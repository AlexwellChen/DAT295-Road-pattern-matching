from codecs import utf_8_encode
import googlemaps
from datetime import datetime, timedelta, timezone
from pexpect import TIMEOUT
import requests
import folium
from folium.vector_layers import PolyLine
import polyline
from geopy import distance
import csv
import os
import random
import math
from PIL import Image
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from io import BytesIO
from tqdm import tqdm
import cv2 as cv
import calendar
import time
from datetime import datetime
from GRQ import *
gmaps = googlemaps.Client(key="AIzaSyAFqM8oZMQAJaPvS2qYcFzZpAluCr0KywQ")

# G, Y, R, D
BG_COLOR = [[148, 200, 97], [225, 131, 49], [211, 45, 31], [146, 36, 29]]
# color similarity threshold
COLOR_SIMILARITY = 3000
ENABLE_PROXY = False
TIMEOUT= 120

OVERPASS = "http://overpass-api.de/api/interpreter"
MAX_SEGMENT_LENGTH = 100
FOCUS_POINT_THRESHOLD = 40
ZOOM_LEVEL = 16
# https://www.epochconverter.com/
# e.g. 1658556000
# Saturday 8AM in local time zone
# MODIFY MANUALLY
# Current timestamp
ts = calendar.timegm(time.gmtime())
DEPARTURE_TIME = ts
# TIME = "Friday, July 29, 2022 2:00:00 PM GMT+01:00"
# TIME = "Sunday, July 24, 2022 6:00:00 PM GMT+01:00"
# TIME = "Tuesday, July 26, 2022 8:00:00 PM GMT+02:00"
week_dict = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 0: ' Sunday'}
d = week_dict.get(datetime.today().isoweekday())
date = datetime.today().strftime("%m %d, %Y")
TIME = "Sunday, July 24, 2022 8:00:00 AM GMT+02:00"
# TIME = d + ", " + date
LOC = [
   
"Nuremberg, Germany",
"Stuttgart, Germany",
"Brunswick, Germany",
"Richmond, Virginia, USA"]

ORIGIN =     "Nuremberg, Germany"
DESTINATION = "Hamburg, Germany" 
WAYPOINTS = None
# WAYPOINTS = "A9, 85095 Denkendorf, Germany"
# WAYPOINTS = "E43, 97215 Simmershofen, Germany"
# WAYPOINTS = "A3, 96193 Wachenroth, Germany"
TRIP_IDX = 40

FILE_NAME = str(TRIP_IDX) + "_" + ORIGIN
if WAYPOINTS is not None:
    FILE_NAME += '_' +   WAYPOINTS

FILE_NAME += '_' + DESTINATION + '_' + TIME
print(FILE_NAME)
CSV_FILE_LOC = "./output/" + FILE_NAME + ".csv"
CSV_HEADER = [
    'lat',
    'lng',
    'elevation',
    'distance_to_previous_point',
    'gradient',
    'traffic_signals',
    'crossing',
    'bus_stops',
    'other',
    'intersection',
    'road_name',
    'speedlimit',
    'lane',
    'type_of_road',
    'gp',
    'yp',
    'rp',
    'dp',
    'duration']

# --------VISUALIZATION---------
map = folium.Map(prefer_canvas=True, tiles="cartodbpositron")
t = folium.TileLayer("cartodbpositron").add_to(map)
if not os.path.exists('./cache'):
    os.makedirs('cache')
if not os.path.exists('output'):
    os.makedirs('output')



class Proxy:
    def __init__(self, google=False):
        self.proxy = None
        self.proxy_list = []

        if(google):
            self.addr = 'https://www.google-proxy.net/'
        else:
            self.addr = 'https://free-proxy-list.net/'
        self.build_proxy_list()
        self.cnt = 0

    def build_proxy_list(self):
        self.proxy_list = []

        ua = UserAgent()
        header = {'User-Agent': str(ua.random)}
        proxies_doc = requests.get(
            # 'https://www.google-proxy.net/',
            self.addr,
            headers=header).text

        soup = BeautifulSoup(proxies_doc, 'html.parser')
        proxies_table = soup.find(class_='fpl-list')

        for row in proxies_table.tbody.find_all('tr'):
            self.proxy_list.append(
                row.find_all('td')[0].string + ":" +
                row.find_all('td')[1].string
            )
        self.proxy = self.proxy_list[0]

    def get_proxy(self, flag=True):
        if(flag):
            return self.proxy
        self.cnt += 1
        if(self.cnt == min(75, len(self.proxy_list))):
            self.cnt = 0
            self.build_proxy_list()
        self.proxy = self.proxy_list[self.cnt]
        return self.proxy


proxy = Proxy()
gproxy = Proxy(True)


class Scraper:

    def coordinate_to_tile(self, lat, lon, zoom):
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)

    def scraper(self, time, zoom_level, region, filename):
        output_file = "./" + filename
        w = region['x_end'] - region['x_start'] + 1
        h = region['y_end'] - region['y_start'] + 1

        url_template = (
            "http://mt%i.google.com/vt?lyrs=h,traffic|seconds_into_week:%i&x=%i&y=%i&z=%i&apistyle=p.v%%3Aoff")
        image_out = Image.new('RGB', (256 * w, 256 * h), color=(255, 255, 255))

        for i in range(w):
            for j in range(h):
                while True:
                    x = region['x_start'] + i
                    y = region['y_start'] + j
                    url = url_template % (
                        random.randrange(4), time, x, y, zoom_level)
                    p = gproxy.get_proxy()

                    try:
                        im = self.get_traffic_image(url, p)
                        break
                    except Exception as err:
                        print("PROXY ERROR::", err)
                        p = gproxy.get_proxy(False)
                        continue
                im = im.convert('RGBA')
                image_out.paste(im, (256 * i, 256 * j), im)
            image_out.save(output_file)

    def build_tile_region(self, start_lat, start_lon, end_lat, end_lon, zoom):
        sx, sy = self.coordinate_to_tile(start_lat, start_lon, zoom)
        
        ex, ey = self.coordinate_to_tile(end_lat, end_lon, zoom)
        
        if(sx > ex):
            sx, ex = ex, sx
        if(sy > ey):
            sy, ey = ey, sy

        return {
            'x_start': sx,
            'x_end': ex,
            'y_start': sy,
            'y_end': ey
        }

    def get_traffic_image(self, url, proxy):

        image_data = requests.get(
            url,
            proxies={
                "http": proxy,
                "https": proxy},
            timeout=3).content

        return Image.open(BytesIO(image_data))

    def num2deg(self, xtile, ytile, zoom):
        n = math.pow(2, zoom)
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return (lat_deg, lon_deg)

    def getColor(self, lat, lng, time):
        def calc_diff(pixel, i):
            return (
                (pixel[0] - BG_COLOR[i][0]) ** 2
                + (pixel[1] - BG_COLOR[i][1]) ** 2
                + (pixel[2] - BG_COLOR[i][2]) ** 2
            ) < COLOR_SIMILARITY
        region = self.build_tile_region(
            lat, lng, lat, lng, 15)
        # print(region)
        from os.path import exists
        filename = str(region['x_start']) + '_' + \
            str(region['y_start']) + '_' + str(time)
        file_exists = exists('./cache/' + filename + '.png')
        if(file_exists is False):
            self.scraper(time, 15, region, './cache/' + filename + '.png')

        xs = region["x_start"]
        xe = xs + 1
        ys = region["y_start"]
        ye = ys + 1
        (xs, ys) = self.num2deg(xs, ys, 15)
        (xe, ye) = self.num2deg(xe, ye, 15)
        lng = (lng - ys) / (ye - ys)
        lat = (lat - xs) / (xe - xs)

        logo = cv.imread('./cache/' + filename + '.png')
        logo = cv.cvtColor(logo, cv.COLOR_BGR2RGB)
        h, w = logo.shape[0:2]
        x = 8

        # print(h*lng-x, h*lng+x, w*lat-x, w*lat+x)
        lng = int(h * lng)
        lat = int(w * lat)
        cropped = logo[max(0, lat - x): min(w, lat + x),
                       max(0, lng - x): min(h, lng + x)]  #
        # cv.imwrite("./cache/TMP_CROP.png",
        #            cv.cvtColor(cropped, cv.COLOR_RGB2BGR))
        # print(cropped.shape)
        # Show the image
        # cv.imshow('image', cropped)
        t = [0, 0, 0, 0]
        for i in range(cropped.shape[0]):
            for j in range(cropped.shape[1]):
                if(cropped[i][j][0] == 255 and cropped[i][j][1] == 255 and cropped[i][j][2] == 255):
                    continue
                for k in range(len(BG_COLOR)):
                    if calc_diff(cropped[i][j], k):
                        t[k] += 1
        s = sum(t)
        try:
            t = [i / s for i in t]
        except BaseException:
            t = [0, 0, 0, 0]
        return t


scraper = Scraper()


class Node:
    def __init__(self, lat, lon, _type=None):
        self.loc = [lat, lon]
        self.type = _type

    def closeTo(self, loc, direction):
        if(distance.distance(loc, self.loc).m < FOCUS_POINT_THRESHOLD):
            if(loc[0] - self.loc[0] > 0) == direction[0] and (loc[1] - self.loc[1] > 0) == direction[1]:
                return True
        return False

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Location: " + ' '.join([str(value) for value in self.loc])


class Road:
    def __init__(self, start_pos, end_pos, distance, duration, polyline):
        self.p = polyline
        self.way_locs = []
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.distance = distance
        self.tags = ""
        self.name = None
        self.ref = None
        self.wayid = 0
        self.lane = None
        self.maxspeed = None
        self.type_of_road = None
        self.duration = duration
        self.distance = []
        self.slope = []
        self.focus_points = []
        self.intersections = []
        self.elevation = []
        self.current = []
        self.free = []
        self.sum_weight = 0
        self.decode()
        # self.getElevation()
        self.getWayAttr()

    def getElevation(self):
        for p in self.way_locs:
            self.elevation.append([i["elevation"] for i in gmaps.elevation(p)])

    def getWayAttr(self):
        if(self.wayid == 0):
            self.reverseGeoCoding()
        # print(self.wayid)
        query_sentence = """[out:json];way(%s);(._;>;);out tags;"""

        # response = requests.get(OVERPASS,
        # params={'data': query_sentence % self.wayid})
        # print("way", response.status_code)
        p = proxy.get_proxy()
        counter = 1

        while True:
            try:
                response = requests.get(
                    OVERPASS,
                    params={
                        'data': query_sentence % self.wayid},
                    timeout=TIMEOUT)
                break
            except Exception as err:
                counter += 1
                print("Road Attribute ERROR::", err)


        if(response.status_code !=200):
            print(query_sentence % self.wayid)
            print("ROAD ATTR:: Give UP")
            return
        response = response.json()
        # with open("sample.json", "w") as outfile:
        #     import json
        #     json.dump(response, outfile)
        if len(response["elements"]) == 0:
            print(query_sentence % self.wayid)
            return
        if "tags" not in response["elements"][-1]:
            return
        self.tags = response["elements"][-1]["tags"]
        if "ref" in self.tags:
            self.ref = self.tags["ref"]
            # print(self.ref)
        if "name" in self.tags:
            self.name = self.tags["name"]
        if "maxspeed" in self.tags:
            import re
            if "mph" in self.tags["maxspeed"]:
                self.maxspeed =str(int(re.search("\d+", self.tags["maxspeed"])[0]) * 1.60934)
            else:
            # print(self.wayid, self.tags["maxspeed"])
                self.maxspeed = self.tags["maxspeed"]
        if "highway" in self.tags:
            self.type_of_road = self.tags["highway"]
        if "lane" in self.tags:
            self.lane = self.tags["lanes"]
        # self.getFocusPoint()
        # self.getIntersection()

    def getIntersection(self):
        flag = False
        query = """[out:json];way"""
        if self.ref is not None:
            query =  query +"""[ref="%s"]""" % self.ref
            flag = True
        if self.name is not None:
            query = query + """["name"="%s"]""" % self.name
            flag = True
        if(self.lane is not None):
            query = query + """["lanes"="%s"]""" % self.lane
        if "source" in self.tags:
            query = query + """["source"= "%s"]""" % self.tags["source"]

        if(flag == False):
            return
        query = query + \
            """->.relevant_ways;
                foreach.relevant_ways->.this_way{
                node(w.this_way)->.this_ways_nodes;
                way(bn.this_ways_nodes)->.linked_ways;
                way.linked_ways
                    ["highway"]
                    ["highway"!~"footway|cycleway|path|service|track"]
                    ->.linked_ways;
                (
                    .linked_ways->.linked_ways;
                    -
                    .this_way->.this_way;
                )->.linked_ways_only;
                node(w.linked_ways_only)->.linked_ways_only_nodes;
                node.linked_ways_only_nodes.this_ways_nodes;
                out;
                }
                """
        # print(query)
        counter = 1

        p = proxy.get_proxy()
        while True:
            try:
                if ENABLE_PROXY:
                    response = requests.get(
                        OVERPASS, params={
                            'data': query}, timeout=TIMEOUT, proxies={
                            "http": p})
                    break
                else:
                    response = requests.get(
                        OVERPASS, params={
                            'data': query}, timeout=TIMEOUT)
                    break
            except Exception as err:
                counter += 1
                print("OVERPASS INTERSECTION ERROR::", err)
                p = proxy.get_proxy(False)

        if(response.status_code != 200):
            print(
                "OVERPASS INTERSECTION QUERY ERROR: Status Code ",
                response.status_code)
            return
        response = response.json()
        for element in response["elements"]:
            self.intersections.append(Node(element["lat"], element["lon"]))
        return

    def getFocusPoint(self):
        flag = False
        query = """[out:json];way"""
        if self.ref is not None:
            query =  query +"""[ref="%s"]""" % self.ref
            flag = True
        if self.name is not None:
            query = query + """["name"="%s"]""" % self.name
            flag = True
        if(self.lane is not None):
            query = query + """["lanes"="%s"]""" % self.lane
        if "source" in self.tags:
            query = query + """["source"= "%s"]""" % self.tags["source"]

        if(flag  == False):
            return
        query = query + \
            """->.sameway;foreach.sameway->.this_way{node(w.this_way)->.this_ways_nodes;node.this_ways_nodes(if:is_tag("highway")||is_tag("railway"));way.linked_ways["highway"]->.linked_ways;out;}"""

        p = proxy.get_proxy()
        counter = 1
        while True:
            try:
                if ENABLE_PROXY:
                    response = requests.get(
                        OVERPASS, params={
                            'data': query}, timeout=TIMEOUT, proxies={
                            "http": p})
                    break
                else:
                    response = requests.get(
                        OVERPASS, params={
                            'data': query}, timeout=TIMEOUT)
                    break

            except Exception as err:
                print("OVERPASS FOCAL ERROR::", err)
                counter += 1
                if(counter == 50):
                    print("GIVE UP FOCAL POINT") 
                if ENABLE_PROXY:
                    p = proxy.get_proxy(False)

        # response = requests.get(OVERPASS, params={'data': query})
        # counter  = 1
        # while(counter < 5 and response.status_code != 200):
        #     print("Query Focus Point Timeout, Retrying ", counter)
        #     counter +=1
        #     response = requests.get(OVERPASS, params={'data': query})
        if(response.status_code != 200):
            print("OVERPASS FOCAL ERROR: Status Code ", response.status_code)
            print(query)
            return
        response = response.json()
        for element in response["elements"]:
            t = None
            if ("highway" in element["tags"]):
                t = element["tags"]["highway"]
            if ("crossing" in element["tags"]):
                t = "crossing"
            if("railway" in element["tags"]):
                t = element["tags"]["railway"]

            self.focus_points.append(Node(element["lat"], element["lon"], t))
        return

    def reverseGeoCoding(self):
        test = "https://nominatim.geocoding.ai/reverse?lat=%s&lon=%s&zoom=" + \
            str(ZOOM_LEVEL) + "&format=jsonv2"
        # Zoom 16 to filter building
        # might sitll locate to wrong place
        # (instead of way we are interested, try use another location or make the zoom be lower)
        l = int(len(self.way_locs) / 2)
        response = requests.get(test %
                                (self.way_locs[l][0], self.way_locs[l][1]))
        response = response.json()
        # print(response)
        self.wayid = response["osm_id"]

    def getRealTimeTraffic(self):
        # Replace with your TomTom API key here
        key = "YOUR_API"
        for loc in self.way_locs:
            query = """https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/16/json?point=%s%%2C%s&unit=KMPH&openLr=false&key="""
            query = query % (loc[0], loc[1]) + key
            response = requests.get(query)
            response = response.json()
            # print(response["flowSegmentData"]["currentSpeed"])
            # print(response["flowSegmentData"]["freeFlowSpeed"])
            self.current.append(
                float(response["flowSegmentData"]["currentSpeed"]))
            self.free.append(
                float(
                    response["flowSegmentData"]["freeFlowSpeed"]))
            self.sum_weight = self.sum_weight + \
                (self.current[-1] / self.free[-1])

    def decode(self):
        tmp = polyline.decode(self.p)
        sz = len(tmp) - 1
        idx = 0
        while idx < sz:
            distance_geopy = distance.distance(tmp[idx], tmp[idx + 1]).m
            # print(distance_geopy)
            while(distance_geopy > MAX_SEGMENT_LENGTH):
                tmp.insert(idx + 1,
                           ((tmp[idx][0] + tmp[idx + 1][0]) / 2,
                            (tmp[idx][1] + tmp[idx + 1][1]) / 2))
                distance_geopy = distance.distance(tmp[idx], tmp[idx + 1]).m
            sz = len(tmp) - 1
            idx += 1

        self.distance = [0]
        for i in range(len(tmp) - 1):
            d = distance.distance(tmp[i], tmp[i + 1]).m
            self.distance.append(d)

        self.way_locs = tmp

    def export(self):
        # _filter = "traffic_signals"
        with open(CSV_FILE_LOC, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            idx = 0
            # print(self.way_locs)
            for idx in range(1, len(self.way_locs)):
                isTrafficSignal, isCrossing, isBusStop, isOther, isIntersection = False, False, False, False, False
                # isCrossing = False
                # isBusStop = False
                # isOther = False
                # isIntersection = False
                for node in self.focus_points:
                    if(node.closeTo(self.way_locs[idx],
                                    [self.way_locs[idx][0] - self.way_locs[idx - 1][0] > 0, self.way_locs[idx][1] - self.way_locs[idx - 1][1] > 0])):
                        if node.type == "traffic_signals":
                            isTrafficSignal = True
                        elif node.type == "crossing":
                            isCrossing = True
                        elif node.type == "bus_stop":
                            isBusStop = True
                        else:
                            isOther = True
                        if isBusStop and isCrossing and isTrafficSignal and isOther:
                            break
                for node in self.intersections:
                    if(node.closeTo(self.way_locs[idx],
                                    [self.way_locs[idx][0] - self.way_locs[idx - 1][0] > 0, self.way_locs[idx][1] - self.way_locs[idx - 1][1] > 0])):
                        isIntersection = True
                        break
                color_vec = scraper.getColor(
                    self.way_locs[idx][0],
                    self.way_locs[idx][1],
                    self.duration)
                writer.writerow([
                    self.way_locs[idx][0],
                    self.way_locs[idx][1],
                    0,  # self.elevation[idx][0],
                    self.distance[idx],
                    0,
                    # float(self.elevation[idx][0] - self.elevation[idx - 1][0]) / self.distance[idx],
                    isTrafficSignal,
                    isCrossing,
                    isBusStop,
                    isOther,
                    isIntersection,
                    self.name,
                    self.maxspeed,
                    self.lane,
                    self.type_of_road,
                    color_vec[0],
                    color_vec[1],
                    color_vec[2],
                    color_vec[3]
                    # None,
                ])

    def plot_map(self):
        folium.Marker(location=self.start_pos, popup=self.name).add_to(map)
        folium.Marker(location=self.end_pos).add_to(map)
        idx = 0
        for intermidate_point in self.way_locs:
            # grade = 0
            # if idx > 0:
            #     grade = (
            # self.elevation[idx][0] - self.elevation[idx - 1][0]) /
            # (self.distance[idx])
            tag = "Distance to previous point: " + str(self.distance[idx])
            # +"\nElevation: " + str(self.elevation[idx])
            #+ "\nGrade: " + str(grade)
            # "\nTraffic indicator: " + str(self.current[idx]/self.free[idx]            )
            folium.CircleMarker(
                location=intermidate_point,
                radius=2,
                fill=False,
                color='red',
                opacity=0.2).add_to(map)
            idx += 1
        p = PolyLine(
            locations=polyline.decode(
                self.p),
            opacity=0.7).add_to(map)

        for focus_node in self.focus_points:
            folium.CircleMarker(
                location=focus_node.loc,
                color='green', radius=5, popup=focus_node.type
            ).add_to(map)
        for focus_node in self.intersections:
            folium.CircleMarker(
                location=focus_node.loc,
                color='yellow', radius=5
            ).add_to(map)

        return map


def test_single_tile():
    sol = Scraper()

    level = 15
    # time = 64800
    res = sol.getColor(
        42.32532134284961, -83.67431783690408, -1)


def timeCalc(mid):
    weekday = (mid.weekday() + 1) % 7
    return weekday * 86400 + mid.hour * 3600 + mid.minute * 60


def testTime():
    t = datetime.fromtimestamp(DEPARTURE_TIME) + timedelta(hours=1)
    sec_from_SUN_mid_night = timeCalc(t)
    assert(sec_from_SUN_mid_night == 550800)




if __name__ == '__main__':
    GMAPS_API = "AIzaSyAFqM8oZMQAJaPvS2qYcFzZpAluCr0KywQ"
    url = "https://maps.googleapis.com/maps/api/directions/json?origin=%s&destination=%s&departure_time=%s&key=" + GMAPS_API
    if WAYPOINTS is not None:
        url = "https://maps.googleapis.com/maps/api/directions/json?origin=%s&destination=%s&waypoints=via:%s&departure_time=%s&key=" + GMAPS_API

    payload = {}
    headers = {}
    use_local = False
    if use_local is False:
        if WAYPOINTS is not None:
            directions_result = requests.request(
                "GET",
                url %
                (ORIGIN,
                DESTINATION,
                WAYPOINTS,
                str(DEPARTURE_TIME)),
                headers=headers,
                data=payload).json()
        else:
            directions_result = requests.request(
                "GET",
                url %
                (ORIGIN,
                DESTINATION,
                str(DEPARTURE_TIME)),
                headers=headers,
                data=payload).json()
        with open("sample.json", "w") as outfile:
            import json
            json.dump(directions_result, outfile)
        with open("current_time.txt", "w") as outfile:
            outfile.write(str(DEPARTURE_TIME))
        print("Get way points from Google Done!")
    else:
        # use local sample.json
        with open("sample.json", "r") as outfile:
            import json
            directions_result = json.load(outfile)
        with open("current_time.txt", "r") as outfile:
            DEPARTURE_TIME = int(outfile.read())

    departure_time = datetime.fromtimestamp(DEPARTURE_TIME, timezone.utc)
    duration = directions_result['routes'][0]["legs"][0]["duration_in_traffic"]["value"]
    # COLOR_TIME = timeCalc(departure_time + timedelta(seconds=duration // 2))
    COLOR_TIME = timeCalc(departure_time)
    with open(CSV_FILE_LOC, 'w+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        # departure time
        writer.writerow([None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         departure_time])

        # total duration
        writer.writerow(
            [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                duration
            ])

    ne_lat = float(directions_result['routes']
                   [0]["bounds"]["northeast"]["lat"])
    ne_lng = float(directions_result['routes']
                   [0]["bounds"]["northeast"]["lng"])
    sw_lat = float(directions_result['routes']
                   [0]["bounds"]["southwest"]["lat"])
    sw_lng = float(directions_result['routes']
                   [0]["bounds"]["southwest"]["lng"])

    map.fit_bounds([[ne_lat, ne_lng], [sw_lat, sw_lng]])

    ways = []

    import pandas as pd
    # Create a csv file, cols: img_id, color_vec, human_check
    df = pd.DataFrame(columns=['img_id', 'color_vec', 'human_check'])
    prefix = ORIGIN.split(',')[0] + '_' + DESTINATION.split(',')[0] + '_'
    img_path = 'validate_imgs/' + prefix
    for i, leg in enumerate(
            tqdm(directions_result['routes'][0]["legs"][0]["steps"])):
        ways.append(
            Road(
                ([leg["start_location"]["lat"], leg["start_location"]["lng"]]),
                ([leg["end_location"]["lat"], leg["end_location"]["lng"]]),
                leg["distance"]["value"],
                COLOR_TIME,
                leg["polyline"]["points"]))
        # GRQ should be added here
        # ways[i].export()
        # ways[i].plot_map()
        color_vec, img = GRQ(ways[i], img_name=prefix + str(i) + '.png')
        # numpy array to image
        # img = Image.fromarray(img)
        # img.save(img_path + prefix + str(i) + '.png')
        # store the color_vec
        df.loc[i] = [prefix + str(i), color_vec, None]

        # # print(GRQ(ways[i]))
        # if i == 1:
        #     GRQ(ways[i])
        #     break
    df.to_csv('validate_imgs/' + prefix + 'color_vec.csv', index=False)
    map.save('./output/MAP_' + FILE_NAME + '.html')

    



