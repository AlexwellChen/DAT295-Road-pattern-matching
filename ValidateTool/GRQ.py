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
# TIME = "Sunday, July 24, 2022 8:00:00 AM GMT+02:00"
TIME = d + ", " + date
LOC = [
   
"Nuremberg, Germany",
"Stuttgart, Germany",
"Brunswick, Germany",
"Richmond, Virginia, USA"]

ORIGIN =     "Hanover, Germany"
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

from tool import Scraper
from tool import Road
import numpy as np
import sys 
sys.path.append("..")
import ColorExtractor.tools as tools
import matplotlib.pyplot as plt

scraper = Scraper()

def getKernel(s_x, s_y, e_x, e_y):
    # end point as destination
    maxDelta = max(abs(e_x - s_x), abs(e_y - s_y))
    print("maxDelta: ", maxDelta)
    # start point as center
    center_x = maxDelta
    center_y = maxDelta
    # end point cordination
    end_x = e_x - s_x + maxDelta
    end_y = e_y - s_y + maxDelta
    interpolate = []
    # calculate the interpolation
    tools.DDA(center_x, center_y, end_x, end_y, interpolate)
    kernel = np.zeros((2*maxDelta+1, 2*maxDelta+1))
    kernel[center_x][center_y] = 1
    kernel[end_x][end_y] = 1
    for i in range(len(interpolate)):
        kernel[interpolate[i][0]][interpolate[i][1]] = 1
    # crop all zero rows
    kernel = kernel[~np.all(kernel == 0, axis=1)]
    # crop all zero columns
    kernel = kernel[:, ~np.all(kernel == 0, axis=0)]
    return kernel

def PathFinding(s_x, s_y, e_x, e_y, img):
    '''
    Use DFS to find the path, Start from (s_x, s_y), End at (e_x, e_y)
    '''
    # First convert img to binary image, img is a PIL image
    # if the pixel is not white (255, 255, 255), then set it to 1
    # else set it to 0
    img = np.array(img)
    h, w = img.shape[:2]
    bin_img = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] == 255 and img[i][j][1] == 255 and img[i][j][2] == 255:
                bin_img[i][j] = 0
            else:
                bin_img[i][j] = 1
    
    # in bin_img, set the start point to 10, end point to 20
    bin_img[s_x][s_y] = 10
    bin_img[e_x][e_y] = 20
    
    m_y = int((s_y + e_y) / 2)
    m_x = int((s_x + e_x) / 2)

    # calculate the max distance from middle point to start/end point
    d_m_s = (m_x - s_x) ** 2 + (m_y - s_y) ** 2
    d_m_e = (m_x - e_x) ** 2 + (m_y - e_y) ** 2
    # sqrt the distance
    d_m_s = int(d_m_s ** 0.5)
    d_m_e = int(d_m_e ** 0.5)
    # max distance from middle point to start/end point
    max_distance = max(d_m_s, d_m_e)
    crop_distance = max_distance
    cropped_bin = bin_img[max(0, m_x - crop_distance): min(w, m_x + crop_distance),
                        max(0, m_y - crop_distance): min(h, m_y + crop_distance)]
    cropped_img = img[max(0, m_x - crop_distance): min(w, m_x + crop_distance),
                        max(0, m_y - crop_distance): min(h, m_y + crop_distance)]
    # get new start/end point
    new_s_x, new_s_y, new_e_x, new_e_y = 0, 0, 0, 0
    for i in range(cropped_bin.shape[0]):
        for j in range(cropped_bin.shape[1]):
            if cropped_bin[i][j] == 10:
                new_s_x = i
                new_s_y = j
            if cropped_bin[i][j] == 20:
                new_e_x = i
                new_e_y = j
    
    # plt.imshow(cropped_img)
    # plt.imshow(cropped_bin, alpha=0.5)
    # plt.show()

    # # Then use DFS to find the path from s_x, s_y to e_x, e_y in bin_img
    px = [-1, 0, 1, 0]
    py = [0, 1, 0, -1]
    visited = np.zeros((cropped_bin.shape[0], cropped_bin.shape[1]))
    path = []
    cropped_h, cropped_w = cropped_bin.shape[0:2]
    def dfs(cropped_bin, visited, x, y):
        # boundary check
        if x < 0 or x >= cropped_w or y < 0 or y >= cropped_h:
            print("Out of boundary")
            return
        if x == new_e_x and y == new_e_y:
            print("Find the path")
            path.append((x, y))
            return
        for i in range(4):
            new_x = x + px[i]
            new_y = y + py[i]
            if new_x >= 0 and new_x < cropped_w and new_y >= 0 and new_y < cropped_h and visited[new_x][new_y] == 0 and bin_img[new_x][new_y] == 1:
                visited[new_x][new_y] = 1
                path.append((x, y))
                print("Append at: ", x, y)
                dfs(cropped_bin, visited, new_x, new_y)
                print("Pop at: ", x, y)
                path.pop()
                visited[new_x][new_y] = 0
    visited[new_s_x][new_s_y] = 1
    dfs(cropped_bin, visited, new_s_x, new_s_y)
    # print("path: ", path)


def GRQ(way: Road):
    def calc_diff(pixel, i):
        return (
            (pixel[0] - BG_COLOR[i][0]) ** 2
            + (pixel[1] - BG_COLOR[i][1]) ** 2
            + (pixel[2] - BG_COLOR[i][2]) ** 2
        ) < COLOR_SIMILARITY

    start_point_lat = way.start_pos[0]
    start_point_lng = way.start_pos[1]

    end_point_lat = way.end_pos[0]
    end_point_lng = way.end_pos[1]

    region = scraper.build_tile_region(start_point_lat, start_point_lng, end_point_lat, end_point_lng, 15)
    print(region)
    from os.path import exists
    filename = str(region['x_start']) + '_' + \
        str(region['y_start']) + '_' + str(way.duration)
    file_exists = exists('./cache/' + filename + '.png')
    if(file_exists is False):
        scraper.scraper(way.duration, 15, region, './cache/' + filename + '.png')

    xs = region["x_start"]
    xe = xs + 1
    ys = region["y_start"]
    ye = ys + 1
    (xs, ys) = scraper.num2deg(xs, ys, 15)
    (xe, ye) = scraper.num2deg(xe, ye, 15)

    s_lng = (start_point_lng - ys) / (ye - ys)
    s_lat = (start_point_lat - xs) / (xe - xs)
    e_lng = (end_point_lng - ys) / (ye - ys)
    e_lat = (end_point_lat - xs) / (xe - xs)

    logo = cv.imread('./cache/' + filename + '.png')
    logo = cv.cvtColor(logo, cv.COLOR_BGR2RGB)
    h, w = logo.shape[0:2]
    
    x = 20

    # 图中只有start point，没有end point
    # Convert to pixel
    s_lng_pixel = int(h * s_lng)
    s_lat_pixel = int(w * s_lat)
    e_lng_pixel = int(h * e_lng)
    e_lat_pixel = int(w * e_lat)

    print("image size: ", h, w)
    print("start point: ", s_lat_pixel, s_lng_pixel)
    print("end point: ", e_lat_pixel, e_lng_pixel)

    # Create a zero mask same size as image
    mask = np.zeros(logo.shape[0:2])
    # Set start and end point to 1
    mask[s_lat_pixel][s_lng_pixel] = 1
    mask[e_lat_pixel][e_lng_pixel] = 2
    PathFinding(s_lat_pixel, s_lng_pixel, e_lat_pixel, e_lng_pixel, logo)
    
    # max_lat_delta = abs(s_lat_pixel - e_lat_pixel)
    # max_lng_delta = abs(s_lng_pixel - e_lng_pixel)
    # max_delta = max(max_lat_delta, max_lng_delta)
    # rest_point_delta = []
    # for i in range(1, len(way.way_locs)):
    #     lng_ = int(w * way.way_locs[i][1])  # y 经度
    #     lat_ = int(h * way.way_locs[i][0])  # x 纬度
    #     max_lat_delta = max(max_lat_delta, abs(lat_ - s_lat))
    #     max_lng_delta = max(max_lng_delta, abs(lng_ - s_lng))
    #     max_delta = max(max_delta, max(max_lat_delta, max_lng_delta))
    #     rest_point_delta.append((lat_ - s_lat, lng_ - s_lng))  # 上下是x，左右是y
    
    # # Generate filter
    # kernel = getKernel(s_lat_pixel, s_lng_pixel, e_lat_pixel, e_lng_pixel)
    # # Apply filter
    # # print("Kernel:")
    # # print(kernel)
    # # output = cv.filter2D(mask, 1, kernel)
    # middle point
    m_lng_pixel = int((s_lng_pixel + e_lng_pixel) / 2)
    m_lat_pixel = int((s_lat_pixel + e_lat_pixel) / 2)

    # max distance from middle point to start/end point
    max_distance = max(abs(s_lat_pixel - m_lat_pixel), abs(s_lng_pixel - m_lng_pixel))
    crop_distance = 20 + max_distance
    cropped = logo[max(0, m_lat_pixel - crop_distance): min(w, m_lat_pixel + crop_distance),
                    max(0, m_lng_pixel - crop_distance): min(h, m_lng_pixel + crop_distance)]
    # mask_cropped = mask[max(0, m_lat_pixel - crop_distance): min(w, m_lat_pixel + crop_distance),
    #                 max(0, m_lng_pixel - crop_distance): min(h, m_lng_pixel + crop_distance)]
    
    # # cv.imwrite("./cache/TMP_CROP.png",
    # #            cv.cvtColor(cropped, cv.COLOR_RGB2BGR))
    # # print(cropped.shape)
    # # Show the image
    # from matplotlib import pyplot as plt
    # plt.imshow(cropped)
    # plt.imshow(mask_cropped, alpha=0.5)
    # plt.show()
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