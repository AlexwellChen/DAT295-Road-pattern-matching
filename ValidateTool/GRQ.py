from codecs import utf_8_encode
import googlemaps
from datetime import datetime, timedelta, timezone
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
TIME = "Sunday, July 24, 2022 8:00:00 AM GMT+02:00"
# TIME = d + ", " + date
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
    cropped_mask = np.zeros((cropped_bin.shape[0], cropped_bin.shape[1]))
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

    # # Then use DFS to find the path from s_x, s_y to e_x, e_y in bin_img
    print("New start point: ", new_s_x, new_s_y)
    print("New end point: ", new_e_x, new_e_y)
    px = [-1, 0, 1, 0]
    py = [0, 1, 0, -1]
    visited = np.zeros((cropped_bin.shape[0], cropped_bin.shape[1]))
    path = []
    cropped_h, cropped_w = cropped_bin.shape[0:2]
    print("Cropped image size: ", cropped_h, cropped_w)
    find_flag = False
    result = []
    animation = False
    def dfs(cropped_bin, visited, x, y, predistance):
        # boundary check
        nonlocal animation
        nonlocal find_flag
        nonlocal path
        nonlocal result
        if x < 0 or x >= cropped_w or y < 0 or y >= cropped_h:
            print("Out of boundary")
            return
        if animation:
            # Animation to show the finding process
            orignialValue = cropped_bin[x][y]
            cropped_bin[x][y] = 15
            plt.cla()
            plt.imshow(cropped_bin)
            plt.imshow(visited, alpha=0.5)
            plt.pause(0.01)
            cropped_bin[x][y] = orignialValue
        if x == new_e_x and y == new_e_y:
            print("Find the path")
            find_flag = True
            path.append((x, y))
            # print(path)
            result = path[:]
            plt.imshow(cropped_bin)
            plt.imshow(visited, alpha=0.5)
            plt.show()
            return
        for i in range(4):
            new_x = x + px[i]
            new_y = y + py[i]
            if new_x >= 0 and new_x < cropped_h and new_y >= 0 and new_y < cropped_w and find_flag == False:
                distance = (new_x - new_e_x) ** 2 + (new_y - new_e_y) ** 2
                distance = int(distance ** 0.5)
                if distance > predistance:
                    # Greedy optimization, might ignord the path in some cases
                    continue
                if visited[new_x][new_y] == 0 and cropped_bin[new_x][new_y] > 0:
                    visited[new_x][new_y] = 1
                    path.append((x, y))
                    dfs(cropped_bin, visited, new_x, new_y, distance)
                    path.pop()
                    visited[new_x][new_y] = 0
    visited[new_s_x][new_s_y] = 1
    distance_s_e = (new_s_x - new_e_x) ** 2 + (new_s_y - new_e_y) ** 2
    distance_s_e = int(distance_s_e ** 0.5)
    dfs(cropped_bin, visited, new_s_x, new_s_y, distance_s_e)
    
    
    for point in result:
        cropped_mask[point[0]][point[1]] = 1
        # set round pixels to 1
        for i in range(4):
            new_x = point[0] + px[i]
            new_y = point[1] + py[i]
            if new_x >= 0 and new_x < cropped_h and new_y >= 0 and new_y < cropped_w:
                cropped_mask[new_x][new_y] = 1
    
    t = [0, 0, 0, 0]
    for i in range(cropped_img.shape[0]):
        for j in range(cropped_img.shape[1]):
            if(cropped_img[i][j][0] == 255 and cropped_img[i][j][1] == 255 and cropped_img[i][j][2] == 255 and cropped_mask[i][j] == 0):
                continue
            for k in range(len(BG_COLOR)):
                if calc_diff(cropped_img[i][j], k):
                    t[k] += 1
        s = sum(t)
        try:
            t = [i / s for i in t]
        except BaseException:
            t = [0, 0, 0, 0]
        return t
    
def calc_diff(pixel, i):
        return (
            (pixel[0] - BG_COLOR[i][0]) ** 2
            + (pixel[1] - BG_COLOR[i][1]) ** 2
            + (pixel[2] - BG_COLOR[i][2]) ** 2
        ) < COLOR_SIMILARITY

def PathFinding_Conv(s_x, s_y, e_x, e_y, img, img_name):

    img = np.array(img)
    h, w = img.shape[:2]
    bin_img = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] == 255 and img[i][j][1] == 255 and img[i][j][2] == 255:
                bin_img[i][j] = 0
            else:
                bin_img[i][j] = 1
    
    # mask = np.zeros((img.shape[0], img.shape[1]))
    # Set start point to blue
    img[s_x][s_y] = [0, 0, 0]
    img[s_x+1][s_y] = [0, 0, 0]
    img[s_x][s_y+1] = [0, 0, 0]
    img[s_x-1][s_y] = [0, 0, 0]
    img[s_x][s_y-1] = [0, 0, 0]

    crop_distance = 60
    cropped_bin = bin_img[max(0, s_x - crop_distance): min(w, s_x + crop_distance),
                        max(0, s_y - crop_distance): min(h, s_y + crop_distance)]
    cropped_img = img[max(0, s_x - crop_distance): min(w, s_x + crop_distance),
                        max(0, s_y - crop_distance): min(h, s_y + crop_distance)]

    # Direction vector from start to end
    directionVector = [e_x - s_x, e_y - s_y]
    # Normalize the direction vector
    directionVector = directionVector / np.linalg.norm(directionVector)
    # Slope of the direction vector
    slope = directionVector[1] / directionVector[0]

    # Gen Convolution kernel by direction vector
    kernel_size = 21
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        j = int(slope * (i - kernel_size//2) + kernel_size//2)
        # check whether the point is in the kernel
        if j >= 0 and j < kernel_size:
            kernel[i][j] = 1
    
    # Convolution
    output = cv.filter2D(cropped_bin, 1, kernel)

    threshold = int(kernel.sum() * 0.8)
    # Filter the output

	# Without interest area
    output[output < threshold] = 0
    output[output >= threshold] = 1

    showImgFlag = True
    if showImgFlag:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(cropped_img)
        ax.imshow(output, alpha=0.3)
        # Save fig
        fig.savefig("validate_imgs/" + img_name + ".png")
        
        # ax.set_title("Actual trajectory")
        # human_check = int(input("Human check: 1. Mistake 2.Fair enough 3. Good 4. Perfect\n"))
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        # plt.close(fig)
        
    
    # Save cropped image overlayed with output
    
    
    t = [0, 0, 0, 0]
    for i in range(cropped_img.shape[0]):
        for j in range(cropped_img.shape[1]):
            # Check whether the pixel is in the output
            if output[i][j] != 0:
                if(cropped_img[i][j][0] == 255 and cropped_img[i][j][1] == 255 and cropped_img[i][j][2] == 255):
                    continue
                for k in range(len(BG_COLOR)):
                    if calc_diff(cropped_img[i][j], k):
                        t[k] += 1

    s = sum(t)
    try:
        t = [i / s for i in t]
    except BaseException:
        t = [0, 0, 0, 0]
    return t, cropped_img



def GRQ(way: Road, img_name):
    start_point_lat = way.start_pos[0]
    start_point_lng = way.start_pos[1]

    end_point_lat = way.end_pos[0]
    end_point_lng = way.end_pos[1]

    region = scraper.build_tile_region(start_point_lat, start_point_lng, start_point_lat, start_point_lng, 15)
    print(region)
    from os.path import exists
    filename = str(region['x_start']) + '_' + \
        str(region['y_start']) + '_' + str(way.duration)
    file_exists = exists('./cache/' + filename + '.png') # Use local cache
    if(file_exists is False):
        scraper.scraper(way.duration, 15, region, './cache/' + filename + '.png')

    xs = region["x_start"]
    xe = xs + 1
    ys = region["y_start"]
    ye = ys + 1
    (xs, ys) = scraper.num2deg(xs, ys, 15)
    (xe, ye) = scraper.num2deg(xe, ye, 15)

    # print("Start point: ", xs, ys)
    # print("End point: ", xe, ye)

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
    try: 
        # t = PathFinding(s_lat_pixel, s_lng_pixel, e_lat_pixel, e_lng_pixel, logo)
        return PathFinding_Conv(s_lat_pixel, s_lng_pixel, e_lat_pixel, e_lng_pixel, logo, img_name)
    except Exception as e:
        print("Error: ", e)
        print("Conv error, use default")
        # middle point
        m_lng_pixel = int((s_lng_pixel + e_lng_pixel) / 2)
        m_lat_pixel = int((s_lat_pixel + e_lat_pixel) / 2)

        # max distance from middle point to start/end point
        max_distance = max(abs(s_lat_pixel - m_lat_pixel), abs(s_lng_pixel - m_lng_pixel))
        crop_distance = 20 + max_distance
        cropped = logo[max(0, m_lat_pixel - crop_distance): min(w, m_lat_pixel + crop_distance),
                        max(0, m_lng_pixel - crop_distance): min(h, m_lng_pixel + crop_distance)]
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
        return t, cropped