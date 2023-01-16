import random
import math
import pandas as pd
import cv2 as cv
import numpy as np
from tqdm import tqdm
import time
from datetime import date, datetime, timedelta
# Load the page


# (DayNum 1 = Nov, 1st, 2017, 00:00:00, DayNum 1.5 = Nov, 1st, 2017, 12:00:00)

# file_names = ['VED_171101_week', 'VED_171108_week', 'VED_171115_week', 
#'VED_171122_week', 'VED_171129_week', 'VED_171206_week', # 'VED_171213_week', 'VED_171220_week', 'VED_171227_week',# 'VED_180103_week', 'VED_180110_week', 'VED_180117_week',  #'VED_180124_week', 'VED_180131_week', 'VED_180207_week', #'VED_180214_week', 'VED_180221_week', 'VED_180228_week', 'VED_180307_week',  'VED_180314_week', 'VED_180321_week', 
#'VED_180328_week', 'VED_180404_week', 'VED_180411_week', 
#'VED_180418_week', 'VED_180425_week', 'VED_180502_week', 
#'VED_180509_week', 'VED_180516_week', 'VED_180523_week',
#               
#'VED_180530_week', 'VED_180606_week', 'VED_180613_week', 'VED_180620_week', 'VED_180627_week', 'VED_180704_week', 
#'0-0--0-'
#'VED_180711_week', 'VED_180718_week', 'VED_180725_week', 'VED_180801_week', 'VED_180808_week'], 'VED_180815_week', 
#-----
#'VED_180822_week', ]'VED_180829_week', 
#'VED_180905_week', 'VED_180912_week', 'VED_180919_week', 'VED_180926_week', 'VED_181003_week', 
# ------------
#'VED_181010_week',  'VED_181017_week', 
#'VED_181024_week', 'VED_181031_week', 'VED_181107_week']

# file_names = ['VED_180905_week', 'VED_180912_week']
file_names = ['VED_181003_week']
# file_names = ['VED_180801_week', 'VED_180808_week', 'VED_180815_week']
# file_names = ['VED_171213_week', 'VED_171220_week', 'VED_171227_week']
# file_names = ['VED_180103_week', 'VED_180110_week', 'VED_180117_week']
# file_names = ['VED_180124_week', 'VED_180131_week', 'VED_180207_week']
# file_names = ['VED_180307_week', 'VED_180314_week', 'VED_180321_week']


# file_names = ['sample_input']

# green, yellow, red, dark red
bg_color = [[148, 200, 97], [225, 131, 49], [211, 45, 31],  [146, 36, 29]]
# color similarity threshold
threshold = 3000


def calc_diff(pixel, i):
    return (
        (pixel[0] - bg_color[i][0]) ** 2
        + (pixel[1] - bg_color[i][1]) ** 2
        + (pixel[2] - bg_color[i][2]) ** 2
    ) < threshold


def image_process(lng: float, lat: float, idx, output=None):
    try:
        logo = cv.imread('./output/8756_8768_12124_12137_z15_t' +
                         str(idx*600)+'.png')
        logo = cv.cvtColor(logo, cv.COLOR_BGR2RGB)
        h, w = logo.shape[0:2]
        x = 8

        lng = int(h*lng)
        lat = int(w*lat)
        cropped = logo[max(0, lat-x): min(w, lat+x),
                       max(0, lng-x): min(h, lng+x)]  # [y0:y1, x0:x1]


        t = [0, 0, 0, 0]
        for i in range(cropped.shape[0]):
            for j in range(cropped.shape[1]):
                if(cropped[i][j][0] == 255 and cropped[i][j][1] == 255 and cropped[i][j][2] == 255):
                    continue
                for k in range(len(bg_color)):
                    if calc_diff(cropped[i][j], k):
                        t[k] += 1
        t_sum = sum(t)
        return 1-t[0]/t_sum, 1-t[1]/t_sum, 1- t[2]/t_sum, 1- t[3]/t_sum
    except:
        return 1,1,1,1


def num2deg(xtile, ytile, zoom):
    n = math.pow(2, zoom)
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def timeCalc(daynum):
    start = datetime(2017, 11, 1)
    # UTC -> Local time
    start = start - timedelta(hours=5)
    test = start + timedelta(days=daynum-1)
    # Monday is 0 and Sunday is 6.
    d = test.weekday()
    h = test.hour
    m = test.minute
    if(m % 10 > 5):
        m += (10 - m % 10)
    else:
        m -= (m % 10)
    idx = d*(144) + h * 6 + m
    idx %= 1008
    if d == 6:
        d = 0
    else:
        d += 1
    return idx


if __name__ == "__main__":
    import re
    res = [int(s) for s in re.findall(
        r'\d+', "8756_8768_12124_12137_z15_t110400")]
    # print(res)
    zoom = res[4]
    # print(res[0], res[2])
    # print(res[1], res[3])
    NW_lat, NW_lng = num2deg(res[0], res[2]-1, zoom)
    SE_lat, SE_lng = num2deg(res[1]+1, res[3], zoom)
    # print(NW_lat, NW_lng)
    # print(SE_lat, SE_lng)
    X = SE_lng - NW_lng
    Y = SE_lat - NW_lat
    import typer
    ved_final_path = "./ved-final/"
    for filename in file_names:
        length = sum(1 for row in open(ved_final_path+filename+'.csv', 'r'))

        typer.secho(f"Reading file: {filename}", fg="red", bold=True)
        typer.secho(f"total rows: {length}", fg="green", bold=True)
        _input = pd.read_csv(ved_final_path+filename+'.csv',
                             dtype={"Matchted Latitude[deg]": float, 'Matched Longitude[deg]': float,
                                    'Vehicle Speed[km/h]': float, 'Speed Limit[km/h]': 'string'},
                             chunksize=1000)
        with tqdm(total=length, desc="chunks read: ") as bar:
            for i, chunk in enumerate(_input):
                mode = 'w' if i == 0 else 'a'

                # chunk = chunk[[
                #     'DayNum', 'Matchted Latitude[deg]', 'Matched Longitude[deg]', 'Vehicle Speed[km/h]', 'Speed Limit[km/h]']]
                chunk['Speed Limit[km/h]'] = pd.to_numeric(
                    chunk['Speed Limit[km/h]'], errors='coerce')
                chunk = chunk.assign(gp='1') # Green
                chunk = chunk.assign(op='1') # Orange
                chunk = chunk.assign(rp='1') # Red
                chunk = chunk.assign(dp='1') # Dark Red
                chunk['Index'] = chunk['DayNum'].apply(timeCalc)
                for index, row in chunk.iterrows():
                    chunk['gp'].at[index],chunk['op'].at[index],chunk['rp'].at[index],chunk['dp'].at[index] = image_process(
                        (chunk['Matched Longitude[deg]'].at[index] -   NW_lng)/X,
                        (chunk['Matchted Latitude[deg]'].at[index] -  NW_lat)/Y,
                        chunk['Index'].at[index],
                        str(i)
                    )
                    bar.update(1)

                header = i == 0

                chunk.to_csv(
                    "./vec_res/" + filename+".csv",
                    na_rep='nan',
                    index=False,  # Skip index column
                    header=header,
                    mode=mode)
