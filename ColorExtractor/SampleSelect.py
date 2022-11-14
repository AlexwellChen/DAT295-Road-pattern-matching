import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import post
import typer
from tqdm import tqdm
import tools
import re
import sys

if __name__ == "__main__":
    res = [int(s) for s in re.findall(
    r'\d+', "8756_8768_12124_12137_z15_t110400")]
    # print(res)
    zoom = res[4]
    # print(res[0], res[2])
    # print(res[1], res[3])
    NW_lat, NW_lng = post.num2deg(res[0], res[2], zoom)
    SE_lat, SE_lng = post.num2deg(res[1], res[3], zoom)
    print(NW_lat, NW_lng)
    print(SE_lat, SE_lng)

    # Estabilish the size of the image, Northwest as zero point
    X = SE_lat - NW_lat
    Y = SE_lng - NW_lng

    file_names = ['VED_171129_week']
    path = "/Users/alexwell/Desktop/DAT295-Road-pattern-matching/"

    selectedSample = pd.DataFrame(columns=['Filename', 'index', 'color_code'])

    lenght_limit = 50 # 50 manually selected samples
    rand_target_index = 7
    
    plt.ion()
    
    for filename in file_names:
        length = sum(1 for row in open(path + 'ved_data_enrichment/data/ved-final/'+filename+'.csv', 'r'))

        typer.secho(f"Reading file: {filename}", fg="red", bold=True)
        typer.secho(f"total rows: {length}", fg="green", bold=True)
        _input = pd.read_csv(path + 'ved_data_enrichment/data/ved-final/'+filename+'.csv',
                             dtype={"Latitude[deg]": float, 'Longitude[deg]': float,
                                    'Vehicle Speed[km/h]': float, 'Speed Limit[km/h]': 'string'},
                             chunksize=50000)
        with tqdm(total=length, desc="chunks read: ") as bar:
            for i, chunk in enumerate(_input):
                mode = 'w' if i == 0 else 'a'

                chunk = chunk[[
                    'VehId', 'DayNum', 'Latitude[deg]', 'Longitude[deg]', 'Vehicle Speed[km/h]', 'Speed Limit[km/h]']]
                chunk['Index'] = chunk['DayNum'].apply(post.timeCalc)
                grouped = chunk.groupby(['VehId'])

                for name, group in grouped:
                    cnt = 0
                    for index, row in group.iterrows():
                        up_limit = 30
                        rand_num = np.random.randint(0, 10)
                        if cnt > up_limit and cnt % 18 == 0 and rand_num == rand_target_index:
                            longitude_seq = []
                            latitude_seq = []
                            for i in range(up_limit):
                                # Change every 6 items
                                new_index = index
                                lng_val = (group['Longitude[deg]'].at[index - i] - NW_lng)/Y
                                lat_val = (group['Latitude[deg]'].at[index - i] - NW_lat)/X
                                # adptiv track length by delta
                                longitude_seq.append(lng_val)
                                latitude_seq.append(lat_val)
                                idx = group['Index'].at[index]
                                rest_point_delta, max_delta = tools.point_delta(longitude_seq, latitude_seq, idx)
                                if max_delta > 20:
                                    break
                            color_code = tools.show_trajectory_color(longitude_seq, latitude_seq, idx)
                            print("Sample color_code: ", color_code)
                            if color_code in [0, 1, 2, 3, 4]:
                                selectedSample = selectedSample.append(
                                    {'Filename': filename, 'index': index, 'color_code': color_code}, ignore_index=True)
                                print("Current sample number: ", len(selectedSample))
                                if len(selectedSample) > lenght_limit or color_code == 4:
                                    # save the selectedSample
                                    selectedSample.to_csv(
                                        'rand_selected_samples.csv', mode=mode, index=False)
                                    sys.exit()
                            print("----------------------------------------")
                        cnt += 1
                bar.update(len(chunk))
