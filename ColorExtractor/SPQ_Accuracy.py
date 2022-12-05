import pandas as pd
import numpy as np
from post import *
import tools 
import re

if __name__ == "__main__":
    # Read the data
    # Columns: [Filename, index, color_code]
    manual_sample = pd.read_csv('manual_selected_samples.csv')
    random_sample = pd.read_csv('rand_selected_samples.csv')
    
    res = [int(s) for s in re.findall(
    r'\d+', "8756_8768_12124_12137_z15_t110400")]
    # print(res)
    zoom = res[4]
    # print(res[0], res[2])
    # print(res[1], res[3])
    NW_lat, NW_lng = num2deg(res[0], res[2], zoom)
    SE_lat, SE_lng = num2deg(res[1], res[3], zoom)
    print(NW_lat, NW_lng)
    print(SE_lat, SE_lng)
    
    # Estabilish the size of the image, Northwest as zero point
    X = SE_lat - NW_lat
    Y = SE_lng - NW_lng
    
    path = "/Users/alexwell/Desktop/DAT295-Road-pattern-matching/"
    samples_groupby_filename = manual_sample.groupby('Filename')
    
    new_accuract = 0
    old_accuract = 0
    total = 0
    for filename, group in samples_groupby_filename:
        # Read the data
        _input = pd.read_csv(path + 'ved_data_enrichment/data/ved-final/'+filename+'.csv',
                                 dtype={"Latitude[deg]": float, 'Longitude[deg]': float,
                                        'Vehicle Speed[km/h]': float, 'Speed Limit[km/h]': 'string'},
                                 nrows=10000)
        _input['Index'] = _input['DayNum'].apply(timeCalc)
        # sample index in this file
        sample_index = group['index'].tolist()
        # print(sample_index)
        for index, row in _input.iterrows():
            up_limit = 30
            longitude_seq = []
            latitude_seq = []
            if index == 1927:
                print('here')
                for i in range(up_limit):
                    # Change every 6 items
                    new_index = index
                    lng_val = (_input['Longitude[deg]'].at[index - i] - NW_lng)/Y
                    lat_val = (_input['Latitude[deg]'].at[index - i] - NW_lat)/X
                    # adptiv track length by delta
                    longitude_seq.append(lng_val)
                    latitude_seq.append(lat_val)
                    idx = _input['Index'].at[index]
                    rest_point_delta, max_delta = tools.point_delta(longitude_seq, latitude_seq, idx)
                    if max_delta > 20:
                        break
                color_code = tools.image_process_position_seq(longitude_seq, latitude_seq, idx, rest_point_delta, max_delta)
                # old_color_code = tools.image_process(longitude_seq[0], latitude_seq[0], idx)
                # acutal_index = group[group['index'] == index].index.tolist()[0]
                break
                # acutal_color_code = group['color_code'].at[acutal_index]
                # if color_code == acutal_color_code:
                #     new_accuract += 1
                # if old_color_code == acutal_color_code:
                #     old_accuract += 1
                # total += 1
        break
        
    # print("New accuracy: ", new_accuract/total)
    # print("Old accuracy: ", old_accuract/total)
    
