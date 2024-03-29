import pandas as pd
import numpy as np
from post import *
import tools 
import re

# Close pandas warning
pd.options.mode.chained_assignment = None

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
    
    ved_final_path = "./ved-final/"

    # samples_groupby_filename = manual_sample.groupby('Filename')
    samples_groupby_filename = random_sample.groupby('Filename')

    # result df
    result = pd.DataFrame(columns=['Filename', 'index', 'Speed Limit', 'color_code', 'SPQ_color_code', 'Old_color_code'])
    
    new_accuract = 0
    old_accuract = 0
    total = 0
    for filename, group in samples_groupby_filename:
        # Read the data
        _input = pd.read_csv(ved_final_path+filename+'.csv',
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
            # if index in sample_index:
            if index == 6281 and filename == 'VED_171122_week':
                for i in range(up_limit):
                    # Change every 6 items
                    lng_val = (_input['Longitude[deg]'].at[index - i] - NW_lng)/Y
                    lat_val = (_input['Latitude[deg]'].at[index - i] - NW_lat)/X
                    # adptive track length by delta
                    longitude_seq.append(lng_val)
                    latitude_seq.append(lat_val)
                    idx = _input['Index'].at[index]
                    rest_point_delta, max_delta = tools.point_delta(longitude_seq, latitude_seq, idx)
                    if max_delta > 20: # 20 is the threshold, set as you wish
                        break
                try:
                    dir_color_code, orig_color_code = tools.image_process_position_seq(longitude_seq, latitude_seq, idx, max_delta, flag=True)
                except:
                    dir_color_code = tools.image_process(longitude_seq[0], latitude_seq[0], idx)
                    orig_color_code = dir_color_code
                old_color_code = tools.image_process(longitude_seq[0], latitude_seq[0], idx)
                
    #             acutal_index = group[group['index'] == index].index.tolist()[0]
    #             acutal_color_code = group['color_code'].at[acutal_index]
    #             result = result.append({'Filename': filename, 'index': index, 'Speed Limit': row['Speed Limit[km/h]'], 'color_code': acutal_color_code, 'dir_SPQ_color_code': dir_color_code, 'original_SPQ_color_code': orig_color_code, 'Old_color_code': old_color_code}, ignore_index=True)
                
    # result.to_csv('result/manual_SPQ_Accuracy.csv', index=False)
    
