import pandas as pd

# read result
manual_res = pd.read_csv("result/manual_SPQ_Accuracy.csv")
random_res = pd.read_csv("result/random_SPQ_Accuracy.csv")

# Group by speed limit
manual_res_groupby_speed = manual_res.groupby('Speed Limit')
random_res_groupby_speed = random_res.groupby('Speed Limit')

# result df
manual_result = pd.DataFrame(columns=['Speed Limit', 'SPQ_Accuracy', 'Original_Accuracy'])
random_result = pd.DataFrame(columns=['Speed Limit', 'SPQ_Accuracy', 'Original_Accuracy'])

# calculate accuracy
for speed, group in manual_res_groupby_speed:
    dir_SPQ_Accuracy = group[group['dir_SPQ_color_code'] == group['color_code']].shape[0]/group.shape[0]
    origi_SPQ_Accuracy = group[group['original_SPQ_color_code'] == group['color_code']].shape[0]/group.shape[0]
    Original_Accuracy = group[group['Old_color_code'] == group['color_code']].shape[0]/group.shape[0]
    manual_result = manual_result.append({'Speed Limit': speed, 'dir_SPQ_Accuracy': dir_SPQ_Accuracy, 'original_SPQ_Accuracy':origi_SPQ_Accuracy, 'Original_Accuracy': Original_Accuracy}, ignore_index=True)

for speed, group in random_res_groupby_speed:
    dir_SPQ_Accuracy = group[group['dir_SPQ_color_code'] == group['color_code']].shape[0]/group.shape[0]
    origi_SPQ_Accuracy = group[group['original_SPQ_color_code'] == group['color_code']].shape[0]/group.shape[0]
    Original_Accuracy = group[group['Old_color_code'] == group['color_code']].shape[0]/group.shape[0]
    random_result = random_result.append({'Speed Limit': speed, 'dir_SPQ_Accuracy': dir_SPQ_Accuracy, 'original_SPQ_Accuracy':origi_SPQ_Accuracy, 'Original_Accuracy': Original_Accuracy}, ignore_index=True)

# save result
manual_result.to_csv('result/manual_SPQ_Accuracy_by_speed.csv', index=False)
random_result.to_csv('result/random_SPQ_Accuracy_by_speed.csv', index=False)