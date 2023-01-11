import pandas as pd

# Read the data, columns: [Filename,index,Speed Limit,color_code,Old_color_code,dir_SPQ_color_code,original_SPQ_color_code]
manual_acc = pd.read_csv('manual_SPQ_Accuracy.csv')
random_acc = pd.read_csv('random_SPQ_Accuracy.csv')

# Drop Filename
manual_acc = manual_acc.drop(columns=['Filename'])
random_acc = random_acc.drop(columns=['Filename'])

# split data by Speed Limit range by [0 to 50], (50 to 90), and [90 to 130].
manual_acc['Speed Limit'] = manual_acc['Speed Limit'].astype(int)
random_acc['Speed Limit'] = random_acc['Speed Limit'].astype(int)

manual_acc_0_50 = manual_acc[(manual_acc['Speed Limit'] >= 0) & (manual_acc['Speed Limit'] < 50)]
manual_acc_50_90 = manual_acc[(manual_acc['Speed Limit'] >= 50) & (manual_acc['Speed Limit'] < 90)]
manual_acc_90_130 = manual_acc[(manual_acc['Speed Limit'] >= 90) & (manual_acc['Speed Limit'] <= 130)]

random_acc_0_50 = random_acc[(random_acc['Speed Limit'] >= 0) & (random_acc['Speed Limit'] < 50)]
random_acc_50_90 = random_acc[(random_acc['Speed Limit'] >= 50) & (random_acc['Speed Limit'] < 90)]
random_acc_90_130 = random_acc[(random_acc['Speed Limit'] >= 90) & (random_acc['Speed Limit'] <= 130)]

# Calculate the accuracy
def accuracy(df):
    total = len(df)
    if total == 0:
        return 0, 0, 0, 0, 0, 0, 0
    old_acc = len(df[df['color_code'] == df['Old_color_code']])
    original_acc = len(df[df['color_code'] == df['original_SPQ_color_code']])
    dir_acc = len(df[df['color_code'] == df['dir_SPQ_color_code']])
    return old_acc/total, original_acc/total, dir_acc/total, old_acc, original_acc, dir_acc, total

# Calculate the accuracy for each range
manual_acc_0_50_old, manual_acc_0_50_original, manual_acc_0_50_dir, manual_acc_0_50_old_num, manual_acc_0_50_original_num, manual_acc_0_50_dir_num, manual_acc_0_50_total = accuracy(manual_acc_0_50)
manual_acc_50_90_old, manual_acc_50_90_original, manual_acc_50_90_dir, manual_acc_50_90_old_num, manual_acc_50_90_original_num, manual_acc_50_90_dir_num, manual_acc_50_90_total = accuracy(manual_acc_50_90)
manual_acc_90_130_old, manual_acc_90_130_original, manual_acc_90_130_dir, manual_acc_90_130_old_num, manual_acc_90_130_original_num, manual_acc_90_130_dir_num, manual_acc_90_130_total = accuracy(manual_acc_90_130)

random_acc_0_50_old, random_acc_0_50_original, random_acc_0_50_dir, random_acc_0_50_old_num, random_acc_0_50_original_num, random_acc_0_50_dir_num, random_acc_0_50_total = accuracy(random_acc_0_50)
random_acc_50_90_old, random_acc_50_90_original, random_acc_50_90_dir, random_acc_50_90_old_num, random_acc_50_90_original_num, random_acc_50_90_dir_num, random_acc_50_90_total = accuracy(random_acc_50_90)
random_acc_90_130_old, random_acc_90_130_original, random_acc_90_130_dir, random_acc_90_130_old_num, random_acc_90_130_original_num, random_acc_90_130_dir_num, random_acc_90_130_total = accuracy(random_acc_90_130)

# Print manual accuracy
print('Manual Accuracy')
print('old 0 to 50: ', manual_acc_0_50_old, 'num/total: ', manual_acc_0_50_old_num, '/', manual_acc_0_50_total)
print('original 0 to 50: ', manual_acc_0_50_original, 'num/total: ', manual_acc_0_50_original_num, '/', manual_acc_0_50_total)
print('dir 0 to 50: ', manual_acc_0_50_dir, 'num/total: ', manual_acc_0_50_dir_num, '/', manual_acc_0_50_total)
print()
print('old 50 to 90: ', manual_acc_50_90_old, 'num/total: ', manual_acc_50_90_old_num, '/', manual_acc_50_90_total)
print('original 50 to 90: ', manual_acc_50_90_original, 'num/total: ', manual_acc_50_90_original_num, '/', manual_acc_50_90_total)
print('dir 50 to 90: ', manual_acc_50_90_dir, 'num/total: ', manual_acc_50_90_dir_num, '/', manual_acc_50_90_total)
print()
print('old 90 to 130: ', manual_acc_90_130_old, 'num/total: ', manual_acc_90_130_old_num, '/', manual_acc_90_130_total)
print('original 90 to 130: ', manual_acc_90_130_original, 'num/total: ', manual_acc_90_130_original_num, '/', manual_acc_90_130_total)
print('dir 90 to 130: ', manual_acc_90_130_dir, 'num/total: ', manual_acc_90_130_dir_num, '/', manual_acc_90_130_total)
print()
# Print random accuracy
print('Random Accuracy')
print('old 0 to 50: ', random_acc_0_50_old, 'num/total: ', random_acc_0_50_old_num, '/', random_acc_0_50_total)
print('original 0 to 50: ', random_acc_0_50_original, 'num/total: ', random_acc_0_50_original_num, '/', random_acc_0_50_total)
print('dir 0 to 50: ', random_acc_0_50_dir, 'num/total: ', random_acc_0_50_dir_num, '/', random_acc_0_50_total)
print()
print('old 50 to 90: ', random_acc_50_90_old, 'num/total: ', random_acc_50_90_old_num, '/', random_acc_50_90_total)
print('original 50 to 90: ', random_acc_50_90_original, 'num/total: ', random_acc_50_90_original_num, '/', random_acc_50_90_total)
print('dir 50 to 90: ', random_acc_50_90_dir, 'num/total: ', random_acc_50_90_dir_num, '/', random_acc_50_90_total)
print()
print('old 90 to 130: ', random_acc_90_130_old, 'num/total: ', random_acc_90_130_old_num, '/', random_acc_90_130_total)
print('original 90 to 130: ', random_acc_90_130_original, 'num/total: ', random_acc_90_130_original_num, '/', random_acc_90_130_total)
print('dir 90 to 130: ', random_acc_90_130_dir, 'num/total: ', random_acc_90_130_dir_num, '/', random_acc_90_130_total)
