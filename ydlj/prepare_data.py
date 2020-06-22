"""
prepare data for training/testing
"""
import json
import random

input_file = r'C:\Works\DataSet\CAIL\ydlj_small_data\train.json'
output_train_file = r'data\train.json'
output_dev_file = r'data\dev.json'

with open(input_file, 'r', encoding='utf-8') as f_in:
    all_data = json.load(f_in)

random.shuffle(all_data)
train_set_ratio = 0.8
train_set_size = int(len(all_data) * 0.8)

with open(output_train_file, 'w', encoding='utf-8') as f_out:
    json.dump(all_data[:train_set_size], f_out, ensure_ascii=False, indent=4)

with open(output_dev_file, 'w', encoding='utf-8') as f_out:
    json.dump(all_data[train_set_size:], f_out, ensure_ascii=False, indent=4)


# input_file = r'C:\Works\DataSet\CAIL\CAIL2019\big_train_data.json'
# output_file = r'data\train_2019.json'

# input_file = r'C:\Works\DataSet\CAIL\CAIL2019\dev_ground_truth.json'
# output_file = r'data\dev_2019.json'

# input_file = r'C:\Works\DataSet\CAIL\CAIL2019\test_ground_truth.json'
# output_file = r'data\test_2019.json'
#
#
# with open(input_file, 'r', encoding='utf-8') as f_in:
#     train_data = json.load(f_in)
#
# with open(output_file, 'w', encoding='utf-8') as f_out:
#     json.dump(train_data, f_out, ensure_ascii=False, indent=4)


