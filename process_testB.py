import pandas as pd
import sys
import operator
from functools import reduce

Test_Sample_Data = []
Test_Sample_Data_columns = ['ad_id', 'Ad_material_size', 'Ad_Industry_Id', 'Commodity_type',
                            'Commodity_id', 'Delivery_time', 'Chose_People', 'bid']

with open('../data/Btest_sample.dat', 'r') as f:
    for i, line in enumerate(f):
        # 测试的时候使用的数据
        # if i >= 3:
        # break
        # sys.exit()

        # 原始数据每列属性的含义 修改数据之后每列属性的含义
        # Sample_id ad_id Creation_time Ad_material_size Ad_Industry_Id Commodity_type Commerce_id Account_id
        # Delivery_time Chose_People ad_bid
        # 'ad_id', 'ad_bid', 'num_click', 'Ad_material_size', 'Ad_Industry_Id', 'Commodity_type', 'Delivery_time',

        # 定义一个临时的数组用于缓存数据集 首先加载的属性是直接能够从原始数据中
        save_line = []
        line = line.strip().split('\t')
        # print("line:", line, '\n', 'line[9]:', line[9], type(line))

        save_line.append(line[1])
        save_line.append(line[3])
        # save_line.append(int_num_click)
        save_line.append(line[4])
        save_line.append(line[5])
        save_line.append(line[6])
        # save_line.append(line[8])
        # save_line.append(line[9])

        # 对于属性中存在的多值属性将其中的逗号转化成空格 验证成功
        tmp_line_6 = line[8].strip().split(',')
        line[8] = ' '.join(tmp_line_6)
        save_line.append(line[8])
        save_line.append(line[9])
        save_line.append(line[10])

        Test_Sample_Data.append(save_line)
        # if i == 2:
        #     print("******最后用于保存结果的数据格式是:\n", Test_Sample_Data[3][10], '\n', len(Test_Sample_Data[3][10]))

# 测试成功！！！！！数据集保存正确
user_feature = pd.DataFrame(Test_Sample_Data, columns = Test_Sample_Data_columns)
user_feature.to_csv('../TestB.csv', index=False)

