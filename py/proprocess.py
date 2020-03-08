import numpy as np
import pandas as pd
import gc
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import math
import lightgbm as lgb
from collections import Counter
import time
from scipy.stats import kurtosis,iqr
import seaborn as sns
from scipy import ptp
from tqdm import tqdm
from datetime import timedelta
import datetime
from sklearn.metrics import accuracy_score, roc_auc_score,log_loss,f1_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
from sklearn.utils import shuffle
from sklearn.externals import joblib
from mpl_toolkits.mplot3d import Axes3D
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import warnings
warnings.filterwarnings('ignore')


### 北京市重点区域信息
cols = ['id', 'area_name', 'area_type', 'Center_x', 'Center_y', 'Grid_x', 'Grid_y', 'area']
area_passenger_info = pd.read_csv('/home/poac/AnomalyDetectionDataset/Block_predict/area_passenger_info.csv', names=cols)


### 重点区域人流量情况
cols = ['id', 'date/hour', 'index']
area_passenger_index = pd.read_csv('/home/poac/AnomalyDetectionDataset/Block_predict/area_passenger_index.csv', names=cols)
### 将时间戳拆成月，日和小时
### 去掉年份
area_passenger_index['date'] = area_passenger_index['date/hour'].apply(lambda x: ''.join(str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:8]))
area_passenger_index['date'] = pd.to_datetime(area_passenger_index['date'])
area_passenger_index['hour'] = area_passenger_index['date/hour'].apply(lambda x: int(str(x)[8:]))
### 删除原时间戳
area_passenger_index = area_passenger_index.drop(['date/hour'], axis=1)


### 将北京市重点区域信息和重点区域人流量合并
area_passenger_index = area_passenger_index.merge(area_passenger_info,how = 'left',on = ['id'])
area_passenger_index.shape


### 给数据打上label
### 将每个时刻往后推9天，为了使线上提交测试集的最后一天有数据可用
### 取出对应的index作为预测label
area_passenger_index['label'] = 0
labels = np.zeros((area_passenger_index.shape[0], 1))
labels = pd.DataFrame(labels)
labels.columns = ['label']
max_date = area_passenger_index['date'].iloc[-1]
for index, row in tqdm(area_passenger_index.iterrows()):
    date_forward = row['date']+timedelta(days=9)
    if date_forward <= max_date:
        ### 先找出日期符合的所有数据
        tem = area_passenger_index[area_passenger_index['date']==date_forward]
        ### 再通过id和hour找出唯一符合的数据
        tem = tem[(tem['id']==row['id']) & (tem['hour']==row['hour'])]
        labels.iloc[index]['label'] = tem['index']
    else:
        labels.iloc[index]['label'] = -999

area_passenger_index['label'] = labels


### 保存
area_passenger_index.to_csv('/home/poac/AnomalyDetectionDataset/Block_predict/processed/area_passenger_index.csv', index=None)


### 根据线上测试数，从训练集中找出需要用到的数据
sub = pd.read_csv('/home/poac/code/Block_predict/submit/test_submit_example.csv', names=['id', 'date/hour', 'index'])
sub = sub.merge(area_passenger_info, how='left', on=['id'])
sub['date'] = sub['date/hour'].apply(lambda x: ''.join(str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:8]))
sub['date'] = pd.to_datetime(sub['date'])
sub['hour'] = sub['date/hour'].apply(lambda x: int(str(x)[8:]))
sub = sub.drop(['date/hour'], axis=1)
input_for_sub = pd.DataFrame()
### 将sub的日期往前推9天，找出需要的数据
for index, row in tqdm(sub.iterrows()):
    date_forward = row['date']-timedelta(days=9)
    ### 先找出日期符合的所有数据
    tem = area_passenger_index[(area_passenger_index['date']==date_forward) & (area_passenger_index['id']==row['id']) & (area_passenger_index['hour']==row['hour'])]
    input_for_sub = pd.concat([input_for_sub, tem], axis=0)


### 重置索引
input_for_sub = input_for_sub.reset_index(drop=True)


### 保存
input_for_sub.to_csv('/home/poac/AnomalyDetectionDataset/Block_predict/processed/input_for_sub.csv', index=None)



