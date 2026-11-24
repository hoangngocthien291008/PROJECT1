import pandas as pd 
import plotly.express as px 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np  
from sklearn.model_selection import train_test_split


np.random.seed(42)  # Chọn một số nguyên bất kỳ làm ID

from pathlib  import Path 

data_dir = Path(__file__).resolve().parent

sample_fraction = 0.01
def skip_row(index):  # lấy từ đầu <bao gồm cột tên>  <vì skip nên true sẽ bỏ qua> 
    if index == 0: 
        return False 
    return  np.random.rand() > sample_fraction 

selected_cols = 'pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count,fare_amount'.split(',')  # không cần cột key 
dtypes = { 
    'fare_amount': 'float32',
    'pickup_longitude': 'float32',
    'pickup_latitude': 'float32',
    'dropoff_longitude': 'float32',
    'passenger_count': 'uint8'
}
# skiprows = True thì bỏ qua 
df = pd.read_csv(data_dir+'/new-york-city-taxi-fare-prediction/train.csv', skiprows = skip_row, usecols = selected_cols, dtype = dtypes, parse_dates=['pickup_datetime']) 
# parse_dates chuyển từ dạng dữ liệu thô sang dạng dữ liệu có công thức 
# nrows chỉ lấy được int (số lượng rows lấy được từ đầu <gồm tên>, còn skip_rows có thể truyền 1 dạng hàm ((int) --> bool))

test_df = pd.read_csv(data_dir+'/test.csv', dtype=dtypes, parse_dates=['pickup_datetime']) # vẫn chừa lại cột key để làm file submission 

# print(df.info()) 
# print(test_df.info())
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42) # trả về 0.8 trước 
train_df = train_df.dropna() 
val_df = val_df.dropna()

# xử lý ngày 
def add_dateparts(df, col): 
    df[col + "_year"] = df[col].dt.year
    df[col + "_month"] = df[col].dt.month
    df[col + "_day"] = df[col].dt.day 
    df[col + "_hour"] = df[col].dt.hour 
    df[col + "_weekday"] = df[col].dt.weekday 

add_dateparts(train_df, 'pickup_datetime') # pichup_datetime đã được xử lý rồi 
add_dateparts(val_df, 'pickup_datetime') 
add_dateparts(test_df, 'pickup_datetime') 


# add distance between pickup and drop  (haversine formula)  
def haversine_(lon1,lon2,la1,la2): 
    lon1, lon2, la1, la2 = map(np.radians, [lon1, lon2, la1, la2]) 
    d_la = la2 - la1 
    d_lon = lon2 - lon1 
    d1 = np.sin(d_la / 2) ** 2 + np.cos(la1) *  np.cos(la2) * (np.sin(d_lon / 2) ** 2) 
    return 2 * 6371 * np.arcsin(np.sqrt(d1)) 

def add_distance(df): 
    df['haversine'] = haversine_(df['pickup_longitude'], df['dropoff_longitude'], df['pickup_latitude'], df['dropoff_latitude'])


add_distance(train_df) 
add_distance(val_df)  
add_distance(test_df)  

# add distances to landmark, for purpose of determing cost  (associated charge)

jfk_lonlat = -73.7781, 40.6413
lga_lonlat = -73.8740, 40.7769
ewr_lonlat = -74.1745, 40.6895
met_lonlat = -73.9632, 40.7794
wtc_lonlat = -74.0099, 40.7126

def add_landmark_dropoff_distance(df, landmark_name, landmark_lonlat, point):  # point will be pickup and dropoff
    lon, lat = landmark_lonlat 
    df[landmark_name+'_'+ point] = haversine_(lon, df[f"{point}_longitude"], lat, df[f"{point}_latitude"]) 

def landmarks_distance(df): 
    landmark_map = {'jfk' : jfk_lonlat, 'lga' : lga_lonlat, 'ewr' : ewr_lonlat, 'met' : met_lonlat, 'wtc' : wtc_lonlat} 
    for point in ['pickup', 'dropoff']: 
        for name, lonlat in landmark_map.items(): 
            add_landmark_dropoff_distance(df, name, lonlat, point) 


landmarks_distance(train_df) 
landmarks_distance(val_df)
landmarks_distance(test_df)

# remove outliers (only five main cols because remained cols are created by five original cols)
# print(test_df.describe())

# với kiểu giá trị ngoại lai thì ta xét dựa vào test_df 
def remove_outliers(df):
    return df[(df['fare_amount'] >= 1.) & 
              (df['fare_amount'] <= 500.) &
              (df['pickup_longitude'] >= -75) & 
              (df['pickup_longitude'] <= -72) & 
              (df['dropoff_longitude'] >= -75) & 
              (df['dropoff_longitude'] <= -72) & 
              (df['pickup_latitude'] >= 40) & 
              (df['pickup_latitude'] <= 42) & 
              (df['dropoff_latitude'] >=40) & 
              (df['dropoff_latitude'] <= 42) & 
              (df['passenger_count'] >= 1) & 
              (df['passenger_count'] <= 6)]


train_df = remove_outliers(train_df)
val_df = remove_outliers(val_df)

train_df.to_parquet('train.parquet') 
val_df.to_parquet('val.parquet')
test_df.to_parquet('test.parquet')

