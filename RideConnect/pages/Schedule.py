import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df= pd.read_csv("~/PycharmProjects/Hackathon/pages/ride_sharing.csv")

df1 = pd.DataFrame(
    [[47.62078962369948, -122.33748919643996]],
    columns=['lat', 'lon'])
st.map(df1)


st.write("Please enter travel details")

date_input= st.date_input("Enter Travel date:")

d = pd.Timestamp(date_input)
day= d.day_name()
st.write("Day of Travel: " ,day)


option1 = st.selectbox(
    'Select time to leave from home:',
    options=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]) ##todo make this like time stamps and also for half hour slots

# st.write(option1)
option2 = st.selectbox(
    'Select time to return home:',
    options=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    # options=['00:00', '01:00', '02:00', '03:00','04:00','05:00','06:00','07:00','08:00','09:00',
    #          '10:00','11:00','12:00','13:00','14:00','15:00','16:00','17:00','18:00','19:00',
    #          '20:00','21:00','22:00','23:00'])


# st.write(option2[0])

def preprocessing(df):
    # Initialize label encoder
    le = LabelEncoder()
    # Fit and transform the column
    # df['day_code'] = le.fit_transform(df['day'])
    #df['preference_code'] = le.fit_transform(df['preference'])
    # df['lat'] = df['lat'].astype(float)
    df['lat'] = pd.to_numeric(df['lat'])
    df['long'] = pd.to_numeric(df['long'])
    return df
df=preprocessing(df)
def nearest_neighbor(df, test_point, k=4):
    # df=df[df['day_code']==day]
    data = df.to_numpy()
    test_point = np.array(test_point)
    distances = get_distances(data, test_point)
    #print(distances)
    nearest_indices = np.argsort(distances)
    top_indices = nearest_indices[:k]
    #print(data[nearest_indices[1]])
    #print('nearest row indices:', top_indices)
    return top_indices
def get_distances(data, test_point, dist_type = 'euclidean'):
    distances = []
    test_point = test_point.reshape(1, -1)
    for i in range(len(data)):
          data_point = data[i, :].reshape(1, -1)
          # print(type(test_point[0][0]), data_point.shape, type(data_point[0][0]), test_point.shape)
          # dist = spatial.distance.cdist(test_point, data_point, metric='euclidean')
          dist = spatial.distance.euclidean(test_point, data_point)
          distances.append(dist)
    return distances
#   Required input: user email, day, depart time
# get this from streamlit app
query_user = "Pulkit" ##todo: get username from other script
query_day = day
query_depart_time = option1
# get details from dataframe for test point
test_point = df.loc[(df['email']==query_user) &  (df['day']==query_day) & (df['departure time']==query_depart_time)]
def generate_suggestions(data, test_point,k):
    cols = ['lat', 'long']
    day_data = data[(data['day'] == test_point['day'].iloc[0]) & (data['email'] != test_point['email'].iloc[0])]
    indices = nearest_neighbor(day_data[cols], test_point[cols], k=k)
    results = day_data.iloc[indices]
    results=results[['first_name', 'last_name','preference', 'phone','departure time','return time']]
    depart_indices = nearest_neighbor(results[['departure time']], test_point[['departure time']], k=k)
    depart_results = results.iloc[depart_indices].drop(['return time'],axis=1)
    return_indices = nearest_neighbor(results[['return time']], test_point[['return time']], k=k)
    return_results = results.iloc[return_indices].drop(['departure time'],axis=1)
    return depart_results, return_results
# get suggestions for test point
depart_results, return_results = generate_suggestions(df, test_point,4)


