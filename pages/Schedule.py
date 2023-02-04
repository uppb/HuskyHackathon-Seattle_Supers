import time
import streamlit as st
from datetime import datetime,date
from streamlit_extras.switch_page_button import switch_page
import datetime
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


if st.session_state['logged_in'] == "false" or 'logged_in' not in st.session_state:
    st.write("You are not logged in, Please head back to main page to log in")
    st.write("Redirecting...")
    time.sleep(2)
    switch_page("TravelBuddy")

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

# def get_distances(data, test_point, dist_type = 'euclidean'):
#     distances = []
#     test_point = test_point.reshape(-1,)
#     for i in range(len(data)):
#           data_point = data[i, :]#.reshape(1, -1)
#           # print(type(test_point[0][0]), data_point.shape, type(data_point[0][0]), test_point.shape)
#           # dist = spatial.distance.cdist(test_point, data_point, metric='euclidean')
#           dist = spatial.distance.euclidean(test_point, data_point)
#           distances.append(dist)
#     return distances
def get_distances(data, test_point, dist_type = 'euclidean'):
    distances = []
    test_point = test_point.reshape(-1,)
    for i in range(len(data)):
          data_point = data[i, :]#.reshape(1, -1)
          # print(data_point.shape, test_point.shape)
          # dist = spatial.distance.cdist(test_point, data_point, metric='euclidean')
          dist = spatial.distance.euclidean(test_point, data_point)
          distances.append(dist)
    return distances

def generate_suggestions(data, test_point,k):
    cols = ['lat', 'long']
    day_data = data[(data['day'] == query_day) & (data['email'] != query_user)]
    # st.write(day_data[cols])
    # st.write(test_point[cols])
    indices = nearest_neighbor(day_data[cols], test_point[cols], k=k)
    results = day_data.iloc[indices]
    results=results[['first_name', 'last_name','preference', 'phone','departure time','return time']]
    depart_indices = nearest_neighbor(results[['departure time']], query_depart_time, k=k)
    depart_results = results.iloc[depart_indices].drop(['return time'],axis=1)
    return_indices = nearest_neighbor(results[['return time']], query_return_time, k=k)
    return_results = results.iloc[return_indices].drop(['departure time'],axis=1)
    return depart_results, return_results

dt = datetime.datetime
df= pd.read_csv(st.secrets["csv"])


df1 = pd.DataFrame(
    [[47.62078962369948, -122.33748919643996]],
    columns=['lat', 'lon'])
st.map(df1)


st.write("Please enter travel details")

date_input= st.date_input( "Enter travel date : " , min_value=dt.today(), max_value=dt.today()+datetime.timedelta(days=7))

d = pd.Timestamp(date_input)
day= d.day_name()
st.write("Day of Travel: " ,day)


option1 = st.selectbox(
    'Select time to leave from home:',
    options=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], format_func=lambda x: str(x)+":00") ##todo make this like time stamps and also for half hour slots

# st.write(option1)
option2 = st.selectbox(
    'Select time to return home:',
    options=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], format_func=lambda x: str(x)+":00")

preference = st.selectbox("Select travel mode preference:" , options=["Walk","Public Transport","Car","Bike"])
# st.button("Submit")


# dfNew = pd.DataFrame(
#    np.random.randn(10, 5))

if st.button("Submit"):
    dfNew = pd.DataFrame({'Date':[date_input],'Day':[day],'Leaving Time':[option1],
                          'Return Time':[option2],'Travel Preference':[preference]})
    st.table(dfNew)
    df = preprocessing(df)
    #   Required input: user email, day, depart time
    # get this from streamlit app
    query_user = st.session_state['key']
    query_day = day
    query_depart_time = option1
    query_return_time = option2
    query_user = str(query_user)

    # get details from dataframe for test point
    test_point = df.loc[(df['email'] == query_user) & (df['day'] == query_day)]
    if test_point.empty:
        st.write("No matches found. Please try another time or day")
    else:
        hide_table_row_index = """
                    <style>
                    thead tr th:first-child {display:none}
                    tbody th {display:none}
                    </style>
                    """

        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        # get suggestions for test point
        depart_results, return_results = generate_suggestions(df, test_point, 4)
        st.write("Leaving time Recommendations: ")
        st.table(depart_results)
        st.write("Return time Recommendations:")
        st.table(return_results)


