import pandas as pd
import streamlit as st
import sys
sys.path.append('../')
from RideConnect import username

st.write(username)
st.write(st.session_state['key'])
data= pd.read_csv("~/PycharmProjects/Hackathon/pages/ride_sharing.csv")
email=username
# df = data.loc[data['email']==email][['first_name','last_name','email','phone']][:1]
#
# st.title("My Profile")
# st.write("Name:",df["first_name"][0], df["last_name"][0])
# #st.write("LastName:",)
# st.write("Email id:",df["email"][0])
# st.write("Phone Number:",df["phone"][0])
