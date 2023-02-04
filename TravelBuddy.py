import streamlit as st
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
from PIL import Image
import pandas as pd
import numpy as np

df = pd.read_csv(st.secrets["csv"])

import calendar
import matplotlib.pyplot as plt

st.title("TravelBuddy App")
#image = Image.open("Bus-Scooter-App-Getty-Images.webp")
image = Image.open(st.secrets["logo"])
st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

if "logged_in" not in st.session_state:
    st.session_state['logged_in'] = "false"

st.sidebar.write("Please enter your Northeastern id")
username = st.sidebar.text_input('Username')
password = st.sidebar.text_input('Password',type='password')
#button1 = st.sidebar.button("Login")

successUser=""
temp = False
if st.sidebar.button("Login"):
    for i in range(len(df)):
        if df.loc[i,"email"] == username and df.loc[i,"password"] == password:
            temp = True

    if temp == True:
        st.sidebar.success("Logged In as {}".format(username)) ## username is case sensitive
        st.session_state['key'] = username
        st.session_state['logged_in'] = "true"

    else:
        st.sidebar.error("Username and password is incorrect")



