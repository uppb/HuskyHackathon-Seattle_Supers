import pandas as pd
import time
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import numpy as np
import sys
sys.path.append('../')
from TravelBuddy import username

if st.session_state['logged_in'] == "false" or 'logged_in' not in st.session_state:
    st.write("You are not logged in, Please head back to main page to log in")
    st.write("Redirecting...")
    time.sleep(2)
    switch_page("TravelBuddy")

#st.write(username)
# st.write(st.session_state['key'])
data= pd.read_csv(st.secrets["csv"])
email=st.session_state['key']
df = data.loc[data['email']==email][['first_name','last_name','email','phone']][:1]

# st.write(df)

st.title("My Profile")
st.write("Name:",df["first_name"].values[0], df["last_name"].values[0])
#st.write("LastName:",)
st.write("Email id:",df["email"].values[0])
st.write("Phone Number:",df["phone"].values[0])



