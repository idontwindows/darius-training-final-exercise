import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', font_scale=5)

@st.cache
def load_data():
    return pd.read_csv('diamonds_regression.csv')

data = load_data()

st.title("Diamond Pricing App")

# Show datafrome
if st.sidebar.checkbox("Show Dataframe"):
    st.subheader("Dataframe")
    st.write(data.head(20))

# Show graphs 
if st.sidebar.checkbox("Show Graph"):
    st.subheader("Graphs")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    columns = ['carat', 'price']
    sns.pairplot(data[['carat', 'depth','table', 'price']].head(20),height=8, kind='reg', diag_kind='kde')
    st.pyplot()
    sns.set(font_scale = 5)
    data_final = data.head(100)
    sns.violinplot(data['price'])
    st.pyplot()
    sns.countplot(data['carat'].head(20))
    st.pyplot()
    
### Set Sidebar Options
st.sidebar.title('About')
st.sidebar.info('Change parameters to see how prices change.')

st.sidebar.title('Parameters')

carat = st.sidebar.number_input('Carat',0.50)

depth = st.sidebar.number_input('Depth',0.50)

cut = st.sidebar.selectbox("Cut", ['Fair','Good','Ideal','Premium','Very Good'])

if cut == 'Fair':
    cut_list = [1,0,0,0,0]
elif cut == 'Good':
    cut_list = [0,1,0,0,0]
elif cut == 'Ideal':
    cut_list = [0,0,1,0,0]
elif cut == 'Premium':
    cut_list = [0,0,0,1,0]
elif cut == 'Very Good':
    cut_list = [0,0,0,0,1]

color = st.sidebar.selectbox("Color", ['D','E','F','G','H','I','J'])

if color == 'D':
    color_list = [1,0,0,0,0,0,0]
elif color == 'E':
    color_list = [0,1,0,0,0,0,0]
elif color == 'F':
    color_list = [0,0,1,0,0,0,0]
elif color == 'G':
    color_list = [0,0,0,1,0,0,0]
elif color == 'H':
    color_list = [0,0,0,0,1,0,0]
elif color == 'I':
    color_list = [0,0,0,0,0,1,0]
elif color == 'J':
    color_list = [0,0,0,0,0,0,1]

# Prediction
st.subheader("Prediction")
filename = 'finalized_model.sav'
loaded_model = joblib.load(filename)

prediction = round(loaded_model.predict([[carat,depth] + cut_list + color_list])[0])

st.write(f"Suggested Price is: {prediction}")