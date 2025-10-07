import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    return pd.read_csv('diamonds_regression.csv')

@st.cache_resource
def load_model():
    return joblib.load('finalized_model.sav')

data = load_data()

st.title("Diamond Pricing App")

# Show dataframe
if st.sidebar.checkbox("Show Dataframe"):
    st.subheader("Dataframe")
    st.write(data.head(20))

# Show graphs 
if st.sidebar.checkbox("Show Graph"):
    st.subheader("Graphs")
    
    # Pairplot
    sns.set(style='white', font_scale=1.2)  # Adjusted font scale for pairplot
    fig1 = sns.pairplot(data[['carat', 'depth','table', 'price']].head(20), height=8, kind='reg', diag_kind='kde')
    st.pyplot(fig1.fig)
    plt.close(fig1.fig)  # Close to free memory
    
    # Violinplot
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.violinplot(data=data, y='price', ax=ax2)
    ax2.set_title('Price Distribution', fontsize=16)
    st.pyplot(fig2)
    plt.close(fig2)
    
    # Countplot
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.countplot(data=data.head(20), x='carat', ax=ax3)
    ax3.set_title('Carat Count', fontsize=16)
    plt.xticks(rotation=45)
    st.pyplot(fig3)
    plt.close(fig3)
    
### Set Sidebar Options
st.sidebar.title('About')
st.sidebar.info('Change parameters to see how prices change.')

st.sidebar.title('Parameters')

carat = st.sidebar.number_input('Carat', min_value=0.20, max_value=5.0, value=1.0, step=0.1)

depth = st.sidebar.number_input('Depth', min_value=40.0, max_value=80.0, value=60.0, step=0.1)

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
loaded_model = load_model()

prediction = round(loaded_model.predict([[carat,depth] + cut_list + color_list])[0])

st.write(f"### Suggested Price: ${prediction:,}")