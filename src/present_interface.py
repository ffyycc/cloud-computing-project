import streamlit as st
import pandas as pd
import numpy as np

def present_interface(scores):

    st.title("Cloud Type Prediction")

    st.sidebar.header("User Input Parameters")

    # Define your sliders
    sliders = {
        # Define sliders for each variable
        st.sidebar.slider('BEDROOMS', 0, 10, 3, step=1),
        st.sidebar.slider('BATHROOMS', 0, 10, 2, step=1),
        st.sidebar.slider('GARAGE', 0, 10, 1, step=1),
        st.sidebar.slider('LAND_AREA', 0, 5000, 1000, step=50),
        st.sidebar.slider('FLOOR_AREA', 0, 5000, 1000, step=50),
        st.sidebar.slider('OTHERS_ROOMS_AREA', 0, 5000, 1000, step=50),
        st.sidebar.slider('GARAGE_AREA', 0, 5000, 500, step=50),
        st.sidebar.slider('BATHROOMS_AREA', 0, 5000, 200, step=50),
        st.sidebar.slider('BEDROOMS_AREA', 0, 5000, 300, step=50),
    }

    st.header("User Current Input") 
    st.write(sliders)

    # Show classifcation result
    st.header("Type of Cloud")
    # Display the mean in a large, bold, and colorful font
    type = 0
    st.markdown(f"""
        <div style="
            color: #39FF14;
            font-size: 40px;
            font-weight: bold;
            text-shadow: 3px 3px 6px #000000;
        ">
            {type}
        </div>
    """, unsafe_allow_html=True)

    # Show a probability (normalized nice number)
    st.header("Likelihood")
    probability = 0
    st.markdown(f"""
        <div style="
            color: #39FF14;
            font-size: 40px;
            font-weight: bold;
            text-shadow: 3px 3px 6px #000000;
        ">
            {probability:,.2f}
        </div>
    """, unsafe_allow_html=True)
