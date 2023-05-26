import streamlit as st
import pandas as pd
import numpy as np
import itertools
from sklearn.compose import ColumnTransformer
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import pdb

def present_interface(model,preprocessor):

    st.title("We can Make House Price Prediction in Perth for You!")

    st.sidebar.header("User Input Parameters")

        
    user_info = {
        'SUBURB': None,
        'BEDROOMS': None,
        'BATHROOMS': None,
        'GARAGE': None,
        'LAND_AREA': None,
        'FLOOR_AREA': None,
        'NEAREST_STN': None,
        'LATITUDE': None,
        'NEAREST_SCH': None,
        'OTHERS_ROOMS_AREA': None,
        'GARAGE_AREA': None,
        'BATHROOMS_AREA': None,
        'BEDROOMS_AREA': None,
    }

    # Define your sliders with keys
    sliders = {
        'BEDROOMS': st.sidebar.slider('BEDROOMS', 0, 8, 3, step=1),
        'BATHROOMS': st.sidebar.slider('BATHROOMS', 0, 5, 2, step=1),
        'GARAGE': st.sidebar.slider('GARAGE', 0, 6, 1, step=1),
        'LAND_AREA': st.sidebar.slider('LAND_AREA', 0, 5000, 1000, step=50),
        'FLOOR_AREA': st.sidebar.slider('FLOOR_AREA', 0, 3500, 1000, step=50),
        'OTHERS_ROOMS_AREA': st.sidebar.slider('OTHERS_ROOMS_AREA', 0, 2000, 1000, step=50),
        'GARAGE_AREA': st.sidebar.slider('GARAGE_AREA', 0, 400, 200, step=50),
        'BATHROOMS_AREA': st.sidebar.slider('BATHROOMS_AREA', 0, 500, 200, step=50),
        'BEDROOMS_AREA': st.sidebar.slider('BEDROOMS_AREA', 0, 1000, 300, step=50),
    }

    with open('config/suburb.txt', 'r') as f:
        suburb_options = [line.strip() for line in f]
    suburbs = st.sidebar.selectbox('SUBURB', suburb_options)

    with open('config/nearest_stn.txt', 'r') as f:
        nearest_stn_options = [line.strip() for line in f]
    nearest_stn = st.sidebar.selectbox('NEAREST_STN', nearest_stn_options)

    with open('config/nearest_sch.txt', 'r') as f:
        nearest_sch_options = [line.strip() for line in f]
    nearest_sch = st.sidebar.selectbox('NEAREST_SCH', nearest_sch_options)

    ############################################## UI part above ##############################################
    
    # example ['Landsdale',3,2,2,420,164,'Greenwood Station',-31.81008664,'LANDSDALE CHRISTIAN SCHOOL',41.0,24.6,24.6,30.75]
    # Set the constant values for other features
    other_values = {
        'SUBURB': suburbs,
        'NEAREST_STN': nearest_stn,
        'LATITUDE': -31.95,
        'NEAREST_SCH': nearest_sch,
        'ADDRESS': ' ',
        'PRICE': 0,
        'DATE_SOLD': ' '
    }

    user_info.update(sliders)
    user_info.update(other_values)
    
    row_df = pd.DataFrame.from_records([user_info])
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])

    # Now you can use user_input_df for prediction
    price = pipeline.predict(row_df)[0]

    print("price:",price)
    
    ############################################### UI part below ###############################################
    st.header("Housing Information:")

    # Set up some CSS properties
    st.markdown("""
        <style>
            .key {
                color: #ff8f00;  /* bright yellow color */
                font-weight: bold;
            }
            .value {
                color: #ffffff;  /* light green color */
            }
        </style>
        """, unsafe_allow_html=True)

    # Initialize columns
    columns = st.columns(3)

    # Create a list of sliders to maintain their order
    slider_items = list(sliders.items())
    sidebox_items = [('Suburb', suburbs), ('Nearest Station',
                                           nearest_stn), ('Nearest School', nearest_sch)]
    all_items = slider_items + sidebox_items

    for i in range(len(all_items)):
        # Calculate column index
        column_index = i % 3
        key, value = all_items[i]

        columns[column_index].markdown(
            f"<p class='key'>{key.lower().replace('_', ' ').title()}:</p> <p class='value'>{value}</p>", unsafe_allow_html=True)

    # Show house price
    st.header("Predicted House Price: ")
    st.markdown(f"""
        <div style="
            color: #ea80fc;
            font-size: 40px;
            font-weight: bold;
            text-shadow: 3px 3px 6px #FF69B4;
        ">
            $ {price:,.2f}
        </div>
    """, unsafe_allow_html=True)
