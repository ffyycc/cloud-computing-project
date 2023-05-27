import streamlit as st
import pandas as pd
import numpy as np
import itertools
from sklearn.compose import ColumnTransformer
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from PIL import Image
import base64

def get_image_b64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
# Load video file
def get_video_b64(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')

def present_interface(model,preprocessor):
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

    ############################################## UI Left part above ##############################################
    
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
    
    ############################################### UI main part below ###############################################
    # Embed the video using the HTML video tag
    # Set up a video background
    video_path = 'src/img/perth_video.mp4'  # replace with your video path
    video_b64 = get_video_b64(video_path)
    st.markdown(f"""
        <video id="myVideo" width="100%" height="300px" controls autoplay muted playsinline loop>
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
    """, unsafe_allow_html=True)


    
    
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
    
    # Define the levels
    levels = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7']

    # Define corresponding images
    img_dict = {
        'Level 1': 'src/img/level1.png',
        'Level 2': 'src/img/level2.png',
        'Level 3': 'src/img/level3.png',
        'Level 4': 'src/img/level4.png',
        'Level 5': 'src/img/level5.png',
        'Level 6': 'src/img/level6.png',
        'Level 7': 'src/img/level7.png',
    }

    # Add a selectbox to choose level:
    if price <= 400000:
        selected_level = 'Level 1'
    
    elif 400000 <= price < 500000:
        selected_level = 'Level 2'
    
    elif 500000 <= price < 600000:
        selected_level = 'Level 3'
    elif 600000 <= price < 700000:
        selected_level = 'Level 4'
    elif 700000 <= price < 5000000:
        selected_level = 'Level 5'
    elif 5000000 <= price < 6500000:
        selected_level = 'Level 6'
    else:
        selected_level = 'Level 7'
        
    # Display the corresponding image
    if selected_level in img_dict:
        # Create two columns
        col1, col2 = st.columns(2)

        # First column: markdown with price
        col1.markdown(f"""
            <div style="
                color: #ea80fc;
                font-size: 40px;
                font-weight: bold;
                text-shadow: 3px 3px 6px #FF69B4;
                margin-bottom: 50px;
            ">
                $ {price:,.2f}
            </div>
        """, unsafe_allow_html=True)
        
        image_path = 'src/img/best-price.png'
        image_b64 = get_image_b64(image_path)

        col1.markdown(f"""
            <div style="margin-left: 50px;">
                <img src="data:image/png;base64,{image_b64}" width="150" />
            </div>
        """, unsafe_allow_html=True)
        
        # Second column: image
        image = Image.open(img_dict[selected_level])
        col2.image(image, width=300) 
