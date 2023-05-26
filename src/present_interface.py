import streamlit as st
import pandas as pd
import numpy as np
import itertools
from sklearn.compose import ColumnTransformer
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

def present_interface(model):

    st.title("We can Make House Price Prediction in Perth for You!")

    st.sidebar.header("User Input Parameters")

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
    
    ############################################## UI part above ##############################################
    
    # # example ['Landsdale',3,2,2,420,164,'Greenwood Station',-31.81008664,'LANDSDALE CHRISTIAN SCHOOL',41.0,24.6,24.6,30.75]
    # # Set the constant values for other features
    # other_values = {
    #     'SUBURB': 'Landsdale',
    #     'NEAREST_STN': 'Greenwood Station',
    #     'LATITUDE': -31.81008664,
    #     'NEAREST_SCH': 'LANDSDALE CHRISTIAN SCHOOL'
    # }
    
    # # Combine slider values with the other values
    # user_input = {**other_values, **sliders}

    # # Convert the dictionary to a DataFrame
    # # Note: need to put user_input inside [] to make it a 2D structure
    # user_input_df = pd.DataFrame([user_input])
    # numerical_transformer = MinMaxScaler(feature_range=(0,1))
    # categorical_transformer = Pipeline(steps=[('cat_encoder', ce.CatBoostEncoder())])

    # # Define num_cols and cat_cols
    # num_cols = ['BEDROOMS', 'BATHROOMS', 'GARAGE', 'LAND_AREA', 'FLOOR_AREA', 'LATITUDE', 'OTHERS_ROOMS_AREA', 'GARAGE_AREA', 'BATHROOMS_AREA', 'BEDROOMS_AREA']
    # cat_cols = ['SUBURB', 'NEAREST_STN', 'NEAREST_SCH']

    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('num', numerical_transformer, num_cols),
    #         ('cat', categorical_transformer, cat_cols)
    #     ]
    # )
    
    # # Fit and transform user_input_df with the preprocessor
    # user_input_preprocessed = preprocessor.transform(user_input_df)
    
    # pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])

    
    # # Now you can use user_input_df for prediction
    # price = pipeline.predict(user_input_preprocessed)
    # TODO: change default value to predicted price
    price = 666.66
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

    for i in range(len(slider_items)):
        # Calculate column index
        column_index = i % 3
        key, value = slider_items[i]

        columns[column_index].markdown(f"<p class='key'>{key.lower().replace('_', ' ').title()}:</p> <p class='value'>{value}</p>", unsafe_allow_html=True)

    # Show house price
    st.header("Predicted House Price: ")
    st.markdown(f"""
        <div style="
            color: #ea80fc;
            font-size: 40px;
            font-weight: bold;
            text-shadow: 3px 3px 6px #FF69B4;
        ">
            {price:,.2f}
        </div>
    """, unsafe_allow_html=True)
