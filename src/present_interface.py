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

    suburbs = st.sidebar.selectbox('SUBURB', ['Alexander Heights', 'Alfred Cove', 'Alkimos', 'Applecross',
                                              'Ardross', 'Armadale', 'Ascot', 'Ashby', 'Ashfield', 'Attadale',
                                              'Atwell', 'Aubin Grove', 'Balcatta', 'Baldivis', 'Balga',
                                              'Ballajura', 'Banksia Grove', 'Bassendean', 'Bateman', 'Bayswater',
                                              'Beaconsfield', 'Beckenham', 'Bedford', 'Bedfordale', 'Beechboro',
                                              'Beeliar', 'Beldon', 'Bellevue', 'Belmont', 'Bennett Springs',
                                              'Bentley', 'Bertram', 'Bibra Lake', 'Bicton', 'Booragoon', 'Boya',
                                              'Brabham', 'Brentwood', 'Brookdale', 'Bull Creek', 'Burns Beach',
                                              'Burswood', 'Butler', 'Byford', 'Calista', 'Camillo',
                                              'Canning Vale', 'Cannington', 'Carine', 'Carlisle', 'Carramar',
                                              'Caversham', 'Champion Lakes', 'Churchlands', 'City Beach',
                                              'Claremont', 'Clarkson', 'Cloverdale', 'Cockburn Central', 'Como',
                                              'Connolly', 'Coogee', 'Coolbellup', 'Coolbinia', 'Cooloongup',
                                              'Cottesloe', 'Craigie', 'Crawley', 'Currambine', 'Daglish',
                                              'Dalkeith', 'Darch', 'Darling Downs', 'Darlington', 'Dayton',
                                              'Dianella', 'Doubleview', 'Duncraig', 'East Cannington',
                                              'East Fremantle', 'East Perth', 'East Victoria Park', 'Eden Hill',
                                              'Edgewater', 'Eglinton', 'Embleton', 'Ferndale', 'Floreat',
                                              'Forrestfield', 'Fremantle', 'Girrawheen', 'Glen Forrest',
                                              'Glendalough', 'Gnangara', 'Golden Bay', 'Gooseberry Hill',
                                              'Gosnells', 'Greenmount', 'Greenwood', 'Guildford', 'Gwelup',
                                              'Hamersley', 'Hamilton Hill', 'Hammond Park', 'Harrisdale',
                                              'Haynes', 'Hazelmere', 'Heathridge', 'Helena Valley',
                                              'Henley Brook', 'Herne Hill', 'High Wycombe', 'Highgate',
                                              'Hilbert', 'Hillarys', 'Hillman', 'Hilton', 'Hocking',
                                              'Huntingdale', 'Iluka', 'Inglewood', 'Innaloo', 'Jandakot',
                                              'Jane Brook', 'Jindalee', 'Jolimont', 'Joondalup', 'Joondanna',
                                              'Kalamunda', 'Kallaroo', 'Karawara', 'Kardinya', 'Karnup',
                                              'Karrinyup', 'Kelmscott', 'Kensington', 'Kenwick', 'Kewdale',
                                              'Kiara', 'Kingsley', 'Kinross', 'Koondoola', 'Koongamia',
                                              'Kwinana Town Centre', 'Landsdale', 'Langford', 'Lathlain', 'Leda',
                                              'Leederville', 'Leeming', 'Lesmurdie', 'Lockridge', 'Lynwood',
                                              'Maddington', 'Madeley', 'Maida Vale', 'Manning', 'Marangaroo',
                                              'Marmion', 'Martin', 'Maylands', 'Medina', 'Melville', 'Menora',
                                              'Merriwa', 'Middle Swan', 'Midland', 'Midvale', 'Mindarie',
                                              'Mirrabooka', 'Morley', 'Mosman Park', 'Mount Claremont',
                                              'Mount Hawthorn', 'Mount Lawley', 'Mount Nasura', 'Mount Pleasant',
                                              'Mount Richon', 'Mullaloo', 'Murdoch', 'Myaree', 'Nedlands',
                                              'Nollamara', 'Noranda', 'North Beach', 'North Coogee',
                                              'North Fremantle', 'North Lake', 'North Perth', 'Northbridge',
                                              "O'Connor", 'Ocean Reef', 'Orelia', 'Osborne Park', 'Padbury',
                                              'Palmyra', 'Parkwood', 'Parmelia', 'Pearsall', 'Peppermint Grove',
                                              'Piara Waters', 'Port Kennedy', 'Queens Park', 'Quinns Rocks',
                                              'Redcliffe', 'Ridgewood', 'Riverton', 'Rivervale', 'Rockingham',
                                              'Rossmoyne', 'Safety Bay', 'Salter Point', 'Samson', 'Scarborough',
                                              'Secret Harbour', 'Seville Grove', 'Shelley', 'Shenton Park',
                                              'Shoalwater', 'Sinagra', 'Singleton', 'Sorrento',
                                              'South Fremantle', 'South Guildford', 'South Lake', 'South Perth',
                                              'Southern River', 'Spearwood', 'St James', 'Stirling', 'Stratton',
                                              'Subiaco', 'Success', 'Swan View', 'Swanbourne', 'Tapping',
                                              'Thornlie', 'Treeby', 'Trigg', 'Tuart Hill', 'Victoria Park',
                                              'Viveash', 'Waikiki', 'Walliston', 'Wangara', 'Wanneroo',
                                              'Warnbro', 'Warwick', 'Waterford', 'Watermans Bay', 'Wattle Grove',
                                              'Wellard', 'Welshpool', 'Wembley', 'Wembley Downs',
                                              'West Leederville', 'West Perth', 'West Swan', 'Westminster',
                                              'White Gum Valley', 'Willagee', 'Willetton', 'Wilson', 'Winthrop',
                                              'Woodbridge', 'Woodlands', 'Woodvale', 'Yanchep', 'Yangebup',
                                              'Yokine'])

    nearest_stn = st.sidebar.selectbox('NEAREST_STN', ['Armadale Station', 'Ashfield Station', 'Bassendean Station',
                                                       'Bayswater Station', 'Beckenham Station', 'Belmont Park Station',
                                                       'Bull Creek Station', 'Burswood Station', 'Butler Station',
                                                       'Canning', 'Canning Bridge Station', 'Carlisle Station',
                                                       'Challis Station', 'City West Station', 'Claisebrook Station',
                                                       'Claremont Station', 'Clarkson Station',
                                                       'Cockburn Central Station', 'Cottesloe Station',
                                                       'Currambine Station', 'Daglish Station', 'East Guildford Station',
                                                       'East Perth Station', 'Edgewater Station', 'Esplanade Station',
                                                       'Fremantle Station', 'Glendalough Station', 'Gosnells Station',
                                                       'Grant Street Station', 'Greenwood Station', 'Guildford Station',
                                                       'Joondalup Station', 'Karrakatta Station', 'Kelmscott Station',
                                                       'Kenwick Station', 'Kwinana Station', 'Leederville Station',
                                                       'Loch Street Station', 'Madding', 'Mandurah Station',
                                                       'Maylands Station', 'McIver Station', 'Meltham Station',
                                                       'Midland Station', 'Mosman Park Station', 'Mount Lawley Station',
                                                       'Murdoch Station', 'North Fremantle Station',
                                                       'Oats Street Station', 'Perth Station', 'Queens Park Station',
                                                       'Rockingham Station', 'Seaforth Station', 'Shen',
                                                       'Sherwood Station', 'Stirling Station', 'Subiaco Station',
                                                       'Success Hill Station', 'Swanbourne Station', 'Thornlie Station',
                                                       'Vic', 'Warnbro Station', 'Warwick Station', 'Wellard Station',
                                                       'Welshpool Station', 'West Leederville Station',
                                                       'Whitfords Station', 'Woodbridge Station'])

    nearest_sch = st.sidebar.selectbox('NEAREST_SCH', ['ALKIMOS BAPTIST COLLEGE', "ALL SAINTS' COLLEGE", 'ALTA-1',
                                                       'APPLECROSS SENIOR HIGH SCHOOL', 'AQUINAS COLLEGE',
                                                       'ARANMORE CATHOLIC COLLEGE', 'ARMADALE SENIOR HIGH SCHOOL',
                                                       'ASHDALE SECONDARY COLLEGE', 'ATWELL COLLEGE',
                                                       'AUSTRALIAN ISLAMIC COLLEGE - KEWDALE',
                                                       'BALCATTA SENIOR HIGH SCHOOL', 'BALDIVIS SECONDARY COLLEGE',
                                                       'BALGA SENIOR HIGH SCHOOL', 'BALLAJURA COMMUNITY COLLEGE',
                                                       'BELMONT CITY COLLEGE', 'BELRIDGE SECONDARY COLLEGE',
                                                       'BOLD PARK COMMUNITY SCHOOL', 'BUTLER COLLEGE',
                                                       'BYFORD SECONDARY COLLEGE', 'CANNING COLLEGE',
                                                       'CANNING VALE COLLEGE', 'CAREY BAPTIST COLLEGE',
                                                       'CARINE SENIOR HIGH SCHOOL', 'CARMEL SCHOOL',
                                                       'CECIL ANDREWS COLLEGE', 'CHISHOLM CATHOLIC COLLEGE',
                                                       'CHRIST CHURCH GRAMMAR SCHOOL', "CHRISTIAN BROTHERS' COLLEGE",
                                                       'CHURCHLANDS SENIOR HIGH SCHOOL', 'CLARKSON COMMUNITY HIGH SCHOOL',
                                                       'CLONTARF ABORIGINAL COLLEGE', 'COMET BAY COLLEGE',
                                                       'COMMUNICARE ACADEMY', 'COMO SECONDARY COLLEGE',
                                                       'CORPUS CHRISTI COLLEGE', 'CYRIL JACKSON SENIOR CAMPUS',
                                                       'DALE CHRISTIAN SCHOOL', 'DARLING RANGE SPORTS COLLEGE',
                                                       'DIANELLA SECONDARY COLLEGE', 'DIVINE MERCY COLLEGE',
                                                       'DUNCRAIG SENIOR HIGH SCHOOL', 'EMMANUEL CATHOLIC COLLEGE',
                                                       'EMMANUEL CHRISTIAN COMMUNITY SCHOOL', 'FOUNTAIN COLLEGE',
                                                       'FREMANTLE COLLEGE', 'GILMORE COLLEGE',
                                                       'GIRRAWHEEN SENIOR HIGH SCHOOL',
                                                       'GOVERNOR STIRLING SENIOR HIGH SCHOOL', 'GREENWOOD COLLEGE',
                                                       'GUILDFORD GRAMMAR SCHOOL', 'HALE SCHOOL',
                                                       'HAMPTON SENIOR HIGH SCHOOL', 'HELENA COLLEGE',
                                                       'HERITAGE COLLEGE PERTH', 'HILLSIDE CHRISTIAN COLLEGE',
                                                       'INTERNATIONAL SCHOOL OF WESTERN AUSTRALIA',
                                                       'IONA PRESENTATION COLLEGE', 'IRENE MCCORMACK CATHOLIC COLLEGE',
                                                       'JOHN CALVIN CHRISTIAN COLLEGE', 'JOHN CURTIN COLLEGE OF THE ARTS',
                                                       'JOHN FORREST SECONDARY COLLEGE',
                                                       'JOHN SEPTIMUS ROE ANGLICAN COMMUNITY SCHOOL',
                                                       'JOHN WOLLASTON ANGLICAN COMMUNITY SCHOOL', 'JOHN XXIII COLLEGE',
                                                       'JOSEPH BANKS SECONDARY COLLEGE', 'KALAMUNDA SENIOR HIGH SCHOOL',
                                                       'KELMSCOTT SENIOR HIGH SCHOOL', 'KENNEDY BAPTIST COLLEGE',
                                                       'KENT STREET SENIOR HIGH SCHOOL', 'KIARA COLLEGE',
                                                       'KINGSWAY CHRISTIAN COLLEGE', 'KOLBE CATHOLIC COLLEGE',
                                                       'LA SALLE COLLEGE', 'LAKE JOONDALUP BAPTIST COLLEGE',
                                                       'LAKELAND SENIOR HIGH SCHOOL', 'LANDSDALE CHRISTIAN SCHOOL',
                                                       'LANGFORD ISLAMIC COLLEGE', 'LEEMING SENIOR HIGH SCHOOL',
                                                       'LESMURDIE SENIOR HIGH SCHOOL', 'LIVING WATERS LUTHERAN COLLEGE',
                                                       'LUMEN CHRISTI COLLEGE', 'LYNWOOD SENIOR HIGH SCHOOL',
                                                       'MANDURAH BAPTIST COLLEGE', 'MATER DEI COLLEGE',
                                                       'MELVILLE SENIOR HIGH SCHOOL', 'MERCEDES COLLEGE', 'MERCY COLLEGE',
                                                       "METHODIST LADIES' COLLEGE", 'MINDARIE SENIOR COLLEGE',
                                                       'MORLEY SENIOR HIGH SCHOOL', 'MOUNT LAWLEY SENIOR HIGH SCHOOL',
                                                       'NEWMAN COLLEGE', 'NORTH LAKE SENIOR CAMPUS',
                                                       'OCEAN REEF SENIOR HIGH SCHOOL', 'ONESCHOOL GLOBAL WA',
                                                       'PENRHOS COLLEGE', 'PERTH COLLEGE', 'PERTH MODERN SCHOOL',
                                                       'PERTH WALDORF SCHOOL', 'PETER CARNLEY ANGLICAN COMMUNITY SCHOOL',
                                                       'PETER MOYES ANGLICAN COMMUNITY SCHOOL', 'PORT SCHOOL',
                                                       'PRENDIVILLE CATHOLIC COLLEGE', 'PRESBYTERIAN LADIES COLLEGE',
                                                       'PROVIDENCE CHRISTIAN COLLEGE', 'QUINNS BAPTIST COLLEGE',
                                                       'REHOBOTH CHRISTIAN COLLEGE', 'ROCKINGHAM MONTESSORI SCHOOL',
                                                       'ROCKINGHAM SENIOR HIGH SCHOOL', 'ROSSMOYNE SENIOR HIGH SCHOOL',
                                                       'SACRED HEART COLLEGE', 'SAFETY BAY SENIOR HIGH SCHOOL',
                                                       'SANTA MARIA COLLEGE', 'SCOTCH COLLEGE', 'SERVITE COLLEGE',
                                                       'SETON CATHOLIC COLLEGE', 'SEVENOAKS SENIOR COLLEGE',
                                                       'SHENTON COLLEGE', 'SOUTH COAST BAPTIST COLLEGE',
                                                       'SOUTH METROPOLITAN YOUTH LINK COMMUNITY COLLEGE',
                                                       'SOUTHERN RIVER COLLEGE', 'SOWILO COMMUNITY HIGH SCHOOL',
                                                       "ST ANDREW'S GRAMMAR", "ST BRIGID'S COLLEGE", "ST CLARE'S SCHOOL",
                                                       "ST FRANCIS' SCHOOL", "ST GEORGE'S ANGLICAN GRAMMAR SCHOOL",
                                                       "ST HILDA'S ANGLICAN SCHOOL FOR GIRLS",
                                                       "ST MARK'S ANGLICAN COMMUNITY SCHOOL",
                                                       "ST MARY'S ANGLICAN GIRLS' SCHOOL", 'ST NORBERT COLLEGE',
                                                       "ST STEPHEN'S SCHOOL", 'SWAN CHRISTIAN COLLEGE',
                                                       'SWAN VALLEY ANGLICAN COMMUNITY SCHOOL',
                                                       'SWAN VIEW SENIOR HIGH SCHOOL', "THE KING'S COLLEGE",
                                                       'THE MONTESSORI SCHOOL', 'THORNLIE SENIOR HIGH SCHOOL',
                                                       'TRANBY COLLEGE', 'TREETOPS MONTESSORI SCHOOL', 'TRINITY COLLEGE',
                                                       'URSULA FRAYNE CATHOLIC COLLEGE', 'WANNEROO SECONDARY COLLEGE',
                                                       'WARNBRO COMMUNITY HIGH SCHOOL', 'WARWICK SENIOR HIGH SCHOOL',
                                                       'WESLEY COLLEGE', 'WILLETTON SENIOR HIGH SCHOOL',
                                                       'WOODVALE SECONDARY COLLEGE', 'YANCHEP SECONDARY COLLEGE',
                                                       'YOUTH FUTURES COMMUNITY SCHOOL'])

    ############################################## UI part above ##############################################
    
    # example ['Landsdale',3,2,2,420,164,'Greenwood Station',-31.81008664,'LANDSDALE CHRISTIAN SCHOOL',41.0,24.6,24.6,30.75]
    # Set the constant values for other features
    other_values = {
        'SUBURB': 'Landsdale',
        'NEAREST_STN': 'Greenwood Station',
        'LATITUDE': -31.95,
        'NEAREST_SCH': 'LANDSDALE CHRISTIAN SCHOOL',
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
