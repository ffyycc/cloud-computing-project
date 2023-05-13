import pandas as pd

def generate_features(all_data: pd.DataFrame, config: dict):
    """
    Generates additional features based on the given data and configuration.

    Args:
        all_data (pd.DataFrame): The input data.
        config (dict): Configuration parameters.

    Returns:
        pd.DataFrame: The modified data with additional features.
        list: The numerical columns in the modified data.
        list: The categorical columns in the modified data.

    """
    # Calculate additional features
    all_data['FREE_AREA'] = all_data['LAND_AREA'] - all_data['FLOOR_AREA']
    all_data['OTHERS_ROOMS_AREA'] = all_data['FLOOR_AREA'] * config.get('other_room_multiplier', 0.25)
    all_data['GARAGE_AREA'] = (all_data['FLOOR_AREA'] - all_data['OTHERS_ROOMS_AREA']) / (all_data['BEDROOMS'] + all_data['BATHROOMS'])
    all_data['BATHROOMS_AREA'] = (all_data['FLOOR_AREA'] - all_data['OTHERS_ROOMS_AREA']) / (all_data['BEDROOMS'] + all_data['GARAGE'])
    all_data['BEDROOMS_AREA'] = (all_data['FLOOR_AREA'] - all_data['OTHERS_ROOMS_AREA']) / (all_data['BATHROOMS'] + all_data['GARAGE'])

    # Drop unnecessary columns
    data = all_data.drop(['FREE_AREA', 'BUILD_YEAR', 'NEAREST_SCH_DIST', 'NEAREST_STN_DIST', 'POSTCODE', 'LONGITUDE', 'CBD_DIST'], axis=1)

    # Extract features for modeling
    features = data.drop(['ADDRESS', 'DATE_SOLD', 'PRICE'], axis=1)
    num_cols = list(features.select_dtypes(['int64', 'float64']))
    cat_cols = list(features.select_dtypes('object'))

    return data, num_cols, cat_cols


def save_features(all_data: pd.DataFrame, data_path: str):
    """
    Saves the features to a CSV file.

    Args:
        all_data (pd.DataFrame): The features to be saved.
        data_path (str): The path where the features should be saved.

    Returns:
        None

    """
    all_data.to_csv(data_path, index=False)
