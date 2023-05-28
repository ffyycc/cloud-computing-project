import logging
from typing import Tuple, List
import pandas as pd
import numpy as np

def preprocess_dataset(data_path: str, config: dict) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Preprocesses the dataset by handling missing values, converting data types, 
    and dividing into numerical and categorical features.

    Args:
        data_path (str): The path to the dataset file.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        pd.DataFrame: The preprocessed dataset.
        list: The list of numerical columns.
        list: The list of categorical columns.

    """
    all_data = pd.read_csv(data_path)

    # Handling missing values (GARAGE, BUILD_YEAR, NEAREST_SCH_RANK)
    all_data['GARAGE'] = all_data['GARAGE'].fillna(all_data['GARAGE'].median())
    all_data['BUILD_YEAR'] = all_data['BUILD_YEAR'].fillna(all_data['BUILD_YEAR'].quantile(config['quantile']))
    all_data = all_data.drop(['NEAREST_SCH_RANK'], axis=1)

    # Replace float dtype to int dtype in GARAGE and BUILD_YEAR
    cols = ['GARAGE', 'BUILD_YEAR']
    all_data[cols] = all_data[cols].applymap(np.int64)

    # Remove '\r' from DATE_SOLD values and format it to datetime type
    all_data['DATE_SOLD'] = all_data['DATE_SOLD'].str.replace('\r', '')
    all_data['DATE_SOLD'] = pd.to_datetime(all_data['DATE_SOLD'])

    # Divide the dataset into numerical and categorical features
    num_cols = list(all_data.select_dtypes(['int64', 'float64']))
    cat_cols = list(all_data.select_dtypes(['object', 'datetime64[ns]']))
    logging.info('Finished preprocessing the dataset.')

    return all_data, num_cols, cat_cols

def save_dataset(all_data: pd.DataFrame, data_path: str) -> None:
    """
    Saves the dataset to a CSV file.

    Args:
        all_data (pd.DataFrame): The dataset to be saved.
        data_path (str): The path where the dataset should be saved.

    Returns:
        None

    """
    try:
        all_data.to_csv(data_path, index=False)
        logging.info('Data saved successfully to %s', data_path)
    except Exception as e:
        logging.error('Error while saving the data: %s', e)
        raise
