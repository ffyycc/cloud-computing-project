import pandas as pd

def clean_dataset(all_data:pd.DataFrame, config: dict):
    """
    Cleans the dataset based on specified thresholds defined in the 'config' dictionary.

    Args:
        all_data (DataFrame): The dataset to be cleaned.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        DataFrame: The cleaned dataset.

    """
    # Remove rows where price is above the upper threshold (default: $1.5 million)
    all_data.drop(all_data[all_data['PRICE'] > config['outlier']['upper_threshold'].get('price', 1500000)].index, inplace=True)
    
    # Remove rows where LAND_AREA is above the upper threshold (default: 1500)
    all_data.drop(all_data[all_data['LAND_AREA'] > config['outlier']['upper_threshold'].get('land_area', 1500)].index, inplace=True)
    
    # Remove rows where BUILD_YEAR is below the lower threshold (default: 1950)
    all_data.drop(all_data[all_data['BUILD_YEAR'] < config['outlier']['lower_threshold'].get('build_year', 1950)].index, inplace=True)
    
    # Remove rows where NEAREST_STN_DIST is above the upper threshold (default: 10000)
    all_data.drop(all_data[all_data['NEAREST_STN_DIST'] > config['outlier']['upper_threshold'].get('nearest_stn_dist', 10000)].index, inplace=True)
    
    # Remove rows where NEAREST_SCH_DIST is above the upper threshold (default: 4)
    all_data.drop(all_data[all_data['NEAREST_SCH_DIST'] > config['outlier']['upper_threshold'].get('nearest_sch_dist', 4)].index, inplace=True)
    
    # Remove rows where year is NaN
    all_data.dropna(inplace=True)
    
    return all_data