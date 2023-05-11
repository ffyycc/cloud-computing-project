# Temperary Library
import pandas as pd
#importing libaries
import pandas as pd
import numpy as np
#need to deal with numerical and categorical data(used in chapter 6)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import category_encoders as ce




# Library
import argparse
import yaml

from sklearn.model_selection import train_test_split

import src.model_tuning as mt






# Temperary Code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acquire, clean, and create features from clouds data"
    )
    parser.add_argument(
        "--config",
        default="config/config.yml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)






all_data = pd.read_csv('data/perth_house_info.csv')

all_data['GARAGE'] = all_data['GARAGE'].fillna(all_data['GARAGE'].median())
all_data['BUILD_YEAR'] = all_data['BUILD_YEAR'].fillna(all_data['BUILD_YEAR'].quantile(0.75))
all_data = all_data.drop(['NEAREST_SCH_RANK'], axis=1)

#replace float dtype to int dtype in GARAGE and BUILD_YEAR
cols = ['GARAGE', 'BUILD_YEAR']
all_data[cols] = all_data[cols].applymap(np.int64)

#remove '\r' from DATE_SOLD values and formate it to datetime type
all_data['DATE_SOLD'] = all_data['DATE_SOLD'].str.replace('\r', '')
all_data['DATE_SOLD'] = pd.to_datetime(all_data['DATE_SOLD'])

num_cols = list(all_data.select_dtypes(['int64', 'float64']))
cat_cols = list(all_data.select_dtypes(['object', 'datetime64[ns]']))

#remove rows where price is above 1.5$ mln
all_data.drop(all_data[all_data['PRICE'] > 1500000].index, inplace=True)
#remove rows where LAND_AREA > 1500, BUILD_YEAR < 1950, NEAREST_STN_DIST > 10000, NEAREST_SCH_DIST > 4
all_data.drop(all_data[all_data['LAND_AREA'] > 1500].index, inplace=True)
all_data.drop(all_data[all_data['BUILD_YEAR'] < 1950].index, inplace=True)
all_data.drop(all_data[all_data['NEAREST_STN_DIST'] > 10000].index, inplace=True)
all_data.drop(all_data[all_data['NEAREST_SCH_DIST'] > 4].index, inplace=True)
#remove rows where year is NaN
all_data.dropna(inplace=True)

all_data['FREE_AREA'] = all_data['LAND_AREA'] - all_data['FLOOR_AREA']
all_data['OTHERS_ROOMS_AREA'] = all_data['FLOOR_AREA'] * 0.25
all_data['GARAGE_AREA'] = (all_data['FLOOR_AREA'] - all_data['OTHERS_ROOMS_AREA']) / (all_data['BEDROOMS'] + all_data['BATHROOMS'])
all_data['BATHROOMS_AREA'] = (all_data['FLOOR_AREA'] - all_data['OTHERS_ROOMS_AREA']) / (all_data['BEDROOMS'] + all_data['GARAGE'])
all_data['BEDROOMS_AREA'] = (all_data['FLOOR_AREA'] - all_data['OTHERS_ROOMS_AREA']) / (all_data['BATHROOMS'] + all_data['GARAGE'])

data = all_data.drop(['FREE_AREA', 'BUILD_YEAR', 'NEAREST_SCH_DIST', 'NEAREST_STN_DIST', 'POSTCODE', 'LONGITUDE', 'CBD_DIST'], axis=1)


y = data['PRICE']
features = data.drop(['ADDRESS', 'DATE_SOLD', 'PRICE'], axis=1)
num_cols = list(features.select_dtypes(['int64', 'float64']))
cat_cols = list(features.select_dtypes('object'))

numerical_transformer = MinMaxScaler(feature_range=(0,1))
#The best way to encode categorical data is to use CatBoostEncoder
categorical_transformer = Pipeline(steps=[
    ('cat_encoder', ce.CatBoostEncoder())
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)



X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=21)

# Fit and transform the training data
X_train_transformed = preprocessor.fit_transform(X_train, y_train)

# Transform the testing data
X_test_transformed = preprocessor.transform(X_test)

# Create pandas DataFrames from the transformed data
columns = ['SUBURB', 'BEDROOMS', 'BATHROOMS', 'GARAGE', 'LAND_AREA', 'FLOOR_AREA',
           'NEAREST_STN', 'LATITUDE', 'NEAREST_SCH', 'OTHERS_ROOMS_AREA',
           'GARAGE_AREA', 'BATHROOMS_AREA', 'BEDROOMS_AREA']

X_train_transformed = pd.DataFrame(X_train_transformed, columns=columns)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=columns)



rf_model, rf_par = mt.random_forest_tuning(X_train_transformed, y_train,config["model_tuning"])
xgb_model, xgb_par = mt.xgboost_tuning(X_train_transformed, y_train,config["model_tuning"])
lr_model, lr_par = mt.linear_ridge_tuning(X_train_transformed, y_train,config["model_tuning"])
metrics_df, best_model = mt.model_comparison(rf_model, xgb_model, lr_model, X_test_transformed, y_test, config["model_tuning"])

print(metrics_df)








