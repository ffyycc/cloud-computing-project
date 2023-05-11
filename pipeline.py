# Library
import argparse
import yaml
import logging

from sklearn.model_selection import train_test_split

import src.model_tuning as mt


logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("pipeline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acquire, clean, and create features from housing price data"
    )
    parser.add_argument(
        "--config",
        default="config/config.yml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
            logger.error("Error while loading configuration from %s", args.config)
        else:
            logger.info("Configuration file loaded from %s", args.config)









X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=21)

# Tune the models
rf_model, rf_par = mt.random_forest_tuning(X_train_transformed, y_train,config["model_tuning"])
xgb_model, xgb_par = mt.xgboost_tuning(X_train_transformed, y_train,config["model_tuning"])
lr_model, lr_par = mt.linear_ridge_tuning(X_train_transformed, y_train,config["model_tuning"])
metrics_df, best_model = mt.model_comparison(rf_model, xgb_model, lr_model, X_test, y_test, config["model_tuning"])











