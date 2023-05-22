# Library
import argparse
import datetime
import logging.config
from pathlib import Path
import yaml
import pandas as pd

import src.preprocess_data as pp
import src.clean_data as cd
import src.generate_features as gf
import src.model_tuning as mt
import src.generate_preprocessor as gp
import src.split_data as sd
import src.model_evaluation as me

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

    run_config = config.get("run_config", {})

    # Set up output directory for saving artifacts
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path(run_config.get("output", "runs")) / str(now)
    artifacts.mkdir(parents=True)
    
    # Save config file to artifacts directory for traceability
    config_dir = artifacts / Path(run_config.get("output_config", "config"))
    config_dir.mkdir(parents=True)
    
    data_dir = artifacts / Path(run_config.get("output_data", "data"))
    data_dir.mkdir(parents=True)

    with (config_dir / "config.yaml").open("w") as f:
        yaml.dump(config, f)

    # Create structured dataset from raw data; save to disk
    all_data,num_cols,cat_cols = pp.preprocess_dataset("data/perth_house_info.csv", config["preprocess_dataset"])
    pp.save_dataset(all_data, data_dir / "house_preprocessed.csv")
    logger.info("Finished preprocessing the dataset.")

    ####################################
    # TODO: add EDA here (拽姐 part)
    ####################################
    
    # clean the data
    cleaned_data = cd.clean_dataset(all_data, config["clean_data"])
    logger.info("Finished cleaning the dataset.")
    
    # generated features
    features,updated_num_cols,updated_cat_cols = gf.generate_features(cleaned_data, config["generate_features"])
    # Save the features dataset in the disk
    gf.save_features(features, data_dir / "house_features.csv")
    logger.info("Finished generating features.")
    
    preprocessor = gp.generate_preprocessor(updated_num_cols,updated_cat_cols)
    logger.info("Finished generating preprocessor.")

    X_train_transformed, X_test_transformed, y_train, y_test = sd.split_data(features, preprocessor, config["split_data"])
    sd.save_splited_data(X_train_transformed, X_test_transformed, y_train, y_test, data_dir)
    logger.info("Finished splitting the data.")

    # Tune the models
    rf_model, rf_par = mt.random_forest_tuning(X_train_transformed, y_train,config["model_tuning"])
    xgb_model, xgb_par = mt.xgboost_tuning(X_train_transformed, y_train,config["model_tuning"])
    lr_model, lr_par = mt.linear_ridge_tuning(X_train_transformed, y_train,config["model_tuning"])
    metrics_df, best_model, best_model_name = mt.model_comparison(rf_model, xgb_model, lr_model, X_test_transformed, y_test, config["model_tuning"])
    # Save the metrics and best model
    mt.save_metrics(metrics_df, artifacts)
    mt.save_model(best_model, artifacts / "best_model_object.pkl")
    
    # Model evaluation
    model_results = me.evaluate_model(best_model, X_test_transformed, y_test, config["model_evaluation"]["evaluate_model"])
    fig_dict = me.plot_results(model_results, config["model_evaluation"]["plot_results"])
    me.save_graphs(fig_dict, artifacts / config["model_evaluation"]["plot_results"]["output_dir"])









