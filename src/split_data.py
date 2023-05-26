import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Any

def split_data(data: Any, preprocessor: Any, config: dict) -> tuple:
    """
    Splits the data into train and test sets and applies the preprocessor transformation.

    Args:
        data (Any): The input data.
        preprocessor (Any): The preprocessor object.
        config (dict): Configuration parameters.

    Returns:
        tuple: The transformed train and test data along with the corresponding labels.

    """
    y = data[config['target']]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=config['test_size'], random_state=config['random_state'])

    # Fit and transform the training data
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)

    # Transform the testing data
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor


def save_splited_data(X_train_transformed: np.ndarray, X_test_transformed: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, data_dir: Path):
    """
    Saves the split data to separate CSV files.

    Args:
        X_train_transformed (np.ndarray): The transformed training data.
        X_test_transformed (np.ndarray): The transformed testing data.
        y_train (np.ndarray): The labels for the training data.
        y_test (np.ndarray): The labels for the testing data.
        data_dir (Path): The directory where the data should be saved.

    Returns:
        None

    """
    np.savetxt(data_dir / "X_train_transformed.csv", X_train_transformed, delimiter=",")
    np.savetxt(data_dir / "X_test_transformed.csv", X_test_transformed, delimiter=",")

    # Assuming y_train and y_test are 1D, we might want to save them in a column format
    np.savetxt(data_dir / "y_train.csv", y_train, delimiter=",", fmt="%s")
    np.savetxt(data_dir / "y_test.csv", y_test, delimiter=",", fmt="%s")
