import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# configure logging
logging.basicConfig(filename='pipeline.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test data and returns a DataFrame
    with the actual and predicted values.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets

    Returns:
        DataFrame with actual and predicted values
    """
    y_pred = model.predict(X_test)
    results = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})

    return results


def plot_results(model_results: pd.DataFrame) -> dict:
    """
    Function to plot the results of the model

    Args:
        model_results: DataFrame containing actual and predicted prices

    Returns:
        dict: Dictionary containing plot figures
    """
    fig_dict = {}
    try:
        # Histogram
        fig1 = model_results.hist(figsize=(20, 8))
        fig_dict["histogram"] = fig1

        # Density plot
        fig2 = model_results.plot(kind='kde', figsize=(20, 8))
        fig_dict["density_plot"] = fig2

        # Scatterplot of actual price vs predicted price   
        fig3 = plt.figure(figsize=(20, 8)) 
        plt.scatter(model_results['Actual Price'], model_results['Predicted Price'], color='green')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual Price vs Predicted Price')    
        fig_dict["scatterplot"] = fig3

        logging.info("Plots created successfully.")
        return fig_dict
    except Exception as e:
        logging.error("Error occurred while generating plots: " + str(e))
        return None


def save_graphs(fig_dict: dict, plot_dir: Path) -> None:
    """
    Function to save the plotted graphs

    Args:
        fig_dict: Dictionary containing plot figures
        plot_dir: Directory to save the plots

    Returns:
        None
    """
    try:
        for fig_name, fig in fig_dict.items():
            fig.savefig(plot_dir / f'{fig_name}.png')
            plt.close(fig)
        
        logging.info("Graphs saved successfully.")
    except Exception as e:
        logging.error("Error occurred while saving graphs: " + str(e))
