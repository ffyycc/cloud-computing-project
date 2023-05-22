# Clouds Project

This project aims to acquire, clean, process, and analyze data related to clouds in order to develop a machine learning model for predicting cloud properties. The pipeline provided covers the entire process, from data acquisition to model evaluation and storage.

## Fetch Data Overview

This project uses data from the following source:

- Data source: [UCI Machine Learning Repository - Cloud Data](https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/taylor/cloud.data)

The Python script includes a function called `get_data` that downloads the data from the specified URL. The function accepts parameters for the URL, the maximum number of download attempts, the initial wait time between attempts, and a multiplier to increase the wait time between attempts. It returns the downloaded data as bytes if successful, otherwise None.


## Pipeline Overview

The pipeline consists of the following steps:

1. Acquire data from an online repository and save it to disk.
2. Create a structured dataset from the raw data and save it to disk.
3. Enrich the dataset with additional features for model training and save it to disk.
4. Generate statistics and visualizations to summarize the data and save them to disk.
5. Split the data into train and test sets, train the model based on the provided configuration, and save the model and dataset to disk.
6. Score the model on the test set and save the scores to disk.
7. Evaluate the model's performance using various metrics and save the results to disk.
8. Optionally, upload all artifacts to an AWS S3 bucket for storage and future analysis.

## Setting Up the Project

### Clone the Repository
Clone this repository to your local machine using Git:

```bash
$ git clone https://github.com/ffyycc/cloud-computing-project
```

### Install Requirements
Navigate to the cloned repository and install the required packages using `pip`:

```bash
$ cd cloud-computing-project
```

Install requirements for cloud app
```bash
$ pip install -r requirements.txt
```

Install requirements for cloud test
```bash
$ pip install -r requirements.txt
```



## YAML File Overview

The YAML file for user to modify is in the directory ```config/```, we provide a default configuration file ready for use.

This YAML configuration file contains settings for a machine learning pipeline designed to analyze cloud data and predict cloud properties. The file is organized into several sections:

**create_dataset:** Specifies the parameters for creating a structured dataset from the raw data, including the number of rows and columns, the start and end lines for each data segment, the names of the column features, and the target variable name.

**generate_features:** Defines the feature engineering steps to be applied to the dataset, including the calculation of the normalized range for the IR channel, log transformation of the visible entropy, and multiplication of visible entropy and contrast.

**mpl_update:** Sets the customization parameters for Matplotlib visualizations, including color schemes, font styles, and figure sizes.

**train_model:** Configures the model training process, including the model type (Random Forest Classifier), model parameters (number of estimators, maximum depth, and random state), the target variable, the initial features to be used in the model, and the proportion of data to be used for testing.

**score_model:** Specifies the parameters for scoring the model on the test set, including the response columns, target variable, and initial features.

**evaluate_performance:** Provides the response columns for evaluating the model's performance.

**aws:** Configures the AWS S3 settings for optionally uploading artifacts, including STS settings, the S3 bucket name, and the prefix for storing experiments.

To upload artifacts to the S3 bucket, ensure that aws:upload is set to True in the YAML configuration file, and update the bucket_name to your own AWS S3 bucket name.

The below is an example which has the bucket name called cloud-classification-project:

```yaml
aws:
  ...
  upload: True
  bucket_name: cloud-classification-project
  ...
```

By setting `upload` to `True` and specifying the desired S3 bucket name, the pipeline will automatically upload the artifacts to the specified bucket once the process is completed.

## Usage

To run the pipeline, execute the main script with the following command:

### Create the virtual environment
To create a new Python environment, you can use a virtual environment management tool like venv.

Run the following command to create a new virtual environment

```bash
python3 -m venv cloud_env
```

### Activate the virtual environment

On Linux or macOS:
```bash
source cloud_env/bin/activate
```

On Windows:

```bash
cloud_env\Scripts\activate
```

Run the following command to install the packages listed in the requirements.txt file:

```bash
pip install -r requirements.txt
```

To run the pipeline locally, simply execute the following command in your terminal or command prompt:

```bash
python3 pipeline.py
```
This will initiate the pipeline using the default configuration settings. Make sure you have all the required dependencies installed and the appropriate configuration file in place before executing the command.


### Deactivate the virtual environment

To deactivate the virtual environment, input the following command:
```bash
deactivate
```

## How to Run in Docker

We've also provided a Dockerfile that sets up everything, including installing packages and running the pipeline. To run the Docker container, follow the steps below:

### Connect to AWS through SSO

Ensure that the AWS CLI is installed and available

```bash
aws --version
```

Be sure that you have logged in to your account using SSO

```bash
aws --profile personal-sso-admin sso login
```

Set your default AWS profile for the remainder of your shell session:

```bash
export AWS_DEFAULT_PROFILE=personal-sso-admin
```

Confirm that you are authenticated as the proper identity:

```bash
aws sts get-caller-identity
```

Follow the steps below to build and run the Docker image for the app and tests:

### Build the Docker Image for the App

```bash
docker build -t appimage -f dockerfiles/DockerfileApp .
```

### Run the entire model pipeline

```bash
docker run -v ~/.aws:/root/.aws -e AWS_PROFILE=personal-sso-admin 
```
