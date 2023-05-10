# cloud-computing-project

Please put data in /data directory

You can view login info in /s3_access directory

s3 raw bucket refer video: https://www.youtube.com/watch?v=eeRl5bjf9Bs

Kaggle link: https://www.kaggle.com/datasets/syuzai/perth-house-prices

Our s3 raw bucket name: **cloud-comp-raw-data-bucket**

## Requirement:
1. Planning and Budgeting: High level estimates of project timelines and cloud infrastructure costs with
accompanying Architecture Diagram (Diagrams.net) and Cost Estimate (AWS Cost Calculator).

2. Data Collection and Preparation: Identify and collect relevant data for their project. They will then
clean and preprocess the data to make it ready for use in training the machine learning model. Raw
and processed data should be stored in the cloud using services such as S3 or RDS.

3. Model Building and Training: Students will use popular machine learning libraries like SkLearn,
TensorFlow, etc. to build and train their machine learning model. They will experiment with different
model architectures and hyperparameters to find the best-performing model. Model training should
leverage cloud compute like EC2 or ECS and cloud storage like S3 or RDS for artifacts and data.

4. Model Deployment: Once the machine learning model is trained, students will deploy it on AWS using
best practices in cloud architecture. They will use services like EC2, ECS, or Lambda to host their model
in a scalable and secure manner.

5. Configuration Files, Logging, and Monitoring: Students will use configuration management to securely
store parameters used in their project, like hyperparameters, API keys, and AWS credentials. They will
implement logging and monitoring to track the performance of their model and ensure it is working
correctly. NOTE: Significant penalty will be applied for hard coding/exposing any API or AWS keys.

6. Good Coding Practices: Students will follow best coding practices, like using version control, writing
readable code (PEP 8), and commenting their code. They will also use tools like linters and formatters
to ensure their code meets quality standards

Due date: May 30

### Week 1 May 10 - May 17
Finish Model (ipynb) (Linear Regression, decision Tree, RF, KNN, XGBooost...)
Finish pipeline (Docker) -> plt., 

### Week 2 May 17 - May 24
Model Deployment (EC2, ECS, or Lambda)
Configuration Files, Logging, and Monitoring

### Week3 May 24 - May 30
Write Report
