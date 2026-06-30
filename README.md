# Telemarketing Targeting Classifier 
## Executive Summary
A (nearly) end-to-end machine learning project for identifying targets for marketing was built. The focus of this project is not the machine learning models themselves, but rather its deployment preparation. Models were tracked using MLFlow, an endpoint was built using Gradio with the model being served using FastAPI, and Docker images built into DockerHub.

## Problem Statement
Reaching customers through calls is an effective campaign strategy for making customers subscribe to long-term deposits. However, this strategy is costly due to the external institutions invested in to execute the campaigns. To reduce costs, calling is best done to customers who are likely to subscribe in the first place. 

The goal is to have as much of the contacted customers subscribe to a deposit, thus precision is the main metric. A customer flagged as "positive" means that they are targeted by telemarketing, while a "false positive" means that the customer did not subscribe after being contacted.

Through this machine learning model, customer acquistion costs could be lowered for the marketing and operations team leading to reduced company losses.

## Dataset Acquisition
The dataset is used is modified from https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets/data classified under public domain. It is a dataset containing a total of 45000+ rows.

## Actions Done
- Conducted EDA of the train dataset using Jupyter Notebook, seaborn, and matplotlib to identify necessary actions for features before model training
- Trained logistic regression models and performed cross-validation through GridSearchCV and hyperparameter tuning with experiment tracking using MLFlow
- Modularized code for scalability and used best-performing model for predictions.
- Built an API using FastAPI and built a front-end UI using Gradio
- Built a Docker images through GitHub actions

The picture below shows the endpoint for the ML model:
![alt text](<img/gradio ui.png>)

## Roadblocks encountered
The main focus of this project is the pre-deployment aspects, instead of the development of the classifier itself. Despite not being able to deploy to a cloud service, I learned a lot from debugging the roadblocks encountered.

- Test dataset column has categories not present in train dataset. Solution: Added validation for preprocessing
- MLFlow was not able to log models. Solution: terminate processes using port 5000 and export
- MLFlow was not able to load models from the folder. Solution: Access the 'artifacts' folder hidden by the IDE first, then access the model fromt there.
- Docker building time ran out. Solution: Reduced unnecessary imports from requirements.txt

## Next Steps
- Add data validation for input using Great Expectations
- Document training data and experimentation using DVC
- Deploy model into a cloud service e.g. AWS EC2

## Dataset Acknowledgement
Dataset created by: Paulo Cortez (Univ. Minho) and Sérgio Moro (ISCTE-IUL) @ 2012. Thanks to Berkin Kaplanoğlu for helping with the proper column descriptions.