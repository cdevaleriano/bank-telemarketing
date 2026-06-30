# Telemarketing Targeting Classifier 
## Executive Summary
## Problem Statement
Reaching customers through calls is an effective campaign strategy for making customers subscribe to long-term deposits. However, this strategy is costly due to the external institutions invested in to execute the campaigns. To reduce costs, calling is best done to customers who are likely to subscribe in the first place. 

The goal is to have as much of the contacted customers subscribe to a deposit, thus precision is the main metric. A customer flagged as "positive" means that they are targeted by telemarketing, while a "false positive" means that the customer did not subscribe after being contacted.

Through this machine learning model, customer acquistion costs could be lowered for the marketing and operations team leading to reduced company losses.
## Dataset Acquisition
The dataset is used is modified from https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets/data classified under public domain. It is a dataset containing a total of 45000+ rows. I split the dataset into 5 subsets (May '08 - Dec '08, Jan '09 - Jun '09, Jul '09 - Dec '09, Jan '10 - Jun '10, Jul '10 - Nov '10) for ingesting and model retraining purposes once I built the automation pipeline for ingestion for testing.

## Data Contents


## Roadblocks encountered
- Test dataset column has categories not present in train dataset. Solution: Added validation for preprocessing
- MLFlow was not able to log models. Solution: terminate processes using port 5000 and export
- MLFlow was not able to load models from the folder. Solution: Access the 'artifacts' folder hidden by the IDE first, then access the model fromt there.

## Dataset Acknowledgement
Created by: Paulo Cortez (Univ. Minho) and Sérgio Moro (ISCTE-IUL) @ 2012. Thanks to Berkin Kaplanoğlu for helping with the proper column descriptions.