# Bulldozer Sale Price Prediction  – Time Series Machine Learning
Learning & Implementation – Time Series Machine Learning

<img width="749" height="436" alt="Screenshot 2026-01-09 at 18 45 21" src="https://github.com/user-attachments/assets/463f5bc4-d461-4f59-be0b-06d8309d6105" />


## Project Purpose

This project is created as a learning-oriented, hands-on implementation of an end-to-end machine learning workflow using a real-world dataset.

The main focus is understanding and practicing:

- How time series constraints affect ML modeling
- How to prepare messy, real-world tabular data
- How to build, evaluate, and interpret a regression model step by step

Dataset used: Kaggle Bluebook for Bulldozers https://www.kaggle.com/c/bluebook-for-bulldozers/data

<img width="630" height="431" alt="plot scatter saledate and SalePrice" src="https://github.com/user-attachments/assets/a7ef2689-ada5-4b1a-a9fd-8d15a035cdb8" />

## Learning Objectives

Through this project, I aimed to learn and practice:

- Time-aware data splitting (avoiding data leakage)
- Feature engineering from date-time data
- Handling missing values in large tabular datasets
- Training and evaluating regression models
- Understanding model performance using RMSLE
- Interpreting feature importance

This project prioritizes process and understanding over leaderboard performance.

## Dataset
Source: Kaggle – [Bluebook for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers/data)

<img width="1108" height="310" alt="first 5 row DataFrame" src="https://github.com/user-attachments/assets/55380191-e73c-49f7-b976-1a883e660138" />

Files Used
File	Description
- Train.csv	Training data (sales before 2012)
- Valid.csv	Validation data (Jan–Apr 2012)
- Test.csv	Test data (May–Nov 2012, no labels)

Why This Dataset?
- Large and realistic
- Many missing values
- Mix of categorical, numerical, and time-based features
- Commonly used to practice tabular ML + time series logic

## Tech Stack

- Python
- Pandas & NumPy – data processing
- Scikit-learn – modeling & evaluation
- Matplotlib – visualization
- Jupyter Notebook / Google Colab

## Machine Learning Workflow
The notebook is structured to follow a learning-friendly ML pipeline.

1. Data Loading
- CSV files loaded efficiently
- Global dtype handling to reduce warnings
- Early inspection of data types and missing values

2. Exploratory Data Analysis (EDA)
- Target distribution (SalePrice)
- Relationship between time (saledate) and price
- Identifying patterns and anomalies

3. Feature Engineering
   <img width="742" height="102" alt="Screenshot 2026-01-09 at 19 07 20" src="https://github.com/user-attachments/assets/ffbaf37c-afda-4860-b7a8-509cbeab58de" />
  Extracted features from saledate:
  - Year
  - Month
  - Day
  - Day of week
  - Day of year

This step helps the model understand temporal patterns in pricing.

4. Handling Missing Values
- Numerical features filled using median
- Categorical features filled using most frequent value
- Ensures consistency between training, validation, and test sets

## Model
Algorithm Used
- RandomForestRegressor

Chosen because:
- Works well on tabular data
- Handles non-linear relationships
- Requires minimal feature scaling
- Provides feature importance for learning purposes

## Hyperparameter Tuning

<img width="827" height="622" alt="Screenshot 2026-01-09 at 19 10 11" src="https://github.com/user-attachments/assets/0e40a65c-38d5-422c-adbd-6a845a13ca79" />

Hyperparameters:
- n_estimators
- max_features
- min_samples_leaf
- min_samples_split
- n_jobs = -1 (parallel processing)

## Evaluation

<img width="237" height="388" alt="prediction format" src="https://github.com/user-attachments/assets/b28ab2d5-a265-4c6e-a7c2-b366e85ba71d" />

Metric
RMSLE (Root Mean Squared Log Error)

Why RMSLE?

- Suitable for skewed price data
- Penalizes large relative errors
- Commonly used in real-world price prediction problems

The goal here is understanding model behavior, not perfect accuracy.

## Model Interpretation

<img width="509" height="263" alt="Screenshot 2026-01-09 at 19 04 32" src="https://github.com/user-attachments/assets/dfa9e489-d760-4c11-9dbb-77d3479fa12f" />

Feature importance analysis is used to:

- See which features influence predictions most
- Understand the role of time-based features
- Improve intuition about the dataset and model

This repository is intended for:

- Learning machine learning step by step
- Understanding time series regression in tabular data
- Reference for future ML projects

## Run Options

- Google Colab (need to configure path to folder)
- Local Jupyter Notebook (need to configure path to folder and install manual conda enviroment, jupyter notebook, pandas, numpy, matplotlib, scikit-learn)

## Key Takeaways

- Real-world ML data is messy and requires careful preprocessing
- Time series problems need different validation strategies
- Feature engineering often matters more than model choice
- Model evaluation should match the problem context

## What This Project Demonstrates

- End-to-end ML workflow
- Time series–aware modeling
- Proper data leakage prevention
- Feature engineering skills
- Model evaluation using appropriate metrics
- Clean, readable, and reproducible code

## Author

Surya Ramadhani

Apple Developer Academy @Infinite Learning Graduade Cohort 2025

Machine Learning Enthusiast

This project is part of my journey in understanding and implementing machine learning concepts through real datasets.
