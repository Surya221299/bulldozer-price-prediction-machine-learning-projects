# Bulldozer Sale Price Prediction  – Time Series Machine Learning
Learning & Implementation – Time Series Machine Learning

## Project Purpose

This project is created as a learning-oriented, hands-on implementation of an end-to-end machine learning workflow using a real-world dataset.

The main focus is understanding and practicing:

- How time series constraints affect ML modeling
- How to prepare messy, real-world tabular data
- How to build, evaluate, and interpret a regression model step by step

Dataset used: Kaggle Bluebook for Bulldozers https://www.kaggle.com/c/bluebook-for-bulldozers/data

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

Hyperparameters:
- n_estimators
- max_features
- min_samples_leaf
- min_samples_split
- n_jobs = -1 (parallel processing)

## Evaluation
Metric
RMSLE (Root Mean Squared Log Error)

Why RMSLE?

- Suitable for skewed price data
- Penalizes large relative errors
- Commonly used in real-world price prediction problems

The goal here is understanding model behavior, not perfect accuracy.

## Model Interpretation

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
