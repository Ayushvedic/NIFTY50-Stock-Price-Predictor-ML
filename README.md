# NIFTY50-Stock-Price-Predictor-ML
Nifty-50 Value Predictor
This project implements a Linear Regression model from scratch to predict the closing prices of stocks in the NIFTY-50 index based on historical market data. It includes data preprocessing, feature engineering, model training with gradient descent, and accuracy evaluation.

Project Overview
Goal: Predict monthly closing prices of NIFTY-50 stocks using key market features.

Data: Historical stock data with features like Open, High, Low, Volume, VWAP, and others.

Method: Custom-built Linear Regression model optimized via gradient descent.

Evaluation: Model performance measured using Mean Squared Error (MSE) and R-squared (R²).

Features
Robust data cleaning and preprocessing, handling missing and invalid values.

Feature normalization using standard scaling to improve training stability.

Manual implementation of gradient descent for weight and bias optimization.

Calculation of model accuracy with R² score to assess prediction quality.

Model persistence using Python's pickle module for saving and loading.

How to Use
Prepare your data in CSV format with necessary columns.

Run the training script to fit the model on your data.

Evaluate the trained model with testing data.

Save the trained model for later use and reload it as needed.

Technologies Used
Python (NumPy, Pandas)

Custom linear algebra and optimization algorithms

Data preprocessing techniques

Future Improvements
Extend model to support other regression algorithms.

Add a user-friendly interface (e.g., web app or CLI for predictions).

Incorporate additional financial indicators and feature selection.

Improve error handling and input validations.
