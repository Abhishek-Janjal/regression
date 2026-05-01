# Forest Fire Area Prediction

This project predicts the burned area of forest fires using machine learning and deep learning models.

## Overview

The workflow includes:

* Data preprocessing and feature engineering
* Model training with multiple regression algorithms
* Hyperparameter tuning using GridSearchCV
* Evaluation using standard regression metrics
* A neural network implementation using PyTorch

## Dataset

Dataset used: Forest Fires dataset
Source:
https://raw.githubusercontent.com/Abhishek-Janjal/regression/main/forest%2Bfires/forestfires.csv

Target variable:

* `area` (log-transformed using `log1p`)

Features include weather indices and cyclic time features derived from month and day.

## Machine Learning

Pipeline includes:

* StandardScaler and MinMaxScaler
* ColumnTransformer for preprocessing
* GridSearchCV for model selection

Models evaluated:

* Linear Regression
* Lasso, Ridge, ElasticNet
* Decision Tree
* Random Forest
* AdaBoost
* Support Vector Regressor

## Evaluation Metrics

* R² Score
* Adjusted R²
* MSE, RMSE
* MAE, Median Absolute Error
* Explained Variance
* Max Error

## Deep Learning

Model: Multi-layer Perceptron (MLP)

* Input: 12 features
* Hidden layers: 64 → 32
* Output: 1
* Loss: MSELoss
* Optimizer: Adam

## Usage

```bash
git clone https://github.com/your-username/forest-fire-prediction.git
cd forest-fire-prediction
pip install -r requirements.txt
python main.py
```

## Tech Stack

* Python
* NumPy, Pandas
* Scikit-learn
* PyTorch

## Notes

* Target variable is log-transformed to reduce skewness
* Deep learning model can be improved with batch training and tuning
* Additional models like gradient boosting can be explored

## License

MIT License
