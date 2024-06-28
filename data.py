# !pip install xgboost
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def get_data():
    # Load the dataset
    path=r'./egypt housing data/houses_data_v2.csv'
    data = pd.read_csv(path)

    # Separate numeric and categorical columns
    numeric_cols = data.select_dtypes(include='number').columns
    categorical_cols = data.select_dtypes(exclude='number').columns

    # Impute missing values for numeric columns
    numeric_imputer = SimpleImputer(strategy='mean')  # Impute with mean value
    data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])

    # Impute missing values for categorical columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')  # Impute with most frequent value (mode)
    data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

    def encode_floor(floor):
        if floor == 'Ground':
            floor = 0
        if floor == '10+':
            floor = 11
        if floor == 'Highest':
            floor = 12
        return int(floor)

    data['Floor'] = data['Floor'].apply(encode_floor)

    # Assuming these columns are numeric
    numeric_cols = ['Size', 'Bedrooms', 'Bathrooms', 'Floor', 'Price']
    # Separating out the numeric columns for normalization
    data_numeric = data[numeric_cols]
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Scale the numeric columns
    data_numeric_scaled = scaler.fit_transform(data_numeric)

    # Convert back to a DataFrame
    data_numeric_scaled = pd.DataFrame(data_numeric_scaled, columns=numeric_cols)
    # Drop the original numeric columns from the original data
    data_non_numeric = data.drop(columns=numeric_cols)

    # Concatenate the scaled numeric data and non-numeric data
    data_preprocessed = pd.concat([data_numeric_scaled, data_non_numeric.reset_index(drop=True)], axis=1)


    # Perform one-hot encoding for categorical variables

    data_preprocessed = pd.get_dummies(data_preprocessed , columns=['House_Type', 'Furnished', 'For_rent','Region', 'City'])

    # Split data into features and target
    X = data_preprocessed.drop('Price', axis=1)  # Assume 'Price' is the target
    y = data_preprocessed['Price']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train

def process_output(preds):
    # Load the dataset
    path=r'./egypt housing data/houses_data_v2.csv'
    data = pd.read_csv(path)

    # Separate numeric and categorical columns
    numeric_cols = data.select_dtypes(include='number').columns
    categorical_cols = data.select_dtypes(exclude='number').columns

    # Impute missing values for numeric columns
    numeric_imputer = SimpleImputer(strategy='mean')  # Impute with mean value
    data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])

    # Impute missing values for categorical columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')  # Impute with most frequent value (mode)
    data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

    def encode_floor(floor):
        if floor == 'Ground':
            floor = 0
        if floor == '10+':
            floor = 11
        if floor == 'Highest':
            floor = 12
        return int(floor)

    data['Floor'] = data['Floor'].apply(encode_floor)
    new_data_predictions= (preds *(np.max(data['Price'])-np.min(data['Price'])))+np.min(data['Price'])
    rounded_data_predictions = np.round(new_data_predictions)
    if rounded_data_predictions<0:
        rounded_data_predictions = abs(rounded_data_predictions)
    return rounded_data_predictions