import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data import *
from dotenv import load_dotenv
from pyngrok import ngrok
import os


load_dotenv()


app = Flask(__name__)
voting_model = joblib.load('voting_model.joblib')

def process_data(data):
    def encode_floor(floor):
        if floor == 'Ground':
            floor = 0
        if floor == '10+':
            floor = 11
        if floor == 'Highest':
            floor = 12
        return int(floor)
    df = pd.DataFrame(data)
    scaler = MinMaxScaler()
    df['Floor'] = df['Floor'].apply(encode_floor)

    # Assuming these columns are numeric
    numeric_cols_new = ['Size', 'Bedrooms', 'Bathrooms', 'Floor']

    # Separating out the numeric columns for normalization
    new_data_numeric = df[numeric_cols_new]

    scaler_new = MinMaxScaler()

    # Scale the numeric columns
    new_data_numeric_scaled = scaler_new.fit_transform(new_data_numeric)

    # Convert back to a DataFrame
    new_data_numeric_scaled = pd.DataFrame(new_data_numeric_scaled, columns=numeric_cols_new)



    # Drop the original numeric columns from the original data
    new_data_non_numeric = df.drop(columns=numeric_cols_new)

    # Concatenate the scaled numeric data and non-numeric data
    new_data_preprocessed = pd.concat([new_data_numeric_scaled, new_data_non_numeric.reset_index(drop=True)], axis=1)
    new_data_preprocessed = pd.get_dummies(new_data_preprocessed, columns=['House_Type', 'Furnished', 'For_rent', 'Region', 'City'])
    
    X_train = get_data()

    missing_cols = set(X_train.columns) - set(new_data_preprocessed.columns)
    for col in missing_cols:
        new_data_preprocessed[col] = 0
        
    new_data_preprocessed = new_data_preprocessed[X_train.columns]


# Reorder columns to match X_train
    return new_data_preprocessed
    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    raw_features = request.get_json()
    features = process_data(raw_features)
    prediction = voting_model.predict(features)
    output = process_output(prediction)
#     return render_template('index.html',prediction_text="price :  {}".format(output))
    return {"price":float(output[0])}


if __name__ == '__main__':
    NGROK_AUTH=os.getenv("NGROK_AUTHEN")
    PORT=5000
    ngrok.set_auth_token = NGROK_AUTH
    tunnel=ngrok.connect(PORT, domain="excited-central-shrimp.ngrok-free.app")
    print("Public URL : ", tunnel.public_url)
    app.run(host="0.0.0.0", port=5000)