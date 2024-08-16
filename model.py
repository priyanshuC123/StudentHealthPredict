import pickle
import pandas as pd

# Load the saved model
with open('best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the column names
with open('column_names.pkl', 'rb') as file:
    column_names = pickle.load(file)

def preprocess_input(input_data):
    # Convert the input into a DataFrame
    df = pd.DataFrame([input_data])

    # Remove 'Do you have Depression?' if it's included in the input data
    if 'Do you have Depression?' in df.columns:
        df = df.drop(columns=['Do you have Depression?'])

    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df)

    # Reindex to ensure it matches the training data columns
    df_encoded = df_encoded.reindex(columns=column_names, fill_value=0)

    # Scale the features
    df_scaled = scaler.transform(df_encoded)

    # Log the processed input data
    print("Processed input data:")
    print(df_scaled)

    return df_scaled

def predict_depression(input_data):
    # Log the raw input data
    print("Raw input data:")
    print(input_data)
    
    # Preprocess the input data
    df_processed = preprocess_input(input_data)
    
    # Predict the risk score
    prediction_proba = best_model.predict_proba(df_processed)[:, 1]
    risk_score = prediction_proba[0] * 100

    # Log the resulting risk score
    print("Risk score:", risk_score)

    return risk_score