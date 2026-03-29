import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(file_path):
    """
    Load dataset from CSV file
    """
    print("Loading dataset...")
    data = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with {data.shape[0]} rows")
    return data

def load_data(path):
    return pd.read_csv(path)


def get_features(data):
    """
    Select input features for the model
    """
    features = data[['Temperature(°C)', 'Humidity(%)']]
    return features


def get_aqi_values(data):
    """
    Target for regression (predict AQI value)
    """
    return data['Air Quality Index(AQI) Value']


def encode_aqi_category(data):
    """
    Convert AQI category (text) into numbers for classification
    """
    encoder = LabelEncoder()
    data['AQI_Category_Encoded'] = encoder.fit_transform(data['AQI Category'])
    
    print("AQI categories encoded successfully")
    return data['AQI_Category_Encoded'], encoder