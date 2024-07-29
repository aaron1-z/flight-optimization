import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data, data_scaled, scaler
