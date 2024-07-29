import pandas as pd
import numpy as np

def generate_synthetic_data(output_path='data/flight_data.csv', num_samples=1000):
    np.random.seed(42)
    data = {
        'altitude': np.random.uniform(30000, 40000, num_samples),
        'speed': np.random.uniform(400, 600, num_samples),
        'fuel_flow': np.random.uniform(2000, 5000, num_samples),
        'flight_time': np.random.uniform(1, 5, num_samples),
        'route_efficiency': np.random.uniform(0.7, 1, num_samples)
    }
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Synthetic data generated and saved to {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()
