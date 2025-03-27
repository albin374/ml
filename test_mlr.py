import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

print("Testing MLR model...")

# Load the model
try:
    with open('mlr_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"Model type: {type(model)}")
    print(f"Model has predict method: {hasattr(model, 'predict')}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Test prediction
try:
    # Test data
    test_data = pd.DataFrame({
        'Win_Margin': [12],
        'Toss_Decision': [1]  # 1 for Bat
    })
    
    print("\nTest data:")
    print(test_data)
    
    # Make prediction
    prediction = model.predict(test_data)
    print(f"\nPrediction for Win_Margin=12, Toss_Decision=Bat: {prediction[0]:.2f} runs")
    
except Exception as e:
    print(f"Error making prediction: {e}")
    exit(1) 