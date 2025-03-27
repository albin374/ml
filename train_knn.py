import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pickle

print("Loading and preprocessing data...")

# Load the cricket dataset
data = pd.read_csv('data/cricket_dataset_knn.csv')

print("\nData Info:")
print(data.info())
print("\nSample of data:")
print(data.head())

# Calculate the threshold for high/low scoring
runs_median = data['Runs_Scored'].median()
data['High_Low_Score'] = (data['Runs_Scored'] > runs_median).astype(int)

print(f"\nRuns Scored Statistics:")
print(f"Median (threshold): {runs_median:.2f}")
print(f"Mean: {data['Runs_Scored'].mean():.2f}")
print(f"Min: {data['Runs_Scored'].min():.2f}")
print(f"Max: {data['Runs_Scored'].max():.2f}")

# Prepare features and target
X = data[['Innings', 'Runs_Scored']]
y = data['High_Low_Score']

print("\nFeature Statistics:")
print(X.describe())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\nTraining K-Nearest Neighbors model...")

# Create and train the model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nK-Nearest Neighbors Results:")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Plot decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(X['Innings'], X['Runs_Scored'], c=y, cmap='coolwarm', alpha=0.6)
plt.axhline(y=runs_median, color='g', linestyle='--', label='Median Runs (Threshold)')
plt.xlabel('Innings')
plt.ylabel('Runs Scored')
plt.title('KNN: High/Low Scoring Classification')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('knn_results.png')
plt.close()

# Save the model and related data
model_data = {
    'model': model,
    'scaler': scaler,
    'runs_median': runs_median,
    'stats': {
        'accuracy': accuracy,
        'mean_runs': data['Runs_Scored'].mean(),
        'std_runs': data['Runs_Scored'].std()
    }
}

print("\nSaving model and related data...")

with open('knn_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model and scaler saved as 'knn_model.pkl'")

# Test prediction
print("\nTesting model with sample data...")
test_cases = [
    {'innings': 1, 'runs': 150},
    {'innings': 1, 'runs': 250},
    {'innings': 2, 'runs': 180},
    {'innings': 2, 'runs': 280}
]

print("\nSample Predictions:")
print("Innings | Runs | Prediction")
print("-" * 40)

for case in test_cases:
    # Scale the input
    input_data = np.array([[case['innings'], case['runs']]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]
    confidence = prob[prediction]
    
    result = "High" if prediction == 1 else "Low"
    print(f"{case['innings']:7} | {case['runs']:4} | {result:10} (confidence: {confidence:.2f})")

# Verify model can be loaded and used
print("\nVerifying model loading and prediction...")
try:
    with open('knn_model.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    
    test_case = {'innings': 2, 'runs': 286}
    input_data = np.array([[test_case['innings'], test_case['runs']]])
    input_scaled = loaded_data['scaler'].transform(input_data)
    prediction = loaded_data['model'].predict(input_scaled)[0]
    prob = loaded_data['model'].predict_proba(input_scaled)[0]
    confidence = prob[prediction]
    
    result = "High" if prediction == 1 else "Low"
    print(f"\nTest prediction for innings {test_case['innings']}, runs {test_case['runs']}:")
    print(f"Classification: {result} (confidence: {confidence:.2f})")
    print("Model verification successful!")
except Exception as e:
    print(f"Error during model verification: {str(e)}")

print("\nModel training and testing completed successfully!")
