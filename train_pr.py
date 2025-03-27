import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

print("Loading and preprocessing data...")

# Load the cricket dataset
data = pd.read_csv('data/cricket_dataset_pr.csv')

print("\nData Info:")
print(data.info())
print("\nSample of data:")
print(data.head())

# Prepare features and target
X = data[['Innings']].values  # Convert to numpy array
y = data['Runs_Scored'].values

print("\nFeature Statistics:")
print(f"Innings - Mean: {X.mean():.2f}, Min: {X.min():.2f}, Max: {X.max():.2f}")
print(f"Runs - Mean: {y.mean():.2f}, Min: {y.min():.2f}, Max: {y.max():.2f}")

# Create polynomial features
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

print("\nPolynomial Features Shape:", X_poly.shape)
print("Feature Names:", poly.get_feature_names_out(['Innings']))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

print("\nTraining Polynomial Regression model...")

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nPolynomial Regression Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Print coefficients
print("\nModel Coefficients:")
feature_names = poly.get_feature_names_out(['Innings'])
for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual')

# Create smooth curve for plotting
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.plot(X_plot, y_plot, color='red', label='Polynomial Fit')
plt.xlabel('Innings')
plt.ylabel('Runs Scored')
plt.title('PR: Runs Scored vs Innings')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('pr_results.png')
plt.close()

# Save the model and polynomial transformer
model_data = {
    'model': model,
    'poly': poly,
    'feature_names': feature_names,
    'stats': {
        'mse': mse,
        'r2': r2,
        'mean_runs': y.mean(),
        'std_runs': y.std()
    }
}

print("\nSaving model and related data...")

with open('pr_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model and transformer saved as 'pr_model.pkl'")

# Test prediction
print("\nTesting model with sample data...")
test_innings = [1, 2, 3, 4]

print("\nSample Predictions:")
print("Innings | Predicted Runs")
print("-" * 30)

for innings in test_innings:
    input_data = np.array([[innings]])
    input_poly = poly.transform(input_data)
    prediction = model.predict(input_poly)[0]
    print(f"{innings:7} | {prediction:13.2f}")

# Verify model can be loaded and used
print("\nVerifying model loading and prediction...")
try:
    with open('pr_model.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    
    test_innings = 2
    input_data = np.array([[test_innings]])
    input_poly = loaded_data['poly'].transform(input_data)
    prediction = loaded_data['model'].predict(input_poly)[0]
    print(f"\nTest prediction for innings {test_innings}: {prediction:.2f} runs")
    print("Model verification successful!")
except Exception as e:
    print(f"Error during model verification: {str(e)}")

print("\nModel training and testing completed successfully!")

