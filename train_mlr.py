import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

# Load the cricket dataset
data = pd.read_csv('data/cricket_dataset_mlr.csv')

# Convert Toss_Decision to numeric values
data['Toss_Decision'] = data['Toss_Decision'].map({'Bat': 1, 'Bowl': 0})

# Print data info
print("\nData Info:")
print(data.info())
print("\nSample of data:")
print(data.head())

# Prepare features and target
X = data[['Win_Margin', 'Toss_Decision']]
y = data['Runs_Scored']

# Create and train the model using all data
model = LinearRegression()
model.fit(X, y)

# Print model coefficients
print("\nModel Coefficients:")
print(f"Win_Margin coefficient: {model.coef_[0]:.2f}")
print(f"Toss_Decision coefficient: {model.coef_[1]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Make some example predictions
print("\nExample Predictions:")
print("For Win_Margin=10:")
print(f"Bowl (0): {model.predict([[10, 0]])[0]:.2f} runs")
print(f"Bat (1): {model.predict([[10, 1]])[0]:.2f} runs")

print("\nFor Win_Margin=20:")
print(f"Bowl (0): {model.predict([[20, 0]])[0]:.2f} runs")
print(f"Bat (1): {model.predict([[20, 1]])[0]:.2f} runs")

# Calculate metrics using all data
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\nMultiple Linear Regression Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, color='blue', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Runs')
plt.ylabel('Predicted Runs')
plt.title('MLR: Actual vs Predicted Runs')
plt.savefig('mlr_results.png')
plt.close()

# Save the model
with open('mlr_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as 'mlr_model.pkl'")
