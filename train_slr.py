import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

# Load the cricket dataset
data = pd.read_csv('data/cricket_dataset_slr.csv')

# Print unique values and their counts
print("\nUnique values in Toss_Decision:")
print(data['Toss_Decision'].value_counts())

# Convert Toss_Decision to numeric values
data['Toss_Decision'] = data['Toss_Decision'].map({'Bat': 1, 'Bowl': 0})

# Drop rows with missing values
data = data.dropna()

# Print summary statistics
print("\nSummary statistics for Runs_Scored grouped by Toss_Decision:")
print(data.groupby('Toss_Decision')['Runs_Scored'].describe())

# Calculate average runs for each decision
avg_runs_by_decision = data.groupby('Toss_Decision')['Runs_Scored'].mean()
print("\nAverage runs by decision:")
print("Bowl first:", avg_runs_by_decision[0])
print("Bat first:", avg_runs_by_decision[1])

# Prepare features and target
X = data[['Toss_Decision']]
y = data['Runs_Scored']

# Create and train the model
model = LinearRegression()
model.fit(X, y)  # Using all data since we want the most accurate coefficients

# Print model coefficients
print("\nModel Coefficients:")
print(f"Slope (coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Make predictions for both cases
bowl_prediction = model.predict([[0]])[0]
bat_prediction = model.predict([[1]])[0]
print("\nPredictions:")
print(f"Predicted runs when bowling first (0): {bowl_prediction:.2f}")
print(f"Predicted runs when batting first (1): {bat_prediction:.2f}")
print(f"Difference in predictions: {bat_prediction - bowl_prediction:.2f}")

# Plot results with actual data points
plt.figure(figsize=(10, 6))

# Add some jitter to x-coordinates for better visualization
bowl_x = np.random.normal(0, 0.02, size=len(data[data['Toss_Decision'] == 0]))
bat_x = np.random.normal(1, 0.02, size=len(data[data['Toss_Decision'] == 1]))

plt.scatter(bowl_x, 
           data[data['Toss_Decision'] == 0]['Runs_Scored'], 
           color='blue', label='Bowl First (Actual)', alpha=0.5)
plt.scatter(bat_x, 
           data[data['Toss_Decision'] == 1]['Runs_Scored'], 
           color='green', label='Bat First (Actual)', alpha=0.5)

# Plot the regression line
plt.plot([0, 1], [bowl_prediction, bat_prediction], color='red', label='Predicted', linewidth=2)

plt.xlabel('Toss Decision (0: Bowl, 1: Bat)')
plt.ylabel('Runs Scored')
plt.title('SLR: Runs Scored vs Toss Decision')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('slr_results.png')
plt.close()

# Save the model
with open('slr_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as 'slr_model.pkl'")
