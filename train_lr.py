import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

print("Loading and preprocessing data...")

# Load the cricket dataset
data = pd.read_csv('data/cricket_dataset_lr.csv')

# Create High/Low Margin classification based on median
margin_median = data['Win_Margin'].median()
data['High_Low_Margin'] = (data['Win_Margin'] > margin_median).astype(int)

print(f"\nWin Margin Statistics:")
print(f"Median (threshold): {margin_median:.2f}")
print(f"Mean: {data['Win_Margin'].mean():.2f}")
print(f"Min: {data['Win_Margin'].min():.2f}")
print(f"Max: {data['Win_Margin'].max():.2f}")

# Encode venue using LabelEncoder
venue_encoder = LabelEncoder()
data['Venue_Encoded'] = venue_encoder.fit_transform(data['Venue'])

print("\nVenue Encoding Mapping:")
for venue in sorted(data['Venue'].unique()):
    encoded_value = venue_encoder.transform([venue])[0]
    print(f"{venue}: {encoded_value}")

# Prepare features and target
X = data[['Venue_Encoded', 'Win_Margin']]
y = data['High_Low_Margin']

print("\nFeature Statistics:")
print(X.describe())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Logistic Regression model...")

# Create and train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
confusion_matrix = pd.crosstab(y_test, y_pred, margins=True)
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('LR: Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('lr_results.png')
plt.close()

# Save the model and related data
model_data = {
    'model': model,
    'venue_encoder': venue_encoder,
    'margin_median': margin_median,
    'venues': sorted(data['Venue'].unique())
}

print("\nSaving model and related data...")

with open('lr_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model and encoders saved as 'lr_model.pkl'")

# Test prediction
print("\nTesting model with sample data...")
test_venues = sorted(data['Venue'].unique())
test_margins = [10, 30, 50, 70, 90]

print("\nSample Predictions:")
print("Venue | Win Margin | Prediction | Confidence")
print("-" * 50)

for venue in test_venues[:3]:  # Test first 3 venues
    venue_encoded = venue_encoder.transform([venue])[0]
    for margin in [30, 70]:  # Test with two different margins
        X_test = pd.DataFrame({
            'Venue_Encoded': [venue_encoded],
            'Win_Margin': [margin]
        })
        pred = model.predict(X_test)[0]
        pred_proba = model.predict_proba(X_test)[0]
        confidence = pred_proba[pred]
        result = "High" if pred == 1 else "Low"
        print(f"{venue:8} | {margin:10} | {result:10} | {confidence:.2f}")

print("\nModel training and testing completed successfully!")
