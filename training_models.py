import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Simple Linear Regression (SLR)
def train_slr():
    # Load data
    data = pd.read_csv('data/cricket_dataset_slr.csv')
    
    # Prepare features and target
    X = data[['Toss_Decision']]
    y = data['Runs_Scored']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nSimple Linear Regression Results:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', label='Predicted')
    plt.xlabel('Toss Decision')
    plt.ylabel('Runs Scored')
    plt.title('SLR: Runs Scored vs Toss Decision')
    plt.legend()
    plt.savefig('slr_results.png')
    plt.close()

# 2. Multiple Linear Regression (MLR)
def train_mlr():
    # Load data
    data = pd.read_csv('data/cricket_dataset_mlr.csv')
    
    # Prepare features and target
    X = data[['Win_Margin', 'Toss_Decision']]
    y = data['Runs_Scored']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nMultiple Linear Regression Results:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Runs')
    plt.ylabel('Predicted Runs')
    plt.title('MLR: Actual vs Predicted Runs')
    plt.savefig('mlr_results.png')
    plt.close()

# 3. Logistic Regression (LR)
def train_lr():
    # Load data
    data = pd.read_csv('data/cricket_dataset_lr.csv')
    
    # Prepare features and target
    X = data[['Venue', 'Win_Margin']]
    y = data['High_Low_Margin']  # Binary classification
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression()
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
    sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('LR: Confusion Matrix')
    plt.savefig('lr_results.png')
    plt.close()

# 4. Polynomial Regression (PR)
def train_pr():
    # Load data
    data = pd.read_csv('data/cricket_dataset_pr.csv')
    
    # Prepare features and target
    X = data[['Innings']]
    y = data['Runs_Scored']
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    
    # Train model
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
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], y, color='blue', label='Actual')
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)
    plt.plot(X_plot, y_plot, color='red', label='Predicted')
    plt.xlabel('Innings')
    plt.ylabel('Runs Scored')
    plt.title('PR: Runs Scored vs Innings')
    plt.legend()
    plt.savefig('pr_results.png')
    plt.close()

# 5. K-Nearest Neighbors (KNN)
def train_knn():
    # Load data
    data = pd.read_csv('data/cricket_dataset_knn.csv')
    
    # Prepare features and target
    X = data[['Innings', 'Runs_Scored']]
    y = data['High_Low_Score']  # Binary classification
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
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
    h = 0.02  # step size in the mesh
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, alpha=0.8)
    plt.xlabel('Innings')
    plt.ylabel('Runs Scored')
    plt.title('KNN: Decision Boundary')
    plt.savefig('knn_results.png')
    plt.close()

if __name__ == "__main__":
    print("Training all cricket prediction models...")
    train_slr()
    train_mlr()
    train_lr()
    train_pr()
    train_knn()
    print("\nAll models have been trained and results have been saved!") 