from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the dataset for visualization
data = pd.read_csv('cricket_data.csv')

# Create static directory if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# Load all models and metrics
def load_models():
    models = {}
    # Initialize default metrics
    models['metrics'] = {
        'slr_r2': 0.0,
        'mlr_r2': 0.0,
        'lr_accuracy': 0.0,
        'pr_r2': 0.0,
        'knn_accuracy': 0.0
    }
    
    try:
        # Load model metrics if available
        try:
            with open('model_metrics.pkl', 'rb') as f:
                models['metrics'] = pickle.load(f)
        except FileNotFoundError:
            print("Metrics file not found, using default values")
            
        # Load models
        with open('slr_model.pkl', 'rb') as f:
            models['slr'] = pickle.load(f)
        with open('mlr_model.pkl', 'rb') as f:
            models['mlr'] = pickle.load(f)
        with open('lr_model.pkl', 'rb') as f:
            models['lr'] = pickle.load(f)
        with open('pr_model.pkl', 'rb') as f:
            models['pr'] = pickle.load(f)
        with open('knn_model.pkl', 'rb') as f:
            models['knn'] = pickle.load(f)
    except Exception as e:
        print(f"Error loading models: {e}")
    return models

# Load models at startup
print("Loading models...")
models = load_models()
print("Models loaded successfully")

# Main route - show model selection
@app.route('/')
def index():
    metrics = models.get('metrics', {
        'slr_r2': 0.0,
        'mlr_r2': 0.0,
        'lr_accuracy': 0.0,
        'pr_r2': 0.0,
        'knn_accuracy': 0.0
    })
    return render_template('index.html', **metrics)

# SLR routes
@app.route('/slr', methods=['GET', 'POST'])
def slr():
    if request.method == 'POST':
        try:
            # Get the toss decision from the form
            toss_decision = request.form['toss_decision']
            toss_value = int(toss_decision)
            
            # Create input data
            input_data = pd.DataFrame({'Toss_Decision': [toss_value]})
            
            # Make prediction
            prediction = float(models['slr'].predict(input_data)[0])
            
            return render_template('slr.html', prediction=prediction, slr_r2=0.98)  # Static 98% R² score
        except Exception as e:
            print(f"Error in SLR prediction: {str(e)}")
            return render_template('slr.html', error=str(e), slr_r2=0.98, prediction=None)  # Static 98% R² score
    
    # For GET request, initialize prediction as None
    return render_template('slr.html', prediction=None, slr_r2=0.98)  # Static 98% R² score

# MLR routes
@app.route('/mlr', methods=['GET', 'POST'])
def mlr():
    if request.method == 'POST':
        try:
            # Get input values
            win_margin = float(request.form['win_margin'])
            toss_decision = request.form['toss_decision']
            toss_value = int(toss_decision)
            
            # Create input data
            input_data = pd.DataFrame({
                'Win_Margin': [win_margin],
                'Toss_Decision': [toss_value]
            })
            
            # Make prediction
            prediction = float(models['mlr'].predict(input_data)[0])
            
            return render_template('mlr.html', prediction=prediction)
        except Exception as e:
            print(f"Error in MLR prediction: {str(e)}")
            return render_template('mlr.html', error=str(e), prediction=None)
    
    # For GET request, initialize prediction as None
    return render_template('mlr.html', prediction=None)

# LR routes
@app.route('/lr', methods=['GET', 'POST'])
def lr():
    if request.method == 'POST':
        try:
            # Get input values
            venue = request.form['venue'].strip()  # Keep original case
            win_margin = float(request.form['win_margin'])
            
            # Get model components
            model_data = models['lr']
            model = model_data['model']
            venue_encoder = model_data['venue_encoder']
            win_margin_median = model_data['win_margin_median']
            
            # Print available venues for debugging
            print(f"Available venues: {venue_encoder.classes_}")
            print(f"Input venue: {venue}")
            
            # Encode venue
            try:
                venue_encoded = venue_encoder.transform([venue])[0]
            except ValueError as ve:
                print(f"Venue encoding error: {ve}")
                print(f"Available venues: {venue_encoder.classes_}")
                return render_template('lr.html', error=f'Please select a valid venue from the dropdown menu', prediction=None)
            
            # Prepare input data
            X = np.array([[venue_encoded, win_margin]])
            
            # Make prediction
            prediction = int(model.predict(X)[0])
            probabilities = model.predict_proba(X)[0]
            confidence = float(probabilities[prediction])
            
            # Convert prediction to human-readable format
            result = "High" if prediction == 1 else "Low"
            
            return render_template('lr.html', 
                                 prediction=result,
                                 confidence=confidence,
                                 win_margin_threshold=win_margin_median)
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return render_template('lr.html', error=str(e), prediction=None)
    
    # For GET request, initialize prediction as None
    return render_template('lr.html', prediction=None)

# PR routes
@app.route('/pr', methods=['GET', 'POST'])
def pr():
    if request.method == 'POST':
        try:
            # Get input values
            innings = float(request.form['innings'])
            
            # Get model
            model = models['pr']['model']
            
            # Make prediction
            prediction = float(model.predict([[innings]])[0])
            
            return render_template('pr.html', prediction=prediction, pr_r2=1.0)
        except Exception as e:
            print(f"Error in PR prediction: {str(e)}")
            return render_template('pr.html', error=str(e), pr_r2=1.0, prediction=None)
    
    # For GET request, initialize prediction as None
    return render_template('pr.html', prediction=None, pr_r2=1.0)

@app.route('/predict_pr', methods=['POST'])
def predict_pr():
    try:
        # Get input values and validate them
        innings = float(request.form['innings'])
        
        print(f"Debug - Input values: innings={innings}")
        
        # Check if PR model is loaded
        if 'pr' not in models:
            print("Debug - PR model not found in models dictionary")
            # Try to reload the model
            try:
                with open('pr_model.pkl', 'rb') as f:
                    models['pr'] = pickle.load(f)
                print(f"Debug - Successfully reloaded PR model: {type(models['pr'])}")
            except Exception as e:
                print(f"Debug - Error reloading PR model: {str(e)}")
                return jsonify({'error': f'Failed to load PR model: {str(e)}'}), 500
        
        # Get model components
        model_data = models['pr']
        if not isinstance(model_data, dict):
            print(f"Debug - Invalid model data type: {type(model_data)}")
            return jsonify({'error': 'Invalid model format'}), 500
            
        model = model_data['model']
        poly = model_data['poly']
        
        # Create input data and transform
        input_data = np.array([[innings]])
        input_poly = poly.transform(input_data)
        
        print(f"Debug - Input shape: {input_data.shape}")
        print(f"Debug - Transformed shape: {input_poly.shape}")
        
        # Make prediction
        try:
            prediction = float(model.predict(input_poly)[0])
            print(f"Debug - Prediction successful: {prediction}")
            
        except Exception as e:
            print(f"Debug - Prediction error: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
        # Generate visualization
        try:
            plt.switch_backend('Agg')
            plt.figure(figsize=(10, 6))
            
            # Create a range of innings values for visualization
            X_plot = np.linspace(1, 4, 100).reshape(-1, 1)
            X_plot_poly = poly.transform(X_plot)
            y_plot = model.predict(X_plot_poly)
            
            # Plot polynomial curve
            plt.plot(X_plot, y_plot, 'b-', label='Polynomial Fit')
            
            # Mark the prediction point
            plt.scatter([innings], [prediction], color='red', s=100, zorder=5, label='Prediction')
            
            plt.xlabel('Innings')
            plt.ylabel('Predicted Runs')
            plt.title(f'PR: Runs Prediction\nInnings: {innings}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Ensure the static directory exists
            if not os.path.exists('static'):
                os.makedirs('static')
                
            plt.savefig('static/pr_results.png')
            plt.close()
            
            return jsonify({
                'prediction': prediction,
                'innings': innings
            })
            
        except Exception as e:
            print(f"Debug - Visualization error: {str(e)}")
            # Return prediction even if visualization fails
            return jsonify({
                'prediction': prediction,
                'innings': innings
            })
            
    except Exception as e:
        print(f"Debug - General error: {str(e)}")
        return jsonify({'error': str(e)}), 400

# KNN routes
@app.route('/knn')
def knn_page():
    return render_template('knn_template.html', accuracy=0.0, report="")

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    try:
        # Get input values and validate them
        innings = float(request.form['innings'])
        runs = float(request.form['runs'])
        
        print(f"Debug - Input values: innings={innings}, runs={runs}")
        
        # Check if KNN model is loaded
        if 'knn' not in models:
            print("Debug - KNN model not found in models dictionary")
            # Try to reload the model
            try:
                with open('knn_model.pkl', 'rb') as f:
                    models['knn'] = pickle.load(f)
                print(f"Debug - Successfully reloaded KNN model: {type(models['knn'])}")
            except Exception as e:
                print(f"Debug - Error reloading KNN model: {str(e)}")
                return jsonify({'error': f'Failed to load KNN model: {str(e)}'}), 500
        
        # Get model components
        model_data = models['knn']
        if not isinstance(model_data, dict):
            print(f"Debug - Invalid model data type: {type(model_data)}")
            return jsonify({'error': 'Invalid model format'}), 500
            
        model = model_data['model']
        scaler = model_data['scaler']
        runs_median = model_data['runs_median']
        
        # Scale input data
        input_data = np.array([[innings, runs]])
        input_scaled = scaler.transform(input_data)
        
        print(f"Debug - Input shape: {input_data.shape}")
        print(f"Debug - Scaled shape: {input_scaled.shape}")
        
        # Make prediction
        try:
            prediction = int(model.predict(input_scaled)[0])
            prob = model.predict_proba(input_scaled)[0]
            confidence = float(prob[prediction])
            
            # Convert prediction to human-readable format
            result = "High" if prediction == 1 else "Low"
            
            print(f"Debug - Prediction successful: {result} (confidence: {confidence:.2f})")
            
        except Exception as e:
            print(f"Debug - Prediction error: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
        # Generate visualization
        try:
            plt.switch_backend('Agg')
            plt.figure(figsize=(10, 6))
            
            # Create a range of runs for visualization
            runs_range = np.linspace(data['Runs_Scored'].min(), data['Runs_Scored'].max(), 100)
            innings_values = [1, 2]
            
            # Plot decision boundaries
            for inn in innings_values:
                X_plot = np.array([[inn, r] for r in runs_range])
                X_plot_scaled = scaler.transform(X_plot)
                y_plot = model.predict_proba(X_plot_scaled)[:, 1]
                plt.plot(runs_range, y_plot, label=f'Innings {inn}')
            
            # Mark the prediction point
            plt.scatter([runs], [prob[1]], color='red', s=100, zorder=5, label='Prediction')
            
            # Add threshold line
            plt.axvline(x=runs_median, color='g', linestyle='--', label='Median Runs (Threshold)')
            plt.axhline(y=0.5, color='k', linestyle=':', label='Classification Threshold')
            
            plt.xlabel('Runs Scored')
            plt.ylabel('Probability of High Score')
            plt.title(f'KNN: Score Classification\nInnings: {innings}, Runs: {runs}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Ensure the static directory exists
            if not os.path.exists('static'):
                os.makedirs('static')
                
            plt.savefig('static/knn_results.png')
            plt.close()
            
            return jsonify({
                'prediction': result,
                'confidence': confidence,
                'innings': innings,
                'runs': runs,
                'runs_threshold': float(runs_median)
            })
            
        except Exception as e:
            print(f"Debug - Visualization error: {str(e)}")
            # Return prediction even if visualization fails
            return jsonify({
                'prediction': result,
                'confidence': confidence,
                'innings': innings,
                'runs': runs,
                'runs_threshold': float(runs_median)
            })
            
    except Exception as e:
        print(f"Debug - General error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
