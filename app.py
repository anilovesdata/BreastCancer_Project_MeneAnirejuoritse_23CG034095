from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler (do this once when app starts)
MODEL_PATH = os.path.join('model', 'breast_cancer_model.joblib')
SCALER_PATH = os.path.join('model', 'scaler.joblib')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    error = None

    if request.method == 'POST':
        try:
            # Get form data
            radius = float(request.form['radius'])
            perimeter = float(request.form['perimeter'])
            area = float(request.form['area'])
            concavity = float(request.form['concavity'])
            concave_points = float(request.form['concave_points'])

            # Create DataFrame (must match training feature names exactly)
            input_data = pd.DataFrame([{
                'mean radius': radius,
                'mean perimeter': perimeter,
                'mean area': area,
                'mean concavity': concavity,
                'mean concave points': concave_points
            }])

            # Scale input
            input_scaled = scaler.transform(input_data)

            # Predict
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0]
            max_prob = prob.max()

            label = "Benign" if pred == 1 else "Malignant"
            prob_str = f"{max_prob:.1%}"

            prediction = label
            probability = prob_str

        except ValueError:
            error = "Please enter valid numeric values for all fields."
        except Exception as e:
            error = f"An error occurred: {str(e)}"

    return render_template('index.html',
                           prediction=prediction,
                           probability=probability,
                           error=error)


if __name__ == '__main__':
    app.run(debug=True)