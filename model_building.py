# model.py
"""
Breast Cancer Prediction Model Training Script
---------------------------------------------
Educational project only — NOT for medical use.

Trains a Logistic Regression model using 5 selected mean features
from the Breast Cancer Wisconsin (Diagnostic) dataset.
Saves the trained model and scaler for later use in the web app.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib
import os

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────

SELECTED_FEATURES = [
    'mean radius',
    'mean perimeter',
    'mean area',
    'mean concavity',
    'mean concave points'
]

MODEL_SAVE_PATH = os.path.join("model", "breast_cancer_model.joblib")
SCALER_SAVE_PATH = os.path.join("model", "scaler.joblib")

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ────────────────────────────────────────────────
# MAIN FUNCTIONS
# ────────────────────────────────────────────────

def load_and_prepare_data():
    """Load dataset and select allowed features + target"""
    print("Loading Breast Cancer Wisconsin dataset...")
    data = load_breast_cancer()
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target  # 0 = malignant, 1 = benign
    
    print("\nClass distribution:")
    print(df['diagnosis'].value_counts(normalize=True).round(3))
    print("\nMissing values:", df.isnull().sum().sum())
    
    # Select only permitted features
    X = df[SELECTED_FEATURES]
    y = df['diagnosis']
    
    return X, y


def train_and_evaluate():
    """Train model, evaluate, save artifacts"""
    X, y = load_and_prepare_data()
    
    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size    : {X_test.shape[0]} samples")
    
    # Feature scaling (very important for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver='lbfgs'
    )
    model.fit(X_train_scaled, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE ON TEST SET")
    print("="*50)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Malignant (0)', 'Benign (1)']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model and scaler
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"\nModel saved to:     {MODEL_SAVE_PATH}")
    print(f"Scaler saved to:    {SCALER_SAVE_PATH}")
    
    return model, scaler


def demo_prediction(model, scaler):
    """Quick demonstration of model inference"""
    print("\n" + "="*50)
    print("DEMO PREDICTION (fake patient data)")
    print("="*50)
    
    # Example values (mid-range realistic)
    sample = pd.DataFrame([{
        'mean radius': 15.12,
        'mean perimeter': 92.85,
        'mean area': 712.4,
        'mean concavity': 0.1425,
        'mean concave points': 0.0896
    }])
    
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)[0]
    prob = model.predict_proba(sample_scaled)[0].max()
    
    result = "Benign" if pred == 1 else "Malignant"
    print(f"Prediction          : {result}")
    print(f"Confidence          : {prob:.1%}")
    print(f"Raw probability [Malignant, Benign]: {model.predict_proba(sample_scaled)[0]}")


if __name__ == "__main__":
    print("Breast Cancer Prediction Model Training\n")
    print("IMPORTANT: This is an educational project only.\n"
          "           NOT a medical diagnostic tool.\n")
    
    model, scaler = train_and_evaluate()
    demo_prediction(model, scaler)
    
    print("\nDone. Model and scaler are ready for use in the Flask web app.")