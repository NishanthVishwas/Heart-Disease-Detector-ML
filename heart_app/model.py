# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
import os

def train_models():
    # Load the dataset
    df = pd.read_csv('heart.csv')
    
    # Preprocessing Stage
    # Step 1: Define features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Step 2: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 3: Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for future use
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Model Training Stage
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    # Train and save each model
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        
        # Save model
        filename = f"{name.lower().replace(' ', '_')}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
    
    # Return column names for the front-end form
    return X.columns.tolist()

# This will run when the script is executed directly
if __name__ == "__main__":
    train_models()
    print("Models trained and saved successfully")