Heart Disease Detector ML

A machine learning project that predicts the presence of heart disease using patient clinical data. It leverages models like Logistic Regression, Random Forest, and Support Vector Machine (SVM). The project includes data preprocessing, model evaluation, and persistence using pickle. 
GitHub

Table of Contents

About

Tech Stack

Project Structure

Setup & Usage

Modeling Workflow

Evaluation & Output

Future Enhancements

License & Contributions

About

This repository is centered on a heart disease detection system built through machine learning. It utilizes the UCI Heart Disease dataset and applies classification models to determine the probability of a patient having heart disease. The trained models are serialized using pickle for easy reuse. 
GitHub

Tech Stack

Language: Python

Libraries: Likely includes scikit-learn (LogisticRegression, RandomForestClassifier, SVC), pandas, numpy, pickle for model persistence

Dataset: UCI Heart Disease dataset

Model Serialization: Pickle 
GitHub

Project Structure (Assumed from file list)

Based on the visible files, a probable layout might be:

/
├── heart_app/               # Main application scripts
├── .gitignore
└── README.md                # (You’re creating this)


Files inside heart_app/ may include:

Data loading and preprocessing scripts

Model training routines (e.g., train_models.py)

Evaluation and visualization modules

Model saving (pickle.dump) and loading

Setup & Usage
Prerequisites

Python 3.x

Install dependencies (e.g., scikit-learn, numpy, pandas) via:

pip install -r requirements.txt

Steps

Clone the repository

git clone https://github.com/NishanthVishwas/Heart-Disease-Detector-ML.git
cd Heart-Disease-Detector-ML


Prepare the dataset

Ensure the UCI Heart Disease dataset is in the expected location (e.g., heart_app/data/).

Run training script

python heart_app/train_models.py


This likely handles:

Data preprocessing

Model training (Logistic Regression, Random Forest, SVM)

Model evaluation (metrics like accuracy, ROC)

Saving models using pickle

Use the trained model

python heart_app/predict.py --input "<feature_vector>"


(Assuming there's a script to make new predictions.)

Modeling Workflow

Load dataset – Read the UCI dataset and inspect features

Preprocess data – Handle missing values, scale features, encode categorical variables

Train models – Fit multiple classifiers (Logistic Regression, Random Forest, SVM)

Evaluate performance – Compare using cross-validation and metrics (accuracy, precision, recall, ROC AUC)

Save best model – Serialize using pickle for future inference

Evaluation & Output

Typical outputs you might generate:

Accuracy comparison across models

Confusion matrices and ROC AUC scores

Saved model files (model_logreg.pkl, model_rf.pkl, model_svm.pkl)

Prediction scripts to load and test new patient data

Future Enhancements

🌟 Bundle into a Streamlit or Flask app for a user-friendly interface

Add hyperparameter tuning via GridSearchCV or RandomizedSearchCV

Implement better feature engineering and model explainability (e.g., SHAP)

Provide a well-formatted requirements.txt and usage documentation

Integrate unit tests for data processing and model inference

License & Contributions

License: (Add license details if any, e.g., MIT, Apache)

Contributions: Contributions are welcome! Feel free to open issues or submit pull requests.
