# M.Tech_Project_Stress_level_prediction# ================================
# STRESS LYSIS FULL PROJECT (.py)
# Training + EDA + Model + Streamlit App
# ================================

# ------------ IMPORTS ------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import streamlit as st

# ------------ DATA COLLECTION ------------
data = pd.read_csv(r"E:\Stress-Lysis.csv")

# ------------ BASIC ANALYSIS ------------
print("Shape:", data.shape)
print(data.info())
print(data.describe())

print("Unique Stress Levels:", data['Stress_Level'].unique())
print("Missing values:", data.isnull().sum())
print("Duplicate values:", data.duplicated().sum())

# ------------ EDA ------------
print("Skewness:")
print(data.skew())

# Fixing Step Count outliers with IQR limits
Q1 = data["Step_count"].quantile(0.25)
Q3 = data["Step_count"].quantile(0.75)

data["Step_count"] = np.where(data["Step_count"] < Q1, Q1, data["Step_count"])
data["Step_count"] = np.where(data["Step_count"] > Q3, Q3, data["Step_count"])

print("Skewness after Step Count fixing:", data['Step_count'].skew())

# ------------ MODELLING ------------
X = data.drop(['Stress_Level'], axis=1)
y = data['Stress_Level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# --- Logistic Regression ---
regressor = LogisticRegression(C=1.0, random_state=2)
regressor.fit(X_train, y_train)
pred_lr = regressor.predict(X_test)
print("LR Accuracy:", accuracy_score(y_test, pred_lr))

# --- Random Forest ---
model_rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
model_rf.fit(X, y)
pred_rf = model_rf.predict(X_test)
print("RF Accuracy:", accuracy_score(y_test, pred_rf))

# --- SVM MODEL ---
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train_s, y_train)
y_pred_svm = classifier.predict(X_test_s)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# ------------ SAVE MODEL ------------
pickle.dump(classifier, open('stress_trained.sav', 'wb'))
pickle.dump(scaler, open('scaler.sav', 'wb'))

print("Model Saved Successfully!")

# ------------ STREAMLIT APP ------------

st.title("STRESS LEVEL PREDICTION WEB APP")

# load model & scaler
loaded_model = pickle.load(open('stress_trained.sav', 'rb'))
loaded_scaler = pickle.load(open('scaler.sav', 'rb'))

def stresslevel_prediction(input_data):
    features = np.asarray(input_data, dtype=float)
    features = features.reshape(1, -1)
    features_scaled = loaded_scaler.transform(features)
    prediction = loaded_model.predict(features_scaled)

    if prediction[0] == 0:
        return "Stress Level: LOW"
    elif prediction[0] == 1:
        return "Stress Level: MEDIUM"
    else:
        return "Stress Level: HIGH"

# input fields
Humidity = st.text_input("Humidity Value")
Temperature = st.text_input("Body Temperature")
Step_count = st.text_input("Number of Steps")

diagnosis = ""

if st.button("PREDICT"):
    diagnosis = stresslevel_prediction([Humidity, Temperature, Step_count])

st.success(diagnosis)
