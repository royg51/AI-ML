import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Data
data = pd.DataFrame({ 
    'EnginePower': np.random.uniform(1, 100, 100), 
    'FuelEfficiency': np.random.uniform(1, 100, 100), 
    'ShieldStrength': np.random.uniform(1, 100, 100), 
    'MissionSuccess': np.random.randint(0, 2, 100) # 0: Failure, 1: Success 
})

# Model Training
X = data[['EnginePower', 'FuelEfficiency', 'ShieldStrength']]
y = data['MissionSuccess']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Baseline Model
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
Y_pred_baseline = svm_clf.predict(X_test)
baseline_accuracy = accuracy_score(y_test, Y_pred_baseline)

param_dist = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly']
}

random_search = RandomizedSearchCV(
    estimator=SVC(), 
    param_distributions=param_dist, 
    n_iter=10, 
    cv=5, 
    n_jobs=-1, 
    scoring='accuracy', 
    random_state=42
)

random_search.fit(X_train, y_train)
best_svm = random_search.best_estimator_

# Predictions
y_pred = best_svm.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_pred)

st.title("Space Mission Success Prediction")

st.header("Baseline SVM Model")
st.write(f"Baseline Accuracy: {baseline_accuracy:.2f}")

st.header("Tuned SVM Model with Randomized Search")
st.write(f"Best Hyperparameters: {random_search.best_params_}")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.text("Classification Report:\n" + classification_report(y_test, y_pred))
st.text(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

st.header("Improvement")
improvement = tuned_accuracy - baseline_accuracy
st.write(f"Accuracy Improvement after Hyperparameter Tuning: {improvement:.2f}")