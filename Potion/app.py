import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Data
data = pd.DataFrame({ 
    'Herbs': np.random.randint(1, 10, 100), 
    'Crystal': np.random.randint(1, 10, 100), 
    'MagicPower': np.random.randint(1, 10, 100), 
    'Effectiveness': np.random.randint(0, 2, 100) # 0: Low, 1: High 
})

# Model Training
X = data[['Herbs', 'Crystal', 'MagicPower']]
y = data['Effectiveness']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_clf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Predictions
y_pred = best_rf.predict(X_test)

st.title("Magic Potion Effectiveness Prediction")
st.write(f"Best Hyperparameters: {grid_search.best_params_}")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.text("Classification Report:\n" + classification_report(y_test, y_pred))
st.text(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")