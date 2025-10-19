import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("Dinosaur Era Classifier")

# create dataset
np.random.seed(42)
data = pd.DataFrame({
    'BoneLength': np.random.uniform(1, 50, 100),
    'BoneDensity': np.random.uniform(1, 50, 100),
    'TeethCount': np.random.randint(1, 50, 100),
    'Era': np.random.randint(0, 4, 100)  # 0: Triassic, 1: Jurassic, 2: Cretaceous, 3: Quaternary
})

X = data[['BoneLength', 'BoneDensity', 'TeethCount']]
y = data['Era']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# sidebar for hyperparameters
st.sidebar.header("Gradient Boosting Hyperparameters")
n_estimators = st.sidebar.slider("Number of estimators", 50, 500, 100, step=10)
learning_rate = st.sidebar.slider("Learning rate", 0.01, 1.0, 0.1, step=0.01)
max_depth = st.sidebar.slider("Maximum depth of trees", 1, 10, 3, step=1)

# baseline model
baseline_gb = GradientBoostingClassifier()
baseline_gb.fit(X_train, y_train)
y_pred_baseline = baseline_gb.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
st.write(f"Baseline Accuracy (default params): {baseline_accuracy:.2f}")

# tuned model
gb_clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
gb_clf.fit(X_train, y_train)
y_pred_tuned = gb_clf.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

st.write(f"Tuned Accuracy: {tuned_accuracy:.2f}")
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred_tuned))
st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred_tuned))

st.subheader("Hyperparameter Impact Summary")
st.write("""
- Increasing `n_estimators` usually improves accuracy but increases training time.
- `learning_rate` controls how much each tree contributes; too high can overfit, too low can underfit.
- `max_depth` controls tree complexity; deeper trees can overfit small datasets.
- Proper tuning of these hyperparameters can significantly improve model performance over the baseline.
""")
