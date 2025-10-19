import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.title("Potion Overfitting Demonstration")

# sliders to control dataset size and model parameters
n_samples = st.slider("Number of potion samples", 20, 100, 50, step=5)
C_value = st.slider("Regularization parameter C", 0.1, 1000.0, 100.0, step=10.0)
gamma_value = st.slider("Gamma", 0.001, 1.0, 0.1, step=0.01)
kernel_choice = st.selectbox("Select kernel", ['rbf', 'linear', 'poly', 'sigmoid'])

# add more features option
add_features = st.checkbox("Add extra features (magic type)")

# create dataset
np.random.seed(42)
X_base = np.linspace(0, 10, n_samples).reshape(-1, 1)
y = 3 * X_base.flatten()**2 + np.random.normal(0, 5, n_samples)

# optionally add extra features
if add_features:
    X_extra = np.random.uniform(0, 10, (n_samples, 2))
    X = np.hstack((X_base, X_extra))
else:
    X = X_base

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train SVR model
svr = SVR(kernel=kernel_choice, C=C_value, gamma=gamma_value)
svr.fit(X_train, y_train)
y_train_pred = svr.predict(X_train)
y_test_pred = svr.predict(X_test)

# compute MSE
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# display MSE
st.write(f"Mean Squared Error on Training set: {mse_train:.2f}")
st.write(f"Mean Squared Error on Test set: {mse_test:.2f}")

# plot predictions
st.subheader("Potion Effect Predictions")
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(range(len(y_train)), y_train, color='blue', alpha=0.5, label='Training Data')
ax.scatter(range(len(y_train), len(y_train)+len(y_test)), y_test, color='green', alpha=0.5, label='Test Data')
ax.plot(range(len(y_train)), y_train_pred, color='orange', label='SVR Prediction (train)')
ax.plot(range(len(y_train), len(y_train)+len(y_test)), y_test_pred, color='red', label='SVR Prediction (test)')
ax.set_xlabel("Sample Index")
ax.set_ylabel("Potion Effect")
ax.legend()
st.pyplot(fig)

# wizard summary
st.subheader("Wizard Leo's Findings on Overfitting")
st.write("""
- Overfitting happens when a model fits the training data too perfectly (low training MSE) 
  but fails to generalize to new data (high test MSE).
- Increasing C or using complex kernels like 'rbf' or 'poly' can cause overfitting.
- Adding more data points generally improves generalization.
- Adding meaningful features can help the model capture true patterns but may also increase complexity if too many irrelevant features are added.
- To avoid overfitting, Wizard Leo can use:
  - Simpler kernels (linear)
  - Lower C values (stronger regularization)
  - More training data
  - Feature selection to remove irrelevant ingredients
""")
