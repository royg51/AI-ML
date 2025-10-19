import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

st.title("Potion Temperature Predictor")

# sliders for dataset size and noise
n_samples = st.slider("Number of potion samples", min_value=50, max_value=500, value=200, step=10)
noise_level = st.slider("Noise level", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

# create dataset
np.random.seed(42)
X = np.random.uniform(1, 10, (n_samples, 5))
noise = np.random.normal(0, noise_level, n_samples)
y = 5*X[:,0] + 3*X[:,1] - 2*X[:,2] + 4*np.sin(X[:,3]) + 2*np.cos(X[:,4]) + noise

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# kernel selection
kernel = st.selectbox("Choose SVM kernel", ['linear', 'poly', 'rbf', 'sigmoid'])

# train SVR model
svr = SVR(kernel=kernel, C=100, gamma=0.1)
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)

# calculate MSE
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error (MSE) on test set: {mse:.3f}")

# plot actual vs predicted
st.subheader("Potion Temperature Predictions")
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(range(len(y_test)), y_test, label='Actual', color='blue', alpha=0.5)
ax.plot(range(len(y_test)), y_pred, label='Predicted', color='red')
ax.set_xlabel("Sample index")
ax.set_ylabel("Potion Temperature")
ax.legend()
st.pyplot(fig)

# input new potion ingredients
st.subheader("Predict Temperature for New Potion Ingredients")
new_ingredients = []
for i in range(5):
    val = st.number_input(f"Ingredient {i+1} quantity", value=5.0)
    new_ingredients.append(val)

if st.button("Predict Temperature"):
    X_new = np.array([new_ingredients])
    y_new_pred = svr.predict(X_new)
    st.write(f"Predicted Potion Temperature: {y_new_pred[0]:.2f}")