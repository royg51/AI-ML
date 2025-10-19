import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("Fantasy Creature Classifier")

# create dataset
np.random.seed(42)
data = pd.DataFrame({
    'Wingspan': np.random.randint(1, 20, 100),
    'FurLength': np.random.randint(1, 20, 100),
    'MagicLevel': np.random.randint(1, 20, 100),
    'CreatureType': np.random.randint(0, 3, 100)  # 0: Dragon, 1: Unicorn, 2: Griffin
})

X = data[['Wingspan', 'FurLength', 'MagicLevel']]
y = data['CreatureType']

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# sidebar for hyperparameters
st.sidebar.header("KNN Hyperparameters")
n_neighbors = st.sidebar.slider("Number of neighbors (k)", 1, 15, 5)
weight_option = st.sidebar.selectbox("Weighting scheme", ['uniform', 'distance'])
p_option = st.sidebar.selectbox("Distance metric (p)", [1, 2])  # 1=Manhattan, 2=Euclidean

# train KNN model
knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight_option, p=p_option)
knn.fit(X_train, y_train)

# predictions
y_pred = knn.predict(X_test)

# evaluation
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Performance")
st.write(f"Accuracy on test set: {accuracy:.2f}")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

# interactive prediction for new creature
st.subheader("Predict New Creature Type")
wingspan = st.number_input("Wingspan", min_value=1, max_value=20, value=10)
furlength = st.number_input("Fur Length", min_value=1, max_value=20, value=10)
magiclevel = st.number_input("Magic Level", min_value=1, max_value=20, value=10)

if st.button("Predict Creature Type"):
    X_new = np.array([[wingspan, furlength, magiclevel]])
    prediction = knn.predict(X_new)[0]
    mapping = {0: 'Dragon', 1: 'Unicorn', 2: 'Griffin'}
    st.write(f"Predicted Creature Type: {mapping[prediction]}")
