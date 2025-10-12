import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load models
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'creature_kmeans_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

pca_path = os.path.join(base_path, 'creature_pca_model.pkl')
with open(pca_path, 'rb') as file:
    pca_model = pickle.load(file)

scaler_path = os.path.join(base_path, 'creature_scaler.pkl')
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

st.title("Magical Creature Classifier")
st.write("This app classifies magical creatures based on their features.")

st.subheader("Input Creature Features")

uploaded_file = st.file_uploader("Upload a CSV file with creature features", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df.head())
else:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

# Apply scaling, PCA, and clustering

scaled_data = scaler.transform(df)
pca_data = pca_model.transform(scaled_data)
pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])

clusters = model.predict(pca_df)
pca_df['Cluster'] = clusters

st.write("PCA and Clustering Results:")
st.dataframe(pca_df.head())

# Visualization
st.subheader("PCA Clustering Visualization")
fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(
    pca_df["PC1"], pca_df["PC2"], 
    c=pca_df["Cluster"], cmap='viridis', s=60, edgecolor ='k'
)
centers = model.cluster_centers_
ax.scatter(
    centers[:, 0], centers[:, 1], 
    c='red', s=200, alpha=0.75, marker='X', label='Centroids'
)
ax.set_title("K-Means Clustering on PCA-Reduced Data")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()
st.pyplot(fig)
