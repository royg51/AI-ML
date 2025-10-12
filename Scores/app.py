import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score

# Professor Datta’s dataset

data = pd.DataFrame({

  'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

  'Quiz_Score': [55, 60, 65, 70, 75, 80, 85, 90, 95, 100],

  'Pass/Fail': ['Fail', 'Fail', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass'

]})

st.title("Professor Datta's Dataset Analysis")

# Display the dataset
st.header("Professor Datta's Dataset")
st.write("Quiz Scores Overview")
st.write(data['Quiz_Score'].describe())
# Mean, median, mode
mode_score = data['Quiz_Score'].mode()[0]
st.write(f"Most Frequent Quiz Score (Mode): {mode_score}")
mean_score = data['Quiz_Score'].mean()
st.write(f"Average Quiz Score (Mean): {mean_score}")
median_score = data['Quiz_Score'].median()
st.write(f"Middle Quiz Score (Median): {median_score}")

# Data Visualization
st.header("Data Visualization")
sns.set(style="whitegrid")
fig = plt.figure()
sns.scatterplot(x='Study_Hours', y='Quiz_Score', hue='Pass/Fail', data=data, s=100, palette={'Pass': 'green', 'Fail': 'red'})
plt.title("Study Hours vs Quiz Score")
st.pyplot(fig)

# Linear Regression Model
st.header("Linear Regression Model")
X = data[['Study_Hours']]
y = data['Quiz_Score']
lin_reg = LinearRegression()
lin_reg.fit(X, y)
data['Predicted_Score'] = lin_reg.predict(X)
r2_score = lin_reg.score(X, y)
st.write(f"R² Score of the Linear Regression Model: {r2_score:.3f}")

# Plot regression line
fig2 = plt.figure()
sns.lineplot(x='Study_Hours', y='Predicted_Score', data=data, color='blue', label='Regression Line')
sns.scatterplot(x='Study_Hours', y='Quiz_Score', hue='Pass/Fail', data=data, s=100, palette={'Pass': 'green', 'Fail': 'red'})
plt.title("Linear Regression: Study Hours vs Quiz Score")
st.pyplot(fig2)

# Logistic Regression Model
st.header("Logistic Regression Model")
data['Pass_Fail_Binary'] = data['Pass/Fail'].map({'Pass': 1, 'Fail': 0})
X_log = data[['Study_Hours']]
y_log = data['Pass_Fail_Binary']

log_reg = LogisticRegression()
log_reg.fit(X_log, y_log)
data['Predicted_Pass'] = log_reg.predict(X_log)
accuracy = accuracy_score(y_log, data['Predicted_Pass'])
st.write(f"Accuracy of the Logistic Regression Model: {accuracy:.3f}")
st.write("Logistic Regression Coefficients:")
st.write(f"Intercept: {log_reg.intercept_[0]:.3f}")
st.write(f"Coefficient for Study Hours: {log_reg.coef_[0][0]:.3f}")

x_grid = np.linspace(data["Study_Hours"].min(), data["Study_Hours"].max(), 100)
y_prob = log_reg.predict_proba(x_grid.reshape(-1, 1))[:, 1]

# Plot logistic regression curve
fig3 = plt.figure()
sns.scatterplot(x='Study_Hours', y='Pass_Fail_Binary', hue='Pass/Fail', data=data, s=100, palette={'Pass': 'green', 'Fail': 'red'})
sns.lineplot(x=x_grid, y=y_prob, color='blue', label='Logistic Regression Curve')
plt.title("Logistic Regression: Study Hours vs Pass Probability")
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
st.pyplot(fig3)

# Final data
st.header("Predictions vs Actuals")
st.dataframe(data)