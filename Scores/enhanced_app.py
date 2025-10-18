import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Data creation
np.random.seed(42)
data = pd.DataFrame({
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Quiz_Score":   [55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    "Study_Group":  [0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    "Extracurriculars": [0, 1, 2, 1, 3, 0, 2, 1, 2, 3],
})

# Pass/Fail label and binary column
data['Pass/Fail'] = ["Fail" if score < 65 else "Pass" for score in data['Quiz_Score']]
data['Pass_Fail_Binary'] = data['Pass/Fail'].map({'Pass': 1, 'Fail': 0})

st.title("Professor Datta's Advanced Dataset Analysis")

# Display dataset + simple stats
st.header("Professor Datta's Dataset")
st.dataframe(data)

st.write("Quiz Scores Overview")
st.write(data['Quiz_Score'].describe())

mode_score = data['Quiz_Score'].mode()[0]
mean_score = data['Quiz_Score'].mean()
median_score = data['Quiz_Score'].median()

st.write(f"Most Frequent Quiz Score (Mode): {mode_score}")
st.write(f"Average Quiz Score (Mean): {mean_score}")
st.write(f"Middle Quiz Score (Median): {median_score}")

# Data visualization: pairplot + scatter
st.header("Data Visualization")
sns.set(style="whitegrid")

# Pairplot (use PairGrid.fig to pass to st.pyplot)
pair_plot = sns.pairplot(data, hue='Pass/Fail', diag_kind="kde", palette={'Pass': 'green', 'Fail': 'red'})
st.pyplot(pair_plot.fig)
plt.close(pair_plot.fig)

# Simple scatter: Study_Hours vs Quiz_Score
fig = plt.figure(figsize=(7, 4))
sns.scatterplot(
    x='Study_Hours', y='Quiz_Score', hue='Pass/Fail', data=data,
    s=100, palette={'Pass': 'green', 'Fail': 'red'}
)
plt.title("Study Hours vs Quiz Score")
st.pyplot(fig)
plt.close(fig)

# Correlation heatmap
st.header("Correlation Heatmap")
corr = data.corr(numeric_only=True)  # avoid string columns
fig_corr = plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Numeric Features")
st.pyplot(fig_corr)
plt.close(fig_corr)

# Model preparation
st.header("Model Preparation")
X = data[['Study_Hours', 'Quiz_Score', 'Study_Group', 'Extracurriculars']]
y = data['Pass_Fail_Binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
st.write(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Classification models
st.header("Classification Models")

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
st.subheader("Decision Tree Classifier")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
st.text("Classification Report:\n" + classification_report(y_test, y_pred_dt))
st.text(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_dt)}")

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
st.subheader("Random Forest Classifier")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
st.text("Classification Report:\n" + classification_report(y_test, y_pred_rf))
st.text(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
st.subheader("K-Nearest Neighbors Classifier")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")
st.text("Classification Report:\n" + classification_report(y_test, y_pred_knn))
st.text(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_knn)}")


# Underfitting vs Overfitting
st.header("Underfitting and Overfitting Analysis")

underfit_model = DecisionTreeClassifier(max_depth=1, random_state=42)
underfit_model.fit(X_train, y_train)
under_acc = accuracy_score(y_test, underfit_model.predict(X_test))

overfit_model = DecisionTreeClassifier(max_depth=10, random_state=42)
overfit_model.fit(X_train, y_train)
over_acc = accuracy_score(y_test, overfit_model.predict(X_test))

st.write(f"Underfitting Model Accuracy (max_depth = 1): {under_acc:.2f}")
st.write(f"Overfitting Model Accuracy (max_depth = 10): {over_acc:.2f}")

# Visualization of boundaries (use Study_Hours axis, keep other features fixed/interpolated)
st.header("Visualization: Underfitting vs Overfitting")
fig_uv = plt.figure(figsize=(8, 5))

# Plot actual points (Study_Hours vs pass/fail)
plt.scatter(data["Study_Hours"], data["Pass_Fail_Binary"], c=data["Pass_Fail_Binary"], cmap='coolwarm', s=80)

# Create grid for Study_Hours and build full-feature rows for prediction
X_plot = np.linspace(data["Study_Hours"].min(), data["Study_Hours"].max(), 200)
X_plot_full = pd.DataFrame({
    "Study_Hours": X_plot,
    "Quiz_Score": np.interp(X_plot, data["Study_Hours"], data["Quiz_Score"]),
    "Study_Group": 1,          
    "Extracurriculars": 1
})
y_under = underfit_model.predict(X_plot_full)
y_over = overfit_model.predict(X_plot_full)
plt.plot(X_plot, y_under, linestyle='--', label='Underfitting (max_depth=1)', linewidth=2)
plt.plot(X_plot, y_over, linestyle='-', label='Overfitting (max_depth=10)', linewidth=2)
plt.xlabel("Study Hours")
plt.ylabel("Pass (1) / Fail (0)")
plt.title("Underfitting vs Overfitting Decision Boundaries")
plt.legend()
st.pyplot(fig_uv)
plt.close(fig_uv)

# Linear Regression
st.header("Regression Analysis: Linear Regression")
X_reg = data[['Study_Hours']]
y_reg = data['Quiz_Score']
lin_reg = LinearRegression()
lin_reg.fit(X_reg, y_reg)
data['Predicted_Score'] = lin_reg.predict(X_reg)
r2_score = lin_reg.score(X_reg, y_reg)

fig_lr = plt.figure(figsize=(8, 5))

# Scatter actual quiz scores
sns.scatterplot(x='Study_Hours', y='Quiz_Score', hue='Pass/Fail', data=data, s=100, palette={'Pass':'green','Fail':'red'})

# Regression line
plt.plot(data['Study_Hours'], data['Predicted_Score'], color='blue', label='Regression Line', linewidth=2)
plt.title("Linear Regression: Study Hours vs Quiz Score")
plt.xlabel("Study Hours")
plt.ylabel("Quiz Score")
plt.legend()
st.pyplot(fig_lr)
plt.close(fig_lr)

st.write("Linear Regression Coefficients:")
st.write(f"Intercept: {lin_reg.intercept_:.3f}")
st.write(f"Coefficient for Study Hours: {lin_reg.coef_[0]:.3f}")
st.write(f"R² Score of the Linear Regression Model: {r2_score:.3f}")
st.write(f"Regression Equation: Quiz_Score = {lin_reg.intercept_:.3f} + {lin_reg.coef_[0]:.3f} * Study_Hours")

# Logistic Regression
st.header("Logistic Regression Model: Predicting Pass/Fail (from Study Hours)")
X_log = data[['Study_Hours']]
y_log = data['Pass_Fail_Binary']
log_reg = LogisticRegression()
log_reg.fit(X_log, y_log)
data['Predicted_Pass'] = log_reg.predict(X_log)
accuracy_log = accuracy_score(y_log, data['Predicted_Pass'])

st.write(f"Accuracy of the Logistic Regression Model: {accuracy_log:.3f}")
st.write("Logistic Regression Coefficients:")
st.write(f"Intercept: {log_reg.intercept_[0]:.3f}")
st.write(f"Coefficient for Study Hours: {log_reg.coef_[0][0]:.3f}")

# Logistic curve plot
x_grid = np.linspace(data["Study_Hours"].min(), data["Study_Hours"].max(), 200)
y_prob = log_reg.predict_proba(x_grid.reshape(-1, 1))[:, 1]
fig_lr2 = plt.figure(figsize=(8, 5))
sns.scatterplot(x='Study_Hours', y='Pass_Fail_Binary', hue='Pass/Fail', data=data, s=100, palette={'Pass':'green','Fail':'red'})
plt.plot(x_grid, y_prob, color='blue', linewidth=2, label='Logistic Regression Probability')
plt.title("Logistic Regression: Study Hours vs Probability of Passing")
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.legend()
st.pyplot(fig_lr2)
plt.close(fig_lr2)

# -----------------------------
# Final data view
st.subheader("Predictions vs Actuals")
st.dataframe(data)