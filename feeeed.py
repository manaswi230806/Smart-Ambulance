'''import pandas as pd
from sklearn.model_selection import train_test_split
#Decision Tress:like a flowchart-it asks questions(like weight>150?) to decide the answer
from sklearn.tree import DecisionTreeClassifier
#k-Nearest Neighbours(KNN):Looks at nearby examples and says:"ill choose what must of my neighbours are"
from sklearn.neighbors import KNeighborsClassifier
#Support vector machine:draws the best possible line or curve to separate different groups clearly
from sklearn import svm
#linear regression:draws a straight line to predict a number(like price,height,score)
#logistic regression:tells you the chance of something happening-like yes no,true or false
from sklearn.metrics import accuracy_score
df = pd.read_csv('ice_cream_sales.csv')

df_encoded = pd.get_dummies(df, columns=['Season', 'Flavor', 'Day', 'Weather'])

# Features and target
X = df_encoded.drop('Popular', axis=1)
y = df_encoded['Popular']'''
# Split data
'''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Predicted Defaults:", y_pred)
print("Actual Defaults:", y_test.values)
print("Accuracy:", accuracy_score(y_test, y_pred))
# Split data'''
'''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predicted Defaults:", y_pred)
print("Actual Defaults:", y_test.values)
print("Accuracy:", accuracy_score(y_test, y_pred))'''
# Step : Import libraries
'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Step 1: Sample Dataset
data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female'],
    'SeniorCitizen': [0, 1, 0, 0, 1, 1, 0, 0],
    'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70, 89.10, 25.50, 99.75],
    'Tenure': [1, 34, 2, 45, 5, 10, 1, 60],
    'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Month-to-month', 'Month-to-month', 'Month-to-month', 'Two year'],
    'Churn': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('churn.csv', index=True)
print("Churn created successfully!")

# Step 2: Data Preprocessing
# Handle missing values
df.dropna(inplace=True)

# Encode categorical columns
label_encoders = {}
for column in ['Gender', 'Contract', 'Churn']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Step 3: Feature Selection and Scaling
X = df[['Gender', 'SeniorCitizen', 'MonthlyCharges', 'Tenure', 'Contract']]
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 5: Train Models
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(max_depth=4)
tree_model.fit(X_train, y_train)

# Step 6: Predictions
log_pred = log_model.predict(X_test)
tree_pred = tree_model.predict(X_test)

# Step 7: Evaluation
def evaluate_model(y_test, predictions, model_name):
    print(f"\n🔍 Evaluation for {model_name}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))

evaluate_model(y_test, log_pred, "Logistic Regression")
evaluate_model(y_test, tree_pred, "Decision Tree")

# Step 8: ROC Curve for Logistic Regression
y_prob_log = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_log)
roc_auc = roc_auc_score(y_test, y_prob_log)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()

# Step 9: Better Visualization - Feature Importance (Decision Tree)
plt.figure(figsize=(6, 4))
sns.barplot(
    x=tree_model.feature_importances_,
    y=['Gender', 'SeniorCitizen', 'MonthlyCharges', 'Tenure', 'Contract'],
    palette="viridis"
)
plt.title("Feature Importance - Decision Tree")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()'''
# Clustering Techniques
# K-Means Clustering Example with Elbow Method
# This script demonstrates how to perform K-Means clustering on a dataset
# and use the elbow method to determine the optimal number of clusters.
# # SPDX-License-Identifier: MIT
# # SPDX-FileCopyrightText: 2023 


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample data
df = pd.DataFrame({
    'Age': [25, 34, 22, 27, 45, 52, 23, 43, 36, 29],
    'Spending Score': [77, 62, 88, 71, 45, 20, 85, 40, 50, 65]
})

# Scale the data
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

# Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)

# Plot WCSS vs k
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

