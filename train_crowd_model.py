import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

# 1. Generate Simulated Data
np.random.seed(42)
N = 1000
lat = np.random.uniform(17.2, 17.6, N)  # around Hyderabad
lng = np.random.uniform(78.3, 78.6, N)
hour = np.random.randint(0, 24, N)
day_of_week = np.random.randint(0, 7, N)

# Crowd Level Logic (Simulated Pattern)
crowd = ((hour >= 8) & (hour <= 10)) | ((hour >= 17) & (hour <= 20))
crowd = crowd.astype(int)

# Create DataFrame
df = pd.DataFrame({
    'lat': lat,
    'lng': lng,
    'hour': hour,
    'day_of_week': day_of_week,
    'crowd': crowd
})

# 2. Train ML Model
X = df[['lat', 'lng', 'hour', 'day_of_week']]
y = df['crowd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 3. Save Model
with open("crowd_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ ML Crowd model trained and saved as crowd_model.pkl")
