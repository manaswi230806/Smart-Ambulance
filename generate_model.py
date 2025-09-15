import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import random

# Create dummy training data
data = []
for _ in range(500):
    lat = round(random.uniform(17.0, 17.6), 6)
    lng = round(random.uniform(78.0, 78.6), 6)
    hour = random.randint(0, 23)
    
    # Crowded in peak hours 8-10 AM and 5-8 PM
    crowded = int(hour in [8, 9, 17, 18, 19])
    
    data.append([lat, lng, hour, crowded])

df = pd.DataFrame(data, columns=['lat', 'lng', 'hour', 'crowded'])

X = df[['lat', 'lng', 'hour']]
y = df['crowded']

model = RandomForestClassifier()
model.fit(X, y)

# Save the model
with open('crowd_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as crowd_model.pkl")
