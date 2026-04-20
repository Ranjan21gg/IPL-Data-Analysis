import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# 1. Load Data
# Note: Ensure your terminal is in the same folder as this script 
# or that ../data/matches.csv is the correct path relative to your location.
df = pd.read_csv('../data/matches.csv')

# 2. Clean Data
df = df[['team1', 'team2', 'toss_winner', 'toss_decision', 'winner']]
df.dropna(inplace=True)

# 3. Split Features (X) and Target (y) BEFORE One-hot encoding
# This prevents the "winner not found" error
y = df['winner']
X = df.drop('winner', axis=1)

# 4. One-hot encoding on Features only
X = pd.get_dummies(X)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# 7. Create folder and Save Model
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Success! Model saved to models/model.pkl")
