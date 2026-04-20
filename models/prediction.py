import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv('../data/matches.csv')

df = df[['team1','team2','toss_winner','toss_decision','winner']]
df.dropna(inplace=True)

# Encode categorical values
le = LabelEncoder()

for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Split data
X = df.drop('winner', axis=1)
y = df['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Prediction
sample = X_test.iloc[0:1]
pred = model.predict(sample)
print("Prediction:", pred)