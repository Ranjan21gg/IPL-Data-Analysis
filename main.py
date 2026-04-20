import pandas as pd

# Load dataset
df = pd.read_csv('data/matches.csv')

# Basic cleaning
df.dropna(inplace=True)

# Team wins
print("\nTeam Wins:\n", df['winner'].value_counts())

# Toss impact
toss_win = df[df['toss_winner'] == df['winner']]
print("\nToss impact %:", len(toss_win)/len(df)*100)