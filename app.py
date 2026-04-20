import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open('models/model.pkl', 'rb'))

st.title("🏏 IPL Match Winner Prediction")

team1 = st.selectbox("Select Team 1", ["MI","CSK","RCB","KKR","RR"])
team2 = st.selectbox("Select Team 2", ["MI","CSK","RCB","KKR","RR"])
toss_winner = st.selectbox("Toss Winner", ["MI","CSK","RCB","KKR","RR"])
toss_decision = st.selectbox("Toss Decision", ["bat","field"])

if st.button("Predict Winner"):
    input_data = pd.DataFrame({
        'team1': [team1],
        'team2': [team2],
        'toss_winner': [toss_winner],
        'toss_decision': [toss_decision]
    })

    input_data = pd.get_dummies(input_data)

    # Align columns
    model_columns = model.feature_names_in_
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.success(f"Predicted Winner: {prediction[0]}")
    st.info(f"Winning Probability: {max(probability[0])*100:.2f}%")