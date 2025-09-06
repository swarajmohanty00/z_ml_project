# app.py

import streamlit as st
from predictor import predict_win_probability, teams, cities

st.set_page_config(layout='wide')
st.title("🏏 IPL Win Predictor")

# Team & city selection
col1, col2, col3 = st.columns(3)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))
with col3:
    selected_city = st.selectbox('Select the host city', sorted(cities))

# Target input
target = st.number_input('Target Score', min_value=0, max_value=720, step=1)

# Match state inputs
col4, col5, col6 = st.columns(3)
with col4:
    score = st.number_input('Score', min_value=0, max_value=720, step=1)
with col5:
    overs = st.number_input('Overs Done', min_value=0, max_value=20, step=1)
with col6:
    wickets = st.number_input('Wickets fell', min_value=0, max_value=10, step=1)

# Prediction
if st.button("Predict Probabilities"):
    loss, win = predict_win_probability(
        batting_team, bowling_team, selected_city,
        target, score, overs, wickets
    )
    st.header(f"{batting_team} - {round(win * 100)}%")
    st.header(f"{bowling_team} - {round(loss * 100)}%")
