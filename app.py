# app.py
import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoders
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🎮 Player Engagement Prediction App")

# --- User Inputs ---
age = st.slider("Age", 10, 60, 25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
location = st.selectbox("Location", ["Urban", "Rural", "Suburban"])
genre = st.selectbox("Favorite Game Genre", ["Action", "Strategy", "Puzzle", "RPG", "Sports"])
play_time = st.slider("Play Time (Hours/Week)", 0, 100, 10)
in_game_purchases = st.selectbox("In-Game Purchases", ["Yes", "No"])
game_difficulty = st.selectbox("Preferred Game Difficulty", ["Easy", "Medium", "Hard"])
sessions_per_week = st.slider("Gaming Sessions per Week", 0, 30, 7)
avg_session_duration = st.slider("Average Session Duration (min)", 0, 300, 45)
player_level = st.slider("Player Level", 1, 100, 10)
achievements = st.slider("Achievements Unlocked", 0, 1000, 50)

# --- Convert inputs into dataframe for model ---
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Location": [location],
    "GameGenre": [genre],
    "PlayTimeHours": [play_time],
    "InGamePurchases": [in_game_purchases],
    "GameDifficulty": [game_difficulty],
    "SessionsPerWeek": [sessions_per_week],
    "AvgSessionDurationMinutes": [avg_session_duration],
    "PlayerLevel": [player_level],
    "AchievementsUnlocked": [achievements]
})


# Predict
if st.button("Predict Engagement"):
    prediction = model.predict(input_data)
    engagement_level = label_encoder.inverse_transform(prediction)
    st.success(f"Predicted Engagement Level: **{engagement_level[0]}**")
