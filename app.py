import streamlit as st
import pandas as pd
import pickle
import nbformat
from nbconvert import HTMLExporter
import streamlit.components.v1 as components
from datetime import datetime

# --- Render Jupyter Notebook ---
def render_notebook_html(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    html_exporter = HTMLExporter()
    html_exporter.template_name = 'lab'
    (body, _) = html_exporter.from_notebook_node(notebook)
    components.html(body, height=1500, scrolling=True)

# --- App UI ---
def main():
    st.set_page_config(
        page_title="Online Game Behavior Prediction",
        layout="wide",
        page_icon="🎮",
    )

    st.sidebar.title("🎮 Online Game Behavior")
    page = st.sidebar.radio("Go to", ["🏠 Home", "📓 Notebook", "🔮 Predict"])

    if page == "🏠 Home":
        st.title("🎮 Online Game Behavior Prediction")
        st.image("img.png", use_container_width=True)

        st.markdown("""
        Predict how engaged a player is in an online game based on their gaming behavior and profile!
        """)

        st.markdown("""
        <div style='display: flex; gap: 20px;'>
            <a href='https://github.com/NevilWilson107/Predicting_Online_gaming_behavior.git' target='_blank'>
                <button style='background-color:#007BFF; color:white; padding:10px; border:none; border-radius:5px;'>🔗 GitHub Repo</button>
            </a>
            <a href='https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset' target='_blank'>
                <button style='background-color:#28a745; color:white; padding:10px; border:none; border-radius:5px;'>📂 Dataset Source</button>
            </a>
        </div>
        """, unsafe_allow_html=True)

    elif page == "📓 Notebook":
        st.subheader("📓 Code and Output")
        render_notebook_html("online_game_behavior.ipynb")

    elif page == "🔮 Predict":
        st.subheader("🔮 Predict Player Engagement Level")

        col1, col2 = st.columns(2)

        with col1:
            Age = st.slider("Age", 10, 60, 25)
            PlayTimeHours = st.slider("Play Time (hours/week)", 0, 100, 10)
            SessionsPerWeek = st.slider("Sessions per Week", 0, 50, 5)
            AvgSessionDurationMinutes = st.slider("Avg Session Duration (minutes)", 0, 300, 60)
            PlayerLevel = st.slider("Player Level", 0, 100, 50)
            AchievementsUnlocked = st.slider("Achievements Unlocked", 0, 100, 10)

        with col2:
            InGamePurchases = st.slider("In-Game Purchases", 0, 10, 1)
            Gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            Location = st.selectbox("Location", ["Asia", "Europe", "North America", "Other"])
            GameGenre = st.selectbox("Game Genre", ["Puzzle", "RPG", "Shooter", "Sports", "Strategy"])
            GameDifficulty = st.selectbox("Game Difficulty", ["Easy", "Hard", "Medium"])

        # --- Hardcoded Label Encoding ---
        gender_map = {"Female": 0, "Male": 1, "Other": 2}
        location_map = {"Asia": 0, "Europe": 1, "North America": 2, "Other": 3}
        genre_map = {"Puzzle": 0, "RPG": 1, "Shooter": 2, "Sports": 3, "Strategy": 4}
        difficulty_map = {"Easy": 0, "Hard": 1, "Medium": 2}

        # --- Load model and scaler ---
        model = pickle.load(open("game.sav", "rb"))
        scaler = pickle.load(open("game_scaler.sav", "rb"))

                # --- Scale only the 6 features scaler was trained on ---
        scaled = scaler.transform([[
            Age,
            PlayTimeHours,
            SessionsPerWeek,
            AvgSessionDurationMinutes,
            PlayerLevel,
            AchievementsUnlocked
        ]])[0]

        # --- Combine scaled numerical features + unscaled InGamePurchases + encoded categoricals ---
        final_input = list(scaled) + [
            InGamePurchases,  # NOT scaled
            gender_map[Gender],
            location_map[Location],
            genre_map[GameGenre],
            difficulty_map[GameDifficulty]
        ]

        if st.button("🎯 Predict Engagement Level"):
            result = model.predict([final_input])[0]

            engagement_map = {
                0: "High",
                1: "Low",
                2: "Medium"
            }

            label = engagement_map.get(result, f"Level {result}")
            st.success(f"🕹️ Predicted Engagement Level: **{label}**")


if __name__ == "__main__":
    main()
