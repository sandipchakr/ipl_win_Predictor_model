import pickle
import numpy as np
import pandas as pd
import streamlit as st

# load the dataset:-
df = pd.read_csv("ipl_dataset.csv",encoding="latin1",low_memory=False)

# load model:-
with open("ipl_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
encoders = data["encoders"]
# print(encoders)

def predict_winner(team_a, team_b, venue, city, toss_decision,
                   team_a_toss_win, team_a_NRR, team_b_NRR, team_a_last5_win_pct, 
                   team_b_last5_win_pct, h2h_win_pct, team_a_home, team_b_home):
    
    # input_data = np.array([[team_a, team_b, venue, city, toss_decision, team_a_toss_win, 
    #                         team_a_NRR, team_b_NRR, team_a_last5_win_pct, team_b_last5_win_pct, 
    #                         h2h_win_pct, team_a_home, team_b_home]])

    input_data = pd.DataFrame([{
    "team_a": team_a,
    "team_b": team_b,
    "venue": venue,
    "city": city,
    "toss_decision": toss_decision,
    "team_a_toss_win": team_a_toss_win,
    "team_a_NRR": team_a_NRR,
    "team_b_NRR": team_b_NRR,
    "team_a_last5_win_pct": team_a_last5_win_pct,
    "team_b_last5_win_pct": team_b_last5_win_pct,
    "h2h_win_pct": h2h_win_pct,
    "team_a_home": team_a_home,
    "team_b_home": team_b_home
}])
    
    # input_data[:, 0] = encoders["team_a"].transform(input_data[:, 0])
    # input_data[:, 1] = encoders["team_b"].transform(input_data[:, 1])
    # input_data[:, 2] = encoders["venue"].transform(input_data[:, 2])
    # input_data[:, 3] = encoders["city"].transform(input_data[:, 3])
    # input_data[:, 4] = encoders["toss_decision"].transform(input_data[:, 4])

    input_data["team_a"] = encoders["team_a"].transform(input_data["team_a"])
    input_data["team_b"] = encoders["team_b"].transform(input_data["team_b"])
    input_data["venue"] = encoders["venue"].transform(input_data["venue"])
    input_data["city"] = encoders["city"].transform(input_data["city"])
    input_data["toss_decision"] = encoders["toss_decision"].transform(input_data["toss_decision"])

    prediction = model.predict(input_data)
    return prediction[0]

# print(predict_winner(
#     "Lucknow Super Giants",
#     "Royal Challengers Bengaluru",
#     "Eden Gardens",
#     "Lucknow",
#     "Field",
#     0,-1.07,
#     1.42,
#     0.70,
#     0.20,
#     0.50,
#     1,
#     0
# ))

# create UI:-


st.markdown(
    "<h1>🏏 IPL Winner Predictor 🤖</h1>",
    unsafe_allow_html=True
)
def calculate_h2h(team_a, team_b):

    matches = df[
        (
            (df["team_a"] == team_a) &
            (df["team_b"] == team_b)
        ) |
        (
            (df["team_a"] == team_b) &
            (df["team_b"] == team_a)
        )
    ]

    if len(matches) == 0:
        return 0.5

    team_a_wins = (matches["winner"] == team_a).sum()

    return round((team_a_wins / len(matches)*100), 2)

st.write("## input data")
col1,col2 = st.columns(2)
team_a = col1.selectbox(
    "Select Team_A",
    df["team_a"].dropna().unique()
)
team_b = col2.selectbox(
    "Select Team_B",
    df["team_b"].dropna().unique()
)
if team_a == team_b:
    st.error("Both teams cannot be same")
    st.stop()

h2h_win_pct = calculate_h2h(team_a, team_b)
st.write(f"Head-to-Head Win % for {team_a}: {h2h_win_pct}")
venue = st.selectbox(
    "Select Venue",
    df["venue"].dropna().unique()
)

city = st.selectbox(
    "Select City",
    df["city"].dropna().unique()
)
col1,col2 = st.columns(2)
team_a_toss_win = col1.radio(
    "Did Team A Win Toss?",
    ["Yes", "No"]
)
team_a_toss_win = 1 if team_a_toss_win == "Yes" else 0

toss_decision = col2.selectbox(
    "Toss Decision",
    df["toss_decision"].dropna().unique()
)
team_a_last5_wins = st.slider(
    f"{team_a} Wins in Last 5 Matches",
    0,
    5,
    3
)

team_b_last5_wins = st.slider(
    f"{team_b} Wins in Last 5 Matches",
    0,
    5,
    2
)
st.subheader("Net Run Rate")

col1, col2 = st.columns(2)

team_a_NRR = col1.slider(
    f"{team_a} NRR",
    min_value=-5.0,
    max_value=5.0,
    value=0.0,
    step=0.01
)

team_b_NRR = col2.slider(
    f"{team_b} NRR",
    min_value=-5.0,
    max_value=5.0,
    value=0.0,
    step=0.01
)

col1,col2 = st.columns(2)
team_a_home = col1.radio(
    f"{team_a} home ground ...?",
    ["Yes", "No"]
)
team_b_home = col2.radio(
    f"{team_b} home ground ...?",
    ["Yes", "No"]
)

if st.button("Predict Winner"):
    prediction = predict_winner(
        team_a,
        team_b,
        venue,
        city,
        toss_decision,
        team_a_toss_win,
        team_a_NRR,
        team_b_NRR,
        team_a_last5_wins/5, # convert to win percentage
        team_b_last5_wins/5, # convert to win percentage
        h2h_win_pct/100, # convert to decimal
        1 if team_a_home == "Yes" else 0,
        1 if team_b_home == "Yes" else 0
    )
    winner = team_a if prediction == 1 else team_b
    st.success(f"Predicted Winner: {winner}")
