import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Winner Predictor",
    page_icon="🏏",
    layout="centered",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── root / background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 40%, #0a1628 70%, #12082a 100%) !important;
    min-height: 100vh;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 20% 20%, rgba(99,60,255,0.18) 0%, transparent 70%),
        radial-gradient(ellipse 50% 60% at 80% 80%, rgba(255,140,0,0.12) 0%, transparent 70%),
        radial-gradient(ellipse 40% 40% at 60% 30%, rgba(0,200,150,0.09) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}
[data-testid="stHeader"],
[data-testid="stToolbar"]          { background: transparent !important; }
[data-testid="stSidebar"]          { display: none; }
.block-container {
    padding: 2rem 1.5rem 4rem !important;
    max-width: 780px !important;
    position: relative;
    z-index: 1;
}

/* ── typography ── */
* { font-family: 'DM Sans', sans-serif !important; }
h1, h2, h3, .syne { font-family: 'Syne', sans-serif !important; }

/* ── glass card helper ── */
.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
}

/* ── hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -40px; left: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,60,255,0.25) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: "";
    position: absolute;
    bottom: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(255,140,0,0.2) 0%, transparent 70%);
    pointer-events: none;
}
.hero h1 {
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #ffffff 0%, #c9b8ff 50%, #ff9c40 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem !important;
    line-height: 1.15 !important;
}
.hero p {
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
    letter-spacing: 0.5px;
}

/* ── section labels ── */
.section-label {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: rgba(180,160,255,0.7) !important;
    margin: 0 0 0.8rem !important;
}

/* ── glass panel ── */
.panel {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}

/* ── H2H badge ── */
.h2h-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: rgba(99,60,255,0.15);
    border: 1px solid rgba(140,100,255,0.3);
    border-radius: 40px;
    padding: 0.5rem 1.2rem;
    margin-top: 0.5rem;
    font-size: 0.88rem;
    color: rgba(200,185,255,0.95);
}
.h2h-badge .pct {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #b49fff;
}

/* ── Streamlit widget overrides ── */
/* Labels */
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: rgba(255,255,255,0.65) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}
/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-size: 0.9rem !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(140,100,255,0.5) !important;
}
/* Number input */
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-size: 0.9rem !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: rgba(140,100,255,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,60,255,0.15) !important;
}
/* Slider */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #6b3cff, #ff9c40) !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] {
    color: rgba(255,255,255,0.3) !important;
}

/* ── Predict button ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    padding: 0.9rem 2rem !important;
    background: linear-gradient(135deg, #6b3cff 0%, #9b59f5 50%, #ff7c20 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 30px rgba(107,60,255,0.4) !important;
    margin-top: 0.5rem !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 40px rgba(107,60,255,0.55) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Success / error boxes ── */
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}
.winner-box {
    text-align: center;
    padding: 2rem 1.5rem;
    background: linear-gradient(135deg, rgba(99,60,255,0.2), rgba(255,140,0,0.15));
    border: 1px solid rgba(180,150,255,0.3);
    border-radius: 22px;
    margin-top: 0.5rem;
    backdrop-filter: blur(20px);
}
.winner-box .trophy { font-size: 3rem; line-height: 1; margin-bottom: 0.5rem; }
.winner-box .label {
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.45);
    margin-bottom: 0.4rem;
}
.winner-box .team {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff, #c9b8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.08) !important;
    margin: 0.5rem 0 1.2rem !important;
}

/* ── column gap fix ── */
[data-testid="column"] { padding: 0 0.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Load data & model ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("ipl_dataset.csv", encoding="latin1", low_memory=False)

@st.cache_resource
def load_model():
    with open("ipl_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["encoders"]

df      = load_data()
model, encoders = load_model()


# ── Helpers ────────────────────────────────────────────────────────────────────
def calculate_h2h(team_a, team_b):
    matches = df[
        ((df["team_a"] == team_a) & (df["team_b"] == team_b)) |
        ((df["team_a"] == team_b) & (df["team_b"] == team_a))
    ]
    if len(matches) == 0:
        return 50.0
    return round((matches["winner"] == team_a).sum() / len(matches) * 100, 2)


def predict_winner(team_a, team_b, venue, city, NRR_diff, last5_diff, h2h_win_pct):
    row = pd.DataFrame([{
        "team_a": team_a, "team_b": team_b, "venue": venue, "city": city,
        "NRR_diff": NRR_diff, "last5_diff": last5_diff, "h2h_win_pct": h2h_win_pct,
    }])
    row["team_a"] = encoders["team_a"].transform(row["team_a"])
    row["team_b"] = encoders["team_b"].transform(row["team_b"])
    row["venue"]  = encoders["venue"].transform(row["venue"])
    row["city"]   = encoders["city"].transform(row["city"])
    return model.predict(row)[0]


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏏 IPL Winner Predictor</h1>
    <p>Machine-learning powered match outcome prediction</p>
</div>
""", unsafe_allow_html=True)

# ── Teams ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Select Teams</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
team_a = col1.selectbox("🔵 Team A", df["team_a"].dropna().unique(), key="ta")
team_b = col2.selectbox("🔴 Team B", df["team_b"].dropna().unique(), key="tb")

if team_a == team_b:
    st.error("⚠️ Both teams cannot be the same. Please choose different teams.")
    st.stop()

h2h = calculate_h2h(team_a, team_b)
st.markdown(f"""
<div class="h2h-badge">
    Head-to-Head &nbsp;·&nbsp; <strong>{team_a}</strong> win rate
    <span class="pct">{h2h}%</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Venue & City ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Match Venue</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
venue = col1.selectbox("🏟️ Stadium / Venue", df["venue"].dropna().unique())
city  = col2.selectbox("📍 City", df["city"].dropna().unique())

st.markdown("<br>", unsafe_allow_html=True)

# ── Form ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Recent Form  —  Last 5 Matches</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
a_wins = col1.slider(f"{team_a} wins", 0, 5, 3)
b_wins = col2.slider(f"{team_b} wins", 0, 5, 2)

st.markdown("<br>", unsafe_allow_html=True)

# ── NRR ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Net Run Rate</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
a_nrr = col1.number_input(f"{team_a} NRR", min_value=-5.0, max_value=5.0, value=0.0, step=0.01, format="%.2f")
b_nrr = col2.number_input(f"{team_b} NRR", min_value=-5.0, max_value=5.0, value=0.0, step=0.01, format="%.2f")

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("⚡ Predict Winner", use_container_width=True):
    NRR_diff   = a_nrr - b_nrr
    last5_diff = (a_wins / 5) - (b_wins / 5)
    pred = predict_winner(
        team_a, team_b, venue, city,
        NRR_diff, last5_diff, h2h / 100
    )
    winner = team_a if pred == 1 else team_b
    st.markdown(f"""
<div class="winner-box">
    <div class="trophy">🏆</div>
    <div class="label">Predicted Winner</div>
    <div class="team">{winner}</div>
</div>
""", unsafe_allow_html=True)