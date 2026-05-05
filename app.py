import streamlit as st
import pandas as pd
import joblib

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Mental Health Recommendation System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- Load model + scaler ----------------
best_model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- Mappings ----------------
FREQ_MAP = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Always": 5}

TIME_MAP = {
    "Less than 30 minutes": 1,
    "30–60 minutes": 2,
    "1–2 hours": 3,
    "2–4 hours": 4,
    "More than 4 hours": 5
}

GENDER_MAP = {"Male": 0, "Female": 1, "Prefer not to say": 2}

PROFESSION_MAP = {
    "Student": 0,
    "Unemployed": 1,
    "Self-employed / Freelancer": 2,
    "Private Job": 3,
    "Government Job": 4,
    "Business Owner": 5,
    "Healthcare": 6,
    "Teacher / Education": 7,
    "Engineer / IT": 8,
    "Other": 9
}

COMP_FEEL_MAP = {
    "Much Better": 0,
    "Slightly Better": 1,
    "No Change": 2,
    "Slightly Worse": 3,
    "Much Worse": 4
}

OUTCOME_LABELS = {0: "Normal", 1: "Mild", 2: "Severe"}

# ---------------- Recommendation ----------------
def recommend(outcome: int):
    if outcome == 0:
        return {
            "title": "You seem to be doing okay",
            "message": """Keep healthy habits going. Try to maintain regular sleep, balanced screen time, movement, and meaningful offline breaks.""",
            "resources": [
                "WHO Mental Health Guide: https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response",
                "Sleep Tips: https://www.sleepfoundation.org/how-sleep-works/healthy-sleep-tips"
            ]
        }

    if outcome == 1:
        return {
            "title": "Mild symptoms detected",
            "message": """Some warning signs are showing up. This is a good time to improve routines, reduce mindless scrolling, and pay attention to stress, sleep, and mood.""",
            "resources": [
                "APA Stress Guide: https://www.apa.org/topics/stress",
                "Research: https://journals.sagepub.com/doi/10.1177/2167702617723376"
            ]
        }

    return {
        "title": "Higher risk detected",
        "message": """Your responses suggest elevated difficulty. Reaching out to a qualified mental health professional would be a strong next step.""",
        "resources": [
            "WHO Depression: https://www.who.int/news-room/fact-sheets/detail/depression",
            "NIMH: https://www.nimh.nih.gov/health/topics/depression"
        ]
    }

# ---------------- Feature engineering ----------------
def build_features(
    age, gender, daily_time_spent,
    mindless_use_freq, distraction_when_busy_freq, restless_without_sm, distraction_impact,
    concentration_difficulty_freq, sm_negative_impact_freq,
    social_comparison_freq, comparison_feelings, validation_seeking_freq,
    low_mood_freq, interest_fluctuation_freq, sleep_issues_freq
):
    gender_n = GENDER_MAP[gender]
    time_n = TIME_MAP[daily_time_spent]

    purpose = (
        FREQ_MAP[mindless_use_freq]
        + FREQ_MAP[distraction_when_busy_freq]
        + FREQ_MAP[restless_without_sm]
        + distraction_impact
    )

    anxiety = (
        FREQ_MAP[sm_negative_impact_freq]
        + FREQ_MAP[concentration_difficulty_freq]
    )

    self_esteem = (
        FREQ_MAP[social_comparison_freq]
        + COMP_FEEL_MAP[comparison_feelings]
        + FREQ_MAP[validation_seeking_freq]
    )

    depression = (
        FREQ_MAP[low_mood_freq]
        + FREQ_MAP[interest_fluctuation_freq]
        + FREQ_MAP[sleep_issues_freq]
    )

    X = pd.DataFrame([{
        "age": age,
        "gender": gender_n,
        "daily_time_spent": time_n,
        "purpose": purpose,
        "Anxiety Score": anxiety,
        "Self Esteem Score": self_esteem,
        "Depression Score": depression
    }])

    scores = {
        "purpose": purpose,
        "anxiety": anxiety,
        "self_esteem": self_esteem,
        "depression": depression,
        "total": purpose + anxiety + self_esteem + depression
    }

    return X, scores

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(139, 92, 246, 0.16), transparent 25%),
        radial-gradient(circle at bottom center, rgba(34, 211, 238, 0.10), transparent 30%),
        linear-gradient(135deg, #050816 0%, #081121 32%, #09152b 62%, #050814 100%);
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2.5rem;
    max-width: 1250px;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    right: 1rem;
}

h1, h2, h3, h4, h5, h6, p, label {
    color: #e5eefc !important;
}
            
.hero-wrap {
    padding: 1.2rem 0 0.6rem 0;
    margin-bottom: 1rem;
}

.hero-card {
    background: linear-gradient(135deg, rgba(10, 17, 35, 0.95), rgba(7, 12, 24, 0.88));
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 28px;
    padding: 28px 30px 24px 30px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.08;
    margin: 0;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, #67e8f9 0%, #60a5fa 30%, #818cf8 60%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #cbd5e1 !important;
    font-size: 1.02rem;
    margin-top: 0.8rem;
    margin-bottom: 0.2rem;
}

.hero-note {
    color: #94a3b8 !important;
    font-size: 0.95rem;
}

.section-card {
    background: linear-gradient(180deg, rgba(11, 18, 34, 0.88), rgba(8, 13, 25, 0.95));
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 24px;
    padding: 18px 18px 8px 18px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.22);
    margin-bottom: 1rem;
}

.section-title {
    font-size: 1.45rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
    color: #f8fafc !important;
}

.section-subtitle {
    color: #94a3b8 !important;
    font-size: 0.94rem;
    margin-bottom: 1rem;
}

.label-soft {
    color: #a5b4fc !important;
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
    display: inline-block;
}

[data-testid="stForm"] {
    background: transparent;
    border: none;
    padding: 0;
}

div[data-testid="stNumberInput"],
div[data-testid="stSelectbox"],
div[data-testid="stSlider"] {
    margin-bottom: 0.45rem;
}

.stNumberInput > div > div,
.stSelectbox > div > div {
    background: rgba(15, 23, 42, 0.90) !important;
    border: 1px solid rgba(100, 116, 139, 0.32) !important;
    border-radius: 16px !important;
    min-height: 3.2rem;
    color: #f8fafc !important;
    box-shadow: none !important;
}

.stNumberInput input {
    color: #111827 !important;
    background: #ffffff !important;
    font-weight: 600 !important;
}

.stSelectbox [data-baseweb="select"] > div {
    background: rgba(15, 23, 42, 0.90) !important;
    color: #f8fafc !important;
}

.stSelectbox [data-baseweb="select"] span {
    color: #f8fafc !important;
}

div[role="listbox"] {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 14px !important;
}

div[role="option"] {
    color: #111827 !important;
    background: #ffffff !important;
}

div[role="option"]:hover {
    color: #111827 !important;
    background: #e5e7eb !important;
}

div[role="option"][aria-selected="true"] {
    color: #111827 !important;
    background: #dbeafe !important;
}

div[role="listbox"] {
    background: #0f172a !important;
    border: 1px solid rgba(96,165,250,0.25) !important;
    border-radius: 14px !important;
}

@media (prefers-color-scheme: dark) {
    div[role="option"] {
        color: #e2e8f0 !important;
        background: #0f172a !important;
    }
}

@media (prefers-color-scheme: light) {
    div[role="option"] {
        color: #111827 !important;
        background: #ffffff !important;
    }
}
            
div[role="option"]:hover {
    background: rgba(59, 130, 246, 0.16) !important;
}

.stSlider [data-baseweb="slider"] {
    padding-top: 0.4rem;
    padding-bottom: 0.2rem;
}

.stSlider [role="slider"] {
    background: linear-gradient(90deg, #38bdf8, #8b5cf6) !important;
    border: 3px solid #ffffff22 !important;
    box-shadow: 0 0 18px rgba(56, 189, 248, 0.35);
}

.stSlider div[data-testid="stTickBar"] {
    background: linear-gradient(90deg, #38bdf8, #8b5cf6) !important;
}

.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    min-height: 3.4rem;
    border-radius: 18px;
    border: 0 !important;
    background: linear-gradient(90deg, #22d3ee 0%, #3b82f6 45%, #8b5cf6 100%) !important;
    color: white !important;
    font-size: 1rem;
    font-weight: 800;
    box-shadow: 0 14px 35px rgba(59, 130, 246, 0.28);
    transition: 0.25s ease;
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 42px rgba(99, 102, 241, 0.35);
}

[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.74);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 18px;
    padding: 12px 14px;
}

[data-testid="stMetricLabel"] {
    color: #93c5fd !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}

.result-card {
    background: linear-gradient(180deg, rgba(8, 15, 29, 0.94), rgba(10, 17, 35, 0.95));
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 26px;
    padding: 20px;
    box-shadow: 0 18px 40px rgba(0,0,0,0.26);
}

.result-title {
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 0.35rem;
}

.result-chip {
    display: inline-block;
    padding: 0.55rem 0.95rem;
    border-radius: 999px;
    font-weight: 800;
    font-size: 0.95rem;
    margin: 0.4rem 0 1rem 0;
}

.result-normal {
    background: rgba(34, 197, 94, 0.15);
    color: #86efac !important;
    border: 1px solid rgba(34, 197, 94, 0.28);
}

.result-mild {
    background: rgba(245, 158, 11, 0.14);
    color: #fcd34d !important;
    border: 1px solid rgba(245, 158, 11, 0.30);
}

.result-severe {
    background: rgba(239, 68, 68, 0.14);
    color: #fca5a5 !important;
    border: 1px solid rgba(239, 68, 68, 0.30);
}

.resource-item {
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(148, 163, 184, 0.14);
    padding: 0.9rem 1rem;
    border-radius: 14px;
    margin-bottom: 0.6rem;
    color: #e2e8f0 !important;
}

.small-help {
    color: #94a3b8 !important;
    font-size: 0.88rem;
    margin-top: -0.2rem;
    margin-bottom: 0.8rem;
}

hr {
    border-color: rgba(148,163,184,0.15);
}

@media (max-width: 900px) {
    .hero-title {
        font-size: 2.25rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<div class="hero-wrap"><div class="hero-card">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Mental Health Recommendation System</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="hero-note">This tool provides an AI-assisted screening result and guidance. It is not a medical diagnosis.</div>',
    unsafe_allow_html=True
)
st.markdown("</div></div>", unsafe_allow_html=True)

left, right = st.columns([1.65, 1], gap="large")

# ---------------- Form ----------------
with left:
    with st.form("survey"):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Personal Information</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Basic details to personalize the screening input.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=10, max_value=100, value=22, step=1)
            gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
        with col2:
            profession = st.selectbox("Profession", list(PROFESSION_MAP.keys()))
            daily_time_spent = st.selectbox("Social Media Time", list(TIME_MAP.keys()))

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Behavior</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">How social media habits affect your attention and daily flow.</div>', unsafe_allow_html=True)

        mindless_use_freq = st.selectbox("Mindless browsing", list(FREQ_MAP.keys()))
        distraction_when_busy_freq = st.selectbox("Distracted during tasks", list(FREQ_MAP.keys()))
        restless_without_sm = st.selectbox("Restless without social media", list(FREQ_MAP.keys()))
        distraction_impact = st.slider("Distraction impact", 1, 5, 3)
        st.caption("1 means very low impact, 5 means very high impact.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Focus & Productivity</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">A quick look at concentration and work or study disruption.</div>', unsafe_allow_html=True)

        concentration_difficulty_freq = st.selectbox("Difficulty concentrating", list(FREQ_MAP.keys()))
        sm_negative_impact_freq = st.selectbox("Social media affects productivity", list(FREQ_MAP.keys()))

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Self-Esteem</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">How comparison and validation habits may be shaping self-perception.</div>', unsafe_allow_html=True)

        social_comparison_freq = st.selectbox("Social comparison", list(FREQ_MAP.keys()))
        comparison_feelings = st.selectbox("Feeling after comparison", list(COMP_FEEL_MAP.keys()))
        validation_seeking_freq = st.selectbox("Seek validation", list(FREQ_MAP.keys()))

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Mood & Sleep</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Patterns linked to emotional state and rest quality.</div>', unsafe_allow_html=True)

        low_mood_freq = st.selectbox("Felt Depressed or Sad", list(FREQ_MAP.keys()))
        interest_fluctuation_freq = st.selectbox("Motivation change", list(FREQ_MAP.keys()))
        sleep_issues_freq = st.selectbox("Sleep issues", list(FREQ_MAP.keys()))

        st.markdown('</div>', unsafe_allow_html=True)

        submit = st.form_submit_button("Predict")

# ---------------- Side panel ----------------
with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Quick Guide</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Answer honestly. The model works best when your responses reflect your recent habits and feelings.</div>',
        unsafe_allow_html=True
    )

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Questions", "13")
    with m2:
        st.metric("Scale", "1–5")

    st.markdown('<div class="small-help">This version focuses on a cleaner experience: stronger contrast, better spacing, clearer sections, and a more modern dark aesthetic.</div>', unsafe_allow_html=True)
    st.info("Tip: Use the dropdowns in order from top to bottom. The structure is designed to feel like a guided flow instead of a wall of fields.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What This Covers</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - Personal information  
        - Social media behavior  
        - Focus and productivity  
        - Self-esteem patterns  
        - Mood and sleep signals
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Prediction ----------------
if submit:
    X, scores = build_features(
        age, gender, daily_time_spent,
        mindless_use_freq, distraction_when_busy_freq, restless_without_sm, distraction_impact,
        concentration_difficulty_freq, sm_negative_impact_freq,
        social_comparison_freq, comparison_feelings, validation_seeking_freq,
        low_mood_freq, interest_fluctuation_freq, sleep_issues_freq
    )

    X_scaled = scaler.transform(X)
    pred = int(best_model.predict(X_scaled)[0])
    label = OUTCOME_LABELS.get(pred, str(pred))
    rec = recommend(pred)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">Screening Result</div>', unsafe_allow_html=True)

    chip_class = "result-normal"
    if label == "Mild":
        chip_class = "result-mild"
    elif label == "Severe":
        chip_class = "result-severe"

    st.markdown(
        f'<div class="result-chip {chip_class}">Prediction: {label}</div>',
        unsafe_allow_html=True
    )
    st.write("As per Analysis: "+rec["title"])
    st.write(rec["message"])

    st.markdown("### Score Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Purpose", scores["purpose"])
    c2.metric("Anxiety", scores["anxiety"])
    c3.metric("Self-Esteem", scores["self_esteem"])
    c4.metric("Depression", scores["depression"])
    c5.metric("Total", scores["total"])

    st.markdown("### Recommended Resources")
    for item in rec["resources"]:
        st.markdown(f'<div class="resource-item">• {item}</div>', unsafe_allow_html=True)

    st.warning("This is a screening aid, not a clinical diagnosis. If distress feels intense or persistent, reaching out to a trusted adult or qualified mental health professional is important.")
    st.markdown('</div>', unsafe_allow_html=True)
