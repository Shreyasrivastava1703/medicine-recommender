import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(page_title="Medicine Recommender", layout="wide")

st.markdown("""
<style>

/* Main app title */
.main-title{
    font-size:48px;
    font-weight:700;
    text-align:center;
    color:#2C7BE5;
    margin-bottom:10px;
}

/* Subtitle */
.subtitle{
    text-align:center;
    font-size:18px;
    color:#666;
    margin-bottom:30px;
}

/* Card container */
.card{
    background-color:#f8f9fa;
    padding:25px;
    border-radius:16px;
    box-shadow:0px 6px 15px rgba(0,0,0,0.06);
    margin-bottom:25px;
}

/* Section headings inside cards */
.section-title{
    font-size:26px;
    font-weight:600;
    margin-bottom:15px;
}

/* Bullet list text */
.card p{
    font-size:18px;
}

/* Streamlit default text */
html, body, [class*="css"]{
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)
# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_symptoms():
    df = pd.read_csv("Training.csv")
    return list(df.columns[:-1])

@st.cache_data
def load_encoder():
    df = pd.read_csv("Training.csv")
    le = LabelEncoder()
    le.fit(df["prognosis"])
    return le

model = load_model()
symptoms = load_symptoms()
label_encoder = load_encoder()

# --------------------------------------------------
# UI Header
# --------------------------------------------------

st.markdown('<p class="main-title">🩺 Medicine & Lifestyle Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered disease prediction based on symptoms</p>', unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# Symptom Selector
# --------------------------------------------------

features = symptoms
display_labels = [feat.replace('_', ' ').title() for feat in features]
label_to_feat = dict(zip(display_labels, features))

selected = st.multiselect(
    "Select your symptoms",
    options=display_labels,
    placeholder="Choose symptoms..."
)

predict_btn = st.button("🔍 Predict Disease", use_container_width=True)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_btn:

    if not selected:
        st.warning("Please select at least one symptom.")
        st.stop()

    X = pd.DataFrame(0, index=[0], columns=features)

    for label in selected:
        feat = label_to_feat[label]
        X.at[0, feat] = 1

    pred_label = model.predict(X)[0]
    disease = label_encoder.inverse_transform([pred_label])[0]

    st.success(f"🎯 Predicted Disease: **{disease}**")

    # --------------------------------------------------
    # Load recommendation datasets
    # --------------------------------------------------

    desc_df = pd.read_csv("description.csv")
    prec_df = pd.read_csv("precautions_df.csv")
    meds_df = pd.read_csv("medications.csv")
    diets_df = pd.read_csv("diets.csv")
    workout_df = pd.read_csv("workout_df.csv")

    # normalize column names
    desc_df.columns = ["Disease", "Description"]
    meds_df.columns = ["Disease", "Medication"]
    diets_df.columns = ["Disease", "Diet"]

    workout_df = workout_df.loc[:, ~workout_df.columns.str.contains('^Unnamed')]
    workout_df.columns = workout_df.columns.str.strip().str.title()

    if "Workout1" in workout_df.columns and "Workout2" in workout_df.columns:
        workout_df["Workout"] = workout_df[["Workout1", "Workout2"]].apply(
            lambda row: ", ".join(row.dropna().astype(str)), axis=1
        )

    if "Workout" not in workout_df.columns:
        workout_df["Workout"] = ""

    workout_df = workout_df[["Disease", "Workout"]]

    # --------------------------------------------------
    # Description Card
    # --------------------------------------------------

    desc = desc_df.loc[desc_df.Disease == disease, "Description"]

    if not desc.empty:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown('<p class="section-title">📝 Disease Description</p>', unsafe_allow_html=True)
        st.write(desc.values[0])

        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------
    # Precautions + Medications
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">⚠️ Precautions</p>', unsafe_allow_html=True)

        prec_cols = [c for c in prec_df.columns if c.lower().startswith("p")]
        pr = prec_df.loc[prec_df.Disease == disease, prec_cols]

        if not pr.empty:
            items = [str(x) for x in pr.values.flatten() if pd.notna(x)]
            for i in items:
                st.write("•", i)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">💊 Medications</p>', unsafe_allow_html=True)

        med_row = meds_df.loc[meds_df.Disease == disease, "Medication"]

        if not med_row.empty:
            meds = eval(med_row.values[0])
            for m in meds:
                st.write("•", m)

        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------
    # Diet + Workout
    # --------------------------------------------------

    col3, col4 = st.columns(2)

    with col3:

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">🥗 Recommended Diet</p>', unsafe_allow_html=True)

        diet_row = diets_df.loc[diets_df.Disease == disease, "Diet"]

        if not diet_row.empty:
            diets = eval(diet_row.values[0])
            for d in diets:
                st.write("•", d)

        st.markdown('</div>', unsafe_allow_html=True)

    with col4:

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">🏃 Lifestyle / Workout</p>', unsafe_allow_html=True)

        wo = workout_df.loc[workout_df.Disease == disease, "Workout"].tolist()

        for w in wo:
            st.write("•", w)

        st.markdown('</div>', unsafe_allow_html=True)

    st.warning("⚠️ This is AI-generated advice. Always consult a medical professional.")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
st.caption("Built with ❤️ by Shreya | © 2025 Health Predictor")