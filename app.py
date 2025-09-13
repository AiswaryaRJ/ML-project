# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
from chatbot import get_response as chatbot_response
from recommender import recommend

st.set_page_config(page_title="Career Guidance AI", layout="centered")
st.title("Career Guidance AI")
st.caption("Type your interests/skills and get career suggestions, chatbot help, and course/job recommendations.")

# Load career info
career_info = {
    "Software Engineer": {
        "description": "Designs and develops software applications.",
        "next_steps": ["Learn Python/Java/C++", "Build projects", "Study algorithms"]
    },
    "Data Scientist": {
        "description": "Analyzes data to extract insights and build predictive models.",
        "next_steps": ["Learn Python/R", "Study statistics & ML", "Kaggle projects"]
    },
    # Add other careers here...
}

# Load model + vectorizer
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("career_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
    except FileNotFoundError:
        st.error("career_model.pkl or vectorizer.pkl not found.")
        st.stop()
    return model, vectorizer

model, vectorizer = load_artifacts()

# Helper: softmax
def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum(axis=-1, keepdims=True)

def get_top_k(text, k=3):
    X = vectorizer.transform([text])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        probs = softmax(scores.ravel())
    else:
        pred = model.predict(X)[0]
        return [(pred, None)]
    classes = model.classes_
    top_idx = np.argsort(probs)[::-1][:k]
    return [(classes[i], float(probs[i])) for i in top_idx]

def important_terms(text, top_n=8):
    X = vectorizer.transform([text])
    fn = vectorizer.get_feature_names_out()
    row = X.toarray().ravel()
    idx = np.argsort(row)[::-1][:top_n]
    terms = [(fn[i], float(row[i])) for i in idx if row[i] > 0]
    return terms

# --- Single prediction ---
st.subheader("Describe your interests or skills")
examples = [
    "I enjoy building mobile apps and solving algorithmic problems",
    "I like designing clothes and following fashion trends",
    "I love teaching math and helping students understand concepts",
    "I like analyzing sales data and creating dashboards",
    "I enjoy flying and learning about aerodynamics"
]
example_choice = st.selectbox("Quick examples", ["— choose example —"] + examples)
text = st.text_area("Your description", value=(example_choice if example_choice != "— choose example —" else ""), height=140)
k = st.slider("How many career suggestions?", 1, 6, 3)

if st.button("Suggest careers"):
    if not text.strip():
        st.warning("Please enter a description.")
    else:
        results = get_top_k(text, k=k)
        df = pd.DataFrame(
            [(r[0], f"{r[1]:.2f}" if r[1] is not None else "—",
              ", ".join(career_info[r[0]]["next_steps"]) if r[0] in career_info else "N/A",
              career_info[r[0]]["description"] if r[0] in career_info else "N/A") for r in results],
            columns=["Career", "Confidence", "Next Steps", "Description"]
        )
        st.subheader("Top suggestions")
        st.table(df)

        if results and results[0][1] is not None:
            chart_df = pd.Series({r[0]: r[1] for r in results})
            st.bar_chart(chart_df)

        terms = important_terms(text)
        if terms:
            st.subheader("Important words from your input")
            st.write(", ".join([f"{t[0]} ({t[1]:.2f})" for t in terms]))

        # Recommended jobs/courses
        st.subheader("Recommended Jobs/Courses")
        for career, _ in results:
            recs = recommend(career)
            st.write(f"**{career}** → {', '.join(recs)}")

# --- Bulk CSV predictions ---
st.subheader("Bulk CSV Predictions")
uploaded_file = st.file_uploader("Upload CSV (with 'description' column)", type=["csv"])
if uploaded_file:
    df_bulk = pd.read_csv(uploaded_file)
    if "description" not in df_bulk.columns:
        st.error("CSV must have a 'description' column.")
    else:
        df_bulk['predicted_career'] = df_bulk['description'].apply(lambda x: get_top_k(x, k=1)[0][0])
        df_bulk['career_description'] = df_bulk['predicted_career'].apply(lambda x: career_info[x]["description"] if x in career_info else "N/A")
        df_bulk['next_steps'] = df_bulk['predicted_career'].apply(lambda x: ", ".join(career_info[x]["next_steps"]) if x in career_info else "N/A")
        st.write(df_bulk)
        df_bulk.to_csv("career_predictions.csv", index=False)
        st.success("Bulk predictions saved to career_predictions.csv")

# --- Chatbot ---
st.subheader("Chatbot Assistant 🤖")
user_msg = st.text_input("Ask a question about careers or courses:")
if st.button("Chat"):
    if user_msg.strip():
        reply = chatbot_response(user_msg)
        st.write(f"**Bot:** {reply}")
