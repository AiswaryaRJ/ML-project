# app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Optional: these imports go at the top, not in the middle
import chatbot
import recommender

from chatbot import get_response

import streamlit as st

st.header("💬 Career Chatbot")
user_question = st.text_input("Ask me anything about careers or courses:")
if user_question:
    answer = get_response(user_question)
    st.write("🤖", answer)


st.set_page_config(page_title="Career Suggestor", layout="centered")
st.title("Career Suggestor")
st.caption("Type your interests/skills and get top career suggestions.")

# --- Career Info Dictionary ---
career_info = {
    "Software Engineer": {
        "description": "Designs and develops software applications.",
        "next_steps": ["Learn Python/Java/C++", "Build small projects", "Study algorithms"]
    },
    "Data Scientist": {
        "description": "Analyzes data to extract insights and build predictive models.",
        "next_steps": ["Learn Python/R", "Study statistics & ML", "Work on Kaggle projects"]
    },
    "Artist": {
        "description": "Creates visual or performing art.",
        "next_steps": ["Practice drawing/painting", "Learn digital tools", "Build portfolio"]
    }
}

# --- Load model and vectorizer ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("career_model.pkl")
    except FileNotFoundError:
        st.error("career_model.pkl not found. Run your training script first.")
        st.stop()
    try:
        vectorizer = joblib.load("vectorizer.pkl")
    except FileNotFoundError:
        st.error("vectorizer.pkl not found. Run your training script first.")
        st.stop()
    return model, vectorizer

model, vectorizer = load_artifacts()

# --- Helper Functions ---
def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum(axis=-1, keepdims=True)

def get_top_k(text, k=3):
    X = vectorizer.transform([text])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    elif hasattr(model, "decision_function"):
        probs = softmax(model.decision_function(X).ravel())
    else:
        pred = model.predict(X)[0]
        return [(pred, None)]
    classes = model.classes_
    top_idx = np.argsort(probs)[::-1][:k]
    return [(classes[i], float(probs[i])) for i in top_idx]

def important_terms(text, top_n=8):
    X = vectorizer.transform([text])
    features = (
        vectorizer.get_feature_names_out()
        if hasattr(vectorizer, "get_feature_names_out")
        else np.array(vectorizer.get_feature_names())
    )
    row = X.toarray().ravel()
    idx = np.argsort(row)[::-1][:top_n]
    return [(features[i], float(row[i])) for i in idx if row[i] > 0]

# --- Single Prediction Section ---
st.write("Describe your interests or skills. Examples: 'I enjoy coding', 'I love designing', 'I like teaching'.")
examples = [
    "I enjoy building mobile apps and solving algorithmic problems",
    "I like designing clothes and following fashion trends",
    "I love teaching math and helping students understand concepts",
    "I like analyzing sales data and creating dashboards",
    "I enjoy flying and learning about aerodynamics"
]
example_choice = st.selectbox("Quick examples", ["— choose example —"] + examples)
text = st.text_area(
    "Your description",
    value=(example_choice if example_choice != "— choose example —" else ""),
    height=140
)

k = st.slider("How many suggestions?", 1, 6, 3)

results = []
if st.button("Suggest careers"):
    if not text.strip():
        st.warning("Please enter a description.")
    else:
        results = get_top_k(text, k=k)
        df = pd.DataFrame(
            [
                (
                    r[0],
                    f"{r[1]:.2f}" if r[1] is not None else "—",
                    ", ".join(career_info[r[0]]["next_steps"]) if r[0] in career_info else "N/A",
                    career_info[r[0]]["description"] if r[0] in career_info else "N/A"
                )
                for r in results
            ],
            columns=["Career", "Confidence", "Next Steps", "Description"]
        )
        st.subheader("Top Suggestions")
        st.table(df)

        if results and results[0][1] is not None:
            st.bar_chart(pd.Series({r[0]: r[1] for r in results}))

        terms = important_terms(text)
        if terms:
            st.subheader("Important words from your input")
            st.write(", ".join([f"{t[0]} ({t[1]:.2f})" for t in terms]))

# --- Bulk Prediction Section ---
st.subheader("Bulk CSV Predictions")
uploaded_file = st.file_uploader("Upload CSV (with 'description' column)", type=["csv"])
if uploaded_file:
    df_bulk = pd.read_csv(uploaded_file)
    if "description" not in df_bulk.columns:
        st.error("CSV must have a 'description' column.")
    else:
        df_bulk['predicted_career'] = df_bulk['description'].apply(lambda x: get_top_k(x, k=1)[0][0])
        df_bulk['career_description'] = df_bulk['predicted_career'].apply(
            lambda x: career_info.get(x, {}).get("description", "N/A")
        )
        df_bulk['next_steps'] = df_bulk['predicted_career'].apply(
            lambda x: ", ".join(career_info.get(x, {}).get("next_steps", ["N/A"]))
        )
        st.write(df_bulk)
        df_bulk.to_csv("career_predictions.csv", index=False)
        st.success("Bulk predictions saved to career_predictions.csv")

# --- Chatbot Section ---
st.subheader("Chatbot Assistant 🤖")
user_msg = st.text_input("Ask me something about careers or courses:")
if st.button("Chat"):
    if user_msg.strip():
        reply = chatbot.get_response(user_msg)
        st.write(f"**Bot:** {reply}")

# --- Recommendations Section ---
if results:
    st.subheader("Recommended Jobs/Courses")
    for career, _ in results:
        rec = recommender.recommend(career)
        st.markdown(f"**{career}**")
        st.write("**Summary:**", rec.get("summary", ""))
        st.write("**Top skills:**", ", ".join(rec.get("skills", [])))
        st.write("**Suggested courses:**", ", ".join(rec.get("sample_courses", [])))
        st.write("**Job titles to search:**", ", ".join(rec.get("job_titles", [])))
        st.write("**Next steps:**", ", ".join(rec.get("next_steps", [])))
        st.write("---")

