# chatbot.py
import random
import pandas as pd
import difflib
import wikipedia
from functools import lru_cache
from transformers import pipeline

# --- Setup Wikipedia ---
wikipedia.set_lang("en")

# --- Load career data ---
try:
    career_df = pd.read_csv("career_data.csv")
    career_df['description'] = career_df['description'].astype(str).str.lower()
except FileNotFoundError:
    career_df = pd.DataFrame(columns=['description', 'career'])

# --- Predefined responses ---
responses = {
    "hello": [
        "Hi there! 👋 How can I assist with your career planning today?",
        "Hello! 😊 Tell me a bit about your interests so I can suggest a path."
    ],
    "career options": [
        "There are many paths! Describe what you enjoy (coding, design, helping people, etc.) and I’ll suggest careers."
    ],
    "courses": [
        "Try Coursera, Udemy, or edX. Want a suggestion for a specific career?"
    ],
    "resume": [
        "Highlight your projects, use action verbs, and keep it concise (1 page for freshers)."
    ],
    "interview tips": [
        "Research the company, practice common questions, and use STAR method for answers."
    ],
    "thanks": [
        "You're welcome! 😊", "Glad to help — good luck!"
    ]
}

keyword_map = {
    "hi": "hello", "hey": "hello",
    "career": "career options", "job": "career options",
    "course": "courses", "study": "courses",
    "resume": "resume", "cv": "resume",
    "interview": "interview tips", "tips": "interview tips",
    "thank": "thanks", "thanks": "thanks"
}

# --- Local fuzzy career search ---
def search_careers_fuzzy(user_input, threshold=0.4):
    if career_df.empty or not isinstance(user_input, str) or not user_input.strip():
        return None
    text = user_input.lower().strip()
    scores = {}
    for desc, career in zip(career_df['description'], career_df['career']):
        ratio = difflib.SequenceMatcher(None, text, desc).ratio()
        scores[career] = max(scores.get(career, 0), ratio)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    matches = [c for c, s in ranked if s >= threshold]
    if matches:
        return f"Based on your input, consider: {', '.join(matches[:4])}."
    return None

# --- Wikipedia integration ---
@lru_cache(maxsize=256)
def get_wikipedia_summary(query, sentences=3):
    try:
        search_results = wikipedia.search(query, results=5)
        if not search_results:
            return None
        for title in search_results[:3]:
            page = wikipedia.page(title, auto_suggest=False)
            parts = page.summary.split('. ')
            short = '. '.join(parts[:sentences])
            if not short.endswith('.'): short += '.'
            return {"title": page.title, "summary": short, "url": page.url}
    except Exception:
        return None

# --- Transformer-based Q&A ---
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

context = """
Software Engineers build apps and systems. Data Scientists analyze data to find patterns.
Doctors treat patients. AI/ML Engineers train models. Teachers guide students.
Lawyers defend rights. Pilots navigate airplanes. Architects design buildings.
UX/UI Designers create interfaces. Social Workers support communities.
"""

# --- Unified get_response ---
def get_response(user_input):
    text = (user_input or "").lower().strip()
    if not text:
        return "Please ask me something about careers or your interests."

    # 1. Predefined answers
    for key, answers in responses.items():
        if key in text:
            return random.choice(answers)

    for word, mapped in keyword_map.items():
        if word in text:
            return random.choice(responses[mapped])

    # 2. Fuzzy career match
    local = search_careers_fuzzy(text)
    if local:
        return local + " Want course suggestions for any of these?"

    # 3. Transformer Q&A
    try:
        result = qa_pipeline(question=user_input, context=context)
        if result and result['score'] > 0.25:
            return result['answer']
    except Exception:
        pass

    # 4. Wikipedia fallback
    wiki = get_wikipedia_summary(user_input, sentences=2)
    if wiki:
        return f"From Wikipedia about **{wiki['title']}**:\n{wiki['summary']}\nRead more: {wiki['url']}"

    # 5. Fallback
    return "I’m not sure 🤔. Try describing what you like doing or ask about a career or course."

# --- Test Run ---
if __name__ == "__main__":
    print("Chatbot ready! Type 'exit' to quit.")
    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            print("Bot: Goodbye 👋")
            break
        print("Bot:", get_response(q))
