# chatbot.py
import random
import pandas as pd
import difflib
import wikipedia
from functools import lru_cache

# Set Wikipedia language
wikipedia.set_lang("en")

# --- Load career data (local) ---
try:
    career_df = pd.read_csv("career_data.csv")
    career_df['description'] = career_df['description'].astype(str).str.lower()
except FileNotFoundError:
    career_df = pd.DataFrame(columns=['description', 'career'])

# --- Predefined simple responses ---
responses = {
    "hello": [
        "Hi there! 👋 How can I assist with your career planning today?",
        "Hello! 😊 Tell me a bit about your interests so I can suggest a path."
    ],
    "thanks": ["You're welcome! 😊", "Glad to help — good luck!"]
}

# Keyword-to-response mapping
keyword_map = {"hi": "hello", "hey": "hello", "thank": "thanks", "thanks": "thanks"}

# --- Fuzzy career search ---
def search_careers_fuzzy(user_input, threshold=0.4):
    if career_df.empty or not user_input.strip():
        return None
    text = user_input.lower().strip()
    words = [w for w in text.split() if len(w) > 2]
    if not words:
        return None
    scores = {}
    for desc, career in zip(career_df['description'], career_df['career']):
        for word in words:
            ratio = difflib.SequenceMatcher(None, word, desc).ratio()
            scores[career] = max(scores.get(career, 0), ratio)
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    matches = [c for c, s in ranked if s >= threshold]
    if matches:
        return matches[:4]  # top 4 matches
    return None

# --- Wikipedia integration ---
@lru_cache(maxsize=256)
def get_wikipedia_summary(query, sentences=3):
    if not query.strip():
        return None
    try:
        search_results = wikipedia.search(query, results=5)
    except Exception:
        return None
    if not search_results:
        return None
    for title in search_results[:3]:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            summary = page.summary
            if summary:
                parts = summary.split('. ')
                short = '. '.join(parts[:sentences])
                if not short.endswith('.'):
                    short += '.'
                return {"title": page.title, "summary": short, "url": page.url}
        except (wikipedia.DisambiguationError, wikipedia.PageError):
            continue
        except Exception:
            continue
    return None

# --- Main chatbot response ---
def get_response(user_input):
    text = (user_input or "").lower().strip()
    if not text:
        return "Please type something about your interests or skills."
    
    # 1) Predefined responses
    for key in responses:
        if key in text:
            return random.choice(responses[key])
    
    # 2) Keyword map
    for word, mapped_key in keyword_map.items():
        if word in text:
            return random.choice(responses[mapped_key])
    
    # 3) Fuzzy local career suggestion
    careers = search_careers_fuzzy(text)
    if careers:
        msg = f"Based on your interests, you might explore: {', '.join(careers)}. "
        msg += "These careers align with skills or hobbies you mentioned."
        return msg
    
    # 4) Wikipedia lookup
    wiki = get_wikipedia_summary(text, sentences=3)
    if wiki:
        return f"I found this on Wikipedia about **{wiki['title']}**:\n\n{wiki['summary']}\n\nRead more: {wiki['url']}"
    
    # 5) Fallback
    fallbacks = [
        "I’m not sure about that yet 🤔. Could you describe your interests (skills, subjects, or hobbies)?",
        "I don't have a direct answer, but I can suggest careers if you tell me what you enjoy doing!",
        "Try asking: 'What careers suit someone who likes <activity>?' or 'Recommend courses for <career>'."
    ]
    return random.choice(fallbacks)

# --- Quick test ---
if __name__ == "__main__":
    print("Career Guidance Chatbot (type 'exit' to quit)")
    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            print("Bot: Goodbye 👋")
            break
        print("Bot:", get_response(q))
