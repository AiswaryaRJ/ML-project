import random
import pandas as pd
import difflib

# --- Load career data ---
try:
    career_df = pd.read_csv("career_data.csv")
    career_df['description'] = career_df['description'].str.lower()
except FileNotFoundError:
    career_df = pd.DataFrame(columns=['description', 'career'])

# --- Predefined responses ---
responses = {
    "hello": [
        "Hi there! 👋 How can I assist with your career planning today?",
        "Hello! 😊 Tell me a bit about your interests so I can suggest a path."
    ],
    "career options": [
        "There are many exciting paths! 🌟 Try describing what you enjoy doing—like coding, design, or teaching.",
        "Careers today are diverse—tech, healthcare, design, and more. 🚀 Tell me your interests!"
    ],
    "courses": [
        "Explore Coursera, Udemy, or edX to build skills for your desired career.",
        "LinkedIn Learning and Khan Academy are great places to start."
    ],
    "resume": [
        "Highlight projects, use action verbs, and keep it concise (1 page for freshers).",
        "Include education, skills, and achievements. Tailor it to each job application!"
    ],
    "interview tips": [
        "Research the company, practice common questions, and prepare a few to ask them.",
        "Use the STAR method for behavioral questions and show enthusiasm!"
    ],
    "thanks": [
        "You're welcome! 😊 Wishing you the best on your career journey!",
        "Happy to help! 🙌 Keep exploring your options!"
    ]
}

keyword_map = {
    "hi": "hello", "hey": "hello",
    "career": "career options", "job": "career options", "jobs": "career options",
    "course": "courses", "study": "courses",
    "resume": "resume", "cv": "resume",
    "interview": "interview tips", "tips": "interview tips",
    "thank": "thanks", "thanks": "thanks", "thank you": "thanks"
}

def search_careers_fuzzy(user_input):
    """
    Fuzzy search for career matches.
    """
    if career_df.empty:
        return None

    text = user_input.lower()
    # Split input into words and find closest matches against all descriptions
    words = text.split()
    scores = {}

    for desc, career in zip(career_df['description'], career_df['career']):
        for word in words:
            ratio = difflib.SequenceMatcher(None, word, desc).ratio()
            scores[career] = max(scores.get(career, 0), ratio)

    # Filter top matches
    top_matches = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_matches = [c for c, s in top_matches if s > 0.3]  # threshold for fuzzy match

    if top_matches:
        return f"Based on what you said, you might explore these careers: {', '.join(top_matches[:5])}"
    return None

def get_response(user_input):
    text = user_input.lower().strip()

    # Check predefined responses
    for key, answers in responses.items():
        if key in text:
            return random.choice(answers)

    # Keyword mapping
    for word, mapped_key in keyword_map.items():
        if word in text:
            return random.choice(responses[mapped_key])

    # Try fuzzy search
    csv_suggestion = search_careers_fuzzy(text)
    if csv_suggestion:
        return csv_suggestion

    # Fallbacks
    fallbacks = [
        "Hmm, I’m not sure about that 🤔. Could you describe your interests for career suggestions?",
        "I don’t have an exact answer, but I can recommend jobs or courses if you tell me what you like doing!",
        "I’m still learning 🌱. Try asking about career options, courses, resume tips, or interviews!"
    ]
    return random.choice(fallbacks)

# --- Test mode ---
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye! 👋")
            break
        print("Bot:", get_response(user_input))
