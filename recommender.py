# recommender.py
import json
from typing import List, Dict, Any

# load recommender.json (generated above)
def load_recs(path: str = "recommender.json") -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

RECS = load_recs()

def recommend(career: str) -> Dict[str, Any]:
    """
    Return recommendation dict for a given career.
    If career not found, returns a generic fallback.
    """
    if not career:
        return {
            "summary": "No career provided.",
            "skills": [],
            "sample_courses": [],
            "job_titles": [],
            "next_steps": []
        }
    return RECS.get(career, {
        "summary": f"No specific recommendations for {career}.",
        "skills": ["Learn fundamentals", "Build projects"],
        "sample_courses": ["Search for beginner course on the topic"],
        "job_titles": [f"Entry-level {career}"],
        "next_steps": ["Learn fundamentals", "Do projects", "Apply for internships"]
    })

def recommend_by_text(text: str, predictor_fn) -> Dict[str, Any]:
    """
    Given an input text and a predictor function (e.g., get_top_k or predict_career),
    return the best career and its recommendations.

    predictor_fn can be:
      - a function that accepts text and returns a career string, OR
      - a function that returns a list of (career, score) tuples (like get_top_k)
    """
    if not text:
        return {"error": "No input text provided."}

    # Try to get single career from predictor_fn
    career = None
    try:
        maybe = predictor_fn(text)
        # If predictor returns list of (career, score)
        if isinstance(maybe, list):
            career = maybe[0][0]
        elif isinstance(maybe, dict) and "predicted_career" in maybe:
            career = maybe["predicted_career"]
        elif isinstance(maybe, str):
            career = maybe
    except Exception:
        # fallback: try calling predictor_fn as predict_career
        try:
            career = predictor_fn(text)["predicted_career"]
        except Exception as e:
            return {"error": "Predictor function failed", "details": str(e)}

    rec = recommend(career)
    return {"career": career, "recommendation": rec}



# Simple job/course recommendations
recommendations = {
    "Data Scientist": ["Complete a Machine Learning course on Coursera", "Look for Data Analyst internships"],
    "Software Engineer": ["Learn Full-Stack Development on Udemy", "Explore junior developer jobs"],
    "UI/UX Designer": ["Try Figma tutorials", "Build a Behance portfolio"],
    "Teacher": ["Earn a teaching certification", "Apply for tutoring opportunities"],
    # Add more career → recommendation pairs
}

def recommend(career):
    return recommendations.get(career, ["Explore online resources", "Research entry-level opportunities"])
