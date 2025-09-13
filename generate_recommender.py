# generate_recommender.py
import pandas as pd
import json
from collections import defaultdict

# load careers from your dataset
df = pd.read_csv("career_data.csv")
careers = sorted(df["career"].unique())

# template generator for each career (generic placeholders)
def make_entry(career):
    return {
        "summary": f"Overview: A career as a {career}. (Edit this to add a short summary.)",
        "skills": [
            "Fundamental skills (add specifics)",
            "Communication/teamwork",
            "Problem solving / domain fundamentals"
        ],
        "sample_courses": [
            "Introductory course to get started (replace with platform & course name)",
            "Intermediate/practical project course (replace)",
            "Certification or advanced topic course (replace)"
        ],
        "job_titles": [
            f"Junior {career}",
            f"{career} Intern",
            f"Associate {career}"
        ],
        "next_steps": [
            "Learn the fundamentals listed above",
            "Complete 2–3 hands-on projects and add to portfolio",
            "Apply for internships / entry-level roles"
        ]
    }

recs = {c: make_entry(c) for c in careers}

with open("recommender.json", "w", encoding="utf-8") as f:
    json.dump(recs, f, indent=2, ensure_ascii=False)

print(f"recommender.json created with {len(careers)} careers.")
