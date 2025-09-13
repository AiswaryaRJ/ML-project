import pandas as pd
import json

# Load your CSV with at least "career" column
df = pd.read_csv("career_data.csv")

# Get unique careers
careers = df["career"].unique()

# Build career_info dictionary with placeholders
career_info = {}
for career in careers:
    career_info[career] = {
        "description": f"A professional career in {career}. (Add a detailed description here.)",
        "next_steps": [
            f"Learn fundamental skills for {career}",
            f"Take an online course or certification for {career}",
            f"Work on small {career.lower()} projects to build experience"
        ]
    }

# Save to JSON for easy editing
with open("career_info.json", "w") as f:
    json.dump(career_info, f, indent=4)

print(f"Generated career_info.json with {len(careers)} careers.")
