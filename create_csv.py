import pandas as pd

# Sample career descriptions
career_descriptions = [
    "I love coding and solving problems",
    "I enjoy painting and sketching",
    "I like analyzing data and statistics",
    "I enjoy designing websites",
    "I am interested in artificial intelligence and machine learning"
]

# Create DataFrame
df = pd.DataFrame({"description": career_descriptions})

# Save CSV
csv_file = "descriptions_to_predict.csv"
df.to_csv(csv_file, index=False)

print(f"CSV file '{csv_file}' created successfully!")
