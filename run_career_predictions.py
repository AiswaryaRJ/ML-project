import joblib
import pandas as pd
import os

# Load the trained model and vectorizer
model = joblib.load('career_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Career information dictionary
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

# Function to predict career for a single input
def predict_career(user_input):
    X = vectorizer.transform([user_input])
    prediction = model.predict(X)[0]
    
    info = career_info.get(prediction, {"description": "N/A", "next_steps": []})
    
    return {
        "predicted_career": prediction,
        "description": info["description"],
        "next_steps": info["next_steps"]
    }

# Function to do bulk prediction from CSV
def bulk_predict(csv_path):
    if not os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' does not exist!")
        return
    
    df = pd.read_csv(csv_path)
    df['predicted_career'] = df['description'].apply(lambda x: predict_career(x)['predicted_career'])
    df['career_description'] = df['description'].apply(lambda x: predict_career(x)['description'])
    df['next_steps'] = df['description'].apply(lambda x: ", ".join(predict_career(x)['next_steps']))
    
    output_file = "career_predictions.csv"
    df.to_csv(output_file, index=False)
    print(f"Bulk predictions saved to '{output_file}'")

# Step 1: Create sample CSV if not exists
sample_csv = "descriptions_to_predict.csv"
if not os.path.exists(sample_csv):
    sample_descriptions = [
        "I love coding and solving problems",
        "I enjoy painting and sketching",
        "I like analyzing data and statistics",
        "I enjoy designing websites",
        "I am interested in artificial intelligence and machine learning"
    ]
    pd.DataFrame({"description": sample_descriptions}).to_csv(sample_csv, index=False)
    print(f"Sample CSV '{sample_csv}' created successfully!")

# Step 2: Interactive single prediction
print("\n=== Advanced Career Guidance System ===")
print("Type 'exit' to quit or enter your career interests.\n")

while True:
    user_input = input("Enter your interests: ")
    if user_input.lower() == 'exit':
        break
    result = predict_career(user_input)
    print(f"\nSuggested career path: {result['predicted_career']}")
    print(f"Description: {result['description']}")
    print(f"Next steps: {', '.join(result['next_steps'])}\n")

# Step 3: Ask if user wants bulk prediction
csv_input = input("Do you want to predict careers from a CSV file? (yes/no): ")
if csv_input.lower() == "yes":
    csv_path = input(f"Enter CSV file path (default '{sample_csv}'): ").strip()
    if csv_path == "":
        csv_path = sample_csv
    bulk_predict(csv_path)
