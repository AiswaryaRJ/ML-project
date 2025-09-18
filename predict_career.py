import pickle
import pandas as pd
import os

# --- Load the saved vectorizer and both models safely ---
def safe_load(file_name, fallback_name=None):
    # Try the primary file
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    # Try a fallback name if provided
    if fallback_name and os.path.exists(fallback_name):
        with open(fallback_name, 'rb') as f:
            return pickle.load(f)
    print(f"⚠️ Warning: Could not find '{file_name}' or '{fallback_name}'.")
    return None

vectorizer = safe_load('vectorizer.pkl')
# Match the actual file you said exists: career_model_lr.pkl
model_logreg = safe_load('career_model_logreg.pkl', fallback_name='career_model_lr.pkl')
model_rf = safe_load('career_model_rf.pkl')

# --- Career info dictionary ---
career_info = {
    "Software Engineer": {
        "description": "Designs and develops software applications.",
        "next_steps": ["Learn Python/Java/C++", "Build small projects", "Study algorithms"]
    },
    "Data Scientist": {
        "description": "Analyzes data to extract insights and build predictive models.",
        "next_steps": ["Learn Python/R", "Study statistics & ML", "Work on Kaggle projects"]
    },
    "Graphic Designer": {
        "description": "Creates visual or digital graphics.",
        "next_steps": ["Practice design tools", "Build portfolio", "Study color theory"]
    },
    "Mechanical Engineer": {
        "description": "Designs and analyzes mechanical systems.",
        "next_steps": ["Learn CAD tools", "Work on projects", "Study thermodynamics"]
    },
    "Teacher": {
        "description": "Educates and guides students.",
        "next_steps": ["Plan lessons", "Practice teaching", "Engage in classroom activities"]
    }
}

# --- Predict career ---
def predict_career(description):
    if not vectorizer or not model_logreg or not model_rf:
        return {
            "LogisticRegression": {
                "career": "Model not available",
                "description": "Please upload or retrain the model files.",
                "next_steps": []
            },
            "RandomForest": {
                "career": "Model not available",
                "description": "Please upload or retrain the model files.",
                "next_steps": []
            }
        }

    desc_vector = vectorizer.transform([description])
    career_logreg = model_logreg.predict(desc_vector)[0]
    career_rf = model_rf.predict(desc_vector)[0]

    info_logreg = career_info.get(career_logreg, {"description": "N/A", "next_steps": []})
    info_rf = career_info.get(career_rf, {"description": "N/A", "next_steps": []})

    return {
        "LogisticRegression": {
            "career": career_logreg,
            "description": info_logreg["description"],
            "next_steps": info_logreg["next_steps"]
        },
        "RandomForest": {
            "career": career_rf,
            "description": info_rf["description"],
            "next_steps": info_rf["next_steps"]
        }
    }

# --- Bulk CSV prediction ---
def bulk_predict(csv_path, output_file="career_predictions.csv"):
    df = pd.read_csv(csv_path)
    df['LogReg_Career'] = df['Description'].apply(lambda x: predict_career(x)['LogisticRegression']['career'])
    df['LogReg_Description'] = df['Description'].apply(lambda x: predict_career(x)['LogisticRegression']['description'])
    df['LogReg_NextSteps'] = df['Description'].apply(lambda x: ", ".join(predict_career(x)['LogisticRegression']['next_steps']))
    df['RF_Career'] = df['Description'].apply(lambda x: predict_career(x)['RandomForest']['career'])
    df['RF_Description'] = df['Description'].apply(lambda x: predict_career(x)['RandomForest']['description'])
    df['RF_NextSteps'] = df['Description'].apply(lambda x: ", ".join(predict_career(x)['RandomForest']['next_steps']))
    df.to_csv(output_file, index=False)
    print(f"✅ Bulk predictions saved to '{output_file}'")

# --- Interactive mode ---
if __name__ == "__main__":
    print("Career Prediction System (LogReg & RandomForest)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter your career interests or description: ")
        if user_input.lower() == 'exit':
            break
        result = predict_career(user_input)
        print("\n--- Logistic Regression Prediction ---")
        print(f"Predicted Career: {result['LogisticRegression']['career']}")
        print(f"Description: {result['LogisticRegression']['description']}")
        print(f"Next Steps: {', '.join(result['LogisticRegression']['next_steps'])}")
        print("\n--- Random Forest Prediction ---")
        print(f"Predicted Career: {result['RandomForest']['career']}")
        print(f"Description: {result['RandomForest']['description']}")
        print(f"Next Steps: {', '.join(result['RandomForest']['next_steps'])}\n")

    csv_input = input("Do you want to predict careers from a CSV file? (yes/no): ")
    if csv_input.lower() == 'yes':
        csv_path = input("Enter CSV file path: ")
        bulk_predict(csv_path)
