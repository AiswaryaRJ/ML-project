import joblib
import pandas as pd

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


from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("career_bert_model")
model = BertForSequenceClassification.from_pretrained("career_bert_model")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def predict_career(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1)
    career = le.inverse_transform(pred.detach().numpy())[0]
    
    info = career_info.get(career, {"description": "N/A", "next_steps": []})
    
    return {
        "predicted_career": career,
        "description": info["description"],
        "next_steps": info["next_steps"]
    }


# Function to do bulk prediction from CSV
def bulk_predict(csv_path):
    df = pd.read_csv(csv_path)
    df['predicted_career'] = df['description'].apply(lambda x: predict_career(x)['predicted_career'])
    df['career_description'] = df['description'].apply(lambda x: predict_career(x)['description'])
    df['next_steps'] = df['description'].apply(lambda x: ", ".join(predict_career(x)['next_steps']))
    
    output_file = "career_predictions.csv"
    df.to_csv(output_file, index=False)
    print(f"Bulk predictions saved to '{output_file}'")

# Interactive single prediction mode
if __name__ == "__main__":
    print("Advanced Career Guidance System")
    print("Type 'exit' to quit or enter your career interests.")
    
    while True:
        user_input = input("\nEnter your interests: ")
        if user_input.lower() == 'exit':
            break
        result = predict_career(user_input)
        print(f"\nSuggested career path: {result['predicted_career']}")
        print(f"Description: {result['description']}")
        print(f"Next steps: {', '.join(result['next_steps'])}")
    
    # Optional: ask if user wants to do bulk prediction
    csv_input = input("\nDo you want to predict careers from a CSV file? (yes/no): ")
    if csv_input.lower() == "yes":
        csv_path = input("Enter CSV file path: ")
        bulk_predict(csv_path)
