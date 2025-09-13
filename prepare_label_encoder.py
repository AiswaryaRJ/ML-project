from sklearn.preprocessing import LabelEncoder
import pickle

# List of all career labels you want your model to predict
career_labels = ["Software Engineer", "Data Scientist", "Artist"]

# Fit label encoder
le = LabelEncoder()
le.fit(career_labels)

# Save the encoder for later use in predictions
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Label encoder saved as 'label_encoder.pkl'")
