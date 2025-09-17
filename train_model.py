import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Download NLTK resources (only first time)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
data = pd.read_csv('generated_dataset_fixed.csv')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation & numbers
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing
data['Description'] = data['Description'].apply(preprocess_text)

X = data['Description']
y = data['Career']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Define models and hyperparameter grids
models_params = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
        'params': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(class_weight='balanced', random_state=42),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    }
}

best_model = None
best_accuracy = 0
best_model_name = ""

# Hyperparameter tuning and training
for name, mp in models_params.items():
    print(f"\nTuning {name}...")
    grid = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_vec, y_train)
    
    y_pred = grid.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Best Params: {grid.best_params_}")
    print(f"{name} Accuracy after tuning: {acc}")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = grid.best_estimator_
        best_model_name = name

# Save best model and vectorizer
with open('career_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"\nBest model: {best_model_name} with accuracy {best_accuracy}")
print("Best model and vectorizer saved successfully!")
