import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load your dataset
data = pd.read_csv('sms_data.csv')  # Assuming you have a CSV file with 'text' and 'label'

# Preprocess data
X = data['text']
y = data['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Save model and vectorizer
joblib.dump(model, 'sms_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
