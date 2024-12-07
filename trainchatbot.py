import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

#dataset
df=pd.read_csv('/content/drive/MyDrive/Chatbot/chatbot.csv', encoding='ISO-8859-1')
df.info()
print(df)

df=df.drop(["Question I.D"],axis="columns")

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df['Questions'] = df['Questions'].apply(preprocess_text)

# Prepare X and y
X = df['Questions']
y = df['Answers']

# Vectorize the questions with bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Experiment with different models
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Create a classification report
class_report = classification_report(y_test, y_pred, digits=4)
print("\nClassification Report:\n", class_report)

# Chatbot interaction
def chatbot_response(question):
    question = preprocess_text(question)
    question_tfidf = vectorizer.transform([question])
    answer = model.predict(question_tfidf)
    return answer[0]

# Example interaction
user_question = "How can I renew a book?"
response = chatbot_response(user_question)
print("Chatbot Response:", response)

import pickle

# Save the trained model
with open('logistic_regression_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)