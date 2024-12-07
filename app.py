from flask import Flask, request, render_template, jsonify
import pickle

# Load the model and vectorizer
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['question']
    user_input_transformed = vectorizer.transform([user_input])
    prediction = model.predict(user_input_transformed)
    return jsonify({'answer': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
