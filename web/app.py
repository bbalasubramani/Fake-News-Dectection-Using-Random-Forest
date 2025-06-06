from flask import Flask, render_template, request
import pickle
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open('../notebooks/models/model.pkl', 'rb'))
vectorizer = pickle.load(open('../notebooks/models/vectorizer.pkl', 'rb'))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        processed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)
        result = "Fake News" if prediction[0] == 0 else "Real News"
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
