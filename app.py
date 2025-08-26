from flask import Flask, request, jsonify
import joblib
import numpy as np

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

model = joblib.load('fake_news_model (1).pkl')
tfidf = joblib.load('tfidf_vectorizer (1).pkl')
@app.route('/', methods=['GET'])
def home():
    return "Flask backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news_text = data['text']
    transformed = tfidf.transform([news_text])
    prediction = model.predict(transformed)[0]
    result = 'FAKE' if prediction == 0 else 'REAL'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)

