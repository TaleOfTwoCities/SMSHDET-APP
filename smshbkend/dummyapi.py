from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('sms_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    message = data['message']
    
    # Preprocess and predict
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)
    
    return jsonify({'result': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
