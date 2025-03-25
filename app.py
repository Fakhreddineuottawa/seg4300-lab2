import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from transformers import pipeline

# 1) Load variables from .env into environment
load_dotenv()  

# 2) Retrieve variables using os.getenv(...)
SECRET_KEY = os.getenv("SECRET_KEY")
DB_USER    = os.getenv("DB_USER")
DB_PASS    = os.getenv("DB_PASSWORD")

# 3) Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY  # for session or other usage

# 4) Example: If you want to see them, do NOT do this in production
# print("Debug -> SECRET_KEY:", SECRET_KEY)
# print("Debug -> DB_USER:", DB_USER, "DB_PASS:", DB_PASS)

# 5) Example pipeline code (as before)
model = pipeline("sentiment-analysis")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    prediction = model(text)
    return jsonify(prediction)

# 6) Optional: demonstration route to confirm secrets injection
@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "secret_key_used_by_flask": app.config['SECRET_KEY'],
        "db_user": DB_USER  # do not reveal actual secrets in production
    })

if __name__ == '__main__':
    # 7) Run your Flask app
    app.run(host='0.0.0.0', port=5000)
