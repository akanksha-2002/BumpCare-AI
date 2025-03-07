from transformers import pipeline
from flask import Flask, request, jsonify
import re
import joblib
import pandas as pd

app = Flask(__name__)

# Load a free model from Hugging Face (mistral or llama2)
chatbot = pipeline("text2text-generation", model="google/flan-t5-large")

SYSTEM_PROMPT = """
You are MaternAI, an AI-powered maternal health assistant.
You provide expert maternal health guidance in simple, understandable language.
If a user reports severe symptoms like high blood pressure or excessive bleeding, strongly advise them to see a doctor.
"""

# Load the trained model
model_filename = "maternal_risk_model.pkl"
model = joblib.load(model_filename)

# Function to predict risk level
def predict_risk(input_data):
    df_input = pd.DataFrame([input_data])
    feature_columns = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
    df_input = df_input[feature_columns]
    prediction = model.predict(df_input)[0]
    risk_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    return risk_mapping[prediction]

# Function to extract health data from user input
def extract_health_data(user_input):
    try:
        age = re.search(r"(\d+)\s*(?:years|year|yo|age)?", user_input, re.IGNORECASE)
        bp = re.search(r"(\d{2,3})\/(\d{2,3})", user_input)
        bs = re.search(r"Blood Sugar\s*(\d+(\.\d+)?)", user_input, re.IGNORECASE)
        temp = re.search(r"Temperature\s*(\d+(\.\d+)?)", user_input, re.IGNORECASE)
        hr = re.search(r"Heart Rate\s*(\d+)", user_input, re.IGNORECASE)

        if age and bp and bs and temp and hr:
            return {
                "Age": float(age.group(1)),
                "SystolicBP": float(bp.group(1)),
                "DiastolicBP": float(bp.group(2)),
                "BS": float(bs.group(1)),
                "BodyTemp": float(temp.group(1)),
                "HeartRate": float(hr.group(1))
            }
        return None
    except Exception:
        return None

# Function to process structured health data
def chatbot_response(input_data):
    try:
        risk_level = predict_risk(input_data)
        return f"✅ Based on the input, the predicted maternal health risk level is: {risk_level}."
    except Exception as e:
        return f"⚠️ Error: Unable to process data. Details: {str(e)}"

# Function to handle chatbot and risk prediction responses
def maternai_response(user_input):
    structured_data = extract_health_data(user_input)
    if structured_data:
        return chatbot_response(structured_data)
    
    prompt = f"{SYSTEM_PROMPT}\nUser: {user_input}\nAI:"
    response = chatbot(user_input, max_length=200, temperature=0.7, do_sample=True)


    return response[0]["generated_text"].split("AI:")[-1].strip()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    bot_reply = maternai_response(user_message)
    return jsonify({"response": bot_reply})

if __name__ == '__main__':
    app.run(debug=True, port=5001)