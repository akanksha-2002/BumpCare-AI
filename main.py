from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load a free model from Hugging Face (mistral or llama2)
chatbot = pipeline("text2text-generation", model="google/flan-t5-small")


SYSTEM_PROMPT = """
You are MaternAI, an AI-powered maternal health assistant.
You provide expert maternal health guidance in simple, understandable language.
If a user reports severe symptoms like high blood pressure or excessive bleeding, strongly advise them to see a doctor.
"""

def maternai_response(user_input):
    """Generate AI-based responses for maternal health queries."""
    prompt = f"{SYSTEM_PROMPT}\nUser: {user_input}\nAI:"
    response = chatbot(prompt, max_length=200, do_sample=True)
    return response[0]["generated_text"].split("AI:")[-1].strip()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    bot_reply = maternai_response(user_message)
    return jsonify({"response": bot_reply})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
