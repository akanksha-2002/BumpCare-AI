import streamlit as st
import requests
import speech_recognition as sr
import pyttsx3
import pyaudio  # Ensure PyAudio is imported before using SpeechRecognition

# Backend API URL (Ensure your Flask app is running on port 5001)
API_URL = "http://127.0.0.1:5001/chat"

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    """Convert text to speech"""
    engine.say(text)
    engine.runAndWait()

def transcribe_audio():
    """Capture voice input and convert to text"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text.strip() if text.strip() else "Sorry, I didn't catch that."
        except sr.UnknownValueError:
            return "Could not understand the audio. Please try again."
        except sr.RequestError:
            return "Speech Recognition service is unavailable."

# Streamlit UI
st.title("🤰 MaternAI - Maternal Health Assistant")
st.write("Ask any maternal health-related questions or enter vital signs for risk assessment.")

# Text input from user
user_input = st.text_input("Type your message:")

# Voice input button
if st.button("🎙️ Speak"):
    user_input = transcribe_audio()
    st.text(f"You said: {user_input}")

# Fallback for empty input
if st.button("Send"):
    user_input = user_input.strip()  # Remove unnecessary spaces

    if not user_input:
        st.warning("⚠️ Please enter a message or use voice input.")
    else:
        response = requests.post(API_URL, json={"message": user_input})

        if response.status_code == 200:
            reply = response.json().get("response", "No response from AI.")
            st.success(f"🤖 AI: {reply}")
            speak(reply)  # Convert AI response to speech
        else:
            st.error("Error connecting to AI service.")
