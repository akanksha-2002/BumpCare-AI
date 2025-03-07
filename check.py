import speech_recognition as sr

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Say something...")
    
    recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjust for background noise
    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)  # Increase listening time

try:
    text = recognizer.recognize_google(audio)
    print(f"You said: {text}")
except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError:
    print("Could not request results; check your internet connection")
