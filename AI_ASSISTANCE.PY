import os
import cv2
import speech_recognition as sr
import pyttsx3
import requests
import pyjokes
import pyaudio
from pyowm import OWM
from deepface import DeepFace
from bs4 import BeautifulSoup
import pyautogui
import webbrowser
import time
import mediapipe as mp

# Initialize text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Set to female voice (or change as needed)
engine.setProperty('rate', 150)  # Adjust speaking rate

# Function to speak text
def speak(text):
    print(f"Speaking: {text}")  # Debugging statement
    engine.say(text)
    engine.runAndWait()

# Function to check available audio devices
def check_microphone():
    p = pyaudio.PyAudio()
    print("Available audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"{i}: {info['name']}")
    p.terminate()

# Function to listen for voice commands
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        r.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = r.listen(source)
        command = ""
        try:
            command = r.recognize_google(audio)
            print(f"You said: {command}")
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        return command

# Function to get weather information
def get_weather(city):
    owm = OWM('ee260a5ca6b08d4add841baf06773b33')  # Replace with your API key
    mgr = owm.weather_manager()
    observation = mgr.weather_at_place(city)
    weather = observation.weather
    return f"The weather in {city} is {weather.status} with a temperature of {weather.temperature('celsius')['temp']}°C."

# Function to tell a joke
def tell_joke():
    joke = pyjokes.get_joke()
    return joke

# Function to detect facial expression
def detect_expression():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emotions
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        # Display the resulting frame with emotion
        cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Facial Expression Detection', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return emotion

# Function to perform a web search
def web_search(query):
    url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('a', class_='result__a', limit=5)
    
    search_results = []
    for result in results:
        title = result.get_text()
        link = result['href']
        search_results.append(f"{title}: {link}")
    
    return search_results

# Function to open applications or websites
def open_application(app_name):
    if "whatsapp" in app_name.lower():
        webbrowser.open("https://web.whatsapp.com")
        speak("Opening WhatsApp.")
    elif "instagram" in app_name.lower():
        webbrowser.open("https://www.instagram.com")
        speak("Opening Instagram.")
    elif "browser" in app_name.lower():
        webbrowser.open("https://www.google.com")
        speak("Opening your browser.")
    else:
        speak("I cannot open that application.")

# Function to scroll Instagram Reels
def scroll_instagram_reels():
    speak("Scrolling through Instagram Reels.")
    time.sleep(2)
    pyautogui.scroll(-300)

# Hand Gesture Detection (MediaPipe)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to detect hand gestures and interpret finger positions
def detect_gestures():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror-like effect
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hand landmarks
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Interpret gestures based on finger positions
                landmarks = hand_landmarks.landmark

                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
                pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y

                thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP].y
                index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
                middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
                ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y
                pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP].y

                # Define finger gestures
                if thumb_tip < index_tip and index_tip > middle_tip:
                    gesture = "Thumbs Up"
                    cv2.putText(frame, "Gesture: Thumbs Up", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    speak("Good job!")
                
                elif middle_tip < index_tip < thumb_tip:  # Middle finger raised
                    gesture = "Middle Finger"
                    cv2.putText(frame, "Gesture: Middle Finger", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    speak("Fuck you!")
                
                elif index_tip < thumb_tip and middle_tip > index_tip:  # Raised index finger
                    gesture = "Index Finger"
                    cv2.putText(frame, "Gesture: Index Finger", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    speak("That's number one!")
                
                elif thumb_tip < index_tip and index_tip < middle_tip < ring_tip < pinky_tip:  # Peace sign
                    gesture = "Peace Sign"
                    cv2.putText(frame, "Gesture: Peace", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    speak("Peace!")
                
                elif pinky_tip < ring_tip < middle_tip < index_tip < thumb_tip:  # Pinky raised
                    gesture = "Pinky Up"
                    cv2.putText(frame, "Gesture: Pinky Up", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    speak("Pinky promise!")
                
                elif thumb_tip < thumb_mcp and index_tip < index_mcp and middle_tip < middle_mcp and ring_tip < ring_mcp and pinky_tip < pinky_mcp:
                    gesture = "Closed Fist"
                    cv2.putText(frame, "Gesture: Closed Fist", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    speak("Fist bump!")

                else:
                    gesture = "Unknown Gesture"
                    cv2.putText(frame, "Gesture: Unknown", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Hand Gesture Recognition', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main loop
check_microphone()  # Check available microphones

while True:
    command = listen().lower()
    
    if 'weather' in command:
        city = command.split("in")[-1].strip()
        weather_info = get_weather(city)
        speak(weather_info)
    elif 'joke' in command:
        joke = tell_joke()
        speak(joke)
    elif 'facial expression' in command:
        emotion = detect_expression()
        speak(f"You seem to be feeling {emotion}.")
    elif 'search for' in command:
        query = command.replace('search for', '').strip()
        search_results = web_search(query)
        for result in search_results:
            speak(result)
    elif 'open' in command:
        app_name = command.replace('open', '').strip()
        open_application(app_name)
    elif 'scroll instagram' in command:
        scroll_instagram_reels()
    elif 'gesture' in command:
        detect_gestures()
    elif 'quit' in command or 'exit' in command:
        speak("Goodbye!")
        break
    else:
        speak("I didn't understand that command.")
