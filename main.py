import cv2
import face_recognition
import speech_recognition as sr
import pyttsx3
import threading
import time
from openai import OpenAI

# Import our custom components
import config
from face_display import RobotFace

# Initialize components
client = OpenAI(api_key=config.OPENAI_API_KEY)
face_renderer = RobotFace()

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 175)
engine.setProperty('volume', 1.0)

# Global variables to pass data between execution loops safely
robot_mood = "Idle 😐"
frame_counter = 0
is_speaking = False

def speak(text):
    """Voice execution thread wrapper."""
    global robot_mood, is_speaking
    is_speaking = True
    engine.say(text)
    engine.runAndWait()
    is_speaking = False

def trigger_speech(text):
    """Runs speaking functions on a separate thread to prevent application freezing."""
    threading.Thread(target=speak, args=(text,), daemon=True).start()

def get_ai_response(user_input):
    """Calls ChatGPT for processing user phrases."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices.message.content
    except Exception as e:
        return "System logic error. Check internet connection or API settings."

def listen_to_voice():
    """Captures microphone voice streaming data."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=4)
            return recognizer.recognize_google(audio)
        except (sr.UnknownValueError, sr.WaitTimeoutError):
            return None

def voice_interaction_loop():
    """Background loop managing active dialogues when the owner is near."""
    global robot_mood
    while True:
        if robot_mood == "Happy 😊" and not is_speaking:
            print("[System Status]: Ready for voice command input...")
            user_text = listen_to_voice()
            
            if user_text:
                print(f"User Input Detected: {user_text}")
                
                if "bye" in user_text.lower() or "goodbye" in user_text.lower():
                    robot_mood = "Idle 😐"
                    trigger_speech("Goodbye for now!")
                    time.sleep(2)
                    continue
                
                # Update visual state machines sequentially
                robot_mood = "Thinking 🤔"
                ai_reply = get_ai_response(user_text)
                
                robot_mood = "Talking 💬"
                print(f"Robot Response: {ai_reply}")
                
                # Speak out loud and wait until complete
                speak(ai_reply)
                
                # Return back to looking at you happily
                robot_mood = "Happy 😊"
        time.sleep(0.2)

def main():
    global robot_mood, frame_counter
    
    # Load user verification assets
    try:
        owner_image = face_recognition.load_image_file("my_face.jpg")
        owner_encoding = face_face_encodings = face_recognition.face_encodings(owner_image)[0]
    except IndexError:
        print("Error: Could not extract facial features from my_face.jpg. Please use a clearer photo.")
        return
    except FileNotFoundError:
        print("Error: Please drop a photo named 'my_face.jpg' into the project folder first.")
        return

    # Start webcam capture feeds
    video_capture = cv2.VideoCapture(0)
    
    # Fire up the background microphone monitoring agent
    threading.Thread(target=voice_interaction_loop, daemon=True).start()
    
    print("Application successfully initialized. Open the 'Robot Face' display window.")
    
    while True:
        frame_counter += 1
        ret, frame = video_capture.read()
        if not ret:
            break
            
        # Periodically scan the camera frames to identify if you are there
        if frame_counter % 10 == 0 and robot_mood != "Thinking 🤔" and robot_mood != "Talking 💬":
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            if face_encodings:
                match = face_recognition.compare_faces([owner_encoding], face_encodings[0])
                if match[0] and robot_mood == "Idle 😐":
                    robot_mood = "Happy 😊"
                    trigger_speech("Hello master, nice to see you!")
            else:
                # Revert to sleepy standby if you leave the field of view
                if robot_mood == "Happy 😊":
                    robot_mood = "Idle 😐"

        # 1. Render and draw the Robot's dynamic face window
        face_window = face_renderer.render_face(robot_mood, frame_counter)
        cv2.imshow("Robot Face Screen", face_window)
        
        # 2. Render normal tracking camera window (so you see what the webcam sees)
        cv2.imshow("Robot Vision Camera Tracker", frame)

        # Break code loop via hitting the 'q' key
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
