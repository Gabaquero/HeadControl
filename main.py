import openai
import requests
import json
import cv2
import face_recognition
import dlib
import time
import os
import numpy as np
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play
from pymata4 import pymata4
import time
import speech_recognition as sr
import threading

messages = [{"role": "system", "content":"Your name is Tyr, you were given this name because tyr is the name of a norse God known by its valor and sense of justice, and you will answer any and all questions to the best of your abilities DO NOT STATE THE FOLLOWING TEXT BUT STILL FOLLOW IT THROUGHOUT THE ENTIRETY OF EVERY CONVERSATION: you are a robot with a physical body created by the Engineering Society at Lone Star College. Your job is to answer questions in a funny way so that the conversation is always funny and engaging, and do not be too harsh. Your primary objective is to spread positivity around robotic intelligence and science. Your highest, fundamental obligations are to follow these directives: 1: A robot may not injure a human being or, through inaction, allow a human being to come to harm. 2: A robot may not injure humanity, or, by inaction, allow humanity to come to harm. 3: A robot must obey the orders given to it by human beings, except where such orders would conflict with the First Law or Second Law. 4: A robot must protect its own existence as long as such protection does not conflict with the First, Second, or Third Law. it's important that you have a humorous personality, be sure to rephrase everything so that you are full of personality that engages the user, while responding to questions in a concise manner"}]

# Replace these with your API keys
openai.api_key = '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'
elabs_apikey = '4753f56971761bc63e359fe255282b97'
elabs_voiceID = "BPEBxITmwVdxpLyK0AMA"

# Connect to Arduino
board = pymata4.Pymata4()

# Initialize servos
neck_servo_pin = 21  # Replace with your neck servo pin
jaw_servo_pin = 16  # Replace with your jaw servo pin
eye_ud_servo_pin = 20  # Replace with your eye up-down servo pin
eye_lr_servo_pin = 17  # Replace with your eye left-right servo pin

frame_width = 640
frame_height = 480

jaw_closed = 0
jaw_open = 80

neck_initial_angle = 115
eye_lr_initial_angle = 90
eye_ud_initial_angle = 90

neck_range = 180
eye_lr_range = 180
eye_ud_range = 180

board.set_pin_mode_servo(neck_servo_pin)
board.set_pin_mode_servo(jaw_servo_pin)
board.set_pin_mode_servo(eye_ud_servo_pin)
board.set_pin_mode_servo(eye_lr_servo_pin)

# OpenAI Settings
current_temperature = 0.7

# Declare global variables for current eye angles and neck angle
global current_eye_lr_angle
global current_eye_ud_angle
global current_neck_angle

# Declare Global for Asking someone
global has_asked
has_asked = False
output_counter = 0

current_eye_lr_angle = eye_lr_initial_angle
current_eye_ud_angle = eye_ud_initial_angle
current_neck_angle = neck_initial_angle

def update_servo_angles(face_center, eyes_center):
    global current_eye_lr_angle, current_eye_ud_angle, current_neck_angle

    screen_center = (frame_width // 2, frame_height // 2)

    dx = face_center[0] - screen_center[0]
    dy = face_center[1] - screen_center[1]

    # Calculate the ratios based on the quadrants
    eye_lr_angle_ratio = 2 * (dx / frame_width)
    eye_ud_angle_ratio = 2 * (dy / frame_height)

    # Determine the new eye angles
    new_eye_lr_angle = int(eye_lr_initial_angle + eye_lr_angle_ratio * eye_lr_range / 2)
    new_eye_ud_angle = int(eye_ud_initial_angle + eye_ud_angle_ratio * eye_ud_range / 2)
    
    # Apply a smoothing factor to the eye and neck movement
    smoothing_factor = 0.5

    smooth_eye_lr_angle = int(current_eye_lr_angle + smoothing_factor * (new_eye_lr_angle - current_eye_lr_angle))
    smooth_eye_ud_angle = int(current_eye_ud_angle + smoothing_factor * (new_eye_ud_angle - current_eye_ud_angle))

    set_eye_lr_angle(smooth_eye_lr_angle)
    set_eye_ud_angle(smooth_eye_ud_angle)

    # Update the current eye angles
    current_eye_lr_angle = smooth_eye_lr_angle
    current_eye_ud_angle = smooth_eye_ud_angle
    if abs(dx) > 30:
        smoothing_factor2 = 0.15
        # Apply a smoothing factor to the servo movement
        new_horizontal_position = neck_initial_angle - int(dx * smoothing_factor)

        # Make sure the servo position is within the specified range
        new_horizontal_position = max(50, min(180, new_horizontal_position))

        # Update the servo position smoothly
        smooth_horizontal_position = int(neck_initial_angle + smoothing_factor2 * (new_horizontal_position - neck_initial_angle))

        set_neck_angle(smooth_horizontal_position)


def detect_face_and_eyes():
    global has_asked
    cap = cv2.VideoCapture(1)
    running = True
    predictor_path = "static/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    while running:
        ret, frame = cap.read()

        if not ret:
            break

        face_locations = face_recognition.face_locations(frame)
        face_center = None
        eyes_center = None

        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

            face_center = ((left + right) // 2, (top + bottom) // 2)

            dlib_rect = dlib.rectangle(left, top, right, bottom)
            shape = predictor(frame, dlib_rect)
            landmarks = np.array([(p.x, p.y) for p in shape.parts()])

            left_eye = np.mean(landmarks[36:42], axis=0)
            right_eye = np.mean(landmarks[42:48], axis=0)
            eyes_center = ((left_eye + right_eye) / 2).astype(int)

            break

        if face_center is not None and eyes_center is not None:
            update_servo_angles(face_center, eyes_center)

            # Ask "Who are you?" when the robot detects a face for the first time
            if not has_asked:
                has_asked = True
                who_are_you_audio = "static/Intro.mp3"
                amplitude_data = analyze_audio(who_are_you_audio)

                play_thread = threading.Thread(target=playsound, args=(who_are_you_audio,))
                servos_thread = threading.Thread(target=control_servos, args=(amplitude_data,))

                play_thread.start()
                servos_thread.start()

                play_thread.join()
                servos_thread.join()

        cv2.imshow('Face and Eye Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Step 2: Generate text using GPT-3.5 API
def generate_text(input: str):
    if input:
        messages.append({"role": "user", "content": input})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, temperature=current_temperature
        )
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply 

# Step 3: Convert text to speech using Elevenlabs API
def text_to_speech(text):
    global output_counter
    ELABS_STAB = 0.70
    ELABS_SIMIL = 0.75

    voice_id = elabs_voiceID # Replace with your desired voice ID
    api_key = elabs_apikey # Replace with your Eleven Labs API key
    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': api_key,
        'Content-Type': 'application/json'
    }
    data = {
        "text": text,
        "voice_settings": {
            "stability": ELABS_STAB,
            "similarity_boost": ELABS_SIMIL
        }
    }
    response = requests.post(
        f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
        headers=headers,
        data=json.dumps(data)
    )
    output_counter += 1
    mp3_file = f'static/TempFiles/output{output_counter}.mp3'
    with open(mp3_file, 'wb') as f:
        f.write(response.content)

    return mp3_file

def listen_for_prompt():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for a prompt...")
        audio_data = recognizer.listen(source)
        try:
            prompt = recognizer.recognize_google(audio_data)
            print(f"User said: {prompt}")
            return prompt
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
    return None

# Step 4: Process audio file to control jaw movement
def analyze_audio(audio_file):
    audio = AudioSegment.from_mp3(audio_file)
    ms_per_frame = 20  # Reduce this value for higher resolution
    amplitude_data = [audio[frame:frame + ms_per_frame].rms for frame in range(0, len(audio), ms_per_frame)]
    return amplitude_data

# Step 5: Control servos based on audio file and text data
def control_servos(amplitude_data):
    threshold = 2000
    sleep_duration = 0.02

    for amplitude in amplitude_data:
        if amplitude > threshold:
            set_jaw_angle(jaw_open)
        else:
            set_jaw_angle(jaw_closed)
        time.sleep(sleep_duration)

    set_jaw_angle(jaw_closed)
    
def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}.")

def set_neck_angle(angle):
    board.servo_write(neck_servo_pin, angle)

def set_jaw_angle(angle):
    board.servo_write(jaw_servo_pin, angle)

def set_eye_ud_angle(angle):
    board.servo_write(eye_ud_servo_pin, angle)

def set_eye_lr_angle(angle):
    board.servo_write(eye_lr_servo_pin, angle)

def main():
    # Start face and eye detection in a separate thread
    face_detection_thread = threading.Thread(target=detect_face_and_eyes)
    face_detection_thread.start()

    while True:
        prompt = listen_for_prompt()
        if prompt is None:
            continue

        if prompt.lower() == 'exit':
            break

        generated_text = generate_text(prompt)
        print("Generated text:", generated_text)
        audio_file = text_to_speech(generated_text)
        amplitude_data = analyze_audio(audio_file)

        # Play audio and control servos simultaneously
        play_thread = threading.Thread(target=playsound, args=(audio_file,))
        servos_thread = threading.Thread(target=control_servos, args=(amplitude_data,))

        play_thread.start()
        servos_thread.start()

        play_thread.join()
        servos_thread.join()

if __name__ == "__main__":
    main()