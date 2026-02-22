import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from gtts import gTTS
import threading
import subprocess
import os

# -----------------------------
# Load model & labels
# -----------------------------
model = tf.keras.models.load_model('./models/asl_model.h5')
labels = ['goodbye', 'hello', 'I_Love_You', 'thanks', 'where', 'yes']  # same as training

# -----------------------------
# MediaPipe hands
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# -----------------------------
# Helper functions
# -----------------------------
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def speak(text):
    # Run in separate thread so it doesn't block webcam
    def _speak():
        tts = gTTS(text=text, lang='en')
        temp_file = 'temp.mp3'
        tts.save(temp_file)
        # macOS audio player
        subprocess.run(['afplay', temp_file])
        os.remove(temp_file)
    threading.Thread(target=_speak).start()

# -----------------------------
# Webcam loop
# -----------------------------
cap = cv2.VideoCapture(0)
print("Starting webcam... Press 'q' to quit.")

last_prediction = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Preprocess and predict
        img_input = preprocess_frame(frame)
        prediction = model.predict(img_input)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction)

        if confidence > 0.7:
            label_text = f"{labels[class_id]} ({confidence*100:.1f}%)"
            cv2.putText(frame, label_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Speak only if prediction changed
            if last_prediction != class_id:
                speak(labels[class_id])
                last_prediction = class_id

    cv2.imshow('ASL Real-Time Detection', frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()


# import cv2
# import mediapipe as mp
# import tensorflow as tf
# import numpy as np
# from gtts import gTTS
# import subprocess
# import threading
# import os
# import time

# # -----------------------------
# # Load trained model
# # -----------------------------
# model = tf.keras.models.load_model('./models/asl_model.h5')

# labels = ['goodbye', 'hello', 'I_Love_You', 'thanks', 'where', 'yes']

# # -----------------------------
# # MediaPipe Hands setup
# # -----------------------------
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# image_size = (224, 224)

# # -----------------------------
# # Webcam
# # -----------------------------
# cap = cv2.VideoCapture(0)
# print("Starting webcam... Press 'q' to quit.")

# last_prediction = None
# last_spoken_time = 0
# speech_delay = 2  # seconds

# # -----------------------------
# # Text-to-Speech (Google TTS)
# # -----------------------------
# def gtts_tts(text):
#     tts = gTTS(text=text, lang='en')
#     temp_audio_file = 'temp_audio.mp3'
#     tts.save(temp_audio_file)

#     # macOS audio player
#     subprocess.run(["afplay", temp_audio_file], stdout=subprocess.DEVNULL)
#     os.remove(temp_audio_file)

# def async_generate_voice(text):
#     threading.Thread(target=gtts_tts, args=(text,), daemon=True).start()

# # -----------------------------
# # Real-time detection loop
# # -----------------------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape

#     # Convert to RGB
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(frame_rgb)

#     if result.multi_hand_landmarks:
#         hand_landmarks = result.multi_hand_landmarks[0]

#         # Get bounding box of hand
#         x_coords = [lm.x for lm in hand_landmarks.landmark]
#         y_coords = [lm.y for lm in hand_landmarks.landmark]

#         x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
#         y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

#         # Add padding
#         padding = 20
#         x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
#         x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

#         hand_img = frame[y_min:y_max, x_min:x_max]

#         if hand_img.size > 0:
#             # Resize & normalize
#             hand_img = cv2.resize(hand_img, image_size)
#             hand_img = hand_img / 255.0
#             hand_img = np.expand_dims(hand_img, axis=0)

#             # Prediction
#             predictions = model.predict(hand_img, verbose=0)
#             class_id = np.argmax(predictions)
#             confidence = predictions[0][class_id]

#             label = labels[class_id]

#             # Display result
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#             cv2.putText(
#                 frame,
#                 f"{label} ({confidence:.2f})",
#                 (x_min, y_min - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (0, 255, 0),
#                 2
#             )

#             # Speak only if confidence is high and label changes
#             current_time = time.time()
#             if confidence > 0.85 and label != last_prediction and current_time - last_spoken_time > speech_delay:
#                 async_generate_voice(label.replace('_', ' '))
#                 last_prediction = label
#                 last_spoken_time = current_time

#         # Draw landmarks
#         mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     cv2.imshow("ASL Real-Time Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # -----------------------------
# # Cleanup
# # -----------------------------
# cap.release()
# cv2.destroyAllWindows()
# hands.close()


# # import cv2
# # import mediapipe as mp
# # import tensorflow as tf
# # import numpy as np
# # from gtts import gTTs
# # import os
# # import subprocess
# # import threading

# # model = tf.keras.models.load_model('.models/asl_model.h5')
# # labels = ['goodbye', 'hello', 'I_Love_You', 'thanks', 'where', 'yes']

# # mp_hands = mp.solution.hands
# # hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
# # mp_drawing = mp.solution.drawing_utils

# # image_size = (224, 224)

# # cap cv2.VideoCapture(0)
# # print("Satrting webcam... Press 'q' to quit.")

# # last_prediction = None

# # #gTTs function
# # def gtts_tts(tts_text):
# #     tts = gTTS(tts_text, lang='ja')

# #     temp_audio_file = 'temp_audio.mp3'
# #     tts.save(temp_audio_file)

# #     subprocess.run(["afplay", temp_audio_file])

# # def async_generate_voice(text):

# #     threading.Thread(target=gtts_tts, args=(text,)).start()

# # while True:

