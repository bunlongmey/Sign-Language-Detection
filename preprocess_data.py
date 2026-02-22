import cv2
import os
import mediapipe as mp

# -----------------------------
# Paths
# -----------------------------
input_dir = './data'                # folder from Step 1
output_dir = './processed_data'     # where preprocessed images go
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# MediaPipe setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,      # images, not video
    max_num_hands=1,             # only one hand per image
    min_detection_confidence=0.7
)

# -----------------------------
# Image processing
# -----------------------------
for label_folder in os.listdir(input_dir):
    label_path = os.path.join(input_dir, label_folder)
    if not os.path.isdir(label_path):
        continue

    output_path = os.path.join(output_dir, label_folder)
    os.makedirs(output_path, exist_ok=True)

    for image_file in os.listdir(label_path):
        image_path = os.path.join(label_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            # Draw landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Save the processed image
        save_path = os.path.join(output_path, image_file)
        cv2.imwrite(save_path, image)

print("Preprocessing complete!")
hands.close()


# import cv2
# import os
# import mediapipe as mp
# import numpy as np

# # -----------------------------
# # Directories
# # -----------------------------
# input_dir = './data'  # fixed: remove comma
# output_dir = './processed_data'
# os.makedirs(output_dir, exist_ok=True)

# # -----------------------------
# # MediaPipe setup
# # -----------------------------
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# # Hands detection setup
# hands = mp_hands.Hands(
#     static_image_mode=True,      # processing images, not video
#     max_num_hands=1,             # detect one hand
#     min_detection_confidence=0.7
# )

# image_size = (224, 224)  # optional: resize images for visualization

# # -----------------------------
# # Process each label folder
# # -----------------------------
# for label_folder in os.listdir(input_dir):
#     label_path = os.path.join(input_dir, label_folder)
#     if not os.path.isdir(label_path):
#         continue

#     output_path = os.path.join(output_dir, label_folder)
#     os.makedirs(output_path, exist_ok=True)

#     for image_file in os.listdir(label_path):
#         image_path = os.path.join(label_path, image_file)
#         image = cv2.imread(image_path)
#         if image is None:
#             continue

#         # Resize image (optional)
#         image = cv2.resize(image, image_size)

#         # Convert BGR to RGB
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Process with MediaPipe
#         result = hands.process(image_rgb)

#         if result.multi_hand_landmarks:
#             hand_landmarks = result.multi_hand_landmarks[0]

#             # Extract 21 landmarks (x, y, z)
#             landmarks = []
#             for lm in hand_landmarks.landmark:
#                 landmarks.append([lm.x, lm.y, lm.z])

#             # Flatten landmarks to 1D array
#             landmarks = np.array(landmarks).flatten()

#             # Save landmarks as .npy file
#             save_path = os.path.join(output_path, image_file.replace('.jpg', '.npy'))
#             np.save(save_path, landmarks)

#             # Optional: draw landmarks and save visualization
#             mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             vis_path = os.path.join(output_path, image_file)
#             cv2.imwrite(vis_path, image)

# print("Preprocessing complete!")
# hands.close()



# # import cv2
# # import os 
# # import mediapipe as mp

# # input_dir = ',/data'
# # output_dir = './processed_data'
# # os.makedirs(output_dir, exist_ok=True)

# # mp_hands = mp.solutions.hands
# # hands = mp_hands.Hands(static_image_mode=True, man_num_hand)
# # mp_drawing = mp.solution.drawing_utils

# # image_size = (224, 224)

# # for label_folder in os.listdir(input_dir):
# #     label_path = os.path.join(input_dir, label_folder)
# #     output_path = os.path.join(output_dir, label_folder)
# #     os.makedirs(output_path, exist_ok=True)

# #     for image_file in os.listdir(label_path):
# #         image_path = os.path.join(label_path, image_file)
# #         image = cv2.imread(image_path)

# #         image_rgb = cv2.cvColor(image, cv2.COLOR_BGR2RGB)


