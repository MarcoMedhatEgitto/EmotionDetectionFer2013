# # Inserting the main libraries
# import cv2
# import numpy as np
# from tensorflow.keras.models import model_from_json
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer

# #Definig the emotion categories
# emitionCat = {
#     0: "Angry",
#     1: "Disgusted",
#     2: "Neutral",
#     3: "Happy",
#     4: "Fearful",
#     5: "Sad",
#     6: "Surprised",
# }

# # Starting streamlit server
# onlineFrame = st.empty()
# st.title("Emotion detector")
# # Loading the model and model weights
# jsonFile = open('Models/model.json', 'r')
# loaded_model_json = jsonFile.read()
# jsonFile.close()
# model = model_from_json(loaded_model_json)
# model.load_weights("Models/model.weights.h5")
# print("Loaded model from disk")

# # Starting the camera
# cap = cv2.VideoCapture(1)

# # Starting the Video and the application
# while True:
#     ret, frame = cap.read()
#     # ret, frame = cap.read()
#     if not ret:
#         st.write("Camera is not working")
#         break

#     frame = cv2.resize(frame, (1280, 720))
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     faceDetected = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceDetected.detectMultiScale(gray, 1.3, 5)

#     if faces is not None:
#         for (x, y, w, h) in faces:
#             cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#             roi_gray = gray[y:y + h, x:x + w]
#             cropped_face = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), axis=-1), axis=0)
#             result = model.predict(cropped_face)
#             maxindex = int(np.argmax(result))
#             cv2.putText(rgb_frame, emitionCat[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
#                         cv2.LINE_4)
#     onlineFrame.image(rgb_frame)
#     # cv2.imshow('Video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import av
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Defining the emotion categories
emotionCat = {
    0: "Angry",
    1: "Disgusted",
    2: "Neutral",
    3: "Happy",
    4: "Fearful",
    5: "Sad",
    6: "Surprised",
}

# Load the model and model weights
jsonFile = open('model.json', 'r')
loaded_model_json = jsonFile.read()
jsonFile.close()
model = model_from_json(loaded_model_json)
model.load_weights("Models/model.weights.h5")
st.title("Emotion Detector")
st.success("Model loaded successfully!")
face_detected = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Define the Video Transformer
class EmotionDetector(VideoTransformerBase):
    def recv(self, frame):
        # Convert the frame to OpenCV BGR format
        img = frame.to_ndarray(format="bgr24")
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detected.detectMultiScale(gray, 1.3, 5)

        # Process detected faces
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Draw rectangle around the face in the original image (BGR)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Extract the region of interest (face) for emotion detection
                roi_gray = gray[y:y + h, x:x + w]
                cropped_face = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), axis=-1), axis=0)

                # Predict emotion
                result = model.predict(cropped_face)
                maxindex = int(np.argmax(result))

                # Add the emotion label to the frame
                cv2.putText(img, emotionCat[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_4)

        # Return the processed frame to the WebRTC stream (display the updated image)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Run the WebRTC streamer with Emotion Detection
webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionDetector,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
