import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import av
import os
from twilio.rest import Client
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model paths
FACE_PROTO = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

# Model parameters
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_LIST = ["Male", "Female"]

def load_models():
    try:
        face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
        age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
        gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
        logger.info("Successfully loaded all models")
        return face_net, age_net, gender_net
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error(f"Error loading models: {e}")
        return None, None, None

face_net, age_net, gender_net = load_models()

def detect_age_and_gender(frame, confidence_threshold=0.7):
    if face_net is None or age_net is None or gender_net is None:
        return frame, "Model loading failed", "N/A"

    frame_opencv = frame.copy()
    h, w = frame_opencv.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_opencv, 1.0, (320, 240), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    age = "N/A"
    gender = "N/A"
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame_opencv[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Gender detection
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]

            # Age detection
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]

            # Draw rectangle and label
            label = f"{gender}, {age}, Conf: {confidence:.2f}"
            cv2.rectangle(frame_opencv, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame_opencv, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            break

    return frame_opencv, age, gender

class AgeGenderProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence_threshold = 0.7

    def update_confidence(self, threshold):
        self.confidence_threshold = threshold

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img, age, gender = detect_age_and_gender(img, self.confidence_threshold)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            st.error(f"Error processing frame: {e}")
            return frame

# Twilio TURN server configuration
def get_twilio_ice_servers():
    try:
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        if not account_sid or not auth_token:
            logger.warning("Twilio credentials not set, falling back to STUN")
            return [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]}
            ]
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        logger.info("Successfully retrieved Twilio ICE servers")
        return token.ice_servers
    except Exception as e:
        logger.error(f"Error retrieving Twilio ICE servers: {e}")
        return [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]}
        ]

st.title("Real-Time Age and Gender Detection")
st.write("**Instructions**: Click 'Start' to enable your webcam. Grant browser permissions. Adjust the confidence threshold to fine-tune detection. Streaming may take a few seconds to initialize.")

confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05, key="confidence")
try:
    ctx = webrtc_streamer(
        key="age-gender-detection",
        video_processor_factory=AgeGenderProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": get_twilio_ice_servers(), "iceTransportPolicy": "all"}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        verbose=True
    )
    if ctx.video_processor:
        ctx.video_processor.update_confidence(confidence)
except Exception as e:
    logger.error(f"WebRTC error: {e}")
    st.error(f"WebRTC error: {e}. Check network or browser permissions.")