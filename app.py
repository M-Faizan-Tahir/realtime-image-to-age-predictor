import streamlit as st
import cv2
import numpy as np
import os
import logging
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress excessive logging from streamlit-webrtc dependencies
st_webrtc_logger = logging.getLogger("streamlit_webrtc")
st_webrtc_logger.setLevel(logging.WARNING)
aioice_logger = logging.getLogger("aioice")
aioice_logger.setLevel(logging.WARNING)

def load_models():
    try:
        model_files = {
            "face_proto": "opencv_face_detector.pbtxt",
            "face_model": "opencv_face_detector_uint8.pb",
            "age_proto": "age_deploy.prototxt",
            "age_model": "age_net.caffemodel",
            "gender_proto": "gender_deploy.prototxt",
            "gender_model": "gender_net.caffemodel"
        }
        
        for key, file in model_files.items():
            if not os.path.exists(file):
                logger.error(f"Missing model file: {file}")
                st.error(f"Model file {file} not found. Please ensure all model files are in the correct directory.")
                return None, None, None
                
        face_net = cv2.dnn.readNet(model_files["face_model"], model_files["face_proto"])
        age_net = cv2.dnn.readNet(model_files["age_model"], model_files["age_proto"])
        gender_net = cv2.dnn.readNet(model_files["gender_model"], model_files["gender_proto"])
        
        logger.info("Successfully loaded all models")
        return face_net, age_net, gender_net
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def faceBox(faceNet, frame, confidence_threshold=0.7):
    try:
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
        faceNet.setInput(blob)
        detection = faceNet.forward()
        bboxs = []
        for i in range(detection.shape[2]):
            confidence = detection[0, 0, i, 2]
            if confidence > confidence_threshold:
                x1 = int(detection[0, 0, i, 3] * frame_width)
                y1 = int(detection[0, 0, i, 4] * frame_height)
                x2 = int(detection[0, 0, i, 5] * frame_width)
                y2 = int(detection[0, 0, i, 6] * frame_height)
                bboxs.append([x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        return frame, bboxs
    except Exception as e:
        logger.error(f"Error in faceBox: {str(e)}")
        return frame, []

def detect_age_and_gender(frame, face_net, age_net, gender_net, confidence_threshold):
    try:
        if frame is None or frame.size == 0:
            logger.error("Invalid frame received")
            return None, False
        
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        GENDER_LIST = ['Male', 'Female']
        
        # Store original frame and its dimensions
        original_frame = frame.copy()
        original_height, original_width = frame.shape[:2]
        
        # Resize frame for processing
        target_size = (320, 240)
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        proc_height, proc_width = frame.shape[:2]
        
        # Calculate scaling factors
        width_scale = original_width / proc_width
        height_scale = original_height / proc_height
        
        # Detect faces
        frame, bboxs = faceBox(face_net, frame, confidence_threshold)
        faces_detected = len(bboxs) > 0
        
        # Draw on original frame
        output_frame = original_frame.copy()
        for bbox in bboxs:
            x1, y1, x2, y2 = bbox
            # Scale bounding box coordinates to original resolution
            x1 = int(x1 * width_scale)
            y1 = int(y1 * height_scale)
            x2 = int(x2 * width_scale)
            y2 = int(y2 * height_scale)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original_width, x2), min(original_height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Extract face ROI from original frame
            face = original_frame[y1:y2, x1:x2]
            if face.size == 0:
                logger.warning("Empty face ROI detected")
                continue
                
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Gender detection
            gender_net.setInput(blob)
            gender_pred = gender_net.forward()
            gender = GENDER_LIST[gender_pred[0].argmax()]
            
            # Age detection
            age_net.setInput(blob)
            age_pred = age_net.forward()
            age = AGE_LIST[age_pred[0].argmax()]
            
            # Add label on original frame
            label = f"{gender},{age}"
            cv2.rectangle(output_frame, (x1, y1-10), (x2, y1), (0, 255, 0), -1)
            cv2.putText(output_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
        return output_frame, faces_detected
    except Exception as e:
        logger.error(f"Error in detect_age_and_gender: {str(e)}")
        return None, False

class AgeGenderProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_net, self.age_net, self.gender_net = load_models()
        self.confidence_threshold = 0.7  # Default value
        self.faces_detected = False
        self.frame_counter = 0
        self.last_processed_frame = None

    def update_confidence(self, confidence):
        self.confidence_threshold = confidence / 100.0

    def recv(self, frame):
        try:
            if self.face_net is None or self.age_net is None or self.gender_net is None:
                return frame

            # Convert WebRTC frame to OpenCV format
            img = frame.to_ndarray(format="bgr24")
            
            # Increment frame counter
            self.frame_counter += 1
            
            # Process every 6th frame
            if self.frame_counter % 6 == 0:
                # Process frame for age and gender detection
                processed_frame, self.faces_detected = detect_age_and_gender(
                    img, self.face_net, self.age_net, self.gender_net, self.confidence_threshold
                )
                
                if processed_frame is not None:
                    self.last_processed_frame = processed_frame
            else:
                # Use the last processed frame if available, otherwise return the current frame
                processed_frame = self.last_processed_frame if self.last_processed_frame is not None else img

            if processed_frame is None:
                return frame

            # Convert back to WebRTC frame
            return av.VideoFrame.from_ndarray(processed_frame, format="rgb24")
        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
            return frame

def main():
    st.set_page_config(page_title="Real-Time Age and Gender Detection", layout="centered")
    st.title("Real-Time Age and Gender Detection")
    st.write("Click 'Start' to begin real-time webcam streaming for age and gender detection.")

    # Initialize session state
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = 70
    if 'streamer' not in st.session_state:
        st.session_state.streamer = None

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        st.session_state.confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
            key="confidence_slider"
        )

    # WebRTC configuration
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Initialize WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="age-gender-detection",
        video_processor_factory=AgeGenderProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    # Status and frame display
    status_placeholder = st.empty()
    frame_placeholder = st.empty()

    if webrtc_ctx.video_processor:
        # Update confidence threshold dynamically
        webrtc_ctx.video_processor.update_confidence(st.session_state.confidence_threshold)

        if webrtc_ctx.state.playing:
            status_placeholder.write(f"Streaming... Confidence Threshold: {st.session_state.confidence_threshold}%")
        else:
            status_placeholder.write("Click 'Start' to begin streaming.")
    else:
        status_placeholder.write("Camera not initialized.")

if __name__ == "__main__":
    main()