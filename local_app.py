import streamlit as st
import cv2
import numpy as np
import os
import logging
import platform
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Function to list available cameras
def list_available_cameras(max_index=20):
    available_cameras = []
    backends = [
        (cv2.CAP_DSHOW, "DirectShow") if platform.system() == "Windows" else (None, "Default"),
        (cv2.CAP_V4L2, "V4L2") if platform.system() == "Linux" else (None, "Default"),
        (cv2.CAP_FFMPEG, "FFMPEG"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
        (None, "Default")
    ]
    
    for index in range(max_index):
        for backend, backend_name in backends:
            if backend is None and backend_name != "Default":
                continue
            try:
                cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
                if cap.isOpened():
                    camera_id = f"Camera {index + 1}"
                    available_cameras.append((camera_id, index, backend))
                    logger.info(f"Found camera: {camera_id}")
                    cap.release()
                    break
                cap.release()
            except Exception as e:
                logger.debug(f"No camera at index {index} with {backend_name} backend: {str(e)}")
    
    if not available_cameras:
        logger.error("No cameras detected")
    return available_cameras

# Function to load models with caching
@st.cache_resource
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

# Function to detect faces and return bounding boxes
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

# Function to process frame for age and gender detection
def detect_age_and_gender(frame, face_net, age_net, gender_net, confidence_threshold):
    try:
        if frame is None or frame.size == 0:
            logger.error("Invalid frame received")
            return None, False
        
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        GENDER_LIST = ['Male', 'Female']
        
        # Resize frame while maintaining aspect ratio
        target_size = (640, 480)
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        
        # Detect faces
        frame, bboxs = faceBox(face_net, frame, confidence_threshold)
        faces_detected = len(bboxs) > 0
        
        for bbox in bboxs:
            x1, y1, x2, y2 = bbox
            # Ensure valid bounding box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
                
            face = frame[y1:y2, x1:x2]
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
            
            # Add label
            label = f"{gender},{age}"
            cv2.rectangle(frame, (x1, y1-10), (x2, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame, faces_detected
    except Exception as e:
        logger.error(f"Error in detect_age_and_gender: {str(e)}")
        return None, False

# Main Streamlit app
async def main():
    st.set_page_config(page_title="Real-Time Age and Gender Detection", layout="centered")
    st.title("Real-Time Age and Gender Detection")
    st.write("Configure settings in the sidebar and press Start to begin real-time age and gender detection.")
    
    # Initialize session state
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = 70
        
    try:
        # Load models
        face_net, age_net, gender_net = load_models()
        if face_net is None or age_net is None or gender_net is None:
            return
            
        # List available cameras
        cameras = list_available_cameras()
        if not cameras:
            st.error("No cameras detected. Please ensure a webcam is connected and try again.")
            return
            
        # Sidebar configuration
        with st.sidebar:
            st.header("Configuration")
            # Camera selection
            camera_options = [cam[0] for cam in cameras]
            selected_camera = st.selectbox("Select Camera", camera_options, index=0)
            
            # Confidence threshold slider (0-100%)
            st.session_state.confidence_threshold = st.slider(
                "Confidence Threshold (%)",
                min_value=0,
                max_value=100,
                value=70,
                step=1,
                key="confidence_slider"
            )
        
        # Convert percentage to decimal for face detection
        confidence_threshold_decimal = st.session_state.confidence_threshold / 100.0
        
        # Find selected camera details
        selected_camera_info = next(cam for cam in cameras if cam[0] == selected_camera)
        camera_index, backend = selected_camera_info[1], selected_camera_info[2]
        
        # UI controls
        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("Start", key="start")
        with col2:
            stop_button = st.button("Stop", key="stop")
            
        status_placeholder = st.empty()
        frame_placeholder = st.empty()
        
        if start_button and not st.session_state.streaming:
            st.session_state.streaming = True
            # Initialize webcam
            st.session_state.cap = cv2.VideoCapture(camera_index, backend) if backend is not None else cv2.VideoCapture(camera_index)
            if not st.session_state.cap.isOpened():
                st.error(f"Cannot access camera {selected_camera}. Ensure no other apps are using the camera.")
                st.session_state.streaming = False
                return
                
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            st.session_state.cap.set(cv2.CAP_PROP_FPS, 30)
            st.session_state.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
            
        if stop_button and st.session_state.streaming:
            st.session_state.streaming = False
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            status_placeholder.write("Camera stopped")
        
        # Streaming loop
        if st.session_state.streaming and st.session_state.cap is not None:
            status_placeholder.write(f"Streaming... Confidence Threshold: {st.session_state.confidence_threshold}%. Press Stop to end.")
            while st.session_state.streaming:
                ret, frame = st.session_state.cap.read()
                if not ret or frame is None:
                    status_placeholder.error("Failed to capture frame. Ensure the camera is connected.")
                    break
                
                processed_frame, faces_detected = detect_age_and_gender(
                    frame, face_net, age_net, gender_net, confidence_threshold_decimal
                )
                if processed_frame is not None:
                    frame_placeholder.image(
                        processed_frame,
                        channels="RGB",
                        use_container_width=True,
                        caption="No faces detected" if not faces_detected else ""
                    )
                
                # Control frame rate
                await asyncio.sleep(0.033)  # ~30 FPS
                
            # Cleanup
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
                status_placeholder.write("Camera stopped")
                
    except Exception as e:
        logger.error(f"Main app error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        st.session_state.streaming = False

if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())