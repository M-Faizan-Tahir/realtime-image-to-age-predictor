import cv2
import streamlit as st
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
st.title("My app")
frame_placeholder = st.empty()

stopbtn = st.button("STOP")


while cap.isOpened() and not stopbtn:

    ret, frame =cap.read()
    if not ret:
        st.write("vedio ended")
        break
    frame = cv2.cvtColor(frame, channels="RGB")
    frame_placeholder.image(frame, channels="RGB")

    if cv3.waitKey(1)& 0xFF == ord("q") or stopbtn:
        break

cap.release()
cv2.destroyAllWindows()