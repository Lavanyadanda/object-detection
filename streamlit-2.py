# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import subprocess
# import os

# # =========================
# # PAGE SETTINGS
# # =========================
# st.set_page_config(page_title="VisionGuard CCTV", layout="wide")
# st.title("🚨 VisionGuard CCTV Dashboard")
# st.write("Select an option from the sidebar to start monitoring.")

# # =========================
# # SIDEBAR MENU
# # =========================
# st.sidebar.header("Navigation")
# option = st.sidebar.radio("Choose an action:", ["Upload Image/Video", "Live Monitoring (Webcam)"])

# # =========================
# # UPLOAD IMAGE/VIDEO MODE
# # =========================
# if option == "Upload Image/Video":
#     st.subheader("📁 Upload an Image or Video")
#     uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi"])

#     if uploaded_file is not None:
#         file_extension = uploaded_file.name.split(".")[-1].lower()

#         # Save to temp file
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_file.read())

#         if file_extension in ["jpg", "jpeg", "png"]:
#             st.image(tfile.name, caption="Uploaded Image", use_column_width=True)

#             # Here you can run your detection script
#             # Example: subprocess.run(["python", "detect.py", "--source", tfile.name])
#             st.success("✅ Image uploaded successfully! Run detection from detect.py.")
        
#         elif file_extension in ["mp4", "avi"]:
#             st.video(tfile.name)
#             st.success("✅ Video uploaded successfully! Run detection from detect.py.")

# # =========================
# # LIVE MONITORING MODE
# # =========================
# elif option == "Live Monitoring (Webcam)":
#     st.subheader("🎥 Live Monitoring via Webcam")
#     st.info("Click the button below to start detection using your webcam.")

#     if st.button("▶ Start Live Monitoring"):
#         # Run detection script (detect.py should handle webcam)
#         subprocess.Popen(["python", "detect.py", "--source", "0"])
#         st.warning("Webcam detection started. Press 'Q' in the webcam window to stop.")
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from ultralytics import YOLO
import subprocess
import os

# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(page_title="VisionGuard CCTV", layout="wide")
st.title("🚨 VisionGuard CCTV Dashboard")
st.write("Select an option from the sidebar to start monitoring.")

# Load YOLO model
model = YOLO("yolov8n.pt")

# =========================
# SIDEBAR MENU
# =========================
st.sidebar.header("Navigation")
option = st.sidebar.radio("Choose an action:", ["Upload Image/Video", "Live Monitoring (Webcam)"])

# =========================
# UPLOAD IMAGE/VIDEO MODE
# =========================
if option == "Upload Image/Video":
    st.subheader("📁 Upload an Image or Video")
    uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        # Save to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
        tfile.write(uploaded_file.read())
        temp_path = tfile.name

        # ================= IMAGE =================
        if file_extension in ["jpg", "jpeg", "png"]:
            image = Image.open(temp_path)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("🔍 Run Detection on Image"):
                results = model.predict(source=temp_path, save=False, conf=0.25)
                for r in results:
                    img_array = r.plot()  # NumPy array (BGR)
                    st.image(img_array[:, :, ::-1], caption="Detection Result", use_container_width=True)

        # ================= VIDEO =================
        elif file_extension in ["mp4", "avi"]:
            st.video(temp_path)

            if st.button("🔍 Run Detection on Video"):
                stframe = st.empty()  # placeholder for video frames
                cap = cv2.VideoCapture(temp_path)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(source=frame, save=False, conf=0.25, verbose=False)
                    annotated_frame = results[0].plot()

                    # Display annotated frame in Streamlit
                    stframe.image(annotated_frame[:, :, ::-1], channels="RGB")

                cap.release()
                st.success("✅ Video processing finished!")

# =========================
# LIVE MONITORING MODE
# =========================
elif option == "Live Monitoring (Webcam)":
    st.subheader("🎥 Live Monitoring via Webcam")
    st.info("Click the button below to start detection using your webcam.")

    if st.button("▶ Start Live Monitoring"):
        # Save webcam detections (continuous stream) to runs/detect folder
        subprocess.Popen(["python", "detect.py", "--source", "0"])
        st.warning("Webcam detection started. Press 'Q' in the webcam window to stop.")
