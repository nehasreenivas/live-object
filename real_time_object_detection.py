# Import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import pyttsx3
import streamlit as st

# Streamlit UI setup
st.title("Real-Time Object Detection")
st.sidebar.header("Model Configuration")

# File uploader for prototxt and model files
prototxt_file = st.sidebar.file_uploader("Upload Prototxt File", type=["txt", "prototxt"])
model_file = st.sidebar.file_uploader("Upload Model File", type=["caffemodel"])

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.1, max_value=1.0, value=0.2, step=0.05
)

# Classes MobileNetSSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Function to load the model and run object detection
def run_object_detection(prototxt_path, model_path, confidence):
    st.write("[INFO] Loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    st.write("[INFO] Starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()

    # Text-to-speech engine setup
    engine = pyttsx3.init()

    # Loop over frames from the video stream
    while True:
        # Grab the frame and resize it
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # Grab frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)

        # Pass the blob through the network to get detections
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
            confidence_score = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence_score > confidence:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box and label
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence_score * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                # Text-to-speech
                engine.say(f"This is a {CLASSES[idx]}")
                engine.runAndWait()

        # Show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # If the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # Update the FPS counter
        fps.update()

    # Stop the timer and display FPS information
    fps.stop()
    st.write("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
    st.write("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

    # Clean up
    cv2.destroyAllWindows()
    vs.stop()

# Streamlit logic for handling file uploads
if prototxt_file and model_file:
    # Save the uploaded files to disk
    with open("uploaded_prototxt.prototxt", "wb") as f:
        f.write(prototxt_file.read())
    with open("uploaded_model.caffemodel", "wb") as f:
        f.write(model_file.read())

    # Run the object detection
    st.success("Model files uploaded successfully. Starting object detection...")
    run_object_detection("uploaded_prototxt.prototxt", "uploaded_model.caffemodel", confidence_threshold)
else:
    st.warning("Please upload both the Prototxt and Model files to proceed.")
