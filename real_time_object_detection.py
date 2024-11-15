import streamlit as st
import cv2
import numpy as np
from imutils.video import VideoStream
from gtts import gTTS
import os
import imutils
import time

# Hardcode the paths to the prototxt file and model file
prototxt_path = "MobileNetSSD_deploy.prototxt.txt"
model_path = "MobileNetSSD_deploy.caffemodel"
confidence_threshold = 0.2

# Initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "mobile"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Function for text-to-speech using gTTS
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("mpg321 output.mp3")  # You can change this command depending on how to play audio on your cloud server

# Initialize Streamlit app and camera
st.title('Real-Time Object Detection with Sound')

# Create a video stream object
video_capture = st.camera_input("Capture Video")

# Check if video input exists
if video_capture:
    # Decode the frame from bytes into an OpenCV image
    np_frame = np.frombuffer(video_capture.read(), np.uint8)
    frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
    
    # Check if frame was decoded successfully
    if frame is None:
        st.write("[ERROR] Failed to decode frame")
    else:
        # Start the video stream
        video_stream = VideoStream(src=0).start()
        time.sleep(2.0)  # Allow the camera to warm up

        # Loop over frames from the video stream
        while True:
            frame = video_stream.read()

            # Check if the frame is None
            if frame is None:
                st.write("[ERROR] Failed to capture frame")
                break  # Exit if no frame is captured

            # Resize the frame to a max width of 400 pixels
            frame = imutils.resize(frame, width=400)

            # Grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5)

            # Pass the blob through the network and obtain the detections and predictions
            net.setInput(blob)
            detections = net.forward()

            # Loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # Extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # Filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                if confidence > confidence_threshold:
                    # Extract the index of the class label from the detections, then compute the (x, y)-coordinates of the bounding box for the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Draw the prediction on the frame
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                    # Read aloud the detected object
                    speak("This is a " + label)

            # Show the output frame
            st.image(frame, channels="BGR")
