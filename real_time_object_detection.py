import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
import io

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

# Function for text-to-speech using gTTS and returning as byte stream
def speak(text):
    try:
        # Convert text to speech
        tts = gTTS(text=text, lang='en')
        
        # Save it to a BytesIO object
        audio_stream = io.BytesIO()
        tts.save(audio_stream)
        audio_stream.seek(0)  # Rewind the stream to the beginning
        
        return audio_stream
    except Exception as e:
        st.write(f"Error in TTS: {str(e)}")
        return None

# Initialize Streamlit app and camera
st.title('Real-Time Object Detection with Sound')

# Create a video capture object (this will capture an image from the webcam)
image_file = st.camera_input("Capture Image")

if image_file is not None:
    # Read the image as bytes
    img_bytes = image_file.getvalue()
    
    # Convert the image bytes to a NumPy array
    np_arr = np.frombuffer(img_bytes, np.uint8)
    
    # Decode the image to an OpenCV format
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Check if frame was successfully decoded
    if frame is not None:
        # Resize the frame (optional, depending on the size)
        frame = cv2.resize(frame, (400, 400))
        
        # Check the channels (should be 3 channels for color image)
        if frame.shape[2] == 3:
            # Ensure the frame is in the correct color format (BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Initialize blob
            blob = cv2.dnn.blobFromImage(frame_bgr, 0.007843, (400, 400), (127.5, 127.5, 127.5), swapRB=False)
            
            # Pass the blob through the network and obtain the detections
            net.setInput(blob)
            detections = net.forward()
            
            # Display results on Streamlit
            detected_objects = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > confidence_threshold:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    color = COLORS[idx]
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    detected_objects.append(CLASSES[idx])
            
            # Convert the frame to RGB and display it in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, channels="RGB")
            
            # Check if any objects were detected
            if detected_objects:
                detected_text = " and ".join(detected_objects) + " detected"
                st.write(f"Detected Objects: {detected_text}")
                
                # Generate speech for detected objects
                audio_stream = speak(detected_text)
                if audio_stream:
                    st.audio(audio_stream, format="audio/mp3")
                else:
                    st.write("[ERROR] Audio generation failed")
            else:
                st.write("No objects detected")
        else:
            st.write("[ERROR] Frame has invalid number of channels.")
    else:
        st.write("[ERROR] Failed to decode the image")
else:
    st.write("[ERROR] No image file received.")
