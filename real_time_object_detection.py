import cv2
import imutils
import pyttsx3
import time
import streamlit as st
from imutils.video import FPS

def run_object_detection(prototxt_path, model_path, confidence_threshold):
    # Initialize the camera
    camera = cv2.VideoCapture(0)
    
    # Check if the camera is opened correctly
    if not camera.isOpened():
        print("Error: Camera is not accessible.")
        return
    
    # Load the pre-trained model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()
    
    # Start FPS counter
    fps = FPS().start()

    # Loop over frames from the video stream
    while True:
        ret, frame = camera.read()
        
        # Check if the frame was successfully captured
        if not ret or frame is None:
            print("Error: Failed to capture frame.")
            break
        
        # Resize the frame to a width of 400 pixels
        frame = imutils.resize(frame, width=400)
        
        # Get the frame dimensions and create a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Set the blob as input to the network
        net.setInput(blob)
        
        # Perform forward pass
        detections = net.forward()

        # Loop over detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                # Get the bounding box coordinates for the detected object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Draw the bounding box around the object
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # Optionally, speak the detected object's label (if available)
                label = "Object detected"
                engine.say(label)
                engine.runAndWait()

        # Display the frame with the bounding box
        cv2.imshow("Object Detection", frame)
        
        # Check for a key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Update the FPS counter
        fps.update()

    # Stop the FPS counter and display the FPS
    fps.stop()
    print(f"[INFO] FPS: {fps.fps():.2f}")

    # Release the camera and close all windows
    camera.release()
    cv2.destroyAllWindows()

# Example usage (make sure you pass correct paths to the model files)
confidence_threshold = 0.5  # Adjust confidence threshold as needed
run_object_detection("uploaded_prototxt.prototxt", "uploaded_model.caffemodel", confidence_threshold)
