import cv2
import imutils
import pyttsx3
import time

# Initialize text-to-speech engine
def init_text_to_speech():
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)  # Use the first voice (can be changed to another voice)
        engine.setProperty('rate', 150)  # Set speech speed
        return engine
    except Exception as e:
        print(f"Error initializing TTS: {e}")
        return None

# Function to start object detection
def run_object_detection(prototxt, model, confidence_threshold=0.5):
    # Initialize camera
    camera = cv2.VideoCapture(0)  # Try changing to 1 or other index if 0 doesn't work

    if not camera.isOpened():
        print("Error: Camera not accessible")
        return

    # Load model
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # Initialize text-to-speech engine
    engine = init_text_to_speech()
    
    # Start processing frames
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Resize the frame to a fixed width
        frame = imutils.resize(frame, width=400)

        # Prepare the frame for object detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123), False, crop=False)
        net.setInput(blob)
        detections = net.forward()

        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                idx = int(detections[0, 0, i, 1])  # Get the index of the detected object
                box = detections[0, 0, i, 3:7] * \
                    np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw bounding box and label on the frame
                label = f"Confidence: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Optionally use TTS to announce detected objects
                if engine:
                    engine.say(f"Object detected with confidence: {confidence:.2f}")
                    engine.runAndWait()

        # Display the frame with detected objects
        cv2.imshow("Object Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close any OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    prototxt = "MobileNetSSD_deploy.prototxt.txt"  # Path to your Prototxt file
    model = "uploaded_model.caffemodel"      # Path to your model file
    run_object_detection(prototxt, model)
