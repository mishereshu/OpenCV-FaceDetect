import cv2
import numpy as np

# Load the trained recognizer and known names
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer.yml")
known_names = np.load("known_names.npy", allow_pickle=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to classify a face as "Registered User" or "Intruder"
def classify_face(frame, face_cascade, recognizer):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_region = gray_frame[y:y + h, x:x + w]

        # Resize the face region to a standard size (e.g., 100x100 pixels)
        face_region_resized = cv2.resize(face_region, (100, 100))

        # Use the recognizer to predict the label (id) and confidence level
        label, confidence = recognizer.predict(face_region_resized)

        # Increase the threshold to improve accuracy (lower value = higher confidence)
        # Increase the threshold from 50 to a higher value, like 70 for more accurate detection
        if confidence < 70:  # Only recognize faces with confidence above 70%
            label_text = known_names[label]  # Get the name of the person
            color = (0, 255, 0)  # Green if recognized
        else:
            label_text = "Intruder"
            color = (0, 0, 255)  # Red if not recognized

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame, f"{label_text} - {round(100 - confidence, 2)}%", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )

    return frame

# Main function
def main():
    # Open the laptop's default camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Detect, classify, and label faces with confidence values
        frame_with_labels = classify_face(frame, face_cascade, recognizer)

        # Display the live feed with classification
        cv2.imshow("Live Feed with Classification", frame_with_labels)

        # Close window when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
