import cv2
import numpy as np
import os
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize MTCNN (for face detection) and InceptionResnetV1 (for face embeddings)
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Path to your known faces folder
KNOWN_FACES_DIR = "C:/Users/mithi/Desktop/intrusion detection/known_faces"

# Initialize lists for known face embeddings and names
known_faces = []
known_names = []

# Process each file in the known faces folder
for filename in os.listdir(KNOWN_FACES_DIR):
    filepath = os.path.join(KNOWN_FACES_DIR, filename)

    # Load the image
    image = cv2.imread(filepath)

    # Detect faces using MTCNN
    faces, probs = mtcnn.detect(image)
    if faces is not None:
        for face in faces:
            x1, y1, x2, y2 = [int(i) for i in face]
            face_region = image[y1:y2, x1:x2]

            # Get the embedding of the face
            face_embedding = model(face_region)
            if face_embedding is not None:
                known_faces.append(face_embedding[0].detach().numpy())  # Save the embedding
                known_names.append(os.path.splitext(filename)[0])  # Use filename as the name

# Save the embeddings and names for later use
np.save("known_faces_cnn.npy", known_faces)
np.save("known_names.npy", known_names)
print("Face embeddings and names saved successfully!")
