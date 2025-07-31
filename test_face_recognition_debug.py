import cv2
import pickle
import face_recognition
import numpy as np
import torch
from facenet_pytorch import MTCNN

# Load the saved student database
db_path = "students.pkl"

try:
    with open(db_path, "rb") as f:
        data = pickle.load(f)
        known_encodings = data["encodings"]
        known_ids = data["ids"]
        known_names = data["names"]
except FileNotFoundError:
    print("❌ ERROR: 'students.pkl' not found. Make sure you enrolled and saved the database.")
    exit()

if not known_encodings:
    print("⚠️ No face encodings found in the database. Please re-enroll students.")
    exit()

print("✅ Enrolled Students:")
for sid, name in zip(known_ids, known_names):
    print(f"  - {sid}: {name}")

# Initialize webcam
cap = cv2.VideoCapture(0)
print("\n📸 Starting webcam. Press Q to quit.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

tolerance = 0.6  # You can raise this to 0.65 or 0.7 if matches fail

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)
    face_locations = [] if boxes is None else [
        (int(y1), int(x2), int(y2), int(x1)) for x1, y1, x2, y2 in boxes
    ]
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for face_encoding, location in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        print("\n🔍 Match Distances:", distances)

        if len(distances) == 0:
            print("⚠️ No known faces to compare against.")
            continue

        min_dist = np.min(distances)
        best_match_index = np.argmin(distances)

        if min_dist < tolerance:
            name = known_names[best_match_index]
            print(f"✅ Face recognized as {name} (Distance: {min_dist:.3f})")
            color = (0, 255, 0)
            label = f"{name} ({min_dist:.2f})"
        else:
            print(f"❌ Face not recognized. Closest distance: {min_dist:.3f}")
            color = (0, 0, 255)
            label = f"Unknown ({min_dist:.2f})"

        # Draw the bounding box and label
        top, right, bottom, left = location
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Debug Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
