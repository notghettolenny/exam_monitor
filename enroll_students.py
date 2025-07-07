from exam_monitor_system import ExamMonitorConfig, FaceRecognitionModule
import cv2
import os

# Set up
config = ExamMonitorConfig()
face_recognition = FaceRecognitionModule(config)

# Load a student photo
student_id = "001"
student_name = "Test Student"
image_path = f"student_photos/{student_id}.jpg"

# Make sure the photo exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Missing photo: {image_path}")

# Read and enroll
image = cv2.imread(image_path)
success = face_recognition.enroll_student(student_id, student_name, image)

if success:
    print(f"Enrolled {student_name}")
    face_recognition.save_student_database("students.pkl")
else:
    print("Enrollment failed")
