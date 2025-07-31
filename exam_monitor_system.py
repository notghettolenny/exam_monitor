import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import torch
import json
import os
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
import threading
import queue
import pickle

# Configuration and data structures
@dataclass
class Alert:
    timestamp: float
    alert_type: str
    description: str
    confidence: float
    student_id: Optional[str] = None
    frame_path: Optional[str] = None

@dataclass
class StudentData:
    student_id: str
    name: str
    face_encoding: np.ndarray
    seat_position: Optional[Tuple[int, int]] = None

class ExamMonitorConfig:
    """Configuration class for tuning detection thresholds"""
    
    def __init__(self):
        # Face recognition thresholds
        self.face_recognition_tolerance = 0.6
        self.unknown_face_threshold = 0.7
        
        # Gaze tracking thresholds
        self.max_gaze_deviation = 45.0  # degrees
        self.gaze_alert_duration = 3.0  # seconds
        self.head_turn_threshold = 30.0  # degrees
        
        # Pose estimation thresholds
        self.leaning_threshold = 0.3  # relative position change
        self.gesture_repeat_threshold = 3  # repeated gestures
        self.pose_alert_duration = 2.0  # seconds
        
        # Object detection thresholds
        self.phone_confidence_threshold = 0.5
        self.book_confidence_threshold = 0.4
        self.object_detection_frequency = 5  # every N frames
        
        # Alert system
        self.alert_cooldown = 10.0  # seconds between similar alerts
        self.video_clip_duration = 5.0  # seconds
        
        # Performance settings
        self.frame_skip = 2  # process every N frames
        self.max_faces_per_frame = 10
        # Scale factor for face recognition processing (0.25 reduces resolution
        # to 25% for faster computation)
        self.face_recognition_scale = 0.25

class FaceRecognitionModule:
    """Handles student attendance and face verification"""
    
    def __init__(self, config: ExamMonitorConfig):
        self.config = config
        self.known_students: Dict[str, StudentData] = {}
        self.attendance_log: Dict[str, float] = {}
        self.unknown_faces_buffer = deque(maxlen=100)
        
    def load_student_database(self, database_path: str):
        """Load pre-enrolled student face encodings"""
        try:
            with open(database_path, 'rb') as f:
                student_data = pickle.load(f)
                for student in student_data:
                    self.known_students[student.student_id] = student
            print(f"Loaded {len(self.known_students)} students from database")
        except FileNotFoundError:
            print(f"Student database not found at {database_path}")
            
    def save_student_database(self, database_path: str):
        """Save student database with face encodings"""
        student_list = list(self.known_students.values())
        with open(database_path, 'wb') as f:
            pickle.dump(student_list, f)
            
    def enroll_student(self, student_id: str, name: str, face_image: np.ndarray):
        """Enroll a new student with face encoding"""
        face_encodings = face_recognition.face_encodings(face_image)
        if len(face_encodings) > 0:
            student = StudentData(student_id, name, face_encodings[0])
            self.known_students[student_id] = student
            return True
        return False
        
    def recognize_faces(self, frame: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Recognize faces in frame and return student IDs with bounding boxes"""
        # Convert to RGB and optionally scale for faster processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.config.face_recognition_scale != 1.0:
            small_frame = cv2.resize(
                rgb_frame,
                (0, 0),
                fx=self.config.face_recognition_scale,
                fy=self.config.face_recognition_scale,
            )
        else:
            small_frame = rgb_frame

        # Find face locations and encodings on the scaled frame
        small_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, small_locations)

        # Scale face locations back to original frame size
        face_locations = [
            (
                int(top / self.config.face_recognition_scale),
                int(right / self.config.face_recognition_scale),
                int(bottom / self.config.face_recognition_scale),
                int(left / self.config.face_recognition_scale),
            )
            for top, right, bottom, left in small_locations
        ]
        
        results = []
        
        known_ids = list(self.known_students.keys())
        known_encodings = [student.face_encoding for student in self.known_students.values()]

        for face_encoding, face_location in zip(face_encodings, face_locations):
            if known_encodings:
                # Compare with known students
                matches = face_recognition.compare_faces(
                    known_encodings,
                    face_encoding,
                    tolerance=self.config.face_recognition_tolerance,
                )

                face_distances = face_recognition.face_distance(
                    known_encodings, face_encoding
                )
                best_match_index = int(np.argmin(face_distances))
            else:
                matches = []
                face_distances = []
                best_match_index = 0
            student_id = "unknown"
            confidence = 0.0
            
            if (
                matches
                and matches[best_match_index]
                and face_distances[best_match_index] < self.config.unknown_face_threshold
            ):
                student_id = known_ids[best_match_index]
                confidence = 1.0 - face_distances[best_match_index]
                
                # Log attendance
                if student_id not in self.attendance_log:
                    self.attendance_log[student_id] = time.time()
            else:
                # Store unknown face for review
                self.unknown_faces_buffer.append((time.time(), face_encoding, face_location))
                
            results.append((student_id, confidence, face_location))
            
        return results

class GazeTrackingModule:
    """Handles gaze and head pose tracking using MediaPipe"""
    
    def __init__(self, config: ExamMonitorConfig):
        self.config = config
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=config.max_faces_per_frame,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.gaze_history = defaultdict(lambda: deque(maxlen=30))
        self.head_pose_history = defaultdict(lambda: deque(maxlen=30))
        
    def calculate_gaze_direction(self, landmarks, frame_shape) -> Tuple[float, float]:
        """Calculate gaze direction from facial landmarks"""
        # Key landmarks for gaze estimation
        left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Get eye center points
        left_eye_center = np.mean([(landmarks[i].x, landmarks[i].y) for i in left_eye_landmarks], axis=0)
        right_eye_center = np.mean([(landmarks[i].x, landmarks[i].y) for i in right_eye_landmarks], axis=0)
        
        # Calculate gaze direction (simplified)
        eye_center = (left_eye_center + right_eye_center) / 2
        nose_tip = (landmarks[1].x, landmarks[1].y)
        
        # Calculate horizontal and vertical gaze angles
        horizontal_angle = np.arctan2(eye_center[0] - nose_tip[0], 0.1) * 180 / np.pi
        vertical_angle = np.arctan2(eye_center[1] - nose_tip[1], 0.1) * 180 / np.pi
        
        return horizontal_angle, vertical_angle
        
    def calculate_head_pose(self, landmarks, frame_shape) -> Tuple[float, float, float]:
        """Calculate head pose angles"""
        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # 2D image points
        image_points = np.array([
            (landmarks[1].x * frame_shape[1], landmarks[1].y * frame_shape[0]),    # Nose tip
            (landmarks[175].x * frame_shape[1], landmarks[175].y * frame_shape[0]), # Chin
            (landmarks[33].x * frame_shape[1], landmarks[33].y * frame_shape[0]),   # Left eye
            (landmarks[263].x * frame_shape[1], landmarks[263].y * frame_shape[0]), # Right eye
            (landmarks[61].x * frame_shape[1], landmarks[61].y * frame_shape[0]),   # Left mouth
            (landmarks[291].x * frame_shape[1], landmarks[291].y * frame_shape[0])  # Right mouth
        ], dtype="double")
        
        # Camera parameters (approximate)
        focal_length = frame_shape[1]
        center = (frame_shape[1] / 2, frame_shape[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )
        
        # Convert rotation vector to angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose_angles = cv2.RQDecomp3x3(rotation_matrix)[0]
        
        return pose_angles[0], pose_angles[1], pose_angles[2]  # pitch, yaw, roll
        
    def process_frame(self, frame: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> List[Alert]:
        """Process frame for gaze and head pose tracking"""
        alerts = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for i, landmarks in enumerate(results.multi_face_landmarks):
                # Calculate gaze direction
                horizontal_gaze, vertical_gaze = self.calculate_gaze_direction(landmarks.landmark, frame.shape)
                
                # Calculate head pose
                pitch, yaw, roll = self.calculate_head_pose(landmarks.landmark, frame.shape)
                
                # Store in history
                face_id = f"face_{i}"
                self.gaze_history[face_id].append((time.time(), horizontal_gaze, vertical_gaze))
                self.head_pose_history[face_id].append((time.time(), pitch, yaw, roll))
                
                # Check for suspicious gaze patterns
                if abs(horizontal_gaze) > self.config.max_gaze_deviation:
                    if len(self.gaze_history[face_id]) > 5:
                        recent_gazes = list(self.gaze_history[face_id])[-5:]
                        if all(abs(g[1]) > self.config.max_gaze_deviation for g in recent_gazes):
                            alerts.append(Alert(
                                timestamp=time.time(),
                                alert_type="gaze_deviation",
                                description=f"Excessive gaze deviation: {horizontal_gaze:.1f}°",
                                confidence=min(abs(horizontal_gaze) / 90.0, 1.0)
                            ))
                
                # Check for excessive head turning
                if abs(yaw) > self.config.head_turn_threshold:
                    alerts.append(Alert(
                        timestamp=time.time(),
                        alert_type="head_turn",
                        description=f"Excessive head turning: {yaw:.1f}°",
                        confidence=min(abs(yaw) / 90.0, 1.0)
                    ))
                    
        return alerts

class PoseEstimationModule:
    """Handles pose estimation using MediaPipe"""
    
    def __init__(self, config: ExamMonitorConfig):
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_history = defaultdict(lambda: deque(maxlen=30))
        self.gesture_patterns = defaultdict(list)
        
    def detect_leaning(self, landmarks) -> Tuple[bool, float]:
        """Detect if person is leaning excessively"""
        # Get shoulder and hip landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate torso angle
        shoulder_center = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
        hip_center = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
        
        # Calculate lean angle
        lean_angle = np.arctan2(shoulder_center[0] - hip_center[0], hip_center[1] - shoulder_center[1]) * 180 / np.pi
        
        is_leaning = abs(lean_angle) > self.config.leaning_threshold * 90
        return is_leaning, abs(lean_angle)
        
    def detect_suspicious_gestures(self, landmarks, person_id: str) -> List[str]:
        """Detect suspicious hand gestures and repetitive movements"""
        suspicious_gestures = []
        
        # Get hand landmarks
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # Store current hand positions
        current_time = time.time()
        hand_positions = {
            'left': (left_wrist.x, left_wrist.y),
            'right': (right_wrist.x, right_wrist.y)
        }
        
        # Add to history
        self.pose_history[person_id].append((current_time, hand_positions))
        
        # Check for repetitive movements
        if len(self.pose_history[person_id]) >= 10:
            recent_positions = list(self.pose_history[person_id])[-10:]
            
            # Detect repetitive hand movements
            for hand in ['left', 'right']:
                positions = [pos[1][hand] for pos in recent_positions]
                
                # Calculate movement variance
                x_positions = [pos[0] for pos in positions]
                y_positions = [pos[1] for pos in positions]
                
                if len(set(x_positions)) < 3 and len(set(y_positions)) < 3:
                    # Very little movement - potentially writing/using device
                    suspicious_gestures.append(f"{hand}_hand_activity")
                    
        return suspicious_gestures
        
    def process_frame(self, frame: np.ndarray) -> List[Alert]:
        """Process frame for pose estimation"""
        alerts = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Check for leaning
            is_leaning, lean_angle = self.detect_leaning(landmarks)
            if is_leaning:
                alerts.append(Alert(
                    timestamp=time.time(),
                    alert_type="leaning",
                    description=f"Student leaning excessively: {lean_angle:.1f}°",
                    confidence=min(lean_angle / 90.0, 1.0)
                ))
            
            # Check for suspicious gestures
            person_id = "person_0"  # Simplified for single person tracking
            suspicious_gestures = self.detect_suspicious_gestures(landmarks, person_id)
            
            for gesture in suspicious_gestures:
                alerts.append(Alert(
                    timestamp=time.time(),
                    alert_type="suspicious_gesture",
                    description=f"Suspicious hand activity detected: {gesture}",
                    confidence=0.7
                ))
                
        return alerts

class ObjectDetectionModule:
    """Handles object detection using YOLOv5"""
    
    def __init__(self, config: ExamMonitorConfig):
        self.config = config
        self.model = None
        self.prohibited_objects = ['cell phone', 'book', 'laptop', 'tablet']
        self.detection_history = defaultdict(lambda: deque(maxlen=10))
        self.frame_counter = 0
        
    def load_model(self, model_path: str = 'yolov5s'):
        """Load YOLOv5 model"""
        try:
            self.model = torch.hub.load('ultralytics/yolov5', model_path, pretrained=True)
            print(f"YOLOv5 model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLOv5 model: {e}")
            
    def process_frame(self, frame: np.ndarray) -> List[Alert]:
        """Process frame for object detection"""
        alerts = []
        self.frame_counter += 1
        
        # Skip frames for performance
        if self.frame_counter % self.config.object_detection_frequency != 0:
            return alerts
            
        if self.model is None:
            return alerts
            
        # Run inference
        results = self.model(frame)
        
        # Process detections
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            class_name = self.model.names[int(cls)]
            
            # Check if object is prohibited
            if any(prohibited in class_name.lower() for prohibited in self.prohibited_objects):
                if conf > self.config.phone_confidence_threshold:
                    alerts.append(Alert(
                        timestamp=time.time(),
                        alert_type="prohibited_object",
                        description=f"Prohibited object detected: {class_name}",
                        confidence=conf
                    ))
                    
                    # Store detection history
                    self.detection_history[class_name].append((time.time(), conf, box))
                    
        return alerts

class VideoClipManager:
    """Manages video clip saving for alerts"""
    
    def __init__(self, config: ExamMonitorConfig, output_dir: str = "alert_clips"):
        self.config = config
        self.output_dir = output_dir
        self.frame_buffer = deque(maxlen=150)  # ~5 seconds at 30fps
        self.recording_clips = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
    def add_frame(self, frame: np.ndarray, timestamp: float):
        """Add frame to buffer"""
        self.frame_buffer.append((timestamp, frame.copy()))
        
    def save_clip_for_alert(self, alert: Alert) -> str:
        """Save video clip for alert"""
        if len(self.frame_buffer) == 0:
            return ""
            
        # Create filename
        timestamp_str = datetime.fromtimestamp(alert.timestamp).strftime("%Y%m%d_%H%M%S")
        filename = f"{alert.alert_type}_{timestamp_str}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        # Get frames around alert time
        alert_frames = []
        for timestamp, frame in self.frame_buffer:
            if abs(timestamp - alert.timestamp) <= self.config.video_clip_duration / 2:
                alert_frames.append(frame)
                
        if len(alert_frames) == 0:
            return ""
            
        # Save video
        height, width = alert_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, 30.0, (width, height))
        
        for frame in alert_frames:
            out.write(frame)
            
        out.release()
        return filepath

class AlertManager:
    """Manages alerts and prevents spam"""
    
    def __init__(self, config: ExamMonitorConfig):
        self.config = config
        self.recent_alerts = defaultdict(lambda: deque(maxlen=10))
        self.alert_queue = queue.Queue()
        self.alert_log = []
        
    def add_alert(self, alert: Alert) -> bool:
        """Add alert if not in cooldown period"""
        alert_key = f"{alert.alert_type}_{alert.student_id}"
        
        # Check cooldown
        if self.recent_alerts[alert_key]:
            last_alert_time = self.recent_alerts[alert_key][-1]
            if time.time() - last_alert_time < self.config.alert_cooldown:
                return False
                
        # Add alert
        self.recent_alerts[alert_key].append(time.time())
        self.alert_queue.put(alert)
        self.alert_log.append(alert)
        
        return True
        
    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """Get recent alerts"""
        return self.alert_log[-limit:]
        
    def save_alert_log(self, filepath: str):
        """Save alert log to file"""
        with open(filepath, 'w') as f:
            for alert in self.alert_log:
                json.dump(asdict(alert), f)
                f.write('\n')

class ExamMonitoringSystem:
    """Main exam monitoring system"""
    
    def __init__(self, config: ExamMonitorConfig):
        self.config = config
        self.face_recognition = FaceRecognitionModule(config)
        self.gaze_tracking = GazeTrackingModule(config)
        self.pose_estimation = PoseEstimationModule(config)
        self.object_detection = ObjectDetectionModule(config)
        self.video_clip_manager = VideoClipManager(config)
        self.alert_manager = AlertManager(config)
        
        self.is_running = False
        self.frame_count = 0
        
    def setup(self, student_database_path: str = "students.pkl", yolo_model: str = "yolov5s"):
        """Setup the monitoring system"""
        print("Setting up exam monitoring system...")
        
        # Load student database
        self.face_recognition.load_student_database(student_database_path)
        
        # Load YOLO model
        self.object_detection.load_model(yolo_model)
        
        print("Setup complete!")
        
    def process_frame(self, frame: np.ndarray) -> List[Alert]:
        """Process a single frame and return alerts"""
        all_alerts = []
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_count % self.config.frame_skip != 0:
            return all_alerts
            
        # Add frame to video buffer
        self.video_clip_manager.add_frame(frame, time.time())
        
        # Face recognition
        face_results = self.face_recognition.recognize_faces(frame)
        face_locations = [result[2] for result in face_results]
        
        # Check for unknown faces
        for student_id, confidence, location in face_results:
            if student_id == "unknown":
                alert = Alert(
                    timestamp=time.time(),
                    alert_type="unknown_face",
                    description="Unknown person detected in exam room",
                    confidence=confidence
                )
                all_alerts.append(alert)
        
        # Gaze tracking
        gaze_alerts = self.gaze_tracking.process_frame(frame, face_locations)
        all_alerts.extend(gaze_alerts)
        
        # Pose estimation
        pose_alerts = self.pose_estimation.process_frame(frame)
        all_alerts.extend(pose_alerts)
        
        # Object detection
        object_alerts = self.object_detection.process_frame(frame)
        all_alerts.extend(object_alerts)
        
        # Process alerts
        processed_alerts = []
        for alert in all_alerts:
            if self.alert_manager.add_alert(alert):
                # Save video clip for alert
                clip_path = self.video_clip_manager.save_clip_for_alert(alert)
                alert.frame_path = clip_path
                processed_alerts.append(alert)
                
        return processed_alerts
        
    def start_monitoring(self, video_source: int = 0):
        """Start monitoring from video source"""
        self.is_running = True
        cap = cv2.VideoCapture(video_source)
        
        print("Starting exam monitoring...")
        print("Press 'q' to quit")
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            alerts = self.process_frame(frame)
            
            # Display alerts
            for alert in alerts:
                print(f"ALERT: {alert.alert_type} - {alert.description} (Confidence: {alert.confidence:.2f})")
                
            # Display frame with annotations
            self.draw_annotations(frame)
            cv2.imshow('Exam Monitoring', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                
        cap.release()
        cv2.destroyAllWindows()
        
        # Save final logs
        self.save_session_data()
        
    def draw_annotations(self, frame: np.ndarray):
        """Draw annotations on frame"""
        # Draw face recognition results
        face_results = self.face_recognition.recognize_faces(frame)
        for student_id, confidence, (top, right, bottom, left) in face_results:
            color = (0, 255, 0) if student_id != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            label = f"{student_id} ({confidence:.2f})"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # Draw alert count
        recent_alerts = self.alert_manager.get_recent_alerts(5)
        for i, alert in enumerate(recent_alerts):
            cv2.putText(frame, f"{alert.alert_type}: {alert.description}", 
                       (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
    def save_session_data(self):
        """Save session data and logs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save alert log
        self.alert_manager.save_alert_log(f"alert_log_{timestamp}.json")
        
        # Save attendance
        attendance_data = {
            'session_timestamp': timestamp,
            'attendance': self.face_recognition.attendance_log,
            'total_alerts': len(self.alert_manager.alert_log)
        }
        
        with open(f"attendance_{timestamp}.json", 'w') as f:
            json.dump(attendance_data, f, indent=2)
            
        print(f"Session data saved with timestamp: {timestamp}")

def create_sample_student_database():
    """Create a sample student database for testing"""
    config = ExamMonitorConfig()
    face_recognition_module = FaceRecognitionModule(config)
    
    # This would typically be done with actual student photos
    # For demo purposes, we'll create placeholder data
    print("To create a student database, you need to:")
    print("1. Collect student photos")
    print("2. Use face_recognition_module.enroll_student() for each student")
    print("3. Save the database with face_recognition_module.save_student_database()")
    
    return face_recognition_module

if __name__ == "__main__":
    # Example usage
    config = ExamMonitorConfig()
    
    # Adjust thresholds as needed
    config.face_recognition_tolerance = 0.5
    config.max_gaze_deviation = 30.0
    config.alert_cooldown = 5.0
    
    # Create monitoring system
    monitor = ExamMonitoringSystem(config)
    
    # Setup (load student database and models)
    monitor.setup()
    
    # Start monitoring
    # monitor.start_monitoring(video_source=0)  # Use 0 for webcam, or path to video file
    
    print("Exam monitoring system initialized!")
    print("To start monitoring, call monitor.start_monitoring()")
    print("To enroll students, use create_sample_student_database()")