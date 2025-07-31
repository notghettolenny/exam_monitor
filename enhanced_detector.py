import cv2
import numpy as np
import mediapipe as mp
import math
from collections import deque
import time

class EnhancedDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        # Initialize face mesh for talking detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize hands for hand movement detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize pose for body movement detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize phone detection (using color and shape detection)
        self.phone_cascade = None
        try:
            # You can download a phone cascade classifier or use object detection
            # For now, we'll use color and shape detection
            pass
        except:
            pass
        
        # Buffers for temporal analysis
        self.mouth_aspect_ratios = deque(maxlen=10)
        self.hand_positions = deque(maxlen=10)
        self.head_positions = deque(maxlen=10)
        
        # Thresholds
        self.MOUTH_OPEN_THRESHOLD = 0.02
        self.HAND_MOVEMENT_THRESHOLD = 0.15
        self.PHONE_COLOR_RANGES = [
            # Common phone colors in HSV
            ([0, 0, 0], [180, 255, 50]),      # Black
            ([0, 0, 200], [180, 30, 255]),    # White/Silver
            ([100, 50, 50], [130, 255, 255]) # Blue
        ]
        
    def detect_phone_in_frame(self, frame):
        """Detect phone using color detection and rectangular shape analysis"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Skip detection in extremely dark frames to reduce false positives
            if hsv[..., 2].mean() < 40:
                return False

            # Create masks for common phone colors
            phone_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

            for lower, upper in self.PHONE_COLOR_RANGES:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)
                phone_mask = cv2.bitwise_or(phone_mask, mask)
            
            # Morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            phone_mask = cv2.morphologyEx(phone_mask, cv2.MORPH_CLOSE, kernel)
            phone_mask = cv2.morphologyEx(phone_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(phone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # Filter by area (phone should be reasonably sized)
                if 1000 < area < 50000:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)

                    # Skip very dark candidate regions
                    region_v = hsv[y:y + h, x:x + w, 2]
                    if region_v.size > 0 and region_v.mean() < 60:
                        continue

                    # Additional rectangularity checks to reduce false positives
                    rect_area = float(w * h)
                    if rect_area <= 0:
                        continue
                    extent = area / rect_area
                    if extent < 0.6:
                        continue

                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                    if len(approx) < 4 or len(approx) > 6:
                        continue

                    # Check aspect ratio (phones are typically rectangular)
                    aspect_ratio = w / h
                    if 0.3 < aspect_ratio < 0.8:  # Portrait phone
                        # Check if it's in hand area (upper part of frame)
                        if y < frame.shape[0] * 0.7:  # Upper 70% of frame
                            return True
                    elif 1.0 < aspect_ratio < 3.5:  # Landscape phone
                        if y < frame.shape[0] * 0.7:
                            return True
            
            return False
            
        except Exception as e:
            print(f"Error in phone detection: {e}")
            return False
    
    def detect_speaking_in_frame(self, frame):
        """Detect speaking using mouth aspect ratio analysis"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get mouth landmarks
                    mouth_landmarks = []
                    
                    # Upper lip: 13, 14, 15, 16, 17, 18
                    # Lower lip: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
                    # We'll use specific points for mouth opening calculation
                    
                    # Key mouth points
                    mouth_points = [13, 14, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                    
                    # Get mouth corner and center points
                    mouth_left = face_landmarks.landmark[61]   # Left corner
                    mouth_right = face_landmarks.landmark[291] # Right corner
                    mouth_top = face_landmarks.landmark[13]    # Top center
                    mouth_bottom = face_landmarks.landmark[14] # Bottom center
                    
                    # Calculate mouth aspect ratio
                    mouth_width = abs(mouth_right.x - mouth_left.x)
                    mouth_height = abs(mouth_top.y - mouth_bottom.y)
                    
                    if mouth_width > 0:
                        mouth_aspect_ratio = mouth_height / mouth_width
                        self.mouth_aspect_ratios.append(mouth_aspect_ratio)
                        
                        # Check if mouth is open (speaking)
                        if len(self.mouth_aspect_ratios) >= 5:
                            avg_mar = sum(self.mouth_aspect_ratios) / len(self.mouth_aspect_ratios)
                            if avg_mar > self.MOUTH_OPEN_THRESHOLD:
                                return True
            
            return False
            
        except Exception as e:
            print(f"Error in speaking detection: {e}")
            return False
    
    def detect_hand_movements(self, frame):
        """Detect suspicious hand movements"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                current_hand_positions = []
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get wrist position as reference
                    wrist = hand_landmarks.landmark[0]
                    current_hand_positions.append((wrist.x, wrist.y))
                
                self.hand_positions.append(current_hand_positions)
                
                # Analyze hand movement if we have enough data
                if len(self.hand_positions) >= 5:
                    # Calculate movement between frames
                    movements = []
                    for i in range(1, len(self.hand_positions)):
                        prev_positions = self.hand_positions[i-1]
                        curr_positions = self.hand_positions[i]
                        
                        # Calculate movement for each hand
                        for j in range(min(len(prev_positions), len(curr_positions))):
                            prev_x, prev_y = prev_positions[j]
                            curr_x, curr_y = curr_positions[j]
                            
                            movement = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                            movements.append(movement)
                    
                    # Check if movement exceeds threshold
                    if movements:
                        avg_movement = sum(movements) / len(movements)
                        if avg_movement > self.HAND_MOVEMENT_THRESHOLD:
                            return True
            
            return False
            
        except Exception as e:
            print(f"Error in hand movement detection: {e}")
            return False
    
    def detect_head_movements(self, frame):
        """Detect suspicious head movements (looking around)"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get nose tip as head position reference
                    nose_tip = face_landmarks.landmark[1]
                    current_head_pos = (nose_tip.x, nose_tip.y)
                    
                    self.head_positions.append(current_head_pos)
                    
                    # Analyze head movement
                    if len(self.head_positions) >= 8:
                        # Calculate head movement variance
                        x_positions = [pos[0] for pos in self.head_positions]
                        y_positions = [pos[1] for pos in self.head_positions]
                        
                        x_variance = np.var(x_positions)
                        y_variance = np.var(y_positions)
                        
                        # If head is moving too much (looking around)
                        if x_variance > 0.001 or y_variance > 0.001:
                            return True
            
            return False
            
        except Exception as e:
            print(f"Error in head movement detection: {e}")
            return False
    
    def detect_object_near_ear(self, frame):
        """Detect objects near ear area (phone calls)"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get ear area landmarks
                    left_ear = face_landmarks.landmark[234]   # Left ear
                    right_ear = face_landmarks.landmark[454]  # Right ear
                    
                    # Convert to pixel coordinates
                    h, w = frame.shape[:2]
                    left_ear_px = (int(left_ear.x * w), int(left_ear.y * h))
                    right_ear_px = (int(right_ear.x * w), int(right_ear.y * h))
                    
                    # Check for objects near ear areas
                    ear_regions = [
                        (max(0, left_ear_px[0] - 50), max(0, left_ear_px[1] - 30),
                         min(w, left_ear_px[0] + 20), min(h, left_ear_px[1] + 30)),
                        (max(0, right_ear_px[0] - 20), max(0, right_ear_px[1] - 30),
                         min(w, right_ear_px[0] + 50), min(h, right_ear_px[1] + 30))
                    ]
                    
                    for x1, y1, x2, y2 in ear_regions:
                        ear_region = frame[y1:y2, x1:x2]
                        if ear_region.size > 0:
                            # Simple object detection near ear
                            if self.detect_phone_in_frame(ear_region):
                                return True
            
            return False
            
        except Exception as e:
            print(f"Error in object near ear detection: {e}")
            return False

# Usage example for integrating with your app.py
def create_enhanced_detector():
    """Create enhanced detector instance"""
    return EnhancedDetector()

# Replace the placeholder functions in your app.py with these:
def detect_phone_in_frame_enhanced(frame, detector):
    """Enhanced phone detection"""
    return detector.detect_phone_in_frame(frame) or detector.detect_object_near_ear(frame)

def detect_speaking_in_frame_enhanced(frame, detector):
    """Enhanced speaking detection"""
    return detector.detect_speaking_in_frame(frame)

def detect_hand_movements_enhanced(frame, detector):
    """Enhanced hand movement detection"""
    return detector.detect_hand_movements(frame)

def detect_head_movements_enhanced(frame, detector):
    """Enhanced head movement detection"""
    return detector.detect_head_movements(frame)