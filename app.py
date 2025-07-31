from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit
import cv2
import base64
import threading
import time
import json
from datetime import datetime
import os
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import numpy as np
import face_recognition
from PIL import Image
import io
import mediapipe as mp
import math
from collections import deque

# Import the enhanced detector
from enhanced_detector import EnhancedDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'student_photos'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

socketio = SocketIO(app, cors_allowed_origins="*")

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for monitoring
monitoring_active = False
video_thread = None
cap = None
current_detected_student = None
student_encodings_cache = []
enhanced_detector = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class UserManager:
    def __init__(self):
        self.init_db()
    
    def init_db(self):
        """Initialize user database"""
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'lecturer',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create default admin user
        admin_hash = generate_password_hash('admin123', method='pbkdf2:sha256')
        cursor.execute('''
            INSERT OR IGNORE INTO users (username, password_hash, role)
            VALUES (?, ?, ?)
        ''', ('admin', admin_hash, 'admin'))
        
        conn.commit()
        conn.close()
    
    def verify_user(self, username, password):
        """Verify user credentials"""
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        conn.close()
        
        if result and check_password_hash(result[0], password):
            return True
        return False
    
    def add_user(self, username, password, role='lecturer'):
        """Add new user"""
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            password_hash = generate_password_hash(password)
            cursor.execute('''
                INSERT INTO users (username, password_hash, role)
                VALUES (?, ?, ?)
            ''', (username, password_hash, role))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False

class StudentManager:
    def __init__(self):
        self.init_db()
    
    def init_db(self):
        """Initialize student database"""
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                photo_path TEXT,
                face_encoding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def add_student(self, student_id, name, photo_path, face_encoding):
        """Add new student"""
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            
            # Convert face encoding to bytes for storage
            encoding_bytes = face_encoding.tobytes() if face_encoding is not None else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO students (student_id, name, photo_path, face_encoding)
                VALUES (?, ?, ?, ?)
            ''', (student_id, name, photo_path, encoding_bytes))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding student: {e}")
            return False
    
    def get_all_students(self):
        """Get all students"""
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT student_id, name, photo_path FROM students ORDER BY name')
        students = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': student[0],
                'name': student[1],
                'photo': f'/static/student_photos/{os.path.basename(student[2])}' if student[2] else None,
                'online': False
            }
            for student in students
        ]
    
    def get_student_encodings(self):
        """Get all student face encodings for recognition"""
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT student_id, name, face_encoding FROM students WHERE face_encoding IS NOT NULL')
        results = cursor.fetchall()
        conn.close()
        
        encodings = []
        for student_id, name, encoding_bytes in results:
            if encoding_bytes:
                # Convert bytes back to numpy array
                encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
                encodings.append({
                    'id': student_id,
                    'name': name,
                    'encoding': encoding
                })
        
        return encodings

    def get_student_by_id(self, student_id):
        """Get student information by ID"""
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT student_id, name, photo_path FROM students WHERE student_id = ?', (student_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'name': result[1],
                'photo': f'/static/student_photos/{os.path.basename(result[2])}' if result[2] else None
            }
        return None

user_manager = UserManager()
student_manager = StudentManager()

def detect_student_in_frame(frame):
    """Detect and identify student in frame using face recognition"""
    global student_encodings_cache, current_detected_student
    
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if not face_encodings:
            current_detected_student = None
            return None
        
        # Compare with known students
        for face_encoding in face_encodings:
            for student_data in student_encodings_cache:
                # Calculate face distance
                face_distance = face_recognition.face_distance([student_data['encoding']], face_encoding)[0]
                confidence = 1 - face_distance
                
                # If confidence is high enough (adjust threshold as needed)
                if confidence > 0.6:
                    student_info = {
                        'id': student_data['id'],
                        'name': student_data['name'],
                        'confidence': confidence
                    }
                    current_detected_student = student_info
                    return student_info
        
        # No match found
        current_detected_student = None
        return None
        
    except Exception as e:
        print(f"Error in student detection: {e}")
        current_detected_student = None
        return None

def create_alert_with_student_info(alert_type, description, confidence=0.8):
    """Create alert with current detected student information"""
    global current_detected_student
    
    # Use detected student info if available, otherwise use "Unknown Student"
    if current_detected_student:
        student_name = current_detected_student['name']
        student_id = current_detected_student['id']
        # Update description to include student info
        description = f"{student_name} (ID: {student_id}) - {description}"
    else:
        student_name = "Unknown Student"
        student_id = "N/A"
    
    return {
        'type': alert_type,
        'description': description,
        'confidence': confidence,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'student_name': student_name,
        'student_id': student_id,
        'severity': 'danger' if alert_type in ['phone_detected', 'multiple_faces', 'unauthorized_person'] else 'warning'
    }

def detect_multiple_faces_in_frame(frame):
    """Detect multiple faces in frame"""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        return len(face_locations) > 1
    except:
        return False

@app.route('/')
def index():
    """Main page - redirect to login if not authenticated"""
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if user_manager.verify_user(username, password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/static/student_photos/<filename>')
def student_photo(filename):
    """Serve student photos"""
    return app.send_static_file(f'student_photos/{filename}')

@app.route('/api/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start monitoring endpoint"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    global monitoring_active, video_thread, cap, student_encodings_cache, enhanced_detector
    
    try:
        # Load student encodings for recognition
        student_encodings_cache = student_manager.get_student_encodings()
        print(f"Loaded {len(student_encodings_cache)} student encodings")
        
        # Initialize enhanced detector
        enhanced_detector = EnhancedDetector()
        print("Enhanced detector initialized")
        
        # Start video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open camera'}), 500
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        monitoring_active = True
        video_thread = threading.Thread(target=video_processing_thread)
        video_thread.daemon = True
        video_thread.start()
        
        return jsonify({'success': True, 'message': 'Monitoring started with enhanced detection'})
    
    except Exception as e:
        print(f"Error starting monitoring: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop monitoring endpoint"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    global monitoring_active, cap, current_detected_student
    
    monitoring_active = False
    current_detected_student = None
    if cap:
        cap.release()
        cap = None
    
    return jsonify({'success': True, 'message': 'Monitoring stopped'})

@app.route('/api/monitoring_status')
def monitoring_status():
    """Get current monitoring status"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    return jsonify({
        'active': monitoring_active,
        'alerts_count': 0,  # You can implement alert counting
        'students_detected': len(student_encodings_cache) if student_encodings_cache else 0,
        'enhanced_detection': enhanced_detector is not None
    })

@app.route('/api/get_students')
def get_students():
    """Get all students"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    students = student_manager.get_all_students()
    return jsonify({'students': students})

@app.route('/api/upload_students', methods=['POST'])
def upload_students():
    """Upload student photos"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if 'photos' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('photos')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    uploaded_count = 0
    errors = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Secure filename
                filename = secure_filename(file.filename)
                
                # Extract student info from filename (format: "StudentID_Name.jpg")
                name_part = filename.rsplit('.', 1)[0]
                if '_' in name_part:
                    student_id, name = name_part.split('_', 1)
                else:
                    student_id = name_part
                    name = name_part
                
                # Save file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Process face encoding
                face_encoding = None
                try:
                    # Load and process image for face recognition
                    image = face_recognition.load_image_file(file_path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        face_encoding = face_encodings[0]
                    else:
                        errors.append(f"No face found in {filename}")
                        continue
                        
                except Exception as e:
                    errors.append(f"Error processing {filename}: {str(e)}")
                    continue
                
                # Add student to database
                if student_manager.add_student(student_id, name, file_path, face_encoding):
                    uploaded_count += 1
                else:
                    errors.append(f"Failed to add {name} to database")
                    
            except Exception as e:
                errors.append(f"Error uploading {file.filename}: {str(e)}")
        else:
            errors.append(f"Invalid file type: {file.filename}")
    
    # Refresh student encodings cache
    global student_encodings_cache
    student_encodings_cache = student_manager.get_student_encodings()
    
    if uploaded_count > 0:
        return jsonify({
            'success': True,
            'count': uploaded_count,
            'errors': errors if errors else None
        })
    else:
        return jsonify({
            'error': 'No files were uploaded successfully',
            'details': errors
        }), 400

def video_processing_thread():
    """Enhanced video processing thread with advanced detection"""
    global monitoring_active, cap, current_detected_student, enhanced_detector
    
    frame_count = 0
    detection_interval = 5  # Process detection every 5 frames for better performance
    
    while monitoring_active and cap and cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                # Transient camera failure - retry instead of exiting
                print("Failed to read frame from camera")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            alerts_to_send = []
            
            # Perform student detection periodically
            if frame_count % detection_interval == 0:
                detected_student = detect_student_in_frame(frame)
                
                if detected_student:
                    # Send student detection update
                    socketio.emit('student_detected', {
                        'name': detected_student['name'],
                        'id': detected_student['id'],
                        'confidence': detected_student['confidence']
                    })
            
            # Enhanced detection checks
            if frame_count % 15 == 0 and enhanced_detector:  # Check every 15 frames
                
                # Check for phone detection
                if enhanced_detector.detect_phone_in_frame(frame) or enhanced_detector.detect_object_near_ear(frame):
                    alert = create_alert_with_student_info(
                        'phone_detected', 
                        'Phone or mobile device detected',
                        confidence=0.85
                    )
                    alerts_to_send.append(alert)
                
                # Check for speaking/talking
                if enhanced_detector.detect_speaking_in_frame(frame):
                    alert = create_alert_with_student_info(
                        'talking_detected',
                        'Talking or mouth movement detected during exam',
                        confidence=0.75
                    )
                    alerts_to_send.append(alert)
                
                # Check for hand movements
                if enhanced_detector.detect_hand_movements(frame):
                    alert = create_alert_with_student_info(
                        'hand_movement_detected',
                        'Suspicious hand movements detected',
                        confidence=0.70
                    )
                    alerts_to_send.append(alert)
                
                # Check for head movements (looking around)
                if enhanced_detector.detect_head_movements(frame):
                    alert = create_alert_with_student_info(
                        'head_movement_detected',
                        'Excessive head movement detected (looking around)',
                        confidence=0.65
                    )
                    alerts_to_send.append(alert)
                
                # Check for multiple faces
                if detect_multiple_faces_in_frame(frame):
                    alert = create_alert_with_student_info(
                        'multiple_faces',
                        'Multiple people detected in exam area',
                        confidence=0.90
                    )
                    alerts_to_send.append(alert)
                
                # Check if no student is detected but monitoring is active
                if not current_detected_student and len(student_encodings_cache) > 0:
                    # Only send this alert occasionally to avoid spam
                    if frame_count % 150 == 0:  # Every 5 seconds
                        alert = create_alert_with_student_info(
                            'no_student_detected',
                            'No registered student detected in exam area',
                            confidence=0.80
                        )
                        alerts_to_send.append(alert)
            
            # Send alerts to frontend
            for alert in alerts_to_send:
                socketio.emit('new_alert', alert)
            
            # Send video frame to frontend (every 3rd frame to reduce bandwidth)
            if frame_count % 3 == 0:
                try:
                    # Draw detection visualizations
                    display_frame = frame.copy()
                    detection_info = {}
                    
                    # Draw face detection boxes
                    try:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_frame)
                        
                        for (top, right, bottom, left) in face_locations:
                            # Draw rectangle around face
                            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            
                            # Add student name if detected
                            if current_detected_student:
                                label = f"{current_detected_student['name']} ({current_detected_student['confidence']:.2f})"
                                cv2.putText(display_frame, label, (left, top - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        detection_info['faces'] = len(face_locations)
                    except Exception as e:
                        print(f"Error drawing face detection: {e}")
                        detection_info['faces'] = 0
                    
                    # Add detection status indicators
                    if enhanced_detector and frame_count % 15 == 0:
                        detection_info.update({
                            'phone': enhanced_detector.detect_phone_in_frame(frame),
                            'speaking': enhanced_detector.detect_speaking_in_frame(frame),
                            'hand_movement': enhanced_detector.detect_hand_movements(frame),
                            'head_movement': enhanced_detector.detect_head_movements(frame)
                        })
                    
                    # Draw detection status on frame
                    status_y = 30
                    if detection_info.get('phone'):
                        cv2.putText(display_frame, "PHONE DETECTED", (10, status_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        status_y += 30
                    
                    if detection_info.get('speaking'):
                        cv2.putText(display_frame, "TALKING DETECTED", (10, status_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        status_y += 30
                    
                    if detection_info.get('hand_movement'):
                        cv2.putText(display_frame, "HAND MOVEMENT", (10, status_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        status_y += 30
                    
                    if detection_info.get('head_movement'):
                        cv2.putText(display_frame, "HEAD MOVEMENT", (10, status_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    
                    # Encode frame as base64
                    _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_data = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send frame with detection info
                    socketio.emit('video_frame', {
                        'frame': frame_data,
                        'detections': detection_info,
                        'student_info': current_detected_student,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"Error in video processing thread: {e}")
            break
    
    print("Video processing thread ended")

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connection_status', {'status': 'connected', 'enhanced_detection': enhanced_detector is not None})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('get_detection_settings')
def handle_get_detection_settings():
    """Get current detection settings"""
    settings = {
        'phone_detection': True,
        'speaking_detection': True,
        'hand_movement_detection': True,
        'head_movement_detection': True,
        'face_recognition': True,
        'multiple_face_detection': True
    }
    emit('detection_settings', settings)

@socketio.on('update_detection_settings')
def handle_update_detection_settings(data):
    """Update detection settings (for future implementation)"""
    print(f"Detection settings update requested: {data}")
    # Here you could implement settings persistence
    emit('settings_updated', {'success': True})

@app.route('/api/detection_stats')
def detection_stats():
    """Get detection statistics"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    stats = {
        'enhanced_detection_active': enhanced_detector is not None,
        'student_recognition_active': len(student_encodings_cache) > 0,
        'total_students': len(student_encodings_cache),
        'monitoring_active': monitoring_active,
        'current_student': current_detected_student
    }
    
    return jsonify(stats)

@app.route('/api/test_detection', methods=['POST'])
def test_detection():
    """Test detection capabilities"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if not cap or not cap.isOpened():
        return jsonify({'error': 'Camera not available'}), 400
    
    try:
        # Capture a test frame
        ret, frame = cap.read()
        if not ret:
            return jsonify({'error': 'Failed to capture test frame'}), 400
        
        test_results = {}
        
        if enhanced_detector:
            test_results['phone_detection'] = enhanced_detector.detect_phone_in_frame(frame)
            test_results['speaking_detection'] = enhanced_detector.detect_speaking_in_frame(frame)
            test_results['hand_movement'] = enhanced_detector.detect_hand_movements(frame)
            test_results['head_movement'] = enhanced_detector.detect_head_movements(frame)
            test_results['object_near_ear'] = enhanced_detector.detect_object_near_ear(frame)
        
        test_results['face_detection'] = len(face_recognition.face_locations(frame)) > 0
        test_results['multiple_faces'] = detect_multiple_faces_in_frame(frame)
        test_results['student_detected'] = detect_student_in_frame(frame) is not None
        
        return jsonify({
            'success': True,
            'test_results': test_results,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({'error': f'Test failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/student_photos', exist_ok=True)
    
    # Copy student photos to static directory for serving
    if os.path.exists('student_photos'):
        import shutil
        for filename in os.listdir('student_photos'):
            src = os.path.join('student_photos', filename)
            dst = os.path.join('static/student_photos', filename)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
    
    print("Starting Enhanced Exam Monitoring System...")
    print("Features enabled:")
    print("- Face Recognition")
    print("- Phone Detection")
    print("- Speaking Detection") 
    print("- Hand Movement Detection")
    print("- Head Movement Detection")
    print("- Multiple Face Detection")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)