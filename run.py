from exam_monitor.exam_monitor_system import ExamMonitoringSystem, ExamMonitorConfig

# Step 1: Create config object
config = ExamMonitorConfig()

# (Optional) Adjust thresholds
config.face_recognition_tolerance = 0.5
config.max_gaze_deviation = 30.0
config.alert_cooldown = 5.0

# Step 2: Initialize the system
monitor = ExamMonitoringSystem(config)

# Step 3: Setup system (load models, load student database)
monitor.setup()

# Step 4: Start monitoring using webcam (0) or test video file path
monitor.start_monitoring(video_source=0)  # 0 = webcam
# monitor.start_monitoring(video_source="test_video.mp4")  # For video file
