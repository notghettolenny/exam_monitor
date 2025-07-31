import threading
import time
import cv2
from exam_monitor_system import ExamMonitorConfig, ExamMonitoringSystem

class ThreadedMonitor:
    def __init__(self, video_source=0, display_scale=0.5):
        self.video_source = video_source
        self.display_scale = display_scale

        self.config = ExamMonitorConfig()
        self.monitor = ExamMonitoringSystem(self.config)
        self.monitor.setup()

        self.cap = cv2.VideoCapture(self.video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        self.frame_lock = threading.Lock()
        self.frame = None
        self.processed_frame = None
        self.running = False

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)

    def start(self):
        self.running = True
        self.capture_thread.start()
        self.detection_thread.start()

    def stop(self):
        self.running = False
        self.capture_thread.join()
        self.detection_thread.join()
        self.cap.release()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.frame = frame.copy()
            time.sleep(0.005)  # Yield CPU, ~200 FPS max read rate

    def _detection_loop(self):
        while self.running:
            with self.frame_lock:
                frame = self.frame.copy() if self.frame is not None else None
            if frame is not None:
                self.monitor.process_frame(frame)
                with self.frame_lock:
                    self.processed_frame = frame.copy()
            time.sleep(0.01)  # Adjust based on expected frame processing time

    def get_frame(self):
        with self.frame_lock:
            return self.processed_frame.copy() if self.processed_frame is not None else None
# threaded_monitor.py

import threading
import time
import cv2
from exam_monitor_system import ExamMonitorConfig, ExamMonitoringSystem

class ThreadedMonitor:
    def __init__(self):
        self.config = ExamMonitorConfig()
        self.monitor = ExamMonitoringSystem(self.config)
        self.monitor.setup()

        self.frame_lock = threading.Lock()
        self.alert_lock = threading.Lock()

        self.latest_frame = None
        self.latest_alerts = []

        self.running = False
        self.capture_thread = None
        self.detection_thread = None
        self.cap = None

    def start(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.capture_thread.start()
        self.detection_thread.start()

    def stop(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
        if self.detection_thread:
            self.detection_thread.join()
        if self.cap:
            self.cap.release()

    def _capture_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
            time.sleep(0.005)

    def _detection_loop(self):
        while self.running:
            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
            if frame is not None:
                alerts = self.monitor.process_frame(frame)
                with self.alert_lock:
                    self.latest_alerts = alerts
            time.sleep(0.01)

    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_latest_alerts(self):
        with self.alert_lock:
            return list(self.latest_alerts)

    def get_alerts(self):
        return self.monitor.alert_manager.get_recent_alerts()