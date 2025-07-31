import cv2
import threading
import time

class ThreadedMonitor:
    def __init__(self, camera_index=0, scale=1.0):
        self.camera_index = camera_index
        self.scale = scale  # 0.5 for half-res preview, 1.0 for full-res
        self.cap = cv2.VideoCapture(self.camera_index)

        # Set properties if supported
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        self.frame_lock = threading.Lock()
        self.running = False
        self.frame = None

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        if not self.running:
            self.running = True
            self.capture_thread.start()

    def stop(self):
        self.running = False
        self.capture_thread.join()
        if self.cap:
            self.cap.release()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if self.scale != 1.0:
                    frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
                with self.frame_lock:
                    self.frame = frame
            time.sleep(0.001)  # yield CPU

    def get_latest_frame(self):
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None
