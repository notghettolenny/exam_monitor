
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import cv2
from exam_monitor_system import ExamMonitorConfig, ExamMonitoringSystem

class ExamMonitorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Exam Monitoring System")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")

        self.setup_ui()

        self.config = ExamMonitorConfig()
        self.monitor = ExamMonitoringSystem(self.config)
        self.monitor.setup()

        self.running = False
        self.cap = None
        self.current_frame = None
        # Performance tuning for smoother video display
        self.frame_interval = 50  # milliseconds between GUI frame updates (~20 FPS)
        self.display_scale = 0.5  # scale factor for GUI preview

    def setup_ui(self):
        title = tk.Label(self.root, text="AI Exam Monitoring Dashboard", font=("Helvetica", 18), bg="#f0f0f0")
        title.pack(pady=10)

        self.video_label = tk.Label(self.root)
        self.video_label.pack(pady=10)

        start_btn = tk.Button(self.root, text="Start Monitoring", command=self.start_monitoring, width=20, bg="#4CAF50", fg="white")
        start_btn.pack(pady=5)

        stop_btn = tk.Button(self.root, text="Stop Monitoring", command=self.stop_monitoring, width=20, bg="#f44336", fg="white")
        stop_btn.pack(pady=5)

        alert_label = tk.Label(self.root, text="Alerts", font=("Helvetica", 14), bg="#f0f0f0")
        alert_label.pack(pady=10)

        self.alert_display = tk.Listbox(self.root, height=10, width=80, font=("Courier", 10))
        self.alert_display.pack(pady=5)

        self.status_label = tk.Label(self.root, text="Status: Idle", font=("Helvetica", 12), bg="#f0f0f0", fg="gray")
        self.status_label.pack(pady=10)

    def start_monitoring(self):
        if not self.running:
            self.running = True
            self.status_label.config(text="Status: Monitoring", fg="green")
            self.cap = cv2.VideoCapture(0)
            # Set a modest resolution to reduce CPU usage
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.update_video()
            self.schedule_alert_update()

    def stop_monitoring(self):
        self.running = False
        self.status_label.config(text="Status: Stopped", fg="red")
        if self.cap:
            self.cap.release()
            self.cap = None

    def monitor_loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.monitor.process_frame(frame)
            self.current_frame = frame.copy()

    def update_video(self):
        if self.running and self.current_frame is not None:
            display_frame = cv2.resize(
                self.current_frame,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
            )
            cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        self.root.after(self.frame_interval, self.update_video)

    def schedule_alert_update(self):
        self.update_alert_display()
        if self.running:
            self.root.after(1000, self.schedule_alert_update)

    def update_alert_display(self):
        alerts = self.monitor.alert_manager.get_recent_alerts()
        self.alert_display.delete(0, 'end')
        for alert in alerts[-10:]:
            timestamp = time.strftime('%H:%M:%S', time.localtime(alert.timestamp))
            self.alert_display.insert('end', f"[{timestamp}] {alert.alert_type}")
        self.alert_display.yview(tk.END)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ExamMonitorGUI()
    app.run()
