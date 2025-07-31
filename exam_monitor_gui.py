import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import cv2
from threaded_monitor import ThreadedMonitor

class ExamMonitorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Exam Monitoring System")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")

        self.setup_ui()

        self.monitor = ThreadedMonitor(display_scale=0.5)
        self.running = False

        # GUI update interval (aiming for ~60 FPS = ~16ms per frame)
        self.frame_interval = 16

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
            self.monitor.start()
            self.update_video()
            self.schedule_alert_update()

    def stop_monitoring(self):
        if self.running:
            self.running = False
            self.status_label.config(text="Status: Stopped", fg="red")
            self.monitor.stop()

    def update_video(self):
        if self.running:
            frame = self.monitor.get_latest_frame()
            if frame is not None:
                display_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
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
        alerts = self.monitor.detector.alert_manager.get_recent_alerts() if self.monitor else []
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
