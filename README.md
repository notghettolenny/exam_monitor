# 🎓 exam\_monitor

![Python](https://img.shields.io/badge/built%20with-Python-3776AB?style=flat\&logo=python\&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An intelligent Python-based exam monitoring system that combines face recognition, YOLO object detection, and a GUI to track students, record attendance, and raise alerts during exams.

---

## ✨ Features

✅ Face recognition for student identification
✅ YOLO-based detection for enhanced monitoring
✅ GUI for managing sessions and real-time monitoring
✅ Logs attendance and alerts to JSON/text files
✅ Stores student data and session photos
✅ Easy to extend and customize

---

## 🏗️ Tech Stack

| Layer                | Technology                   |
| -------------------- | ---------------------------- |
| **Language**         | Python 3                     |
| **Machine Learning** | YOLOv5, face\_recognition    |
| **GUI**              | Tkinter (or similar)  |
| **Data Storage**     | `.db`, `.pkl` files          |
| **Dependencies**     | Listed in `requirements.txt` |

---

## 📁 Project Structure

```
exam_monitor/
├── app.py
├── run.py
├── enroll_students.py
├── exam_monitor_gui.py
├── exam_monitor_system.py
├── enhanced_detector.py
├── test_face_recognition_debug.py
├── requirements.txt
├── templates/                 # GUI or HTML templates
├── static/student_photos/     # Captured student photos
├── student_photos/            # Additional photo storage
├── users.db                   # User database
├── students.pkl               # Pickled student data
├── yolov5s.pt                 # YOLOv5 model weights
├── alert_log_*.json           # Alert logs
├── attendance_log.txt         # Attendance records
```

---

## 🚀 Getting Started

### 📋 Prerequisites

* Python 3.8+
* pip
* virtualenv (optional but recommended)
* GPU (optional, for faster YOLO inference)

---

### ⚙️ Installation

1️⃣ Clone the repository:

```bash
git clone https://github.com/notghettolenny/exam_monitor.git
cd exam_monitor
```

2️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

3️⃣ Run the application:

```bash
python3 app.py
```

(or use `exam_monitor_gui.py` if it has the main GUI)

---

## 📄 Logs & Data

* Attendance logs → `attendance_log.txt`
* Alerts → `alert_log_*.json`
* Student photos → `static/student_photos/`
* Model weights → `yolov5s.pt`

---

## 👥 User Roles

| Role        | Description                |
| ----------- | -------------------------- |
| **Admin**   | Runs & supervises sessions |
| **Proctor** | Monitors students          |
| **Student** | Identified and tracked     |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

* Fork the repo
* Create a feature branch (`git checkout -b feature/your-feature`)
* Commit & push your changes
* Open a Pull Request

---

## 📬 Contact

For questions or feedback, open an [issue](https://github.com/notghettolenny/exam_monitor/issues).

---

### 🔗 Notes

* Make sure `yolov5s.pt` is present in the root folder.
* Ensure your camera is properly configured if running face recognition.
* Logs and captured photos are saved automatically during a session.

---


