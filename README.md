# 🎓 exam\_monitor

![Django](https://img.shields.io/badge/built%20with-Django-092E20?style=flat\&logo=django\&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A robust web application for managing and monitoring exams — online or in-person.
It allows administrators to schedule exams, enroll students, assign proctors, and monitor sessions seamlessly.

---

## ✨ Features

✅ Secure user authentication and role-based access (Admin, Proctor, Student)
✅ Create and manage exam sessions with ease
✅ Enroll students and assign them to specific sessions
✅ Assign proctors to monitor exams
✅ Track student activity and monitor session progress
✅ Generate reports of completed sessions

---

## 🏗️ Tech Stack

| Layer        | Technology                                             |
| ------------ | ------------------------------------------------------ |
| **Backend**  | [Django](https://www.djangoproject.com/)               |
| **Frontend** | Django Templates + Bootstrap (optional)                |
| **Database** | SQLite (default) — can be swapped for PostgreSQL/MySQL |

---

## 📁 Project Structure

```
exam_monitor/
├── authentication/      # User authentication & roles
├── exam_sessions/        # Exam session management
├── proctors/             # Proctor assignment & tracking
├── students/             # Student enrollment & profiles
├── static/               # Static assets (CSS, JS, images)
├── templates/            # HTML templates
├── exam_monitor/         # Project settings & URLs
├── manage.py
```

---

## 🚀 Getting Started

### 📋 Prerequisites

* Python 3.8+
* pip
* virtualenv (recommended)

---

### ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/notghettolenny/exam_monitor.git

cd exam_monitor

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Create an admin user
python manage.py createsuperuser

# Start the development server
python manage.py runserver
```

Visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to see it in action!

---

## 👤 User Roles

| Role        | Permissions                                  |
| ----------- | -------------------------------------------- |
| **Admin**   | Full access: manage sessions, users, reports |
| **Proctor** | Monitor assigned exam sessions               |
| **Student** | Participate in assigned exams                |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

* Fork the repo
* Create your feature branch (`git checkout -b feature/your-feature`)
* Commit your changes
* Push to the branch (`git push origin feature/your-feature`)
* Open a Pull Request

---

## 📬 Contact

For questions or feedback, feel free to reach out via [GitHub Issues](https://github.com/notghettolenny/exam_monitor/issues).

