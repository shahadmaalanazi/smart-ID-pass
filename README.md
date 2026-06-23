# Smart ID Verification System

## Overview

Smart ID Verification System is an AI-powered identity authentication platform designed to enhance security and prevent identity fraud. The system combines Computer Vision and Machine Learning techniques to verify user identities through facial recognition and liveness detection.

Unlike traditional verification systems that rely solely on face matching, this solution performs a multi-layer verification process. It first determines whether the presented face belongs to a real person using a liveness detection model, then compares the captured face against registered identities stored in the database. This approach helps protect the system from spoofing attacks such as printed photos, screenshots, or images displayed on mobile devices.

The project integrates TensorFlow, OpenCV, Flask, and SQLite into a complete web-based application, providing an automated, secure, and user-friendly verification experience. The system is designed to simulate real-world identity verification scenarios used in universities, secure facilities, attendance systems, and access control environments.

By combining AI-driven facial analysis with anti-spoofing mechanisms, Smart ID Verification System delivers a reliable and intelligent authentication solution capable of distinguishing between genuine users and fraudulent attempts.

## Features
- Face Detection and Recognition
- Liveness Detection (Real vs Fake Face)
- Identity Verification
- Secure User Authentication
- Web-Based Interface
- SQLite Database Integration
- Audit Logging and Verification Records

## Technologies Used
- Python
- Flask
- TensorFlow / Keras
- OpenCV
- SQLite
- HTML/CSS

## System Architecture
### Frontend
- Web interface using HTML/CSS
- Camera integration for image capture

### Backend
- Flask server
- Handles model execution and decision logic

### Pipeline Flow
1. User captures or uploads a facial image.
2. The system performs liveness detection.
3. If the face is real, face matching is performed.
4. The captured face is compared against stored identities.
5. Access is granted if a valid match is found.
6. Verification results are stored for auditing purposes.

## Main Functions

### predict_liveness()
Determines whether the detected face belongs to a live person or a spoof attempt.

### face_similarity()
Calculates facial similarity between the captured image and stored records.

### smart_id_pass()
Executes the complete verification workflow and returns the final authentication decision.

## Project Structure

```text
Smart-ID-Pass/
│
├── app.py
├── requirements.txt
├── cnn_liveness_model.keras
├── student_face_database.sqlite
├── verification_audit.sqlite
│
├── templates/
│   └── *.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
└── README.md
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/smart-id-pass.git
cd smart-id-pass
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

## Future Improvements
- Multi-factor authentication
- Advanced anti-spoofing techniques
- Cloud deployment
- Real-time webcam verification
- Integration with national identity systems

## Authors
shahad alanazi 
rana alashur
lama alshehri
lina albdrani

Developed as an Artificial Intelligence project focused on secure identity verification using Computer Vision and Machine Learning technologies.

## License
This project was developed for educational and academic purposes.
