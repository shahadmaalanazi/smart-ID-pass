# Smart ID Verification System

## Overview
Smart ID Verification System is an AI-powered web application designed to verify user identities through facial recognition and liveness detection. The system ensures that the presented face belongs to a real person and matches the registered identity stored in the database.

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

## System Workflow
1. User uploads or captures a facial image.
2. The system performs liveness detection.
3. If the face is verified as real, facial matching is executed.
4. The captured face is compared with registered users.
5. Access is granted if a match is found.
6. Verification results are stored for auditing purposes.

## Project Structure
