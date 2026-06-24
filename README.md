# 🪪 Smart ID Pass

> AI-powered student identity verification system — CAI360, Princess Nourah Bint Abdulrahman University

---

## 👥 Team

| Name | Student ID | Section |
|------|-----------|---------|
| Shahad Alanazi | 445008913 | 61U |
| Rana Alashur | 445008907 | 64U |
| Lama Alshehri | 445008898 | 64U |
| Lina Albdrani | 445008881 | 61U |
| Hessa Alhuwail | 445008908 | 61U |

---

## 📌 Overview

Smart ID Pass verifies student identity by combining three AI components in a single pipeline:

1. **Liveness Detection** — checks the selfie is from a real, live person (not a photo/screen)
2. **OCR** — reads the student ID number from the university ID card image
3. **Face Matching** — compares the live selfie against the registered face in the database

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3, Flask |
| Face Recognition | InsightFace `buffalo_l` |
| OCR | EasyOCR (Arabic + English) |
| Liveness Detection | OpenCV (Haar Cascade + texture analysis) |
| Database | SQLite |
| Frontend | HTML / CSS / JavaScript |

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/smart-id-pass.git
cd smart-id-pass
```

### 2. Install dependencies
```bash
pip install flask insightface easyocr opencv-python numpy werkzeug
```

### 3. Run the app
```bash
python app.py
```

The server starts at **http://localhost:5000**

---

## 📁 Project Structure

```
smart-id-pass/
│
├── app.py                        # Flask backend + AI pipeline
├── student_face_database.sqlite  # SQLite database (auto-created)
├── static/
│   └── uploads/                  # Temporary uploaded images
├── templates/
│   └── index.html                # Frontend interface
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/verify` | Run full verification pipeline |
| `GET` | `/health` | Server health check |
| `GET` | `/api/students` | List all registered students |
| `GET` | `/api/students/<id>` | Get a specific student |
| `POST` | `/api/students` | Register a new student |
| `DELETE` | `/api/students/<id>` | Delete a student record |
| `GET` | `/api/stats` | System statistics & thresholds |

### Example — POST /verify

**Request (multipart/form-data):**
```
selfie    → image file (jpg/png)
id_card   → image file (jpg/png)
```

**Response:**
```json
{
  "success": true,
  "decision": "Access Granted",
  "student_name": "Lama Alshehri",
  "student_id": "445008898",
  "similarity_score": 0.87,
  "liveness_score": 0.91
}
```

---

## 🔍 How It Works

```
User submits selfie + ID card
        │
        ▼
 [1] Liveness Detection
   OpenCV checks face size, pixel variance,
   Laplacian sharpness, and color variation
   → Score < 0.50 → ❌ Access Denied
        │
        ▼
 [2] OCR Extraction
   CLAHE preprocessing → EasyOCR
   extracts 9-digit student ID
   (confidence ≥ 0.70)
        │
        ▼
 [3] Face Matching
   InsightFace buffalo_l computes
   512-D embeddings → cosine similarity
   → Score < 0.50 → ❌ Access Denied
        │
        ▼
   ✅ Access Granted
```

---

## ⚙️ Configuration

In `app.py`:

```python
LIVENESS_THRESHOLD  = 0.5   # minimum liveness score to pass
SIMILARITY_THRESHOLD = 0.5  # minimum face similarity to pass
MAX_FILE_SIZE        = 5MB  # maximum upload size
ALLOWED_EXTENSIONS   = {'png', 'jpg', 'jpeg', 'gif'}
```

---

## 📊 Performance

| Component | Metric | Value |
|-----------|--------|-------|
| Liveness Detection | Test Accuracy | 99.91% |
| OCR Extraction | ID Accuracy | ~96% |
| End-to-End Pipeline | F1-Score | 1.0 (6 test cases) |
| Average Response Time | — | ~68.95 seconds |

> ⚠️ Response time is a known limitation. Model pre-loading and in-memory image processing are planned optimisations.

---

## ⚠️ Known Limitations

- Average verification time (~69s) is too slow for real-time access control
- Liveness detection uses OpenCV heuristics, not a deep CNN — susceptible to sophisticated spoofing
- Face similarity threshold (0.50) is permissive; production systems typically use ≥ 0.85
- Performance depends heavily on image quality and lighting conditions

---

## 🔮 Future Improvements

- [ ] Reduce response time to < 5 seconds via in-memory processing
- [ ] Replace heuristic liveness with a trained anti-spoofing CNN
- [ ] Raise face similarity threshold to ≥ 0.75 with ROC analysis
- [ ] Add JWT authentication to the API
- [ ] Deploy on cloud (Docker + gunicorn + HTTPS)
- [ ] Real-time video stream verification

---

## 📄 License

This project was developed for academic purposes as part of CAI360.
