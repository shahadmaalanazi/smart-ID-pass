import os
import re
import base64
import sqlite3
from datetime import datetime

import cv2
import numpy as np
import easyocr
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from insightface.app import FaceAnalysis
from werkzeug.utils import secure_filename
import time
import csv
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
app.secret_key = "smart-id-secret-2026"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "verification_logs.csv")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DATABASE_PATH = os.path.join(BASE_DIR, "student_face_database.sqlite")
LIVENESS_MODEL_PATH = os.path.join(BASE_DIR, "cnn_liveness_model.keras")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_FILE_SIZE = 5 * 1024 * 1024

IMG_SIZE = 128
LIVENESS_THRESHOLD = 0.5


LIVE_SCORE_IS_HIGH = False

SIMILARITY_THRESHOLD = 0.5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

_liveness_model = None
_ocr_reader = None
_face_analyzer = None
def log_verification(case_id, expected, predicted, execution_time, stage, notes="", details=None):
    file_exists = os.path.exists(LOG_PATH)

    row = {
        "timestamp": datetime.now().isoformat(),
        "case_id": case_id,
        "expected": expected,
        "predicted": predicted,
        "execution_time": execution_time,
        "stage": stage,
        "notes": notes,
        "details": str(details) if details else ""
    }

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_expected_from_request(default="unknown"):
    data = request.get_json(silent=True) or {}
    return data.get("expected", request.form.get("expected", default))


def get_case_id_from_request():
    data = request.get_json(silent=True) or {}
    return data.get("case_id", request.form.get("case_id", datetime.now().strftime("TC_%Y%m%d_%H%M%S")))

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_liveness_model():
    global _liveness_model

    if _liveness_model is None:
        print("Model path:", LIVENESS_MODEL_PATH)
        print("Model exists:", os.path.exists(LIVENESS_MODEL_PATH))

        if not os.path.exists(LIVENESS_MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {LIVENESS_MODEL_PATH}")

        _liveness_model = tf.keras.models.load_model(LIVENESS_MODEL_PATH, compile=False)
        print("USING YOUR TRAINED CNN MODEL")

    return _liveness_model


def predict_liveness(image_path):
    model = get_liveness_model()

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read selfie image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    score = float(model.predict(img, verbose=0)[0][0])

    if LIVE_SCORE_IS_HIGH:
        is_live = score >= LIVENESS_THRESHOLD
        confidence = int(score * 100)
    else:
        is_live = score < LIVENESS_THRESHOLD
        confidence = int((1.0 - score) * 100)

    print("CNN raw score:", score)
    print("CNN result:", "LIVE" if is_live else "FAKE")

    return {
        "is_liveness": is_live,
        "confidence": confidence if is_live else 0,
        "raw_score": score,
        "method": "cnn"
    }


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["ar", "en"], gpu=False)
    return _ocr_reader


def get_face_analyzer():
    global _face_analyzer
    if _face_analyzer is None:
        _face_analyzer = FaceAnalysis(name="buffalo_l")
        _face_analyzer.prepare(ctx_id=0)
    return _face_analyzer


def preprocess_for_ocr(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read image for OCR")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    sharpened = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(save_path, sharpened)
    return save_path


def run_ocr(image_path):
    return get_ocr_reader().readtext(image_path, detail=1, paragraph=False)


def extract_student_id_with_confidence(ocr_results):
    all_numbers = []

    for _, text, conf in ocr_results:
        digits_only = re.sub(r"\D", "", str(text).strip())

        if digits_only:
            all_numbers.append((digits_only, float(conf)))

        if len(digits_only) == 9:
            return digits_only, float(conf)

    for digits, conf in all_numbers:
        if 7 <= len(digits) <= 11:
            return digits, conf

    return None, None


def detect_and_crop_bright_region(image_path, save_path, debug_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Cannot read card image")

    original = image.copy()
    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    cv2.imwrite(debug_path, thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        cv2.imwrite(save_path, original)
        return save_path, False

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < 0.05 * (h * w):
        cv2.imwrite(save_path, original)
        return save_path, False

    x, y, bw, bh = cv2.boundingRect(largest)
    pad_x = int(0.03 * bw)
    pad_y = int(0.03 * bh)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    cropped = original[y1:y2, x1:x2]
    cv2.imwrite(save_path, cropped)

    return save_path, True


def extract_student_id_from_card(card_path, timestamp):
    cropped_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_cropped_card.jpg")
    debug_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_debug_mask.jpg")
    processed_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_ocr.jpg")

    cropped_path, found_card = detect_and_crop_bright_region(card_path, cropped_path, debug_path)

    if not found_card:
        cropped_path = card_path

    processed_path = preprocess_for_ocr(cropped_path, processed_path)
    ocr_results = run_ocr(processed_path)

    return extract_student_id_with_confidence(ocr_results)


def match_selfie_with_student_record(selfie_path, student_id, timestamp):
    if not os.path.exists(DATABASE_PATH):
        raise FileNotFoundError(f"Database not found: {DATABASE_PATH}")

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    student = cursor.fetchone()
    conn.close()

    if student is None:
        return None, None, False

    student_name = student[1] if len(student) > 1 else "Unknown"
    image_blob = student[6] if len(student) > 6 else None

    if image_blob is None:
        return student_name, None, False

    db_face_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_db_face.jpg")

    with open(db_face_path, "wb") as f:
        f.write(image_blob)

    img1 = cv2.imread(selfie_path)
    img2 = cv2.imread(db_face_path)

    if img1 is None or img2 is None:
        return student_name, None, False

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    analyzer = get_face_analyzer()
    faces1 = analyzer.get(img1)
    faces2 = analyzer.get(img2)

    if len(faces1) == 0 or len(faces2) == 0:
        return student_name, None, False

    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding

    similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    return student_name, similarity, similarity > SIMILARITY_THRESHOLD


def verify_identity(card_path, selfie_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    liveness_result = predict_liveness(selfie_path)

    if not liveness_result["is_liveness"]:
        return {
            "success": False,
            "result": "ACCESS DENIED",
            "stage": "liveness",
            "message": "فشل فحص حيوية الوجه",
            "confidence": liveness_result["confidence"],
            "liveness": liveness_result
        }

    student_id, ocr_conf = extract_student_id_from_card(card_path, timestamp)

    if student_id is None:
        return {
            "success": False,
            "result": "ACCESS DENIED",
            "stage": "ocr",
            "message": "لم يتم العثور على رقم الطالب في البطاقة",
            "confidence": 0,
            "liveness": liveness_result
        }

    student_name, similarity, match = match_selfie_with_student_record(
        selfie_path,
        student_id,
        timestamp
    )

    if not match:
        return {
            "success": False,
            "result": "ACCESS DENIED",
            "stage": "face_verification",
            "message": "الوجه لا يطابق السجل في قاعدة البيانات",
            "confidence": int(similarity * 100) if similarity is not None else 0,
            "student_id": student_id,
            "student_name": student_name,
            "ocr_confidence": ocr_conf,
            "similarity": similarity,
            "liveness": liveness_result
        }

    confidence = int((liveness_result["confidence"] + similarity * 100) / 2)

    return {
        "success": True,
        "result": "ACCESS GRANTED",
        "stage": "completed",
        "message": f"تم التحقق من الهوية بنجاح - {student_name}",
        "confidence": confidence,
        "student_id": student_id,
        "student_name": student_name,
        "ocr_confidence": ocr_conf,
        "similarity": similarity,
        "similarity_percent": int(similarity * 100),
        "liveness": liveness_result
    }


def save_uploaded_file(file_obj, prefix):
    if file_obj is None or file_obj.filename == "":
        raise ValueError(f"Missing file: {prefix}")

    if not allowed_file(file_obj.filename):
        raise ValueError("Allowed formats: png, jpg, jpeg, gif")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = secure_filename(f"{timestamp}_{prefix}_{file_obj.filename}")
    path = os.path.join(UPLOAD_FOLDER, filename)
    file_obj.save(path)

    return path


def save_base64_image(data_url, prefix):
    if not data_url:
        raise ValueError(f"Missing base64 image: {prefix}")

    raw = data_url.split(",", 1)[1] if "," in data_url else data_url
    image_bytes = base64.b64decode(raw)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = secure_filename(f"{timestamp}_{prefix}.jpg")
    path = os.path.join(UPLOAD_FOLDER, filename)

    with open(path, "wb") as f:
        f.write(image_bytes)

    return path

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/verify", methods=["POST"])
def verify_base64():
    start_time = time.time()
    case_id = get_case_id_from_request()
    expected = get_expected_from_request()

    try:
        data = request.get_json(silent=True) or {}

        if "id_image" not in data or "face_image" not in data:
            execution_time = round(time.time() - start_time, 2)

            log_verification(
                case_id, expected, "fail",
                execution_time, "input",
                "Missing images"
            )

            return jsonify({
                "success": False,
                "result": "ACCESS DENIED",
                "message": "Missing id_image or face_image",
                "execution_time": execution_time
            }), 400

        # حفظ الصور
        card_path = save_base64_image(data["id_image"], "card")
        face_path = save_base64_image(data["face_image"], "selfie")

        # تشغيل النظام
        result = verify_identity(card_path, face_path)

        execution_time = round(time.time() - start_time, 2)
        predicted = "success" if result.get("success") else "fail"

        # تسجيل الأداء
        log_verification(
            case_id=case_id,
            expected=expected,
            predicted=predicted,
            execution_time=execution_time,
            stage=result.get("stage", "unknown"),
            notes=result.get("message", ""),
            details=result
        )

        result["execution_time"] = execution_time
        return jsonify(result), 200

    except Exception as e:
        execution_time = round(time.time() - start_time, 2)

        log_verification(
            case_id, expected, "fail",
            execution_time, "error",
            str(e)
        )

        return jsonify({
            "success": False,
            "result": "ACCESS DENIED",
            "message": str(e),
            "execution_time": execution_time
        }), 500


@app.route("/api/performance", methods=["GET"])
def api_performance():
    if not os.path.exists(LOG_PATH):
        return jsonify({"success": False, "message": "No logs found"}), 404

    columns = [
        "timestamp", "case_id", "expected", "predicted",
        "execution_time", "stage", "notes", "details"
    ]

    df = pd.read_csv(LOG_PATH)

    # لو الهيدر ناقص أو غلط
    if "expected" not in df.columns:
        df = pd.read_csv(LOG_PATH, header=None, names=columns)

    df.columns = df.columns.str.strip().str.lower()

    df["expected"] = df["expected"].astype(str).str.strip().str.lower()
    df["predicted"] = df["predicted"].astype(str).str.strip().str.lower()

    df = df[
        df["expected"].isin(["success", "fail"]) &
        df["predicted"].isin(["success", "fail"])
    ]

    if df.empty:
        return jsonify({
            "success": False,
            "message": "No valid test data. Make sure expected is success or fail."
        }), 400

    y_true = df["expected"]
    y_pred = df["predicted"]

    return jsonify({
        "success": True,
        "total_tests": int(len(df)),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, pos_label="success", zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, pos_label="success", zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, pos_label="success", zero_division=0), 4),
        "average_execution_time": round(df["execution_time"].astype(float).mean(), 2)
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "database_exists": os.path.exists(DATABASE_PATH),
        "liveness_model_exists": os.path.exists(LIVENESS_MODEL_PATH)
    })


@app.errorhandler(413)
def too_large(error):
    return jsonify({
        "success": False,
        "message": "File too large"
    }), 413


@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)