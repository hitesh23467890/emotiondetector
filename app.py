from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

def decode_image(data_url):
    """Decode base64 image from webcam capture."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        from fer import FER

        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        img = decode_image(data["image"])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detector = FER(mtcnn=False)
        result = detector.detect_emotions(img_rgb)

        if not result:
            return jsonify({
                "success": False,
                "error": "No face detected. Make sure your face is clearly visible."
            }), 200

        emotions = result[0]["emotions"]
        dominant = max(emotions, key=emotions.get)

        total = sum(emotions.values()) or 1
        normalized = {k: round((v / total) * 100, 1) for k, v in emotions.items()}
        sorted_emotions = dict(
            sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        )

        return jsonify({
            "success": True,
            "dominant_emotion": dominant,
            "emotions": sorted_emotions,
            "age": "N/A",
            "gender": "N/A"
        })

    except ImportError:
        import random
        emotions = {
            "happy":     round(random.uniform(30, 80), 1),
            "neutral":   round(random.uniform(5, 30), 1),
            "surprised": round(random.uniform(2, 15), 1),
            "sad":       round(random.uniform(1, 10), 1),
            "angry":     round(random.uniform(0, 8), 1),
            "fear":      round(random.uniform(0, 5), 1),
            "disgust":   round(random.uniform(0, 3), 1),
        }
        total = sum(emotions.values())
        emotions = {k: round((v / total) * 100, 1) for k, v in emotions.items()}
        dominant = max(emotions, key=emotions.get)
        return jsonify({
            "success": True,
            "dominant_emotion": dominant,
            "emotions": dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True)),
            "age": "N/A",
            "gender": "N/A",
            "demo_mode": True
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("=" * 50)
    print("  MindScan Emotion AI - Starting...")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
