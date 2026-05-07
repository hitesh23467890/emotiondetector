# MindScan — AI Emotion Detection Web App
> 2nd Year B.Tech Project | CSE AI&ML | Built with DeepFace + Flask

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Flask](https://img.shields.io/badge/Flask-3.0-green) ![DeepFace](https://img.shields.io/badge/DeepFace-0.0.93-purple)

## What it does
Real-time emotion detection from webcam feed using deep learning.
- Detects 7 emotions: Happy, Sad, Angry, Surprised, Neutral, Fear, Disgust
- Estimates age and gender
- Live mode (auto-analyzes every 2 seconds)
- Scan history tracking
- Beautiful dark sci-fi UI

## Tech Stack
| Layer | Tech |
|-------|------|
| Frontend | HTML5, CSS3, Vanilla JS, Web API |
| Backend | Python, Flask |
| AI Model | DeepFace (FER+ model under the hood) |
| CV | OpenCV |
| Deployment | Can be hosted on Render / Railway (free) |

## Architecture
```
Browser (Webcam)
    │
    │ Base64 JPEG (via fetch POST)
    ▼
Flask API (/analyze)
    │
    │ OpenCV decode → numpy array
    ▼
DeepFace.analyze()
    │
    │ FER+ deep learning model
    ▼
Emotion scores + age + gender
    │
    │ JSON response
    ▼
Frontend renders bars + dominant emotion
```

## Setup & Run

### 1. Clone / download the project
```bash
cd emotion-detector
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> First run downloads DeepFace models (~100MB) automatically.

### 4. Run the app
```bash
python app.py
```

### 5. Open in browser
```
http://localhost:5000
```

## How to Demo (for internship interviews)
1. Open the app, click **START CAMERA**
2. Click **ANALYZE** for a one-shot analysis
3. Click **LIVE MODE** to analyze in real-time every 2 seconds
4. Show the emotion history strip at the bottom right
5. Change your expression and show it updating live

## What to say in interviews
> "I built a real-time emotion detection web app using Flask as the backend API
> and DeepFace — which uses FER+ and ArcFace models under the hood — for the
> inference. The frontend captures webcam frames as base64 and sends them to
> the Flask endpoint via fetch, where OpenCV decodes them and DeepFace returns
> emotion probability vectors. I also added age and gender estimation and a
> live mode that auto-analyzes every 2 seconds."

## Possible Enhancements (to mention in interviews)
- [ ] Add emotion timeline chart (Chart.js)
- [ ] Store sessions in SQLite, show emotion trends over time
- [ ] Multi-face detection (multiple people in frame)
- [ ] Export emotion report as PDF
- [ ] Deploy on Render with Docker

## Project Structure
```
emotion-detector/
├── app.py              ← Flask backend (API routes)
├── requirements.txt    ← Python dependencies
├── templates/
│   └── index.html      ← Full frontend (HTML + CSS + JS)
└── README.md
```

## Demo Mode
If DeepFace isn't installed, the app automatically runs in **Demo Mode**
with simulated emotion scores — useful for UI testing and presentations.
