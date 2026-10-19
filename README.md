
# SENTINEL-X 🛡️
### AI-Powered Cybersecurity Threat Detection Platform

SENTINEL-X is a cybersecurity project designed to analyse network traffic, identify suspicious activity, and generate alerts for potential threats.

The project is being developed as a lightweight security intelligence platform, with a focus on explainable detection, machine learning, and network intrusion analysis.

## 🎯 Project Objectives

- Detect suspicious network behaviour.
- Identify potential brute-force login attempts.
- Flag unusually high request rates.
- Detect unusually large outbound data transfers.
- Explore machine-learning-based intrusion detection using the UNSW-NB15 dataset.
- Present security findings through a future interactive dashboard.

## ✅ Current Features

- Rule-based network event analysis.
- Detection of multiple failed login attempts.
- High request-rate detection.
- Large outbound data-transfer detection.
- JSON API built with Python's standard library.
- Input validation and structured JSON responses.
- Automated tests for four representative scenarios.

## 🧠 Planned Features

- Supervised machine-learning classification.
- Unsupervised anomaly detection.
- Evaluation using precision, recall, F1-score, and a confusion matrix.
- Interactive security dashboard.
- Alert history and visual analytics.

## 🧰 Technology Stack

- **Language:** Python
- **Backend:** Python HTTP server and JSON API
- **Dataset:** UNSW-NB15
- **Version control:** Git and GitHub
- **Planned frontend:** React, TypeScript, and Vite
- **Planned storage:** SQLite

## 📊 Dataset

SENTINEL-X uses the UNSW-NB15 network intrusion detection dataset for research and experimentation.

The training and testing CSV files are stored locally and excluded from this repository to avoid uploading large dataset files.

Dataset source: [UNSW-NB15 — UNSW Canberra](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

Please consult the dataset's documentation and citation requirements before using or redistributing its contents.

## 🚀 Running the Backend

Requirements: Python 3.10 or later.

From the project root, run:

```bash
python3 backend/app.py
```

The server starts at:

`http://127.0.0.1:8000`

### Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Project information |
| GET | `/api/status` | Backend status |
| POST | `/api/analyse` | Analyse a network event |

### Example Network Event

Send a JSON request to `POST /api/analyse`:

```json
{
  "failed_logins": 15,
  "requests_per_minute": 120,
  "bytes_sent": 5000
}
```

The current rule engine should flag this event because it contains at least 10 failed login attempts.

## 🧪 Testing

Run the existing detector tests from the project root:

```bash
python3 backend/test_detector.py
```

The initial test suite covers brute-force login attempts, high request rates, large outbound transfers, and normal network activity.

## ⚠️ Project Status

SENTINEL-X is under active development.

The current implementation uses transparent, threshold-based rules. Machine-learning models, the frontend dashboard, and database integration are planned components and should not be considered implemented until they are built and tested.

This project is intended for educational and research purposes. It is not a substitute for a production security monitoring system.

## 👨‍💻 Author

Farhan

GitHub: [@farhan15016](https://github.com/farhan15016)
