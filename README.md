# SENTINEL-X 🛡️

### AI-Powered Cybersecurity Threat Detection Platform

SENTINEL-X is an educational cybersecurity prototype that analyses network-event data to identify suspicious activity. It combines rule-based detection with supervised machine learning and unsupervised anomaly detection, presented through a React dashboard.

## Features

- **Rule-based detection:** Identifies suspicious events using configurable detection rules.
- **Random Forest classifier:** Predicts normal or attack traffic using the UNSW-NB15 dataset.
- **Isolation Forest anomaly detection:** Detects anomalous traffic patterns using a separately trained, tuned unsupervised model.
- **Model evaluation:** Reports accuracy, attack precision, attack recall, F1-score and confusion matrices.
- **Feature importance:** Highlights input features that influence Random Forest predictions.
- **REST-style JSON API:** Exposes analysis, prediction, batch-prediction and anomaly-detection endpoints.
- **Security dashboard:** React, TypeScript and Vite frontend for viewing model results and project metrics.
- **Input validation and automated tests:** Includes request validation and backend regression checks.

## Technology Stack

- Python
- scikit-learn and joblib
- React, TypeScript and Vite
- Python standard-library HTTP server
- UNSW-NB15 network intrusion detection dataset
- Git and GitHub

## Architecture

1. Network-event features are submitted to the backend API.
2. The rule engine evaluates events against configured rules.
3. The Random Forest model performs supervised classification.
4. The tuned Isolation Forest model performs unsupervised anomaly detection.
5. The frontend presents predictions and evaluation metrics.

## Dataset

SENTINEL-X uses the UNSW-NB15 dataset for model training and evaluation.

The training and testing CSV files are stored locally and excluded from Git. Obtain the dataset through its official source and follow its documentation and citation requirements.

Dataset source: [UNSW-NB15 — UNSW Canberra](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

## Model Evaluation

The following results were recorded on the project's official test set.

| Metric | Random Forest | Isolation Forest (baseline) | Isolation Forest (tuned) |
|---|---:|---:|---:|
| Accuracy | 90.73% | 55.30% | 70.66% |
| Attack precision | 87.63% | 96.43% | 76.06% |
| Attack recall | 96.83% | 19.53% | 68.17% |
| Attack F1-score | 92.00% | 32.48% | 71.90% |

The tuned Isolation Forest uses a contamination setting of 0.20, selected using a separate validation split. The `/api/anomaly` endpoint uses the tuned model. Baseline results are retained for comparison.

These metrics reflect evaluation on UNSW-NB15 data; they do not establish performance on live network traffic.

## Requirements

- Python 3.10 or later
- Node.js and npm
- Local UNSW-NB15 dataset files for model training or retraining

Use the Python environment and dependency versions configured for this project where possible.

## Run the Backend

From the project root:

```bash
source .venv/bin/activate
python backend/app.py
```

The backend runs at `http://127.0.0.1:8000`.

## Run the Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Project information |
| GET | `/api/status` | Backend status |
| POST | `/api/analyse` | Rule-based network-event analysis |
| POST | `/api/predict` | Random Forest prediction |
| POST | `/api/predict-batch` | Batch Random Forest predictions |
| POST | `/api/anomaly` | Tuned Isolation Forest anomaly detection |

Prediction endpoints require request data matching the input schema expected by the backend. For machine-learning predictions, provide the required dataset features.

## Testing

From the project root:

```bash
source .venv/bin/activate
python -m pytest -q
python -m compileall -q backend
```

Build-check the frontend:

```bash
cd frontend
npm run build
```

The backend test suite has passed with **8 tests** in the current verified run.

## Important Limitations

- The dashboard currently uses a fixed demonstration record for its model-prediction buttons; it does not capture live network traffic.
- An anomaly prediction is not definitive proof of a cyberattack.
- The Isolation Forest tuning results are experimental and require validation on additional real-world traffic.
- This prototype is intended for educational and research purposes, not as a replacement for a production security monitoring system.

## Author

Farhan

GitHub: [@farhan15016](https://github.com/farhan15016)