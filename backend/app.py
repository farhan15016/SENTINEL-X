
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import json
import math

import joblib
import pandas as pd


# ============================================================
# SENTINEL-X CONFIGURATION
# ============================================================

PROJECT_NAME = "SENTINEL-X"
HOST = "127.0.0.1"
PORT = 8000

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"

_model = None


# ============================================================
# RULE-BASED DETECTION ENGINE
# ============================================================

def analyse_network_event(event):
    """Detect suspicious activity using transparent rules."""

    alerts = []

    failed_logins = event.get("failed_logins", 0)
    requests_per_minute = event.get("requests_per_minute", 0)
    bytes_sent = event.get("bytes_sent", 0)

    if failed_logins >= 10:
        alerts.append({
            "rule": "MULTIPLE_FAILED_LOGINS",
            "severity": "high",
            "reason": f"{failed_logins} failed login attempts detected."
        })

    if requests_per_minute >= 500:
        alerts.append({
            "rule": "HIGH_REQUEST_RATE",
            "severity": "high",
            "reason": f"{requests_per_minute} requests per minute detected."
        })

    if bytes_sent >= 100_000_000:
        alerts.append({
            "rule": "HIGH_OUTBOUND_DATA",
            "severity": "medium",
            "reason": f"{bytes_sent} bytes sent."
        })

    return {
        "project": PROJECT_NAME,
        "engine": "Rule-based",
        "flagged": len(alerts) > 0,
        "alert_count": len(alerts),
        "alerts": alerts,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ============================================================
# MACHINE-LEARNING MODEL
# ============================================================

def load_model():
    """Load the trained Random Forest pipeline once."""

    global _model

    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}. "
                "Run backend/train_model.py first."
            )

        _model = joblib.load(MODEL_PATH)

    return _model


def predict_network_event(event):
    """Predict normal or attack using the trained ML pipeline."""

    model = load_model()

    if not hasattr(model, "feature_names_in_"):
        raise RuntimeError(
            "The saved model has no feature schema. "
            "Check backend/train_model.py."
        )

    feature_names = list(model.feature_names_in_)

    missing_features = [
        name for name in feature_names if name not in event
    ]

    if missing_features:
        return {
            "error": "Missing required model features",
            "missing_features": missing_features,
            "required_feature_count": len(feature_names)
        }, 400

    features = {}

    for name in feature_names:
        value = event[name]

        if value is None or isinstance(value, bool):
            raise ValueError(
                f"Invalid value for feature '{name}'."
            )

        if isinstance(value, str):
            if not value.strip():
                raise ValueError(
                    f"Feature '{name}' cannot be empty."
                )
            features[name] = value

        elif isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise ValueError(
                    f"Feature '{name}' must be a finite number."
                )
            features[name] = value

        else:
            raise ValueError(
                f"Unsupported value type for feature '{name}'."
            )

    # The saved pipeline handles categorical encoding and imputation.
    input_data = pd.DataFrame(
        [features],
        columns=feature_names
    )

    prediction = model.predict(input_data)[0]

    probabilities = None

    if hasattr(model, "predict_proba"):
        probability_values = model.predict_proba(input_data)[0]

        probabilities = {
            str(label): round(float(probability), 4)
            for label, probability in zip(
                model.classes_,
                probability_values
            )
        }

    label = int(prediction)

    return {
        "project": PROJECT_NAME,
        "engine": "Random Forest",
        "prediction": "attack" if label == 1 else "normal",
        "label": label,
        "probabilities": probabilities,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Prediction uses the trained UNSW-NB15 pipeline, "
            "including categorical encoding and imputation."
        )
    }, 200


# ============================================================
# HTTP API
# ============================================================

class SentinelHandler(BaseHTTPRequestHandler):

    def send_json(self, data, status_code=200):
        body = json.dumps(data, allow_nan=False).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("Invalid Content-Length.")

        if length <= 0 or length > 100_000:
            raise ValueError(
                "Request body must be between 1 and 100000 bytes."
            )

        raw_body = self.rfile.read(length)
        data = json.loads(raw_body.decode("utf-8"))

        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")

        return data

    def do_GET(self):

        if self.path == "/":
            self.send_json({
                "project": PROJECT_NAME,
                "status": "running",
                "message": "AI Cybersecurity Intelligence Platform",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "available_endpoints": [
                    "GET /",
                    "GET /api/status",
                    "POST /api/analyse",
                    "POST /api/predict"
                ]
            })

        elif self.path == "/api/status":
            self.send_json({
                "project": PROJECT_NAME,
                "status": "operational",
                "rule_engine": "ready",
                "machine_learning": (
                    "model_file_available"
                    if MODEL_PATH.is_file()
                    else "model_file_missing"
                ),
                "database": "not_configured"
            })

        else:
            self.send_json(
                {"error": "Endpoint not found"},
                404
            )

    def do_POST(self):

        if self.path not in (
            "/api/analyse",
            "/api/predict"
        ):
            self.send_json(
                {"error": "Endpoint not found"},
                404
            )
            return

        try:
            event = self.read_json_body()

            if self.path == "/api/analyse":
                limits = {
                    "failed_logins": 1_000_000,
                    "requests_per_minute": 1_000_000_000,
                    "bytes_sent": 1_000_000_000_000
                }

                for key, maximum in limits.items():
                    value = event.get(key, 0)

                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                        or value > maximum
                        or not float(value).is_integer()
                    ):
                        self.send_json({
                            "error": f"Invalid value for {key}"
                        }, 400)
                        return

                self.send_json(analyse_network_event(event))
                return

            # Machine-learning prediction endpoint.
            try:
                result, status_code = predict_network_event(event)
                self.send_json(result, status_code)

            except FileNotFoundError as error:
                self.send_json({
                    "error": "Machine-learning model unavailable",
                    "details": str(error)
                }, 503)

            except (ValueError, TypeError) as error:
                self.send_json({
                    "error": "Invalid model input",
                    "details": str(error)
                }, 400)

            except Exception as error:
                self.send_json({
                    "error": "Prediction failed",
                    "details": str(error)
                }, 500)

        except (
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError
        ) as error:
            self.send_json({
                "error": "Invalid request",
                "details": str(error)
            }, 400)

    def log_message(self, format_string, *args):
        print(
            f"[API] {self.address_string()} - "
            f"{format_string % args}"
        )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("SENTINEL-X CYBERSECURITY API")
    print("=" * 55)
    print(f"Model file: {MODEL_PATH}")
    print(f"Model available: {MODEL_PATH.is_file()}")
    print(f"Server URL: http://{HOST}:{PORT}")

    server = HTTPServer((HOST, PORT), SentinelHandler)

    try:
        print("Server started. Press Ctrl+C to stop.")
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nSENTINEL-X server stopped.")

    finally:
        server.server_close()
