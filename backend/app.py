
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import json
import logging
import math

import joblib
import pandas as pd


# ============================================================
# SENTINEL-X CONFIGURATION
# ============================================================

PROJECT_NAME = "SENTINEL-X"
HOST = "127.0.0.1"
PORT = 8000
MAX_BODY_BYTES = 100_000
MAX_BATCH_RECORDS = 100
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"
ANOMALY_MODEL_PATH = BASE_DIR / "models" / "isolation_forest_tuned.joblib"

ALLOWED_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}

_model = None
_anomaly_model = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("sentinel_x")


# ============================================================
# SHARED UTILITIES
# ============================================================

def utc_timestamp():
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def request_path(handler):
    """Return the URL path without query parameters."""
    return urlsplit(handler.path).path


# ============================================================
# RULE-BASED DETECTION ENGINE
# ============================================================

def analyse_network_event(event):
    """Detect suspicious activity using transparent rules."""

    alerts = []

    failed_logins = event["failed_logins"]
    requests_per_minute = event["requests_per_minute"]
    bytes_sent = event["bytes_sent"]

    if failed_logins >= 10:
        alerts.append({
            "rule": "MULTIPLE_FAILED_LOGINS",
            "severity": "high",
            "reason": f"{failed_logins} failed login attempts detected.",
        })

    if requests_per_minute >= 500:
        alerts.append({
            "rule": "HIGH_REQUEST_RATE",
            "severity": "high",
            "reason": f"{requests_per_minute} requests per minute detected.",
        })

    if bytes_sent >= 100_000_000:
        alerts.append({
            "rule": "HIGH_OUTBOUND_DATA",
            "severity": "medium",
            "reason": f"{bytes_sent} bytes sent.",
        })

    return {
        "project": PROJECT_NAME,
        "engine": "Rule-based",
        "flagged": bool(alerts),
        "alert_count": len(alerts),
        "alerts": alerts,
        "timestamp": utc_timestamp(),
    }


def validate_rule_event(event):
    """Validate all required fields for rule-based analysis."""

    limits = {
        "failed_logins": 1_000_000,
        "requests_per_minute": 1_000_000_000,
        "bytes_sent": 1_000_000_000_000,
    }

    for key, maximum in limits.items():
        if key not in event:
            raise ValueError(f"Missing required field: {key}")

        value = event[key]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or value > maximum
            or not float(value).is_integer()
        ):
            raise ValueError(f"Invalid value for {key}")

    return event


# ============================================================
# MACHINE-LEARNING MODEL
# ============================================================

def load_model():
    """Load the trained Random Forest pipeline once."""

    global _model

    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                "Trained model is unavailable. "
                "Run backend/train_model.py first."
            )

        loaded_model = joblib.load(MODEL_PATH)

        if not hasattr(loaded_model, "feature_names_in_"):
            raise RuntimeError("The saved model has no feature schema.")

        _model = loaded_model
        logger.info("Random Forest model loaded successfully.")

    return _model


def predict_network_event(event):
    """Predict normal or attack using the trained ML pipeline."""

    model = load_model()
    feature_names = list(model.feature_names_in_)

    missing_features = [
        name for name in feature_names if name not in event
    ]

    if missing_features:
        return {
            "error": "Missing required model features",
            "missing_features": missing_features,
            "required_feature_count": len(feature_names),
        }, 400

    features = {}

    for name in feature_names:
        value = event[name]

        if value is None or isinstance(value, bool):
            raise ValueError(f"Invalid value for feature '{name}'.")

        if isinstance(value, str):
            if not value.strip():
                raise ValueError(f"Feature '{name}' cannot be empty.")
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

    input_data = pd.DataFrame([features], columns=feature_names)

    prediction = model.predict(input_data)[0]
    label = int(prediction)

    probabilities = None

    if hasattr(model, "predict_proba"):
        probability_values = model.predict_proba(input_data)[0]

        probabilities = {
            str(class_label): round(float(probability), 4)
            for class_label, probability in zip(
                model.classes_,
                probability_values,
            )
        }

    return {
        "project": PROJECT_NAME,
        "engine": "Random Forest",
        "prediction": "attack" if label == 1 else "normal",
        "label": label,
        "probabilities": probabilities,
        "timestamp": utc_timestamp(),
        "note": (
            "Prediction uses the trained UNSW-NB15 pipeline, "
            "including categorical encoding and imputation. "
            "Probabilities are model estimates, not guarantees."
        ),
    }, 200


def load_anomaly_model():
    """Load the separately trained tuned Isolation Forest pipeline."""

    global _anomaly_model

    if _anomaly_model is None:
        if not ANOMALY_MODEL_PATH.is_file():
            raise FileNotFoundError(
                "Tuned Isolation Forest model is unavailable. "
                "Train and save it as backend/models/isolation_forest_tuned.joblib."
            )

        loaded_model = joblib.load(ANOMALY_MODEL_PATH)
        if not hasattr(loaded_model, "feature_names_in_"):
            raise RuntimeError("The Isolation Forest model has no feature schema.")

        _anomaly_model = loaded_model
        logger.info("Tuned Isolation Forest model loaded successfully.")

    return _anomaly_model


def detect_network_anomaly(event):
    """Classify a flow as normal/anomalous and return its anomaly score."""

    model = load_anomaly_model()
    feature_names = list(model.feature_names_in_)
    missing_features = [name for name in feature_names if name not in event]

    if missing_features:
        return {
            "error": "Missing required model features",
            "missing_features": missing_features,
            "required_feature_count": len(feature_names),
        }, 400

    features = {}
    for name in feature_names:
        value = event[name]
        if value is None or isinstance(value, bool):
            raise ValueError(f"Invalid value for feature '{name}'.")
        if isinstance(value, str):
            if not value.strip():
                raise ValueError(f"Feature '{name}' cannot be empty.")
            features[name] = value
        elif isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise ValueError(f"Feature '{name}' must be a finite number.")
            features[name] = value
        else:
            raise ValueError(f"Unsupported value type for feature '{name}'.")

    input_data = pd.DataFrame([features], columns=feature_names)
    raw_prediction = int(model.predict(input_data)[0])
    # Isolation Forest returns 1 for an inlier and -1 for an outlier.
    is_anomaly = raw_prediction == -1
    anomaly_score = float(-model.decision_function(input_data)[0])

    return {
        "project": PROJECT_NAME,
        "engine": "Isolation Forest",
        "prediction": "attack" if is_anomaly else "normal",
        "label": 1 if is_anomaly else 0,
        "anomaly_score": round(anomaly_score, 6),
        "contamination": 0.20,
        "timestamp": utc_timestamp(),
        "note": (
            "Unsupervised anomaly detection using the separately trained "
            "tuned Isolation Forest. An anomaly is not definitive proof of an attack."
        ),
    }, 200


# ============================================================
# HTTP API
# ============================================================


def predict_network_batch(records):
    """Predict up to MAX_BATCH_RECORDS network-flow records."""

    if not isinstance(records, list) or not records:
        raise ValueError("'records' must be a non-empty list.")

    if len(records) > MAX_BATCH_RECORDS:
        raise ValueError(
            f"A batch can contain at most {MAX_BATCH_RECORDS} records."
        )

    results = []

    for index, event in enumerate(records):
        if not isinstance(event, dict):
            raise ValueError(f"Record {index + 1} must be a JSON object.")

        result, status_code = predict_network_event(event)

        if status_code != 200:
            raise ValueError(
                f"Record {index + 1}: "
                f"{result.get('error', 'Invalid record')}."
            )

        results.append({
            "record": index + 1,
            **result,
        })

    attack_count = sum(
        result["label"] == 1 for result in results
    )
    normal_count = len(results) - attack_count

    return {
        "project": PROJECT_NAME,
        "engine": "Random Forest",
        "records_analyzed": len(results),
        "attack_count": attack_count,
        "normal_count": normal_count,
        "results": results,
        "timestamp": utc_timestamp(),
    }


class SentinelHandler(BaseHTTPRequestHandler):
    """HTTP request handler for SENTINEL-X."""

    def send_cors_headers(self):
        """Allow requests from the local development frontend."""

        origin = self.headers.get("Origin", "")

        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type",
        )

    def send_json(self, data, status_code=200):
        """Send a JSON response with consistent headers."""

        body = json.dumps(data, allow_nan=False).encode("utf-8")

        self.send_response(status_code)
        self.send_cors_headers()
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Handle browser CORS preflight requests."""

        self.send_response(204)
        self.send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def read_json_body(self):
        """Read and validate a bounded JSON request body."""

        content_type = self.headers.get("Content-Type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()

        if media_type != "application/json":
            raise ValueError("Content-Type must be application/json.")

        content_length = self.headers.get("Content-Length")

        if content_length is None:
            raise ValueError("Content-Length header is required.")

        try:
            length = int(content_length)
        except (TypeError, ValueError):
            raise ValueError("Invalid Content-Length.")

        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError(
                f"Request body must be between 1 and "
                f"{MAX_BODY_BYTES} bytes."
            )

        raw_body = self.rfile.read(length)

        if len(raw_body) != length:
            raise ValueError("Incomplete request body.")

        try:
            data = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("Request body must contain valid UTF-8 JSON.")

        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")

        return data

    def do_GET(self):
        """Handle health and API information endpoints."""

        path = request_path(self)

        if path == "/":
            self.send_json({
                "project": PROJECT_NAME,
                "status": "running",
                "message": "AI Cybersecurity Intelligence Platform",
                "timestamp": utc_timestamp(),
                "available_endpoints": [
                    "GET /",
                    "GET /api/status",
                    "POST /api/analyse",
                    "POST /api/predict",
                    "POST /api/predict-batch",
                    "POST /api/anomaly",
                ],
            })

        elif path == "/api/status":
            model_status = (
                "loaded" if _model is not None else
                "model_file_available" if MODEL_PATH.is_file() else
                "model_file_missing"
            )
            anomaly_status = (
                "loaded" if _anomaly_model is not None else
                "model_file_available" if ANOMALY_MODEL_PATH.is_file() else
                "model_file_missing"
            )

            self.send_json({
                "project": PROJECT_NAME,
                "status": "operational",
                "rule_engine": "ready",
                "machine_learning": model_status,
                "random_forest": model_status,
                "isolation_forest": anomaly_status,
                "database": "not_configured",
                "timestamp": utc_timestamp(),
            })

        else:
            self.send_json({"error": "Endpoint not found"}, 404)

    def do_POST(self):
        """Handle rule-based, single-record, and batch ML predictions."""

        path = request_path(self)

        if path == "/api/predict-batch":
            try:
                payload = self.read_json_body()
                result = predict_network_batch(payload.get("records"))
                self.send_json(result)
            except ValueError as error:
                self.send_json({
                    "error": "Invalid batch input",
                    "details": str(error),
                }, 400)
            except FileNotFoundError:
                logger.exception("Batch prediction unavailable: model file missing.")
                self.send_json({
                    "error": "Machine-learning model unavailable",
                    "details": "The trained model is not available on the server.",
                }, 503)
            except Exception:
                logger.exception("Unexpected batch prediction failure.")
                self.send_json({
                    "error": "Batch prediction failed",
                    "details": "An unexpected server error occurred.",
                }, 500)
            return

        if path not in ("/api/analyse", "/api/predict", "/api/anomaly"):
            self.send_json({"error": "Endpoint not found"}, 404)
            return

        try:
            event = self.read_json_body()
        except ValueError as error:
            self.send_json({
                "error": "Invalid request",
                "details": str(error),
            }, 400)
            return

        if path == "/api/analyse":
            try:
                validate_rule_event(event)
                self.send_json(analyse_network_event(event))
            except ValueError as error:
                self.send_json({
                    "error": "Invalid rule-based input",
                    "details": str(error),
                }, 400)
            return

        if path == "/api/anomaly":
            try:
                result, status_code = detect_network_anomaly(event)
                self.send_json(result, status_code)
            except FileNotFoundError:
                logger.exception("Anomaly detection unavailable: model file missing.")
                self.send_json({
                    "error": "Isolation Forest model unavailable",
                    "details": (
                        "The tuned model is not available. Train and save "
                        "isolation_forest_tuned.joblib first."
                    ),
                }, 503)
            except (ValueError, TypeError) as error:
                self.send_json({
                    "error": "Invalid anomaly-detection input",
                    "details": str(error),
                }, 400)
            except Exception:
                logger.exception("Unexpected anomaly-detection failure.")
                self.send_json({
                    "error": "Anomaly detection failed",
                    "details": "An unexpected server error occurred.",
                }, 500)
            return

        try:
            result, status_code = predict_network_event(event)
            self.send_json(result, status_code)

        except FileNotFoundError:
            logger.exception("Prediction unavailable: model file missing.")
            self.send_json({
                "error": "Machine-learning model unavailable",
                "details": "The trained model is not available on the server.",
            }, 503)

        except (ValueError, TypeError) as error:
            self.send_json({
                "error": "Invalid model input",
                "details": str(error),
            }, 400)

        except Exception:
            logger.exception("Unexpected prediction failure.")
            self.send_json({
                "error": "Prediction failed",
                "details": "An unexpected server error occurred.",
            }, 500)

    def log_message(self, format_string, *args):
        """Log HTTP access messages."""

        logger.info(
            "%s - %s",
            self.address_string(),
            format_string % args,
        )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("SENTINEL-X CYBERSECURITY API")
    print("=" * 55)
    print(f"Random Forest model: {MODEL_PATH}")
    print(f"Random Forest available: {MODEL_PATH.is_file()}")
    print(f"Tuned Isolation Forest model: {ANOMALY_MODEL_PATH}")
    print(f"Isolation Forest available: {ANOMALY_MODEL_PATH.is_file()}")
    print(f"Server URL: http://{HOST}:{PORT}")

    server = ThreadingHTTPServer((HOST, PORT), SentinelHandler)

    try:
        print("Server started. Press Ctrl+C to stop.")
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nSENTINEL-X server stopped.")

    finally:
        server.server_close()
