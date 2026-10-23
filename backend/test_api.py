import json
import math
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from app import (
    SentinelHandler,
    analyse_network_event,
    detect_network_anomaly,
    validate_rule_event,
)


# ============================================================
# RULE-BASED DETECTION TESTS
# ============================================================

class TestSentinelRuleEngine(unittest.TestCase):

    def test_normal_event_has_no_alerts(self):
        event = {
            "failed_logins": 0,
            "requests_per_minute": 100,
            "bytes_sent": 1000,
        }

        result = analyse_network_event(event)

        self.assertFalse(result["flagged"])
        self.assertEqual(result["alert_count"], 0)
        self.assertEqual(result["alerts"], [])

    def test_multiple_failed_logins(self):
        event = {
            "failed_logins": 10,
            "requests_per_minute": 100,
            "bytes_sent": 1000,
        }

        result = analyse_network_event(event)

        self.assertTrue(result["flagged"])
        self.assertIn(
            "MULTIPLE_FAILED_LOGINS",
            [alert["rule"] for alert in result["alerts"]],
        )

    def test_high_request_rate(self):
        event = {
            "failed_logins": 0,
            "requests_per_minute": 500,
            "bytes_sent": 1000,
        }

        result = analyse_network_event(event)

        self.assertIn(
            "HIGH_REQUEST_RATE",
            [alert["rule"] for alert in result["alerts"]],
        )

    def test_high_outbound_data(self):
        event = {
            "failed_logins": 0,
            "requests_per_minute": 100,
            "bytes_sent": 100_000_000,
        }

        result = analyse_network_event(event)

        self.assertIn(
            "HIGH_OUTBOUND_DATA",
            [alert["rule"] for alert in result["alerts"]],
        )

    def test_all_three_alerts(self):
        event = {
            "failed_logins": 15,
            "requests_per_minute": 800,
            "bytes_sent": 150_000_000,
        }

        result = analyse_network_event(event)

        self.assertTrue(result["flagged"])
        self.assertEqual(result["alert_count"], 3)

    def test_negative_login_count_is_rejected(self):
        event = {
            "failed_logins": -1,
            "requests_per_minute": 100,
            "bytes_sent": 1000,
        }

        with self.assertRaises(ValueError):
            validate_rule_event(event)

    def test_missing_field_is_rejected(self):
        event = {
            "failed_logins": 0,
            "requests_per_minute": 100,
        }

        with self.assertRaises(ValueError):
            validate_rule_event(event)

    def test_boolean_value_is_rejected(self):
        event = {
            "failed_logins": True,
            "requests_per_minute": 100,
            "bytes_sent": 1000,
        }

        with self.assertRaises(ValueError):
            validate_rule_event(event)


# ============================================================
# ISOLATION FOREST DETECTOR TESTS
# ============================================================

class TestIsolationForestAnomalyDetection(unittest.TestCase):

    def test_missing_features_returns_400(self):
        mock_model = type(
            "MockModel",
            (),
            {"feature_names_in_": ["feature_a", "feature_b"]},
        )()

        with patch(
            "app.load_anomaly_model",
            return_value=mock_model,
        ):
            result, status_code = detect_network_anomaly(
                {"feature_a": 1.0}
            )

        self.assertEqual(status_code, 400)
        self.assertEqual(
            result["error"],
            "Missing required model features",
        )
        self.assertEqual(
            result["missing_features"],
            ["feature_b"],
        )

    def test_boolean_feature_is_rejected(self):
        mock_model = type(
            "MockModel",
            (),
            {"feature_names_in_": ["feature_a"]},
        )()

        with patch(
            "app.load_anomaly_model",
            return_value=mock_model,
        ):
            with self.assertRaises(ValueError):
                detect_network_anomaly({"feature_a": True})

    def test_non_finite_feature_is_rejected(self):
        mock_model = type(
            "MockModel",
            (),
            {"feature_names_in_": ["feature_a"]},
        )()

        with patch(
            "app.load_anomaly_model",
            return_value=mock_model,
        ):
            with self.assertRaises(ValueError):
                detect_network_anomaly({"feature_a": math.nan})

    def test_missing_model_is_propagated(self):
        with patch(
            "app.load_anomaly_model",
            side_effect=FileNotFoundError("model missing"),
        ):
            with self.assertRaises(FileNotFoundError):
                detect_network_anomaly({})


# ============================================================
# HTTP-LEVEL ANOMALY ENDPOINT TESTS
# ============================================================

class TestAnomalyEndpointHTTP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Use an available ephemeral port to avoid port conflicts.
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            SentinelHandler,
        )
        cls.port = cls.server.server_address[1]

        cls.thread = threading.Thread(
            target=cls.server.serve_forever,
            daemon=True,
        )
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def post_anomaly(self, payload):
        request = Request(
            f"http://127.0.0.1:{self.port}/api/anomaly",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=3) as response:
                body = json.loads(
                    response.read().decode("utf-8")
                )
                return response.status, body

        except HTTPError as error:
            body = json.loads(
                error.read().decode("utf-8")
            )
            return error.code, body

    def test_successful_anomaly_response(self):
        mock_result = {
            "project": "SENTINEL-X",
            "engine": "Isolation Forest",
            "prediction": "normal",
            "label": 0,
            "anomaly_score": -0.01,
            "contamination": 0.20,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "note": "Mock anomaly detection result.",
        }

        with patch(
            "app.detect_network_anomaly",
            return_value=(mock_result, 200),
        ):
            status, body = self.post_anomaly({})

        self.assertEqual(status, 200)
        self.assertEqual(
            body["engine"],
            "Isolation Forest",
        )
        self.assertEqual(body["prediction"], "normal")
        self.assertEqual(body["label"], 0)

    def test_invalid_input_returns_400(self):
        with patch(
            "app.detect_network_anomaly",
            side_effect=ValueError("Invalid feature"),
        ):
            status, body = self.post_anomaly({})

        self.assertEqual(status, 400)
        self.assertEqual(
            body["error"],
            "Invalid anomaly-detection input",
        )

    def test_missing_model_returns_503(self):
        with patch(
            "app.detect_network_anomaly",
            side_effect=FileNotFoundError("Model missing"),
        ):
            status, body = self.post_anomaly({})

        self.assertEqual(status, 503)
        self.assertEqual(
            body["error"],
            "Isolation Forest model unavailable",
        )


if __name__ == "__main__":
    unittest.main()