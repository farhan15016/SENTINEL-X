
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import json


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
        "flagged": len(alerts) > 0,
        "alert_count": len(alerts),
        "alerts": alerts
    }


class SentinelHandler(BaseHTTPRequestHandler):

    def send_json(self, data, status_code=200):
        body = json.dumps(data).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send_json({
                "project": "SENTINEL-X",
                "status": "running",
                "message": "AI Cybersecurity Intelligence Platform",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "available_endpoints": [
                    "GET /",
                    "GET /api/status",
                    "POST /api/analyse"
                ]
            })

        elif self.path == "/api/status":
            self.send_json({
                "project": "SENTINEL-X",
                "status": "operational",
                "rule_engine": "ready",
                "machine_learning": "not_configured",
                "database": "not_configured"
            })

        else:
            self.send_json({"error": "Endpoint not found"}, 404)

    def do_POST(self):
        if self.path != "/api/analyse":
            self.send_json({"error": "Endpoint not found"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))

            if length <= 0 or length > 10000:
                self.send_json({"error": "Invalid request size"}, 400)
                return

            event = json.loads(self.rfile.read(length))

            if not isinstance(event, dict):
                self.send_json({
                    "error": "Expected a JSON object"
                }, 400)
                return

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
                    or value < 0
                    or value > maximum
                    or not float(value).is_integer()
                ):
                    self.send_json({
                        "error": f"Invalid value for {key}"
                    }, 400)
                    return

            result = analyse_network_event(event)
            self.send_json(result)

        except (ValueError, UnicodeDecodeError):
            self.send_json({"error": "Invalid JSON"}, 400)


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8000), SentinelHandler)
    print(
        "SENTINEL-X running at http://127.0.0.1:8000",
        flush=True
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSENTINEL-X server stopped.")
    finally:
        server.server_close()
