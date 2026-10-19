
def analyse_network_event(event):
    """Identify suspicious network activity using simple rules."""

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
