
import json
from urllib.request import Request, urlopen
from urllib.error import URLError

URL = "http://127.0.0.1:8000/api/analyse"

tests = [
    {
        "name": "Brute-force login attempts",
        "event": {
            "failed_logins": 15,
            "requests_per_minute": 100,
            "bytes_sent": 5000
        },
        "expected": True
    },
    {
        "name": "High request rate",
        "event": {
            "failed_logins": 0,
            "requests_per_minute": 800,
            "bytes_sent": 5000
        },
        "expected": True
    },
    {
        "name": "Large outbound data transfer",
        "event": {
            "failed_logins": 0,
            "requests_per_minute": 100,
            "bytes_sent": 150_000_000
        },
        "expected": True
    },
    {
        "name": "Normal network activity",
        "event": {
            "failed_logins": 0,
            "requests_per_minute": 100,
            "bytes_sent": 5000
        },
        "expected": False
    }
]

passed = 0

print("\nSENTINEL-X DETECTOR TESTS")
print("=" * 40)

for test in tests:
    request = Request(
        URL,
        data=json.dumps(test["event"]).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urlopen(request, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))

        correct = result["flagged"] == test["expected"]

        if correct:
            passed += 1

        print(
            f"\n[{'PASS' if correct else 'FAIL'}] "
            f"{test['name']}"
        )
        print(f"Flagged: {result['flagged']}")
        print(f"Alerts: {result['alert_count']}")

    except (URLError, ValueError, KeyError) as error:
        print(f"\n[ERROR] {test['name']}: {error}")
        print("Check that app.py is running.")
        break

print("\n" + "=" * 40)
print(f"RESULT: {passed}/{len(tests)} tests passed")
