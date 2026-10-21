
import unittest

from app import analyse_network_event, validate_rule_event


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


if __name__ == "__main__":
    unittest.main()
