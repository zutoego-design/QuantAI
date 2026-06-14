import requests

from qss.ingestion.fred import _fetch_series


class _Response:
    text = "DATE,UNRATE\n2026-01-01,4.0\n"

    def raise_for_status(self):
        return None


def test_fred_fetch_retries_transient_network_error(monkeypatch):
    calls = 0

    def _get(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise requests.ConnectionError("temporary")
        return _Response()

    monkeypatch.setattr("qss.ingestion.fred.requests.get", _get)
    monkeypatch.setattr("qss.ingestion.fred.time.sleep", lambda seconds: None)

    result = _fetch_series("UNRATE", retry_delays=(0.0,))

    assert calls == 2
    assert result.iloc[0]["value"] == 4.0


def test_fred_fetch_uses_api_when_key_is_configured(monkeypatch):
    class _ApiResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "observations": [
                    {"date": "2026-01-01", "value": "4.0"},
                ]
            }

    request = {}

    def _get(url, **kwargs):
        request["url"] = url
        request["params"] = kwargs["params"]
        return _ApiResponse()

    monkeypatch.setattr("qss.ingestion.fred.requests.get", _get)

    result = _fetch_series("UNRATE", api_key="secret", retry_delays=())

    assert request["url"].endswith("/fred/series/observations")
    assert request["params"]["api_key"] == "secret"
    assert result.iloc[0]["value"] == 4.0
