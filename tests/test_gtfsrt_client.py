from etaslip.gtfsrt.client import fetch_feed


def test_fetch_feed_sets_x_api_key_when_provided(monkeypatch):
    captured = {}

    class DummyResp:
        def __init__(self):
            self.content = b"abc"

        def raise_for_status(self):
            return None

    def fake_get(url, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResp()

    import etaslip.gtfsrt.client as client_mod

    monkeypatch.setattr(client_mod.requests, "get", fake_get)

    out = fetch_feed("https://example.com/feed", api_key="secret123", timeout_sec=20)
    assert out == b"abc"
    assert captured["headers"]["x-api-key"] == "secret123"
    assert captured["timeout"] == 20


def test_fetch_feed_omits_x_api_key_when_none(monkeypatch):
    captured = {}

    class DummyResp:
        def __init__(self):
            self.content = b"xyz"

        def raise_for_status(self):
            return None

    def fake_get(url, headers, timeout):
        captured["headers"] = headers
        return DummyResp()

    import etaslip.gtfsrt.client as client_mod

    monkeypatch.setattr(client_mod.requests, "get", fake_get)

    out = fetch_feed("https://example.com/feed", api_key=None, timeout_sec=20)
    assert out == b"xyz"
    assert "x-api-key" not in captured["headers"]
