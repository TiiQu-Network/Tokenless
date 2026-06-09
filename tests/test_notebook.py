import json
import queue

import requests

from tokenless.notebook import KaggleNotebookManager


def test_ntfy_listener_reconnects_after_transient_failure(monkeypatch):
    expected = "https://reconnected.trycloudflare.com"
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=True):
            yield json.dumps({"event": "message", "message": expected})

    def fake_get(*_args, **_kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise requests.ConnectionError("temporary ntfy failure")
        return Response()

    monkeypatch.setattr("tokenless.notebook.requests.get", fake_get)
    monkeypatch.setattr("tokenless.notebook.time.sleep", lambda _seconds: None)
    public_url_queue = queue.Queue(maxsize=1)

    KaggleNotebookManager._listen_for_public_url("test-topic", public_url_queue, 5)

    assert public_url_queue.get_nowait() == expected
    assert len(calls) == 2
