import base64
import json
import queue
from pathlib import Path

import pytest
import requests

from tokenless import GPT_OSS_MODEL_ID, TokenlessLLM
from tokenless.notebook import KaggleNotebookManager


class Response:
    def __init__(self, data, *, error=None):
        self._data = data
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        return self._data


def endpoint_llm() -> TokenlessLLM:
    llm = TokenlessLLM(model=GPT_OSS_MODEL_ID)
    llm._running = True
    llm._base_url = "https://ready.trycloudflare.com"
    llm._openai_client = object()
    return llm


def test_send_uploads_pdf_then_uses_page_mapping(tmp_path, monkeypatch):
    pdf = tmp_path / "legal source.pdf"
    pdf.write_bytes(b"%PDF-1.4\nsource")
    llm = endpoint_llm()
    calls = []

    def fake_post(url, data, headers, timeout):
        calls.append(
            {
                "url": url,
                "body": data.read(),
                "headers": headers,
                "timeout": timeout,
            }
        )
        return Response({"status": "running"})

    def fake_get(url, timeout):
        calls.append({"poll_url": url, "timeout": timeout})
        return Response({"status": "complete", "pages": 1})

    monkeypatch.setattr("tokenless.client.requests.post", fake_post)
    monkeypatch.setattr("tokenless.client.requests.get", fake_get)
    monkeypatch.setattr("tokenless.client.PDF_PAGE_JOB_POLL_INTERVAL", 0)
    monkeypatch.setattr(
        llm,
        "_send_pdf_page_mapped",
        lambda message, **kwargs: json.dumps({"message": message}),
    )

    result = llm.send("x" * 4000, file_path=str(pdf), file_upload_timeout=45)

    assert json.loads(result) == {"message": "x" * 4000}
    assert calls[0]["url"].startswith(
        "https://ready.trycloudflare.com/tokenless/pdf/upload/"
    )
    assert calls[0]["body"] == b"%PDF-1.4\nsource"
    assert calls[0]["headers"] == {
        "Content-Type": "application/pdf",
        "X-Tokenless-Filename-B64": base64.b64encode(
            "legal source.pdf".encode("utf-8")
        ).decode("ascii"),
    }
    assert 44 <= calls[0]["timeout"] <= 45
    assert calls[1]["poll_url"] == calls[0]["url"]
    assert calls[1]["timeout"] == 60
    assert llm._pdf_context is True


def test_upload_retries_transient_dns_failure_with_same_job_id(
    tmp_path,
    monkeypatch,
):
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4\nsource")
    llm = endpoint_llm()
    post_urls = []

    def fake_post(url, data, headers, timeout):
        post_urls.append(url)
        if len(post_urls) == 1:
            raise requests.ConnectionError("temporary DNS failure")
        return Response({"status": "complete", "pages": 1})

    monkeypatch.setattr("tokenless.client.requests.post", fake_post)
    monkeypatch.setattr("tokenless.client.PDF_PAGE_JOB_POLL_INTERVAL", 0)
    monkeypatch.setattr(
        llm,
        "_send_pdf_page_mapped",
        lambda message, **kwargs: json.dumps({"units": []}),
    )

    result = llm.send("x" * 4000, file_path=str(pdf))

    assert json.loads(result) == {"units": []}
    assert len(post_urls) == 2
    assert post_urls[0] == post_urls[1]


def test_endpoint_readiness_retries_dns_failure(monkeypatch):
    calls = []
    doh_calls = []

    def fake_get(url, timeout):
        calls.append((url, timeout))
        if len(calls) == 1:
            raise requests.ConnectionError("DNS not propagated yet")
        return Response({"version": "ready"})

    monkeypatch.setattr("tokenless.client.requests.get", fake_get)
    monkeypatch.setattr("tokenless.client.PDF_PAGE_JOB_POLL_INTERVAL", 0)
    monkeypatch.setattr(
        TokenlessLLM,
        "_register_doh_override",
        lambda hostname: doh_calls.append(hostname),
    )

    TokenlessLLM._wait_for_endpoint_ready(
        "https://ready.trycloudflare.com",
        timeout=30,
    )

    assert len(calls) == 2
    assert calls[0][0] == "https://ready.trycloudflare.com/api/version"
    assert doh_calls == ["ready.trycloudflare.com"]


def test_doh_override_registers_valid_ipv4_addresses(monkeypatch):
    class DnsResponse(Response):
        def json(self):
            return {
                "Answer": [
                    {"data": "104.16.230.132"},
                    {"data": "not-an-address"},
                    {"data": "104.16.231.132"},
                ]
            }

    monkeypatch.setattr(
        "tokenless.client.requests.get",
        lambda *args, **kwargs: DnsResponse({}),
    )

    addresses = TokenlessLLM._register_doh_override(
        "fallback.trycloudflare.com"
    )

    assert addresses == ("104.16.230.132", "104.16.231.132")


def test_failed_upload_disables_previous_pdf_context(tmp_path, monkeypatch):
    pdf = tmp_path / "replacement.pdf"
    pdf.write_bytes(b"%PDF-1.4\nreplacement")
    llm = endpoint_llm()
    llm._pdf_context = True

    def fail_request(*args, **kwargs):
        raise requests.HTTPError("conversion failed")

    monkeypatch.setattr(llm, "_retry_pdf_job_request", fail_request)

    with pytest.raises(requests.HTTPError, match="conversion failed"):
        llm.send("prompt", file_path=str(pdf))

    assert llm._pdf_context is False


def test_send_file_path_requires_endpoint(tmp_path):
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    llm = TokenlessLLM(model=GPT_OSS_MODEL_ID)
    llm._running = True

    with pytest.raises(RuntimeError, match="requires a running Tokenless endpoint"):
        llm.send("prompt", file_path=str(pdf))


def test_document_mapping_sends_prompt_once_and_receives_only_json(monkeypatch):
    llm = endpoint_llm()
    llm._pdf_context = True
    posts = []
    get_responses = iter(
        [
            Response(
                {
                    "status": "running",
                    "current_page": 1,
                    "total_pages": 2,
                    "page_number": 1,
                }
            ),
            Response(
                {
                    "status": "complete",
                    "pages": [
                        {
                            "page": 1,
                            "values": [{"units": [{"unit_id": "U1"}]}],
                            "repaired": False,
                        },
                        {
                            "page": 2,
                            "values": [{"units": [{"unit_id": "U2"}]}],
                            "repaired": True,
                        },
                    ],
                }
            ),
        ]
    )

    def fake_post(url, json, timeout):
        posts.append((url, json, timeout))
        return Response({"status": "running"})

    monkeypatch.setattr("tokenless.client.requests.post", fake_post)
    monkeypatch.setattr(
        "tokenless.client.requests.get",
        lambda url, timeout: next(get_responses),
    )
    monkeypatch.setattr("tokenless.client.PDF_PAGE_JOB_POLL_INTERVAL", 0)

    result = llm.send("x" * 4000, page_progress=False)

    assert json.loads(result) == {
        "units": [{"unit_id": "U1"}, {"unit_id": "U2"}]
    }
    assert len(posts) == 1
    payload = posts[0][1]
    assert payload["messages"] == [{"role": "user", "content": "x" * 4000}]
    assert payload["tokenless_pdf_map"] == {
        "repair_malformed_json": True,
        "json_repair_attempts": 2,
    }
    assert "pdf_markdown" not in json.dumps(payload)


def test_document_poll_recovers_new_tunnel_and_keeps_same_job(monkeypatch):
    llm = endpoint_llm()
    llm._pdf_context = True
    post_urls = []
    get_urls = []
    recoveries = []

    def fake_post(url, json, timeout):
        post_urls.append(url)
        return Response({"status": "running"})

    def fake_get(url, timeout):
        get_urls.append(url)
        if len(get_urls) == 1:
            raise requests.HTTPError("530 tunnel unavailable")
        return Response(
            {
                "status": "complete",
                "pages": [{"page": 1, "values": [], "repaired": False}],
            }
        )

    def recover():
        recoveries.append(True)
        llm._base_url = "https://replacement.trycloudflare.com"
        return True

    monkeypatch.setattr("tokenless.client.requests.post", fake_post)
    monkeypatch.setattr("tokenless.client.requests.get", fake_get)
    monkeypatch.setattr(llm, "_recover_public_endpoint", recover)
    monkeypatch.setattr("tokenless.client.PDF_PAGE_JOB_POLL_INTERVAL", 0)

    assert json.loads(llm.send("x" * 4000, page_progress=False)) == []
    assert len(post_urls) == 1
    assert recoveries == [True]
    assert get_urls[0].startswith("https://ready.trycloudflare.com/")
    assert get_urls[1].startswith("https://replacement.trycloudflare.com/")
    assert get_urls[0].split("/tokenless/jobs/")[1] == get_urls[1].split(
        "/tokenless/jobs/"
    )[1]


def test_public_url_monitor_keeps_listening_for_replacement(monkeypatch):
    first = "https://first.trycloudflare.com"
    second = "https://second.trycloudflare.com"
    calls = 0

    class StreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=True):
            yield json.dumps({"event": "message", "message": first})
            yield json.dumps({"event": "message", "message": second})

    def fake_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return StreamResponse()
        raise KeyboardInterrupt

    monkeypatch.setattr("tokenless.notebook.requests.get", fake_get)
    notebook = KaggleNotebookManager(model=GPT_OSS_MODEL_ID)
    first_url_queue = queue.Queue(maxsize=1)

    with pytest.raises(KeyboardInterrupt):
        notebook._monitor_public_urls("topic", first_url_queue, 60)

    assert first_url_queue.get_nowait() == first
    assert notebook.public_url == second


def test_recovery_prefers_monitored_replacement_url(monkeypatch):
    llm = endpoint_llm()

    class Notebook:
        public_url = "https://replacement.trycloudflare.com"

        def latest_public_url(self):
            raise AssertionError("live monitored URL should be used before logs")

    llm._notebook = Notebook()
    monkeypatch.setattr(llm, "_register_doh_override", lambda hostname: ())
    monkeypatch.setattr(llm, "_wait_for_endpoint_ready", lambda *args, **kwargs: None)

    assert llm._recover_public_endpoint() is True
    assert llm._base_url == "https://replacement.trycloudflare.com"


def test_server_template_supports_reusable_pdf_uploads():
    root = Path(__file__).resolve().parents[1]
    body = (
        root
        / "tokenless"
        / "kernels"
        / "gpt_oss_20b"
        / "serve.template.py"
    ).read_text(encoding="utf-8")

    assert 'r"/tokenless/pdf/upload/' in body
    assert "def _replace_pdf_context" in body
    assert "def _clear_pdf_context" in body
    assert "PAGE_JOBS.clear()" in body
    assert "existing_job = _get_pdf_upload_job(job_id)" in body
    assert "def _run_pdf_document_job" in body
    assert 'payload.get("tokenless_pdf_map")' in body
    assert '"pages": page_results' in body
    assert "def _start_public_tunnel" in body
    assert "Restarting public tunnel without stopping active jobs" in body
    assert '"http://localhost:8000"' in body
    assert "UPLOADED_PDF" in body
