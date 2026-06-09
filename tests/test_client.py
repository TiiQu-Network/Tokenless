<<<<<<< HEAD
=======
import json

>>>>>>> 7f54bf7 (JSON extractor)
import pytest
import requests

from tokenless import GPT_OSS_MODEL_ID, TokenlessLLM


class DummyNotebook:
    public_url = None
    smoke_test_message = None

    def launch(self, **_kwargs):
        raise AssertionError("unit tests must not launch Kaggle")

    def stop(self):
        return None


class CapturingNotebook:
    public_url = None
    smoke_test_message = "converted markdown"

    def __init__(self):
        self.launch_kwargs = None

    def launch(self, **kwargs):
        self.launch_kwargs = kwargs

    def stop(self):
        return None


class PublicUrlNotebook:
    public_url = "https://ready.trycloudflare.com"
    smoke_test_message = None

    def launch(self, **kwargs):
        self.launch_kwargs = kwargs

    def stop(self):
        return None


<<<<<<< HEAD
=======
class FailingNotebook:
    public_url = None
    smoke_test_message = None

    def __init__(self):
        self.stopped = False

    def launch(self, **_kwargs):
        raise requests.HTTPError("temporary Kaggle failure")

    def stop(self):
        self.stopped = True


>>>>>>> 7f54bf7 (JSON extractor)
class CapturingTunnel:
    def __init__(self, url):
        self.url = url
        self.timeout = None

    def get_url(self, timeout=300):
        self.timeout = timeout
        return self.url

    def close(self):
        return None


def make_llm():
    llm = TokenlessLLM(model=GPT_OSS_MODEL_ID)
    llm._notebook = DummyNotebook()
    return llm


def test_gpt_oss_model_is_supported():
    llm = make_llm()

    assert llm.model == "gpt-oss:20b"


def test_send_requires_start():
    llm = make_llm()

    with pytest.raises(RuntimeError, match=r"Call \.start\(\) first"):
        llm.send("hello")


def test_agents_model_requires_running_endpoint():
    llm = make_llm()

    with pytest.raises(RuntimeError, match="Call .start"):
        llm.as_agents_model()


def test_strands_model_requires_running_endpoint():
    llm = make_llm()

    with pytest.raises(RuntimeError, match="Call .start"):
        llm.as_strands_model()


def test_langchain_model_requires_running_endpoint():
    llm = make_llm()

    with pytest.raises(RuntimeError, match="Call .start"):
        llm.as_langchain_llm()


def test_start_with_public_url_does_not_launch_kaggle():
    llm = make_llm()

    url = llm.start(public_url="https://example.trycloudflare.com", show_progress=False)

    assert url == "https://example.trycloudflare.com"
    assert llm.base_url == "https://example.trycloudflare.com"


def test_start_rejects_file_path_with_public_url(tmp_path):
    llm = make_llm()
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(ValueError, match="cannot use public_url"):
        llm.start(
            public_url="https://example.trycloudflare.com",
            file_path=str(pdf),
            show_progress=False,
        )


def test_start_passes_file_path_to_kaggle_launch(tmp_path):
    llm = TokenlessLLM(model=GPT_OSS_MODEL_ID)
    notebook = CapturingNotebook()
    llm._notebook = notebook
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    result = llm.start(file_path=str(pdf), show_progress=False)

    assert result == "converted markdown"
    assert notebook.launch_kwargs["file_path"] == str(pdf)


def test_start_passes_pdf_context_to_kaggle_launch(tmp_path):
    llm = TokenlessLLM(model=GPT_OSS_MODEL_ID)
    notebook = CapturingNotebook()
    llm._notebook = notebook
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    result = llm.start(file_path=str(pdf), pdf_context=True, show_progress=False)

    assert result == "converted markdown"
    assert notebook.launch_kwargs["file_path"] == str(pdf)
    assert notebook.launch_kwargs["pdf_context"] is True


<<<<<<< HEAD
def test_start_waits_for_public_endpoint_readiness(monkeypatch):
    llm = TokenlessLLM(model=GPT_OSS_MODEL_ID)
    llm._notebook = PublicUrlNotebook()
    llm._tunnel = CapturingTunnel("https://ready.trycloudflare.com")
    requests = []

    class Response:
        def raise_for_status(self):
            return None

    def fake_get(url, timeout):
        requests.append((url, timeout))
        return Response()

    monkeypatch.setattr("tokenless.client.requests.get", fake_get)
=======
def test_start_cleans_up_notebook_when_launch_fails():
    llm = TokenlessLLM(model=GPT_OSS_MODEL_ID)
    notebook = FailingNotebook()
    llm._notebook = notebook

    with pytest.raises(requests.HTTPError, match="temporary Kaggle failure"):
        llm.start(show_progress=False)

    assert notebook.stopped is True


def test_start_uses_discovered_tunnel_url():
    llm = TokenlessLLM(model=GPT_OSS_MODEL_ID)
    llm._notebook = PublicUrlNotebook()
    llm._tunnel = CapturingTunnel("https://ready.trycloudflare.com")
>>>>>>> 7f54bf7 (JSON extractor)

    result = llm.start(show_progress=False)

    assert result == "https://ready.trycloudflare.com"
    assert llm.base_url == "https://ready.trycloudflare.com"
<<<<<<< HEAD
    assert llm._tunnel.timeout is None
    assert requests == [("https://ready.trycloudflare.com/api/version", 10)]
=======
    assert llm._tunnel.timeout == 300
>>>>>>> 7f54bf7 (JSON extractor)


def test_chat_uses_long_timeout_for_pdf_page_mapping():
    llm = make_llm()
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)

            class Message:
                content = "ok"

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    llm._running = True
    llm._base_url = "https://ready.trycloudflare.com"
    llm._openai_client = Client()

    assert llm.chat("hello") == "ok"
    assert calls[0]["timeout"] == 3600


def test_pdf_page_mapping_submits_and_polls_async_jobs(monkeypatch):
    llm = make_llm()
    llm._running = True
    llm._pdf_context = True
    llm._base_url = "https://ready.trycloudflare.com"
    llm._openai_client = object()
    calls = []
    statuses = iter(["running", "complete"])

    class Response:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self.data

    def fake_get(url, timeout):
        calls.append(("get", url, timeout))
        if url.endswith("/tokenless/pdf/pages"):
            return Response({"pages": [{"page": 4, "markdown": "page markdown"}]})
        status = next(statuses)
<<<<<<< HEAD
        return Response({"status": status, "content": "page result"})
=======
        return Response(
            {
                "status": status,
                "content": 'Page result:\n```json\n[{"name": "first"}, {"name": "second"}]\n```',
            }
        )
>>>>>>> 7f54bf7 (JSON extractor)

    def fake_post(url, json, timeout):
        calls.append(("post", url, json, timeout))
        return Response({"status": "running"})

    monkeypatch.setattr("tokenless.client.requests.get", fake_get)
    monkeypatch.setattr("tokenless.client.requests.post", fake_post)
    monkeypatch.setattr("tokenless.client.PDF_PAGE_JOB_POLL_INTERVAL", 0)

    result = llm.send("x" * 4000, page_progress=False)

<<<<<<< HEAD
    assert result == "## Page 4\n\npage result"
=======
    assert json.loads(result) == [{"name": "first"}, {"name": "second"}]
>>>>>>> 7f54bf7 (JSON extractor)
    post = next(call for call in calls if call[0] == "post")
    assert post[1].startswith("https://ready.trycloudflare.com/tokenless/jobs/")
    assert post[2]["messages"][0]["content"].startswith("<tokenless_pdf_page_context>")
    assert len([call for call in calls if call[0] == "get"]) == 3


<<<<<<< HEAD
=======
def test_pdf_page_mapping_collects_json_without_printing_page_results(monkeypatch, capsys):
    llm = make_llm()
    llm._running = True
    llm._pdf_context = True
    llm._base_url = "https://ready.trycloudflare.com"
    llm._openai_client = object()
    page_results = iter(
        [
            'Analysis before JSON: {"page": 1, "value": "secret one"}',
            '```json\n[{"page": 2}, {"page": 2, "value": "secret two"}]\n```',
        ]
    )

    monkeypatch.setattr(
        llm,
        "_fetch_pdf_pages",
        lambda: [
            {"page": 1, "markdown": "first page"},
            {"page": 2, "markdown": "second page"},
        ],
    )
    monkeypatch.setattr(llm, "_run_pdf_page_job", lambda *args, **kwargs: next(page_results))

    result = llm.send("x" * 4000)

    assert json.loads(result) == [
        {"page": 1, "value": "secret one"},
        {"page": 2},
        {"page": 2, "value": "secret two"},
    ]
    stderr = capsys.readouterr().err
    assert "secret one" not in stderr
    assert "secret two" not in stderr
    assert "extracted JSON from page 1" in stderr
    assert "extracted JSON from page 2" in stderr


def test_pdf_page_mapping_repairs_malformed_json_with_running_model(monkeypatch, capsys):
    llm = make_llm()
    llm._running = True
    llm._pdf_context = True
    llm._base_url = "https://ready.trycloudflare.com"
    llm._openai_client = object()
    calls = []
    responses = iter(
        [
            '{"units": [{"unit_id": "U1"}]',
            '{"units": [{"unit_id": "U1"}]}',
        ]
    )

    monkeypatch.setattr(
        llm,
        "_fetch_pdf_pages",
        lambda: [{"page": 7, "markdown": "source page"}],
    )

    def fake_page_job(message, *, system_prompt=None, **kwargs):
        calls.append((message, system_prompt, kwargs))
        return next(responses)

    monkeypatch.setattr(llm, "_run_pdf_page_job", fake_page_job)

    result = llm.send("x" * 4000)

    assert json.loads(result) == {"units": [{"unit_id": "U1"}]}
    assert len(calls) == 2
    assert "<tokenless_json_repair_agent>" in calls[1][1]
    assert "Malformed response from repair attempt 0" in calls[1][0]
    assert calls[1][2]["temperature"] == 0
    assert calls[1][2]["max_tokens"] == 16_384
    stderr = capsys.readouterr().err
    assert "repairing malformed JSON from page 7" in stderr
    assert "repaired JSON from page 7" in stderr


def test_pdf_page_mapping_retries_json_repair_and_skips_failed_page(monkeypatch):
    llm = make_llm()
    llm._running = True
    llm._pdf_context = True
    llm._base_url = "https://ready.trycloudflare.com"
    llm._openai_client = object()
    calls = []
    responses = iter(['{"units": [', "still invalid", "also invalid"])

    monkeypatch.setattr(
        llm,
        "_fetch_pdf_pages",
        lambda: [{"page": 8, "markdown": "source page"}],
    )

    def fake_page_job(message, *, system_prompt=None, **kwargs):
        calls.append((message, system_prompt, kwargs))
        return next(responses)

    monkeypatch.setattr(llm, "_run_pdf_page_job", fake_page_job)

    result = llm.send("x" * 4000, page_progress=False, json_repair_attempts=2)

    assert json.loads(result) == []
    assert len(calls) == 3
    assert "Malformed response from repair attempt 1" in calls[2][0]


def test_pdf_page_mapping_can_disable_json_repair(monkeypatch):
    llm = make_llm()
    llm._running = True
    llm._pdf_context = True
    llm._base_url = "https://ready.trycloudflare.com"
    llm._openai_client = object()
    calls = []

    monkeypatch.setattr(
        llm,
        "_fetch_pdf_pages",
        lambda: [{"page": 9, "markdown": "source page"}],
    )

    def fake_page_job(*args, **kwargs):
        calls.append((args, kwargs))
        return '{"units": ['

    monkeypatch.setattr(llm, "_run_pdf_page_job", fake_page_job)

    result = llm.send("x" * 4000, page_progress=False, repair_malformed_json=False)

    assert json.loads(result) == []
    assert len(calls) == 1


def test_extract_json_values_skips_non_json_text_and_handles_nested_values():
    content = (
        'Not JSON {broken}; then {"nested": {"text": "a } brace"}} '
        'and ```json\n[{"id": 1}]\n```'
    )

    assert TokenlessLLM._extract_json_values(content) == [
        {"nested": {"text": "a } brace"}},
        [{"id": 1}],
    ]


def test_extract_json_values_does_not_salvage_nested_fragments_from_truncated_json():
    content = '{"units": [{"unit_id": "U1", "metadata": {"page": 1}}]'

    assert TokenlessLLM._extract_json_values(content) == []


def test_concatenate_json_values_merges_repeated_list_wrappers():
    values = [
        {"units": []},
        {"units": [{"unit_id": "U1"}]},
        {"units": [{"unit_id": "U2"}]},
    ]

    assert TokenlessLLM._concatenate_json_values(values) == {
        "units": [{"unit_id": "U1"}, {"unit_id": "U2"}]
    }


def test_json_shape_hint_keeps_schema_without_copying_values():
    value = {
        "units": [
            {
                "unit_id": "U1",
                "is_exception": False,
                "threshold": {"value": 12.5},
            }
        ]
    }

    assert TokenlessLLM._json_shape_hint(value) == {
        "units": [
            {
                "unit_id": "str",
                "is_exception": "bool",
                "threshold": {"value": "float"},
            }
        ]
    }


>>>>>>> 7f54bf7 (JSON extractor)
def test_pdf_page_job_submit_retry_reuses_job_id(monkeypatch):
    llm = make_llm()
    llm._running = True
    llm._base_url = "https://ready.trycloudflare.com"
    post_urls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "complete", "content": "done"}

    def fake_post(url, json, timeout):
        post_urls.append(url)
        if len(post_urls) == 1:
            raise requests.ConnectionError("temporary tunnel failure")
        return Response()

    monkeypatch.setattr("tokenless.client.requests.post", fake_post)
    monkeypatch.setattr("tokenless.client.requests.get", lambda url, timeout: Response())
    monkeypatch.setattr("tokenless.client.PDF_PAGE_JOB_POLL_INTERVAL", 0)

    assert llm._run_pdf_page_job("prompt") == "done"
    assert len(post_urls) == 2
    assert post_urls[0] == post_urls[1]
