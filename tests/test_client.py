import pytest

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

    result = llm.start(show_progress=False)

    assert result == "https://ready.trycloudflare.com"
    assert llm.base_url == "https://ready.trycloudflare.com"
    assert requests == [("https://ready.trycloudflare.com/api/version", 10)]


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
