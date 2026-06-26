"""
Core TokenlessLLM client — manages notebook lifecycle and exposes an
OpenAI-compatible endpoint backed by Kaggle's free GPUs.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import logging
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import openai
import requests

from tokenless.notebook import (
    DEFAULT_GPT_OSS_KERNEL_SLUG,
    DEFAULT_SMOKE_KERNEL_SLUG,
    GPT_OSS_MODEL_ID,
    KaggleNotebookManager,
)
from tokenless.tunnel import TunnelManager

logger = logging.getLogger(__name__)
PDF_PAGE_MAP_MIN_PROMPT_CHARS = 4000
PDF_PAGE_JOB_POLL_INTERVAL = 2.0
PDF_PAGE_JOB_TIMEOUT = 3600
PDF_DOCUMENT_JOB_STALL_TIMEOUT = 3600
PDF_JSON_REPAIR_MAX_TOKENS = 16_384
DNS_OVER_HTTPS_URL = "https://dns.google/resolve"

_ORIGINAL_GETADDRINFO = getattr(
    socket,
    "_tokenless_original_getaddrinfo",
    socket.getaddrinfo,
)
setattr(socket, "_tokenless_original_getaddrinfo", _ORIGINAL_GETADDRINFO)
_DNS_OVERRIDES: dict[str, tuple[str, ...]] = {}
_DNS_OVERRIDES_LOCK = threading.Lock()


def _tokenless_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    normalized = host.decode("ascii") if isinstance(host, bytes) else str(host)
    with _DNS_OVERRIDES_LOCK:
        addresses = _DNS_OVERRIDES.get(normalized.casefold(), ())
    if not addresses:
        return _ORIGINAL_GETADDRINFO(host, port, family, type, proto, flags)

    results = []
    for address in addresses:
        results.extend(
            _ORIGINAL_GETADDRINFO(address, port, family, type, proto, flags)
        )
    return results


socket.getaddrinfo = _tokenless_getaddrinfo

SUPPORTED_MODELS = [
    "llama3.1-8b",
    "llama3.1-70b",
    "mistral-7b",
    "gemma-2-9b",
    "qwen2.5-7b",
    GPT_OSS_MODEL_ID,
]


class _StartProgress:
    """Tiny dependency-free terminal progress bar for long Kaggle startup."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._message = "Starting"
        self._done = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_at = 0.0
        self._last_width = 0
        self._lock = threading.Lock()

    def __enter__(self):
        if self.enabled:
            self._started_at = time.time()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, *_):
        if not self.enabled:
            return
        self._done.set()
        if self._thread:
            self._thread.join(timeout=1)
        if exc_type is None:
            self.finish("Ready")
        else:
            self.finish("Failed")

    def update(self, message: str) -> None:
        if message:
            self._message = message

    def finish(self, message: str) -> None:
        elapsed = int(time.time() - self._started_at) if self._started_at else 0
        bar = "#" * 24
        self._write(f"Tokenless [{bar}] {message} ({elapsed}s)\n")

    def _run(self) -> None:
        width = 24
        pos = 0
        direction = 1
        while not self._done.is_set():
            elapsed = int(time.time() - self._started_at)
            chars = ["-"] * width
            for offset in range(5):
                idx = pos + offset
                if 0 <= idx < width:
                    chars[idx] = "#"
            self._write(f"Tokenless [{''.join(chars)}] {self._message} ({elapsed}s)")
            pos += direction
            if pos <= 0 or pos >= width - 5:
                direction *= -1
            time.sleep(0.2)

    def _write(self, text: str) -> None:
        with self._lock:
            padding = " " * max(0, self._last_width - len(text))
            sys.stderr.write(f"\r{text}{padding}")
            sys.stderr.flush()
            self._last_width = len(text.rstrip("\n"))


class TokenlessLLM:
    """
    High-level client for running LLM inference on Kaggle's free GPU notebooks.

    Example
    -------
    >>> llm = TokenlessLLM(model="gpt-oss:20b")
    >>> llm.start()
    >>> msg = llm.send("Explain transformers in 3 sentences.")
    >>> print(msg)
    >>> llm.stop()

    For ``gpt-oss:20b``, ``start()`` installs Ollama, pulls the model, starts a
    persistent Kaggle kernel, and exposes an OpenAI-compatible endpoint. ``send()``
    reuses that endpoint for each prompt.

    With ``TOKENLESS_PUBLIC_URL`` set, ``start()`` returns the tunnel base URL and
    ``chat()`` talks to your remote OpenAI-compatible server.
    """

    def __init__(
        self,
        model: str = "llama3.1-8b",
        kaggle_username: Optional[str] = None,
        kaggle_key: Optional[str] = None,
        backend: str = "ollama",          # "ollama" | "vllm"
        keepalive_interval: int = 300,    # seconds between keepalive pings
        verbose: bool = False,
    ):
        if model not in SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model}' not supported. Choose from: {SUPPORTED_MODELS}"
            )

        self.model = model
        self.backend = backend
        self.keepalive_interval = keepalive_interval
        self.verbose = verbose

        kernel_slug = (
            DEFAULT_GPT_OSS_KERNEL_SLUG if model == GPT_OSS_MODEL_ID else DEFAULT_SMOKE_KERNEL_SLUG
        )
        self._notebook = KaggleNotebookManager(
            username=kaggle_username,
            key=kaggle_key,
            model=model,
            backend=backend,
            kernel_slug=kernel_slug,
        )
        self._tunnel = TunnelManager()
        self._base_url: Optional[str] = None
        self._openai_client: Optional[openai.OpenAI] = None
        self._running = False
        self._batch_pdf_file_path: Optional[str] = None
        self._pdf_context = False

    # ------------------------------------------------------------------
    # Lifecycle
    # 
    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        timeout: int = 300,
        *,
        public_url: Optional[str] = None,
        file_path: Optional[str] = None,
        pdf_context: bool = False,
        kaggle_prompt: Optional[str] = None,
        kaggle_system_prompt: Optional[str] = None,
        smoke_status_timeout: int = 1800,
        smoke_poll_interval: float = 3.0,
        smoke_kernel_session_timeout: int = 600,
        gpt_oss_status_timeout: int = 36_000,
        gpt_oss_poll_interval: float = 5.0,
        gpt_oss_kernel_session_timeout: int = 36_000,
        gpt_oss_accelerator: Optional[str] = "NvidiaTeslaT4",
        show_progress: bool = True,
    ) -> str:
        """
        Start remote lifecycle.

        * If ``public_url`` is passed: connect to that existing endpoint and return it.
        * If ``TOKENLESS_PUBLIC_URL`` is set: wait for the tunnel URL and return it;
          ``chat()`` uses that endpoint.
        * If ``file_path`` points to a PDF: upload it as a private Kaggle dataset,
          convert it to Markdown in the Kaggle kernel, and return the Markdown
          directly unless ``kaggle_prompt`` is also passed.
        * Else with Kaggle credentials: run a script kernel — lightweight smoke for
          most models, or a full ``gpt-oss:20b`` Ollama batch job when ``model`` is
          ``gpt-oss:20b``. The returned string is stored in the notebook manager and
          returned here (smoke marker or model reply). ``chat()`` is unavailable until
          you set ``TOKENLESS_PUBLIC_URL`` and start again.
        """
        logger.info("Starting Kaggle notebook with model '%s'...", self.model)
        progress = _StartProgress(enabled=show_progress)
        progress.update(f"Starting Kaggle notebook for {self.model}")
        progress.__enter__()
        try:
            return self._start(
                timeout=timeout,
                public_url=public_url,
                file_path=file_path,
                pdf_context=pdf_context,
                kaggle_prompt=kaggle_prompt,
                kaggle_system_prompt=kaggle_system_prompt,
                smoke_status_timeout=smoke_status_timeout,
                smoke_poll_interval=smoke_poll_interval,
                smoke_kernel_session_timeout=smoke_kernel_session_timeout,
                gpt_oss_status_timeout=gpt_oss_status_timeout,
                gpt_oss_poll_interval=gpt_oss_poll_interval,
                gpt_oss_kernel_session_timeout=gpt_oss_kernel_session_timeout,
                gpt_oss_accelerator=gpt_oss_accelerator,
                progress_callback=progress.update,
            )
        except BaseException:
            logger.exception("Error occurred while starting the notebook")
            self._notebook.stop()
            self._tunnel.close()
            self._base_url = None
            self._openai_client = None
            self._running = False
            self._pdf_context = False
            raise
        finally:
            progress.__exit__(*sys.exc_info())

    def _start(
        self,
        timeout: int = 300,
        *,
        public_url: Optional[str] = None,
        file_path: Optional[str] = None,
        pdf_context: bool = False,
        kaggle_prompt: Optional[str] = None,
        kaggle_system_prompt: Optional[str] = None,
        smoke_status_timeout: int = 1800,
        smoke_poll_interval: float = 3.0,
        smoke_kernel_session_timeout: int = 600,
        gpt_oss_status_timeout: int = 36_000,
        gpt_oss_poll_interval: float = 5.0,
        gpt_oss_kernel_session_timeout: int = 36_000,
        gpt_oss_accelerator: Optional[str] = "NvidiaTeslaT4",
        progress_callback=None,
    ) -> str:
        if public_url and file_path:
            raise ValueError("file_path PDF conversion runs on Kaggle and cannot use public_url.")
        self._pdf_context = False
        if public_url:
            if progress_callback:
                progress_callback("Connecting to existing endpoint")
            self._base_url = public_url.rstrip("/")
            self._openai_client = openai.OpenAI(
                base_url=f"{self._base_url}/v1",
                api_key="kaggle-free",
                timeout=3600,
            )
            self._running = True
            logger.info("Connected to existing endpoint: %s", self._base_url)
            return self._base_url

        self._notebook.launch(
            file_path=file_path,
            pdf_context=pdf_context,
            kaggle_prompt=kaggle_prompt,
            kaggle_system_prompt=kaggle_system_prompt,
            smoke_status_timeout=smoke_status_timeout,
            smoke_poll_interval=smoke_poll_interval,
            smoke_kernel_session_timeout=smoke_kernel_session_timeout,
            gpt_oss_status_timeout=gpt_oss_status_timeout,
            gpt_oss_poll_interval=gpt_oss_poll_interval,
            gpt_oss_kernel_session_timeout=gpt_oss_kernel_session_timeout,
            gpt_oss_accelerator=gpt_oss_accelerator,
            progress_callback=progress_callback,
        )
        self._pdf_context = pdf_context

        if self._notebook.public_url:
            logger.info("Waiting for tunnel URL (timeout=%ds)...", timeout)
            if progress_callback:
                progress_callback("Connecting to public endpoint")
            self._base_url = self._tunnel.get_url(timeout=timeout)
            self._wait_for_endpoint_ready(
                self._base_url,
                timeout=timeout,
                progress_callback=progress_callback,
            )
            self._openai_client = openai.OpenAI(
                base_url=f"{self._base_url}/v1",
                api_key="kaggle-free",  # dummy key — no auth needed
            )
            self._running = True
            logger.info("Endpoint ready: %s", self._base_url)
            return self._base_url

        if self._notebook.smoke_test_message is not None:
            self._base_url = None
            self._openai_client = None
            self._running = True
            msg = self._notebook.smoke_test_message
            logger.info("Kaggle script lifecycle ready (no inference URL configured).")
            return msg

        raise RuntimeError(
            "Notebook launch produced neither TOKENLESS_PUBLIC_URL nor a script kernel message."
        )

    def stop(self):
        """Tear down the Kaggle notebook and tunnel."""
        if self._running:
            self._notebook.stop()
            self._tunnel.close()
            self._running = False
            self._pdf_context = False
            logger.info("Notebook stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def chat(self, message: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Single-turn chat. Returns the assistant reply as a string."""
        self._assert_inference()
        kwargs.setdefault("timeout", 3600)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    def send(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        *,
        file_path: Optional[str] = None,
        file_upload_timeout: int = 3600,
        gpt_oss_status_timeout: int = 36_000,
        gpt_oss_poll_interval: float = 5.0,
        gpt_oss_kernel_session_timeout: int = 36_000,
        gpt_oss_accelerator: Optional[str] = "NvidiaTeslaT4",
        page_progress: bool = True,
        repair_malformed_json: bool = True,
        json_repair_attempts: int = 2,
        **kwargs,
    ) -> str:
        """
        Send one prompt and return the assistant reply.

        In endpoint mode this delegates to ``chat()``. For ``gpt-oss:20b``,
        ``start()`` creates the endpoint so repeated ``send()`` calls reuse the same
        Kaggle kernel and downloaded model. Passing ``file_path`` uploads a PDF to
        the running endpoint and replaces any previous PDF context before inference.
        """
        self._assert_running()
        if self._openai_client is not None and self._base_url is not None:
            if file_path is not None:
                self._upload_pdf_context(file_path, timeout=file_upload_timeout)
            if self._pdf_context and len(message) >= PDF_PAGE_MAP_MIN_PROMPT_CHARS:
                return self._send_pdf_page_mapped(
                    message,
                    system_prompt=system_prompt,
                    page_progress=page_progress,
                    repair_malformed_json=repair_malformed_json,
                    json_repair_attempts=json_repair_attempts,
                    **kwargs,
                )
            return self.chat(message, system_prompt=system_prompt, **kwargs)

        if self.model == GPT_OSS_MODEL_ID:
            if file_path is not None:
                raise RuntimeError(
                    "send(file_path=...) requires a running Tokenless endpoint. "
                    "Call start() without file_path first."
                )
            if kwargs:
                unsupported = ", ".join(sorted(kwargs))
                raise TypeError(
                    "Unsupported keyword argument(s) for gpt-oss:20b batch send: "
                    f"{unsupported}"
                )
            return self._notebook.run_ollama_gpt_oss_20b_prompt(
                message,
                system_prompt=system_prompt or "You are a helpful AI assistant.",
                status_timeout=gpt_oss_status_timeout,
                poll_interval=gpt_oss_poll_interval,
                kernel_session_timeout=gpt_oss_kernel_session_timeout,
                accelerator=gpt_oss_accelerator,
            )

        raise RuntimeError(
            "send() needs an OpenAI-compatible inference endpoint for this model. "
            "Set TOKENLESS_PUBLIC_URL and call .start(), or use model='gpt-oss:20b' "
            "for Kaggle batch prompts."
        )

    def chat_stream(self, message: str, system_prompt: Optional[str] = None, **kwargs):
        """Streaming chat — yields text chunks as they arrive."""
        self._assert_inference()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        stream = self._openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @property
    def openai_client(self) -> openai.OpenAI:
        """Raw OpenAI client pointed at the Kaggle endpoint."""
        self._assert_inference()
        return self._openai_client

    @property
    def base_url(self) -> str:
        self._assert_inference()
        return self._base_url

    # ------------------------------------------------------------------
    # Framework integrations
    # ------------------------------------------------------------------

    def as_strands_model(self, **kwargs):
        """Return a Strands-compatible model provider backed by this endpoint."""
        self._assert_inference()
        from tokenless.providers.strands import TokenlessStrandsModel

        return TokenlessStrandsModel(base_url=self._base_url, model=self.model, **kwargs)

    def as_langchain_llm(self, **kwargs):
        """Return a LangChain ChatOpenAI instance backed by this endpoint."""
        self._assert_inference()
        from tokenless.providers.langchain import TokenlessLangChainLLM

        return TokenlessLangChainLLM(base_url=self._base_url, model=self.model, **kwargs)

    def as_agents_model(self, **kwargs):
        """Return an OpenAI Agents SDK model backed by this endpoint."""
        self._assert_inference()
        from tokenless.providers.openai_agents import TokenlessAgentsModel

        return TokenlessAgentsModel(base_url=self._base_url, model=self.model, **kwargs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _wait_for_endpoint_ready(
        base_url: str,
        *,
        timeout: int,
        progress_callback=None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        url = f"{base_url.rstrip('/')}/api/version"
        hostname = urlparse(base_url).hostname
        deadline = time.time() + timeout
        last_error = None
        last_doh_attempt = 0.0
        while time.time() < deadline:
            if progress_callback:
                progress_callback("Waiting for endpoint DNS and health check")
            try:
                response = requests.get(
                    url,
                    timeout=min(10, max(1, deadline - time.time())),
                )
                response.raise_for_status()
                return
            except requests.RequestException as exc:
                last_error = exc
                now = time.time()
                if hostname and now - last_doh_attempt >= 15:
                    last_doh_attempt = now
                    if progress_callback:
                        progress_callback("Resolving endpoint via DNS-over-HTTPS")
                    try:
                        TokenlessLLM._register_doh_override(hostname)
                    except requests.RequestException as doh_exc:
                        logger.debug(
                            "DNS-over-HTTPS lookup failed for %s: %r",
                            hostname,
                            doh_exc,
                        )
                time.sleep(PDF_PAGE_JOB_POLL_INTERVAL)

        raise TimeoutError(
            f"Timed out after {timeout}s waiting for endpoint {base_url}. "
            f"Last error: {last_error!r}"
        )

    @staticmethod
    def _register_doh_override(hostname: str) -> tuple[str, ...]:
        response = requests.get(
            DNS_OVER_HTTPS_URL,
            params={"name": hostname, "type": "A"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        addresses = []
        for answer in data.get("Answer") or []:
            value = str(answer.get("data") or "")
            try:
                address = ipaddress.ip_address(value)
            except ValueError:
                continue
            if address.version == 4:
                addresses.append(str(address))

        if not addresses:
            raise requests.ConnectionError(
                f"DNS-over-HTTPS returned no IPv4 address for {hostname}: {data!r}"
            )

        unique_addresses = tuple(dict.fromkeys(addresses))
        with _DNS_OVERRIDES_LOCK:
            _DNS_OVERRIDES[hostname.casefold()] = unique_addresses
        logger.info(
            "Using DNS-over-HTTPS addresses for %s: %s",
            hostname,
            ", ".join(unique_addresses),
        )
        return unique_addresses

    def _upload_pdf_context(self, file_path: str, *, timeout: int) -> None:
        self._assert_inference()
        if timeout <= 0:
            raise ValueError("file_upload_timeout must be greater than zero.")

        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Input path must be a file: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported by file_path right now.")

        job_path = f"/tokenless/pdf/upload/{uuid.uuid4().hex}"
        deadline = time.time() + timeout
        self._pdf_context = False
        encoded_filename = base64.b64encode(path.name.encode("utf-8")).decode("ascii")

        def submit_upload() -> requests.Response:
            with path.open("rb") as pdf_file:
                return requests.post(
                    f"{self._base_url.rstrip('/')}{job_path}",
                    data=pdf_file,
                    headers={
                        "Content-Type": "application/pdf",
                        "X-Tokenless-Filename-B64": encoded_filename,
                    },
                    timeout=min(300, max(1, deadline - time.time())),
                )

        response = self._retry_pdf_job_request(
            "PDF upload",
            deadline,
            submit_upload,
        )
        data = response.json()
        while time.time() < deadline:
            status = data.get("status")
            if status == "complete":
                self._pdf_context = True
                return
            if status == "error":
                raise RuntimeError(f"PDF upload failed: {data.get('error')}")
            if status != "running":
                raise RuntimeError(
                    "Unexpected PDF upload response from "
                    f"{self._base_url.rstrip('/')}{job_path}: {data!r}"
                )
            time.sleep(PDF_PAGE_JOB_POLL_INTERVAL)
            response = self._retry_pdf_job_request(
                "PDF conversion poll",
                deadline,
                lambda: requests.get(
                    f"{self._base_url.rstrip('/')}{job_path}",
                    timeout=60,
                ),
            )
            data = response.json()

        raise TimeoutError(f"Timed out after {timeout}s replacing the PDF context.")

    def _send_pdf_page_mapped(
        self,
        message: str,
        *,
        system_prompt: Optional[str] = None,
        page_progress: bool = True,
        repair_malformed_json: bool = True,
        json_repair_attempts: int = 2,
        **kwargs,
    ) -> str:
        if json_repair_attempts < 0:
            raise ValueError("json_repair_attempts must be zero or greater.")

        document_kwargs = dict(kwargs)
        document_kwargs.setdefault("max_tokens", PDF_JSON_REPAIR_MAX_TOKENS)
        pages = self._run_pdf_document_job(
            message,
            system_prompt=system_prompt,
            page_progress=page_progress,
            repair_malformed_json=repair_malformed_json,
            json_repair_attempts=json_repair_attempts,
            **document_kwargs,
        )
        outputs: list[object] = []
        total = len(pages)
        for index, page in enumerate(pages, start=1):
            page_number = int(page.get("page") or index)
            page_outputs = page.get("values") or []
            if not isinstance(page_outputs, list):
                raise RuntimeError(
                    f"Unexpected JSON values for PDF page {page_number}: {page!r}"
                )
            outputs.extend(page_outputs)
            if page_progress:
                status = (
                    "repaired JSON"
                    if page.get("repaired")
                    else "extracted JSON"
                    if page_outputs
                    else "no JSON result"
                )
                sys.stderr.write(
                    f"Tokenless PDF page {index}/{total}: "
                    f"{status} from page {page_number}\n"
                )
                sys.stderr.flush()

        return json.dumps(self._concatenate_json_values(outputs), ensure_ascii=False)

    def _repair_pdf_page_json(
        self,
        original_message: str,
        *,
        malformed_content: str,
        page_system_prompt: str,
        schema_hint: Optional[object],
        attempts: int,
        **kwargs,
    ) -> list[object]:
        repair_kwargs = dict(kwargs)
        repair_kwargs["temperature"] = 0
        max_tokens = repair_kwargs.get("max_tokens")
        if not isinstance(max_tokens, int) or max_tokens < PDF_JSON_REPAIR_MAX_TOKENS:
            repair_kwargs["max_tokens"] = PDF_JSON_REPAIR_MAX_TOKENS

        schema_text = (
            json.dumps(self._json_shape_hint(schema_hint), ensure_ascii=False)
            if schema_hint is not None
            else "No earlier valid page response is available."
        )
        repair_system_prompt = (
            f"{page_system_prompt}\n\n"
            "<tokenless_json_repair_agent>\n"
            "You are a JSON repair agent. Re-read the PDF page context and regenerate "
            "the answer to the original extraction request as one complete valid JSON "
            "object or array. Preserve all recoverable facts, match the earlier valid "
            "response schema when applicable, and do not return Markdown fences, "
            "analysis, comments, or trailing text.\n"
            "</tokenless_json_repair_agent>"
        )

        candidate = malformed_content
        for attempt in range(1, attempts + 1):
            repair_message = (
                "Original extraction request:\n"
                f"{original_message}\n\n"
                "Earlier valid page response to use only as a schema hint:\n"
                f"{schema_text}\n\n"
                f"Malformed response from repair attempt {attempt - 1}:\n"
                f"{candidate}\n\n"
                "Return the corrected, complete JSON now."
            )
            candidate = self._run_pdf_page_job(
                repair_message,
                system_prompt=repair_system_prompt,
                **repair_kwargs,
            ).strip()
            repaired_values = self._extract_json_values(candidate)
            if repaired_values:
                return repaired_values

        return []

    @classmethod
    def _json_shape_hint(cls, value: object, depth: int = 0) -> object:
        if depth >= 4:
            return type(value).__name__
        if isinstance(value, dict):
            return {
                key: cls._json_shape_hint(item, depth + 1)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._json_shape_hint(value[0], depth + 1)] if value else []
        if value is None:
            return None
        return type(value).__name__

    @staticmethod
    def _extract_json_values(content: str) -> list[object]:
        """Extract JSON objects and arrays from model text, including fenced JSON."""
        values: list[object] = []
        start: Optional[int] = None
        stack: list[str] = []
        in_string = False
        escaped = False
        matching = {"}": "{", "]": "["}

        for index, char in enumerate(content):
            if start is None:
                if char in "{[":
                    start = index
                    stack = [char]
                continue

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                if not stack or stack[-1] != matching[char]:
                    start = None
                    stack = []
                    continue
                stack.pop()
                if not stack:
                    candidate = content[start : index + 1]
                    try:
                        value = json.loads(candidate)
                    except json.JSONDecodeError:
                        pass
                    else:
                        if isinstance(value, (dict, list)):
                            values.append(value)
                    start = None

        return values

    @staticmethod
    def _concatenate_json_values(values: list[object]) -> object:
        """Join repeated list-valued JSON wrappers, or return one flattened array."""
        if values and all(isinstance(value, dict) for value in values):
            first_keys = set(values[0])
            if first_keys and all(
                set(value) == first_keys
                and all(isinstance(item, list) for item in value.values())
                for value in values
            ):
                merged = {key: [] for key in values[0]}
                for value in values:
                    for key, items in value.items():
                        merged[key].extend(items)
                return merged

        merged_values: list[object] = []
        for value in values:
            if isinstance(value, list):
                merged_values.extend(value)
            else:
                merged_values.append(value)
        return merged_values

    def _run_pdf_document_job(
        self,
        message: str,
        *,
        system_prompt: Optional[str] = None,
        page_progress: bool,
        repair_malformed_json: bool,
        json_repair_attempts: int,
        **kwargs,
    ) -> list[dict]:
        job_id = uuid.uuid4().hex
        job_path = f"/tokenless/jobs/{job_id}"
        stall_timeout = kwargs.pop("timeout", PDF_DOCUMENT_JOB_STALL_TIMEOUT)
        if stall_timeout <= 0:
            raise ValueError("PDF document job timeout must be greater than zero.")
        deadline = time.time() + stall_timeout
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        payload = {
            "model": self.model,
            "messages": messages,
            **kwargs,
            "tokenless_pdf_map": {
                "repair_malformed_json": repair_malformed_json,
                "json_repair_attempts": json_repair_attempts,
            },
        }

        self._retry_pdf_job_request(
            "submit PDF document job",
            deadline,
            lambda: requests.post(
                f"{self._base_url.rstrip('/')}{job_path}",
                json=payload,
                timeout=60,
            ),
        )
        last_progress = None
        while True:
            response = self._retry_pdf_job_request(
                "poll PDF document job",
                deadline,
                lambda: requests.get(
                    f"{self._base_url.rstrip('/')}{job_path}",
                    timeout=60,
                ),
            )
            data = response.json()
            status = data.get("status")
            if status == "complete":
                pages = data.get("pages")
                if not isinstance(pages, list):
                    raise RuntimeError(
                        "Unexpected PDF document job response from "
                        f"{self._base_url.rstrip('/')}{job_path}: {data!r}"
                    )
                return pages
            if status == "error":
                raise RuntimeError(f"PDF document job {job_id} failed: {data.get('error')}")
            if status != "running":
                raise RuntimeError(
                    "Unexpected PDF document job response from "
                    f"{self._base_url.rstrip('/')}{job_path}: {data!r}"
                )

            progress = (
                data.get("current_page"),
                data.get("total_pages"),
                data.get("page_number"),
            )
            if progress != last_progress and progress[0]:
                deadline = time.time() + stall_timeout
            if page_progress and progress != last_progress and progress[0]:
                sys.stderr.write(
                    f"Tokenless PDF page {progress[0]}/{progress[1]}: "
                    f"processing page {progress[2]}\n"
                )
                sys.stderr.flush()
            last_progress = progress
            time.sleep(PDF_PAGE_JOB_POLL_INTERVAL)

            if time.time() >= deadline:
                raise TimeoutError(
                    f"PDF document job {job_id} made no page progress for "
                    f"{stall_timeout}s."
                )

    def _run_pdf_page_job(
        self,
        message: str,
        *,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        job_id = uuid.uuid4().hex
        job_path = f"/tokenless/jobs/{job_id}"
        timeout = kwargs.pop("timeout", PDF_PAGE_JOB_TIMEOUT)
        deadline = time.time() + timeout
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        payload = {"model": self.model, "messages": messages, **kwargs}

        self._retry_pdf_job_request(
            "submit",
            deadline,
            lambda: requests.post(
                f"{self._base_url.rstrip('/')}{job_path}",
                json=payload,
                timeout=60,
            ),
        )
        while time.time() < deadline:
            response = self._retry_pdf_job_request(
                "poll",
                deadline,
                lambda: requests.get(
                    f"{self._base_url.rstrip('/')}{job_path}",
                    timeout=60,
                ),
            )
            data = response.json()
            status = data.get("status")
            if status == "complete":
                return str(data.get("content") or "")
            if status == "error":
                raise RuntimeError(f"PDF page job {job_id} failed: {data.get('error')}")
            if status != "running":
                raise RuntimeError(
                    "Unexpected PDF page job response from "
                    f"{self._base_url.rstrip('/')}{job_path}: {data!r}"
                )
            time.sleep(PDF_PAGE_JOB_POLL_INTERVAL)
        raise TimeoutError(f"Timed out after {timeout}s waiting for PDF page job {job_id}.")

    def _retry_pdf_job_request(
        self,
        action: str,
        deadline: float,
        request,
    ) -> requests.Response:
        last_error = None
        last_recovery_attempt = 0.0
        while time.time() < deadline:
            try:
                response = request()
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                last_error = e
                now = time.time()
                if now - last_recovery_attempt >= 15:
                    last_recovery_attempt = now
                    self._recover_public_endpoint()
                time.sleep(PDF_PAGE_JOB_POLL_INTERVAL)
        raise TimeoutError(f"Timed out during PDF page job {action}. Last error: {last_error!r}")

    def _recover_public_endpoint(self) -> bool:
        current_url = (self._base_url or "").rstrip("/")
        monitored_url = str(getattr(self._notebook, "public_url", "") or "").rstrip("/")
        if monitored_url and monitored_url != current_url:
            latest_url = monitored_url
        else:
            latest_url = None
        latest_url_reader = getattr(self._notebook, "latest_public_url", None)
        if latest_url is None and latest_url_reader is not None:
            latest_url = latest_url_reader()
        if not latest_url:
            return False

        hostname = urlparse(latest_url).hostname
        if hostname:
            try:
                self._register_doh_override(hostname)
            except requests.RequestException:
                pass
        if latest_url.rstrip("/") == current_url:
            return False

        try:
            self._wait_for_endpoint_ready(latest_url, timeout=120)
        except TimeoutError:
            return False

        self._base_url = latest_url.rstrip("/")
        self._openai_client = openai.OpenAI(
            base_url=f"{self._base_url}/v1",
            api_key="kaggle-free",
            timeout=3600,
        )
        logger.info("Recovered Tokenless public endpoint: %s", self._base_url)
        sys.stderr.write(f"Tokenless recovered public endpoint: {self._base_url}\n")
        sys.stderr.flush()
        return True

    def _fetch_pdf_pages(self) -> list[dict]:
        self._assert_inference()
        url = f"{self._base_url.rstrip('/')}/tokenless/pdf/pages"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        pages = data.get("pages")
        if not isinstance(pages, list):
            raise RuntimeError(f"Unexpected PDF pages response from {url}: {data!r}")
        return pages

    @staticmethod
    def _pdf_page_system_prompt(
        page_number: int,
        total_pages: int,
        page_markdown: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> str:
        context = (
            "<tokenless_pdf_page_context>\n"
            "You are processing one page from an uploaded PDF. Apply the user's prompt "
            "only to this page. Return only findings supported by this page. If this "
            "page has no relevant answer, return an empty response.\n\n"
            f"PDF page {page_number} of {total_pages}:\n\n"
            f"<pdf_markdown>\n{page_markdown}\n</pdf_markdown>\n"
            "</tokenless_pdf_page_context>"
        )
        if system_prompt:
            return f"{system_prompt}\n\n{context}"
        return context

    def _assert_running(self):
        if not self._running:
            raise RuntimeError("TokenlessLLM is not running. Call .start() first.")

    def _assert_inference(self):
        self._assert_running()
        if self._openai_client is None or self._base_url is None:
            raise RuntimeError(
                "OpenAI-compatible inference is not available. Only a Kaggle script kernel "
                "result is available (smoke test or batch reply). Set TOKENLESS_PUBLIC_URL "
                "to your notebook tunnel base URL (no /v1 suffix), then call .start() again."
            )
