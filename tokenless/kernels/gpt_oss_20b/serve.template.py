# Long-running Kaggle script kernel for gpt-oss:20b via Ollama.
import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)

OLLAMA_MODEL = "gpt-oss:20b"
NTFY_TOPIC = "__TOKENLESS_NTFY_TOPIC__"
INPUT_DATASET_SLUG_B64 = "__TOKENLESS_INPUT_DATASET_SLUG_B64__"
INPUT_FILENAME_B64 = "__TOKENLESS_INPUT_FILENAME_B64__"

OUT = Path("/kaggle/working/tokenless_public_url.txt")
ERR = Path("/kaggle/working/tokenless_server_error.txt")
CLOUDFLARED_LOG = Path("/tmp/tokenless_cloudflared.log")
PDF_MARKDOWN = Path("/kaggle/working/tokenless_pdf_context.md")
PAGE_MAP_MIN_PROMPT_CHARS = 4000
PAGE_MAP_REQUEST_TIMEOUT = 3600
PAGE_JOBS = {}
PAGE_JOBS_LOCK = threading.Lock()


def _fail(msg: str) -> None:
    ERR.write_text(msg, encoding="utf-8")
    raise SystemExit(1)


def _run(cmd: str, error: str) -> None:
    if os.system(cmd) != 0:
        _fail(error)


def _publish(message: str) -> None:
    if not NTFY_TOPIC or NTFY_TOPIC.startswith("__TOKENLESS_"):
<<<<<<< HEAD
=======
        print(f"NTFY_TOPIC is not set; skipping publish of message: {message}", flush=True)
>>>>>>> 7f54bf7 (JSON extractor)
        return
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10).read()
    except (urllib.error.URLError, TimeoutError):
<<<<<<< HEAD
        pass
=======
        print(f"Failed to publish message to NTFY: {message}", flush=True)
>>>>>>> 7f54bf7 (JSON extractor)


def _decode_b64(value: str) -> str:
    if not value:
        return ""
    return base64.b64decode(value).decode("utf-8")


def _clean_source_label(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    return value.replace("|", "/") if value else "unknown"


def _first_text_line(block: str) -> str:
    for line in block.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _update_source_context(block: str, section: str) -> str:
    first_line = _first_text_line(block)
    heading = re.match(r"^#{1,6}\s+(.+?)\s*#*$", first_line)
    label = heading.group(1).strip() if heading else first_line
    section_match = re.match(
        r"^(?:section|sec\.|§)\s+([A-Za-z0-9IVXLCDMivxlcdm_.-]+)(?:\s*[:.-]\s*(.+))?$",
        label,
        flags=re.IGNORECASE,
    )
    if heading:
        section = label
    if section_match:
        section = label
    return section


def _chunk_page_number(chunk: dict, fallback: int) -> int:
    metadata = chunk.get("metadata") or {}
    for key in ("page", "page_number", "page_index"):
        value = chunk.get(key, metadata.get(key))
        if isinstance(value, int):
            return value + 1 if key == "page_index" else value
    return fallback


def _chunk_text(chunk: dict) -> str:
    for key in ("text", "markdown", "content"):
        value = chunk.get(key)
        if isinstance(value, str):
            return value
    return ""


def _chunk_section_hint(chunk: dict) -> str:
    toc_items = chunk.get("toc_items") or []
    if not toc_items:
        return ""
    item = sorted(toc_items, key=lambda entry: entry[0] if entry else 0)[-1]
    return str(item[1]).strip() if len(item) > 1 else ""


def _add_pdf_source_markers(page_chunks) -> str:
    if isinstance(page_chunks, str):
        page_chunks = [{"text": page_chunks, "page": 1}]

    output = []
    current_section = "unknown"
    for fallback_page, chunk in enumerate(page_chunks, start=1):
        if not isinstance(chunk, dict):
            continue
        page_number = _chunk_page_number(chunk, fallback_page)
        section_hint = _chunk_section_hint(chunk)
        if section_hint:
            current_section = section_hint
        output.append(f"<!-- tokenless-page: page={page_number} -->")

        text = _chunk_text(chunk).strip()
        for block in re.split(r"\n\s*\n", text):
            block = block.strip()
            if not block:
                continue
            current_section = _update_source_context(
                block,
                current_section,
            )
            output.append(
                "[source: "
                f"page={page_number} | "
                f"section={_clean_source_label(current_section)}"
                "]"
            )
            output.append(block)

    return "\n\n".join(output).strip()


def _install_pdf_tools() -> None:
    print("Installing PDF conversion tools...", flush=True)
    _publish("Installing PDF conversion tools")
    _run(
        "apt-get update -qq && apt-get install -y -qq tesseract-ocr ghostscript qpdf unpaper",
        "Failed to install OCR system dependencies.",
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "ocrmypdf", "pymupdf4llm"]
    )


def _find_pdf_path(dataset_slug: str, filename: str) -> Path:
    input_root = Path("/kaggle/input")
    dataset_dirs = [input_root / dataset_slug]
    dataset_dirs.extend(
        p
        for p in sorted(input_root.glob(f"**/{dataset_slug}"))
        if p.is_dir() and p not in dataset_dirs
    )

    expected_paths = [dataset_dir / filename for dataset_dir in dataset_dirs]
    print("Looking for uploaded PDF at", expected_paths[0], flush=True)
    for expected in expected_paths:
        if expected.is_file():
            if expected != expected_paths[0]:
                print("Using uploaded PDF at", expected, flush=True)
            return expected
    candidates = sorted(
        p
        for dataset_dir in dataset_dirs
        for p in dataset_dir.glob("**/*")
        if p.is_file() and p.suffix.lower() == ".pdf"
    )
    if len(candidates) == 1:
        print("Using discovered uploaded PDF at", candidates[0], flush=True)
        return candidates[0]
    available = sorted(str(p) for p in input_root.glob("**/*"))
    _fail(
        f"Uploaded PDF was not found at any of {expected_paths}. "
        f"Available Kaggle input paths: {available[:50]}"
    )


def _pdf_to_markdown(pdf_path: Path) -> str:
    import ocrmypdf  # noqa: E402
    import pymupdf4llm  # noqa: E402

    ocr_pdf = Path("/kaggle/working/tokenless_input_ocr.pdf")
    print("Running OCR pass for image and mixed-content PDFs...", flush=True)
    try:
        ocrmypdf.ocr(
            str(pdf_path),
            str(ocr_pdf),
            skip_text=True,
            deskew=True,
            progress_bar=False,
        )
        source = ocr_pdf
    except ocrmypdf.exceptions.PriorOcrFoundError:
        source = pdf_path
    except Exception as e:  # noqa: BLE001 - fall back for already-readable PDFs
        print(f"OCR pass failed; trying direct Markdown conversion: {e!r}", flush=True)
        source = pdf_path

    print("Converting PDF to source-aware Markdown...", flush=True)
    try:
        page_chunks = pymupdf4llm.to_markdown(str(source), page_chunks=True)
    except TypeError:
        page_chunks = pymupdf4llm.to_markdown(str(source))
    markdown = _add_pdf_source_markers(page_chunks)
    if not markdown.strip():
        _fail("PDF conversion produced empty Markdown.")
    PDF_MARKDOWN.write_text(markdown, encoding="utf-8")
    return markdown


def _prepare_pdf_context() -> None:
    dataset_slug = _decode_b64(INPUT_DATASET_SLUG_B64)
    filename = _decode_b64(INPUT_FILENAME_B64)
    if not dataset_slug or not filename:
        return
    _install_pdf_tools()
    markdown = _pdf_to_markdown(_find_pdf_path(dataset_slug, filename))
    print(f"Stored PDF Markdown context ({len(markdown)} chars).", flush=True)
    _publish("PDF context ready")


def _inject_pdf_context(payload: dict) -> dict:
    if not PDF_MARKDOWN.exists():
        return payload
    markdown = PDF_MARKDOWN.read_text(encoding="utf-8")
    messages = list(payload.get("messages") or [])
    if any("<tokenless_pdf_page_context>" in str(message.get("content") or "") for message in messages):
        return payload
    question = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            question = str(message.get("content") or "")
            break
    selected_context = _select_pdf_context(markdown, question)
    context = (
        "You are answering questions about an uploaded PDF. Use the source-aware "
<<<<<<< HEAD
        "Markdown context below as the source of truth. Each block has page and "
        "section source markers. If the answer is not in the PDF, say you cannot "
=======
        "Markdown context below as the source of truth. Each block has page and section "
        "source markers. If the answer is not in the PDF, say you cannot "
>>>>>>> 7f54bf7 (JSON extractor)
        "find it in the document.\n\n"
        f"<pdf_markdown>\n{selected_context}\n</pdf_markdown>"
    )
    if messages and messages[0].get("role") == "system":
        messages[0] = {**messages[0], "content": f"{messages[0].get('content', '')}\n\n{context}"}
    else:
        messages.insert(0, {"role": "system", "content": context})
    return {**payload, "messages": messages}


def _last_user_message(payload: dict) -> str:
    for message in reversed(payload.get("messages") or []):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _should_map_pdf_pages(payload: dict) -> bool:
    return False


def _split_pdf_pages(markdown: str) -> list[tuple[int, str]]:
    pages = []
    for chunk in re.split(r"(?=<!-- tokenless-page: page=\d+ -->)", markdown):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = re.match(r"<!-- tokenless-page: page=(\d+) -->", chunk)
        page_number = int(match.group(1)) if match else len(pages) + 1
        pages.append((page_number, chunk))
    return pages


def _payload_for_pdf_page(payload: dict, page_number: int, page_markdown: str, total_pages: int) -> dict:
    messages = list(payload.get("messages") or [])
    context = (
        "You are processing one page from an uploaded PDF. Apply the user's prompt "
        "only to this page. If the prompt asks for extraction or analysis, return "
        "only findings supported by this page. If this page has no relevant answer, "
        "return an empty response.\n\n"
        f"PDF page {page_number} of {total_pages}:\n\n"
        f"<pdf_markdown>\n{page_markdown}\n</pdf_markdown>"
    )
    if messages and messages[0].get("role") == "system":
        messages[0] = {**messages[0], "content": f"{messages[0].get('content', '')}\n\n{context}"}
    else:
        messages.insert(0, {"role": "system", "content": context})
    return {**payload, "messages": messages, "stream": False}


def _post_ollama_chat(payload: dict, timeout: int = PAGE_MAP_REQUEST_TIMEOUT) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _assistant_content(response_payload: dict) -> str:
    choices = response_payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip()


def _run_page_job(job_id: str, payload: dict) -> None:
    try:
        content = _assistant_content(_post_ollama_chat(payload))
        update = {"status": "complete", "content": content}
    except Exception as e:  # noqa: BLE001 - return remote inference failures to the client
        update = {"status": "error", "error": repr(e)}
    with PAGE_JOBS_LOCK:
        PAGE_JOBS[job_id] = update


def _start_page_job(job_id: str, payload: dict) -> dict:
    with PAGE_JOBS_LOCK:
        job = PAGE_JOBS.get(job_id)
        if job is not None:
            return job
        PAGE_JOBS[job_id] = {"status": "running"}
    threading.Thread(target=_run_page_job, args=(job_id, payload), daemon=True).start()
    return {"status": "running"}


def _get_page_job(job_id: str) -> dict | None:
    with PAGE_JOBS_LOCK:
        job = PAGE_JOBS.get(job_id)
        return dict(job) if job is not None else None


def _map_pdf_pages(payload: dict) -> dict:
    markdown = PDF_MARKDOWN.read_text(encoding="utf-8")
    pages = _split_pdf_pages(markdown)
    outputs = []
    last_response = {
        "id": "chatcmpl-tokenless-page-map",
        "object": "chat.completion",
        "model": payload.get("model", OLLAMA_MODEL),
        "choices": [],
    }
    for index, (page_number, page_markdown) in enumerate(pages, start=1):
        print(f"Running page-mapped prompt on PDF page {page_number} ({index}/{len(pages)})...", flush=True)
        page_payload = _payload_for_pdf_page(payload, page_number, page_markdown, len(pages))
        last_response = _post_ollama_chat(page_payload)
        content = _assistant_content(last_response)
        if content:
            outputs.append(f"## Page {page_number}\n\n{content}")

    merged_content = "\n\n---\n\n".join(outputs).strip()
    if not merged_content:
        merged_content = "No relevant content was found in the PDF pages."
    return {
        **last_response,
        "id": "chatcmpl-tokenless-page-map",
        "object": "chat.completion",
        "model": last_response.get("model", payload.get("model", OLLAMA_MODEL)),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": merged_content},
                "finish_reason": "stop",
            }
        ],
    }


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]{3,}", text)}


def _split_markdown(markdown: str, chunk_size: int = 3000, overlap: int = 300) -> list[str]:
    chunks = []
    start = 0
    while start < len(markdown):
        end = min(len(markdown), start + chunk_size)
        chunks.append(markdown[start:end])
        if end == len(markdown):
            break
        start = max(0, end - overlap)
    return chunks


def _select_pdf_context(markdown: str, question: str, max_chars: int = 12000) -> str:
    chunks = _split_markdown(markdown)
    query_terms = _tokenize(question)
    if not query_terms:
        return "\n\n---\n\n".join(chunks[: max(1, max_chars // 3000)])
    scored = []
    for index, chunk in enumerate(chunks):
        score = len(query_terms & _tokenize(chunk))
        if score:
            scored.append((score, index, chunk))
    if not scored:
        return "\n\n---\n\n".join(chunks[: max(1, max_chars // 3000)])
    selected = []
    total = 0
    for _score, _index, chunk in sorted(scored, key=lambda item: (-item[0], item[1])):
        if total + len(chunk) > max_chars and selected:
            break
        selected.append(chunk)
        total += len(chunk)
    return "\n\n---\n\n".join(selected)


class TokenlessProxy(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        job_match = re.fullmatch(r"/tokenless/jobs/([a-f0-9]+)", self.path.rstrip("/"))
        if job_match:
            job = _get_page_job(job_match.group(1))
            if job is None:
                self._send_json(404, {"error": "Unknown PDF page job."})
                return
            self._send_json(200, job)
            return
        if self.path.rstrip("/") == "/tokenless/pdf/pages":
            if not PDF_MARKDOWN.exists():
                self._send_json(404, {"error": "PDF Markdown context is not available."})
                return
            pages = [
                {"page": page_number, "markdown": page_markdown}
                for page_number, page_markdown in _split_pdf_pages(
                    PDF_MARKDOWN.read_text(encoding="utf-8")
                )
            ]
            self._send_json(200, {"pages": pages})
            return
        target = f"http://localhost:11434{self.path}"
        try:
            with urllib.request.urlopen(target, timeout=30) as response:
                body = response.read()
                self.send_response(response.status)
                self.send_header(
                    "Content-Type",
                    response.headers.get("Content-Type", "application/json"),
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except Exception as e:  # noqa: BLE001
            self._send_json(502, {"error": repr(e)})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            job_match = re.fullmatch(r"/tokenless/jobs/([a-f0-9]+)", self.path.rstrip("/"))
            if job_match:
                self._send_json(202, _start_page_job(job_match.group(1), payload))
                return
            if self.path.rstrip("/") == "/v1/chat/completions":
                if _should_map_pdf_pages(payload):
                    self._send_json(200, _map_pdf_pages(payload))
                    return
                payload = _inject_pdf_context(payload)
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"http://localhost:11434{self.path}",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=600) as response:
                response_body = response.read()
                self.send_response(response.status)
                self.send_header(
                    "Content-Type",
                    response.headers.get("Content-Type", "application/json"),
                )
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
        except Exception as e:  # noqa: BLE001
            self._send_json(502, {"error": repr(e)})


def _start_proxy() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8000), TokenlessProxy)
    print("Starting Tokenless PDF context proxy on port 8000...", flush=True)
    server.serve_forever()


print("Installing system dependencies...", flush=True)
_publish("Installing system dependencies")
_run(
    "apt-get update -qq && apt-get install -y -qq curl zstd",
    "Failed to install system dependencies.",
)

print("Installing Ollama...", flush=True)
_publish("Installing Ollama")
_run("curl -fsSL https://ollama.com/install.sh | sh 2>/dev/null", "Ollama installer failed.")

print("Starting Ollama server...", flush=True)
_publish("Starting Ollama server")
os.system(
    "OLLAMA_HOST=0.0.0.0:11434 nohup ollama serve "
    "> /tmp/ollama_serve_stdout.log 2>/tmp/ollama_serve_stderr.log &"
)
time.sleep(5)

if os.system("ps aux | grep -E 'ollama serve' | grep -v grep > /dev/null 2>&1") != 0:
    _fail("Ollama server failed to start.")

print("Downloading model (this can take a long time)...", flush=True)
_publish("Downloading gpt-oss:20b")
_run(f"ollama pull {OLLAMA_MODEL}", "Model download failed.")

_prepare_pdf_context()

print("Installing cloudflared...", flush=True)
_publish("Installing public tunnel")
_run(
    "curl -L --output /tmp/cloudflared.deb "
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64.deb "
    "&& dpkg -i /tmp/cloudflared.deb",
    "cloudflared installation failed.",
)

print("Starting public tunnel...", flush=True)
_publish("Starting public tunnel")
target_port = 8000 if PDF_MARKDOWN.exists() else 11434
if PDF_MARKDOWN.exists():
    threading.Thread(target=_start_proxy, daemon=True).start()
    time.sleep(2)
os.system(
    f"nohup cloudflared tunnel --url http://localhost:{target_port} --no-autoupdate "
    f"> {CLOUDFLARED_LOG} 2>&1 &"
)

url = None
deadline = time.time() + 180
pattern = re.compile(r"https://[-a-zA-Z0-9.]+\.trycloudflare\.com")
while time.time() < deadline:
    if CLOUDFLARED_LOG.exists():
        text = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="replace")
        match = pattern.search(text)
        if match:
            url = match.group(0)
            break
    time.sleep(1)

if not url:
    _fail("Timed out waiting for cloudflared public URL.")

OUT.write_text(url, encoding="utf-8")
<<<<<<< HEAD
print(f"TOKENLESS_PUBLIC_URL topic={NTFY_TOPIC} url={url}", flush=True)
_publish(url)
print("Published TOKENLESS_PUBLIC_URL to rendezvous channel.", flush=True)
=======
_publish(url)
print("Published TOKENLESS_PUBLIC_URL to rendezvous channel.", flush=True)
print(f"TOKENLESS_PUBLIC_URL topic={NTFY_TOPIC} url={url}", flush=True)
>>>>>>> 7f54bf7 (JSON extractor)

print("Tokenless GPT-OSS server is ready. Keeping Kaggle kernel alive...", flush=True)

while True:
    try:
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=5).read()
    except Exception as e:  # noqa: BLE001 - keep the kernel log useful
        print(f"Ollama health check failed: {e!r}", flush=True)
    time.sleep(60)
