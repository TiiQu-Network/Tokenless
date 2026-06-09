from pathlib import Path


def test_gpt_oss_kernel_templates_compile():
    root = Path(__file__).resolve().parents[1]
    templates = [
        root / "tokenless" / "kernels" / "gpt_oss_20b" / "run.template.py",
        root / "tokenless" / "kernels" / "gpt_oss_20b" / "serve.template.py",
    ]

    for template in templates:
        compile(template.read_text(encoding="utf-8"), str(template), "exec")


def test_gpt_oss_kernel_templates_request_source_aware_pdf_chunks():
    root = Path(__file__).resolve().parents[1]
    templates = [
        root / "tokenless" / "kernels" / "gpt_oss_20b" / "run.template.py",
        root / "tokenless" / "kernels" / "gpt_oss_20b" / "serve.template.py",
    ]

    for template in templates:
        body = template.read_text(encoding="utf-8")
        assert "page_chunks=True" in body
        assert "page and section" in body
        assert "article=" not in body
        assert "paragraph=" not in body
        assert "[source: " in body


def test_gpt_oss_kernel_templates_find_nested_kaggle_dataset_mounts():
    root = Path(__file__).resolve().parents[1]
    templates = [
        root / "tokenless" / "kernels" / "gpt_oss_20b" / "run.template.py",
        root / "tokenless" / "kernels" / "gpt_oss_20b" / "serve.template.py",
    ]

    for template in templates:
        body = template.read_text(encoding="utf-8")
        assert 'input_root.glob(f"**/{dataset_slug}")' in body
        assert "expected_paths" in body
        assert "Uploaded PDF was not found at any of" in body


def test_gpt_oss_server_template_maps_large_pdf_prompts_by_page():
    root = Path(__file__).resolve().parents[1]
    body = (
        root
        / "tokenless"
        / "kernels"
        / "gpt_oss_20b"
        / "serve.template.py"
    ).read_text(encoding="utf-8")

    assert "PAGE_MAP_MIN_PROMPT_CHARS" in body
    assert "def _split_pdf_pages" in body
    assert "def _map_pdf_pages" in body
    assert "Running page-mapped prompt on PDF page" in body
    assert "## Page {page_number}" in body


def test_gpt_oss_server_template_exposes_async_pdf_page_jobs():
    root = Path(__file__).resolve().parents[1]
    body = (
        root
        / "tokenless"
        / "kernels"
        / "gpt_oss_20b"
        / "serve.template.py"
    ).read_text(encoding="utf-8")

    assert "def _start_page_job" in body
    assert "def _get_page_job" in body
    assert 'r"/tokenless/jobs/([a-f0-9]+)"' in body


def test_gpt_oss_server_template_marks_public_url_with_rendezvous_topic():
    root = Path(__file__).resolve().parents[1]
    body = (
        root
        / "tokenless"
        / "kernels"
        / "gpt_oss_20b"
        / "serve.template.py"
    ).read_text(encoding="utf-8")

    assert "TOKENLESS_PUBLIC_URL topic={NTFY_TOPIC} url={url}" in body
<<<<<<< HEAD
=======


def test_gpt_oss_server_template_does_not_delay_public_url_publish():
    root = Path(__file__).resolve().parents[1]
    body = (
        root
        / "tokenless"
        / "kernels"
        / "gpt_oss_20b"
        / "serve.template.py"
    ).read_text(encoding="utf-8")

    assert "NTFY_URL_PUBLISH_DELAY" not in body
    assert "url_publish_not_before" not in body
    assert "publish_delay" not in body


def test_gpt_oss_server_template_logs_url_after_ntfy_publish_attempt():
    root = Path(__file__).resolve().parents[1]
    body = (
        root
        / "tokenless"
        / "kernels"
        / "gpt_oss_20b"
        / "serve.template.py"
    ).read_text(encoding="utf-8")

    ntfy_publish = body.index("_publish(url)")
    log_fallback_marker = body.index(
        'print(f"TOKENLESS_PUBLIC_URL topic={NTFY_TOPIC} url={url}", flush=True)'
    )

    assert ntfy_publish < log_fallback_marker
>>>>>>> 7f54bf7 (JSON extractor)
