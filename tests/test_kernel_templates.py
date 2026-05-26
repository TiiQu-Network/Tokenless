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
