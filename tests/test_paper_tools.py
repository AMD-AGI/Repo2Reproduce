"""Tests for paper-tool helpers that don't require OpenDataLoader/Java/network."""

from __future__ import annotations

import json
from pathlib import Path

from repo2reproduce.tools import paper as paper_io


def test_normalize_pdf_url():
    assert paper_io.normalize_pdf_url("https://arxiv.org/abs/1234.5678") == (
        "https://arxiv.org/pdf/1234.5678.pdf"
    )
    assert paper_io.normalize_pdf_url("https://arxiv.org/pdf/1234.5678v2") == (
        "https://arxiv.org/pdf/1234.5678v2.pdf"
    )
    assert paper_io.normalize_pdf_url("https://example.com/x.pdf") == "https://example.com/x.pdf"


def test_find_quote(tmp_path: Path):
    parse = tmp_path / "paper_parse.json"
    md = tmp_path / "paper.md"
    md.write_text("We train for 200 epochs with a learning rate of 0.1.", encoding="utf-8")
    parse.write_text(
        json.dumps(
            {
                "markdown_path": str(md),
                "blocks": [
                    {
                        "type": "text",
                        "page": 5,
                        "bbox": None,
                        "text": "We train for 200 epochs with a learning rate of 0.1.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    hit = paper_io.find_quote(parse, "we train for 200   epochs")
    assert hit["found"] is True
    assert "p.5" in hit["locator"]

    miss = paper_io.find_quote(parse, "this sentence is absent from the paper entirely")
    assert miss["found"] is False


def test_extract_blocks_reconstructs_table(tmp_path: Path):
    j = tmp_path / "odl.json"
    j.write_text(
        json.dumps(
            {
                "type": "document",
                "children": [
                    {"type": "heading", "page": 1, "level": 1, "text": "Experiments"},
                    {
                        "type": "table",
                        "page": 6,
                        "rows": [
                            {"cells": [{"text": "Method"}, {"text": "Acc"}]},
                            {"cells": [{"text": "Ours"}, {"text": "94.3"}]},
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    blocks = paper_io._extract_blocks(str(j))
    table = next(b for b in blocks if b["type"] == "table")
    assert "Ours" in table["text"] and "94.3" in table["text"]
    assert table["page"] == 6
    assert any(b.get("is_heading") for b in blocks)
