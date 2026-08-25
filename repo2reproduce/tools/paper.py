"""Strict paper-reading layer.

Paper PDFs are parsed using **OpenDataLoader PDF only**. It is a deterministic,
local-first, Java-based parser (no torch / no GPU), so it never touches the host's
ML environment. We use its full feature set:

* ``format="json,markdown,html"`` -> structured JSON + readable Markdown + HTML.
* ``table_method="cluster"``      -> border + clustering table extraction (merged cells).
* ``reading_order="xycut"``       -> correct human reading order.
* ``markdown_page_separator``     -> page markers in the Markdown so even the MD carries
                                     page locators.
* ``image_output="external"``     -> figures saved alongside (referenced, not inlined).

The JSON carries page numbers + bounding boxes + element types (heading, paragraph,
table, list, ...), which we flatten into grounded "blocks" with precise locators. The
result is a single ``paper_parse.json`` plus ``paper.md`` that the ``paper-analyst``
agent reads. No other PDF library is permitted anywhere in the codebase.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def normalize_pdf_url(url: str) -> str:
    """Resolve common arXiv URL shapes to a direct PDF URL.

    Handles ``arxiv.org/abs/1234.5678``, ``arxiv.org/pdf/1234.5678`` (with or without
    ``.pdf``), and leaves any other URL untouched.
    """
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+(?:v[0-9]+)?)", url)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}.pdf"
    return url


def download_pdf(url: str, dest: Path) -> Path:
    """Download a PDF (following redirects) to ``dest``. Returns the path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    pdf_url = normalize_pdf_url(url)
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        resp = client.get(pdf_url, headers={"User-Agent": "Repo2Reproduce/0.1"})
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        if "pdf" not in ctype and not resp.content[:4] == b"%PDF":
            raise ValueError(
                f"URL did not return a PDF (content-type={ctype!r}). Resolved URL: {pdf_url}"
            )
        dest.write_bytes(resp.content)
    return dest


# --------------------------------------------------------------------------- #
# OpenDataLoader PDF (the only permitted parser)
# --------------------------------------------------------------------------- #
PAGE_SEPARATOR = "\n\n<!-- R2R_PAGE -->\n\n"  # injected between pages in the Markdown


def _ensure_java_on_path() -> None:
    """OpenDataLoader shells out to the literal ``java`` command (it ignores JAVA_HOME),
    so prepend a configured Java 11+ home's bin/ to PATH for this process."""
    java_home = os.environ.get("R2R_JAVA_HOME")
    if not java_home:
        return
    java_bin = str(Path(java_home) / "bin")
    os.environ["JAVA_HOME"] = java_home
    if java_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = f"{java_bin}{os.pathsep}{os.environ.get('PATH', '')}"


def parse_with_opendataloader(pdf_path: Path, out_dir: Path) -> dict[str, Any]:
    """Parse a PDF with OpenDataLoader PDF using its full deterministic feature set.

    Requires Java 11+ (set ``R2R_JAVA_HOME`` to a Java 11+ home if the default ``java``
    is older). Returns the artifact paths.
    """
    try:
        import opendataloader_pdf  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "opendataloader-pdf is not installed. Install with `pip install opendataloader-pdf` "
            "(requires a Java 11+ runtime)."
        ) from e

    import opendataloader_pdf

    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_java_on_path()

    threads = os.environ.get("R2R_ODL_THREADS")  # optional parallelism
    opendataloader_pdf.convert(
        input_path=[str(pdf_path)],
        output_dir=str(out_dir),
        format="json,markdown,html",
        table_method="cluster",          # border + clustering (handles merged cells)
        reading_order="xycut",           # correct reading order
        markdown_page_separator=PAGE_SEPARATOR,
        image_output="external",         # save figures to disk, reference them
        image_dir=str(out_dir / "images"),
        quiet=True,
        **({"threads": threads} if threads else {}),
    )

    return {
        "engine": "opendataloader-pdf",
        "json_path": _as_str(_find_first(out_dir, [".json"])),
        "markdown_path": _as_str(_find_first(out_dir, [".md", ".markdown"])),
        "html_path": _as_str(_find_first(out_dir, [".html", ".htm"])),
        "out_dir": str(out_dir),
    }


# --------------------------------------------------------------------------- #
# Single merged parse the agents read
# --------------------------------------------------------------------------- #
@dataclass
class ParsedPaper:
    pdf_path: str
    opendataloader: dict[str, Any]
    merged_markdown_path: str
    parse_json_path: str
    blocks: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    headings: list[dict] = field(default_factory=list)


def parse_paper(pdf_path: Path, work_dir: Path) -> ParsedPaper:
    """Parse the paper with OpenDataLoader and assemble paper.md + paper_parse.json."""
    odl = parse_with_opendataloader(pdf_path, work_dir / "opendataloader")

    # The Markdown the agent reads.
    merged_md = work_dir / "paper.md"
    md_src = odl.get("markdown_path")
    if md_src and Path(md_src).exists():
        merged_md.write_text(Path(md_src).read_text(encoding="utf-8"), encoding="utf-8")
    else:
        merged_md.write_text("", encoding="utf-8")

    blocks = _extract_blocks(odl.get("json_path"))
    tables = [b for b in blocks if b.get("type", "").lower() in {"table"}]
    headings = [b for b in blocks if "head" in b.get("type", "").lower() or b.get("is_heading")]

    parse_json = work_dir / "paper_parse.json"
    payload = {
        "pdf_path": str(pdf_path),
        "engine": "opendataloader-pdf",
        "artifacts": odl,
        "markdown_path": str(merged_md),
        "page_separator": PAGE_SEPARATOR,
        "num_blocks": len(blocks),
        "num_tables": len(tables),
        "blocks": blocks,
    }
    # Some PDF fonts decode to lone UTF-16 surrogate code points. Escaping
    # non-ASCII keeps the JSON valid instead of failing the entire paper
    # bootstrap with UnicodeEncodeError.
    parse_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    return ParsedPaper(
        pdf_path=str(pdf_path),
        opendataloader=odl,
        merged_markdown_path=str(merged_md),
        parse_json_path=str(parse_json),
        blocks=blocks,
        tables=tables,
        headings=headings,
    )


def _extract_blocks(json_path: Optional[str]) -> list[dict]:
    """Flatten OpenDataLoader JSON into grounded blocks.

    Each block is ``{type, page, bbox, text}``. We also reconstruct table text from
    nested rows/cells when a table node has no direct text. OpenDataLoader's JSON shape
    varies by version, so we walk it defensively.
    """
    if not json_path or not Path(json_path).exists():
        return []
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception:
        return []

    blocks: list[dict] = []

    # OpenDataLoader PDF uses spaced key names: "page number", "bounding box",
    # "heading level", and nests children under "content" / "kids".
    _PAGE_KEYS = ("page number", "page", "page_number", "pageNumber", "page_index")
    _BBOX_KEYS = ("bounding box", "bbox", "bounding_box", "boundingBox")
    _CHILD_KEYS = ("content", "kids", "rows", "cells", "items", "children")

    def _first(node: dict, keys) -> Any:
        for k in keys:
            if k in node and node[k] is not None:
                return node[k]
        return None

    def own_text(node: dict) -> str:
        """The node's OWN text (a string content/text/value), not aggregated children."""
        for key in ("text", "value"):
            t = node.get(key)
            if isinstance(t, str) and t.strip():
                return t.strip()
        c = node.get("content")
        if isinstance(c, str) and c.strip():
            return c.strip()
        return ""

    def aggregated_text(node: dict) -> str:
        """Reconstruct text for container nodes (e.g. a table) from descendants."""
        parts: list[str] = []
        for key in _CHILD_KEYS:
            v = node.get(key)
            if isinstance(v, list):
                for child in v:
                    if isinstance(child, dict):
                        ct = own_text(child) or aggregated_text(child)
                        if ct:
                            parts.append(ct)
                    elif isinstance(child, str) and child.strip():
                        parts.append(child.strip())
        return " | ".join(parts).strip()

    def walk(node: Any, inherited_page: Any = None):
        if isinstance(node, dict):
            ntype = str(node.get("type") or node.get("kind") or "text")
            page = _first(node, _PAGE_KEYS)
            if page is None:
                page = inherited_page  # propagate page to children that omit it
            bbox = _first(node, _BBOX_KEYS)

            text = own_text(node)
            # For tables (and any node that has a page but no own text), reconstruct
            # text from descendants so the table is searchable as one block.
            if not text and ntype.lower() == "table":
                text = aggregated_text(node)

            # Only emit a block when this node has its OWN textual content (or is a
            # reconstructed table). This avoids emitting a giant page-less aggregate for
            # the document root / section containers, keeping page locators accurate.
            if text:
                blocks.append(
                    {
                        "type": ntype,
                        "page": page,
                        "bbox": bbox,
                        "is_heading": bool(
                            "head" in ntype.lower()
                            or node.get("heading level") is not None
                            or node.get("level") is not None
                            or node.get("heading")
                        ),
                        "text": text,
                    }
                )
            for v in node.values():
                walk(v, page)
        elif isinstance(node, list):
            for v in node:
                walk(v, inherited_page)

    walk(data)
    return blocks


def _find_first(root: Path, suffixes: list[str]) -> Optional[Path]:
    """Return the most-likely main file under ``root`` matching any suffix."""
    if not root.exists():
        return None
    candidates: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and any(p.name.endswith(s) for s in suffixes):
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(p.parts), -p.stat().st_size))
    return candidates[0]


def _as_str(p: Optional[Path]) -> Optional[str]:
    return str(p) if p else None


# --------------------------------------------------------------------------- #
# Grounding helper used by verify_quote_in_paper
# --------------------------------------------------------------------------- #
def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def find_quote(parse_json_path: Path, quote: str) -> dict[str, Any]:
    """Fuzzily locate ``quote`` in the parsed paper and return a page locator.

    Matching is whitespace-insensitive substring matching against the structured blocks
    first (for a page locator), then the Markdown as a fallback existence check.
    """
    payload = json.loads(Path(parse_json_path).read_text(encoding="utf-8"))
    nq = _normalize(quote)
    if len(nq) < 8:
        return {"found": False, "locator": None, "matched_text": None,
                "reason": "quote too short to verify reliably"}

    for b in payload.get("blocks", []):
        if nq in _normalize(b.get("text", "")):
            page = b.get("page")
            loc = f"p.{page}" if page is not None else "page unknown"
            btype = b.get("type")
            if btype and btype.lower() != "text":
                loc = f"{loc}, {btype}"
            return {"found": True, "locator": loc, "matched_text": b.get("text")[:500]}

    md_path = payload.get("markdown_path")
    if md_path and Path(md_path).exists():
        md_raw = Path(md_path).read_text(encoding="utf-8")
        if nq in _normalize(md_raw):
            # Try to resolve a page by counting page separators before the match.
            sep = payload.get("page_separator", PAGE_SEPARATOR)
            page = _page_from_markdown(md_raw, quote, sep)
            loc = f"p.{page}" if page else "in paper (page unresolved)"
            return {"found": True, "locator": loc, "matched_text": quote}

    return {"found": False, "locator": None, "matched_text": None,
            "reason": "no whitespace-insensitive match in parsed paper"}


def _page_from_markdown(md_raw: str, quote: str, sep: str) -> Optional[int]:
    """Estimate a page number by counting page separators before the quote match."""
    nmd = _normalize(md_raw)
    nq = _normalize(quote)
    idx = nmd.find(nq)
    if idx < 0 or not sep:
        return None
    # Count separators in the normalized prefix; the separator normalizes too.
    nsep = _normalize(sep)
    if not nsep:
        return None
    prefix = nmd[:idx]
    return prefix.count(nsep) + 1
