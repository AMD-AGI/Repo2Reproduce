"""Unit tests for the pure-Python code-intelligence tools (no Claude, no network)."""

from __future__ import annotations

from pathlib import Path

from repo2reproduce.tools import code as code_intel


def test_extract_argparse(tmp_path: Path):
    src = tmp_path / "train.py"
    src.write_text(
        "import argparse\n"
        "def main():\n"
        "    p = argparse.ArgumentParser()\n"
        "    p.add_argument('--lr', type=float, default=0.1, help='learning rate')\n"
        "    p.add_argument('--epochs', type=int, default=200)\n"
        "    p.add_argument('--model', choices=['resnet50', 'vit'], default='resnet50')\n"
        "    return p.parse_args()\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    res = code_intel.extract_argparse(src)
    names = {n for a in res["arguments"] for n in a["names"]}
    assert {"--lr", "--epochs", "--model"} <= names
    lr = next(a for a in res["arguments"] if "--lr" in a["names"])
    assert lr["default"] == 0.1
    model = next(a for a in res["arguments"] if "--model" in a["names"])
    assert model["choices"] == ["resnet50", "vit"]


def test_verify_flags(tmp_path: Path):
    src = tmp_path / "train.py"
    src.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--lr', default=0.1)\n",
        encoding="utf-8",
    )
    res = code_intel.verify_flags(src, ["--lr", "--nonexistent"])
    assert res["results"]["--lr"] is True
    assert res["results"]["--nonexistent"] is False


def test_find_entry_points(tmp_path: Path):
    (tmp_path / "train.py").write_text(
        "import argparse\nif __name__ == '__main__':\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "utils.py").write_text("x = 1\n", encoding="utf-8")
    res = code_intel.find_entry_points(tmp_path)
    paths = [e["path"] for e in res["entry_points"]]
    assert "train.py" in paths
