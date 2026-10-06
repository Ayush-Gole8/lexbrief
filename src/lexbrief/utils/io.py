"""JSON / JSONL read-write helpers (UTF-8, parent dirs created on write)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

PathLike = str | Path


def iter_jsonl(path: PathLike) -> Iterator[dict[str, Any]]:
    """Yield one record per non-empty line of a JSONL file."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e


def read_jsonl(path: PathLike) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    return list(iter_jsonl(path))


def write_jsonl(records: Iterable[dict[str, Any]], path: PathLike) -> int:
    """Write records to a JSONL file. Returns the number of records written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_json(path: PathLike) -> Any:
    """Read a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: PathLike, indent: int = 2) -> None:
    """Write an object to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.write("\n")
