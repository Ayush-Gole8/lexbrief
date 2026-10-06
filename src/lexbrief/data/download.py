"""Download raw datasets: BUILD (Hugging Face) and IN-Ext/IN-Abs (Zenodo 7152317)."""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

from lexbrief.config import DataConfig

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20


def md5sum(path: Path) -> str:
    """Return the hex MD5 digest of a file."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_instructions(cfg: DataConfig) -> str:
    return (
        f"BUILD ({cfg.build_repo_id}) is a gated Hugging Face dataset.\n"
        "  1. Log in at https://huggingface.co and accept the terms at\n"
        f"     https://huggingface.co/datasets/{cfg.build_repo_id}\n"
        "  2. Create a read token at https://huggingface.co/settings/tokens\n"
        "  3. In the venv run:  hf auth login   (paste the token)\n"
        "  4. Re-run:  lexbrief download\n"
        f"  Or place {', '.join(cfg.build_files)} manually in {Path(cfg.build_dir).resolve()}"
    )


def download_build(cfg: DataConfig, force: bool = False) -> list[Path] | None:
    """Download BUILD train/dev/test JSON files from the Hugging Face Hub into ``build_dir``.

    Returns:
        Local paths, or ``None`` if access failed (instructions are logged).
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    out_dir = Path(cfg.build_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in cfg.build_files:
        target = out_dir / name
        if target.is_file() and not force:
            logger.info("BUILD %s already present (%s)", name, target)
        else:
            logger.info("Downloading BUILD %s from %s", name, cfg.build_repo_id)
            try:
                cached = hf_hub_download(
                    repo_id=cfg.build_repo_id, filename=name, repo_type="dataset"
                )
            except (HfHubHTTPError, OSError) as e:
                logger.error("BUILD download failed: %s", str(e).splitlines()[0])
                logger.error(_build_instructions(cfg))
                return None
            shutil.copyfile(cached, target)
        paths.append(target)
    return paths


def _manual_instructions(cfg: DataConfig, zip_path: Path) -> str:
    return (
        "Automatic download of the Zenodo archive failed.\n"
        "Manual steps:\n"
        "  1. Open https://zenodo.org/records/7152317 in a browser.\n"
        "  2. Download 'dataset.zip' (~98 MB).\n"
        f"  3. Save it as: {zip_path.resolve()}\n"
        f"  4. Check:  (Get-FileHash -Algorithm MD5 '{zip_path}').Hash  ==  {cfg.zenodo_md5}\n"
        "  5. Re-run:  lexbrief download"
    )


def download_zenodo(cfg: DataConfig, force: bool = False) -> Path | None:
    """Download, MD5-verify and extract the Zenodo IN-Ext/IN-Abs archive.

    Returns:
        The extraction directory, or ``None`` if the download failed (instructions are logged).
    """
    out_dir = Path(cfg.zenodo_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "dataset.zip"

    if not zip_path.is_file() or force:
        tmp = zip_path.with_suffix(".zip.part")
        logger.info("Downloading %s", cfg.zenodo_url)
        try:
            req = urllib.request.Request(cfg.zenodo_url, headers={"User-Agent": "lexbrief"})
            with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as f:
                shutil.copyfileobj(resp, f, _CHUNK)
            tmp.replace(zip_path)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            logger.error("Download failed: %s", e)
            logger.error(_manual_instructions(cfg, zip_path))
            return None

    digest = md5sum(zip_path)
    if digest != cfg.zenodo_md5:
        logger.error("MD5 mismatch for %s: got %s, expected %s", zip_path, digest, cfg.zenodo_md5)
        logger.error(_manual_instructions(cfg, zip_path))
        return None
    logger.info("MD5 verified (%s)", digest)

    marker = out_dir / ".extracted"
    if not marker.is_file() or force:
        logger.info("Extracting %s", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        marker.write_text(digest, encoding="utf-8")
    return out_dir


def find_inext_root(cfg: DataConfig) -> Path:
    """Locate the IN-Ext folder inside the extracted Zenodo archive."""
    base = Path(cfg.zenodo_dir)
    hits = sorted(p for p in base.rglob(cfg.inext_subdir) if p.is_dir())
    if not hits:
        raise FileNotFoundError(
            f"No '{cfg.inext_subdir}' folder under {base}; run `lexbrief download` first"
        )
    return hits[0]


def _tree(base: Path, depth: int, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if depth < 0:
        return lines
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        n_files = sum(1 for p in d.iterdir() if p.is_file() and not p.name.startswith("."))
        lines.append(f"{prefix}{d.name}/  [{n_files} files]")
        lines += _tree(d, depth - 1, prefix + "    ")
    return lines


def _first_lines(path: Path, k: int = 2, width: int = 160) -> list[str]:
    from lexbrief.data.inext_parser import read_lines

    return [ln[:width] + ("..." if len(ln) > width else "") for ln in read_lines(path)[:k]]


def inspect_raw(cfg: DataConfig) -> list[str]:
    """Human-readable report of the raw downloads (tree, counts, samples, test labels)."""
    out: list[str] = []
    base = Path(cfg.zenodo_dir)
    out.append(f"== Zenodo extraction tree ({base}, 3 levels) ==")
    out += _tree(base, depth=3) if base.is_dir() else ["(missing: run `lexbrief download`)"]
    try:
        root = find_inext_root(cfg)
    except FileNotFoundError as e:
        out.append(str(e))
        root = None
    if root is not None:
        judg = sorted((root / "judgement").glob("*.txt"))
        out.append("")
        out.append(f"== IN-Ext ({root}) ==")
        out.append(f"judgement files: {len(judg)}")
        for a in ("A1", "A2"):
            full = list((root / "summary" / "full" / a).glob("*.txt"))
            seg = {
                d.name: len(list(d.glob("*.txt")))
                for d in sorted((root / "summary" / "segment-wise" / a).glob("*"))
                if d.is_dir()
            }
            out.append(f"{a}: full={len(full)} segment-wise={seg}")
        if judg:
            sample = judg[0]
            out.append(f"-- sample judgment {sample.name} --")
            out += _first_lines(sample)
            out.append(f"-- sample full summary A1/{sample.name} --")
            out += _first_lines(root / "summary" / "full" / "A1" / sample.name)
            seg_file = root / "summary" / "segment-wise" / "A1" / "facts" / sample.name
            if seg_file.is_file():
                out.append(f"-- sample segment-wise A1/facts/{sample.name} --")
                out += _first_lines(seg_file, k=1)

    out.append("")
    out.append(f"== BUILD ({cfg.build_dir}) ==")
    from lexbrief.data.build_parser import has_labels, load_raw

    for name in cfg.build_files:
        p = Path(cfg.build_dir) / name
        if not p.is_file():
            out.append(f"{name}: MISSING")
            continue
        raw = load_raw(p)
        n_spans = sum(
            len(a.get("result") or []) for d in raw for a in (d.get("annotations") or [])[:1]
        )
        out.append(f"{name}: {len(raw)} docs, {n_spans} spans, labelled={has_labels(raw)}")
        if raw:
            d0 = raw[0]
            out.append(f"   keys={sorted(d0)} meta={d0.get('meta')}")
            res = ((d0.get("annotations") or [{}])[0].get("result") or [{}])[:2]
            for r in res:
                v = r.get("value") or {}
                out.append(f"   value: labels={v.get('labels')} text={str(v.get('text'))[:100]!r}")
    return out


def download_all(cfg: DataConfig, force: bool = False) -> bool:
    """Download every raw source. Returns True if all succeeded."""
    build_ok = download_build(cfg, force=force) is not None
    zenodo_ok = download_zenodo(cfg, force=force) is not None
    return build_ok and zenodo_ok
