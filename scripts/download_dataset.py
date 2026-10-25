"""Download a linked public dataset with YOUR credentials, and record its provenance.

Kaggle (primary source):
    1. Create a Kaggle account, accept the dataset's terms on its page, then
       Account -> Settings -> API -> "Create New Token" (downloads kaggle.json).
    2. Put kaggle.json in %USERPROFILE%\\.kaggle\\kaggle.json (never in this repository).
    3. python -m pip install kaggle
    4. python scripts/download_dataset.py --source kaggle_hair_diseases

Roboflow sources need a Roboflow API key and the export version you choose; this
script prints the manual steps for them instead of guessing a version.

The script writes data/raw/<source>/SOURCE_INFO.json (url, version, license,
download date), which the audit copies into every manifest row.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from scalptrack.config import load_config

KAGGLE_REFS = {"kaggle_hair_diseases": "sundarannamalai/hair-diseases"}


def write_source_info(dest: Path, scfg: dict, version: str | None, method: str) -> Path:
    info = {
        "url": scfg.get("url"),
        "license": scfg.get("license"),
        "version": version,
        "download_date": dt.date.today().isoformat(),
        "method": method,
    }
    p = dest / "SOURCE_INFO.json"
    p.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return p


def kaggle_download(source: str, scfg: dict, dest: Path) -> int:
    if shutil.which("kaggle") is None:
        print("The 'kaggle' command is not installed. Run:  python -m pip install kaggle", file=sys.stderr)
        return 2
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.is_file():
        print(f"No Kaggle API token at {cred}. See the steps at the top of this script.", file=sys.stderr)
        return 2
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_REFS[source], "-p", str(dest)]
    print("[download] " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc
    for z in dest.glob("*.zip"):
        print(f"[download] extracting {z.name}")
        with zipfile.ZipFile(z) as zf:
            for member in zf.namelist():  # refuse path traversal from the archive
                target = (dest / member).resolve()
                if not str(target).startswith(str(dest.resolve())):
                    raise RuntimeError(f"Unsafe path in archive: {member}")
            zf.extractall(dest)
        z.unlink()
    print(f"[download] provenance: {write_source_info(dest, scfg, scfg.get('version'), 'kaggle CLI')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None)
    ap.add_argument(
        "--record-only",
        action="store_true",
        help="data already extracted manually: only write SOURCE_INFO.json",
    )
    ap.add_argument("--version", help="export/version string to record (e.g. Roboflow version number)")
    args = ap.parse_args(argv)
    cfg = load_config()
    source = args.source or cfg["datasets"]["primary"]
    scfg = cfg["datasets"]["sources"][source]
    dest = Path(cfg["root"]) / scfg["dir"]
    if args.record_only:
        dest.mkdir(parents=True, exist_ok=True)
        print(write_source_info(dest, scfg, args.version or scfg.get("version"), "manual download"))
        return 0
    if source in KAGGLE_REFS:
        return kaggle_download(source, scfg, dest)
    print(
        f"'{source}' is a Roboflow Universe export. Manual steps:\n"
        f"  1. Open {scfg['url']} and sign in.\n"
        f"  2. Choose a version -> Download Dataset -> format 'YOLOv8' -> 'Download zip to computer'.\n"
        f"  3. Extract it into {dest} (so that {dest}\\data.yaml exists).\n"
        f"  4. python scripts/download_dataset.py --source {source} --record-only --version <N>\n"
        f"  5. python scripts/audit_dataset.py --source {source}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
