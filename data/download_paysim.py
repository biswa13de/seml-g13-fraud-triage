"""Download the PaySim dataset and record its checksum for provenance.

Uses `kagglehub` first, which downloads this public dataset with NO Kaggle
account, login or API token required -- this is what makes the dataset step
work unattended on Google Colab. Falls back to the `kaggle` CLI (which does
need a token at ~/.kaggle/access_token or ~/.kaggle/kaggle.json) only if
kagglehub is unavailable or fails.

Usage: python data/download_paysim.py
"""
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DATASET = "ealaxi/paysim1"
DATA_DIR = Path(__file__).resolve().parent
RAW_CSV = DATA_DIR / "paysim.csv"
PROVENANCE = DATA_DIR / "provenance.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_via_kagglehub() -> bool:
    try:
        import kagglehub
    except ImportError:
        return False
    try:
        downloaded_dir = Path(kagglehub.dataset_download(DATASET))
    except Exception as exc:
        print(f"kagglehub download failed ({exc}), falling back to the kaggle CLI.", file=sys.stderr)
        return False
    (csv_path,) = downloaded_dir.glob("*.csv")
    shutil.copy(csv_path, RAW_CSV)
    return True


def download_via_kaggle_cli() -> bool:
    kaggle_dir = Path.home() / ".kaggle"
    if not any((kaggle_dir / f).exists() for f in ("access_token", "kaggle.json")):
        print(
            "Missing Kaggle token in ~/.kaggle/ and kagglehub was unavailable. "
            "See README 'Dataset' section.",
            file=sys.stderr,
        )
        return False
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(DATA_DIR)],
        check=True,
    )
    archive = DATA_DIR / "paysim1.zip"
    with zipfile.ZipFile(archive) as zf:
        (csv_name,) = [n for n in zf.namelist() if n.endswith(".csv")]
        zf.extract(csv_name, DATA_DIR)
    (DATA_DIR / csv_name).rename(RAW_CSV)
    archive.unlink()
    return True


def main() -> int:
    if not RAW_CSV.exists():
        if not (download_via_kagglehub() or download_via_kaggle_cli()):
            return 1

    provenance = {
        "source": f"https://www.kaggle.com/datasets/{DATASET}",
        "file": RAW_CSV.name,
        "sha256": sha256(RAW_CSV),
        "bytes": RAW_CSV.stat().st_size,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
