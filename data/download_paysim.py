"""Download the PaySim dataset from Kaggle and record its checksum for provenance.

Requires a Kaggle API token at ~/.kaggle/access_token or ~/.kaggle/kaggle.json (never commit it).
Usage: python data/download_paysim.py
"""
import hashlib
import json
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


def main() -> int:
    kaggle_dir = Path.home() / ".kaggle"
    if not any((kaggle_dir / f).exists() for f in ("access_token", "kaggle.json")):
        print("Missing Kaggle token in ~/.kaggle/, see README 'Dataset' section.", file=sys.stderr)
        return 1

    if not RAW_CSV.exists():
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
