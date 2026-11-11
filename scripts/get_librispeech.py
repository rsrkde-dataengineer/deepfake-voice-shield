"""Download the public LibriSpeech splits used for the real voices (CC BY 4.0, https://www.openslr.org/12).

    python scripts/get_librispeech.py            # dev-clean (demo voices + calibration) and test-clean (held-out test)

Files are checked against the published MD5 and extracted to data/external/librispeech/ (git-ignored).
"""
from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "external"
SPLITS = {"dev-clean": "42e2234ba48799c1f50f24a7926300a1", "test-clean": "32fa31d27d2e1cad72775fee3f4849a9"}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for split, md5 in SPLITS.items():
        if (DEST / "librispeech" / "LibriSpeech" / split).exists():
            print(f"{split}: already there")
            continue
        archive = DEST / f"{split}.tar.gz"
        if not archive.exists():
            print(f"{split}: downloading ~340 MB ...")
            urllib.request.urlretrieve(f"https://www.openslr.org/resources/12/{split}.tar.gz", archive)
        digest = hashlib.md5(archive.read_bytes()).hexdigest()
        if digest != md5:
            raise SystemExit(f"{split}: checksum mismatch ({digest}) - delete {archive} and retry")
        with tarfile.open(archive) as tar:
            tar.extractall(DEST / "librispeech", filter="data")   # 'data' filter blocks unsafe paths
        print(f"{split}: ok")


if __name__ == "__main__":
    main()
