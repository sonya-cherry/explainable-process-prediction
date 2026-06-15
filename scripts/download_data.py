"""Download the project dataset so a fresh checkout can run end to end.

This fetches the BPI Challenge 2013 incidents event log from its public
4TU.ResearchData record, verifies the checksum, and unpacks the XES file into
``data/raw/BPI_Challenge_2013_incidents/``. The pipeline and notebooks expect
the log at that location.

Source record (DOI 10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee):
https://data.4tu.nl/articles/dataset/BPI_Challenge_2013_incidents/12693914

Run from the repository root:

    python scripts/download_data.py

Only the Python standard library is used, so no extra dependencies are needed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "BPI_Challenge_2013_incidents"

# Files published in the 4TU record. The event log is distributed gzip-compressed
# and is decompressed after download; DATA.xml is small dataset documentation.
EVENT_LOG = {
    "url": (
        "https://data.4tu.nl/file/"
        "0fc5c579-e544-4fab-9143-fab1f5192432/"
        "aa51ffbb-25fd-4b5a-b0b8-9aba659b7e8c"
    ),
    "md5": "d4809bd55e3e1c15b017ab4e58228297",
    "compressed_name": "BPI_Challenge_2013_incidents.xes.gz",
    "extracted_name": "BPI_Challenge_2013_incidents.xes",
}
DATA_XML = {
    "url": (
        "https://data.4tu.nl/file/"
        "0fc5c579-e544-4fab-9143-fab1f5192432/"
        "4f915e5e-c6b0-4f77-8e26-a95bd56f169f"
    ),
    "md5": "5ea35c5b94bcdfcb946bea8c89a13dbe",
    "name": "DATA.xml",
}


def _md5(path: Path) -> str:
    """Return the MD5 hex digest of a file, read in chunks."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    """Download ``url`` to ``destination`` via a temporary file."""
    print(f"Downloading {url}")
    with tempfile.NamedTemporaryFile(
        delete=False, dir=destination.parent
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp_path.replace(destination)
    finally:
        tmp_path.unlink(missing_ok=True)


def _verify(path: Path, expected_md5: str) -> None:
    """Raise if the downloaded file does not match the expected checksum."""
    actual = _md5(path)
    if actual != expected_md5:
        path.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {path.name}: "
            f"expected {expected_md5}, got {actual}"
        )
    print(f"Verified checksum for {path.name}")


def download_dataset(force: bool = False) -> Path:
    """Download and unpack the event log. Returns the extracted XES path."""
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    extracted_path = TARGET_DIR / EVENT_LOG["extracted_name"]
    if extracted_path.exists() and not force:
        print(f"Event log already present: {extracted_path}")
    else:
        compressed_path = TARGET_DIR / EVENT_LOG["compressed_name"]
        _download(EVENT_LOG["url"], compressed_path)
        _verify(compressed_path, EVENT_LOG["md5"])

        print(f"Extracting {compressed_path.name}")
        with gzip.open(compressed_path, "rb") as src, extracted_path.open(
            "wb"
        ) as dst:
            shutil.copyfileobj(src, dst)
        compressed_path.unlink(missing_ok=True)
        print(f"Event log ready: {extracted_path}")

    data_xml_path = TARGET_DIR / DATA_XML["name"]
    if data_xml_path.exists() and not force:
        print(f"Documentation already present: {data_xml_path}")
    else:
        _download(DATA_XML["url"], data_xml_path)
        _verify(data_xml_path, DATA_XML["md5"])

    return extracted_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the dataset is already present.",
    )
    args = parser.parse_args(argv)

    try:
        path = download_dataset(force=args.force)
    except (OSError, ValueError) as error:
        print(f"\nDataset download failed: {error}", file=sys.stderr)
        return 1

    print(f"\nDataset available at: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
