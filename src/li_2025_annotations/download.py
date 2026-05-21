"""Download and extract Li 2025 dataset from Zenodo record 15211538."""
import hashlib
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

RECORD_ID = "15211538"
DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "samples" / "Li_2025"

FILES = {
    "Images.zip": {
        "url": f"https://zenodo.org/records/{RECORD_ID}/files/Images.zip",
        "md5": "eb023fe37217d5d52b47b155be510e81",
    },
    "spaceranger_output.zip": {
        "url": f"https://zenodo.org/records/{RECORD_ID}/files/spaceranger_output.zip",
        "md5": "b6373eca73ea86ae15803e861ad8dc20",
    },
}

def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already present: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tmp.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def extract(zip_path: Path, out_dir: Path) -> None:
    marker = out_dir / f".extracted_{zip_path.stem}"
    if marker.exists():
        print(f"  already extracted: {zip_path.name}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in tqdm(zf.infolist(), desc=f"unzip {zip_path.name}"):
            zf.extract(member, out_dir)
    marker.touch()


def main() -> int:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for name, info in FILES.items():
        zip_path = DATA_ROOT / name
        print(f"\n>>> {name}")
        download(info["url"], zip_path)
        actual = md5sum(zip_path)
        if actual != info["md5"]:
            print(f"  MD5 mismatch: {actual} != {info['md5']}", file=sys.stderr)
            return 1
        print(f"  md5 ok ({actual})")
        extract(zip_path, DATA_ROOT)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())