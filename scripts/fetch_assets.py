"""Fetch the pinned MicroDuck scene, meshes and licenses using only the stdlib."""

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
import tomllib
from http.client import IncompleteRead
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets/sources.toml"
DESTINATION = ROOT / "assets/vendor/microduck_rl"


def git_blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def asset_path(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in name:
        raise ValueError(f"Unsafe asset path: {name}")
    result = root.joinpath(*path.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes destination: {name}")
    return result


def verify(content: bytes, entry: dict) -> None:
    if (
        len(content) != entry["size"]
        or git_blob_sha1(content) != entry["git_blob_sha1"]
    ):
        raise ValueError(f"Integrity check failed: {entry['path']}")


def download(url: str, entry: dict) -> bytes:
    request = Request(url, headers={"User-Agent": "duckscout-asset-fetcher"})
    attempts = 5
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=60) as response:
                content = response.read(entry["size"] + 1)
            if len(content) < entry["size"]:
                raise IncompleteRead(content, entry["size"] - len(content))
            return content
        except (URLError, ConnectionError, TimeoutError, IncompleteRead) as error:
            # Permanent HTTP errors such as 404 will not improve with a retry.
            if isinstance(error, HTTPError) and error.code not in {
                408,
                429,
                500,
                502,
                503,
                504,
            }:
                raise OSError(f"{entry['path']}: {error}") from error
            if attempt == attempts:
                raise OSError(
                    f"{entry['path']}: download failed after {attempts} attempts: {error}"
                ) from error
            delay = 2 ** (attempt - 1)
            print(
                f"Retry {attempt}/{attempts - 1} in {delay}s: {entry['path']} ({error})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("Unreachable")


def fetch_file(source: dict, entry: dict, root: Path, check_only: bool) -> str:
    target = asset_path(root, entry["path"])
    if target.exists():
        content = target.read_bytes()
        verify(content, entry)
    else:
        if check_only:
            raise FileNotFoundError(f"Missing asset: {entry['path']}")
        url = f"{source['base_url']}/{source['revision']}/{entry['path']}"
        content = download(url, entry)
        verify(content, entry)
        target.parent.mkdir(parents=True, exist_ok=True)
        # A failed download never leaves a file that looks complete.
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(content)
                handle.close()
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Verify local files without network access"
    )
    args = parser.parse_args()
    try:
        manifest = tomllib.loads(MANIFEST.read_text())
        source = manifest["source"]
        if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
            raise ValueError("Source revision must be a full Git commit SHA")
        hashes = {}
        for entry in manifest["files"]:
            hashes[entry["path"]] = fetch_file(source, entry, DESTINATION, args.check)
            print(f"OK {entry['path']}", flush=True)
        if not args.check:
            receipt = {"revision": source["revision"], "sha256": hashes}
            (DESTINATION / "checksums.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
        print(f"Verified {len(hashes)} files at {source['revision']}")
    except (OSError, URLError, ValueError, KeyError) as error:
        print(f"Asset setup failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
