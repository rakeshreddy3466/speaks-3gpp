"""Download the inputs into data/ (git ignored).

* 3GPP specs: the archive folder of each spec is listed and the newest Release 17 zip
  (version letter ``h``) is taken, so the exact version used is printed and saved in
  ``data/specs/versions.json`` (and copied to results/ by ``make data``).
* TeleQnA: the password protected zip from the authors' GitHub repo, unzipped locally only.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import urllib.request
import zipfile

import pyzipper

from . import DATA, RESULTS, config

ARCHIVE = "https://www.3gpp.org/ftp/Specs/archive/{series}_series/{spec}/"
TELEQNA_ZIP = "https://raw.githubusercontent.com/netop-team/TeleQnA/main/TeleQnA.zip"
TELEQNA_PASSWORD = b"teleqnadataset"  # published in the TeleQnA README
UA = {"User-Agent": "speaks3gpp research script"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def latest_version(listing_html: str, spec: str, prefix: str) -> str | None:
    """Return the newest zip name for a release prefix, e.g. '38300-hc0.zip'.

    3GPP version codes are three characters (major, technical, editorial) from 0-9 then a-z,
    so plain string order inside one release prefix is version order.
    """
    stem = spec.replace(".", "")
    names = set(re.findall(rf"({stem}-{prefix}[0-9a-z]{{2}}\.zip)", listing_html))
    return max(names) if names else None


def fetch_spec(spec: str, prefix: str, out_dir) -> str:
    series = spec.split(".")[0]
    url = ARCHIVE.format(series=series, spec=spec)
    name = latest_version(_get(url).decode("utf-8", "replace"), spec, prefix)
    if name is None:
        raise RuntimeError(f"no Release zip with prefix {prefix!r} found for {spec}")
    target = out_dir / f"{spec}.docx"
    if not target.exists():
        with zipfile.ZipFile(io.BytesIO(_get(url + name))) as z:
            docs = [n for n in z.namelist() if n.lower().endswith(".docx")]
            if not docs:
                raise RuntimeError(f"{name} has no .docx inside: {z.namelist()}")
            biggest = max(docs, key=lambda n: z.getinfo(n).file_size)  # the spec, not an annex
            target.write_bytes(z.read(biggest))
    print(f"{spec}: {name}")
    return name


def fetch_teleqna(out_dir) -> None:
    target = out_dir / "TeleQnA.json"
    if target.exists():
        return
    with pyzipper.AESZipFile(io.BytesIO(_get(TELEQNA_ZIP))) as z:
        z.setpassword(TELEQNA_PASSWORD)
        target.write_bytes(z.read("TeleQnA.txt"))
    print(f"TeleQnA: {len(json.loads(target.read_text()))} questions")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = config.load(args.config) if args.config else config.load()
    specs_dir = DATA / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg["corpus"]["release_prefix"]
    versions = {}
    for spec in [*cfg["corpus"]["specs"], cfg["corpus"]["vocabulary"]]:
        versions[spec] = fetch_spec(spec, prefix, specs_dir)
    (specs_dir / "versions.json").write_text(json.dumps(versions, indent=2))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "spec_versions.json").write_text(json.dumps(versions, indent=2) + "\n")
    fetch_teleqna(DATA)


if __name__ == "__main__":
    main()
