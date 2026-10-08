# /// script
# requires-python = ">=3.10"
# ///
"""Fetches the AMD firmware images rig.amd.pci boots GPUs with into a
firmware directory, the directory Rig_amd_pci.open_ takes as ~firmware.

Run from the worktree root:

  uv run dev/rig/lib/amd/pci/gen/fetch.py DIR

Each image of headers/firmware.tsv is downloaded from its URL in
linux-firmware's tree and written to DIR/<its path> once its BLAKE2b-256
digest matches the one pinned. An image present with that digest is kept;
the script fails, writing nothing for it, on any other.
"""

import argparse
import hashlib
import pathlib
import sys
import urllib.request

FIRMWARE = pathlib.Path(__file__).resolve().parent / "headers" / "firmware.tsv"


def digest(data):
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dir", type=pathlib.Path, help="the firmware directory to fill")
    a = p.parse_args()
    rows = [l.split("\t") for l in FIRMWARE.read_text().splitlines() if l and not l.startswith("#")]
    for path, pinned, url in sorted(rows):
        out = a.dir / path
        if out.exists() and digest(out.read_bytes()) == pinned:
            print(f"kept {path}")
            continue
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "raven-fetch"})) as r:
            data = r.read()
        got = digest(data)
        if got != pinned:
            sys.exit(f"{url}: BLAKE2b-256 {got}, pinned {pinned}")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(out)
        print(f"fetched {path}")


if __name__ == "__main__":
    main()
