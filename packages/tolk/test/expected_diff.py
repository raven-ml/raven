#!/usr/bin/env python3
"""Describe an exact, reviewed difference from an upstream expectation."""

import argparse
import difflib
import hashlib
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()
    try:
        expected = args.expected.read_bytes()
        actual = args.actual.read_bytes()
    except OSError as error:
        parser.exit(1, f"expected_diff: {error}\n")
    if expected == actual:
        parser.exit(
            1,
            "expected_diff: inputs are identical; remove the stale .expected.diff "
            "exception and compare .expected directly with .actual\n",
        )

    output = sys.stdout.buffer
    output.write(f"upstream-sha256: {hashlib.sha256(expected).hexdigest()}\n".encode("ascii"))
    output.write(f"actual-sha256: {hashlib.sha256(actual).hexdigest()}\n".encode("ascii"))
    # Split only at LF: CR and every other byte remain part of the line.
    def lines(data):
        parts = data.split(b"\n")
        return [part + b"\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])

    for line in difflib.diff_bytes(
        difflib.unified_diff,
        lines(expected),
        lines(actual),
        fromfile=b"upstream.expected",
        tofile=b"tolk.actual",
        n=3,
    ):
        output.write(line)
        if not line.endswith(b"\n"):
            output.write(b"\n\\ No newline at end of file\n")


if __name__ == "__main__":
    main()
