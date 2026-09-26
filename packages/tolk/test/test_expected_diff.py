#!/usr/bin/env python3
"""Subprocess checks for the exact expectation-difference artifact."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HELPER = Path(__file__).with_name("expected_diff.py")


class ExpectedDiffTests(unittest.TestCase):
    def run_diff(self, expected, actual, names=("expected", "actual")):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in names]
            for path, data in zip(paths, (expected, actual)):
                path.write_bytes(data)
            return subprocess.run(
                [sys.executable, str(HELPER), *(str(path) for path in paths)],
                capture_output=True,
                check=False,
            )

    def artifact(self, expected, actual, **kwargs):
        result = self.run_diff(expected, actual, **kwargs)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, b"")
        return result.stdout

    def test_known_output_and_path_independence(self):
        expected = b"one\ntwo\nthree\n"
        actual = b"one\nTWO\nthree\n"
        wanted = (
            f"upstream-sha256: {hashlib.sha256(expected).hexdigest()}\n"
            f"actual-sha256: {hashlib.sha256(actual).hexdigest()}\n"
        ).encode("ascii") + (
            b"--- upstream.expected\n+++ tolk.actual\n@@ -1,3 +1,3 @@\n"
            b" one\n-two\n+TWO\n three\n"
        )
        self.assertEqual(self.artifact(expected, actual), wanted)
        self.assertEqual(
            self.artifact(expected, actual, names=("other baseline", "other output")),
            wanted,
        )

    def test_equal_inputs_reject_stale_exception(self):
        for data in (b"", b"same\n", b"same\r\n", b"same"):
            with self.subTest(data=data):
                result = self.run_diff(data, data)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b"")
                self.assertIn(b"remove the stale .expected.diff", result.stderr)

    def test_line_endings_and_missing_final_newline_are_significant(self):
        for changed in (b"new\r\n", b"new"):
            with self.subTest(changed=changed):
                original = self.artifact(b"old\n", b"new\n")
                revised = self.artifact(b"old\n", changed)
                self.assertNotEqual(original, revised)
        no_newline = self.artifact(b"old", b"new")
        self.assertIn(
            b"-old\n\\ No newline at end of file\n"
            b"+new\n\\ No newline at end of file\n",
            no_newline,
        )
        self.assertIn(b"+new\r\n", self.artifact(b"old\n", b"new\r\n"))

    def test_input_changes_outside_diff_context_invalidate_artifact(self):
        baseline = b"".join(f"line {i}\n".encode("ascii") for i in range(30))
        actual = baseline.replace(b"line 15\n", b"changed 15\n")
        original = self.artifact(baseline, actual)
        # A shared change far from the hunk leaves the unified diff unchanged,
        # but both whole-input hashes must invalidate the reviewed artifact.
        new_baseline = baseline.replace(b"line 0\n", b"changed 0\n")
        new_actual = actual.replace(b"line 0\n", b"changed 0\n")
        revised = self.artifact(new_baseline, new_actual)
        self.assertEqual(original.split(b"\n", 2)[2], revised.split(b"\n", 2)[2])
        for old_hash, new_hash in zip(original.splitlines()[:2], revised.splitlines()[:2]):
            self.assertNotEqual(old_hash, new_hash)
        self.assertNotEqual(original, self.artifact(new_baseline, actual))
        self.assertNotEqual(original, self.artifact(baseline, new_actual))

    def test_non_utf8_bytes_are_preserved(self):
        self.assertIn(b"-\xff\n+\xfe\n", self.artifact(b"\xff\n", b"\xfe\n"))


if __name__ == "__main__":
    unittest.main()
