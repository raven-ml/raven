#!/usr/bin/env python3
"""Checks for read-only upstream expectation verification."""

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import generate_reference as reference


class ReferenceCheckTests(unittest.TestCase):
    def test_raw_bytes_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            committed, generated = root / "committed", root / "generated"
            committed.mkdir()
            generated.mkdir()
            values = {"same.expected": (b"same\n", b"same\n"),
                      "crlf.expected": (b"line\n", b"line\r\n"),
                      "eof.expected": (b"line\n", b"line"),
                      "binary.expected": (b"\xff\n", b"\xfe\n")}
            for name, (before, after) in values.items():
                (committed / name).write_bytes(before)
                (generated / name).write_bytes(after)
            report = reference.check_expectations(committed, generated, set(values), {})
            self.assertEqual(report["matched"], ["same.expected"])
            self.assertEqual([item["file"] for item in report["changed"]],
                             ["binary.expected", "crlf.expected", "eof.expected"])
            for item in report["changed"]:
                self.assertNotEqual(item["committed_sha256"], item["generated_sha256"])
            self.assertIn(b"No newline at end of file", (generated / "comparison.diff").read_bytes())
            for name, (before, after) in values.items():
                self.assertEqual((committed / name).read_bytes(), before)
                self.assertEqual((generated / name).read_bytes(), after)

    def test_reference_only_requires_output_and_checks_existing_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            committed, generated = root / "committed", root / "generated"
            committed.mkdir()
            generated.mkdir()
            for name in ("unstored.expected", "stored.expected", "unknown.expected"):
                (generated / name).write_bytes(b"generated\n")
            (committed / "stored.expected").write_bytes(b"old\n")
            reference_only = {name: "reason" for name in
                              ("unstored.expected", "stored.expected", "missing.expected")}
            report = reference.check_expectations(
                committed, generated, set(reference_only) | {"unknown.expected"}, reference_only)
            self.assertEqual(report["reference_only"], ["unstored.expected"])
            self.assertEqual(report["missing"], ["missing.expected"])
            self.assertEqual(report["missing_baselines"], ["unknown.expected"])
            self.assertEqual([item["file"] for item in report["changed"]], ["stored.expected"])

    def test_generate_check_failure_writes_manifest_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            here = root / "packages" / "tolk" / "test"
            driver_dir = here / "parity" / "example"
            driver_dir.mkdir(parents=True)
            (driver_dir / "main.py").write_text(
                "from pathlib import Path\n"
                "Path(__file__).with_name('stage5.expected').write_bytes(b'new\\n')\n")
            baseline = driver_dir / "stage5.expected"
            baseline.write_bytes(b"old\n")
            # Exception files must never enter the upstream output inventory.
            (driver_dir / "stage5.expected.diff").write_bytes(b"reviewed exception\n")
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w"):
                pass
            def git(directory, *args):
                return b"0123456789\n" if args[0] == "rev-parse" else archive.getvalue()
            with patch.object(reference, "HERE", here), patch.object(reference, "ROOT", root), \
                    patch.object(reference, "git", git):
                output = root / "checked"
                with self.assertRaisesRegex(RuntimeError, "changed baselines"):
                    reference.generate(root, "pinned", output, "parity", check=True)
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertFalse(manifest["complete"])
                self.assertTrue(manifest["check"])
                report = manifest["drivers"][0]
                self.assertEqual(report["missing"], [])
                self.assertEqual(report["unexpected"], [])
                self.assertEqual(report["comparison"]["changed"][0]["file"], "stage5.expected")
                self.assertIn(b"-old\n+new\n", (output / "parity/example/comparison.diff").read_bytes())
                self.assertEqual(baseline.read_bytes(), b"old\n")
                # Candidate generation remains possible without accepting or
                # overwriting its differences in the committed baseline.
                reference.generate(root, "candidate", root / "candidate", "parity")
                self.assertEqual(baseline.read_bytes(), b"old\n")
                (driver_dir / "main.py").write_text(
                    "from pathlib import Path\n"
                    "Path(__file__).with_name('stage5.expected').write_bytes(b'old\\n')\n")
                matching = root / "matching"
                reference.generate(root, "pinned", matching, "parity", check=True)
                matching_manifest = json.loads((matching / "manifest.json").read_text())
                self.assertTrue(matching_manifest["complete"])
                self.assertEqual(matching_manifest["drivers"][0]["comparison"]["matched"],
                                 ["stage5.expected"])


if __name__ == "__main__":
    unittest.main()
