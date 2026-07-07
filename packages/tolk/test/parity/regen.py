#!/usr/bin/env python3
"""Regenerate all parity .expected files from the reference clone.

Walks every sibling directory and runs its main.py. Each case's main.py is a
self-contained script that writes *.expected next to itself.

Usage:
    uv run packages/tolk/test/parity/regen.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    cases = sorted(
        name
        for name in os.listdir(HERE)
        if os.path.isdir(os.path.join(HERE, name))
        and os.path.isfile(os.path.join(HERE, name, "main.py"))
    )
    for name in cases:
        main_py = os.path.join(HERE, name, "main.py")
        print(f"case {name}")
        subprocess.run([sys.executable, main_py], check=True)


if __name__ == "__main__":
    main()
