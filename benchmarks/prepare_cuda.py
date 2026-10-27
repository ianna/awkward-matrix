# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause
"""Refresh ignored kernel signatures after switching an editable Awkward checkout."""

import argparse
import subprocess
import sys
from pathlib import Path


def prepare(checkout):
    checkout = Path(checkout).resolve()
    generator = checkout / "dev/generate-kernel-signatures.py"
    if not generator.is_file():
        raise FileNotFoundError(f"Missing Awkward signature generator: {generator}")
    # Editable Python installs do not run this generation step. Always run the
    # checkout's own generator: both a migrated kernel and its legacy helpers
    # can disappear while an ignored, previously generated table remains.
    subprocess.run([sys.executable, str(generator)], cwd=checkout, check=True)
    print(f"Refreshed kernel signatures from {checkout}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout")
    prepare(parser.parse_args().checkout)
