"""Check that archived benchmark sources preserve their recorded provenance."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parents[1] / "benchmarks" / "cuda_workflows"


def test_archived_source_hashes():
    manifest = json.loads((WORKFLOWS / "source-snapshots.json").read_text())
    assert manifest["snapshots"]
    for snapshot in manifest["snapshots"]:
        source = WORKFLOWS / snapshot["file"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == snapshot["sha256"]
        assert len(snapshot["commit"]) == 40


@pytest.mark.skipif(shutil.which("bash") is None, reason="shell syntax needs bash")
def test_della_batch_shell_syntax():
    for script in WORKFLOWS.rglob("*.sbatch"):
        subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
