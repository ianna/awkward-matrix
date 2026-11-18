"""Validate the downloaded 1m sample and create an absolute local input JSON."""

import json
from pathlib import Path

import uproot

root = Path.cwd()
sample = root / "data/mega_rep_1m.root"
template = root / "columnar_gpu/ewkcoffea_on_gpu/reproduce/input_jsons/rep_1m.json"
with uproot.open(sample) as source:
    entries = source["Events"].num_entries
assert entries >= 600000, f"Need several 200k chunks, found {entries} events"
data = json.loads(template.read_text())
data["path"], data["files"] = "", [str(sample.resolve())]
output = root / "data/rep_1m-local.json"
output.write_text(json.dumps(data, indent=2))
print(f"Validated {sample}: {entries} events; wrote {output}")
