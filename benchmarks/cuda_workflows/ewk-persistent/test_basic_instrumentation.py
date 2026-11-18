import ast
import importlib.util
from pathlib import Path


def test_basic_selection_preserves_statements(monkeypatch):
    monkeypatch.setenv("DIAGNOSTIC_PHASE", "obj_basic_selection")
    path = Path(__file__).with_name("scalar_diagnostics.py")
    spec = importlib.util.spec_from_file_location("basic_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    statements = "\n".join(f"        x{i} = {i}" for i in range(40))
    source = (
        "class AnalysisProcessor:\n    def process(self):\n"
        '        print("CHUNK_TIMING phase=data_load_to_gpu seconds=0")\n'
        + statements
        + '\n        print("CHUNK_TIMING phase=obj_basic_selection seconds=0")\n'
    )
    patched = ast.parse(module.instrument(source))
    body = patched.body[0].body[0].body
    region = body[2]
    assert isinstance(region, ast.With)
    assert len(region.body) == 40
    original = ast.parse(source).body[0].body[0].body[1:-1]
    for wrapped, expected in zip(region.body, original, strict=True):
        assert ast.dump(wrapped.body[0]) == ast.dump(expected)
    assert module.PREFIX == "basic"
