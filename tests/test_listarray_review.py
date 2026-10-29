# Copyright (c) 2026, Ianna Osborne
# SPDX-License-Identifier: BSD-3-Clause

import json
from types import SimpleNamespace

import pytest

from benchmarks.cuda_migrations.listarray_review import compare, shapes


def test_shapes_exceed_previous_cache_capacity():
    assert len({rows for rows, _ in shapes()}) > 32
    assert all(rows > 0 and width > 0 for rows, width in shapes())


def test_comparison_preserves_order_and_cache_metadata(tmp_path):
    paths = [tmp_path / name for name in ("baseline.json", "candidate.json")]
    for path, duration in zip(paths, [10, 5], strict=True):
        path.write_text(
            json.dumps(
                dict(
                    cases=[
                        dict(
                            case="long_next_array:4:1048576",
                            host_us=duration,
                            completed_us=duration + 1,
                        )
                    ],
                    setup_cache=dict(misses=1),
                )
            )
        )
    output = tmp_path / "review.md"
    args = SimpleNamespace(baseline=paths[0], candidate=paths[1], output=output)
    compare(args)
    assert "10.000 → 5.000" in output.read_text()
    assert "misses" in output.read_text()
    paths[1].write_text(json.dumps(dict(cases=[], setup_cache={})))
    with pytest.raises(ValueError, match="inventories differ"):
        compare(args)
