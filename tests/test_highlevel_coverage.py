from types import ModuleType

import pytest

from benchmarks.highlevel.cases import Case, Unavailable
from benchmarks.highlevel.coverage import coverage, inventory


@pytest.fixture
def fake_awkward():
    ak = ModuleType("awkward")
    ak.__version__ = "2.test"
    exec(
        """
class Array:
    @property
    def fields(self):
        raise AssertionError("inventory must not evaluate properties")

    __iter__ = None

    def new_method(self):
        pass

    def _private(self):
        pass

def new_operation(array):
    return array

def measured():
    return 1

def unavailable():
    pass

def failed():
    raise ValueError("test failure")
""",
        vars(ak),
    )
    ak.str = ModuleType("awkward.operations.str")
    exec("def new_string_operation(array): return array", vars(ak.str))
    ak.imported = coverage
    return ak


def test_inventory_detects_new_apis_without_using_case_registry(fake_awkward):
    assert inventory(fake_awkward).keys() == {
        "ak.Array",
        "ak.Array.fields",
        "ak.Array.new_method",
        "ak.failed",
        "ak.measured",
        "ak.new_operation",
        "ak.str.new_string_operation",
        "ak.unavailable",
    }


def test_coverage_reports_missing_absent_failed_and_unavailable(fake_awkward):
    def unavailable(_fixtures, _target):
        raise Unavailable("optional dependency missing")

    cases = {
        "ak.old_operation": Case(lambda _, fn: fn),
        "ak.measured": Case(lambda _, fn: fn),
        "ak.failed": Case(lambda _, fn: fn),
        "ak.unavailable": Case(unavailable),
    }
    report = coverage(fake_awkward, cases=cases, smoke=True)
    rows = {row["api"]: row for row in report["coverage"]}
    assert report["awkward_version"] == "2.test"
    assert rows["ak.new_operation"]["status"] == "missing_case"
    assert rows["ak.old_operation"]["status"] == "absent"
    assert rows["ak.measured"]["status"] == "smoke_passed"
    assert rows["ak.failed"]["status"] == "failed"
    assert rows["ak.failed"]["detail"] == "ValueError: test failure"
    assert rows["ak.unavailable"]["status"] == "unavailable"
    assert rows["ak.unavailable"]["detail"] == "optional dependency missing"
