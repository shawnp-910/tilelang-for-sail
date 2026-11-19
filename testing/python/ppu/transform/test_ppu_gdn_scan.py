import hashlib

import pytest

import tilelang.language as T
from tilelang import tvm
from tilelang.ppu import gdn_scan


def _scan_module(two_producers=False):
    @T.prim_func
    def scan(A: T.Tensor((64,), "float32"), O: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=32):
            scores = T.alloc_shared((64,), "float32")
            scratch = T.alloc_shared((64,), "float32")
            for _ in T.serial(4):
                T.copy(A, scores)
                if two_producers:
                    T.copy(A, scratch)
                T.sync_threads()
                if not two_producers:
                    T.copy(A, scratch)
                T.sync_threads()
                for i in T.Parallel(64):
                    O[i] = scores[i] + scratch[i]
                T.sync_threads()

    return tvm.IRModule({"scan": scan})


def _schedule_events(mod):
    events = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call):
            if node.op.name.endswith(".tvm_storage_sync"):
                events.append("sync")
            elif node.op.name == "tl.tileop.copy":
                events.append(str(node.args[1].args[0].buffer.name))

    tvm.tirx.stmt_functor.post_order_visit(mod["scan"].body, visit)
    return events


@pytest.mark.parametrize("arch,listed,rewritten", [("ppu_15", True, True), ("ppu_15", False, False), ("ppu_10", True, False)])
def test_scan_score_staging(monkeypatch, arch, listed, rewritten):
    mod = _scan_module()
    # The small fixture exercises the same dependency without shipping the
    # production operator. Authorize only this fixture for the positive case.
    if listed:
        monkeypatch.setattr(gdn_scan, "_VERIFIED_SCAN_IR", {hashlib.sha256(mod.script().encode()).hexdigest()})
    target = tvm.target.Target({"kind": "ppu", "arch": arch})
    result = gdn_scan.schedule_verified_scan(mod, target)
    if rewritten:
        assert _schedule_events(result) == ["sync", "scratch", "scores", "sync"]
    else:
        assert result.same_as(mod)
    assert _schedule_events(mod) == ["scores", "sync", "scratch", "sync", "sync"]


def test_ambiguous_staging_copies_preserve_barriers(monkeypatch):
    mod = _scan_module(two_producers=True)
    monkeypatch.setattr(gdn_scan, "_VERIFIED_SCAN_IR", {hashlib.sha256(mod.script().encode()).hexdigest()})
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})
    result = gdn_scan.schedule_verified_scan(mod, target)
    assert _schedule_events(result) == ["scores", "scratch", "sync", "sync", "sync"]
