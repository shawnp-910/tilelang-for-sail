"""Specialize the two audited PPU1.5 GDN scan windows before layout inference."""

import hashlib

from tvm import IRModule, tirx
from tvm.target import Target
from tvm.tirx.stmt_functor import ir_transform, post_order_visit

# The tail barrier also protects the score tile across iterations. Until a
# general region-dependence analysis proves that hazard, restrict this
# experimental schedule to the complete IR of the two tested windows.
_VERIFIED_SCAN_IR = {
    "91547cd058b0e01f740edffb2a03c9a5918cea8498cd020846be75b771c9e22b",
    "866ecea08d45fb7e20a359bb518aaeeecb07aac3a1fc385c89187ed9f637211a",
}


def _evaluates(stmt, op_name):
    return isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call) and stmt.value.op.name == op_name


def _accesses_buffer(statements, buffer):
    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, (tirx.BufferLoad, tirx.BufferStore)) and node.buffer.same_as(buffer):
            found = True

    for statement in statements:
        post_order_visit(statement, visit)
        if found:
            break
    return found


def _reschedule_scores(stmt):
    seq = list(stmt.seq)
    barriers = [
        i
        for i, node in enumerate(seq)
        if isinstance(node, tirx.Evaluate)
        and isinstance(node.value, tirx.Call)
        and node.value.op.name.endswith(".tvm_storage_sync")
        and isinstance(node.value.args[0], tirx.StringImm)
        and node.value.args[0].value == "shared"
    ]
    if len(barriers) != 3 or barriers[-1] != len(seq) - 1:
        return None

    producers = []
    for index, node in enumerate(seq[: barriers[0]]):
        if not _evaluates(node, "tl.tileop.copy"):
            continue
        src, dst = [region.args[0].buffer for region in node.value.args]
        if src.scope() != "global" or not dst.scope().startswith("shared"):
            continue
        if _accesses_buffer(seq[index + 1 : barriers[1]], dst):
            continue
        if _accesses_buffer(seq[barriers[1] + 1 : barriers[2]], dst):
            producers.append(index)
    if len(producers) != 1:
        return None
    producer = producers[0]

    # The first barrier releases last iteration's score readers. The second
    # publishes U and the newly staged scores together. Ko is double-buffered,
    # and the next U write is already ordered by the next iteration's head.
    body = seq[:producer] + seq[producer + 1 : barriers[1]]
    body.append(seq[producer])
    body.extend(seq[barriers[1] : barriers[2]])
    return tirx.SeqStmt(body)


def schedule_verified_scan(mod: IRModule, target: Target) -> IRModule:
    if len(mod.functions) != 1 or next(iter(mod.functions)).name_hint != "scan":
        return mod
    if str(target.attrs.get("arch", "")) != "ppu_15":
        return mod
    if hashlib.sha256(mod.script().encode()).hexdigest() not in _VERIFIED_SCAN_IR:
        return mod

    @tirx.transform.prim_func_pass(opt_level=0)
    def schedule(func, _mod, _ctx):
        body = ir_transform(func.body, None, _reschedule_scores, ["tirx.SeqStmt"])
        return func.with_body(body)

    return schedule(mod)
