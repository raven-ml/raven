#!/usr/bin/env python3
"""Generates bnxt_defs.ml, the Broadcom NetXtreme definitions nx.rdma.device
reads.

Run from the repository root:

  uv run --with libclang==18.1.1 packages/nx/lib/rdma/device/gen/gen.py
  uv run --with libclang==18.1.1 packages/nx/lib/rdma/device/gen/gen.py --check

The inputs are five headers of Linux v6.18, pinned by URL and SHA-256 in
pins.json: the firmware interface (hsi.h), the RoCE engine's interface
(roce_hsi.h), and the constants of the kernel drivers that talk to them
(qplib_rcfw.h, qplib_res.h, bnxt_hwrm.h). Downloads are kept in --cache.

Struct layouts come from libclang, for x86_64 Linux; the script checks that
aarch64 Linux lays them out the same. It emits the constants and the structs
the runtime reads, which the lists below name: every field of those structs.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent
sys.path.insert(0, str(HERE.parents[2] / "device" / "gen"))
from devgen import Unit, fetch, key, layout, main, ml_int, stub_dir  # noqa: E402

LINUX = "https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/"
TAG = "v6.18"
FILES = {
    "hsi.h": "include/linux/bnxt/hsi.h",
    "roce_hsi.h": "drivers/infiniband/hw/bnxt_re/roce_hsi.h",
    "qplib_rcfw.h": "drivers/infiniband/hw/bnxt_re/qplib_rcfw.h",
    "qplib_res.h": "drivers/infiniband/hw/bnxt_re/qplib_res.h",
    "bnxt_hwrm.h": "drivers/net/ethernet/broadcom/bnxt/bnxt_hwrm.h",
}

# The firmware's requests (HWRM) and the RoCE engine's commands (RCFW) the
# runtime sends.
HWRM = ["ver_get", "func_reset", "func_qcaps", "func_drv_rgtr", "func_drv_unrgtr", "func_qcfg",
        "func_backing_store_qcaps_v2", "func_backing_store_cfg_v2", "ring_alloc", "stat_ctx_alloc",
        "vnic_alloc", "vnic_cfg", "cfa_l2_filter_alloc"]
RCFW = ["initialize_fw", "add_gid", "create_cq", "create_qp", "modify_qp", "register_mr", "deregister_mr"]

CONSTANTS = [
    *[f"HWRM_{n.upper()}" for n in HWRM],
    *[f"CMDQ_BASE_OPCODE_{n.upper()}" for n in RCFW],
    # HWRM
    "HWRM_MAX_REQ_LEN", "BNXT_HWRM_TARGET", "BNXT_HWRM_NO_CMPL_RING",
    "FUNC_BACKING_STORE_CFG_V2_REQ_FLAGS_BS_CFG_ALL_DONE",
    "RING_ALLOC_REQ_RING_TYPE_NQ", "RING_ALLOC_REQ_RING_TYPE_L2_CMPL", "RING_ALLOC_REQ_RING_TYPE_RX",
    "RING_ALLOC_REQ_INT_MODE_MSIX", "RING_ALLOC_REQ_ENABLES_NQ_RING_ID_VALID",
    "RING_ALLOC_REQ_ENABLES_RX_BUF_SIZE_VALID",
    "VNIC_CFG_REQ_ENABLES_MRU", "VNIC_CFG_REQ_ENABLES_DEFAULT_RX_RING_ID",
    "VNIC_CFG_REQ_ENABLES_DEFAULT_CMPL_RING_ID",
    "CFA_L2_FILTER_ALLOC_REQ_FLAGS_PATH_RX", "CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_ADDR",
    "CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_ADDR_MASK", "CFA_L2_FILTER_ALLOC_REQ_ENABLES_DST_ID",
    # doorbells
    "DBC_DBC_INDEX_MASK", "DBC_DBC_XID_MASK", "DBC_DBC_PATH_ROCE", "DBC_DBC_TYPE_SQ", "DBC_DBC_TYPE_RQ",
    "DBC_DBC_TYPE_CQ", "DBC_DBC_TYPE_NQ_ARM", "BNXT_QPLIB_DBR_VALID", "BNXT_QPLIB_DBR_EPOCH_SHIFT",
    # the RoCE engine's command queue
    "RCFW_COMM_BASE_OFFSET", "RCFW_PF_VF_COMM_PROD_OFFSET", "RCFW_COMM_TRIG_OFFSET", "RCFW_CMDQ_TRIG_VAL",
    "FIRMWARE_FIRST_FLAG", "CMDQ_INIT_CMDQ_SIZE_SFT", "CMDQ_INITIALIZE_FW_FLAGS_HW_REQUESTER_RETX_SUPPORTED",
    "CREQ_BASE_V", "CREQ_BASE_TYPE_QP_EVENT", "CREQ_QP_EVENT_EVENT_QP_ERROR_NOTIFICATION",
    "PTU_PTE_VALID", "PTU_PTE_LAST", "PTU_PTE_NEXT_TO_LAST",
    # queue pairs
    "CMDQ_CREATE_QP_TYPE_RC", "CMDQ_MODIFY_QP_QP_TYPE_RC", "CMDQ_MODIFY_QP_NETWORK_TYPE_ROCEV2_IPV4",
    "CMDQ_MODIFY_QP_PATH_MTU_MTU_4096", "CMDQ_MODIFY_QP_NEW_STATE_INIT", "CMDQ_MODIFY_QP_NEW_STATE_RTR",
    "CMDQ_MODIFY_QP_NEW_STATE_RTS", "CMDQ_MODIFY_QP_ACCESS_LOCAL_WRITE", "CMDQ_MODIFY_QP_ACCESS_REMOTE_WRITE",
    *[f"CMDQ_MODIFY_QP_MODIFY_MASK_{m}" for m in [
        "STATE", "ACCESS", "PKEY", "DGID", "SGID_INDEX", "HOP_LIMIT", "DEST_MAC", "PATH_MTU", "RQ_PSN",
        "MIN_RNR_TIMER", "MAX_DEST_RD_ATOMIC", "DEST_QP_ID", "TIMEOUT", "RETRY_CNT", "RNR_RETRY",
        "MAX_RD_ATOMIC", "SQ_PSN"]],
    # memory regions
    "CMDQ_REGISTER_MR_FLAGS_ALLOC_MR", "CMDQ_REGISTER_MR_LVL_SFT", "CMDQ_REGISTER_MR_LOG2_PG_SIZE_SFT",
    "CMDQ_REGISTER_MR_ACCESS_LOCAL_WRITE", "CMDQ_REGISTER_MR_ACCESS_REMOTE_WRITE",
    # work queues and completions
    "SQ_BASE_WQE_TYPE_SEND", "SQ_SEND_FLAGS_SIGNAL_COMP", "RQ_WQE_WQE_TYPE_RCV",
    "SQ_MSN_SEARCH_START_IDX_SFT", "SQ_MSN_SEARCH_NEXT_PSN_SFT", "CQ_BASE_TOGGLE",
]

STRUCTS = [
    ("hsi.h", "hwrm_cmd_hdr"), ("hsi.h", "hwrm_resp_hdr"),
    *[("hsi.h", f"hwrm_{n}_{d}") for n in HWRM for d in ("input", "output")],
    ("roce_hsi.h", "cmdq_init"), ("roce_hsi.h", "cmdq_base"), ("roce_hsi.h", "creq_base"),
    *[("roce_hsi.h", f"cmdq_{n}") for n in RCFW],
    *[("roce_hsi.h", f"creq_{n}_resp") for n in RCFW],
    ("roce_hsi.h", "sq_send_hdr"), ("roce_hsi.h", "sq_sge"), ("roce_hsi.h", "rq_wqe_hdr"),
    ("roce_hsi.h", "cq_base"), ("roce_hsi.h", "sq_msn_search"),
]


def sources(cache, pins, pin):
    """The directory of the headers, under the digest of the tag."""
    root = cache / "src" / key(LINUX + TAG)
    root.mkdir(parents=True, exist_ok=True)
    for name, path in FILES.items():
        (root / name).write_bytes(fetch(cache, f"{LINUX}{path}?h={TAG}", pins, pin).read_bytes())
    return root


def fields(ci, unit, cname):
    c = unit.struct(cname)
    if c is None:
        sys.exit(f"no struct {cname}")
    size, fs = layout(ci, c)
    if not fs:
        sys.exit(f"{cname} has no fields: its header did not parse")
    counts = {}

    def walk(t, prefix):
        for f in t.get_fields():
            ft = f.type.get_canonical()
            path = (prefix + "__" if prefix else "") + f.spelling
            if ft.kind == ci.TypeKind.CONSTANTARRAY:
                counts[path] = ft.get_array_size()
            elif ft.kind == ci.TypeKind.RECORD:
                walk(ft, path)

    walk(c.type.get_canonical(), "")
    return size, {k: (v[0], v[1], counts[k]) if k in counts else v for k, v in fs.items() if v[0] != "bits"}


# Big-endian fields, which the prelude does not name.
TYPES = "typedef unsigned short __be16; typedef unsigned int __be32; typedef unsigned long long __be64;\n"

KEYWORDS = {"type", "method", "val", "open", "end", "include", "module", "object", "private"}


def ml_name(f):
    s = f.replace("__", "_")
    return s + "_" if s in KEYWORDS else s


def ml_const(v):
    """A constant of 64 bits, as C evaluated it: negative when its top bit is set."""
    return ml_int(v - (1 << 64)) if v >= 1 << 63 else ml_int(v)


def generate(cache, pins, pin, outdir):
    import clang.cindex as ci
    root = sources(cache, pins, pin)
    stub = stub_dir("bnxt")
    units = {h: Unit(ci, [root / h], [root], stub, extra=TYPES) for h in FILES}
    arms = {h: Unit(ci, [root / h], [root], stub, extra=TYPES, target="aarch64-unknown-linux-gnu")
            for h in ("hsi.h", "roce_hsi.h")}
    values = {}
    for u in units.values():
        values.update(u.enums())
        values.update(u.macros([c for c in CONSTANTS if c not in values]))
    missing = [c for c in CONSTANTS if c not in values]
    if missing:
        sys.exit(f"undefined constants {missing}")
    out = ["(* Generated by gen.py; do not edit. The inputs and the command that",
           "   regenerates this file are in gen.py; their digests are in pins.json. *)", ""]
    out.append("(* Constants *)")
    out.append("")
    for c in CONSTANTS:
        v = ml_const(values[c])
        out.append(f"let {c.lower()} = {'(' + v + ')' if v.startswith('-') else v}")
    out.append("")
    out.append("(* Structs: each field is (byte offset, bytes), an array's (offset, bytes")
    out.append("   of an element, elements). *)")
    out.append("")
    for header, cname in STRUCTS:
        size, fs = fields(ci, units[header], cname)
        if fields(ci, arms[header], cname) != (size, fs):
            sys.exit(f"{cname} differs on aarch64")
        out.append(f"module {cname.capitalize()} = struct")
        out.append(f"  let sizeof = {size}")
        for f in sorted(fs, key=lambda f: (fs[f][0], f)):
            out.append(f"  let {ml_name(f)} = ({', '.join(ml_int(x) for x in fs[f])})")
        out.append("end")
        out.append("")
    (outdir / "bnxt_defs.ml").write_text("\n".join(out).rstrip() + "\n")


if __name__ == "__main__":
    main(__doc__, generate, ["bnxt_defs.ml"], HERE, OUT, "bnxt-gen")
