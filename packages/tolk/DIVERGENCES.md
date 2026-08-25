# Divergences

Intentional divergence from the tinygrad reference, in both directions.
Standing rulings; work items live in TODO.md. Anchors reference the tinygrad
clone pin named there.

## Declined ports

Do not re-open without a consumer.

- **Void-RANGE loop header** (`b764599d8`, `39924387b`, its `spec.py` rules).
  Serves HCQ wait loops, a runtime tolk does not port; nothing in tolk
  constructs a void range, so every branch would be unreachable and no test
  could catch a bad port. `Axis_type.Loop` names the concept, deliberately
  without a constructor. Port together with a producer and its test.

- **`Renderer.abi` and CALL-in-C** (`d1f215d37`, `6e979b879`). Both serve the
  CPU uop worker runtime. `abi` only feeds `kernel_typedef`, which tolk spells
  statically; CALL-in-C renders an address-valued CALL nothing constructs.
  Port together with a host-callable runtime.

- **`ParamArg.volatile`** (`a6fda6b10`). Producers are `ops_cpu.py`,
  `ops_qcom.py`, `hcq2.py`; tolk would carry an always-false field. Add with
  the first runtime that needs uncached parameter memory.

- **`ProgramInfo.target`; tolk's `aux` deleted rather than renamed**
  (`9fdaa4bff`). `target` exists so `UOp.to_elf()` can build a `TinyELF` — a
  runtime restructure tolk does not port; `Program_spec.t` already carries
  `device`. `aux` had no reader. (`call_info.aux` is unrelated and stays.)

- **Host-scalar `bitcast` stays in `symbolic.ml`**, upstream moved it to
  `dtype.ml` (`67dc02d7e`). Forced by layering: `Const` depends on `Dtype`.
  The only would-be consumer is host-side rand arithmetic, and
  `lib/frontend/rand.ml` builds `_bits_to_rand` from UOp `Bitcast` nodes.

## Tolk extensions

Code tolk carries that the reference does not. Every site has a comment
containing the phrase "tinygrad counterpart" — `grep -rn "tinygrad
counterpart" lib` lists them. An extension needs a consumer; without one,
delete it rather than registering it.

- **`CALL(CUSTOM_FUNCTION "loop")` — the staged-scan loop**
  (`engine/realize.ml` `exec_loop`; `schedule/rangeify.ml` `find_bufs`
  walking `enter_calls:false` and `split_store` passing a precompiled CALL
  through as its own kernel; builder with its consumer in rune's `jit.ml`).
  Rune stages `Rune.scan` as one compiled body replayed per slice
  (`rune/doc/05-staged-scan.md`); the reference's answer to a recurrence is
  unrolling plus TinyJit, so there is nothing to port. The named-payload
  mechanism is upstream's own ("graph", "encdec", "hcq"); "loop" is a
  tolk-local name in it, and no tinygrad-shaped graph can reach the new
  branches. Pin moves must keep the two rangeify branches — a re-sync of
  `rangeify.py` will not find them upstream. The tolk corpus cannot build a
  loop call; rune's `test_jit.ml` scan groups are this extension's parity
  suite.
