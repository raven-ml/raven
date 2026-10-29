# Reviewing nx

## Design

- A new operation: does it compose from existing ones? A composite enters
  the interface only when composing it correctly takes knowledge the caller
  lacks, such as a stable form.
- A fix inside one kernel or in nx.cpu: is the fact it handles missing
  from the dtype, the layout, the route or `Nx_kernel`'s contract, so that
  every kernel library would need the same fix?
- A per-dtype or per-device branch in frontend code: which layer should know
  that fact?
- Work on the host for a set without kernels, a fallback after a decline,
  or bytes moved between devices outside `Nx.place` and a fold's partials.
- A change to `Nx_kernel.S`: was it asked for, and does each new case have
  its `Shape` rule and its decline bounds?
- `nx.quant`, `nx.ragged` or `nx.wide` reaching past `Nx` and `nx.array`,
  or gaining C.

## Semantics

References: IEEE 754-2019 for floating point, ISO C99 Annex G for complex
special values, and what each `.mli` cites. NumPy is a datapoint for names
and behaviour users expect.

Failure here is a wrong number, or one that changes with layout, thread
count, placement or platform. Most of nx's fixes are this class: libm
differs on Windows, x87 precision leaks through, a narrow float computes
at its own width.

- Every dtype the operation takes: `int4`, `uint4` and `bit` packed below a
  byte, the 8- and 4-bit floats that lack infinities or NaN, complex parts.
- Views: offsets, negative and zero strides, broadcast operands. An
  operand with no elements still needs a valid address. A destination
  shares no byte with an operand but one it reads only at the result's own
  index.
- A kernel answers `Declined` or a refusal before any write or queued work.
- A change to an association, a lane order or an algorithm changes bits:
  the library's `.mli` states the new one.
- NaN quieting, signed zeros and infinities through reductions and scans.
- A kind computed outside `nx_kinds.h`'s functions gives their bits. No
  libm or libc call on a result path.
- Donation: a consumer writes in place only where `Nx.donate`'s conditions
  hold, and a refused operation kills no handle.
- A new operation's result placement and its routing error follow
  `nx.mli`'s Routing section. A random draw is the same at every placement.

## Skip

`test/golden/`, `test/*/golden/`, `bench/*/*.thumper`, `LICENSE-*`.
