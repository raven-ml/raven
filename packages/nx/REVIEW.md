# Reviewing nx

## Design

- A new operation: does it compose from existing ones? A composite enters
  the interface only when composing it correctly takes knowledge the caller
  lacks, such as a stable form.
- A fix inside one kernel or one kernel library: is the fact it handles
  missing from the dtype, the layout, the route or `Nx_kernel`'s contract,
  so that every library would need the same fix?
- A per-dtype or per-device branch in frontend code: which layer should know
  that fact?
- An operation that would compute on the host for a device that lacks the
  kernel.

## Semantics

References: IEEE 754-2019 for floating point, and what each `.mli` cites.
NumPy is a datapoint for names and behaviour users expect.

Failure here is a wrong number, or one that differs by platform or backend.
Most of nx's fixes are this class: libm's `sin`, `round`, `pow`, `fma` and
`expm1` differ on Windows, x87 precision leaks through, a narrow float
computes at its own width, a float prints or parses rounded twice.

- Every dtype the operation accepts, packed 4-bit types included, and the
  null entries of a kernel table rejected before the call.
- Views: offsets, negative and zero strides, broadcast operands, an operand
  with no elements (it still needs a valid address).
- NaN quieting, signed zeros and infinities through reductions and scans.
- A call into libm or libc on a result path. nx computes these with its own
  code.

## Skip

`test/golden/`, `bench/*.thumper`, `LICENSE-*`.
