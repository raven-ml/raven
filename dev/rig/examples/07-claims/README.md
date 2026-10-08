# `07-claims`

A function handed buffers may write over an input in place only if nothing else
sees that memory. This example writes an `add` that takes its inputs under
claims, writes over a donated input when it is exclusive, and allocates
otherwise.

```bash
dune exec dev/rig/examples/07-claims/main.exe
```

## What You'll Learn

- Read claims, held while the host reads memory: `Claim.read`,
  `Claim.release`
- Donation: `Claim.with_ ~read ~donate`, and `Claim.exclusive` for the donated
  buffers nothing else claims
- Consuming: `Claim.consume` makes every older buffer over the memory dead, so a
  stale name raises `Invalid_argument` with the reason
- Buffers that are never exclusive: views of part of a memory, and bigarrays
  the program holds

## Key Functions

| Function                         | Purpose                                     |
| -------------------------------- | ------------------------------------------- |
| `Claim.read b`, `Claim.release b` | Claim `b`'s memory for reading, and end it |
| `Claim.with_ ~read ~donate f`    | `f` under claims on inputs and donations    |
| `Claim.exclusive c b`            | Whether `b` may be written in place         |
| `Claim.consume c ~why b`         | A live name for `b`'s memory; `b` is dead   |

## Next Steps

Continue to [08-loss](../08-loss/).
