# `08-claims`

A function handed buffers may write over an input in place only if nothing
else reads that memory. This example writes an `add` that takes its inputs
under claims, writes over the input it donates when it holds it exclusive,
and allocates otherwise.

```bash
dune exec packages/rig/examples/08-claims/main.exe
```

## What You'll Learn

- Read claims, held while the host reads or writes memory: `Buffer.read`,
  `Buffer.write`
- Donation: `Claim.with_ ~read ~donate`, and `Claim.exclusive` for a
  donated buffer that no other claim holds
- Consuming: `Claim.consume` makes every buffer made over the memory until
  then dead, so a use of one raises `Invalid_argument` with the reason. The
  write in place goes through the buffer `consume` returns.
- Buffers that are never exclusive: views of part of an allocation, buffers
  over a bigarray, and memory that `Buffer.export` exported
- Why `add` uses `Claim.with_`: it claims `x`, which `add` reads, as well as
  `y`. `Claim.take` is for a function of one input.

## Key Functions

| Function                      | Purpose                                     |
| ----------------------------- | ------------------------------------------- |
| `Buffer.read b k f`           | `f` over `b`'s bytes under a read claim     |
| `Claim.with_ ~read ~donate f` | `f` under claims on its inputs              |
| `Claim.exclusive c b`         | Whether `b` may be written in place         |
| `Claim.consume c ~why b`      | A live buffer over `b`'s memory; `b` dead   |
| `Buffer.export b k`           | `b`'s bytes as an array, outside the claims |

## Next Steps

Continue to [09-loss](../09-loss/).
