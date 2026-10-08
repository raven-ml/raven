# `x-metal-kernel`

**Needs a Mac whose GPU supports Metal, on macOS 15 or later.** Elsewhere it
prints a line and exits.

A kernel reaches a Metal device through a fill, `run.c`, that executes an
indirect command buffer. This example loads a metallib, records one dispatch of
its kernel `add` over a million floats, runs it as a step (fixed memory in a
hold, arrays passed to each submit) and checks the result.

```bash
cd dev/rig/examples/x-metal-kernel
dune exec ./main.exe
```

## What You'll Learn

- Loading code on a device: `Image.load`, `Image.entry`
- What compiled code finds in a Metal device's capability:
  `Rig.capability g Rig_metal_abi.key`, its `icb` and `align`
- An argument buffer of GPU addresses: `Buffer.address`, `Buffer.offset`,
  `Buffer.handle`
- A fill in Objective-C that executes the indirect command buffer
- A hold whose release ends the indirect command buffer once the work is done

## Key Functions

| Function                           | Purpose                                     |
| ---------------------------------- | ------------------------------------------- |
| `Image.load g metallib`            | The image on `g`                            |
| `Image.entry p "add"`              | Its kernel's pipeline                       |
| `cap.icb buffer dispatches`        | Record dispatches once                      |
| `Hold.make ~release bs`            | Keep the step's memory and objects          |

## The metallib

`add.metallib` is `add.metal` compiled in this directory on macOS 26.3.1 with
Xcode 26.3's Metal toolchain (`metal` 32023.864):

```bash
xcrun -sdk macosx metal -c add.metal -o add.air
xcrun -sdk macosx metallib add.air -o add.metallib
rm add.air
```
