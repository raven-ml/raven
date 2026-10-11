# `09-loss`

A device whose work fails is lost, once and for good. This example loses a
memory device with a fill that returns a failure (`fail.c`). The lost
device's properties and timeline still answer, another device goes on, and
the name opens a new device.

```bash
dune exec packages/rig/examples/09-loss/main.exe
```

A GPU is lost the same way when its driver reports a fault, such as a page
fault of a kernel, and when its work makes no progress past the hang bound
of a driver that declares one.

## What You'll Learn

- `Lost`, raised by the failing submit and by every later use of the device
- A lost device's memory raises `Lost` too
- A lost device's properties and timeline: `lost`, `name`, `submitted`,
  `signaled`
- Other devices go on
- Opening the name again makes a new device, which `equal` tells apart

## Key Functions

| Function        | Purpose                                     |
| --------------- | ------------------------------------------- |
| `Lost (d, why)` | Raised by every use of a lost device `d`    |
| `lost d`        | `Some why` once `d` is lost; raises nothing |
| `signaled d`    | The last value `d` has reported             |
| `equal d d'`    | Whether two devices are the same            |

## Next Steps

Continue to [10-profiles](../10-profiles/).
