# `08-loss`

A device whose work fails is lost, once and for good. This example loses a
memory device with a fill that answers a failure, `fail.c`, and shows what
still works: the device's facts, other devices, and its name, which opens a
new device.

```bash
dune exec dev/rig/examples/08-loss/main.exe
```

A GPU is lost the same way when its driver reports a fault, such as a page
fault of a kernel. Work that runs long is no loss: only the driver decides that
work failed.

## What You'll Learn

- `Lost`, raised by the failing submit and by every later use of the device
- Memory whose stamps name a lost device raises `Lost` too
- A lost device's facts: `lost`, `name`, `submitted`, `signaled`
- Other devices go on
- Opening the name again makes a new device: `equal` tells them apart

## Key Functions

| Function          | Purpose                                         |
| ----------------- | ----------------------------------------------- |
| `Lost (d, why)`   | Raised by every use of a lost device `d`        |
| `lost d`          | `Some why` once `d` is lost; raises nothing     |
| `signaled d`      | The last value `d`'s work reached               |
| `equal d d'`      | Whether two devices are the same                |

## Next Steps

Continue to [09-profiles](../09-profiles/).
