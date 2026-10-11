# `14-machines`

A job is one process, the controller, and an agent on each other machine,
which opens that machine's devices and runs the work the controller submits.
The controller uses each such device through a proxy, a device of its own
whose memory is that device's, with the same calls as a device of its own
machine. This example starts itself again as an agent on the loopback, which
serves a memory device, and sends bytes there and back.

```bash
dune exec packages/rig/examples/14-machines/main.exe
```

A job across machines runs `rig agent` on each other machine, or a program
that calls `Rig_remote.listen` and `Rig_remote.serve`. Its processes share a
random key, which each reads from a file with `Rig_remote.read_key`.

## What You'll Learn

- An agent: `Rig_remote.listen`, `port`, `serve` with an opener per kind of
  device
- A controller: `Rig_remote.connect`, then the agent's machine's host and
  devices, `Rig_remote.hosts`, `Rig_remote.devices`
- A proxy's buffers, copies and submissions, ordered as on a device of this
  machine
- Ending the job: `Rig_remote.close`, after which the agent's `serve` returns

## Key Functions

| Function                           | Purpose                                   |
| ---------------------------------- | ----------------------------------------- |
| `Rig_remote.listen ~key host port` | Listen for one job's processes            |
| `Rig_remote.serve a kinds`         | Serve one job, opening devices on request |
| `Rig_remote.connect ~key agents`   | Start a job with an agent at each address |
| `Rig_remote.devices h kind`        | A proxy of each device of `kind` at `h`   |
| `Rig_remote.close j`               | End the job in order                      |

## Next Steps

[x-pci](../x-pci/) lists the machine's PCI functions, from which rig boots a
GPU that no kernel driver holds.
