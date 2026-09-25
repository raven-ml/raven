# tolk

A port of [tinygrad](https://github.com/tinygrad/tinygrad) in OCaml. A minimal, readable ML compiler for the [Raven](https://github.com/raven-ml/raven) ecosystem.

## Build

```bash
dune build
dune test
```

## Queue profiling

Set `PROFILE=1` before running a workload. After submission, collect completed
queue timings and write a Chrome trace file:

```ocaml
let events = Tolk.Device.profile device in
Out_channel.with_open_bin "profile.json" (fun channel ->
    Tolk.Profile.output channel events)
```

Collection synchronizes the device and drains its events. Replaying the same
batch before synchronization retains its latest timestamps, matching tinygrad.
Separate batches retain separate records. Trace lanes distinguish compute and
copy queues; each device starts at its own zero because clocks are not yet
calibrated across devices.

## Reference

- [tinygrad](https://github.com/tinygrad/tinygrad) — the original project this is based on

## License

ISC
