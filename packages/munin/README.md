# Munin

Run tracking for [Raven](https://github.com/raven-ml/raven), in plain
files.

munin records what a program does as it runs: its params, its metrics step
by step, readings of its machine and devices, the files it logs, and
versioned artifacts such as checkpoints. A store is a directory of JSON
and text files, specified in `lib/store_format.mli`, so `jq`, `tail -f` or
another program reads it as well as munin does. No server and no account
is involved.

## Quick start

```ocaml
let () =
  Munin.Session.within
    (Munin.Session.start ~experiment:"mnist"
       ~params:[ ("lr", `Float 0.001); ("epochs", `Int 10) ]
       ())
  @@ fun s ->
  let loss = Munin.Session.metric s ~goal:`Minimize "train/loss" in
  for step = 1 to 1000 do
    Munin.Session.log loss ~step (train_step ())
  done
```

```sh
munin runs                     # id, status, start, experiment, name, tags
munin metrics RUN train/loss   # step, time, value
munin compare RUN RUN          # params that differ, each metric's best
```

## Features

- **Runs**: one append-only log per run, written by one session; status
  read from the log and the writer's lock (running, crashed, finished,
  failed, killed)
- **Metrics**: declared once, logged through a handle, with goals and an x
  axis of another metric; statistics and best values that a NaN never
  takes
- **Resume**: a crashed run continues from a saved step, and the samples
  it abandoned become orphans that readers leave out
- **Readings**: CPU, memory and disk of the machine and the process, and
  any device the program reads, sampled on a schedule (`munin.sys`)
- **Artifacts**: immutable numbered versions stored once by digest,
  aliases, and which run produced and used each one
- **Tags and notes**: changed by any process, while a run is written or
  after it ended
- **Provenance**: command, git commit and uncommitted diff, toolchain,
  machine and seeds
- **CLI**: header-free tab-separated output for `cut`, `awk` and `sort`

## Libraries

| Library | Description |
|---------|-------------|
| `munin` | Sessions, stores, runs, histories and artifacts |
| `munin.sys` | Readings of the machine, the process and devices |

## Documentation

The guides are in [`doc/`](doc/index.md), and the examples in
[`examples/`](examples/README.md).

## Contributing

See the [Raven monorepo README](../../README.md) for guidelines.

## License

ISC License. See [LICENSE](../../LICENSE) for details.
