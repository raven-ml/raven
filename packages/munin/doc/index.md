# munin

munin records what a program does as it runs: its params, its metrics step
by step, readings of its machine and devices, the files it logs, and
versioned artifacts such as checkpoints. It writes them to a store, a
directory of JSON and text files, and reads them back. No server is
involved, and `jq`, `tail -f` or another program reads a store as well as
munin does.

## Installation

<!-- $MDX skip -->
```sh
opam install munin
```

A program links the `munin` library, and `munin.sys` for readings of the
machine:

<!-- $MDX skip -->
```lisp
(executable
 (name main)
 (libraries munin munin.sys))
```

## A first run

This program trains nothing. It logs a loss that falls as `100 / (step + 1)`
for 100 steps, which is enough to show every part of a run:

```ocaml
let () =
  let store = Munin.Store.open_ () in
  Munin.Session.within
    (Munin.Session.start ~store ~experiment:"quickstart"
       ~params:[ ("lr", `Float 0.1); ("batch", `Int 32) ]
       ~steps:99 ())
  @@ fun s ->
  let loss = Munin.Session.metric s ~goal:`Minimize "train/loss" in
  for step = 0 to 99 do
    Munin.Session.log loss ~step (100. /. Float.of_int (step + 1))
  done
```

- `Munin.Store.open_ ()` is the store in the directory named by
  `$RAVEN_TRACKING_DIR`, or else `raven/munin` under `$XDG_DATA_HOME` or
  `~/.local/share` (`Store.open_` in [`store.mli`](../lib/store.mli) gives
  the whole rule).
  Opening a store creates nothing; the first write creates the directory.
- `Munin.Session.start` creates a run in `store` and returns its session,
  the one writer of that run. `~experiment` names what the run is an
  attempt at, and runs of one experiment are compared with each other.
  `~params` are the run's settings, each a `Munin.Value.t`. `~steps` is the
  largest step the run will log, from which readers show its progress. A
  run's start never changes after it is written.
- `Munin.Session.within s f` calls `f s`, then ends the run as finished. If
  `f` raises, it ends the run as failed with the exception's text and
  raises the exception again.
- `Munin.Session.metric s ~goal:`Minimize "train/loss"` declares the metric
  `train/loss` and returns the handle that logs it. `~goal` says that lower
  values are better, so readers can pick the best sample. A key is one or
  more segments joined by `/` (the Names section of
  [`store_format.mli`](../lib/store_format.mli)).
- `Munin.Session.log loss ~step v` appends a sample of `train/loss`: the
  value `v` at `step`, at the current time.

Each `log` appends a record that a reader in another process sees once
the call returns, and `within` writes the run's end whether the loop
returns or raises, so the run reads as finished or failed once the program
exits (the preamble of
[`session.mli`](../lib/session.mli)).

## Reading it back

A reader lists the store's runs, newest first, and folds a run's log into
its history:

```ocaml
# let run = List.hd (Munin.Store.runs (Munin.Store.open_ ()));;
val run : Munin.Run.t = <abstr>
# (Munin.Run.experiment run, Munin.Run.status run);;
- : string * Munin.Run.status = ("quickstart", `Finished)
# Munin.Run.params run;;
- : (string * Munin.Value.t) list = [("lr", `Float 0.1); ("batch", `Int 32)]
# let h = Munin.History.read run;;
val h : Munin.History.t = <abstr>
# Munin.Metric.length (Munin.History.series h "train/loss");;
- : int = 100
# Option.map
    (fun (b : Munin.Metric.sample) -> (b.step, b.value))
    (Munin.History.best h "train/loss");;
- : (int * float) option = Some (99, 1.)
```

`Store.runs` reads only the first and last records of each log, so listing
a store costs the same whatever the runs logged. `History.read` reads the
whole log once.

The `munin` command prints the same from a shell:

<!-- $MDX skip -->
```sh
munin runs
munin show 20261101T140307Z-a1f3c9e2
munin metrics 20261101T140307Z-a1f3c9e2 train/loss
```

## Guides

- [Tracking runs](01-tracking.md): sessions, metrics and steps, reading a
  run, resuming and orphans, summaries, tags and notes, files and
  provenance.
- [Artifacts](02-artifacts.md): versions, digests, aliases and which run
  used what.
- [System monitoring](03-system-monitoring.md): readings of the machine,
  the process and devices, sampled into a run.
- [The munin command](04-cli.md): each command and what it prints.
- [Reading a store without munin](05-files.md): the files, with `cat`, `jq`
  and `tail -f`.

The `.mli` files in [`lib/`](../lib) state every rule these pages rely on,
starting with [`munin.mli`](../lib/munin.mli).
