# The munin command

`munin` reads a store and changes what may change in it: a run's tags and
notes, an artifact's aliases, and what is deleted. Each command prints
tab-separated lines without a header, one record a line, so that `cut`,
`awk`, `sort` and `grep` take its output as it is. A value prints as
`Value.pp` prints it ([`value.mli`](../lib/value.mli)): a float in the
fewest digits that read back to it, a NaN as `nan`, and a string with its
tabs and line feeds escaped, so a value never splits a line. Times print
in RFC 3339, in UTC. An error goes to stderr, and the command exits with
status 1.

Every command takes `--root DIR`. Without it, the store is the one
`Store.open_ ()` opens ([`store.mli`](../lib/store.mli)): the directory in
`$RAVEN_TRACKING_DIR`, or else under `$XDG_DATA_HOME` or
`~/.local/share`. A `--root` that does not exist is an error, so a
mistyped one fails.

The output below comes from the store the [examples](../examples) fill.

## Runs

`munin runs` lists the runs, newest first: id, status, start, experiment,
name and tags, with `-` for an absent field. `--experiment E`,
`--status S`, `--tag T` (repeated for every tag) and `--parent ID` keep
the runs that match.

<!-- $MDX skip -->
```sh
$ munin runs --experiment optimisation
20261101T222627Z-38b38cbb	finished	2026-11-01T22:26:27.558396Z	optimisation	conservative	-
20261101T222627Z-388a34ff	finished	2026-11-01T22:26:27.514486Z	optimisation	moderate	-
20261101T222627Z-3a364279	finished	2026-11-01T22:26:27.469755Z	optimisation	aggressive	-
20261101T222627Z-02c598e9	finished	2026-11-01T22:26:27.433609Z	optimisation	sweep-1	-
```

The status is one of `running`, `silent`, `crashed`, `finished`, `failed`
and `killed` (`Run.status` in [`run.mli`](../lib/run.mli)). The runs that
died overnight are

<!-- $MDX skip -->
```sh
munin runs --status crashed
```

`munin show RUN` prints one run as `key value` lines: its start, its
status, the error of a failed run, its end, params as `param.<key>`,
provenance as `provenance.<field>`, each tag as a `tag` line, notes, the
summary as
`summary.<key>`, and for each metric its `count`, `last`, `best` and
`non_finite`. A field the run lacks prints no line, so `grep ^ended`
tells whether the run ended:

<!-- $MDX skip -->
```sh
$ munin show 20261101T222626Z-007549cd
id	20261101T222626Z-007549cd
experiment	solver
name	conjugate-gradient
status	finished
started	2026-11-01T22:26:26.163641Z
ended	2026-11-01T22:26:26.169240Z
updated	2026-11-01T22:26:26.169248Z
steps	500
param.tolerance	1e-06
param.max_iter	500
provenance.command	/home/u/raven/_build/default/packages/munin/examples/02-metrics/main.exe
...
metric.residual.count	20
metric.residual.last	0.002252805587295138
metric.residual.best	0.002252805587295138
metric.residual.non_finite	0
```

## Metrics

`munin metrics RUN` lists each key of the run that has a sample: key,
kind (`metric` or `reading`), count, last value, best value (`-` without a
goal) and the number of samples that are not finite. It reads the
statistics an ended run keeps beside its log, so it costs the same for a
long run as for a short one (`History.stats_of` in
[`history.mli`](../lib/history.mli)).

<!-- $MDX skip -->
```sh
$ munin metrics 20261101T222626Z-007549cd
convergence_rate	metric	20	0.7265482042610354	-	0
epoch	metric	20	20.0	-	0
residual	metric	20	0.002252805587295138	0.002252805587295138	0
```

`munin metrics RUN KEY` prints the key's series, orphans left out: step,
time and value, in log order.

<!-- $MDX skip -->
```sh
$ munin metrics 20261101T222626Z-007549cd residual | head -3
25	2026-11-01T22:26:26.168807Z	0.7583887377325693
50	2026-11-01T22:26:26.168829Z	0.5890780507082984
75	2026-11-01T22:26:26.168852Z	0.44490050687412774
```

The step of the least residual is

<!-- $MDX skip -->
```sh
munin metrics 20261101T222626Z-007549cd residual | sort -t "$(printf '\t')" -k3,3g | head -1 | cut -f1
```

`munin compare RUN...` prints one column per run: a `param.<key>` row for
each param the runs do not all give the same value, then a
`metric.<key>` row for each metric with its best value, or its last
without a goal. A star marks the best of a row whose runs share a goal.
A NaN or an infinity never takes the star.

<!-- $MDX skip -->
```sh
$ munin compare $(munin runs --parent 20261101T222627Z-02c598e9 | cut -f1)
param.max_iter	200	100	50
param.step_size	0.001	0.01	0.1
metric.error	9.098581862777884	3.973154951711871	0.10272175958485176*
```

## Artifacts

`munin artifacts [NAME]` lists the versions, by name, then by number:
name, version, digest, size in bytes, producing run, step, and the aliases
that name it.

<!-- $MDX skip -->
```sh
$ munin artifacts
analysis	1	blake2b-256:935ed569d42370a66d7ab37e632f62954422c1e3629b620e04c7f9da8554152e	25	20261101T222626Z-1b9947e6	-	reviewed
measurements	1	blake2b-256:a5ecbd1f67bc64c46c3b72fff516a42c0a2855f27793098809ba33e4a2c8fa7a	49	20261101T222626Z-28304bb9	-	-
```

`munin alias NAME VERSION ALIAS` makes `ALIAS` name that version, and
`munin alias --remove NAME ALIAS` removes it. `munin rm-artifact NAME
VERSION` deletes a version and the aliases that name it
(`Store.delete_artifact` in [`store.mli`](../lib/store.mli)).

## Tags and notes

`munin tag RUN TAG...` adds tags to a run, and `munin tag --remove RUN
TAG...` removes them. `munin note RUN TEXT` replaces the run's notes,
`munin note RUN` prints them, and `munin note --clear RUN` removes them.
Each works while the run is written and after it ended.

<!-- $MDX skip -->
```sh
$ munin tag 20261101T222625Z-038c51d3 baseline
$ munin note 20261101T222625Z-038c51d3 "Loss flat after step 60."
$ munin runs --tag baseline
20261101T222625Z-038c51d3	finished	2026-11-01T22:26:25.779050Z	demo	baseline	baseline
```

## The dashboard

`munin` without a command shows the store's runs on the terminal, and
`munin watch [RUN]` one run as it trains ([The dashboard](06-dashboard.md)).
Without a terminal, `munin` prints its help and `munin watch` fails.

## Maintenance

`munin rm RUN...` deletes runs and every file they hold. It refuses a run
whose writer holds its lock, and a silent run (`Store.delete_run` in
[`store.mli`](../lib/store.mli)), reports why on stderr, and goes on with
the other runs. The versions a deleted run produced stay.

`munin end RUN` ends a crashed or silent run as killed: it resumes the run
at a step that orphans nothing and finishes it. It refuses a running or
ended run. The lock of a silent run's writer cannot be probed from here,
so its resume trusts that the writer has exited (the resume section of
[`store_format.mli`](../lib/store_format.mli)): end a silent run only when
its writer is known to be gone.

`munin gc` removes the blobs no version names and the work crashed
writers left, and prints `removed N`. `munin check` prints a
`path problem` line for each path that breaks the store's format, and
exits with status 1 if it found one (`Store.check` in
[`store.mli`](../lib/store.mli)). It reads every log and every blob.

`munin root` prints the store's directory, for the programs that read the
files themselves ([Reading a store without munin](05-files.md)):

<!-- $MDX skip -->
```sh
ls "$(munin root)/runs"
```
