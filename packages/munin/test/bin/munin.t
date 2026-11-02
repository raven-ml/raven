The munin command reads and changes a store. Each command prints
header-free tab-separated lines, and an error on stderr with status 1.

Errors quote their values in plain text whatever the terminal: without
NO_COLOR, the command line parser styles them for the terminal TERM names.

  $ export NO_COLOR=1

An explicit root that does not exist is an error. The default root that
does not exist is an empty store.

  $ munin runs --root nowhere
  munin: no store at nowhere
  [1]

  $ RAVEN_TRACKING_DIR=none munin runs

  $ RAVEN_TRACKING_DIR=none munin root
  $TESTCASE_ROOT/none

The store holds a run named base with NaN and infinite losses, a failed run
named wide, a killed child of base named sweep, and a run that crashed at
step 4 and resumed from step 2. Ids and start times change from one test
run to the next, so [ids] names them.

  $ eval "$(./make_store.exe build store)"
  $ resumed=$(./make_store.exe crash store)
  $ ./make_store.exe resume store $resumed 2
  $ ids() {
  >   sed -e "s/$base/BASE/g" -e "s/$wide/WIDE/g" -e "s/$sweep/SWEEP/g" \
  >     -e "s/$resumed/RESUMED/g" \
  >     -e 's/[0-9]\{4\}-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9]\{6\}Z/TIME/g'
  > }

  $ munin root --root store
  $TESTCASE_ROOT/store

Runs, newest first: id, status, start, experiment, name, tags.

  $ munin runs --root store | ids
  RESUMED	finished	TIME	mnist	resumed	-
  SWEEP	killed	TIME	sweep	-	night,sweep
  WIDE	failed	TIME	mnist	wide	-
  BASE	finished	TIME	mnist	base	baseline

  $ munin runs --root store --experiment mnist --status finished | ids
  RESUMED	finished	TIME	mnist	resumed	-
  BASE	finished	TIME	mnist	base	baseline

  $ munin runs --root store --tag night --tag sweep | ids
  SWEEP	killed	TIME	sweep	-	night,sweep

  $ munin runs --root store --tag night --tag baseline

  $ munin runs --root store --parent $base | ids
  SWEEP	killed	TIME	sweep	-	night,sweep

  $ munin runs --root store --status lost
  Usage: munin runs [--help] [OPTION]…
  munin: option '--status': invalid value 'lost', expected one of 'running',
         'silent', 'crashed', 'finished', 'failed' or 'killed'
  [1]

A run as key-value lines. A value prints as a run recorded it, a string
escaped, so that the tab of a summary and the line feed of an error stay on
their line.

  $ munin show --root store $base | ids
  id	BASE
  experiment	mnist
  name	base
  status	finished
  started	TIME
  ended	TIME
  updated	TIME
  param.lr	0.001
  param.batch	64
  param.opt	adam
  param.init	nan
  provenance.command	make_store a\tb
  provenance.cwd	/
  provenance.pid	0
  provenance.ocaml	5.0.0
  provenance.raven	test
  provenance.os	test
  provenance.arch	test
  provenance.git.commit	aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  provenance.git.dirty	false
  provenance.seed.init	7
  tag	baseline
  summary.comment	tab\there
  summary.test/acc	0.98
  metric.lr.count	2
  metric.lr.last	4.58e-72
  metric.lr.non_finite	0
  metric.train/loss.count	5
  metric.train/loss.last	0.25
  metric.train/loss.best	0.25
  metric.train/loss.non_finite	2
  metric.val/acc.count	3
  metric.val/acc.last	0.75
  metric.val/acc.best	0.75
  metric.val/acc.non_finite	0

  $ munin show --root store $wide | ids
  id	WIDE
  experiment	mnist
  name	wide
  status	failed
  error	diverged\nat step 1
  started	TIME
  ended	TIME
  updated	TIME
  param.lr	0.01
  param.batch	64
  param.opt	adam
  param.init	nan
  provenance.command	make_store a\tb
  provenance.cwd	/
  provenance.pid	0
  provenance.ocaml	5.0.0
  provenance.raven	test
  provenance.os	test
  provenance.arch	test
  provenance.git.commit	aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  provenance.git.dirty	false
  provenance.seed.init	7
  metric.train/loss.count	2
  metric.train/loss.last	0.25
  metric.train/loss.best	0.25
  metric.train/loss.non_finite	0
  metric.val/acc.count	2
  metric.val/acc.last	nan
  metric.val/acc.non_finite	2

  $ munin show --root store $sweep | ids
  id	SWEEP
  experiment	sweep
  parent	BASE
  status	killed
  started	TIME
  ended	TIME
  updated	TIME
  provenance.command	make_store a\tb
  provenance.cwd	/
  provenance.pid	0
  provenance.ocaml	5.0.0
  provenance.raven	test
  provenance.os	test
  provenance.arch	test
  provenance.git.commit	aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  provenance.git.dirty	false
  provenance.seed.init	7
  tag	night
  tag	sweep

  $ munin show --root store nope
  munin: no run nope
  [1]

Each metric and reading: key, kind, count, last, best, non-finite count.
The best is finite: base's NaN and infinite losses never are, and wide's
accuracy, never finite, has none.

  $ munin metrics --root store $base
  lr	metric	2	4.58e-72	-	0
  sys/load	reading	1	0.5	-	0
  train/loss	metric	5	0.25	0.25	2
  val/acc	metric	3	0.75	0.75	0

  $ munin metrics --root store $wide
  train/loss	metric	2	0.25	0.25	0
  val/acc	metric	2	nan	-	2

A key's samples: step, time, value. Times are UTC to the microsecond.

  $ munin metrics --root store $base train/loss
  0	2024-03-02T13:25:58.677423Z	2.0
  1	2026-11-01T14:03:09.250000Z	nan
  2	2026-11-01T14:03:10.250000Z	0.5
  3	2026-11-01T14:03:11.250000Z	inf
  4	2026-11-01T14:03:12.250000Z	0.25

The resumed run's samples at steps 3 and 4 before its resume are orphans,
left out of its series and statistics.

  $ munin metrics --root store $resumed
  train/loss	metric	5	2.25	2.25	0

  $ munin metrics --root store $resumed train/loss
  0	2024-03-02T13:25:58.677423Z	5.0
  1	2026-11-01T14:03:09.250000Z	4.0
  2	2026-11-01T14:03:10.250000Z	3.0
  3	2026-11-01T14:03:11.250000Z	2.5
  4	2026-11-01T14:03:12.250000Z	2.25

  $ munin metrics --root store $base no/such 2>&1 | ids
  munin: run BASE has no samples of no/such

Compare: the params that differ, and each metric's best, or its last
without a goal, one column per run. A star marks a row's best, the earliest
of a tie, and never a NaN.

  $ munin compare --root store $base $wide
  param.lr	0.001	0.01
  metric.lr	4.58e-72	-
  metric.train/loss	0.25*	0.25
  metric.val/acc	0.75*	-

  $ munin compare --root store $wide $base $resumed
  param.batch	64	64	-
  param.init	nan	nan	-
  param.lr	0.01	0.001	-
  param.opt	adam	adam	-
  metric.lr	-	4.58e-72	-
  metric.train/loss	0.25*	0.25	2.25
  metric.val/acc	-	0.75*	-

Tags and notes.

  $ munin tag --root store $wide lr-sweep keep
  $ munin runs --root store --tag keep | ids
  WIDE	failed	TIME	mnist	wide	keep,lr-sweep
  $ munin tag --root store --remove $wide keep
  $ munin tag --root store $wide "a,b"
  munin: Store: "a,b" is not a tag: a text without ","
  [1]
  $ munin runs --root store --tag keep | ids

  $ munin note --root store $base
  $ munin note --root store $base 'first line
  > second'
  $ munin note --root store $base
  first line
  second
  $ munin show --root store $base | grep '^notes'
  notes	first line\nsecond
  $ munin note --root store --clear $base
  $ munin note --root store $base
  $ munin note --root store --clear $base text
  Usage: munin note [--help] [--clear] [--root=DIR] [OPTION]… RUN [TEXT]
  munin: note --clear takes no TEXT
  [1]

Artifacts: name, version, digest, size, run, step, aliases.

  $ munin artifacts --root store | ids
  mnist-ckpt	1	blake2b-256:188e41df655bfeffb5654aeab18124803449b498d61874eca09a2159c6172669	3	BASE	4	-
  mnist-ckpt	2	blake2b-256:b4d2febddf41d40dd87e702580b60aab9d6e197ed02348f52cbbbcae5062a164	10	WIDE	-	-

  $ munin alias --root store mnist-ckpt 1 best
  $ munin alias --root store mnist-ckpt 2 latest
  $ munin artifacts --root store mnist-ckpt | ids
  mnist-ckpt	1	blake2b-256:188e41df655bfeffb5654aeab18124803449b498d61874eca09a2159c6172669	3	BASE	4	best
  mnist-ckpt	2	blake2b-256:b4d2febddf41d40dd87e702580b60aab9d6e197ed02348f52cbbbcae5062a164	10	WIDE	-	latest
  $ munin alias --root store --remove mnist-ckpt latest
  $ munin artifacts --root store mnist-ckpt | ids
  mnist-ckpt	1	blake2b-256:188e41df655bfeffb5654aeab18124803449b498d61874eca09a2159c6172669	3	BASE	4	best
  mnist-ckpt	2	blake2b-256:b4d2febddf41d40dd87e702580b60aab9d6e197ed02348f52cbbbcae5062a164	10	WIDE	-	-

  $ munin alias --root store mnist-ckpt 3 best
  munin: no version 3 of mnist-ckpt
  [1]

  $ munin alias --root store mnist-ckpt best
  Usage: munin alias [--help] [--remove] [--root=DIR] [OPTION]… [ARG]…
  munin: alias takes NAME VERSION ALIAS
  [1]

Deleting version 2 leaves its blob, which gc removes.

  $ munin rm-artifact --root store mnist-ckpt 2
  $ munin artifacts --root store | ids
  mnist-ckpt	1	blake2b-256:188e41df655bfeffb5654aeab18124803449b498d61874eca09a2159c6172669	3	BASE	4	best
  $ munin gc --root store
  removed 1
  $ munin gc --root store
  removed 0

A run being written cannot be removed. Once its writer exits, it reads
crashed and can be.

  $ mkfifo input
  $ ./make_store.exe hold store < input > held &
  $ exec 3> input
  $ until [ -s held ]; do sleep 0.1; done
  $ held=$(cat held)
  $ munin runs --root store --status running | sed "s/$held/HELD/" | ids
  HELD	running	TIME	held	-	-
  $ munin rm --root store $held 2>&1 | sed "s/$held/HELD/"
  munin: run HELD is being written
  $ munin end --root store $held 2>&1 | sed "s/$held/HELD/"
  munin: run HELD is being written
  $ exec 3>&-
  $ wait
  $ munin runs --root store --status crashed | sed "s/$held/HELD/" | ids
  HELD	crashed	TIME	held	-	-

end ends a crashed run as killed, and refuses an ended one.

  $ munin end --root store $held
  $ munin runs --root store --status killed | sed "s/$held/HELD/" | ids
  HELD	killed	TIME	held	-	-
  SWEEP	killed	TIME	sweep	-	night,sweep
  $ munin end --root store $held 2>&1 | sed "s/$held/HELD/"
  munin: run HELD has ended
  $ munin rm --root store $held
  $ munin show --root store $held 2>&1 | sed "s/$held/HELD/"
  munin: no run HELD

rm removes what it can and reports the rest.

  $ munin rm --root store $sweep nope
  munin: no run nope
  [1]
  $ munin runs --root store | ids
  RESUMED	finished	TIME	mnist	resumed	-
  WIDE	failed	TIME	mnist	wide	lr-sweep
  BASE	finished	TIME	mnist	base	baseline

check lists each path that breaks the format and why.

  $ munin check --root store

  $ printf 'not a record\n' >> store/runs/$base/events.jsonl
  $ rm store/blobs/blake2b-256/188e41df655bfeffb5654aeab18124803449b498d61874eca09a2159c6172669
  $ mkdir store/artifacts/mnist-ckpt/.left
  $ munin check --root store | ids
  $TESTCASE_ROOT/store/artifacts/mnist-ckpt/.left	left by a writer
  $TESTCASE_ROOT/store/blobs/blake2b-256/188e41df655bfeffb5654aeab18124803449b498d61874eca09a2159c6172669	a version's blob is missing
  $TESTCASE_ROOT/store/runs/BASE/events.jsonl	damaged lines: 1
  $ munin check --root store > /dev/null
  [1]
  $ munin gc --root store
  removed 1
