What the munin command does that prints paths, and with a run held open
through a fifo. The paths print as POSIX systems spell them.

  $ export NO_COLOR=1
  $ eval "$(./make_store.exe build store)"
  $ ids() {
  >   sed -e "s/$base/BASE/g" -e "s/$sweep/SWEEP/g" \
  >     -e 's/[0-9]\{4\}-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9]\{6\}Z/TIME/g'
  > }

The root is absolute, whether it exists or not.

  $ RAVEN_TRACKING_DIR=none munin root
  $TESTCASE_ROOT/none

  $ munin root --root store
  $TESTCASE_ROOT/store

A run being written cannot be removed or ended. The writer prints its run
id once it holds the run, through a fifo, so reading it waits for that, and
ends at the writer's exit if it never prints.

  $ mkfifo input output
  $ ./make_store.exe hold store < input > output &
  $ exec 3> input
  $ read held < output
  $ munin runs --root store --status running | sed "s/$held/HELD/" | ids
  HELD	running	TIME	held	-	-
  $ munin rm --root store $held 2> err
  [1]
  $ sed "s/$held/HELD/" err
  munin: run HELD is being written
  $ munin end --root store $held 2> err
  [1]
  $ sed "s/$held/HELD/" err
  munin: run HELD is being written

Tags and notes change while the run is written.

  $ munin tag --root store $held live
  $ munin note --root store $held 'while held'
  $ munin runs --root store --tag live | sed "s/$held/HELD/" | ids
  HELD	running	TIME	held	-	live
  $ munin note --root store $held
  while held

Once its writer exits, end ends the run as killed, and refuses an ended
one.

  $ exec 3>&-
  $ wait
  $ munin end --root store $held
  $ munin runs --root store --status killed | sed "s/$held/HELD/" | ids
  HELD	killed	TIME	held	-	live
  SWEEP	killed	TIME	sweep	-	night,sweep
  $ munin end --root store $held 2> err
  [1]
  $ sed "s/$held/HELD/" err
  munin: run HELD has ended
  $ munin rm --root store $held
  $ munin show --root store $held 2> err
  [1]
  $ sed "s/$held/HELD/" err
  munin: no run HELD

check names each path that breaks the format.

  $ printf 'not a record\n' >> store/runs/$base/events.jsonl
  $ rm store/blobs/blake2b-256/188e41df655bfeffb5654aeab18124803449b498d61874eca09a2159c6172669
  $ mkdir store/artifacts/mnist-ckpt/.left
  $ munin check --root store | cut -f 1 | ids
  $TESTCASE_ROOT/store/artifacts/mnist-ckpt/.left
  $TESTCASE_ROOT/store/blobs/blake2b-256/188e41df655bfeffb5654aeab18124803449b498d61874eca09a2159c6172669
  $TESTCASE_ROOT/store/runs/BASE/events.jsonl

watch names an unknown run and the store it looked in.

  $ munin watch nope --root store < /dev/null
  munin: no run nope in $TESTCASE_ROOT/store
  [1]
