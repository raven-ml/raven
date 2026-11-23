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

A run being written cannot be removed or ended.

  $ mkfifo input
  $ ./make_store.exe hold store < input > held &
  $ exec 3> input

The wait for the writer's run id gives up after 60 s, so a writer that
never starts fails the test.

  $ i=0
  $ until [ -s held ] || [ $i -ge 600 ]; do i=$((i + 1)); sleep 0.1; done
  $ [ -s held ] || echo "no run id after 60 s"
  $ held=$(cat held)
  $ munin runs --root store --status running | sed "s/$held/HELD/" | ids
  HELD	running	TIME	held	-	-
  $ munin rm --root store $held 2>&1 | sed "s/$held/HELD/"
  munin: run HELD is being written
  $ munin end --root store $held 2>&1 | sed "s/$held/HELD/"
  munin: run HELD is being written

Once its writer exits, end ends the run as killed, and refuses an ended
one.

  $ exec 3>&-
  $ wait
  $ munin end --root store $held
  $ munin runs --root store --status killed | sed "s/$held/HELD/" | ids
  HELD	killed	TIME	held	-	-
  SWEEP	killed	TIME	sweep	-	night,sweep
  $ munin end --root store $held 2>&1 | sed "s/$held/HELD/"
  munin: run HELD has ended
  $ munin rm --root store $held
  $ munin show --root store $held 2>&1 | sed "s/$held/HELD/"
  munin: no run HELD

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

watch names an unknown run and the store it looked in.

  $ munin watch nope --root store < /dev/null
  munin: no run nope in $TESTCASE_ROOT/store
  [1]
