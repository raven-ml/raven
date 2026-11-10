A run's draws are keyed by its key and its stream positions, so a run
stopped and resumed takes the steps the straight run takes, bit for bit.

  $ ./make_corpus.exe corpus targets.npy
  $ run () { ../../bin/main.exe distil --corpus corpus --targets targets.npy --arch nano --steps 6 --every 2 "$@" > /dev/null; }
  $ run --pixels 16 --out straight
  $ run --pixels 16 --out stopped --until 3
  $ run --pixels 16 --out stopped --resume
  $ cut -f 1-5 straight/log.tsv > a; cut -f 1-5 stopped/log.tsv | diff a - && echo same
  same

The steps do not depend on how a step's pixels are cut into groups: two
groups of 16 take the positions one group of 32 takes, and their losses
agree to rounding.

  $ run --groups 2 --pixels 16 --out two
  $ run --groups 1 --pixels 32 --out one
  $ paste two/log.tsv one/log.tsv | awk '{ d = $5 - $12; if (d < 0) d = -d; if ($2 != $9 || $3 != $10 || d > 1e-5 * $12) bad = 1 } END { print (bad ? "differ" : "agree") }'
  agree

A resume refuses a run of other options, and a state that is not there; a
fresh run refuses an output that holds one.

  $ run --pixels 16 --out straight --resume --seed 4 2>&1 | head -1
  tessera: straight/state.safetensors is a run of
  $ run --pixels 16 --out none --resume 2>&1
  tessera: none/state.safetensors: no state to resume
  [2]
  $ run --pixels 16 --out straight 2>&1
  tessera: straight holds a run: --resume it, or write to another --out
  [2]
