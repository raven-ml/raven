A pretraining's draws (views, dropout, mixing) are keyed by its key and its
stream positions, so a run stopped and resumed takes the steps the straight
run takes, bit for bit.

  $ ./make_corpus.exe corpus targets.npy
  $ run () { ../../bin/main.exe pretrain --corpus corpus --arch nano --projector 24,20 --length 4 --pixels 8 --steps 6 --every 2 "$@" > /dev/null; }
  $ run --groups 2 --out straight
  $ run --groups 2 --out stopped --until 3
  $ run --groups 2 --out stopped --resume
  $ cut -f 1-5 straight/log.tsv > a; cut -f 1-5 stopped/log.tsv | diff a - && echo same
  same

A step of two groups takes the positions of two steps of one group, and its
loss is the mean of its groups'. The schedule's rate is 0 at the first
step, so the one-group run's second step is at the initial parameters too,
and the two-group run's first loss is the mean of its first two.

  $ run --groups 1 --out one
  $ cut -f 2,3 straight/log.tsv | head -2
  0	16
  16	16
  $ cut -f 2,3 one/log.tsv | head -2
  0	8
  8	8
  $ awk 'NR == FNR { if (FNR <= 2) s += $5; next } FNR == 1 { d = $5 - s / 2; if (d < 0) d = -d; print (d <= 1e-5 * $5 ? "mean" : "differ") }' one/log.tsv straight/log.tsv
  mean

A resume refuses a run of other options.

  $ run --groups 2 --out straight --resume --length 5 2>&1 | head -1
  tessera: straight/state.safetensors is a run of
