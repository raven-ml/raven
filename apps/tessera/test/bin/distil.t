pretrain trains a model by Barlow Twins with mixup, corpus, targets and
distil train a student from its embeddings, and probe scores a model's
embeddings of a labelled tile: here make_labelled's tile of 36 pixels, one
without an observation, and each other with 8 Sentinel-2 and 5 Sentinel-1
observations.

  $ ./make_labelled.exe tile
  $ ../../main.exe corpus --tiles tile,tile --out corpus --min 1
  tile: 36 pixels, 35 kept
  tile: 36 pixels, 35 kept
  corpus: 70 pixels, shuffled

The corpus is a function of its tiles and --seed, to its files' bits.

  $ cksum corpus/*
  138942469 2928 corpus/s1.npy
  2650980855 696 corpus/s1_offsets.npy
  3800755872 12448 corpus/s2.npy
  2721682866 696 corpus/s2_offsets.npy
  $ ../../main.exe corpus --tiles tile,tile --out seed1 --seed 1 > /dev/null
  $ cmp -s corpus/s2.npy seed1/s2.npy || echo differ
  differ

pretrain's views need an observation of each sensor, which --min 1 keeps.
Its model, two steps of groups of 8 here, is the teacher.

  $ ../../main.exe pretrain --corpus corpus --arch nano --projector 24,20 --pixels 8 --steps 2 --out pre > /dev/null
  $ cut -f 1-4 pre/log.tsv
  1	0	8	8
  2	8	8	8

targets writes the teacher's embedding of each pixel of the corpus, and
distil trains a student toward them, two steps of 16 pixels here.

  $ ../../main.exe targets --teacher pre/model.safetensors --corpus corpus --out targets.npy
  targets: 70 / 70 pixels
  $ distil () { ../../main.exe distil --corpus corpus --targets targets.npy --arch nano --pixels 16 --out run "$@" > /dev/null; }
  $ distil --steps 2
  $ cut -f 1-4 run/log.tsv
  1	0	16	16
  2	16	16	16
  $ ls run
  log.tsv
  state.safetensors
  student.safetensors

A resumed run continues from its state, at the stream position the state
holds, to --steps; a run without --resume refuses an output holding one.

  $ distil --steps 4 --resume
  $ cut -f 1-4 run/log.tsv
  1	0	16	16
  2	16	16	16
  3	32	16	16
  4	48	16	16
  $ distil --steps 4 2>&1
  tessera: run holds a run: --resume it, or write to another --out
  [2]

probe labels each pixel by the field that holds its centre. Column 2's
centres lie in fields b0 and b1, whose edges field a's crosses, so field a,
last in the file, holds columns 0 and 1 alone: 12 pixels. The pixel with no
observation is left out of b1, the smaller field of class b, which v1.0's
split takes for training; b0 is tested.

  $ probe () { ../../main.exe probe --landmask tile/landmask.tiff --labels tile/fields.geojson "$@" | grep -v rankme; }
  $ probe --model run/student.safetensors --tile tile
  probe: class a: 1 fields, 12 pixels, 1 training fields of 12 pixels
  probe: class b: 2 fields, 23 pixels, 1 training fields of 11 pixels
  probe: 2 classes, 23 training pixels, 12 test pixels, 1 labelled pixels without an observation left out, c 0.1
  probe: accuracy 1.000000 weighted F1 1.000000

The probe reads an embedding from files in place of a model: one that reads
the classes scores 1 on the same split, as float32 and in GeoTessera's int8
form.

  $ probe --embedding tile/embedding | tail -1
  probe: accuracy 1.000000 weighted F1 1.000000
  $ probe --embedding tile/embedding_int8 | tail -1
  probe: accuracy 1.000000 weighted F1 1.000000
