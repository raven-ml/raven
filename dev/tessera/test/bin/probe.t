The probe labels a pixel by the field that holds its centre. Column 2's
centres lie in fields b0 and b1, whose edges field a's crosses, so field a,
last in the file, holds columns 0 and 1 alone: 12 pixels. The pixel with no
observation is left out of b1, the smaller field of class b, which v1.0's
split takes for training; b0 is tested.

  $ ./make_corpus.exe corpus targets.npy
  $ ./make_labelled.exe tile
  $ ../../bin/main.exe pretrain --corpus corpus --arch nano --projector 24,20 --length 4 --pixels 8 --steps 2 --out run > /dev/null
  $ probe () { ../../bin/main.exe probe --arch nano --tile tile --landmask tile/landmask.tiff --labels tile/fields.geojson "$@"; }
  $ probe --state run/state.safetensors | grep -v rankme
  probe: class a: 1 fields, 12 pixels, 1 training fields of 12 pixels
  probe: class b: 2 fields, 23 pixels, 1 training fields of 11 pixels
  probe: 2 classes, 23 training pixels, 12 test pixels, 1 labelled pixels without an observation left out, c 0.1
  probe: accuracy 1.000000 weighted F1 1.000000

The probe reads the run's model: its embeddings differ from those of the
model the run started from.

  $ probe --state run/state.safetensors --out trained > /dev/null
  $ probe --out start > /dev/null
  $ cmp -s trained_x_train.npy start_x_train.npy || echo differ
  differ
