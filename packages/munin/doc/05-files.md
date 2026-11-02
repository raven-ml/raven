# Reading a store without munin

A store is a directory of plain files, which
[`store_format.mli`](../lib/store_format.mli) specifies in full. Any
program that follows it reads a store, and a shell reads most of it with
`cat`, `tail`, `jq` and `b2sum`. This page reads the store the
[examples](../examples) fill, from its root:

<!-- $MDX skip -->
```sh
cd "$(munin root)"
ID=20261101T222946Z-302d08a4
```

The layout section of [`store_format.mli`](../lib/store_format.mli) lists
every file. A run is the directory `runs/<id>/`, and its log is
`runs/<id>/events.jsonl`, one JSON record a line.

## A run's start and end

The first record is the run's start, with its experiment, params and
provenance:

<!-- $MDX skip -->
```sh
$ head -n 1 runs/$ID/events.jsonl | jq -c .params
{"tolerance":0.000001,"max_iter":500}
```

An ended run's last record is its end:

<!-- $MDX skip -->
```sh
$ tail -n 1 runs/$ID/events.jsonl | jq -r 'select(.type == "end") | .status'
finished
```

A log without an end belongs to a run that is running, silent or crashed.
On Linux, the writer holds an exclusive `flock` on the log for its whole
life, so on the machine and boot that wrote it, a shared probe tells the
two apart (the status section of
[`store_format.mli`](../lib/store_format.mli) gives the whole rule,
including when a reader cannot probe):

<!-- $MDX skip -->
```sh
flock -s -n runs/$ID/events.jsonl true && echo crashed || echo running
```

## A metric's series

A `metrics` record holds samples of several keys at one step. The samples
of `residual`, as step and value:

<!-- $MDX skip -->
```sh
$ jq -r 'select(.type == "metrics" and (.values | has("residual")))
         | [.step, .values.residual] | @tsv' runs/$ID/events.jsonl | head -3
25	0.7583887377325693
50	0.5890780507082984
75	0.44490050687412774
```

A run that was resumed holds orphans: samples logged at a step above that
of a later `resume` record (the reading section of
[`store_format.mli`](../lib/store_format.mli)). This filter drops them as
it meets each `resume`, and gives the series a reader of munin gets:

<!-- $MDX skip -->
```sh
jq -rn --arg k train/loss '
  reduce inputs as $r ([];
    if $r.type == "resume" then map(select(.[0] <= $r.step))
    elif $r.type == "metrics" and ($r.values | has($k))
    then . + [[$r.step, $r.values[$k]]]
    else . end)
  | .[] | @tsv' runs/$ID/events.jsonl
```

A NaN or an infinity is written as an object, such as
`{"float":"NaN"}`, since JSON has no number for it (the values section of
[`store_format.mli`](../lib/store_format.mli)). `@tsv` prints it as
JSON text.

## Following a live run

A writer appends whole lines, and a reader stops at the first line that is
not whole. `tail -f` follows a run as it writes, and `jq` waits for each
record's end before it prints it:

<!-- $MDX skip -->
```sh
tail -n +1 -f runs/$ID/events.jsonl | jq -c 'select(.type == "metrics") | [.step, .values]'
```

## Statistics of an ended run

An ended run keeps its keys' statistics in `stats.json`, so a reader need
not fold a long log for them. The file is a hint: it holds when its
`format` and `rules` are the ones the reader knows and its `length` is the
log's length in bytes, and otherwise the reader folds the log (the stats
section of [`store_format.mli`](../lib/store_format.mli)):

<!-- $MDX skip -->
```sh
$ [ "$(jq .length runs/$ID/stats.json)" -eq "$(wc -c < runs/$ID/events.jsonl)" ] \
    && jq -c '.keys.residual.finite.least' runs/$ID/stats.json
{"step":500,"time":1793572186.202355,"value":0.002252805587295138}
```

## Tags and notes

Each tag is the file `runs/<id>/tags/<digest>`, named by the BLAKE2b-256
digest of the tag's bytes and holding those bytes without a line feed.
`notes` holds the run's notes as UTF-8 text:

<!-- $MDX skip -->
```sh
for f in runs/$ID/tags/*; do cat "$f"; echo; done
cat runs/$ID/notes
grep -lx baseline runs/*/tags/*          # the tag files of the runs tagged baseline
```

Any process may add a tag, as munin does: it writes the tag to a file
whose name starts with `.`, which readers ignore, then renames it to its
digest (the annotations section of
[`store_format.mli`](../lib/store_format.mli)):

<!-- $MDX skip -->
```sh
tag=baseline
mkdir -p runs/$ID/tags
printf %s "$tag" > runs/$ID/tags/.new
mv runs/$ID/tags/.new "runs/$ID/tags/$(printf %s "$tag" | b2sum -l 256 | cut -d' ' -f1)"
```

## Artifacts and blobs

Version `<n>` of an artifact is `artifacts/<name>/<n>.json`, and an alias
is a file under `artifacts/<name>/aliases/` that holds a version number.
A deleted version's file holds `"deleted":true`. A version's content is
the blob named by its digest:

<!-- $MDX skip -->
```sh
$ cat artifacts/analysis/aliases/reviewed
1
$ jq -r .digest artifacts/measurements/1.json
blake2b-256:a5ecbd1f67bc64c46c3b72fff516a42c0a2855f27793098809ba33e4a2c8fa7a
$ b2sum -l 256 blobs/blake2b-256/a5ecbd1f67bc64c46c3b72fff516a42c0a2855f27793098809ba33e4a2c8fa7a
a5ecbd1f67bc64c46c3b72fff516a42c0a2855f27793098809ba33e4a2c8fa7a  blobs/blake2b-256/a5ecbd1f67bc64c46c3b72fff516a42c0a2855f27793098809ba33e4a2c8fa7a
```

A blob present under its name is complete, so `b2sum` of a file blob
gives back its name. A directory blob's digest is that of a listing of
its entries (the blobs section of
[`store_format.mli`](../lib/store_format.mli)).

The latest version of a name is its highest version not deleted:

<!-- $MDX skip -->
```sh
jq -r 'select(.deleted | not) | input_filename' artifacts/measurements/*.json \
  | sort -V | tail -n 1
```

## Recovering the code

When a run started in a git work tree with uncommitted changes, its
directory holds `git.diff`. Applied to the commit its provenance names, it
gives back the tracked files the run started from:

<!-- $MDX skip -->
```sh
git checkout "$(head -n 1 runs/$ID/events.jsonl | jq -r .provenance.git.commit)"
git apply runs/$ID/git.diff
```
