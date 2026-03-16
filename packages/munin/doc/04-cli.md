# CLI Reference

The `munin` command-line tool inspects and manages the local tracking store.
Every subcommand accepts `--root DIR` to override the default store location
(`$RAVEN_TRACKING_DIR` or `$XDG_DATA_HOME/raven/munin`).

## munin runs

List tracked runs. Output is tab-separated: ID, experiment, status, parent,
name, git commit.

<!-- $MDX skip -->
```
munin runs
```

```
20260317T143201_abc  mnist-sweep  finished  -        lr-0.001  a1b2c3d
20260317T141502_def  mnist-sweep  finished  -        lr-0.01   a1b2c3d
20260317T140003_ghi  cifar10      running   -        baseline  e4f5a6b
```

Filter by experiment:

<!-- $MDX skip -->
```
munin runs --experiment mnist-sweep
```

```
20260317T143201_abc  mnist-sweep  finished  -  lr-0.001  a1b2c3d
20260317T141502_def  mnist-sweep  finished  -  lr-0.01   a1b2c3d
```

## munin show

Display full details for a single run.

<!-- $MDX skip -->
```
munin show 20260317T143201_abc
```

```
id: 20260317T143201_abc
experiment: mnist-sweep
name: lr-0.001
parent: -
status: finished
started_at: 1742224321
ended_at: 1742225180
resumable: false
notes: -
command: ./train.exe --lr 0.001
cwd: /home/user/project
hostname: workstation
pid: 42519
git_commit: a1b2c3d
git_dirty: false
env:
  CUDA_VISIBLE_DEVICES=0
tags:
  sweep
  final
params:
  lr: 0.001
  batch_size: 64
  epochs: 10
summary:
  loss: 0.0312
  accuracy: 0.991
metric_keys:
  accuracy, loss, lr
latest_metrics:
  accuracy: step=9380 value=0.991
  loss: step=9380 value=0.0312
children:
output_artifacts:
  mnist-model v3 aliases=[latest] consumers=[]
input_artifacts:
  mnist-data v1 producer=20260310T090000_xyz
```

## munin compare

Compare two or more runs side by side. Prints a tab-separated table with
parameters and summary values. When a metric has a declared goal (`Minimize` or
`Maximize`), the best value is marked with `*`.

<!-- $MDX skip -->
```
munin compare 20260317T143201_abc 20260317T141502_def
```

```
key           lr-0.001   lr-0.01
batch_size    64         64
epochs        10         10
lr            0.001      0.01
accuracy      0.991*     0.984
loss          0.0312*    0.0587
```

Works with any number of runs:

<!-- $MDX skip -->
```
munin compare abc def ghi
```

## munin metrics

Two modes: listing mode (no `--key`) and history mode (with `--key`).

### Listing mode

Shows all metric keys with their latest value, latest step, and sample count.

<!-- $MDX skip -->
```
munin metrics 20260317T143201_abc
```

```
key              latest_value  latest_step  count
accuracy         0.991         9380         9380
loss             0.0312        9380         9380
lr               0.001         9380         1
sys/cpu_user     12.3          627          627
sys/mem_used_gb  6.82          627          627
```

### History mode

Dump the full time series for a single key. Supports `--format tsv` (default),
`csv`, and `json`.

<!-- $MDX skip -->
```
munin metrics 20260317T143201_abc --key loss
```

```
step	timestamp	value
1	1742224322.123456	2.3026
2	1742224322.234567	1.8451
3	1742224322.345678	1.2107
...
```

<!-- $MDX skip -->
```
munin metrics 20260317T143201_abc --key loss --format csv
```

```
step,timestamp,value
1,1742224322.123456,2.3026
2,1742224322.234567,1.8451
...
```

<!-- $MDX skip -->
```
munin metrics 20260317T143201_abc --key loss --format json
```

```json
[{"step":1,"timestamp":1742224322.123456,"value":2.3026},{"step":2,"timestamp":1742224322.234567,"value":1.8451}]
```

## munin watch

Launch the terminal dashboard. See [Terminal Dashboard](05-dashboard.md) for
full documentation.

Auto-detect the latest run:

<!-- $MDX skip -->
```
munin watch
```

Open a specific run:

<!-- $MDX skip -->
```
munin watch 20260317T143201_abc
```

Filter by experiment (picks the latest run in that experiment):

<!-- $MDX skip -->
```
munin watch --experiment mnist-sweep
```

## munin artifacts

List stored artifacts. Output is tab-separated: name, version, kind, payload
type, size in bytes, producer run, consumer runs.

<!-- $MDX skip -->
```
munin artifacts
```

```
mnist-data   v1  dataset  dir   48000000  20260310T090000_xyz  20260317T143201_abc,20260317T141502_def
mnist-model  v1  model    file  4521984   20260317T141502_def  -
mnist-model  v2  model    file  4521984   20260317T143201_abc  -
mnist-model  v3  model    file  4521984   20260317T143201_abc  -
```

Filter by name:

<!-- $MDX skip -->
```
munin artifacts --name mnist-model
```

```
mnist-model  v1  model  file  4521984  20260317T141502_def  -
mnist-model  v2  model  file  4521984  20260317T143201_abc  -
mnist-model  v3  model  file  4521984  20260317T143201_abc  -
```

## munin delete

Delete a run and its event log from the store. Does not remove shared blobs
(use `munin gc` for that). Removes the experiment directory if no runs remain.

<!-- $MDX skip -->
```
munin delete 20260317T141502_def
```

```
Delete run 20260317T141502_def (mnist-sweep / lr-0.01)? [y/N] y
Deleted.
```

Skip the confirmation prompt with `--yes`:

<!-- $MDX skip -->
```
munin delete 20260317T141502_def --yes
```

## munin gc

Garbage-collect unreferenced blobs from the blob store. Blobs that are no
longer referenced by any artifact are removed.

<!-- $MDX skip -->
```
munin gc
```

```
Removed 3 unreferenced blob(s).
```
