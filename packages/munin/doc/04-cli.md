# CLI Reference

The `munin` command-line tool inspects and manages the local tracking store.
Every subcommand accepts `--root DIR` to override the default store location
(`$RAVEN_TRACKING_DIR` or `$XDG_DATA_HOME/raven/munin`).

## munin runs

List tracked runs. Output is tab-separated: ID, experiment, status, parent,
name, git commit.

```
munin runs
```

```
2026-03-17_14-32-01_a1f3  mnist-sweep  finished  -        lr-0.001  a1b2c3d
2026-03-17_14-15-02_9c2e  mnist-sweep  finished  -        lr-0.01   a1b2c3d
2026-03-17_14-00-03_5b07  cifar10      running   -        baseline  e4f5a6b
```

Filter by experiment:

```
munin runs --experiment mnist-sweep
```

```
2026-03-17_14-32-01_a1f3  mnist-sweep  finished  -  lr-0.001  a1b2c3d
2026-03-17_14-15-02_9c2e  mnist-sweep  finished  -  lr-0.01   a1b2c3d
```

## munin show

Display full details for a single run.

```
munin show 2026-03-17_14-32-01_a1f3
```

```
id: 2026-03-17_14-32-01_a1f3
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
  mnist-data v1 producer=2026-03-10_09-00-00_e81d
```

## munin compare

Compare two or more runs side by side. Prints a tab-separated table with
parameters and summary values. When a metric has a declared goal (`Minimize` or
`Maximize`), the best value is marked with `*`.

```
munin compare 2026-03-17_14-32-01_a1f3 2026-03-17_14-15-02_9c2e
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

```
munin compare abc def ghi
```

## munin metrics

Two modes: listing mode (no `--key`) and history mode (with `--key`).

### Listing mode

Shows all metric keys with their latest value, latest step, and sample count.

```
munin metrics 2026-03-17_14-32-01_a1f3
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

```
munin metrics 2026-03-17_14-32-01_a1f3 --key loss
```

```
step	timestamp	value
1	1742224322.123456	2.3026
2	1742224322.234567	1.8451
3	1742224322.345678	1.2107
...
```

```
munin metrics 2026-03-17_14-32-01_a1f3 --key loss --format csv
```

```
step,timestamp,value
1,1742224322.123456,2.3026
2,1742224322.234567,1.8451
...
```

```
munin metrics 2026-03-17_14-32-01_a1f3 --key loss --format json
```

```json
[{"step":1,"timestamp":1742224322.123456,"value":2.3026},{"step":2,"timestamp":1742224322.234567,"value":1.8451}]
```

## munin watch

Launch the terminal dashboard. See [Terminal Dashboard](05-dashboard.md) for
full documentation.

Auto-detect the latest run:

```
munin watch
```

Open a specific run:

```
munin watch 2026-03-17_14-32-01_a1f3
```

Filter by experiment (picks the latest run in that experiment):

```
munin watch --experiment mnist-sweep
```

## munin artifacts

List stored artifacts. Output is tab-separated: name, version, kind, payload
type, size in bytes, producer run, consumer runs.

```
munin artifacts
```

```
mnist-data   v1  dataset  dir   48000000  2026-03-10_09-00-00_e81d  2026-03-17_14-32-01_a1f3,2026-03-17_14-15-02_9c2e
mnist-model  v1  model    file  4521984   2026-03-17_14-15-02_9c2e  -
mnist-model  v2  model    file  4521984   2026-03-17_14-32-01_a1f3  -
mnist-model  v3  model    file  4521984   2026-03-17_14-32-01_a1f3  -
```

Filter by name:

```
munin artifacts --name mnist-model
```

```
mnist-model  v1  model  file  4521984  2026-03-17_14-15-02_9c2e  -
mnist-model  v2  model  file  4521984  2026-03-17_14-32-01_a1f3  -
mnist-model  v3  model  file  4521984  2026-03-17_14-32-01_a1f3  -
```

## munin delete

Delete a run and its event log from the store. Does not remove shared blobs
(use `munin gc` for that). Removes the experiment directory if no runs remain.

```
munin delete 2026-03-17_14-15-02_9c2e
```

```
Delete run 2026-03-17_14-15-02_9c2e (mnist-sweep / lr-0.01)? [y/N] y
Deleted.
```

Skip the confirmation prompt with `--yes`:

```
munin delete 2026-03-17_14-15-02_9c2e --yes
```

## munin gc

Garbage-collect unreferenced blobs from the blob store. Blobs that are no
longer referenced by any artifact are removed.

```
munin gc
```

```
Removed 3 unreferenced blob(s).
```
