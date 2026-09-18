# Load benchmark

How long a pretrained checkpoint takes to become a running model, and how much
memory the process needs on the way. One process per configuration, because
peak memory is a property of a process.

```sh
packages/kaun/bench/load/run.sh            # Llama 3.2 1B from the cache
packages/kaun/bench/load/run.sh --repo R   # another Llama checkpoint
```

Each configuration loads the checkpoint with `Kaun_hf.load_checkpoint`, imports
it with the Llama example's `of_hf` at a dtype, and, when a device is given,
runs one compiled forward pass over eight tokens, which uploads every weight.
`bench_load.exe` prints the wall time from process start to the end of each
phase; `run.sh` wraps it in `/usr/bin/time` and reports the peak memory
footprint on macOS and the maximum resident set size on Linux. The footprint
excludes the file's own pages, which the system drops under pressure without
writing anything; the resident set size includes them.

It is not a CI job: it needs the 2.5 GB download, several gigabytes of memory
and, for the Metal rows, a Mac.

## Results

2026-09-19, Apple M1 Max, 32 GB, macOS 26.3.1, warm file cache, the machine
under other load (load average about 10). Llama 3.2 1B, 2.47 GB stored at
bfloat16. Times are cumulative from process start.

| dtype | device | load | import | first compiled call | peak memory footprint |
| --- | --- | --- | --- | --- | --- |
| as stored | none | 0.27 s | 0.27 s | | 0.01 GB |
| as stored | CPU | 0.27 s | 0.27 s | 9.60 s | 2.40 GB |
| as stored | METAL | 0.24 s | 0.24 s | 6.07 s | 3.27 GB |
| float32 | none | 0.24 s | 0.73 s | | 4.96 GB |
| float32 | CPU | 0.30 s | 0.80 s | 7.90 s | 9.48 GB |
| float32 | METAL | 0.24 s | 0.74 s | 6.46 s | 11.38 GB |

Reading the table:

- As stored, the import allocates nothing: every leaf is a view of the mapped
  file, and the transposed projections are views of those.
- Nearly all of the load time is one failed network probe for a shard index,
  which an unsharded repository does not have. Mapping the file takes 0.05 s.
- On the CPU device a contiguous leaf is read in place, so the 2.40 GB of the
  first row with a device is the transposed projections, which are copied to
  become contiguous. On Metal every leaf is copied into a device buffer, and
  the staging for those copies is kept.
- At float32 the import casts every leaf, 4.94 GB, and the device copies come
  on top.

The same measurement before the checkpoint was mapped and imported by name
(load and import at float32, no device): 15.82 GB and 4.0 s with the reader
that copied the file into memory and a zero-initialized template of the model,
9.90 GB and 3.0 s with the mapped reader and the template.
