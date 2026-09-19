# Load benchmark

How long a pretrained checkpoint takes to become a running model, and how much
memory the process needs on the way. One process per configuration, because
peak memory is a property of a process.

```sh
packages/kaun/bench/load/run.sh            # Llama 3.2 1B from the cache
packages/kaun/bench/load/run.sh --repo R   # another Llama checkpoint
```

Each configuration loads the checkpoint with `Kaun_hf.load_checkpoint` and
imports it with the Llama example's `of_hf` at a dtype. With a device, the
importer places each leaf on it with `Rune.to_device` as it builds it, and one
compiled forward pass over eight tokens then runs, which binds the placed
buffers. `bench_load.exe` prints the wall time from process start to the end of
each phase; `run.sh` wraps it in `/usr/bin/time` and reports the peak memory
footprint on macOS and the maximum resident set size on Linux. The footprint
excludes the file's own pages, which the system drops under pressure without
writing anything; the resident set size includes them.

It is not a CI job: it needs the 2.5 GB download, several gigabytes of memory
and, for the Metal rows, a Mac.

## Results

2026-09-19, Apple M1 Max, 32 GB, macOS 26.3.1, warm file cache and warm kernel
cache, the machine under other load (load average about 9). Llama 3.2 1B, 2.47
GB stored at bfloat16. Times are cumulative from process start.

| dtype | device | load | import | first compiled call | peak memory footprint |
| --- | --- | --- | --- | --- | --- |
| as stored | none | 0.24 s | 0.24 s | | 0.01 GB |
| as stored | CPU | 0.25 s | 2.64 s | 5.12 s | 2.13 GB |
| as stored | METAL | 0.27 s | 2.94 s | 3.38 s | 2.87 GB |
| float32 | none | 0.23 s | 0.78 s | | 4.96 GB |
| float32 | CPU | 0.23 s | 2.65 s | 3.90 s | 5.17 GB |
| float32 | METAL | 0.28 s | 2.94 s | 3.46 s | 6.49 GB |

Reading the table:

- As stored and with no device, the import allocates nothing: every leaf is a
  view of the mapped file, and the transposed projections are views of those.
- Nearly all of the load time is one failed network probe for a shard index,
  which an unsharded repository does not have. Mapping the file takes 0.05 s.
- With a device the import is where the bytes move: each leaf is copied into
  its device buffer, 64 MiB at a time, and the compiled call uploads nothing.
  On the CPU device placing a leaf makes it contiguous: the embedding stays a
  view of the file and the transposed projections become host copies, which the
  kernels then read in place.
- At float32 each leaf is cast, placed and dropped before the next is read, so
  the casts never add up to a second model on the host.

With a cold file cache (the checkpoint copied by `cold_copy.py`, which leaves
none of its pages cached), wall time to the end of the import and of the first
compiled call, as stored:

| device | mapped upload, cold | reading the file, cold | reading the file, warm |
| --- | --- | --- | --- |
| METAL | 5.82 s, 8.14 s | 3.25 s, 3.70 s | 3.07 s, 3.53 s |
| CPU | 4.81 s, 10.76 s | 2.76 s, 4.99 s | 2.56 s, 4.50 s |

An upload reads a tensor over a mapped file from the file itself, since
walking the mapping is bound by page faults at a fraction of the disk's speed.
Warm, at this size, the mapping was 0.1 to 0.3 s faster. At 13.76 GB it never
is: placing a synthetic gpt-oss-20b on Metal took 28.6 s cold and 26.0 s warm
through the mapping, and 7.3 s either way reading the file.

A synthetic Llama 3.1 8B (the real headers, random payloads, 16.06 GB in four
shards) through the same program:

| dtype | device | import | first compiled call | peak memory footprint |
| --- | --- | --- | --- | --- |
| as stored, bfloat16 | METAL | 50.8 s | 62.0 s | 16.80 GB |
| float16, cast on the host | METAL | 31.5 s | 37.8 s | 18.16 GB |
| as stored, bfloat16 | CPU | 41.2 s | 62.7 s | 15.27 GB |

How the peak memory footprint moved, in GB:

| Configuration | template and copying reader | mapped reader | import by name | chunked copies | placement |
| --- | --- | --- | --- | --- | --- |
| 1B, float32, no device | 15.82 | 9.90 | 4.96 | 4.96 | 4.96 |
| 1B, as stored, METAL | | | 3.27 | 2.83 | 2.79 |
| 1B, float32, METAL | | | 11.38 | 10.28 | 6.49 |
| 1B, as stored, CPU | | | 2.40 | 2.25 | 2.04 |
| 1B, float32, CPU | | | 9.48 | 9.37 | 5.17 |
| 8B, as stored, CPU | | | 18.85 | 17.14 | 15.27 |
