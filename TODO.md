# todo

goalpost: jit-compiled gpt-oss at least as fast as {tinygrad,JAX,llama.cpp/vLLM} on {NVIDIA,NV,AMD}.
Where it stands, with every measurement: doc/performance.md.

## tolk

- Since D479 groups starved reduces, a GPU `Set` election of 16384 and
  24576 rows runs 1.4x and 1.9x slower than rune's former split, and
  elections over several indices near `stream_threads` outputs up to 3.5x;
  a split decided where the renderer is known would close the gap
  (_plans/consolidate/groupgate/adjudication.md)
- tolk's "launches > a schedule does not depend on what the process built
  before it" can pass its 60 s limit under load: it runs after
  `order_independent`, whose schedule cache stays live, so its 200 full
  collections take 10-18 s; running `order_independent` uncached fixes it
- rune's paged attention with pages of 16 keys runs at 168 registers on
  CUDA, one workgroup an SM; its values spilled for a tile are refilled
  every tile, skipped ones included, which keeps K's copy from moving
  before the mask
- gpt-oss's MoE block on CUDA takes 2537 us at 64 tokens and 3195 at 96,
  against 1352 at 128: below 128 tokens the 8-row blocks have no tensor
  cores (gate_up 2076 us at 96 against 687 at 128)
- gpt-oss's prefill scatters gate_up's padded rows back to token rows for
  the bias and activation, then gathers them again for down: about 4 ms at
  512 tokens. A rangeify rule that inlines it slows 32-127 tokens; storing
  the routed product's row operand at its own rows (rune) gets most of the
  gain
- On gpt-oss's cold first call, rangeify's reshape substitution
  (`Indexing`, the `apply_reshape` over placeholder ranges) allocates 8.5%
  of the call's words on the calling domain, before the parallel render
  wave starts
- AMD thread traces are not collected in a profile, and tracing through KFD
  is parked: after a GPU fault KFD runs none of the process's queues, so
  nothing can stop a running trace (one stayed on past its process's exit on
  nonnormal), and an eviction may move its window while it writes. rig's
  traced launch test runs only over PCI. The paths left are PCI and an
  amdgpu DRM submission, which holds its VMID and memory for the job

## talon, hugin, brot, munin, quill

- talon needs a query optimizer, a large design of its own, and the other
  optimization to match the best reference engines; 1.0 has no speed gate.
  Last measured against the faster of DuckDB 1.5.6 and Polars 1.44.2
  (9b82c5b57a, under load): decision support SF1 6.6x geomean, H2O 1e7
  group-by 2.1x, joins 2.0x, with talon using 3-10x the CPU. What the gap
  is made of: join filters pushed at run time and join reordering (DuckDB
  reads far fewer rows on q07, q08 and q12), scans (half the instructions
  of the five worst queries), aggregation (46% of q01), and possibly a
  segment-id reduce kernel in nx (unmeasured). Data: ~/wt/tmp/talon-data
