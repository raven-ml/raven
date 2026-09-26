# Reference fixtures

Each directory pairs an OCaml graph builder (`main.ml`) with a Python graph
builder (`main.py`). Dune compares Tolk's lowered graph (`stage5`) and rendered
source (`stage7`) with independently generated tinygrad expectations.

The following cases were generated from tinygrad
`d0c9745274335e44b5dd7c15b8422c2b684a3ed9`:

- `image_load_store`: OpenCL float32/float16 image reads and writes, selected
  from ordinary buffer accesses with `IMAGE=2` and pitch alignment 8.
- `tc_matmul_128`: Metal float16 inputs, float32 accumulators, 128³ contraction,
  explicit tensor-core optimization followed by full reduction unrolling.
- `multi_stack`: independently placed input shards, `MSTACK`, and elementwise
  computation through CPU and CUDA lowering.
- `weak_movement_width`: weak integer arithmetic requiring 64 bits across
  reshape and permutation before its result is committed to int64.

From the repository root, regenerate these cases against that source tree:

```sh
PYTHONPATH="$PWD/_plans/tinygrad-d0c974527" python3 - <<'PY'
import runpy
import tinygrad
for case in ("image_load_store", "tc_matmul_128", "multi_stack", "weak_movement_width"):
    runpy.run_path(f"packages/tolk/test/parity/{case}/main.py", run_name="__main__")
PY
```

Importing `tinygrad` first pins its package path before the shared helper loads
its submodules. Inspect regenerated expectations before accepting changes.
These fixtures compile and render graphs; they do not establish accelerator
hardware correctness. The Metal runtime suite separately checks numerical
128³ contractions and BF16 inputs/outputs.
