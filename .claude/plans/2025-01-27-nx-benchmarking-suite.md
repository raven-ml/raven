# Task: Build a Comprehensive Benchmarking Suite for Nx

**IMPORTANT: This checklist must be kept updated throughout implementation**

- [x] Extend operation coverage in bench_nx.ml and bench_numpy.py
- [ ] Add Owl benchmarks with same operations
- [ ] Implement JSON export for all benchmarks
- [ ] Create visualization pipeline using Hugin
- [ ] Build report generation system
- [ ] Create automation scripts (run_all.sh, Makefile)
- [ ] Add extended real-world benchmarks
- [ ] Write tests for benchmark infrastructure
- [ ] Update documentation

---

## Objective
Create a robust benchmarking suite for Nx that:
- Compares performance against NumPy and Owl
- Covers a comprehensive set of operations (not just addition)
- Generates visualizations using Hugin
- Produces a comprehensive report document with charts and analysis
- Provides reproducible and automated benchmark execution

## Context
- Current benchmarking is minimal: only Addition operation is tested
- Existing infrastructure: custom ubench library for OCaml, simple timing for Python
- Results are stored as markdown but not in machine-readable format
- No Owl comparison exists yet
- No visualization or automated reporting

## Approach
1. Extend benchmark coverage to all major Nx operations
2. Add Owl benchmarks for comparison
3. Export results in machine-readable format (JSON/CSV)
4. Create visualization pipeline using Hugin
5. Generate comprehensive HTML/PDF report with embedded charts
6. Automate the entire pipeline with a single command

## Implementation Steps

### 1. Extend Operation Coverage
**Files to modify:**
- `nx/bench/bench_nx.ml`
- `nx/bench/bench_numpy.py`

**Operations to benchmark:**
- Arithmetic: add, sub, mul, div, pow
- Unary: sqrt, exp, log, sin, cos, tanh, abs, neg
- Reduction: sum, mean, min, max, std, var
- Matrix operations: matmul, transpose
- Array manipulation: reshape, concatenate, stack, split
- Indexing: get, set, slice operations
- Broadcasting scenarios
- Memory views vs copies

### 2. Add Owl Benchmarks
**Files to create:**
- `nx/bench/bench_owl.ml`
- `nx/bench/dune` (update to include Owl dependency)

**Implementation:**
- Mirror the same operations as Nx benchmarks
- Use same matrix sizes and dtypes
- Integrate with ubench for consistent timing

### 3. Machine-Readable Output
**Files to modify:**
- `nx/vendor/ubench/ubench.ml` (add JSON export)
- `nx/bench/bench_nx.ml` (save results to JSON)
- `nx/bench/bench_numpy.py` (save results to JSON)

**Files to create:**
- `nx/bench/bench_owl.ml` (with JSON output)
- `nx/bench/lib/benchmark_types.ml` (common types for results)

**Format:**
```json
{
  "timestamp": "2025-01-27T10:00:00Z",
  "system_info": { "os": "...", "cpu": "...", "memory": "..." },
  "results": [
    {
      "library": "nx",
      "backend": "native",
      "operation": "add",
      "dtype": "float32",
      "shape": [1000, 1000],
      "time_ns": 1234567,
      "memory_bytes": 4000000,
      "runs": 100
    }
  ]
}
```

### 4. Visualization Pipeline
**Files to create:**
- `nx/bench/visualize/dune`
- `nx/bench/visualize/benchmark_viz.ml`
- `nx/bench/visualize/benchmark_viz.mli`

**Visualizations to generate:**
- Bar charts: operation performance comparison across libraries
- Line plots: performance vs matrix size
- Heatmaps: operation × size performance matrix
- Memory usage charts
- Relative performance charts (normalized to NumPy baseline)

### 5. Report Generation
**Files to create:**
- `nx/bench/report/dune`
- `nx/bench/report/generate_report.ml`
- `nx/bench/report/template.html`

**Report sections:**
- Executive summary with key findings
- System configuration details
- Performance comparison tables
- Interactive charts (exported from Hugin)
- Memory usage analysis
- Backend comparison (native vs metal)
- Recommendations and conclusions

### 6. Automation Script
**Files to create:**
- `nx/bench/run_all.sh`
- `nx/bench/Makefile`

**Script workflow:**
1. Clean previous results
2. Run Nx benchmarks (native and metal backends)
3. Run NumPy benchmarks
4. Run Owl benchmarks
5. Aggregate JSON results
6. Generate visualizations
7. Create HTML report
8. Optionally upload results to tracking system

### 7. Extended Benchmarks
**Files to create:**
- `nx/bench/scenarios/dune`
- `nx/bench/scenarios/real_world.ml`

**Real-world scenarios:**
- Neural network forward pass
- Image processing pipeline
- Signal processing (FFT-like operations)
- Linear algebra workloads (SVD, eigenvalues)
- Statistical computations

## Testing Strategy
- Unit tests for benchmark infrastructure:
  - `nx/bench/test/test_json_export.ml`
  - `nx/bench/test/test_visualization.ml`
- Integration tests:
  - Verify all benchmarks run without errors
  - Check JSON output validity
  - Ensure visualizations are generated
- Manual verification:
  - Review generated report for accuracy
  - Spot-check timing measurements
  - Validate memory usage tracking

## Success Criteria
- [ ] All major Nx operations have benchmarks
- [ ] Owl comparison benchmarks implemented
- [ ] Results exported in machine-readable JSON format
- [ ] Hugin visualizations generated successfully
- [ ] HTML report with embedded charts created
- [ ] Single command runs entire benchmark suite
- [ ] Documentation updated with benchmark usage
- [ ] CI integration for performance regression detection

## Risks & Mitigation
1. **Owl installation complexity**
   - Mitigation: Make Owl benchmarks optional, document installation steps
   
2. **Benchmark variability**
   - Mitigation: Multiple runs, statistical analysis, system isolation recommendations

3. **Memory measurement accuracy**
   - Mitigation: Use OCaml's Gc module, document limitations

4. **Large result files**
   - Mitigation: Compression, selective data retention, cloud storage option

5. **Cross-platform differences**
   - Mitigation: Separate results by platform, document system requirements

## Future Enhancements
- GPU backend benchmarks (CUDA/Metal)
- Continuous performance tracking
- Regression detection and alerts
- Benchmark result database
- Interactive web dashboard
- Profiling integration for bottleneck analysis