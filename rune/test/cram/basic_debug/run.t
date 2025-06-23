Test basic debug functionality

  $ ./basic_debug.exe
  ├─ add
  │  ├─ add [2x3],[2x3] → [2x3 f32] μ=0.33 σ=1.49 range=[-2.30,1.96] nans=0 0.000MB
  ├─ full
  │  ├─ const_scalar → [f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0 0.000MB
  │  ├─ reshape [] → [1x1 f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0 0.000MB
  │  ├─ expand [1x1] → [2x3 f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0 0.000MB
  ├─ mul
  │  ├─ mul [2x3],[2x3] → [2x3 f32] μ=0.65 σ=2.99 range=[-4.59,3.92] nans=0 0.000MB
  Result shape: [2,3]

  $ ./debug_with_grad.exe
  ├─ const_scalar → [f32] zeros nans=0 0.000MB
  ├─ reshape [] → [1x1 f32] zeros nans=0 0.000MB
  ├─ expand [1x1] → [2x3 f32] zeros nans=0 0.000MB
  ├─ add
  │  ├─ add [2x3],[2x3] → [2x3 f32] μ=0.33 σ=1.49 range=[-2.30,1.96] nans=0 0.000MB
  ├─ full
  │  ├─ const_scalar → [f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0 0.000MB
  │  ├─ reshape [] → [1x1 f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0 0.000MB
  │  ├─ expand [1x1] → [2x3 f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0 0.000MB
  ├─ mul
  │  ├─ mul [2x3],[2x3] → [2x3 f32] μ=0.65 σ=2.99 range=[-4.59,3.92] nans=0 0.000MB
  ├─ ∇grad_init
  │  ├─ const_scalar → [f32] zeros nans=0 0.000MB
  │  ├─ reshape [] → [1x1 f32] zeros nans=0 0.000MB
  │  ├─ expand [1x1] → [2x3 f32] zeros nans=0 0.000MB
  │  ├─ const_scalar → [f32] ones nans=0 0.000MB
  │  ├─ reshape [] → [1x1 f32] ones nans=0 0.000MB
  │  ├─ expand [1x1] → [2x3 f32] ones nans=0 0.000MB
  ├─ ∇mul
  │  ├─ const_scalar → [f32] zeros nans=0 0.000MB
  │  ├─ reshape [] → [1x1 f32] zeros nans=0 0.000MB
  │  ├─ expand [1x1] → [2x3 f32] zeros nans=0 0.000MB
  │  ├─ const_scalar → [f32] zeros nans=0 0.000MB
  │  ├─ reshape [] → [1x1 f32] zeros nans=0 0.000MB
  │  ├─ expand [1x1] → [2x3 f32] zeros nans=0 0.000MB
  │  ├─ mul [2x3],[2x3] → [2x3 f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0 0.000MB
  │  ├─ add [2x3],[2x3] → [2x3 f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0 0.000MB
  │  ├─ mul [2x3],[2x3] → [2x3 f32] μ=0.33 σ=1.49 range=[-2.30,1.96] nans=0 0.000MB
  │  ├─ add [2x3],[2x3] → [2x3 f32] μ=0.33 σ=1.49 range=[-2.30,1.96] nans=0 0.000MB
  ├─ ∇expand
  │  ├─ const_scalar → [f32] zeros nans=0 0.000MB
  │  ├─ reshape [] → [1x1 f32] zeros nans=0 0.000MB
  │  ├─ sum [2x3] → [1x1 f32] μ=1.96 σ=0.00 range=[1.96,1.96] nans=0 0.000MB
  │  ├─ add [1x1],[1x1] → [1x1 f32] μ=1.96 σ=0.00 range=[1.96,1.96] nans=0 0.000MB
  ├─ ∇reshape
  │  ├─ const_scalar → [f32] zeros nans=0 0.000MB
  │  ├─ reshape [1x1] → [f32] μ=1.96 σ=0.00 range=[1.96,1.96] nans=0 0.000MB
  │  ├─ add [],[] → [f32] μ=1.96 σ=0.00 range=[1.96,1.96] nans=0 0.000MB
  ├─ ∇const_scalar
  ├─ ∇add
  │  ├─ add [2x3],[2x3] → [2x3 f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0 0.000MB
  │  ├─ add [2x3],[2x3] → [2x3 f32] μ=4.00 σ=0.00 range=[4.00,4.00] nans=0 0.000MB
  ├─ const_scalar → [f32] ones nans=0 0.000MB
  ├─ reshape [] → [1x1 f32] ones nans=0 0.000MB
  ├─ expand [1x1] → [2x3 f32] ones nans=0 0.000MB
  Result shape: [2,3]
