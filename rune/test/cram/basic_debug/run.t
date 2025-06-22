Test basic debug functionality

  $ ./basic_debug.exe
  ├─ add
  │  ├─ add [2x3],[2x3] → [2x3 f32] μ=0.33 σ=1.49 range=[-2.30,1.96] nans=0
  ├─ full
  │  ├─ const_scalar → [f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0
  │  ├─ reshape [] → [1x1 f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0
  │  ├─ expand [1x1] → [2x3 f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0
  ├─ mul
  │  ├─ mul [2x3],[2x3] → [2x3 f32] μ=0.65 σ=2.99 range=[-4.59,3.92] nans=0
  Result shape: [2,3]

  $ ./debug_with_grad.exe
  ├─ const_scalar → [f32] zeros nans=0
  ├─ reshape [] → [1x1 f32] zeros nans=0
  ├─ expand [1x1] → [2x3 f32] zeros nans=0
  ├─ add
  │  ├─ add [2x3],[2x3] → [2x3 f32] μ=0.33 σ=1.49 range=[-2.30,1.96] nans=0
  │  ├─ ∇add
  │  │  ├─ const_scalar → [f32] zeros nans=0
  │  │  ├─ reshape [] → [1x1 f32] zeros nans=0
  │  │  ├─ expand [1x1] → [2x3 f32] zeros nans=0
  │  │  ├─ add [2x3],[2x3] → [2x3 f32] zeros nans=0
  │  │  ├─ add [2x3],[2x3] → [2x3 f32] zeros nans=0
  ├─ full
  │  ├─ const_scalar → [f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0
  │  ├─ ∇const_scalar
  │  │  ├─ const_scalar → [f32] zeros nans=0
  │  ├─ reshape [] → [1x1 f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0
  │  ├─ ∇reshape
  │  │  ├─ const_scalar → [f32] zeros nans=0
  │  │  ├─ reshape [] → [1x1 f32] zeros nans=0
  │  │  ├─ reshape [1x1] → [f32] zeros nans=0
  │  │  ├─ add [],[] → [f32] zeros nans=0
  │  ├─ expand [1x1] → [2x3 f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0
  │  ├─ ∇expand
  │  │  ├─ const_scalar → [f32] zeros nans=0
  │  │  ├─ reshape [] → [1x1 f32] zeros nans=0
  │  │  ├─ expand [1x1] → [2x3 f32] zeros nans=0
  │  │  ├─ sum [2x3] → [1x1 f32] zeros nans=0
  │  │  ├─ add [1x1],[1x1] → [1x1 f32] zeros nans=0
  ├─ mul
  │  ├─ mul [2x3],[2x3] → [2x3 f32] μ=0.65 σ=2.99 range=[-4.59,3.92] nans=0
  │  ├─ ∇mul
  │  │  ├─ const_scalar → [f32] zeros nans=0
  │  │  ├─ reshape [] → [1x1 f32] zeros nans=0
  │  │  ├─ expand [1x1] → [2x3 f32] zeros nans=0
  │  │  ├─ mul [2x3],[2x3] → [2x3 f32] zeros nans=0
  │  │  ├─ add [2x3],[2x3] → [2x3 f32] zeros nans=0
  │  │  ├─ mul [2x3],[2x3] → [2x3 f32] zeros nans=0
  │  │  ├─ add [2x3],[2x3] → [2x3 f32] zeros nans=0
  ├─ ∇grad_init
  │  ├─ const_scalar → [f32] ones nans=0
  │  ├─ reshape [] → [1x1 f32] ones nans=0
  │  ├─ expand [1x1] → [2x3 f32] ones nans=0
  ├─ const_scalar → [f32] ones nans=0
  ├─ reshape [] → [1x1 f32] ones nans=0
  ├─ expand [1x1] → [2x3 f32] ones nans=0
  Result shape: [2,3]
