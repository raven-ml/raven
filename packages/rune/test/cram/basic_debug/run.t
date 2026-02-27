Test basic debug functionality

  $ ./basic_debug.exe
  ├─ buffer → [6 f32] zeros nans=0 0.000MB
  ├─ reshape [6] → [2x3 f32] zeros nans=0 0.000MB
  ├─ add [2x3],[2x3] → [2x3 f32] μ=-1.10 σ=1.86 range=[-3.31,1.63] nans=0 0.000MB
  ├─ buffer → [6 f32] zeros nans=0 0.000MB
  ├─ reshape [6] → [2x3 f32] zeros nans=0 0.000MB
  ├─ mul [2x3],[2x3] → [2x3 f32] μ=-2.19 σ=3.71 range=[-6.62,3.25] nans=0 0.000MB
  Result shape: [2,3]

  $ ./debug_with_grad.exe
  ├─ buffer → [6 f32] zeros nans=0 0.000MB
  ├─ reshape [6] → [2x3 f32] zeros nans=0 0.000MB
  ├─ add [2x3],[2x3] → [2x3 f32] μ=-1.10 σ=1.86 range=[-3.31,1.63] nans=0 0.000MB
  ├─ buffer → [6 f32] zeros nans=0 0.000MB
  ├─ reshape [6] → [2x3 f32] zeros nans=0 0.000MB
  ├─ mul [2x3],[2x3] → [2x3 f32] μ=-2.19 σ=3.71 range=[-6.62,3.25] nans=0 0.000MB
  ├─ buffer → [1 f32] zeros nans=0 0.000MB
  ├─ reshape [1] → [f32] zeros nans=0 0.000MB
  ├─ sum [2x3] → [f32] μ=-13.17 σ=0.00 range=[-13.17,-13.17] nans=0 0.000MB
  ├─ buffer → [1 f32] zeros nans=0 0.000MB
  ├─ reshape [1] → [1x1 f32] zeros nans=0 0.000MB
  ├─ sum [2x3] → [1x1 f32] μ=-13.17 σ=0.00 range=[-13.17,-13.17] nans=0 0.000MB
  ├─ reshape [] → [1x1 f32] ones nans=0 0.000MB
  ├─ expand [1x1] → [2x3 f32] ones nans=0 0.000MB
  ├─ buffer → [6 f32] zeros nans=0 0.000MB
  ├─ reshape [6] → [2x3 f32] zeros nans=0 0.000MB
  ├─ add [2x3],[2x3] → [2x3 f32] ones nans=0 0.000MB
  ├─ reshape [] → [1 f32] ones nans=0 0.000MB
  ├─ buffer → [1 f32] zeros nans=0 0.000MB
  ├─ reshape [1] → [1 f32] zeros nans=0 0.000MB
  ├─ add [1],[1] → [1 f32] ones nans=0 0.000MB
  ├─ buffer → [6 f32] zeros nans=0 0.000MB
  ├─ reshape [6] → [2x3 f32] zeros nans=0 0.000MB
  ├─ mul [2x3],[2x3] → [2x3 f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0 0.000MB
  ├─ buffer → [6 f32] zeros nans=0 0.000MB
  ├─ reshape [6] → [2x3 f32] zeros nans=0 0.000MB
  ├─ add [2x3],[2x3] → [2x3 f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0 0.000MB
  ├─ buffer → [6 f32] zeros nans=0 0.000MB
  ├─ reshape [6] → [2x3 f32] zeros nans=0 0.000MB
  ├─ mul [2x3],[2x3] → [2x3 f32] μ=-1.10 σ=1.86 range=[-3.31,1.63] nans=0 0.000MB
  ├─ buffer → [6 f32] zeros nans=0 0.000MB
  ├─ reshape [6] → [2x3 f32] zeros nans=0 0.000MB
  ├─ add [2x3],[2x3] → [2x3 f32] μ=-1.10 σ=1.86 range=[-3.31,1.63] nans=0 0.000MB
  ├─ reshape [2x3] → [6 f32] ones nans=0 0.000MB
  ├─ buffer → [6 f32] zeros nans=0 0.000MB
  ├─ reshape [6] → [6 f32] zeros nans=0 0.000MB
  ├─ add [6],[6] → [6 f32] ones nans=0 0.000MB
  ├─ buffer → [6 f32] zeros nans=0 0.000MB
  ├─ reshape [6] → [2x3 f32] zeros nans=0 0.000MB
  ├─ add [2x3],[2x3] → [2x3 f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0 0.000MB
  ├─ buffer → [6 f32] zeros nans=0 0.000MB
  ├─ reshape [6] → [2x3 f32] zeros nans=0 0.000MB
  ├─ add [2x3],[2x3] → [2x3 f32] μ=4.00 σ=0.00 range=[4.00,4.00] nans=0 0.000MB
  ├─ reshape [2x3] → [6 f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0 0.000MB
  ├─ buffer → [6 f32] zeros nans=0 0.000MB
  ├─ reshape [6] → [6 f32] zeros nans=0 0.000MB
  ├─ add [6],[6] → [6 f32] μ=2.00 σ=0.00 range=[2.00,2.00] nans=0 0.000MB
  Result shape: [2,3]
