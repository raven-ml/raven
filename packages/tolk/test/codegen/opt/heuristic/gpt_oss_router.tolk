# tolk D280 golden=ddc18d3014ee8b92506a0f3e47d508bf
0 Ops.CONST dtypes.weakint [] 32
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 1)
3 Ops.AFTER dtypes.f32 [1, 2]
4 Ops.INDEX dtypes.f32 [3, 2]
5 Ops.CONST dtypes.weakint [] 2880
6 Ops.PARAM dtypes.f32 [5] ParamArg(slot=1, dtype=dtypes.f32)
7 Ops.RANGE dtypes.weakint [5] (AxisType.WEAK, 0)
8 Ops.INDEX dtypes.f32 [6, 7]
9 Ops.CONST dtypes.weakint [] 92160
10 Ops.PARAM dtypes.bf16 [9] ParamArg(slot=2, dtype=dtypes.bf16)
11 Ops.MUL dtypes.weakint [7, 0]
12 Ops.ADD dtypes.weakint [11, 2]
13 Ops.INDEX dtypes.bf16 [10, 12]
14 Ops.CAST dtypes.f32 [13] dtypes.f32
15 Ops.MUL dtypes.f32 [8, 14]
16 Ops.REDUCE dtypes.f32 [15, 7] (Ops.ADD, 0)
17 Ops.STORE dtypes.void [4, 16]
18 Ops.END dtypes.void [17, 2]
19 Ops.SINK dtypes.void [18] KernelInfo()

