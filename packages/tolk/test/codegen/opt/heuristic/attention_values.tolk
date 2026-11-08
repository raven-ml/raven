# tolk D280 golden=144cca8cc29b9a5d093203f783617aa6
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 64
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
5 Ops.AFTER dtypes.f32 [1, 3, 4]
6 Ops.MUL dtypes.weakint [3, 2]
7 Ops.ADD dtypes.weakint [6, 4]
8 Ops.INDEX dtypes.f32 [5, 7]
9 Ops.CONST dtypes.weakint [] 65536
10 Ops.PARAM dtypes.bf16 [9] ParamArg(slot=1, dtype=dtypes.bf16)
11 Ops.CONST dtypes.weakint [] 1024
12 Ops.MUL dtypes.weakint [3, 11]
13 Ops.RANGE dtypes.weakint [11] (AxisType.WEAK, 0)
14 Ops.ADD dtypes.weakint [12, 13]
15 Ops.INDEX dtypes.bf16 [10, 14]
16 Ops.CAST dtypes.f32 [15] dtypes.f32
17 Ops.PARAM dtypes.bf16 [9] ParamArg(slot=2, dtype=dtypes.bf16)
18 Ops.MUL dtypes.weakint [13, 2]
19 Ops.ADD dtypes.weakint [18, 4]
20 Ops.INDEX dtypes.bf16 [17, 19]
21 Ops.CAST dtypes.f32 [20] dtypes.f32
22 Ops.MUL dtypes.f32 [16, 21]
23 Ops.REDUCE dtypes.f32 [22, 13] (Ops.ADD, 0)
24 Ops.STORE dtypes.void [8, 23]
25 Ops.END dtypes.void [24, 3, 4]
26 Ops.SINK dtypes.void [25] KernelInfo()

