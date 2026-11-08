# tolk D280 golden=e088f4bee02ca2fd0b55afb7ed7453e9
0 Ops.CONST dtypes.weakint [] 16
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.STACK dtypes.void []
3 Ops.PARAM dtypes.weakint [2] ParamArg(slot=-1, dtype=dtypes.weakint, vmin_vmax=(2, 8), name="n", addrspace=AddrSpace.ALU)
4 Ops.RANGE dtypes.weakint [3] (AxisType.LOCAL, 0)
5 Ops.AFTER dtypes.f32 [1, 4]
6 Ops.INDEX dtypes.f32 [5, 4]
7 Ops.CONST dtypes.weakint [] 128
8 Ops.PARAM dtypes.f32 [7] ParamArg(slot=1, dtype=dtypes.f32)
9 Ops.CONST dtypes.weakint [] 8
10 Ops.MUL dtypes.weakint [4, 9]
11 Ops.RANGE dtypes.weakint [9] (AxisType.WEAK, 1)
12 Ops.ADD dtypes.weakint [10, 11]
13 Ops.INDEX dtypes.f32 [8, 12]
14 Ops.REDUCE dtypes.f32 [13, 11] (Ops.ADD, 0)
15 Ops.STORE dtypes.void [6, 14]
16 Ops.END dtypes.void [15, 4]
17 Ops.SINK dtypes.void [16] KernelInfo()

