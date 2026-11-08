# tolk D280 golden=0f6f7273f42700d42bb02024a1582056
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f64 [0] ParamArg(slot=0, dtype=dtypes.f64)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
3 Ops.AFTER dtypes.f64 [1, 2]
4 Ops.INDEX dtypes.f64 [3, 2]
5 Ops.PARAM dtypes.f64 [0] ParamArg(slot=1, dtype=dtypes.f64)
6 Ops.INDEX dtypes.f64 [5, 2]
7 Ops.CONST dtypes.weakfloat [] 1.4426950408889634
8 Ops.MUL dtypes.f64 [6, 7]
9 Ops.EXP2 dtypes.f64 [8]
10 Ops.STORE dtypes.void [4, 9]
11 Ops.END dtypes.void [10, 2]
12 Ops.SINK dtypes.void [11] KernelInfo()

