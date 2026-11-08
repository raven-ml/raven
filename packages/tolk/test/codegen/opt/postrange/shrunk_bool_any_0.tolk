# tolk D280 golden=f33968c91dcd86e1b7328f038cfe6ab9
0 Ops.CONST dtypes.weakint [] 17
1 Ops.PARAM dtypes.bool [0] ParamArg(slot=0, dtype=dtypes.bool)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 1)
3 Ops.AFTER dtypes.bool [1, 2]
4 Ops.INDEX dtypes.bool [3, 2]
5 Ops.CONST dtypes.weakint [] 324
6 Ops.PARAM dtypes.bool [5] ParamArg(slot=1, dtype=dtypes.bool)
7 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
8 Ops.CONST dtypes.weakint [] 18
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.ADD dtypes.weakint [9, 2]
11 Ops.INDEX dtypes.bool [6, 10]
12 Ops.REDUCE dtypes.bool [11, 7] (Ops.ADD, 0)
13 Ops.STORE dtypes.void [4, 12]
14 Ops.END dtypes.void [13, 2]
15 Ops.SINK dtypes.void [14] KernelInfo()

