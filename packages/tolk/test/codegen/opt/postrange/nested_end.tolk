# tolk D280 golden=f915e84ede22f72de514b0b439a963b5
0 Ops.CONST dtypes.weakint [] 32
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 1)
6 Ops.AFTER dtypes.i32 [1, 3, 5]
7 Ops.MUL dtypes.weakint [3, 4]
8 Ops.ADD dtypes.weakint [7, 5]
9 Ops.INDEX dtypes.i32 [6, 8]
10 Ops.CONST dtypes.weakint [] 1
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.STORE dtypes.void [9, 11]
13 Ops.END dtypes.void [12, 5]
14 Ops.END dtypes.void [13, 3]
15 Ops.SINK dtypes.void [14] KernelInfo()

