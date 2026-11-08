# tolk D280 golden=35091d02f87eab4e10acb9728611ba05
0 Ops.CONST dtypes.weakint [] 128
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
3 Ops.AFTER dtypes.i32 [1, 2]
4 Ops.INDEX dtypes.i32 [3, 2]
5 Ops.CAST dtypes.i32 [2] dtypes.i32
6 Ops.STORE dtypes.void [4, 5]
7 Ops.END dtypes.void [6, 2]
8 Ops.SINK dtypes.void [7] KernelInfo()

