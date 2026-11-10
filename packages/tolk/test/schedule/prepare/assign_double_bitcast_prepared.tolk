# tolk D473 golden=17ffed67afaa00eb402847088bf1620f
0 Ops.CONST dtypes.weakint [] 4
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=1, dtype=dtypes.i32, device="CPU")
2 Ops.PARAM dtypes.i32 [0] ParamArg(slot=2, dtype=dtypes.i32, device="CPU")
3 Ops.CONST dtypes.weakint [] 1
4 Ops.ADD dtypes.i32 [2, 3]
5 Ops.STORE dtypes.void [1, 4]
6 Ops.AFTER dtypes.i32 [1, 5]
7 Ops.PARAM dtypes.i32 [0] ParamArg(slot=3, dtype=dtypes.i32, device="CPU")
8 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
9 Ops.BITCAST dtypes.u32 [8] dtypes.u32
10 Ops.BITCAST dtypes.i32 [9] dtypes.i32
11 Ops.STORE dtypes.void [7, 10]
12 Ops.AFTER dtypes.i32 [7, 11]
13 Ops.SINK dtypes.void [6, 12]

