# tolk D473 golden=0f7436ac6387c1dfd415e0e1c812ea53
0 Ops.CONST dtypes.weakint [] 2
1 Ops.PARAM dtypes.u32 [0] ParamArg(slot=1, dtype=dtypes.u32, device="CPU")
2 Ops.PARAM dtypes.u32 [0] ParamArg(slot=2, dtype=dtypes.u32, device="CPU")
3 Ops.CONST dtypes.weakint [] 1
4 Ops.ADD dtypes.u32 [2, 3]
5 Ops.STORE dtypes.void [1, 4]
6 Ops.AFTER dtypes.u32 [1, 5]
7 Ops.PARAM dtypes.u32 [0] ParamArg(slot=3, dtype=dtypes.u32, device="CPU")
8 Ops.CONST dtypes.weakint [] 4
9 Ops.PARAM dtypes.f32 [8] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
10 Ops.CONST dtypes.weakint [] 0
11 Ops.SHRINK dtypes.f32 [9, 10, 0]
12 Ops.BITCAST dtypes.u32 [11] dtypes.u32
13 Ops.STORE dtypes.void [7, 12]
14 Ops.AFTER dtypes.u32 [7, 13]
15 Ops.SINK dtypes.void [6, 14]

