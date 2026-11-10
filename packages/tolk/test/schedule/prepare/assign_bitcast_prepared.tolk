# tolk D473 golden=6be771418080e665ffda11c6458aade5
0 Ops.CONST dtypes.weakint [] 4
1 Ops.PARAM dtypes.u32 [0] ParamArg(slot=1, dtype=dtypes.u32, device="CPU")
2 Ops.PARAM dtypes.u32 [0] ParamArg(slot=2, dtype=dtypes.u32, device="CPU")
3 Ops.CONST dtypes.weakint [] 1
4 Ops.ADD dtypes.u32 [2, 3]
5 Ops.STORE dtypes.void [1, 4]
6 Ops.AFTER dtypes.u32 [1, 5]
7 Ops.PARAM dtypes.u32 [0] ParamArg(slot=3, dtype=dtypes.u32, device="CPU")
8 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
9 Ops.BITCAST dtypes.u32 [8] dtypes.u32
10 Ops.STORE dtypes.void [7, 9]
11 Ops.AFTER dtypes.u32 [7, 10]
12 Ops.SINK dtypes.void [6, 11]

