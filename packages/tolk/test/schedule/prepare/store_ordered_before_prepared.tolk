# tolk D473 golden=ee55f83be2b69ff658384c6aba8a1ab8
0 Ops.CONST dtypes.weakint [] 16
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
3 Ops.EXP2 dtypes.f32 [2]
4 Ops.STORE dtypes.void [1, 3]
5 Ops.AFTER dtypes.f32 [1, 4]
6 Ops.STORE dtypes.void [5, 5]
7 Ops.AFTER dtypes.f32 [5, 6]
8 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
9 Ops.PARAM dtypes.f32 [0] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
10 Ops.STORE dtypes.void [8, 9]
11 Ops.AFTER dtypes.f32 [8, 10]
12 Ops.SINK dtypes.void [7, 11]

