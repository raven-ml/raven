# tolk D473 golden=696203cc828ef548808a431032094c2d
0 Ops.CONST dtypes.weakint [] 4
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
3 Ops.CONST dtypes.weakfloat [] 2.0
4 Ops.MUL dtypes.f32 [2, 3]
5 Ops.STORE dtypes.void [1, 4]
6 Ops.AFTER dtypes.f32 [1, 5]
7 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
8 Ops.CONST dtypes.weakfloat [] 9.0
9 Ops.ADD dtypes.f32 [7, 8]
10 Ops.STORE dtypes.void [6, 9]
11 Ops.AFTER dtypes.f32 [6, 10]
12 Ops.SINK dtypes.void [11]

