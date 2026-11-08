# tolk D330
0 Ops.CONST dtypes.weakint [] 1
1 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
2 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
3 Ops.CONST dtypes.weakfloat [] 1.0
4 Ops.CAST dtypes.f32 [3] dtypes.f32
5 Ops.EXPAND dtypes.f32 [4, 0]
6 Ops.STORE dtypes.void [2, 5]
7 Ops.AFTER dtypes.f32 [2, 6]
8 Ops.CONST dtypes.weakfloat [] 2.0
9 Ops.CAST dtypes.f32 [8] dtypes.f32
10 Ops.EXPAND dtypes.f32 [9, 0]
11 Ops.STORE dtypes.void [7, 10]
12 Ops.AFTER dtypes.f32 [2, 11]
13 Ops.CONST dtypes.weakfloat [] 0.0
14 Ops.ADD dtypes.f32 [12, 13]
15 Ops.STORE dtypes.void [1, 14]
16 Ops.AFTER dtypes.f32 [1, 15]
17 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
18 Ops.ADD dtypes.f32 [7, 13]
19 Ops.STORE dtypes.void [17, 18]
20 Ops.AFTER dtypes.f32 [17, 19]
21 Ops.SINK dtypes.void [16, 20]

