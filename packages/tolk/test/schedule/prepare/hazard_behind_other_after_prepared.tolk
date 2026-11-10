# tolk D473 golden=442e63caa339d5962712933125701559
0 Ops.CONST dtypes.weakint [] 16
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
3 Ops.STORE dtypes.void [1, 2]
4 Ops.AFTER dtypes.f32 [1, 3]
5 Ops.CONST dtypes.weakint [] 4
6 Ops.STACK dtypes.weakint [5, 5]
7 Ops.RESHAPE dtypes.f32 [4, 6]
8 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
9 Ops.RESHAPE dtypes.f32 [8, 6]
10 Ops.CONST dtypes.weakint [] 1
11 Ops.CONST dtypes.weakint [] 0
12 Ops.STACK dtypes.weakint [10, 11]
13 Ops.PERMUTE dtypes.f32 [9, 12]
14 Ops.CONST dtypes.weakfloat [] 1.0
15 Ops.ADD dtypes.f32 [13, 14]
16 Ops.STORE dtypes.void [7, 15]
17 Ops.AFTER dtypes.f32 [4, 16]
18 Ops.SINK dtypes.void [17]

