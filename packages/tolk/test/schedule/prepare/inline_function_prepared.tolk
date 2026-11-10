# tolk D473 golden=d698293e5342c40138e2dfadbd0b1cf3
0 Ops.CONST dtypes.weakint [] 32
1 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=-1, dtype=dtypes.f32, device="CPU")
2 Ops.CONST dtypes.weakint [] 4
3 Ops.CONST dtypes.weakint [] 8
4 Ops.STACK dtypes.weakint [2, 3]
5 Ops.RESHAPE dtypes.f32 [1, 4]
6 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
7 Ops.RESHAPE dtypes.f32 [6, 4]
8 Ops.CONST dtypes.weakfloat [] 2.0
9 Ops.MUL dtypes.f32 [7, 8]
10 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
11 Ops.RESHAPE dtypes.f32 [10, 4]
12 Ops.ADD dtypes.f32 [9, 11]
13 Ops.STORE dtypes.void [5, 12]
14 Ops.AFTER dtypes.f32 [1, 13]
15 Ops.PARAM dtypes.f32 [2] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
16 Ops.RESHAPE dtypes.f32 [14, 4]
17 Ops.CONST dtypes.weakint [] 1
18 Ops.CONST dtypes.weakint [] 0
19 Ops.STACK dtypes.weakint [17, 18]
20 Ops.PERMUTE dtypes.f32 [16, 19]
21 Ops.REDUCE dtypes.f32 [20] (Ops.ADD, 1)
22 Ops.STORE dtypes.void [15, 21]
23 Ops.AFTER dtypes.f32 [15, 22]
24 Ops.SINK dtypes.void [14, 23]

