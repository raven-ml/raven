# tolk D73
0 Ops.CONST dtypes.weakint [] 12
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.CONST dtypes.weakint [] 3
3 Ops.CONST dtypes.weakint [] 4
4 Ops.STACK dtypes.weakint [2, 3]
5 Ops.RESHAPE dtypes.f32 [1, 4]
6 Ops.CONST dtypes.weakint [] 32
7 Ops.PARAM dtypes.f32 [6] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
8 Ops.CONST dtypes.weakint [] 8
9 Ops.STACK dtypes.weakint [8, 3]
10 Ops.RESHAPE dtypes.f32 [7, 9]
11 Ops.MUL dtypes.f32 [10, 10]
12 Ops.ADD dtypes.f32 [11, 10]
13 Ops.PARAM dtypes.i32 [2] ParamArg(slot=2, dtype=dtypes.i32, device="CPU")
14 Ops.CONST dtypes.weakint [] 0
15 Ops.MAX dtypes.i32 [13, 14]
16 Ops.CONST dtypes.weakint [] -1
17 Ops.XOR dtypes.i32 [15, 16]
18 Ops.CONST dtypes.weakint [] 7
19 Ops.XOR dtypes.weakint [18, 16]
20 Ops.MAX dtypes.i32 [17, 19]
21 Ops.XOR dtypes.i32 [20, 16]
22 Ops.CAST dtypes.weakint [21] dtypes.weakint
23 Ops.INDEX dtypes.f32 [12, 22]
24 Ops.STORE dtypes.void [5, 23]
25 Ops.AFTER dtypes.f32 [1, 24]
26 Ops.SINK dtypes.void [25]

