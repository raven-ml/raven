# tolk D477 golden=9316141e00923fbbad5a0155be573591
0 Ops.CONST dtypes.weakint [] 1000
1 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.CONST dtypes.weakint [] 70000
3 Ops.BUFFER dtypes.i8 [2] ParamArg(slot=1, dtype=dtypes.i8, device="CPU")
4 Ops.CONST dtypes.weakint [] 33
5 Ops.BUFFER dtypes.i8 [4] ParamArg(slot=2, dtype=dtypes.i8, device="CPU")
6 Ops.SINK dtypes.void [1, 3, 5]
7 Ops.CONST dtypes.weakint [] 4264704
8 Ops.BUFFER dtypes.i8 [7] ParamArg(slot=7, dtype=dtypes.i8, device="CPU")
9 Ops.CONST dtypes.weakint [] 0
10 Ops.CONST dtypes.weakint [] 4000
11 Ops.SHRINK dtypes.i8 [8, 9, 10]
12 Ops.BITCAST dtypes.f32 [11] dtypes.f32
13 Ops.CONST dtypes.weakint [] 4194304
14 Ops.SHRINK dtypes.i8 [8, 13, 2]
15 Ops.CONST dtypes.weakint [] 16896
16 Ops.SHRINK dtypes.i8 [8, 15, 4]
17 Ops.CALL dtypes.void [6, 12, 14, 16] CallInfo()
18 Ops.CONST dtypes.weakint [] 4096
19 Ops.BUFFER dtypes.f32 [18] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
20 Ops.SINK dtypes.void [19, 3]
21 Ops.CONST dtypes.weakint [] 16384
22 Ops.SHRINK dtypes.i8 [8, 9, 21]
23 Ops.BITCAST dtypes.f32 [22] dtypes.f32
24 Ops.CALL dtypes.void [20, 23, 14] CallInfo()
25 Ops.CONST dtypes.weakint [] 257
26 Ops.BUFFER dtypes.i8 [25] ParamArg(slot=4, dtype=dtypes.i8, device="CPU")
27 Ops.SINK dtypes.void [26, 19, 5]
28 Ops.SHRINK dtypes.i8 [8, 21, 25]
29 Ops.CALL dtypes.void [27, 28, 23, 16] CallInfo()
30 Ops.CONST dtypes.weakint [] 9
31 Ops.BUFFER dtypes.i8 [30] ParamArg(slot=5, dtype=dtypes.i8, device="CPU")
32 Ops.SINK dtypes.void [31, 26]
33 Ops.CONST dtypes.weakint [] 4264448
34 Ops.SHRINK dtypes.i8 [8, 33, 30]
35 Ops.CALL dtypes.void [32, 34, 28] CallInfo()
36 Ops.CONST dtypes.weakint [] 1048576
37 Ops.BUFFER dtypes.f32 [36] ParamArg(slot=6, dtype=dtypes.f32, device="CPU")
38 Ops.SINK dtypes.void [37, 31, 3]
39 Ops.SHRINK dtypes.i8 [8, 9, 13]
40 Ops.BITCAST dtypes.f32 [39] dtypes.f32
41 Ops.CALL dtypes.void [38, 40, 34, 14] CallInfo()
42 Ops.LINEAR dtypes.void [17, 24, 29, 35, 41]

