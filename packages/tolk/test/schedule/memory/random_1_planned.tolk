# tolk D477 golden=032dc38a6db0e119f7143febff4f7a64
0 Ops.CONST dtypes.weakint [] 3299690
1 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=0, dtype=dtypes.i8, device="CPU")
2 Ops.SINK dtypes.void [1]
3 Ops.CALL dtypes.void [2, 1] CallInfo()
4 Ops.CONST dtypes.weakint [] 65536
5 Ops.BUFFER dtypes.f32 [4] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
6 Ops.SINK dtypes.void [5]
7 Ops.CONST dtypes.weakint [] 5509376
8 Ops.BUFFER dtypes.i8 [7] ParamArg(slot=7, dtype=dtypes.i8, device="CPU")
9 Ops.CONST dtypes.weakint [] 5242880
10 Ops.CONST dtypes.weakint [] 262144
11 Ops.SHRINK dtypes.i8 [8, 9, 10]
12 Ops.BITCAST dtypes.f32 [11] dtypes.f32
13 Ops.CALL dtypes.void [6, 12] CallInfo()
14 Ops.CONST dtypes.weakint [] 255
15 Ops.BUFFER dtypes.i8 [14] ParamArg(slot=2, dtype=dtypes.i8, device="CPU")
16 Ops.CONST dtypes.weakint [] 1048576
17 Ops.BUFFER dtypes.f32 [16] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
18 Ops.BUFFER dtypes.i8 [16] ParamArg(slot=4, dtype=dtypes.i8, device="CPU")
19 Ops.SINK dtypes.void [5, 15, 17, 18]
20 Ops.CONST dtypes.weakint [] 5509120
21 Ops.SHRINK dtypes.i8 [8, 20, 14]
22 Ops.CONST dtypes.weakint [] 0
23 Ops.CONST dtypes.weakint [] 4194304
24 Ops.SHRINK dtypes.i8 [8, 22, 23]
25 Ops.BITCAST dtypes.f32 [24] dtypes.f32
26 Ops.SHRINK dtypes.i8 [8, 23, 16]
27 Ops.CALL dtypes.void [19, 12, 21, 25, 26] CallInfo()
28 Ops.SINK dtypes.void [17]
29 Ops.CALL dtypes.void [28, 25] CallInfo()
30 Ops.CONST dtypes.weakint [] 4096
31 Ops.BUFFER dtypes.i8 [30] ParamArg(slot=5, dtype=dtypes.i8, device="CPU")
32 Ops.CONST dtypes.weakint [] 2383702
33 Ops.BUFFER dtypes.f32 [32] ParamArg(slot=6, dtype=dtypes.f32, device="CPU")
34 Ops.SINK dtypes.void [5, 31, 33, 17]
35 Ops.CONST dtypes.weakint [] 5505024
36 Ops.SHRINK dtypes.i8 [8, 35, 30]
37 Ops.CALL dtypes.void [34, 12, 36, 33, 25] CallInfo()
38 Ops.SINK dtypes.void [15, 18, 31]
39 Ops.CALL dtypes.void [38, 21, 26, 36] CallInfo()
40 Ops.SINK dtypes.void [33]
41 Ops.CALL dtypes.void [40, 33] CallInfo()
42 Ops.SINK dtypes.void [15]
43 Ops.CALL dtypes.void [42, 21] CallInfo()
44 Ops.SINK dtypes.void [5, 15]
45 Ops.CALL dtypes.void [44, 12, 21] CallInfo()
46 Ops.SINK dtypes.void [31]
47 Ops.CALL dtypes.void [46, 36] CallInfo()
48 Ops.SINK dtypes.void [17, 5]
49 Ops.CALL dtypes.void [48, 25, 12] CallInfo()
50 Ops.SINK dtypes.void [31, 18, 1, 15]
51 Ops.CALL dtypes.void [50, 36, 26, 1, 21] CallInfo()
52 Ops.SINK dtypes.void [33, 5, 31]
53 Ops.CALL dtypes.void [52, 33, 12, 36] CallInfo()
54 Ops.SINK dtypes.void [18]
55 Ops.CALL dtypes.void [54, 26] CallInfo()
56 Ops.SINK dtypes.void [33, 15, 1]
57 Ops.CALL dtypes.void [56, 33, 21, 1] CallInfo()
58 Ops.SINK dtypes.void [31, 15, 18, 1]
59 Ops.CALL dtypes.void [58, 36, 21, 26, 1] CallInfo()
60 Ops.LINEAR dtypes.void [3, 13, 27, 13, 29, 37, 39, 3, 41, 43, 13, 45, 47, 49, 51, 49, 53, 55, 57, 59]

