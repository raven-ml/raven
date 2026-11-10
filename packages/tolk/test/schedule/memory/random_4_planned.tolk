# tolk D477 golden=8bbeca4b2eaec58fa35eb9e3d15d064f
0 Ops.CONST dtypes.weakint [] 256
1 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.SINK dtypes.void [1]
3 Ops.CALL dtypes.void [2, 1] CallInfo()
4 Ops.CONST dtypes.weakint [] 16
5 Ops.BUFFER dtypes.i8 [4] ParamArg(slot=1, dtype=dtypes.i8, device="CPU")
6 Ops.CONST dtypes.weakint [] 257
7 Ops.BUFFER dtypes.i8 [6] ParamArg(slot=2, dtype=dtypes.i8, device="CPU")
8 Ops.CONST dtypes.weakint [] 65536
9 Ops.BUFFER dtypes.i8 [8] ParamArg(slot=3, dtype=dtypes.i8, device="CPU")
10 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
11 Ops.SINK dtypes.void [5, 7, 9, 10]
12 Ops.CONST dtypes.weakint [] 1379072
13 Ops.BUFFER dtypes.i8 [12] ParamArg(slot=9, dtype=dtypes.i8, device="CPU")
14 Ops.CONST dtypes.weakint [] 1378816
15 Ops.SHRINK dtypes.i8 [13, 14, 4]
16 Ops.CONST dtypes.weakint [] 1378304
17 Ops.SHRINK dtypes.i8 [13, 16, 6]
18 Ops.CONST dtypes.weakint [] 1310720
19 Ops.SHRINK dtypes.i8 [13, 18, 8]
20 Ops.CONST dtypes.weakint [] 1376256
21 Ops.CONST dtypes.weakint [] 1024
22 Ops.SHRINK dtypes.i8 [13, 20, 21]
23 Ops.BITCAST dtypes.f32 [22] dtypes.f32
24 Ops.CALL dtypes.void [11, 15, 17, 19, 23] CallInfo()
25 Ops.CONST dtypes.weakint [] 1048576
26 Ops.BUFFER dtypes.i8 [25] ParamArg(slot=5, dtype=dtypes.i8, device="CPU")
27 Ops.SINK dtypes.void [9, 26, 10, 7]
28 Ops.CONST dtypes.weakint [] 0
29 Ops.SHRINK dtypes.i8 [13, 28, 25]
30 Ops.CALL dtypes.void [27, 19, 29, 23, 17] CallInfo()
31 Ops.SINK dtypes.void [26, 5, 7]
32 Ops.CALL dtypes.void [31, 29, 15, 17] CallInfo()
33 Ops.BUFFER dtypes.f32 [8] ParamArg(slot=6, dtype=dtypes.f32, device="CPU")
34 Ops.SINK dtypes.void [33, 10, 26]
35 Ops.CONST dtypes.weakint [] 262144
36 Ops.SHRINK dtypes.i8 [13, 25, 35]
37 Ops.BITCAST dtypes.f32 [36] dtypes.f32
38 Ops.CALL dtypes.void [34, 37, 23, 29] CallInfo()
39 Ops.SINK dtypes.void [33, 5, 26]
40 Ops.CALL dtypes.void [39, 37, 15, 29] CallInfo()
41 Ops.CONST dtypes.weakint [] 1000
42 Ops.BUFFER dtypes.i8 [41] ParamArg(slot=7, dtype=dtypes.i8, device="CPU")
43 Ops.CONST dtypes.weakint [] 255
44 Ops.BUFFER dtypes.f32 [43] ParamArg(slot=8, dtype=dtypes.f32, device="CPU")
45 Ops.SINK dtypes.void [42, 7, 44]
46 Ops.CONST dtypes.weakint [] 1377280
47 Ops.CONST dtypes.weakint [] 1020
48 Ops.SHRINK dtypes.i8 [13, 46, 47]
49 Ops.BITCAST dtypes.f32 [48] dtypes.f32
50 Ops.CALL dtypes.void [45, 42, 17, 49] CallInfo()
51 Ops.SINK dtypes.void [9, 33, 5, 7]
52 Ops.CALL dtypes.void [51, 19, 37, 15, 17] CallInfo()
53 Ops.SINK dtypes.void [26, 7, 44, 10]
54 Ops.CALL dtypes.void [53, 29, 17, 49, 23] CallInfo()
55 Ops.SINK dtypes.void [44, 26, 33]
56 Ops.CALL dtypes.void [55, 49, 29, 37] CallInfo()
57 Ops.SINK dtypes.void [1, 26, 44, 33]
58 Ops.CALL dtypes.void [57, 1, 29, 49, 37] CallInfo()
59 Ops.LINEAR dtypes.void [3, 24, 30, 32, 38, 40, 50, 52, 54, 56, 58]

