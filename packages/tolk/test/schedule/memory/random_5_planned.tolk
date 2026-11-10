# tolk D41 D477 golden=766a3cca69090332785437477011ad4e
0 Ops.CONST dtypes.weakint [] 65536
1 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=0, dtype=dtypes.i8, device="CPU")
2 Ops.CONST dtypes.weakint [] 4096
3 Ops.BUFFER dtypes.i8 [2] ParamArg(slot=1, dtype=dtypes.i8, device="CPU")
4 Ops.CONST dtypes.weakint [] 256
5 Ops.BUFFER dtypes.f32 [4] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
6 Ops.SINK dtypes.void [1, 3, 5]
7 Ops.CONST dtypes.weakint [] 3159040
8 Ops.BUFFER dtypes.i8 [7] ParamArg(slot=17, dtype=dtypes.i8, device="CPU")
9 Ops.CONST dtypes.weakint [] 0
10 Ops.SHRINK dtypes.i8 [8, 9, 0]
11 Ops.SHRINK dtypes.i8 [8, 0, 2]
12 Ops.CONST dtypes.weakint [] 3157248
13 Ops.CONST dtypes.weakint [] 1024
14 Ops.SHRINK dtypes.i8 [8, 12, 13]
15 Ops.BITCAST dtypes.f32 [14] dtypes.f32
16 Ops.CALL dtypes.void [6, 10, 11, 15] CallInfo()
17 Ops.CONST dtypes.weakint [] 257
18 Ops.BUFFER dtypes.i8 [17] ParamArg(slot=3, dtype=dtypes.i8, device="CPU")
19 Ops.CONST dtypes.weakint [] 255
20 Ops.BUFFER dtypes.i8 [19] ParamArg(slot=4, dtype=dtypes.i8, device="CPU")
21 Ops.BUFFER dtypes.i8 [19] ParamArg(slot=5, dtype=dtypes.i8, device="CPU")
22 Ops.SINK dtypes.void [18, 1, 20, 21]
23 Ops.CONST dtypes.weakint [] 1049600
24 Ops.SHRINK dtypes.i8 [8, 23, 17]
25 Ops.SHRINK dtypes.i8 [8, 0, 19]
26 Ops.CONST dtypes.weakint [] 3158272
27 Ops.SHRINK dtypes.i8 [8, 26, 19]
28 Ops.CALL dtypes.void [22, 24, 10, 25, 27] CallInfo()
29 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=6, dtype=dtypes.i8, device="CPU")
30 Ops.SINK dtypes.void [29]
31 Ops.CALL dtypes.void [30, 10] CallInfo()
32 Ops.BUFFER dtypes.f32 [19] ParamArg(slot=7, dtype=dtypes.f32, device="CPU")
33 Ops.CONST dtypes.weakint [] 1048576
34 Ops.BUFFER dtypes.i8 [33] ParamArg(slot=8, dtype=dtypes.i8, device="CPU")
35 Ops.CONST dtypes.weakint [] 1
36 Ops.BUFFER dtypes.f32 [35] ParamArg(slot=9, dtype=dtypes.f32, device="CPU")
37 Ops.SINK dtypes.void [18, 32, 34, 36]
38 Ops.CONST dtypes.weakint [] 1020
39 Ops.SHRINK dtypes.i8 [8, 33, 38]
40 Ops.BITCAST dtypes.f32 [39] dtypes.f32
41 Ops.SHRINK dtypes.i8 [8, 9, 33]
42 Ops.CONST dtypes.weakint [] 1050112
43 Ops.CONST dtypes.weakint [] 4
44 Ops.SHRINK dtypes.i8 [8, 42, 43]
45 Ops.BITCAST dtypes.f32 [44] dtypes.f32
46 Ops.CALL dtypes.void [37, 24, 40, 41, 45] CallInfo()
47 Ops.CONST dtypes.weakint [] 16
48 Ops.BUFFER dtypes.f32 [47] ParamArg(slot=10, dtype=dtypes.f32, device="CPU")
49 Ops.SINK dtypes.void [5, 48]
50 Ops.CONST dtypes.weakint [] 3158528
51 Ops.CONST dtypes.weakint [] 64
52 Ops.SHRINK dtypes.i8 [8, 50, 51]
53 Ops.BITCAST dtypes.f32 [52] dtypes.f32
54 Ops.CALL dtypes.void [49, 15, 53] CallInfo()
55 Ops.CONST dtypes.weakint [] 28380
56 Ops.BUFFER dtypes.i8 [55] ParamArg(slot=11, dtype=dtypes.i8, device="CPU")
57 Ops.BUFFER dtypes.i8 [2] ParamArg(slot=12, dtype=dtypes.i8, device="CPU")
58 Ops.SINK dtypes.void [56, 21, 32, 57]
59 Ops.SHRINK dtypes.i8 [8, 9, 55]
60 Ops.CONST dtypes.weakint [] 28416
61 Ops.SHRINK dtypes.i8 [8, 60, 2]
62 Ops.CALL dtypes.void [58, 59, 27, 40, 61] CallInfo()
63 Ops.CONST dtypes.weakint [] 3157182
64 Ops.BUFFER dtypes.i8 [63] ParamArg(slot=13, dtype=dtypes.i8, device="CPU")
65 Ops.BUFFER dtypes.i8 [35] ParamArg(slot=14, dtype=dtypes.i8, device="CPU")
66 Ops.SINK dtypes.void [64, 65]
67 Ops.SHRINK dtypes.i8 [8, 9, 63]
68 Ops.CONST dtypes.weakint [] 3158784
69 Ops.SHRINK dtypes.i8 [8, 68, 35]
70 Ops.CALL dtypes.void [66, 67, 69] CallInfo()
71 Ops.PARAM dtypes.i8 [2] ParamArg(slot=0, dtype=dtypes.i8, device="CPU")
72 Ops.PARAM dtypes.f32 [4] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
73 Ops.STORE dtypes.void [71, 72]
74 Ops.SHRINK dtypes.i8 [8, 9, 2]
75 Ops.CALL dtypes.void [73, 74, 15] CallInfo()
76 Ops.SINK dtypes.void [48]
77 Ops.CALL dtypes.void [76, 53] CallInfo()
78 Ops.PARAM dtypes.i8 [19] ParamArg(slot=0, dtype=dtypes.i8, device="CPU")
79 Ops.PARAM dtypes.f32 [47] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
80 Ops.STORE dtypes.void [78, 79]
81 Ops.SHRINK dtypes.i8 [8, 9, 51]
82 Ops.BITCAST dtypes.f32 [81] dtypes.f32
83 Ops.CALL dtypes.void [80, 27, 82] CallInfo()
84 Ops.LINEAR dtypes.void [16, 28, 31, 46, 54, 62, 70, 75, 77, 83]

