# tolk D477 golden=b532b3407036e318007f847204d0228c
0 Ops.CONST dtypes.weakint [] 1000
1 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=0, dtype=dtypes.i8, device="CPU")
2 Ops.CONST dtypes.weakint [] 16
3 Ops.BUFFER dtypes.i8 [2] ParamArg(slot=1, dtype=dtypes.i8, device="CPU")
4 Ops.CONST dtypes.weakint [] 257
5 Ops.BUFFER dtypes.i8 [4] ParamArg(slot=2, dtype=dtypes.i8, device="CPU")
6 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
7 Ops.SINK dtypes.void [1, 3, 5, 6]
8 Ops.CONST dtypes.weakint [] 3655680
9 Ops.BUFFER dtypes.i8 [8] ParamArg(slot=22, dtype=dtypes.i8, device="CPU")
10 Ops.CONST dtypes.weakint [] 3653632
11 Ops.SHRINK dtypes.i8 [9, 10, 0]
12 Ops.CONST dtypes.weakint [] 3649536
13 Ops.CONST dtypes.weakint [] 4000
14 Ops.SHRINK dtypes.i8 [9, 12, 13]
15 Ops.BITCAST dtypes.f32 [14] dtypes.f32
16 Ops.CALL dtypes.void [7, 11, 3, 5, 15] CallInfo()
17 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=4, dtype=dtypes.i8, device="CPU")
18 Ops.CONST dtypes.weakint [] 256
19 Ops.BUFFER dtypes.i8 [18] ParamArg(slot=5, dtype=dtypes.i8, device="CPU")
20 Ops.CONST dtypes.weakint [] 3518378
21 Ops.BUFFER dtypes.i8 [20] ParamArg(slot=6, dtype=dtypes.i8, device="CPU")
22 Ops.CONST dtypes.weakint [] 1048576
23 Ops.BUFFER dtypes.f32 [22] ParamArg(slot=7, dtype=dtypes.f32, device="CPU")
24 Ops.SINK dtypes.void [17, 19, 21, 23]
25 Ops.CONST dtypes.weakint [] 3518464
26 Ops.SHRINK dtypes.i8 [9, 25, 0]
27 Ops.CONST dtypes.weakint [] 3654656
28 Ops.SHRINK dtypes.i8 [9, 27, 18]
29 Ops.CONST dtypes.weakint [] 0
30 Ops.SHRINK dtypes.i8 [9, 29, 20]
31 Ops.CALL dtypes.void [24, 26, 28, 30, 23] CallInfo()
32 Ops.BUFFER dtypes.i8 [22] ParamArg(slot=8, dtype=dtypes.i8, device="CPU")
33 Ops.CONST dtypes.weakint [] 1
34 Ops.BUFFER dtypes.f32 [33] ParamArg(slot=9, dtype=dtypes.f32, device="CPU")
35 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=10, dtype=dtypes.i8, device="CPU")
36 Ops.SINK dtypes.void [32, 34, 35, 17]
37 Ops.CONST dtypes.weakint [] 3520512
38 Ops.CONST dtypes.weakint [] 4
39 Ops.SHRINK dtypes.i8 [9, 37, 38]
40 Ops.BITCAST dtypes.f32 [39] dtypes.f32
41 Ops.CONST dtypes.weakint [] 3519488
42 Ops.SHRINK dtypes.i8 [9, 41, 0]
43 Ops.CALL dtypes.void [36, 32, 40, 42, 26] CallInfo()
44 Ops.BUFFER dtypes.i8 [18] ParamArg(slot=11, dtype=dtypes.i8, device="CPU")
45 Ops.SINK dtypes.void [5, 44]
46 Ops.SHRINK dtypes.i8 [9, 37, 18]
47 Ops.CALL dtypes.void [45, 5, 46] CallInfo()
48 Ops.BUFFER dtypes.f32 [22] ParamArg(slot=12, dtype=dtypes.f32, device="CPU")
49 Ops.BUFFER dtypes.f32 [18] ParamArg(slot=13, dtype=dtypes.f32, device="CPU")
50 Ops.SINK dtypes.void [44, 48, 49]
51 Ops.CALL dtypes.void [50, 46, 48, 49] CallInfo()
52 Ops.BUFFER dtypes.i8 [18] ParamArg(slot=14, dtype=dtypes.i8, device="CPU")
53 Ops.BUFFER dtypes.i8 [2] ParamArg(slot=15, dtype=dtypes.i8, device="CPU")
54 Ops.SINK dtypes.void [17, 52, 53]
55 Ops.CONST dtypes.weakint [] 3654912
56 Ops.SHRINK dtypes.i8 [9, 55, 18]
57 Ops.CONST dtypes.weakint [] 3655168
58 Ops.SHRINK dtypes.i8 [9, 57, 2]
59 Ops.CALL dtypes.void [54, 26, 56, 58] CallInfo()
60 Ops.SINK dtypes.void [23, 49]
61 Ops.CALL dtypes.void [60, 23, 49] CallInfo()
62 Ops.SINK dtypes.void [35]
63 Ops.CALL dtypes.void [62, 42] CallInfo()
64 Ops.SINK dtypes.void [6, 52, 49]
65 Ops.CALL dtypes.void [64, 15, 56, 49] CallInfo()
66 Ops.CONST dtypes.weakint [] 65536
67 Ops.BUFFER dtypes.i8 [66] ParamArg(slot=16, dtype=dtypes.i8, device="CPU")
68 Ops.BUFFER dtypes.i8 [2] ParamArg(slot=17, dtype=dtypes.i8, device="CPU")
69 Ops.BUFFER dtypes.i8 [66] ParamArg(slot=18, dtype=dtypes.i8, device="CPU")
70 Ops.SINK dtypes.void [67, 68, 69]
71 Ops.SHRINK dtypes.i8 [9, 25, 66]
72 Ops.CONST dtypes.weakint [] 3655424
73 Ops.SHRINK dtypes.i8 [9, 72, 2]
74 Ops.CONST dtypes.weakint [] 3584000
75 Ops.SHRINK dtypes.i8 [9, 74, 66]
76 Ops.CALL dtypes.void [70, 71, 73, 75] CallInfo()
77 Ops.SINK dtypes.void [52, 6]
78 Ops.CALL dtypes.void [77, 56, 15] CallInfo()
79 Ops.BUFFER dtypes.f32 [66] ParamArg(slot=19, dtype=dtypes.f32, device="CPU")
80 Ops.SINK dtypes.void [79, 5, 3, 1]
81 Ops.CALL dtypes.void [80, 79, 5, 3, 11] CallInfo()
82 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=20, dtype=dtypes.i8, device="CPU")
83 Ops.SINK dtypes.void [82]
84 Ops.CALL dtypes.void [83, 26] CallInfo()
85 Ops.CONST dtypes.weakint [] 255
86 Ops.BUFFER dtypes.f32 [85] ParamArg(slot=21, dtype=dtypes.f32, device="CPU")
87 Ops.SINK dtypes.void [53, 21, 19, 86]
88 Ops.CONST dtypes.weakint [] 1020
89 Ops.SHRINK dtypes.i8 [9, 25, 88]
90 Ops.BITCAST dtypes.f32 [89] dtypes.f32
91 Ops.CALL dtypes.void [87, 58, 30, 28, 90] CallInfo()
92 Ops.LINEAR dtypes.void [16, 31, 43, 47, 51, 59, 61, 63, 65, 76, 78, 81, 84, 91]

