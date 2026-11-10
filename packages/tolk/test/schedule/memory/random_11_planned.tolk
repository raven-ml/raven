# tolk D41 D477 golden=16ef20339c90b2dd42d0e8765b4950d4
0 Ops.CONST dtypes.weakint [] 255
1 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=0, dtype=dtypes.i8, device="CPU")
2 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
3 Ops.CONST dtypes.weakint [] 16
4 Ops.BUFFER dtypes.i8 [3] ParamArg(slot=2, dtype=dtypes.i8, device="CPU")
5 Ops.CONST dtypes.weakint [] 257
6 Ops.BUFFER dtypes.f32 [5] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
7 Ops.SINK dtypes.void [1, 2, 4, 6]
8 Ops.CONST dtypes.weakint [] 13317120
9 Ops.BUFFER dtypes.i8 [8] ParamArg(slot=16, dtype=dtypes.i8, device="CPU")
10 Ops.CONST dtypes.weakint [] 65536
11 Ops.SHRINK dtypes.i8 [9, 10, 0]
12 Ops.CONST dtypes.weakint [] 13314304
13 Ops.CONST dtypes.weakint [] 1020
14 Ops.SHRINK dtypes.i8 [9, 12, 13]
15 Ops.BITCAST dtypes.f32 [14] dtypes.f32
16 Ops.CONST dtypes.weakint [] 13311744
17 Ops.CONST dtypes.weakint [] 1028
18 Ops.SHRINK dtypes.i8 [9, 16, 17]
19 Ops.BITCAST dtypes.f32 [18] dtypes.f32
20 Ops.CALL dtypes.void [7, 11, 15, 4, 19] CallInfo()
21 Ops.BUFFER dtypes.i8 [10] ParamArg(slot=4, dtype=dtypes.i8, device="CPU")
22 Ops.CONST dtypes.weakint [] 256
23 Ops.BUFFER dtypes.f32 [22] ParamArg(slot=5, dtype=dtypes.f32, device="CPU")
24 Ops.BUFFER dtypes.i8 [10] ParamArg(slot=6, dtype=dtypes.i8, device="CPU")
25 Ops.CONST dtypes.weakint [] 1
26 Ops.BUFFER dtypes.i8 [25] ParamArg(slot=7, dtype=dtypes.i8, device="CPU")
27 Ops.SINK dtypes.void [21, 23, 24, 26]
28 Ops.CONST dtypes.weakint [] 0
29 Ops.SHRINK dtypes.i8 [9, 28, 10]
30 Ops.CONST dtypes.weakint [] 13315328
31 Ops.CONST dtypes.weakint [] 1024
32 Ops.SHRINK dtypes.i8 [9, 30, 31]
33 Ops.BITCAST dtypes.f32 [32] dtypes.f32
34 Ops.CONST dtypes.weakint [] 13229824
35 Ops.SHRINK dtypes.i8 [9, 34, 10]
36 Ops.CONST dtypes.weakint [] 13313024
37 Ops.SHRINK dtypes.i8 [9, 36, 25]
38 Ops.CALL dtypes.void [27, 29, 33, 35, 37] CallInfo()
39 Ops.BUFFER dtypes.i8 [25] ParamArg(slot=8, dtype=dtypes.i8, device="CPU")
40 Ops.SINK dtypes.void [4, 39]
41 Ops.CALL dtypes.void [40, 4, 39] CallInfo()
42 Ops.CONST dtypes.weakint [] 4096
43 Ops.BUFFER dtypes.f32 [42] ParamArg(slot=9, dtype=dtypes.f32, device="CPU")
44 Ops.BUFFER dtypes.f32 [5] ParamArg(slot=10, dtype=dtypes.f32, device="CPU")
45 Ops.SINK dtypes.void [1, 21, 43, 44]
46 Ops.CONST dtypes.weakint [] 13295360
47 Ops.CONST dtypes.weakint [] 16384
48 Ops.SHRINK dtypes.i8 [9, 46, 47]
49 Ops.BITCAST dtypes.f32 [48] dtypes.f32
50 Ops.SHRINK dtypes.i8 [9, 36, 17]
51 Ops.BITCAST dtypes.f32 [50] dtypes.f32
52 Ops.CALL dtypes.void [45, 11, 29, 49, 51] CallInfo()
53 Ops.SINK dtypes.void [39]
54 Ops.CALL dtypes.void [53, 39] CallInfo()
55 Ops.SINK dtypes.void [2]
56 Ops.CALL dtypes.void [55, 15] CallInfo()
57 Ops.PARAM dtypes.f32 [5] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
58 Ops.PARAM dtypes.i8 [0] ParamArg(slot=1, dtype=dtypes.i8, device="CPU")
59 Ops.STORE dtypes.void [57, 58]
60 Ops.CALL dtypes.void [59, 51, 11] CallInfo()
61 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=11, dtype=dtypes.i8, device="CPU")
62 Ops.SINK dtypes.void [1, 2, 61]
63 Ops.CONST dtypes.weakint [] 13316352
64 Ops.SHRINK dtypes.i8 [9, 63, 0]
65 Ops.CALL dtypes.void [62, 11, 15, 64] CallInfo()
66 Ops.PARAM dtypes.f32 [42] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
67 Ops.CONST dtypes.weakint [] 3307416
68 Ops.PARAM dtypes.f32 [67] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
69 Ops.STORE dtypes.void [66, 68]
70 Ops.CONST dtypes.weakint [] 13229664
71 Ops.SHRINK dtypes.i8 [9, 28, 70]
72 Ops.BITCAST dtypes.f32 [71] dtypes.f32
73 Ops.CALL dtypes.void [69, 49, 72] CallInfo()
74 Ops.SINK dtypes.void [4]
75 Ops.CALL dtypes.void [74, 4] CallInfo()
76 Ops.BUFFER dtypes.i8 [25] ParamArg(slot=13, dtype=dtypes.i8, device="CPU")
77 Ops.BUFFER dtypes.i8 [25] ParamArg(slot=14, dtype=dtypes.i8, device="CPU")
78 Ops.BUFFER dtypes.f32 [67] ParamArg(slot=12, dtype=dtypes.f32, device="CPU")
79 Ops.SINK dtypes.void [61, 76, 77, 78]
80 Ops.CONST dtypes.weakint [] 13316608
81 Ops.SHRINK dtypes.i8 [9, 80, 25]
82 Ops.CONST dtypes.weakint [] 13316864
83 Ops.SHRINK dtypes.i8 [9, 82, 25]
84 Ops.CALL dtypes.void [79, 64, 81, 83, 72] CallInfo()
85 Ops.SINK dtypes.void [61]
86 Ops.CALL dtypes.void [85, 64] CallInfo()
87 Ops.SINK dtypes.void [24, 43, 44]
88 Ops.CALL dtypes.void [87, 35, 49, 51] CallInfo()
89 Ops.SINK dtypes.void [23]
90 Ops.CALL dtypes.void [89, 33] CallInfo()
91 Ops.SINK dtypes.void [23, 43, 2, 39]
92 Ops.CALL dtypes.void [91, 33, 49, 15, 39] CallInfo()
93 Ops.SINK dtypes.void [44]
94 Ops.CALL dtypes.void [93, 51] CallInfo()
95 Ops.SINK dtypes.void [2, 39]
96 Ops.CALL dtypes.void [95, 15, 39] CallInfo()
97 Ops.SINK dtypes.void [6, 77, 2, 39]
98 Ops.CALL dtypes.void [97, 19, 83, 15, 39] CallInfo()
99 Ops.PARAM dtypes.i8 [0] ParamArg(slot=0, dtype=dtypes.i8, device="CPU")
100 Ops.PARAM dtypes.i8 [42] ParamArg(slot=1, dtype=dtypes.i8, device="CPU")
101 Ops.STORE dtypes.void [99, 100]
102 Ops.SHRINK dtypes.i8 [9, 28, 42]
103 Ops.CALL dtypes.void [101, 64, 102] CallInfo()
104 Ops.LINEAR dtypes.void [20, 38, 41, 52, 54, 56, 60, 65, 73, 75, 84, 86, 88, 90, 92, 94, 96, 98, 103]

