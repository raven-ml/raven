# tolk D41 D477 golden=7dfe081c75091faaf7f6302e990dd578
0 Ops.CONST dtypes.weakint [] 256
1 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=0, dtype=dtypes.i8, device="CPU")
2 Ops.CONST dtypes.weakint [] 1
3 Ops.BUFFER dtypes.i8 [2] ParamArg(slot=1, dtype=dtypes.i8, device="CPU")
4 Ops.CONST dtypes.weakint [] 16
5 Ops.BUFFER dtypes.i8 [4] ParamArg(slot=2, dtype=dtypes.i8, device="CPU")
6 Ops.SINK dtypes.void [1, 3, 5]
7 Ops.CONST dtypes.weakint [] 9188352
8 Ops.BUFFER dtypes.i8 [7] ParamArg(slot=17, dtype=dtypes.i8, device="CPU")
9 Ops.CONST dtypes.weakint [] 5427200
10 Ops.SHRINK dtypes.i8 [8, 9, 0]
11 Ops.CONST dtypes.weakint [] 9187840
12 Ops.SHRINK dtypes.i8 [8, 11, 2]
13 Ops.CONST dtypes.weakint [] 9188096
14 Ops.SHRINK dtypes.i8 [8, 13, 4]
15 Ops.CALL dtypes.void [6, 10, 12, 14] CallInfo()
16 Ops.CONST dtypes.weakint [] 1000
17 Ops.BUFFER dtypes.i8 [16] ParamArg(slot=3, dtype=dtypes.i8, device="CPU")
18 Ops.SINK dtypes.void [17]
19 Ops.CONST dtypes.weakint [] 9186816
20 Ops.SHRINK dtypes.i8 [8, 19, 16]
21 Ops.CALL dtypes.void [18, 20] CallInfo()
22 Ops.CONST dtypes.weakint [] 1356772
23 Ops.BUFFER dtypes.f32 [22] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
24 Ops.SINK dtypes.void [5, 17, 1, 23]
25 Ops.CONST dtypes.weakint [] 0
26 Ops.CONST dtypes.weakint [] 5427088
27 Ops.SHRINK dtypes.i8 [8, 25, 26]
28 Ops.BITCAST dtypes.f32 [27] dtypes.f32
29 Ops.CALL dtypes.void [24, 14, 20, 10, 28] CallInfo()
30 Ops.CONST dtypes.weakint [] 922461
31 Ops.BUFFER dtypes.f32 [30] ParamArg(slot=5, dtype=dtypes.f32, device="CPU")
32 Ops.SINK dtypes.void [31]
33 Ops.CONST dtypes.weakint [] 3689844
34 Ops.SHRINK dtypes.i8 [8, 9, 33]
35 Ops.BITCAST dtypes.f32 [34] dtypes.f32
36 Ops.CALL dtypes.void [32, 35] CallInfo()
37 Ops.CONST dtypes.weakint [] 4096
38 Ops.BUFFER dtypes.i8 [37] ParamArg(slot=6, dtype=dtypes.i8, device="CPU")
39 Ops.CONST dtypes.weakint [] 65536
40 Ops.BUFFER dtypes.i8 [39] ParamArg(slot=7, dtype=dtypes.i8, device="CPU")
41 Ops.SINK dtypes.void [38, 40]
42 Ops.CONST dtypes.weakint [] 9182720
43 Ops.SHRINK dtypes.i8 [8, 42, 37]
44 Ops.CONST dtypes.weakint [] 9117184
45 Ops.SHRINK dtypes.i8 [8, 44, 39]
46 Ops.CALL dtypes.void [41, 43, 45] CallInfo()
47 Ops.CONST dtypes.weakint [] 1048576
48 Ops.PARAM dtypes.i8 [47] ParamArg(slot=0, dtype=dtypes.i8, device="CPU")
49 Ops.PARAM dtypes.i8 [4] ParamArg(slot=1, dtype=dtypes.i8, device="CPU")
50 Ops.STORE dtypes.void [48, 49]
51 Ops.BUFFER dtypes.i8 [47] ParamArg(slot=8, dtype=dtypes.i8, device="CPU")
52 Ops.SHRINK dtypes.i8 [8, 42, 4]
53 Ops.CALL dtypes.void [50, 51, 52] CallInfo()
54 Ops.CONST dtypes.weakint [] 255
55 Ops.BUFFER dtypes.f32 [54] ParamArg(slot=10, dtype=dtypes.f32, device="CPU")
56 Ops.SINK dtypes.void [55]
57 Ops.CALL dtypes.void [56, 55] CallInfo()
58 Ops.BUFFER dtypes.i8 [37] ParamArg(slot=11, dtype=dtypes.i8, device="CPU")
59 Ops.SINK dtypes.void [58, 5]
60 Ops.CALL dtypes.void [59, 43, 14] CallInfo()
61 Ops.SINK dtypes.void [55, 23, 58, 17]
62 Ops.CALL dtypes.void [61, 55, 28, 43, 20] CallInfo()
63 Ops.PARAM dtypes.i8 [39] ParamArg(slot=0, dtype=dtypes.i8, device="CPU")
64 Ops.STORE dtypes.void [63, 49]
65 Ops.CALL dtypes.void [64, 45, 14] CallInfo()
66 Ops.CONST dtypes.weakint [] 257
67 Ops.BUFFER dtypes.f32 [66] ParamArg(slot=12, dtype=dtypes.f32, device="CPU")
68 Ops.SINK dtypes.void [67, 31, 3]
69 Ops.CONST dtypes.weakint [] 16384
70 Ops.CONST dtypes.weakint [] 1028
71 Ops.SHRINK dtypes.i8 [8, 69, 70]
72 Ops.BITCAST dtypes.f32 [71] dtypes.f32
73 Ops.CALL dtypes.void [68, 72, 35, 12] CallInfo()
74 Ops.BUFFER dtypes.i8 [37] ParamArg(slot=13, dtype=dtypes.i8, device="CPU")
75 Ops.BUFFER dtypes.i8 [16] ParamArg(slot=14, dtype=dtypes.i8, device="CPU")
76 Ops.SINK dtypes.void [74, 67, 75]
77 Ops.SHRINK dtypes.i8 [8, 25, 16]
78 Ops.CALL dtypes.void [76, 74, 72, 77] CallInfo()
79 Ops.SINK dtypes.void [40]
80 Ops.CALL dtypes.void [79, 45] CallInfo()
81 Ops.BUFFER dtypes.f32 [37] ParamArg(slot=15, dtype=dtypes.f32, device="CPU")
82 Ops.SINK dtypes.void [3, 81, 55, 67]
83 Ops.SHRINK dtypes.i8 [8, 25, 69]
84 Ops.BITCAST dtypes.f32 [83] dtypes.f32
85 Ops.CALL dtypes.void [82, 12, 84, 55, 72] CallInfo()
86 Ops.SINK dtypes.void [81]
87 Ops.CALL dtypes.void [86, 84] CallInfo()
88 Ops.SINK dtypes.void [67]
89 Ops.CALL dtypes.void [88, 72] CallInfo()
90 Ops.SINK dtypes.void [17, 81]
91 Ops.CALL dtypes.void [90, 20, 84] CallInfo()
92 Ops.SINK dtypes.void [3]
93 Ops.CALL dtypes.void [92, 12] CallInfo()
94 Ops.SINK dtypes.void [58]
95 Ops.CALL dtypes.void [94, 43] CallInfo()
96 Ops.CONST dtypes.weakint [] 1667422
97 Ops.BUFFER dtypes.f32 [96] ParamArg(slot=16, dtype=dtypes.f32, device="CPU")
98 Ops.SINK dtypes.void [97]
99 Ops.CONST dtypes.weakint [] 6669688
100 Ops.SHRINK dtypes.i8 [8, 25, 99]
101 Ops.BITCAST dtypes.f32 [100] dtypes.f32
102 Ops.CALL dtypes.void [98, 101] CallInfo()
103 Ops.LINEAR dtypes.void [15, 21, 29, 36, 46, 53, 57, 60, 62, 65, 73, 78, 80, 36, 85, 87, 89, 91, 93, 95, 102]

