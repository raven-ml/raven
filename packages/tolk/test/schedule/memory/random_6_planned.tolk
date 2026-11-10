# tolk D41 D477 golden=c92f9000dd25e28525ba7d788a8c86a9
0 Ops.CONST dtypes.weakint [] 256
1 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.SINK dtypes.void [1]
3 Ops.CONST dtypes.weakint [] 9361664
4 Ops.BUFFER dtypes.i8 [3] ParamArg(slot=25, dtype=dtypes.i8, device="CPU")
5 Ops.CONST dtypes.weakint [] 4703744
6 Ops.CONST dtypes.weakint [] 1024
7 Ops.SHRINK dtypes.i8 [4, 5, 6]
8 Ops.BITCAST dtypes.f32 [7] dtypes.f32
9 Ops.CALL dtypes.void [2, 8] CallInfo()
10 Ops.CONST dtypes.weakint [] 4096
11 Ops.BUFFER dtypes.i8 [10] ParamArg(slot=1, dtype=dtypes.i8, device="CPU")
12 Ops.CONST dtypes.weakint [] 257
13 Ops.BUFFER dtypes.f32 [12] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
14 Ops.CONST dtypes.weakint [] 1048576
15 Ops.BUFFER dtypes.i8 [14] ParamArg(slot=3, dtype=dtypes.i8, device="CPU")
16 Ops.SINK dtypes.void [11, 13, 15]
17 Ops.CONST dtypes.weakint [] 4699648
18 Ops.SHRINK dtypes.i8 [4, 17, 10]
19 Ops.CONST dtypes.weakint [] 1028
20 Ops.SHRINK dtypes.i8 [4, 14, 19]
21 Ops.BITCAST dtypes.f32 [20] dtypes.f32
22 Ops.CONST dtypes.weakint [] 0
23 Ops.SHRINK dtypes.i8 [4, 22, 14]
24 Ops.CALL dtypes.void [16, 18, 21, 23] CallInfo()
25 Ops.CONST dtypes.weakint [] 1
26 Ops.BUFFER dtypes.f32 [25] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
27 Ops.CONST dtypes.weakint [] 255
28 Ops.BUFFER dtypes.i8 [27] ParamArg(slot=5, dtype=dtypes.i8, device="CPU")
29 Ops.CONST dtypes.weakint [] 65536
30 Ops.BUFFER dtypes.i8 [29] ParamArg(slot=6, dtype=dtypes.i8, device="CPU")
31 Ops.BUFFER dtypes.i8 [12] ParamArg(slot=7, dtype=dtypes.i8, device="CPU")
32 Ops.SINK dtypes.void [26, 28, 30, 31]
33 Ops.CONST dtypes.weakint [] 4705792
34 Ops.SHRINK dtypes.i8 [4, 33, 27]
35 Ops.CONST dtypes.weakint [] 9230336
36 Ops.SHRINK dtypes.i8 [4, 35, 29]
37 Ops.CONST dtypes.weakint [] 4706816
38 Ops.SHRINK dtypes.i8 [4, 37, 12]
39 Ops.CALL dtypes.void [32, 26, 34, 36, 38] CallInfo()
40 Ops.CONST dtypes.weakint [] 1000
41 Ops.BUFFER dtypes.i8 [40] ParamArg(slot=8, dtype=dtypes.i8, device="CPU")
42 Ops.SINK dtypes.void [41]
43 Ops.CONST dtypes.weakint [] 4704768
44 Ops.SHRINK dtypes.i8 [4, 43, 40]
45 Ops.CALL dtypes.void [42, 44] CallInfo()
46 Ops.BUFFER dtypes.i8 [12] ParamArg(slot=9, dtype=dtypes.i8, device="CPU")
47 Ops.BUFFER dtypes.f32 [25] ParamArg(slot=10, dtype=dtypes.f32, device="CPU")
48 Ops.BUFFER dtypes.f32 [14] ParamArg(slot=11, dtype=dtypes.f32, device="CPU")
49 Ops.SINK dtypes.void [46, 47, 28, 48]
50 Ops.CALL dtypes.void [49, 46, 47, 34, 48] CallInfo()
51 Ops.BUFFER dtypes.i8 [29] ParamArg(slot=12, dtype=dtypes.i8, device="CPU")
52 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=13, dtype=dtypes.f32, device="CPU")
53 Ops.CONST dtypes.weakint [] 1158514
54 Ops.BUFFER dtypes.f32 [53] ParamArg(slot=14, dtype=dtypes.f32, device="CPU")
55 Ops.SINK dtypes.void [51, 52, 54, 47]
56 Ops.CONST dtypes.weakint [] 4634112
57 Ops.SHRINK dtypes.i8 [4, 56, 29]
58 Ops.SHRINK dtypes.i8 [4, 33, 6]
59 Ops.BITCAST dtypes.f32 [58] dtypes.f32
60 Ops.CONST dtypes.weakint [] 4634056
61 Ops.SHRINK dtypes.i8 [4, 22, 60]
62 Ops.BITCAST dtypes.f32 [61] dtypes.f32
63 Ops.CALL dtypes.void [55, 57, 59, 62, 47] CallInfo()
64 Ops.BUFFER dtypes.i8 [29] ParamArg(slot=15, dtype=dtypes.i8, device="CPU")
65 Ops.SINK dtypes.void [52, 46, 64]
66 Ops.CONST dtypes.weakint [] 9295872
67 Ops.SHRINK dtypes.i8 [4, 66, 29]
68 Ops.CALL dtypes.void [65, 59, 46, 67] CallInfo()
69 Ops.SINK dtypes.void [48]
70 Ops.CALL dtypes.void [69, 48] CallInfo()
71 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=16, dtype=dtypes.i8, device="CPU")
72 Ops.SINK dtypes.void [31, 71, 30, 46]
73 Ops.SHRINK dtypes.i8 [4, 33, 0]
74 Ops.CALL dtypes.void [72, 38, 73, 36, 46] CallInfo()
75 Ops.CONST dtypes.weakint [] 16
76 Ops.BUFFER dtypes.i8 [75] ParamArg(slot=17, dtype=dtypes.i8, device="CPU")
77 Ops.BUFFER dtypes.i8 [29] ParamArg(slot=18, dtype=dtypes.i8, device="CPU")
78 Ops.SINK dtypes.void [46, 41, 76, 77]
79 Ops.SHRINK dtypes.i8 [4, 56, 75]
80 Ops.CALL dtypes.void [78, 46, 44, 79, 77] CallInfo()
81 Ops.SINK dtypes.void [26]
82 Ops.CALL dtypes.void [81, 26] CallInfo()
83 Ops.BUFFER dtypes.i8 [10] ParamArg(slot=19, dtype=dtypes.i8, device="CPU")
84 Ops.SINK dtypes.void [77, 83]
85 Ops.SHRINK dtypes.i8 [4, 56, 10]
86 Ops.CALL dtypes.void [84, 77, 85] CallInfo()
87 Ops.BUFFER dtypes.f32 [75] ParamArg(slot=20, dtype=dtypes.f32, device="CPU")
88 Ops.SINK dtypes.void [54, 11, 87]
89 Ops.CONST dtypes.weakint [] 9361408
90 Ops.CONST dtypes.weakint [] 64
91 Ops.SHRINK dtypes.i8 [4, 89, 90]
92 Ops.BITCAST dtypes.f32 [91] dtypes.f32
93 Ops.CALL dtypes.void [88, 62, 18, 92] CallInfo()
94 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=21, dtype=dtypes.i8, device="CPU")
95 Ops.SINK dtypes.void [94, 83]
96 Ops.CONST dtypes.weakint [] 4638208
97 Ops.SHRINK dtypes.i8 [4, 96, 0]
98 Ops.CALL dtypes.void [95, 97, 85] CallInfo()
99 Ops.SINK dtypes.void [1, 94, 47]
100 Ops.CALL dtypes.void [99, 8, 97, 47] CallInfo()
101 Ops.SINK dtypes.void [83]
102 Ops.CALL dtypes.void [101, 85] CallInfo()
103 Ops.BUFFER dtypes.i8 [14] ParamArg(slot=22, dtype=dtypes.i8, device="CPU")
104 Ops.CONST dtypes.weakint [] 3547448
105 Ops.BUFFER dtypes.i8 [104] ParamArg(slot=23, dtype=dtypes.i8, device="CPU")
106 Ops.SINK dtypes.void [103, 64, 105]
107 Ops.CONST dtypes.weakint [] 8181760
108 Ops.SHRINK dtypes.i8 [4, 107, 14]
109 Ops.SHRINK dtypes.i8 [4, 56, 104]
110 Ops.CALL dtypes.void [106, 108, 67, 109] CallInfo()
111 Ops.SINK dtypes.void [30, 87, 54]
112 Ops.CALL dtypes.void [111, 36, 92, 62] CallInfo()
113 Ops.CONST dtypes.weakint [] 747066
114 Ops.BUFFER dtypes.i8 [113] ParamArg(slot=24, dtype=dtypes.i8, device="CPU")
115 Ops.SINK dtypes.void [114]
116 Ops.SHRINK dtypes.i8 [4, 56, 113]
117 Ops.CALL dtypes.void [115, 116] CallInfo()
118 Ops.PARAM dtypes.f32 [53] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
119 Ops.PARAM dtypes.i8 [29] ParamArg(slot=1, dtype=dtypes.i8, device="CPU")
120 Ops.STORE dtypes.void [118, 119]
121 Ops.CALL dtypes.void [120, 62, 67] CallInfo()
122 Ops.LINEAR dtypes.void [9, 24, 39, 45, 50, 63, 68, 70, 74, 80, 82, 86, 93, 98, 100, 102, 110, 112, 117, 121]

