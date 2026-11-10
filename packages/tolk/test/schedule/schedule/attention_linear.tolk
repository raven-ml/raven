# tolk D163 D183 golden=3493cad5ee1edc0d8bac98c96505aed5
0 Ops.CONST dtypes.weakint [] 256
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 3)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 4)
6 Ops.ADD dtypes.weakint [4, 5]
7 Ops.CONST dtypes.weakint [] 2
8 Ops.RANGE dtypes.weakint [7] (AxisType.WEAK, 2)
9 Ops.CONST dtypes.weakint [] 64
10 Ops.MUL dtypes.weakint [8, 9]
11 Ops.ADD dtypes.weakint [6, 10]
12 Ops.RANGE dtypes.weakint [7] (AxisType.WEAK, 1)
13 Ops.CONST dtypes.weakint [] 128
14 Ops.MUL dtypes.weakint [12, 13]
15 Ops.ADD dtypes.weakint [11, 14]
16 Ops.INDEX dtypes.f32 [1, 15]
17 Ops.PARAM dtypes.f32 [13] ParamArg(slot=1, dtype=dtypes.f32)
18 Ops.CONST dtypes.weakint [] 4
19 Ops.MUL dtypes.weakint [3, 18]
20 Ops.RANGE dtypes.weakint [18] (AxisType.WEAK, 0)
21 Ops.ADD dtypes.weakint [19, 20]
22 Ops.CONST dtypes.weakint [] 32
23 Ops.MUL dtypes.weakint [8, 22]
24 Ops.ADD dtypes.weakint [21, 23]
25 Ops.MUL dtypes.weakint [12, 9]
26 Ops.ADD dtypes.weakint [24, 25]
27 Ops.INDEX dtypes.f32 [17, 26]
28 Ops.PARAM dtypes.f32 [13] ParamArg(slot=2, dtype=dtypes.f32)
29 Ops.MUL dtypes.weakint [5, 18]
30 Ops.ADD dtypes.weakint [29, 20]
31 Ops.ADD dtypes.weakint [30, 23]
32 Ops.ADD dtypes.weakint [31, 25]
33 Ops.INDEX dtypes.f32 [28, 32]
34 Ops.MUL dtypes.f32 [27, 33]
35 Ops.REDUCE dtypes.f32 [34, 20] (Ops.ADD, 0)
36 Ops.CONST dtypes.weakfloat [] 0.5
37 Ops.MUL dtypes.f32 [35, 36]
38 Ops.STORE dtypes.void [16, 37]
39 Ops.END dtypes.void [38, 12, 8, 3, 5]
40 Ops.SINK dtypes.void [39] KernelInfo()
41 Ops.CONST dtypes.weakint [] 2560
42 Ops.BUFFER dtypes.i8 [41] ParamArg(slot=10, dtype=dtypes.i8, device="CPU")
43 Ops.CONST dtypes.weakint [] 0
44 Ops.CONST dtypes.weakint [] 1024
45 Ops.SHRINK dtypes.i8 [42, 43, 44]
46 Ops.BITCAST dtypes.f32 [45] dtypes.f32
47 Ops.BUFFER dtypes.f32 [13] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
48 Ops.BUFFER dtypes.f32 [13] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
49 Ops.CALL dtypes.void [40, 46, 47, 48] CallInfo()
50 Ops.PARAM dtypes.f32 [22] ParamArg(slot=0, dtype=dtypes.f32)
51 Ops.MUL dtypes.weakint [8, 2]
52 Ops.ADD dtypes.weakint [51, 3]
53 Ops.CONST dtypes.weakint [] 16
54 Ops.MUL dtypes.weakint [12, 53]
55 Ops.ADD dtypes.weakint [52, 54]
56 Ops.INDEX dtypes.f32 [50, 55]
57 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
58 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
59 Ops.ADD dtypes.weakint [4, 58]
60 Ops.ADD dtypes.weakint [59, 10]
61 Ops.ADD dtypes.weakint [60, 14]
62 Ops.INDEX dtypes.f32 [57, 61]
63 Ops.REDUCE dtypes.f32 [62, 58] (Ops.MAX, 0)
64 Ops.STORE dtypes.void [56, 63]
65 Ops.END dtypes.void [64, 12, 8, 3]
66 Ops.SINK dtypes.void [65] KernelInfo()
67 Ops.CONST dtypes.weakint [] 2048
68 Ops.SHRINK dtypes.i8 [42, 67, 13]
69 Ops.BITCAST dtypes.f32 [68] dtypes.f32
70 Ops.CALL dtypes.void [66, 69, 46] CallInfo()
71 Ops.PARAM dtypes.f32 [22] ParamArg(slot=2, dtype=dtypes.f32)
72 Ops.INDEX dtypes.f32 [71, 55]
73 Ops.CONST dtypes.weakfloat [] -1.0
74 Ops.MUL dtypes.f32 [72, 73]
75 Ops.ADD dtypes.f32 [62, 74]
76 Ops.CONST dtypes.weakfloat [] 1.4426950408889634
77 Ops.MUL dtypes.f32 [75, 76]
78 Ops.EXP2 dtypes.f32 [77]
79 Ops.REDUCE dtypes.f32 [78, 58] (Ops.ADD, 0)
80 Ops.STORE dtypes.void [56, 79]
81 Ops.END dtypes.void [80, 12, 8, 3]
82 Ops.SINK dtypes.void [81] KernelInfo()
83 Ops.CONST dtypes.weakint [] 2304
84 Ops.SHRINK dtypes.i8 [42, 83, 13]
85 Ops.BITCAST dtypes.f32 [84] dtypes.f32
86 Ops.CALL dtypes.void [82, 85, 46, 69] CallInfo()
87 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
88 Ops.MUL dtypes.weakint [87, 2]
89 Ops.ADD dtypes.weakint [88, 3]
90 Ops.ADD dtypes.weakint [89, 25]
91 Ops.RANGE dtypes.weakint [7] (AxisType.WEAK, 0)
92 Ops.MUL dtypes.weakint [91, 13]
93 Ops.ADD dtypes.weakint [90, 92]
94 Ops.INDEX dtypes.f32 [1, 93]
95 Ops.INDEX dtypes.f32 [57, 93]
96 Ops.MUL dtypes.weakint [12, 2]
97 Ops.ADD dtypes.weakint [96, 87]
98 Ops.MUL dtypes.weakint [91, 53]
99 Ops.ADD dtypes.weakint [97, 98]
100 Ops.INDEX dtypes.f32 [71, 99]
101 Ops.MUL dtypes.f32 [100, 73]
102 Ops.ADD dtypes.f32 [95, 101]
103 Ops.MUL dtypes.f32 [102, 76]
104 Ops.EXP2 dtypes.f32 [103]
105 Ops.PARAM dtypes.f32 [22] ParamArg(slot=3, dtype=dtypes.f32)
106 Ops.INDEX dtypes.f32 [105, 99]
107 Ops.RECIP dtypes.f32 [106]
108 Ops.MUL dtypes.f32 [104, 107]
109 Ops.STORE dtypes.void [94, 108]
110 Ops.END dtypes.void [109, 91, 12, 87, 3]
111 Ops.SINK dtypes.void [110] KernelInfo()
112 Ops.SHRINK dtypes.i8 [42, 44, 44]
113 Ops.BITCAST dtypes.f32 [112] dtypes.f32
114 Ops.CALL dtypes.void [111, 113, 46, 69, 85] CallInfo()
115 Ops.PARAM dtypes.f32 [13] ParamArg(slot=0, dtype=dtypes.f32)
116 Ops.RANGE dtypes.weakint [18] (AxisType.WEAK, 4)
117 Ops.ADD dtypes.weakint [19, 116]
118 Ops.ADD dtypes.weakint [117, 23]
119 Ops.ADD dtypes.weakint [118, 25]
120 Ops.INDEX dtypes.f32 [115, 119]
121 Ops.MUL dtypes.weakint [58, 18]
122 Ops.ADD dtypes.weakint [121, 116]
123 Ops.ADD dtypes.weakint [122, 23]
124 Ops.ADD dtypes.weakint [123, 25]
125 Ops.INDEX dtypes.f32 [28, 124]
126 Ops.MUL dtypes.f32 [62, 125]
127 Ops.REDUCE dtypes.f32 [126, 58] (Ops.ADD, 0)
128 Ops.STORE dtypes.void [120, 127]
129 Ops.END dtypes.void [128, 12, 8, 3, 116]
130 Ops.SINK dtypes.void [129] KernelInfo()
131 Ops.BUFFER dtypes.f32 [13] ParamArg(slot=6, dtype=dtypes.f32, device="CPU")
132 Ops.BUFFER dtypes.f32 [13] ParamArg(slot=5, dtype=dtypes.f32, device="CPU")
133 Ops.CALL dtypes.void [130, 131, 113, 132] CallInfo()
134 Ops.LINEAR dtypes.void [49, 70, 86, 114, 133]

