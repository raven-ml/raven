# tolk D342 golden=1f3a25edb8ad26f13f000bf1a3963030
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 8
2 Ops.PARAM dtypes.i64 [1] ParamArg(slot=0, dtype=dtypes.i64)
3 Ops.CONST dtypes.weakint [] 2
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.RANGE dtypes.i32 [4] (AxisType.WEAK, 0)
6 Ops.SHL dtypes.i32 [5, 3]
7 Ops.INDEX dtypes.i64 [2, 6]
8 Ops.PARAM dtypes.i64 [1] ParamArg(slot=1, dtype=dtypes.i64)
9 Ops.INDEX dtypes.i64 [8, 6]
10 Ops.LOAD dtypes.i64 [9]
11 Ops.CONST dtypes.weakint [] 3
12 Ops.SHL dtypes.i64 [10, 11]
13 Ops.STORE dtypes.void [7, 12]
14 Ops.CONST dtypes.weakint [] 1
15 Ops.ADD dtypes.i32 [6, 14]
16 Ops.INDEX dtypes.i64 [2, 15]
17 Ops.INDEX dtypes.i64 [8, 15]
18 Ops.LOAD dtypes.i64 [17]
19 Ops.SHL dtypes.i64 [18, 11]
20 Ops.STORE dtypes.void [16, 19]
21 Ops.ADD dtypes.i32 [6, 3]
22 Ops.INDEX dtypes.i64 [2, 21]
23 Ops.INDEX dtypes.i64 [8, 21]
24 Ops.LOAD dtypes.i64 [23]
25 Ops.SHL dtypes.i64 [24, 11]
26 Ops.STORE dtypes.void [22, 25]
27 Ops.ADD dtypes.i32 [6, 11]
28 Ops.INDEX dtypes.i64 [2, 27]
29 Ops.INDEX dtypes.i64 [8, 27]
30 Ops.LOAD dtypes.i64 [29]
31 Ops.SHL dtypes.i64 [30, 11]
32 Ops.STORE dtypes.void [28, 31]
33 Ops.AFTER dtypes.void [0, 13, 20, 26, 32]
34 Ops.END dtypes.void [33, 5]
35 Ops.SINK dtypes.void [34] KernelInfo(name="E_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
36 Ops.CONST dtypes.weakint [] 16
37 Ops.PARAM dtypes.i32 [36] ParamArg(slot=0, dtype=dtypes.i32)
38 Ops.MUL dtypes.i32 [6, 3]
39 Ops.CONST dtypes.weakint [] 0
40 Ops.ADD dtypes.i32 [38, 39]
41 Ops.INDEX dtypes.i32 [37, 40]
42 Ops.CAST dtypes.i32 [39] dtypes.i32
43 Ops.CMPLT dtypes.bool [42, 42]
44 Ops.CMPEQ dtypes.bool [42, 42]
45 Ops.CAST dtypes.i32 [11] dtypes.i32
46 Ops.BITCAST dtypes.u32 [45] dtypes.u32
47 Ops.CONST dtypes.weakint [] 64
48 Ops.CAST dtypes.i32 [47] dtypes.i32
49 Ops.BITCAST dtypes.u32 [48] dtypes.u32
50 Ops.CMPLT dtypes.bool [46, 49]
51 Ops.AND dtypes.bool [44, 50]
52 Ops.OR dtypes.bool [43, 51]
53 Ops.CONST dtypes.weakint [] 32
54 Ops.CMPLT dtypes.bool [45, 53]
55 Ops.CONST dtypes.bool [] False
56 Ops.CMPEQ dtypes.bool [54, 55]
57 Ops.PARAM dtypes.i32 [36] ParamArg(slot=1, dtype=dtypes.i32)
58 Ops.INDEX dtypes.i32 [57, 40]
59 Ops.LOAD dtypes.i32 [58]
60 Ops.BITCAST dtypes.u32 [59] dtypes.u32
61 Ops.CONST dtypes.weakint [] 31
62 Ops.AND dtypes.i32 [45, 61]
63 Ops.CAST dtypes.u32 [62] dtypes.u32
64 Ops.SHL dtypes.u32 [60, 63]
65 Ops.BITCAST dtypes.i32 [64] dtypes.i32
66 Ops.WHERE dtypes.i32 [56, 42, 65]
67 Ops.WHERE dtypes.i32 [52, 66, 42]
68 Ops.STORE dtypes.void [41, 67]
69 Ops.ADD dtypes.i32 [38, 14]
70 Ops.INDEX dtypes.i32 [37, 69]
71 Ops.INDEX dtypes.i32 [57, 69]
72 Ops.LOAD dtypes.i32 [71]
73 Ops.BITCAST dtypes.u32 [72] dtypes.u32
74 Ops.SHL dtypes.u32 [73, 63]
75 Ops.SHR dtypes.u32 [60, 14]
76 Ops.CONST dtypes.weakint [] 4294967295
77 Ops.MUL dtypes.u32 [63, 76]
78 Ops.ADD dtypes.u32 [61, 77]
79 Ops.SHR dtypes.u32 [75, 78]
80 Ops.OR dtypes.u32 [74, 79]
81 Ops.BITCAST dtypes.i32 [80] dtypes.i32
82 Ops.WHERE dtypes.i32 [56, 65, 81]
83 Ops.WHERE dtypes.i32 [52, 82, 42]
84 Ops.STORE dtypes.void [70, 83]
85 Ops.AFTER dtypes.void [0, 68, 84]
86 Ops.MUL dtypes.i32 [15, 3]
87 Ops.ADD dtypes.i32 [86, 39]
88 Ops.INDEX dtypes.i32 [37, 87]
89 Ops.INDEX dtypes.i32 [57, 87]
90 Ops.LOAD dtypes.i32 [89]
91 Ops.BITCAST dtypes.u32 [90] dtypes.u32
92 Ops.SHL dtypes.u32 [91, 63]
93 Ops.BITCAST dtypes.i32 [92] dtypes.i32
94 Ops.WHERE dtypes.i32 [56, 42, 93]
95 Ops.WHERE dtypes.i32 [52, 94, 42]
96 Ops.STORE dtypes.void [88, 95]
97 Ops.ADD dtypes.i32 [86, 14]
98 Ops.INDEX dtypes.i32 [37, 97]
99 Ops.INDEX dtypes.i32 [57, 97]
100 Ops.LOAD dtypes.i32 [99]
101 Ops.BITCAST dtypes.u32 [100] dtypes.u32
102 Ops.SHL dtypes.u32 [101, 63]
103 Ops.SHR dtypes.u32 [91, 14]
104 Ops.SHR dtypes.u32 [103, 78]
105 Ops.OR dtypes.u32 [102, 104]
106 Ops.BITCAST dtypes.i32 [105] dtypes.i32
107 Ops.WHERE dtypes.i32 [56, 93, 106]
108 Ops.WHERE dtypes.i32 [52, 107, 42]
109 Ops.STORE dtypes.void [98, 108]
110 Ops.AFTER dtypes.void [0, 96, 109]
111 Ops.MUL dtypes.i32 [21, 3]
112 Ops.ADD dtypes.i32 [111, 39]
113 Ops.INDEX dtypes.i32 [37, 112]
114 Ops.INDEX dtypes.i32 [57, 112]
115 Ops.LOAD dtypes.i32 [114]
116 Ops.BITCAST dtypes.u32 [115] dtypes.u32
117 Ops.SHL dtypes.u32 [116, 63]
118 Ops.BITCAST dtypes.i32 [117] dtypes.i32
119 Ops.WHERE dtypes.i32 [56, 42, 118]
120 Ops.WHERE dtypes.i32 [52, 119, 42]
121 Ops.STORE dtypes.void [113, 120]
122 Ops.ADD dtypes.i32 [111, 14]
123 Ops.INDEX dtypes.i32 [37, 122]
124 Ops.INDEX dtypes.i32 [57, 122]
125 Ops.LOAD dtypes.i32 [124]
126 Ops.BITCAST dtypes.u32 [125] dtypes.u32
127 Ops.SHL dtypes.u32 [126, 63]
128 Ops.SHR dtypes.u32 [116, 14]
129 Ops.SHR dtypes.u32 [128, 78]
130 Ops.OR dtypes.u32 [127, 129]
131 Ops.BITCAST dtypes.i32 [130] dtypes.i32
132 Ops.WHERE dtypes.i32 [56, 118, 131]
133 Ops.WHERE dtypes.i32 [52, 132, 42]
134 Ops.STORE dtypes.void [123, 133]
135 Ops.AFTER dtypes.void [0, 121, 134]
136 Ops.MUL dtypes.i32 [27, 3]
137 Ops.ADD dtypes.i32 [136, 39]
138 Ops.INDEX dtypes.i32 [37, 137]
139 Ops.INDEX dtypes.i32 [57, 137]
140 Ops.LOAD dtypes.i32 [139]
141 Ops.BITCAST dtypes.u32 [140] dtypes.u32
142 Ops.SHL dtypes.u32 [141, 63]
143 Ops.BITCAST dtypes.i32 [142] dtypes.i32
144 Ops.WHERE dtypes.i32 [56, 42, 143]
145 Ops.WHERE dtypes.i32 [52, 144, 42]
146 Ops.STORE dtypes.void [138, 145]
147 Ops.ADD dtypes.i32 [136, 14]
148 Ops.INDEX dtypes.i32 [37, 147]
149 Ops.INDEX dtypes.i32 [57, 147]
150 Ops.LOAD dtypes.i32 [149]
151 Ops.BITCAST dtypes.u32 [150] dtypes.u32
152 Ops.SHL dtypes.u32 [151, 63]
153 Ops.SHR dtypes.u32 [141, 14]
154 Ops.SHR dtypes.u32 [153, 78]
155 Ops.OR dtypes.u32 [152, 154]
156 Ops.BITCAST dtypes.i32 [155] dtypes.i32
157 Ops.WHERE dtypes.i32 [56, 143, 156]
158 Ops.WHERE dtypes.i32 [52, 157, 42]
159 Ops.STORE dtypes.void [148, 158]
160 Ops.AFTER dtypes.void [0, 146, 159]
161 Ops.AFTER dtypes.void [0, 85, 110, 135, 160]
162 Ops.END dtypes.void [161, 5]
163 Ops.SINK dtypes.void [162] KernelInfo(name="E_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
164 Ops.SINK dtypes.void [35, 163]

