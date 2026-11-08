# tolk D342 golden=01e1926a3ec7fff7e338d12b23066d54
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
11 Ops.CONST dtypes.weakint [] 5
12 Ops.SHR dtypes.i64 [10, 11]
13 Ops.STORE dtypes.void [7, 12]
14 Ops.CONST dtypes.weakint [] 1
15 Ops.ADD dtypes.i32 [6, 14]
16 Ops.INDEX dtypes.i64 [2, 15]
17 Ops.INDEX dtypes.i64 [8, 15]
18 Ops.LOAD dtypes.i64 [17]
19 Ops.SHR dtypes.i64 [18, 11]
20 Ops.STORE dtypes.void [16, 19]
21 Ops.ADD dtypes.i32 [6, 3]
22 Ops.INDEX dtypes.i64 [2, 21]
23 Ops.INDEX dtypes.i64 [8, 21]
24 Ops.LOAD dtypes.i64 [23]
25 Ops.SHR dtypes.i64 [24, 11]
26 Ops.STORE dtypes.void [22, 25]
27 Ops.CONST dtypes.weakint [] 3
28 Ops.ADD dtypes.i32 [6, 27]
29 Ops.INDEX dtypes.i64 [2, 28]
30 Ops.INDEX dtypes.i64 [8, 28]
31 Ops.LOAD dtypes.i64 [30]
32 Ops.SHR dtypes.i64 [31, 11]
33 Ops.STORE dtypes.void [29, 32]
34 Ops.AFTER dtypes.void [0, 13, 20, 26, 33]
35 Ops.END dtypes.void [34, 5]
36 Ops.SINK dtypes.void [35] KernelInfo(name="E_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
37 Ops.CONST dtypes.weakint [] 16
38 Ops.PARAM dtypes.i32 [37] ParamArg(slot=0, dtype=dtypes.i32)
39 Ops.MUL dtypes.i32 [6, 3]
40 Ops.CONST dtypes.weakint [] 0
41 Ops.ADD dtypes.i32 [39, 40]
42 Ops.INDEX dtypes.i32 [38, 41]
43 Ops.CAST dtypes.i32 [40] dtypes.i32
44 Ops.CMPLT dtypes.bool [43, 43]
45 Ops.CMPEQ dtypes.bool [43, 43]
46 Ops.CAST dtypes.i32 [11] dtypes.i32
47 Ops.BITCAST dtypes.u32 [46] dtypes.u32
48 Ops.CONST dtypes.weakint [] 64
49 Ops.CAST dtypes.i32 [48] dtypes.i32
50 Ops.BITCAST dtypes.u32 [49] dtypes.u32
51 Ops.CMPLT dtypes.bool [47, 50]
52 Ops.AND dtypes.bool [45, 51]
53 Ops.OR dtypes.bool [44, 52]
54 Ops.CONST dtypes.weakint [] 32
55 Ops.CMPLT dtypes.bool [46, 54]
56 Ops.CONST dtypes.bool [] False
57 Ops.CMPEQ dtypes.bool [55, 56]
58 Ops.PARAM dtypes.i32 [37] ParamArg(slot=1, dtype=dtypes.i32)
59 Ops.ADD dtypes.i32 [39, 14]
60 Ops.INDEX dtypes.i32 [58, 59]
61 Ops.LOAD dtypes.i32 [60]
62 Ops.CONST dtypes.weakint [] 31
63 Ops.AND dtypes.i32 [46, 62]
64 Ops.SHR dtypes.i32 [61, 63]
65 Ops.INDEX dtypes.i32 [58, 41]
66 Ops.LOAD dtypes.i32 [65]
67 Ops.BITCAST dtypes.u32 [66] dtypes.u32
68 Ops.CAST dtypes.u32 [63] dtypes.u32
69 Ops.SHR dtypes.u32 [67, 68]
70 Ops.BITCAST dtypes.u32 [61] dtypes.u32
71 Ops.SHL dtypes.u32 [70, 14]
72 Ops.CONST dtypes.weakint [] 4294967295
73 Ops.MUL dtypes.u32 [68, 72]
74 Ops.ADD dtypes.u32 [62, 73]
75 Ops.SHL dtypes.u32 [71, 74]
76 Ops.OR dtypes.u32 [69, 75]
77 Ops.BITCAST dtypes.i32 [76] dtypes.i32
78 Ops.WHERE dtypes.i32 [57, 64, 77]
79 Ops.SHR dtypes.i32 [61, 62]
80 Ops.WHERE dtypes.i32 [53, 78, 79]
81 Ops.STORE dtypes.void [42, 80]
82 Ops.INDEX dtypes.i32 [38, 59]
83 Ops.WHERE dtypes.i32 [57, 79, 64]
84 Ops.WHERE dtypes.i32 [53, 83, 79]
85 Ops.STORE dtypes.void [82, 84]
86 Ops.AFTER dtypes.void [0, 81, 85]
87 Ops.MUL dtypes.i32 [15, 3]
88 Ops.ADD dtypes.i32 [87, 40]
89 Ops.INDEX dtypes.i32 [38, 88]
90 Ops.ADD dtypes.i32 [87, 14]
91 Ops.INDEX dtypes.i32 [58, 90]
92 Ops.LOAD dtypes.i32 [91]
93 Ops.SHR dtypes.i32 [92, 63]
94 Ops.INDEX dtypes.i32 [58, 88]
95 Ops.LOAD dtypes.i32 [94]
96 Ops.BITCAST dtypes.u32 [95] dtypes.u32
97 Ops.SHR dtypes.u32 [96, 68]
98 Ops.BITCAST dtypes.u32 [92] dtypes.u32
99 Ops.SHL dtypes.u32 [98, 14]
100 Ops.SHL dtypes.u32 [99, 74]
101 Ops.OR dtypes.u32 [97, 100]
102 Ops.BITCAST dtypes.i32 [101] dtypes.i32
103 Ops.WHERE dtypes.i32 [57, 93, 102]
104 Ops.SHR dtypes.i32 [92, 62]
105 Ops.WHERE dtypes.i32 [53, 103, 104]
106 Ops.STORE dtypes.void [89, 105]
107 Ops.INDEX dtypes.i32 [38, 90]
108 Ops.WHERE dtypes.i32 [57, 104, 93]
109 Ops.WHERE dtypes.i32 [53, 108, 104]
110 Ops.STORE dtypes.void [107, 109]
111 Ops.AFTER dtypes.void [0, 106, 110]
112 Ops.MUL dtypes.i32 [21, 3]
113 Ops.ADD dtypes.i32 [112, 40]
114 Ops.INDEX dtypes.i32 [38, 113]
115 Ops.ADD dtypes.i32 [112, 14]
116 Ops.INDEX dtypes.i32 [58, 115]
117 Ops.LOAD dtypes.i32 [116]
118 Ops.SHR dtypes.i32 [117, 63]
119 Ops.INDEX dtypes.i32 [58, 113]
120 Ops.LOAD dtypes.i32 [119]
121 Ops.BITCAST dtypes.u32 [120] dtypes.u32
122 Ops.SHR dtypes.u32 [121, 68]
123 Ops.BITCAST dtypes.u32 [117] dtypes.u32
124 Ops.SHL dtypes.u32 [123, 14]
125 Ops.SHL dtypes.u32 [124, 74]
126 Ops.OR dtypes.u32 [122, 125]
127 Ops.BITCAST dtypes.i32 [126] dtypes.i32
128 Ops.WHERE dtypes.i32 [57, 118, 127]
129 Ops.SHR dtypes.i32 [117, 62]
130 Ops.WHERE dtypes.i32 [53, 128, 129]
131 Ops.STORE dtypes.void [114, 130]
132 Ops.INDEX dtypes.i32 [38, 115]
133 Ops.WHERE dtypes.i32 [57, 129, 118]
134 Ops.WHERE dtypes.i32 [53, 133, 129]
135 Ops.STORE dtypes.void [132, 134]
136 Ops.AFTER dtypes.void [0, 131, 135]
137 Ops.MUL dtypes.i32 [28, 3]
138 Ops.ADD dtypes.i32 [137, 40]
139 Ops.INDEX dtypes.i32 [38, 138]
140 Ops.ADD dtypes.i32 [137, 14]
141 Ops.INDEX dtypes.i32 [58, 140]
142 Ops.LOAD dtypes.i32 [141]
143 Ops.SHR dtypes.i32 [142, 63]
144 Ops.INDEX dtypes.i32 [58, 138]
145 Ops.LOAD dtypes.i32 [144]
146 Ops.BITCAST dtypes.u32 [145] dtypes.u32
147 Ops.SHR dtypes.u32 [146, 68]
148 Ops.BITCAST dtypes.u32 [142] dtypes.u32
149 Ops.SHL dtypes.u32 [148, 14]
150 Ops.SHL dtypes.u32 [149, 74]
151 Ops.OR dtypes.u32 [147, 150]
152 Ops.BITCAST dtypes.i32 [151] dtypes.i32
153 Ops.WHERE dtypes.i32 [57, 143, 152]
154 Ops.SHR dtypes.i32 [142, 62]
155 Ops.WHERE dtypes.i32 [53, 153, 154]
156 Ops.STORE dtypes.void [139, 155]
157 Ops.INDEX dtypes.i32 [38, 140]
158 Ops.WHERE dtypes.i32 [57, 154, 143]
159 Ops.WHERE dtypes.i32 [53, 158, 154]
160 Ops.STORE dtypes.void [157, 159]
161 Ops.AFTER dtypes.void [0, 156, 160]
162 Ops.AFTER dtypes.void [0, 86, 111, 136, 161]
163 Ops.END dtypes.void [162, 5]
164 Ops.SINK dtypes.void [163] KernelInfo(name="E_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
165 Ops.SINK dtypes.void [36, 164]

