# tolk D27 golden=27d2ee7eaed51ec11e61c32e04476bd6
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 8
2 Ops.PARAM dtypes.bf16 [1] ParamArg(slot=0, dtype=dtypes.bf16)
3 Ops.CONST dtypes.weakint [] 2
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.RANGE dtypes.i32 [4] (AxisType.WEAK, 0)
6 Ops.SHL dtypes.i32 [5, 3]
7 Ops.INDEX dtypes.bf16 [2, 6]
8 Ops.PARAM dtypes.bf16 [1] ParamArg(slot=1, dtype=dtypes.bf16)
9 Ops.INDEX dtypes.bf16 [8, 6]
10 Ops.LOAD dtypes.bf16 [9]
11 Ops.PARAM dtypes.bf16 [1] ParamArg(slot=2, dtype=dtypes.bf16)
12 Ops.INDEX dtypes.bf16 [11, 6]
13 Ops.LOAD dtypes.bf16 [12]
14 Ops.CMPLT dtypes.bool [10, 13]
15 Ops.WHERE dtypes.bf16 [14, 13, 10]
16 Ops.STORE dtypes.void [7, 15]
17 Ops.CONST dtypes.weakint [] 1
18 Ops.ADD dtypes.i32 [6, 17]
19 Ops.INDEX dtypes.bf16 [2, 18]
20 Ops.INDEX dtypes.bf16 [8, 18]
21 Ops.LOAD dtypes.bf16 [20]
22 Ops.INDEX dtypes.bf16 [11, 18]
23 Ops.LOAD dtypes.bf16 [22]
24 Ops.CMPLT dtypes.bool [21, 23]
25 Ops.WHERE dtypes.bf16 [24, 23, 21]
26 Ops.STORE dtypes.void [19, 25]
27 Ops.ADD dtypes.i32 [6, 3]
28 Ops.INDEX dtypes.bf16 [2, 27]
29 Ops.INDEX dtypes.bf16 [8, 27]
30 Ops.LOAD dtypes.bf16 [29]
31 Ops.INDEX dtypes.bf16 [11, 27]
32 Ops.LOAD dtypes.bf16 [31]
33 Ops.CMPLT dtypes.bool [30, 32]
34 Ops.WHERE dtypes.bf16 [33, 32, 30]
35 Ops.STORE dtypes.void [28, 34]
36 Ops.CONST dtypes.weakint [] 3
37 Ops.ADD dtypes.i32 [6, 36]
38 Ops.INDEX dtypes.bf16 [2, 37]
39 Ops.INDEX dtypes.bf16 [8, 37]
40 Ops.LOAD dtypes.bf16 [39]
41 Ops.INDEX dtypes.bf16 [11, 37]
42 Ops.LOAD dtypes.bf16 [41]
43 Ops.CMPLT dtypes.bool [40, 42]
44 Ops.WHERE dtypes.bf16 [43, 42, 40]
45 Ops.STORE dtypes.void [38, 44]
46 Ops.AFTER dtypes.void [0, 16, 26, 35, 45]
47 Ops.END dtypes.void [46, 5]
48 Ops.SINK dtypes.void [47] KernelInfo(name="E_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
49 Ops.PARAM dtypes.u16 [1] ParamArg(slot=0, dtype=dtypes.u16)
50 Ops.INDEX dtypes.u16 [49, 6]
51 Ops.PARAM dtypes.u16 [1] ParamArg(slot=1, dtype=dtypes.u16)
52 Ops.INDEX dtypes.u16 [51, 6]
53 Ops.LOAD dtypes.u16 [52]
54 Ops.CONST dtypes.weakint [] 32768
55 Ops.AND dtypes.u16 [53, 54]
56 Ops.CAST dtypes.u32 [55] dtypes.u32
57 Ops.CONST dtypes.weakint [] 65536
58 Ops.MUL dtypes.u32 [56, 57]
59 Ops.CONST dtypes.weakint [] 32767
60 Ops.AND dtypes.u16 [53, 59]
61 Ops.CAST dtypes.u32 [60] dtypes.u32
62 Ops.CONST dtypes.weakint [] 128
63 Ops.DIV dtypes.u32 [61, 62]
64 Ops.CONST dtypes.weakint [] 255
65 Ops.CMPEQ dtypes.bool [63, 64]
66 Ops.MUL dtypes.u32 [61, 57]
67 Ops.CONST dtypes.weakint [] 2139095040
68 Ops.OR dtypes.u32 [66, 67]
69 Ops.CONST dtypes.weakint [] 0
70 Ops.CMPEQ dtypes.bool [63, 69]
71 Ops.ADD dtypes.u32 [66, 69]
72 Ops.WHERE dtypes.u32 [70, 66, 71]
73 Ops.WHERE dtypes.u32 [65, 68, 72]
74 Ops.OR dtypes.u32 [58, 73]
75 Ops.BITCAST dtypes.f32 [74] dtypes.f32
76 Ops.PARAM dtypes.u16 [1] ParamArg(slot=2, dtype=dtypes.u16)
77 Ops.INDEX dtypes.u16 [76, 6]
78 Ops.LOAD dtypes.u16 [77]
79 Ops.AND dtypes.u16 [78, 54]
80 Ops.CAST dtypes.u32 [79] dtypes.u32
81 Ops.MUL dtypes.u32 [80, 57]
82 Ops.AND dtypes.u16 [78, 59]
83 Ops.CAST dtypes.u32 [82] dtypes.u32
84 Ops.DIV dtypes.u32 [83, 62]
85 Ops.CMPEQ dtypes.bool [84, 64]
86 Ops.MUL dtypes.u32 [83, 57]
87 Ops.OR dtypes.u32 [86, 67]
88 Ops.CMPEQ dtypes.bool [84, 69]
89 Ops.ADD dtypes.u32 [86, 69]
90 Ops.WHERE dtypes.u32 [88, 86, 89]
91 Ops.WHERE dtypes.u32 [85, 87, 90]
92 Ops.OR dtypes.u32 [81, 91]
93 Ops.BITCAST dtypes.f32 [92] dtypes.f32
94 Ops.CMPLT dtypes.bool [75, 93]
95 Ops.WHERE dtypes.u16 [94, 78, 53]
96 Ops.STORE dtypes.void [50, 95]
97 Ops.INDEX dtypes.u16 [49, 18]
98 Ops.INDEX dtypes.u16 [51, 18]
99 Ops.LOAD dtypes.u16 [98]
100 Ops.AND dtypes.u16 [99, 54]
101 Ops.CAST dtypes.u32 [100] dtypes.u32
102 Ops.MUL dtypes.u32 [101, 57]
103 Ops.AND dtypes.u16 [99, 59]
104 Ops.CAST dtypes.u32 [103] dtypes.u32
105 Ops.DIV dtypes.u32 [104, 62]
106 Ops.CMPEQ dtypes.bool [105, 64]
107 Ops.MUL dtypes.u32 [104, 57]
108 Ops.OR dtypes.u32 [107, 67]
109 Ops.CMPEQ dtypes.bool [105, 69]
110 Ops.ADD dtypes.u32 [107, 69]
111 Ops.WHERE dtypes.u32 [109, 107, 110]
112 Ops.WHERE dtypes.u32 [106, 108, 111]
113 Ops.OR dtypes.u32 [102, 112]
114 Ops.BITCAST dtypes.f32 [113] dtypes.f32
115 Ops.INDEX dtypes.u16 [76, 18]
116 Ops.LOAD dtypes.u16 [115]
117 Ops.AND dtypes.u16 [116, 54]
118 Ops.CAST dtypes.u32 [117] dtypes.u32
119 Ops.MUL dtypes.u32 [118, 57]
120 Ops.AND dtypes.u16 [116, 59]
121 Ops.CAST dtypes.u32 [120] dtypes.u32
122 Ops.DIV dtypes.u32 [121, 62]
123 Ops.CMPEQ dtypes.bool [122, 64]
124 Ops.MUL dtypes.u32 [121, 57]
125 Ops.OR dtypes.u32 [124, 67]
126 Ops.CMPEQ dtypes.bool [122, 69]
127 Ops.ADD dtypes.u32 [124, 69]
128 Ops.WHERE dtypes.u32 [126, 124, 127]
129 Ops.WHERE dtypes.u32 [123, 125, 128]
130 Ops.OR dtypes.u32 [119, 129]
131 Ops.BITCAST dtypes.f32 [130] dtypes.f32
132 Ops.CMPLT dtypes.bool [114, 131]
133 Ops.WHERE dtypes.u16 [132, 116, 99]
134 Ops.STORE dtypes.void [97, 133]
135 Ops.INDEX dtypes.u16 [49, 27]
136 Ops.INDEX dtypes.u16 [51, 27]
137 Ops.LOAD dtypes.u16 [136]
138 Ops.AND dtypes.u16 [137, 54]
139 Ops.CAST dtypes.u32 [138] dtypes.u32
140 Ops.MUL dtypes.u32 [139, 57]
141 Ops.AND dtypes.u16 [137, 59]
142 Ops.CAST dtypes.u32 [141] dtypes.u32
143 Ops.DIV dtypes.u32 [142, 62]
144 Ops.CMPEQ dtypes.bool [143, 64]
145 Ops.MUL dtypes.u32 [142, 57]
146 Ops.OR dtypes.u32 [145, 67]
147 Ops.CMPEQ dtypes.bool [143, 69]
148 Ops.ADD dtypes.u32 [145, 69]
149 Ops.WHERE dtypes.u32 [147, 145, 148]
150 Ops.WHERE dtypes.u32 [144, 146, 149]
151 Ops.OR dtypes.u32 [140, 150]
152 Ops.BITCAST dtypes.f32 [151] dtypes.f32
153 Ops.INDEX dtypes.u16 [76, 27]
154 Ops.LOAD dtypes.u16 [153]
155 Ops.AND dtypes.u16 [154, 54]
156 Ops.CAST dtypes.u32 [155] dtypes.u32
157 Ops.MUL dtypes.u32 [156, 57]
158 Ops.AND dtypes.u16 [154, 59]
159 Ops.CAST dtypes.u32 [158] dtypes.u32
160 Ops.DIV dtypes.u32 [159, 62]
161 Ops.CMPEQ dtypes.bool [160, 64]
162 Ops.MUL dtypes.u32 [159, 57]
163 Ops.OR dtypes.u32 [162, 67]
164 Ops.CMPEQ dtypes.bool [160, 69]
165 Ops.ADD dtypes.u32 [162, 69]
166 Ops.WHERE dtypes.u32 [164, 162, 165]
167 Ops.WHERE dtypes.u32 [161, 163, 166]
168 Ops.OR dtypes.u32 [157, 167]
169 Ops.BITCAST dtypes.f32 [168] dtypes.f32
170 Ops.CMPLT dtypes.bool [152, 169]
171 Ops.WHERE dtypes.u16 [170, 154, 137]
172 Ops.STORE dtypes.void [135, 171]
173 Ops.INDEX dtypes.u16 [49, 37]
174 Ops.INDEX dtypes.u16 [51, 37]
175 Ops.LOAD dtypes.u16 [174]
176 Ops.AND dtypes.u16 [175, 54]
177 Ops.CAST dtypes.u32 [176] dtypes.u32
178 Ops.MUL dtypes.u32 [177, 57]
179 Ops.AND dtypes.u16 [175, 59]
180 Ops.CAST dtypes.u32 [179] dtypes.u32
181 Ops.DIV dtypes.u32 [180, 62]
182 Ops.CMPEQ dtypes.bool [181, 64]
183 Ops.MUL dtypes.u32 [180, 57]
184 Ops.OR dtypes.u32 [183, 67]
185 Ops.CMPEQ dtypes.bool [181, 69]
186 Ops.ADD dtypes.u32 [183, 69]
187 Ops.WHERE dtypes.u32 [185, 183, 186]
188 Ops.WHERE dtypes.u32 [182, 184, 187]
189 Ops.OR dtypes.u32 [178, 188]
190 Ops.BITCAST dtypes.f32 [189] dtypes.f32
191 Ops.INDEX dtypes.u16 [76, 37]
192 Ops.LOAD dtypes.u16 [191]
193 Ops.AND dtypes.u16 [192, 54]
194 Ops.CAST dtypes.u32 [193] dtypes.u32
195 Ops.MUL dtypes.u32 [194, 57]
196 Ops.AND dtypes.u16 [192, 59]
197 Ops.CAST dtypes.u32 [196] dtypes.u32
198 Ops.DIV dtypes.u32 [197, 62]
199 Ops.CMPEQ dtypes.bool [198, 64]
200 Ops.MUL dtypes.u32 [197, 57]
201 Ops.OR dtypes.u32 [200, 67]
202 Ops.CMPEQ dtypes.bool [198, 69]
203 Ops.ADD dtypes.u32 [200, 69]
204 Ops.WHERE dtypes.u32 [202, 200, 203]
205 Ops.WHERE dtypes.u32 [199, 201, 204]
206 Ops.OR dtypes.u32 [195, 205]
207 Ops.BITCAST dtypes.f32 [206] dtypes.f32
208 Ops.CMPLT dtypes.bool [190, 207]
209 Ops.WHERE dtypes.u16 [208, 192, 175]
210 Ops.STORE dtypes.void [173, 209]
211 Ops.AFTER dtypes.void [0, 96, 134, 172, 210]
212 Ops.END dtypes.void [211, 5]
213 Ops.SINK dtypes.void [212] KernelInfo(name="E_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
214 Ops.SINK dtypes.void [48, 213]

