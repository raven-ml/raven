# tolk D27 golden=3887ed3580ef9f5d7eb84fae3f82d904
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 8
2 Ops.PARAM dtypes.fp8e4m3 [1] ParamArg(slot=0, dtype=dtypes.fp8e4m3)
3 Ops.CONST dtypes.weakint [] 2
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.RANGE dtypes.i32 [4] (AxisType.WEAK, 0)
6 Ops.SHL dtypes.i32 [5, 3]
7 Ops.CONST dtypes.weakint [] 4
8 Ops.CAST dtypes.i32 [7] dtypes.i32
9 Ops.SHRINK dtypes.fp8e4m3 [2, 6, 8]
10 Ops.PARAM dtypes.fp8e4m3 [1] ParamArg(slot=1, dtype=dtypes.fp8e4m3)
11 Ops.SHRINK dtypes.fp8e4m3 [10, 6, 8]
12 Ops.LOAD dtypes.fp8e4m3 [11]
13 Ops.CONST dtypes.weakint [] 0
14 Ops.CAST dtypes.i32 [13] dtypes.i32
15 Ops.INDEX dtypes.fp8e4m3 [12, 14]
16 Ops.PARAM dtypes.fp8e4m3 [1] ParamArg(slot=2, dtype=dtypes.fp8e4m3)
17 Ops.SHRINK dtypes.fp8e4m3 [16, 6, 8]
18 Ops.LOAD dtypes.fp8e4m3 [17]
19 Ops.INDEX dtypes.fp8e4m3 [18, 14]
20 Ops.CMPLT dtypes.bool [15, 19]
21 Ops.WHERE dtypes.fp8e4m3 [20, 19, 15]
22 Ops.CONST dtypes.weakint [] 1
23 Ops.CAST dtypes.i32 [22] dtypes.i32
24 Ops.INDEX dtypes.fp8e4m3 [12, 23]
25 Ops.INDEX dtypes.fp8e4m3 [18, 23]
26 Ops.CMPLT dtypes.bool [24, 25]
27 Ops.WHERE dtypes.fp8e4m3 [26, 25, 24]
28 Ops.INDEX dtypes.fp8e4m3 [12, 4]
29 Ops.INDEX dtypes.fp8e4m3 [18, 4]
30 Ops.CMPLT dtypes.bool [28, 29]
31 Ops.WHERE dtypes.fp8e4m3 [30, 29, 28]
32 Ops.CONST dtypes.weakint [] 3
33 Ops.CAST dtypes.i32 [32] dtypes.i32
34 Ops.INDEX dtypes.fp8e4m3 [12, 33]
35 Ops.INDEX dtypes.fp8e4m3 [18, 33]
36 Ops.CMPLT dtypes.bool [34, 35]
37 Ops.WHERE dtypes.fp8e4m3 [36, 35, 34]
38 Ops.STACK dtypes.fp8e4m3 [21, 27, 31, 37]
39 Ops.STORE dtypes.void [9, 38]
40 Ops.AFTER dtypes.void [0, 39]
41 Ops.END dtypes.void [40, 5]
42 Ops.SINK dtypes.void [41] KernelInfo(name="E_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
43 Ops.PARAM dtypes.u8 [1] ParamArg(slot=0, dtype=dtypes.u8)
44 Ops.SHRINK dtypes.u8 [43, 6, 8]
45 Ops.PARAM dtypes.u8 [1] ParamArg(slot=1, dtype=dtypes.u8)
46 Ops.ADD dtypes.i32 [6, 13]
47 Ops.INDEX dtypes.u8 [45, 46]
48 Ops.LOAD dtypes.u8 [47]
49 Ops.CONST dtypes.weakint [] 128
50 Ops.AND dtypes.u8 [48, 49]
51 Ops.CAST dtypes.u32 [50] dtypes.u32
52 Ops.CONST dtypes.weakint [] 16777216
53 Ops.MUL dtypes.u32 [51, 52]
54 Ops.CONST dtypes.weakint [] 127
55 Ops.AND dtypes.u8 [48, 54]
56 Ops.CAST dtypes.u32 [55] dtypes.u32
57 Ops.CMPEQ dtypes.bool [56, 54]
58 Ops.CONST dtypes.weakint [] 1048576
59 Ops.MUL dtypes.u32 [56, 58]
60 Ops.CONST dtypes.weakint [] 2139095040
61 Ops.OR dtypes.u32 [59, 60]
62 Ops.DIV dtypes.u32 [56, 1]
63 Ops.CMPEQ dtypes.bool [62, 13]
64 Ops.CMPEQ dtypes.bool [56, 13]
65 Ops.CAST dtypes.f32 [56] dtypes.f32
66 Ops.BITCAST dtypes.u32 [65] dtypes.u32
67 Ops.CONST dtypes.weakint [] 75497472
68 Ops.CAST dtypes.i64 [67] dtypes.i64
69 Ops.CONST dtypes.weakint [] -1
70 Ops.CAST dtypes.i64 [69] dtypes.i64
71 Ops.MUL dtypes.i64 [68, 70]
72 Ops.CAST dtypes.u32 [71] dtypes.u32
73 Ops.ADD dtypes.u32 [66, 72]
74 Ops.WHERE dtypes.u32 [64, 13, 73]
75 Ops.CONST dtypes.weakint [] 1006632960
76 Ops.ADD dtypes.u32 [59, 75]
77 Ops.WHERE dtypes.u32 [63, 74, 76]
78 Ops.WHERE dtypes.u32 [57, 61, 77]
79 Ops.OR dtypes.u32 [53, 78]
80 Ops.BITCAST dtypes.f32 [79] dtypes.f32
81 Ops.ADD dtypes.i32 [6, 22]
82 Ops.INDEX dtypes.u8 [45, 81]
83 Ops.LOAD dtypes.u8 [82]
84 Ops.AND dtypes.u8 [83, 49]
85 Ops.CAST dtypes.u32 [84] dtypes.u32
86 Ops.MUL dtypes.u32 [85, 52]
87 Ops.AND dtypes.u8 [83, 54]
88 Ops.CAST dtypes.u32 [87] dtypes.u32
89 Ops.CMPEQ dtypes.bool [88, 54]
90 Ops.MUL dtypes.u32 [88, 58]
91 Ops.OR dtypes.u32 [90, 60]
92 Ops.DIV dtypes.u32 [88, 1]
93 Ops.CMPEQ dtypes.bool [92, 13]
94 Ops.CMPEQ dtypes.bool [88, 13]
95 Ops.CAST dtypes.f32 [88] dtypes.f32
96 Ops.BITCAST dtypes.u32 [95] dtypes.u32
97 Ops.ADD dtypes.u32 [96, 72]
98 Ops.WHERE dtypes.u32 [94, 13, 97]
99 Ops.ADD dtypes.u32 [90, 75]
100 Ops.WHERE dtypes.u32 [93, 98, 99]
101 Ops.WHERE dtypes.u32 [89, 91, 100]
102 Ops.OR dtypes.u32 [86, 101]
103 Ops.BITCAST dtypes.f32 [102] dtypes.f32
104 Ops.ADD dtypes.i32 [6, 3]
105 Ops.INDEX dtypes.u8 [45, 104]
106 Ops.LOAD dtypes.u8 [105]
107 Ops.AND dtypes.u8 [106, 49]
108 Ops.CAST dtypes.u32 [107] dtypes.u32
109 Ops.MUL dtypes.u32 [108, 52]
110 Ops.AND dtypes.u8 [106, 54]
111 Ops.CAST dtypes.u32 [110] dtypes.u32
112 Ops.CMPEQ dtypes.bool [111, 54]
113 Ops.MUL dtypes.u32 [111, 58]
114 Ops.OR dtypes.u32 [113, 60]
115 Ops.DIV dtypes.u32 [111, 1]
116 Ops.CMPEQ dtypes.bool [115, 13]
117 Ops.CMPEQ dtypes.bool [111, 13]
118 Ops.CAST dtypes.f32 [111] dtypes.f32
119 Ops.BITCAST dtypes.u32 [118] dtypes.u32
120 Ops.ADD dtypes.u32 [119, 72]
121 Ops.WHERE dtypes.u32 [117, 13, 120]
122 Ops.ADD dtypes.u32 [113, 75]
123 Ops.WHERE dtypes.u32 [116, 121, 122]
124 Ops.WHERE dtypes.u32 [112, 114, 123]
125 Ops.OR dtypes.u32 [109, 124]
126 Ops.BITCAST dtypes.f32 [125] dtypes.f32
127 Ops.ADD dtypes.i32 [6, 32]
128 Ops.INDEX dtypes.u8 [45, 127]
129 Ops.LOAD dtypes.u8 [128]
130 Ops.AND dtypes.u8 [129, 49]
131 Ops.CAST dtypes.u32 [130] dtypes.u32
132 Ops.MUL dtypes.u32 [131, 52]
133 Ops.AND dtypes.u8 [129, 54]
134 Ops.CAST dtypes.u32 [133] dtypes.u32
135 Ops.CMPEQ dtypes.bool [134, 54]
136 Ops.MUL dtypes.u32 [134, 58]
137 Ops.OR dtypes.u32 [136, 60]
138 Ops.DIV dtypes.u32 [134, 1]
139 Ops.CMPEQ dtypes.bool [138, 13]
140 Ops.CMPEQ dtypes.bool [134, 13]
141 Ops.CAST dtypes.f32 [134] dtypes.f32
142 Ops.BITCAST dtypes.u32 [141] dtypes.u32
143 Ops.ADD dtypes.u32 [142, 72]
144 Ops.WHERE dtypes.u32 [140, 13, 143]
145 Ops.ADD dtypes.u32 [136, 75]
146 Ops.WHERE dtypes.u32 [139, 144, 145]
147 Ops.WHERE dtypes.u32 [135, 137, 146]
148 Ops.OR dtypes.u32 [132, 147]
149 Ops.BITCAST dtypes.f32 [148] dtypes.f32
150 Ops.STACK dtypes.f32 [80, 103, 126, 149]
151 Ops.CAST dtypes.f32 [150] dtypes.f32
152 Ops.INDEX dtypes.f32 [151, 14]
153 Ops.PARAM dtypes.u8 [1] ParamArg(slot=2, dtype=dtypes.u8)
154 Ops.INDEX dtypes.u8 [153, 46]
155 Ops.LOAD dtypes.u8 [154]
156 Ops.AND dtypes.u8 [155, 49]
157 Ops.CAST dtypes.u32 [156] dtypes.u32
158 Ops.MUL dtypes.u32 [157, 52]
159 Ops.AND dtypes.u8 [155, 54]
160 Ops.CAST dtypes.u32 [159] dtypes.u32
161 Ops.CMPEQ dtypes.bool [160, 54]
162 Ops.MUL dtypes.u32 [160, 58]
163 Ops.OR dtypes.u32 [162, 60]
164 Ops.DIV dtypes.u32 [160, 1]
165 Ops.CMPEQ dtypes.bool [164, 13]
166 Ops.CMPEQ dtypes.bool [160, 13]
167 Ops.CAST dtypes.f32 [160] dtypes.f32
168 Ops.BITCAST dtypes.u32 [167] dtypes.u32
169 Ops.ADD dtypes.u32 [168, 72]
170 Ops.WHERE dtypes.u32 [166, 13, 169]
171 Ops.ADD dtypes.u32 [162, 75]
172 Ops.WHERE dtypes.u32 [165, 170, 171]
173 Ops.WHERE dtypes.u32 [161, 163, 172]
174 Ops.OR dtypes.u32 [158, 173]
175 Ops.BITCAST dtypes.f32 [174] dtypes.f32
176 Ops.INDEX dtypes.u8 [153, 81]
177 Ops.LOAD dtypes.u8 [176]
178 Ops.AND dtypes.u8 [177, 49]
179 Ops.CAST dtypes.u32 [178] dtypes.u32
180 Ops.MUL dtypes.u32 [179, 52]
181 Ops.AND dtypes.u8 [177, 54]
182 Ops.CAST dtypes.u32 [181] dtypes.u32
183 Ops.CMPEQ dtypes.bool [182, 54]
184 Ops.MUL dtypes.u32 [182, 58]
185 Ops.OR dtypes.u32 [184, 60]
186 Ops.DIV dtypes.u32 [182, 1]
187 Ops.CMPEQ dtypes.bool [186, 13]
188 Ops.CMPEQ dtypes.bool [182, 13]
189 Ops.CAST dtypes.f32 [182] dtypes.f32
190 Ops.BITCAST dtypes.u32 [189] dtypes.u32
191 Ops.ADD dtypes.u32 [190, 72]
192 Ops.WHERE dtypes.u32 [188, 13, 191]
193 Ops.ADD dtypes.u32 [184, 75]
194 Ops.WHERE dtypes.u32 [187, 192, 193]
195 Ops.WHERE dtypes.u32 [183, 185, 194]
196 Ops.OR dtypes.u32 [180, 195]
197 Ops.BITCAST dtypes.f32 [196] dtypes.f32
198 Ops.INDEX dtypes.u8 [153, 104]
199 Ops.LOAD dtypes.u8 [198]
200 Ops.AND dtypes.u8 [199, 49]
201 Ops.CAST dtypes.u32 [200] dtypes.u32
202 Ops.MUL dtypes.u32 [201, 52]
203 Ops.AND dtypes.u8 [199, 54]
204 Ops.CAST dtypes.u32 [203] dtypes.u32
205 Ops.CMPEQ dtypes.bool [204, 54]
206 Ops.MUL dtypes.u32 [204, 58]
207 Ops.OR dtypes.u32 [206, 60]
208 Ops.DIV dtypes.u32 [204, 1]
209 Ops.CMPEQ dtypes.bool [208, 13]
210 Ops.CMPEQ dtypes.bool [204, 13]
211 Ops.CAST dtypes.f32 [204] dtypes.f32
212 Ops.BITCAST dtypes.u32 [211] dtypes.u32
213 Ops.ADD dtypes.u32 [212, 72]
214 Ops.WHERE dtypes.u32 [210, 13, 213]
215 Ops.ADD dtypes.u32 [206, 75]
216 Ops.WHERE dtypes.u32 [209, 214, 215]
217 Ops.WHERE dtypes.u32 [205, 207, 216]
218 Ops.OR dtypes.u32 [202, 217]
219 Ops.BITCAST dtypes.f32 [218] dtypes.f32
220 Ops.INDEX dtypes.u8 [153, 127]
221 Ops.LOAD dtypes.u8 [220]
222 Ops.AND dtypes.u8 [221, 49]
223 Ops.CAST dtypes.u32 [222] dtypes.u32
224 Ops.MUL dtypes.u32 [223, 52]
225 Ops.AND dtypes.u8 [221, 54]
226 Ops.CAST dtypes.u32 [225] dtypes.u32
227 Ops.CMPEQ dtypes.bool [226, 54]
228 Ops.MUL dtypes.u32 [226, 58]
229 Ops.OR dtypes.u32 [228, 60]
230 Ops.DIV dtypes.u32 [226, 1]
231 Ops.CMPEQ dtypes.bool [230, 13]
232 Ops.CMPEQ dtypes.bool [226, 13]
233 Ops.CAST dtypes.f32 [226] dtypes.f32
234 Ops.BITCAST dtypes.u32 [233] dtypes.u32
235 Ops.ADD dtypes.u32 [234, 72]
236 Ops.WHERE dtypes.u32 [232, 13, 235]
237 Ops.ADD dtypes.u32 [228, 75]
238 Ops.WHERE dtypes.u32 [231, 236, 237]
239 Ops.WHERE dtypes.u32 [227, 229, 238]
240 Ops.OR dtypes.u32 [224, 239]
241 Ops.BITCAST dtypes.f32 [240] dtypes.f32
242 Ops.STACK dtypes.f32 [175, 197, 219, 241]
243 Ops.CAST dtypes.f32 [242] dtypes.f32
244 Ops.INDEX dtypes.f32 [243, 14]
245 Ops.CMPLT dtypes.bool [152, 244]
246 Ops.SHRINK dtypes.u8 [153, 6, 8]
247 Ops.LOAD dtypes.u8 [246]
248 Ops.INDEX dtypes.u8 [247, 14]
249 Ops.SHRINK dtypes.u8 [45, 6, 8]
250 Ops.LOAD dtypes.u8 [249]
251 Ops.INDEX dtypes.u8 [250, 14]
252 Ops.WHERE dtypes.u8 [245, 248, 251]
253 Ops.INDEX dtypes.f32 [151, 23]
254 Ops.INDEX dtypes.f32 [243, 23]
255 Ops.CMPLT dtypes.bool [253, 254]
256 Ops.INDEX dtypes.u8 [247, 23]
257 Ops.INDEX dtypes.u8 [250, 23]
258 Ops.WHERE dtypes.u8 [255, 256, 257]
259 Ops.INDEX dtypes.f32 [151, 4]
260 Ops.INDEX dtypes.f32 [243, 4]
261 Ops.CMPLT dtypes.bool [259, 260]
262 Ops.INDEX dtypes.u8 [247, 4]
263 Ops.INDEX dtypes.u8 [250, 4]
264 Ops.WHERE dtypes.u8 [261, 262, 263]
265 Ops.INDEX dtypes.f32 [151, 33]
266 Ops.INDEX dtypes.f32 [243, 33]
267 Ops.CMPLT dtypes.bool [265, 266]
268 Ops.INDEX dtypes.u8 [247, 33]
269 Ops.INDEX dtypes.u8 [250, 33]
270 Ops.WHERE dtypes.u8 [267, 268, 269]
271 Ops.STACK dtypes.u8 [252, 258, 264, 270]
272 Ops.STORE dtypes.void [44, 271]
273 Ops.AFTER dtypes.void [0, 272]
274 Ops.END dtypes.void [273, 5]
275 Ops.SINK dtypes.void [274] KernelInfo(name="E_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
276 Ops.SINK dtypes.void [42, 275]

