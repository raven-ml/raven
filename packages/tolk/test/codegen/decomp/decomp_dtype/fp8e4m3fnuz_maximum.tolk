# tolk D27 golden=f82337b8b948096947b1f0b2a8d460b3
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 8
2 Ops.PARAM dtypes.fp8e4m3fnuz [1] ParamArg(slot=0, dtype=dtypes.fp8e4m3fnuz)
3 Ops.CONST dtypes.weakint [] 2
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.RANGE dtypes.i32 [4] (AxisType.WEAK, 0)
6 Ops.SHL dtypes.i32 [5, 3]
7 Ops.CONST dtypes.weakint [] 4
8 Ops.CAST dtypes.i32 [7] dtypes.i32
9 Ops.SHRINK dtypes.fp8e4m3fnuz [2, 6, 8]
10 Ops.PARAM dtypes.fp8e4m3fnuz [1] ParamArg(slot=1, dtype=dtypes.fp8e4m3fnuz)
11 Ops.SHRINK dtypes.fp8e4m3fnuz [10, 6, 8]
12 Ops.LOAD dtypes.fp8e4m3fnuz [11]
13 Ops.CONST dtypes.weakint [] 0
14 Ops.CAST dtypes.i32 [13] dtypes.i32
15 Ops.INDEX dtypes.fp8e4m3fnuz [12, 14]
16 Ops.PARAM dtypes.fp8e4m3fnuz [1] ParamArg(slot=2, dtype=dtypes.fp8e4m3fnuz)
17 Ops.SHRINK dtypes.fp8e4m3fnuz [16, 6, 8]
18 Ops.LOAD dtypes.fp8e4m3fnuz [17]
19 Ops.INDEX dtypes.fp8e4m3fnuz [18, 14]
20 Ops.CMPLT dtypes.bool [15, 19]
21 Ops.WHERE dtypes.fp8e4m3fnuz [20, 19, 15]
22 Ops.CONST dtypes.weakint [] 1
23 Ops.CAST dtypes.i32 [22] dtypes.i32
24 Ops.INDEX dtypes.fp8e4m3fnuz [12, 23]
25 Ops.INDEX dtypes.fp8e4m3fnuz [18, 23]
26 Ops.CMPLT dtypes.bool [24, 25]
27 Ops.WHERE dtypes.fp8e4m3fnuz [26, 25, 24]
28 Ops.INDEX dtypes.fp8e4m3fnuz [12, 4]
29 Ops.INDEX dtypes.fp8e4m3fnuz [18, 4]
30 Ops.CMPLT dtypes.bool [28, 29]
31 Ops.WHERE dtypes.fp8e4m3fnuz [30, 29, 28]
32 Ops.CONST dtypes.weakint [] 3
33 Ops.CAST dtypes.i32 [32] dtypes.i32
34 Ops.INDEX dtypes.fp8e4m3fnuz [12, 33]
35 Ops.INDEX dtypes.fp8e4m3fnuz [18, 33]
36 Ops.CMPLT dtypes.bool [34, 35]
37 Ops.WHERE dtypes.fp8e4m3fnuz [36, 35, 34]
38 Ops.STACK dtypes.fp8e4m3fnuz [21, 27, 31, 37]
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
54 Ops.CMPEQ dtypes.bool [53, 13]
55 Ops.CONST dtypes.bool [] False
56 Ops.CMPEQ dtypes.bool [54, 55]
57 Ops.CONST dtypes.weakint [] 127
58 Ops.AND dtypes.u8 [48, 57]
59 Ops.CAST dtypes.u32 [58] dtypes.u32
60 Ops.CMPEQ dtypes.bool [59, 13]
61 Ops.AND dtypes.bool [56, 60]
62 Ops.CONST dtypes.weakint [] 2143289344
63 Ops.DIV dtypes.u32 [59, 1]
64 Ops.CMPEQ dtypes.bool [63, 13]
65 Ops.CAST dtypes.f32 [59] dtypes.f32
66 Ops.BITCAST dtypes.u32 [65] dtypes.u32
67 Ops.CONST dtypes.weakint [] 83886080
68 Ops.CAST dtypes.i64 [67] dtypes.i64
69 Ops.CONST dtypes.weakint [] -1
70 Ops.CAST dtypes.i64 [69] dtypes.i64
71 Ops.MUL dtypes.i64 [68, 70]
72 Ops.CAST dtypes.u32 [71] dtypes.u32
73 Ops.ADD dtypes.u32 [66, 72]
74 Ops.WHERE dtypes.u32 [60, 13, 73]
75 Ops.CONST dtypes.weakint [] 1048576
76 Ops.MUL dtypes.u32 [59, 75]
77 Ops.CONST dtypes.weakint [] 998244352
78 Ops.ADD dtypes.u32 [76, 77]
79 Ops.WHERE dtypes.u32 [64, 74, 78]
80 Ops.OR dtypes.u32 [53, 79]
81 Ops.WHERE dtypes.u32 [61, 62, 80]
82 Ops.BITCAST dtypes.f32 [81] dtypes.f32
83 Ops.ADD dtypes.i32 [6, 22]
84 Ops.INDEX dtypes.u8 [45, 83]
85 Ops.LOAD dtypes.u8 [84]
86 Ops.AND dtypes.u8 [85, 49]
87 Ops.CAST dtypes.u32 [86] dtypes.u32
88 Ops.MUL dtypes.u32 [87, 52]
89 Ops.CMPEQ dtypes.bool [88, 13]
90 Ops.CMPEQ dtypes.bool [89, 55]
91 Ops.AND dtypes.u8 [85, 57]
92 Ops.CAST dtypes.u32 [91] dtypes.u32
93 Ops.CMPEQ dtypes.bool [92, 13]
94 Ops.AND dtypes.bool [90, 93]
95 Ops.DIV dtypes.u32 [92, 1]
96 Ops.CMPEQ dtypes.bool [95, 13]
97 Ops.CAST dtypes.f32 [92] dtypes.f32
98 Ops.BITCAST dtypes.u32 [97] dtypes.u32
99 Ops.ADD dtypes.u32 [98, 72]
100 Ops.WHERE dtypes.u32 [93, 13, 99]
101 Ops.MUL dtypes.u32 [92, 75]
102 Ops.ADD dtypes.u32 [101, 77]
103 Ops.WHERE dtypes.u32 [96, 100, 102]
104 Ops.OR dtypes.u32 [88, 103]
105 Ops.WHERE dtypes.u32 [94, 62, 104]
106 Ops.BITCAST dtypes.f32 [105] dtypes.f32
107 Ops.ADD dtypes.i32 [6, 3]
108 Ops.INDEX dtypes.u8 [45, 107]
109 Ops.LOAD dtypes.u8 [108]
110 Ops.AND dtypes.u8 [109, 49]
111 Ops.CAST dtypes.u32 [110] dtypes.u32
112 Ops.MUL dtypes.u32 [111, 52]
113 Ops.CMPEQ dtypes.bool [112, 13]
114 Ops.CMPEQ dtypes.bool [113, 55]
115 Ops.AND dtypes.u8 [109, 57]
116 Ops.CAST dtypes.u32 [115] dtypes.u32
117 Ops.CMPEQ dtypes.bool [116, 13]
118 Ops.AND dtypes.bool [114, 117]
119 Ops.DIV dtypes.u32 [116, 1]
120 Ops.CMPEQ dtypes.bool [119, 13]
121 Ops.CAST dtypes.f32 [116] dtypes.f32
122 Ops.BITCAST dtypes.u32 [121] dtypes.u32
123 Ops.ADD dtypes.u32 [122, 72]
124 Ops.WHERE dtypes.u32 [117, 13, 123]
125 Ops.MUL dtypes.u32 [116, 75]
126 Ops.ADD dtypes.u32 [125, 77]
127 Ops.WHERE dtypes.u32 [120, 124, 126]
128 Ops.OR dtypes.u32 [112, 127]
129 Ops.WHERE dtypes.u32 [118, 62, 128]
130 Ops.BITCAST dtypes.f32 [129] dtypes.f32
131 Ops.ADD dtypes.i32 [6, 32]
132 Ops.INDEX dtypes.u8 [45, 131]
133 Ops.LOAD dtypes.u8 [132]
134 Ops.AND dtypes.u8 [133, 49]
135 Ops.CAST dtypes.u32 [134] dtypes.u32
136 Ops.MUL dtypes.u32 [135, 52]
137 Ops.CMPEQ dtypes.bool [136, 13]
138 Ops.CMPEQ dtypes.bool [137, 55]
139 Ops.AND dtypes.u8 [133, 57]
140 Ops.CAST dtypes.u32 [139] dtypes.u32
141 Ops.CMPEQ dtypes.bool [140, 13]
142 Ops.AND dtypes.bool [138, 141]
143 Ops.DIV dtypes.u32 [140, 1]
144 Ops.CMPEQ dtypes.bool [143, 13]
145 Ops.CAST dtypes.f32 [140] dtypes.f32
146 Ops.BITCAST dtypes.u32 [145] dtypes.u32
147 Ops.ADD dtypes.u32 [146, 72]
148 Ops.WHERE dtypes.u32 [141, 13, 147]
149 Ops.MUL dtypes.u32 [140, 75]
150 Ops.ADD dtypes.u32 [149, 77]
151 Ops.WHERE dtypes.u32 [144, 148, 150]
152 Ops.OR dtypes.u32 [136, 151]
153 Ops.WHERE dtypes.u32 [142, 62, 152]
154 Ops.BITCAST dtypes.f32 [153] dtypes.f32
155 Ops.STACK dtypes.f32 [82, 106, 130, 154]
156 Ops.CAST dtypes.f32 [155] dtypes.f32
157 Ops.INDEX dtypes.f32 [156, 14]
158 Ops.PARAM dtypes.u8 [1] ParamArg(slot=2, dtype=dtypes.u8)
159 Ops.INDEX dtypes.u8 [158, 46]
160 Ops.LOAD dtypes.u8 [159]
161 Ops.AND dtypes.u8 [160, 49]
162 Ops.CAST dtypes.u32 [161] dtypes.u32
163 Ops.MUL dtypes.u32 [162, 52]
164 Ops.CMPEQ dtypes.bool [163, 13]
165 Ops.CMPEQ dtypes.bool [164, 55]
166 Ops.AND dtypes.u8 [160, 57]
167 Ops.CAST dtypes.u32 [166] dtypes.u32
168 Ops.CMPEQ dtypes.bool [167, 13]
169 Ops.AND dtypes.bool [165, 168]
170 Ops.DIV dtypes.u32 [167, 1]
171 Ops.CMPEQ dtypes.bool [170, 13]
172 Ops.CAST dtypes.f32 [167] dtypes.f32
173 Ops.BITCAST dtypes.u32 [172] dtypes.u32
174 Ops.ADD dtypes.u32 [173, 72]
175 Ops.WHERE dtypes.u32 [168, 13, 174]
176 Ops.MUL dtypes.u32 [167, 75]
177 Ops.ADD dtypes.u32 [176, 77]
178 Ops.WHERE dtypes.u32 [171, 175, 177]
179 Ops.OR dtypes.u32 [163, 178]
180 Ops.WHERE dtypes.u32 [169, 62, 179]
181 Ops.BITCAST dtypes.f32 [180] dtypes.f32
182 Ops.INDEX dtypes.u8 [158, 83]
183 Ops.LOAD dtypes.u8 [182]
184 Ops.AND dtypes.u8 [183, 49]
185 Ops.CAST dtypes.u32 [184] dtypes.u32
186 Ops.MUL dtypes.u32 [185, 52]
187 Ops.CMPEQ dtypes.bool [186, 13]
188 Ops.CMPEQ dtypes.bool [187, 55]
189 Ops.AND dtypes.u8 [183, 57]
190 Ops.CAST dtypes.u32 [189] dtypes.u32
191 Ops.CMPEQ dtypes.bool [190, 13]
192 Ops.AND dtypes.bool [188, 191]
193 Ops.DIV dtypes.u32 [190, 1]
194 Ops.CMPEQ dtypes.bool [193, 13]
195 Ops.CAST dtypes.f32 [190] dtypes.f32
196 Ops.BITCAST dtypes.u32 [195] dtypes.u32
197 Ops.ADD dtypes.u32 [196, 72]
198 Ops.WHERE dtypes.u32 [191, 13, 197]
199 Ops.MUL dtypes.u32 [190, 75]
200 Ops.ADD dtypes.u32 [199, 77]
201 Ops.WHERE dtypes.u32 [194, 198, 200]
202 Ops.OR dtypes.u32 [186, 201]
203 Ops.WHERE dtypes.u32 [192, 62, 202]
204 Ops.BITCAST dtypes.f32 [203] dtypes.f32
205 Ops.INDEX dtypes.u8 [158, 107]
206 Ops.LOAD dtypes.u8 [205]
207 Ops.AND dtypes.u8 [206, 49]
208 Ops.CAST dtypes.u32 [207] dtypes.u32
209 Ops.MUL dtypes.u32 [208, 52]
210 Ops.CMPEQ dtypes.bool [209, 13]
211 Ops.CMPEQ dtypes.bool [210, 55]
212 Ops.AND dtypes.u8 [206, 57]
213 Ops.CAST dtypes.u32 [212] dtypes.u32
214 Ops.CMPEQ dtypes.bool [213, 13]
215 Ops.AND dtypes.bool [211, 214]
216 Ops.DIV dtypes.u32 [213, 1]
217 Ops.CMPEQ dtypes.bool [216, 13]
218 Ops.CAST dtypes.f32 [213] dtypes.f32
219 Ops.BITCAST dtypes.u32 [218] dtypes.u32
220 Ops.ADD dtypes.u32 [219, 72]
221 Ops.WHERE dtypes.u32 [214, 13, 220]
222 Ops.MUL dtypes.u32 [213, 75]
223 Ops.ADD dtypes.u32 [222, 77]
224 Ops.WHERE dtypes.u32 [217, 221, 223]
225 Ops.OR dtypes.u32 [209, 224]
226 Ops.WHERE dtypes.u32 [215, 62, 225]
227 Ops.BITCAST dtypes.f32 [226] dtypes.f32
228 Ops.INDEX dtypes.u8 [158, 131]
229 Ops.LOAD dtypes.u8 [228]
230 Ops.AND dtypes.u8 [229, 49]
231 Ops.CAST dtypes.u32 [230] dtypes.u32
232 Ops.MUL dtypes.u32 [231, 52]
233 Ops.CMPEQ dtypes.bool [232, 13]
234 Ops.CMPEQ dtypes.bool [233, 55]
235 Ops.AND dtypes.u8 [229, 57]
236 Ops.CAST dtypes.u32 [235] dtypes.u32
237 Ops.CMPEQ dtypes.bool [236, 13]
238 Ops.AND dtypes.bool [234, 237]
239 Ops.DIV dtypes.u32 [236, 1]
240 Ops.CMPEQ dtypes.bool [239, 13]
241 Ops.CAST dtypes.f32 [236] dtypes.f32
242 Ops.BITCAST dtypes.u32 [241] dtypes.u32
243 Ops.ADD dtypes.u32 [242, 72]
244 Ops.WHERE dtypes.u32 [237, 13, 243]
245 Ops.MUL dtypes.u32 [236, 75]
246 Ops.ADD dtypes.u32 [245, 77]
247 Ops.WHERE dtypes.u32 [240, 244, 246]
248 Ops.OR dtypes.u32 [232, 247]
249 Ops.WHERE dtypes.u32 [238, 62, 248]
250 Ops.BITCAST dtypes.f32 [249] dtypes.f32
251 Ops.STACK dtypes.f32 [181, 204, 227, 250]
252 Ops.CAST dtypes.f32 [251] dtypes.f32
253 Ops.INDEX dtypes.f32 [252, 14]
254 Ops.CMPLT dtypes.bool [157, 253]
255 Ops.SHRINK dtypes.u8 [158, 6, 8]
256 Ops.LOAD dtypes.u8 [255]
257 Ops.INDEX dtypes.u8 [256, 14]
258 Ops.SHRINK dtypes.u8 [45, 6, 8]
259 Ops.LOAD dtypes.u8 [258]
260 Ops.INDEX dtypes.u8 [259, 14]
261 Ops.WHERE dtypes.u8 [254, 257, 260]
262 Ops.INDEX dtypes.f32 [156, 23]
263 Ops.INDEX dtypes.f32 [252, 23]
264 Ops.CMPLT dtypes.bool [262, 263]
265 Ops.INDEX dtypes.u8 [256, 23]
266 Ops.INDEX dtypes.u8 [259, 23]
267 Ops.WHERE dtypes.u8 [264, 265, 266]
268 Ops.INDEX dtypes.f32 [156, 4]
269 Ops.INDEX dtypes.f32 [252, 4]
270 Ops.CMPLT dtypes.bool [268, 269]
271 Ops.INDEX dtypes.u8 [256, 4]
272 Ops.INDEX dtypes.u8 [259, 4]
273 Ops.WHERE dtypes.u8 [270, 271, 272]
274 Ops.INDEX dtypes.f32 [156, 33]
275 Ops.INDEX dtypes.f32 [252, 33]
276 Ops.CMPLT dtypes.bool [274, 275]
277 Ops.INDEX dtypes.u8 [256, 33]
278 Ops.INDEX dtypes.u8 [259, 33]
279 Ops.WHERE dtypes.u8 [276, 277, 278]
280 Ops.STACK dtypes.u8 [261, 267, 273, 279]
281 Ops.STORE dtypes.void [44, 280]
282 Ops.AFTER dtypes.void [0, 281]
283 Ops.END dtypes.void [282, 5]
284 Ops.SINK dtypes.void [283] KernelInfo(name="E_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
285 Ops.SINK dtypes.void [42, 284]

