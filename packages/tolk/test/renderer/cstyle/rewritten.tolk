# tolk D9 D413 D430 golden=96b396bd38ca09c804d3782aa2f5cac7
0 Ops.CONST dtypes.weakint [] 16
1 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=0, dtype=dtypes.bf16)
2 Ops.RANGE dtypes.weakint [0] (AxisType.LOOP, 0)
3 Ops.INDEX dtypes.bf16 [1, 2]
4 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=1, dtype=dtypes.bf16)
5 Ops.INDEX dtypes.bf16 [4, 2]
6 Ops.LOAD dtypes.bf16 [5]
7 Ops.BITCAST dtypes.u16 [6] dtypes.u16
8 Ops.CAST dtypes.u32 [7] dtypes.u32
9 Ops.SHL dtypes.u32 [8, 0]
10 Ops.BITCAST dtypes.f32 [9] dtypes.f32
11 Ops.CONST dtypes.weakfloat [] 1.0
12 Ops.CAST dtypes.bf16 [11] dtypes.bf16
13 Ops.BITCAST dtypes.u16 [12] dtypes.u16
14 Ops.CAST dtypes.u32 [13] dtypes.u32
15 Ops.SHL dtypes.u32 [14, 0]
16 Ops.BITCAST dtypes.f32 [15] dtypes.f32
17 Ops.ADD dtypes.f32 [10, 16]
18 Ops.CMPEQ dtypes.bool [17, 17]
19 Ops.CONST dtypes.bool [] False
20 Ops.CMPEQ dtypes.bool [18, 19]
21 Ops.BITCAST dtypes.u32 [17] dtypes.u32
22 Ops.CONST dtypes.weakint [] 4194304
23 Ops.OR dtypes.u32 [21, 22]
24 Ops.SHR dtypes.u32 [21, 0]
25 Ops.CONST dtypes.weakint [] 1
26 Ops.AND dtypes.u32 [24, 25]
27 Ops.ADD dtypes.u32 [21, 26]
28 Ops.CONST dtypes.weakint [] 32767
29 Ops.ADD dtypes.u32 [27, 28]
30 Ops.WHERE dtypes.u32 [20, 23, 29]
31 Ops.SHR dtypes.u32 [30, 0]
32 Ops.CAST dtypes.u16 [31] dtypes.u16
33 Ops.BITCAST dtypes.bf16 [32] dtypes.bf16
34 Ops.STORE dtypes.void [3, 33]
35 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=2, dtype=dtypes.bf16)
36 Ops.INDEX dtypes.bf16 [35, 2]
37 Ops.LOAD dtypes.bf16 [36]
38 Ops.BITCAST dtypes.u16 [37] dtypes.u16
39 Ops.CAST dtypes.u32 [38] dtypes.u32
40 Ops.SHL dtypes.u32 [39, 0]
41 Ops.BITCAST dtypes.f32 [40] dtypes.f32
42 Ops.MUL dtypes.f32 [10, 41]
43 Ops.CMPEQ dtypes.bool [42, 42]
44 Ops.CMPEQ dtypes.bool [43, 19]
45 Ops.BITCAST dtypes.u32 [42] dtypes.u32
46 Ops.OR dtypes.u32 [45, 22]
47 Ops.SHR dtypes.u32 [45, 0]
48 Ops.AND dtypes.u32 [47, 25]
49 Ops.ADD dtypes.u32 [45, 48]
50 Ops.ADD dtypes.u32 [49, 28]
51 Ops.WHERE dtypes.u32 [44, 46, 50]
52 Ops.SHR dtypes.u32 [51, 0]
53 Ops.CAST dtypes.u16 [52] dtypes.u16
54 Ops.BITCAST dtypes.bf16 [53] dtypes.bf16
55 Ops.STORE dtypes.void [3, 54]
56 Ops.PARAM dtypes.bool [0] ParamArg(slot=0, dtype=dtypes.bool)
57 Ops.INDEX dtypes.bool [56, 2]
58 Ops.CMPLT dtypes.bool [10, 41]
59 Ops.STORE dtypes.void [57, 58]
60 Ops.PARAM dtypes.f32 [0] ParamArg(slot=3, dtype=dtypes.f32)
61 Ops.INDEX dtypes.f32 [60, 2]
62 Ops.LOAD dtypes.f32 [61]
63 Ops.CONST dtypes.weakfloat [] 0.0
64 Ops.CMPLT dtypes.bool [62, 63]
65 Ops.WHERE dtypes.bf16 [64, 6, 37]
66 Ops.STORE dtypes.void [3, 65]
67 Ops.SQRT dtypes.f32 [10]
68 Ops.CMPEQ dtypes.bool [67, 67]
69 Ops.CMPEQ dtypes.bool [68, 19]
70 Ops.BITCAST dtypes.u32 [67] dtypes.u32
71 Ops.OR dtypes.u32 [70, 22]
72 Ops.SHR dtypes.u32 [70, 0]
73 Ops.AND dtypes.u32 [72, 25]
74 Ops.ADD dtypes.u32 [70, 73]
75 Ops.ADD dtypes.u32 [74, 28]
76 Ops.WHERE dtypes.u32 [69, 71, 75]
77 Ops.SHR dtypes.u32 [76, 0]
78 Ops.CAST dtypes.u16 [77] dtypes.u16
79 Ops.BITCAST dtypes.bf16 [78] dtypes.bf16
80 Ops.STORE dtypes.void [3, 79]
81 Ops.EXP2 dtypes.f32 [10]
82 Ops.CMPEQ dtypes.bool [81, 81]
83 Ops.CMPEQ dtypes.bool [82, 19]
84 Ops.BITCAST dtypes.u32 [81] dtypes.u32
85 Ops.OR dtypes.u32 [84, 22]
86 Ops.SHR dtypes.u32 [84, 0]
87 Ops.AND dtypes.u32 [86, 25]
88 Ops.ADD dtypes.u32 [84, 87]
89 Ops.ADD dtypes.u32 [88, 28]
90 Ops.WHERE dtypes.u32 [83, 85, 89]
91 Ops.SHR dtypes.u32 [90, 0]
92 Ops.CAST dtypes.u16 [91] dtypes.u16
93 Ops.BITCAST dtypes.bf16 [92] dtypes.bf16
94 Ops.STORE dtypes.void [3, 93]
95 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
96 Ops.INDEX dtypes.f32 [95, 2]
97 Ops.STORE dtypes.void [96, 10]
98 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
99 Ops.INDEX dtypes.f32 [98, 2]
100 Ops.LOAD dtypes.f32 [99]
101 Ops.CMPEQ dtypes.bool [100, 100]
102 Ops.CMPEQ dtypes.bool [101, 19]
103 Ops.BITCAST dtypes.u32 [100] dtypes.u32
104 Ops.OR dtypes.u32 [103, 22]
105 Ops.SHR dtypes.u32 [103, 0]
106 Ops.AND dtypes.u32 [105, 25]
107 Ops.ADD dtypes.u32 [103, 106]
108 Ops.ADD dtypes.u32 [107, 28]
109 Ops.WHERE dtypes.u32 [102, 104, 108]
110 Ops.SHR dtypes.u32 [109, 0]
111 Ops.CAST dtypes.u16 [110] dtypes.u16
112 Ops.BITCAST dtypes.bf16 [111] dtypes.bf16
113 Ops.STORE dtypes.void [3, 112]
114 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
115 Ops.INDEX dtypes.f16 [114, 2]
116 Ops.LOAD dtypes.f16 [115]
117 Ops.CAST dtypes.f32 [116] dtypes.f32
118 Ops.CMPEQ dtypes.bool [117, 117]
119 Ops.CMPEQ dtypes.bool [118, 19]
120 Ops.BITCAST dtypes.u32 [117] dtypes.u32
121 Ops.OR dtypes.u32 [120, 22]
122 Ops.SHR dtypes.u32 [120, 0]
123 Ops.AND dtypes.u32 [122, 25]
124 Ops.ADD dtypes.u32 [120, 123]
125 Ops.ADD dtypes.u32 [124, 28]
126 Ops.WHERE dtypes.u32 [119, 121, 125]
127 Ops.SHR dtypes.u32 [126, 0]
128 Ops.CAST dtypes.u16 [127] dtypes.u16
129 Ops.BITCAST dtypes.bf16 [128] dtypes.bf16
130 Ops.STORE dtypes.void [3, 129]
131 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
132 Ops.INDEX dtypes.i32 [131, 2]
133 Ops.CAST dtypes.i32 [10] dtypes.i32
134 Ops.STORE dtypes.void [132, 133]
135 Ops.PARAM dtypes.f16 [0] ParamArg(slot=0, dtype=dtypes.f16)
136 Ops.INDEX dtypes.f16 [135, 2]
137 Ops.PARAM dtypes.f64 [0] ParamArg(slot=1, dtype=dtypes.f64)
138 Ops.INDEX dtypes.f64 [137, 2]
139 Ops.LOAD dtypes.f64 [138]
140 Ops.CAST dtypes.f32 [139] dtypes.f32
141 Ops.CAST dtypes.f64 [140] dtypes.f64
142 Ops.CMPLT dtypes.bool [139, 141]
143 Ops.CAST dtypes.f64 [63] dtypes.f64
144 Ops.CMPLT dtypes.bool [143, 139]
145 Ops.AND dtypes.bool [142, 144]
146 Ops.CMPLT dtypes.bool [141, 139]
147 Ops.CMPLT dtypes.bool [139, 143]
148 Ops.AND dtypes.bool [146, 147]
149 Ops.OR dtypes.bool [145, 148]
150 Ops.BITCAST dtypes.u32 [140] dtypes.u32
151 Ops.CONST dtypes.weakint [] -1
152 Ops.MUL dtypes.weakint [25, 151]
153 Ops.ADD dtypes.u32 [150, 152]
154 Ops.WHERE dtypes.u32 [149, 153, 150]
155 Ops.CMPEQ dtypes.bool [141, 139]
156 Ops.CMPEQ dtypes.bool [155, 19]
157 Ops.CAST dtypes.u32 [156] dtypes.u32
158 Ops.OR dtypes.u32 [154, 157]
159 Ops.BITCAST dtypes.f32 [158] dtypes.f32
160 Ops.CAST dtypes.f16 [159] dtypes.f16
161 Ops.STORE dtypes.void [136, 160]
162 Ops.STORE dtypes.void [96, 140]
163 Ops.PARAM dtypes.f16 [0] ParamArg(slot=2, dtype=dtypes.f16)
164 Ops.INDEX dtypes.f16 [163, 2]
165 Ops.LOAD dtypes.f16 [164]
166 Ops.ADD dtypes.f16 [116, 165]
167 Ops.STORE dtypes.void [136, 166]
168 Ops.PARAM dtypes.fp8e4m3 [0] ParamArg(slot=0, dtype=dtypes.fp8e4m3)
169 Ops.INDEX dtypes.fp8e4m3 [168, 2]
170 Ops.PARAM dtypes.fp8e4m3 [0] ParamArg(slot=1, dtype=dtypes.fp8e4m3)
171 Ops.INDEX dtypes.fp8e4m3 [170, 2]
172 Ops.LOAD dtypes.fp8e4m3 [171]
173 Ops.ADD dtypes.fp8e4m3 [172, 11]
174 Ops.STORE dtypes.void [169, 173]
175 Ops.PARAM dtypes.fp8e4m3 [0] ParamArg(slot=2, dtype=dtypes.fp8e4m3)
176 Ops.INDEX dtypes.fp8e4m3 [175, 2]
177 Ops.LOAD dtypes.fp8e4m3 [176]
178 Ops.CMPLT dtypes.bool [172, 177]
179 Ops.STORE dtypes.void [57, 178]
180 Ops.PARAM dtypes.fp8e5m2 [0] ParamArg(slot=0, dtype=dtypes.fp8e5m2)
181 Ops.INDEX dtypes.fp8e5m2 [180, 2]
182 Ops.CAST dtypes.fp8e5m2 [172] dtypes.fp8e5m2
183 Ops.STORE dtypes.void [181, 182]
184 Ops.CAST dtypes.fp8e4m3 [172] dtypes.fp8e4m3
185 Ops.STORE dtypes.void [169, 184]
186 Ops.PARAM dtypes.fp8e4m3fnuz [0] ParamArg(slot=0, dtype=dtypes.fp8e4m3fnuz)
187 Ops.INDEX dtypes.fp8e4m3fnuz [186, 2]
188 Ops.CAST dtypes.fp8e4m3fnuz [172] dtypes.fp8e4m3fnuz
189 Ops.STORE dtypes.void [187, 188]
190 Ops.CAST dtypes.fp8e4m3 [100] dtypes.fp8e4m3
191 Ops.STORE dtypes.void [169, 190]
192 Ops.CAST dtypes.fp8e5m2 [116] dtypes.fp8e5m2
193 Ops.STORE dtypes.void [181, 192]
194 Ops.PARAM dtypes.fp8e5m2 [0] ParamArg(slot=1, dtype=dtypes.fp8e5m2)
195 Ops.INDEX dtypes.fp8e5m2 [194, 2]
196 Ops.LOAD dtypes.fp8e5m2 [195]
197 Ops.CAST dtypes.i32 [196] dtypes.i32
198 Ops.STORE dtypes.void [132, 197]
199 Ops.PARAM dtypes.fp8e4m3fnuz [0] ParamArg(slot=1, dtype=dtypes.fp8e4m3fnuz)
200 Ops.INDEX dtypes.fp8e4m3fnuz [199, 2]
201 Ops.LOAD dtypes.fp8e4m3fnuz [200]
202 Ops.PARAM dtypes.fp8e4m3fnuz [0] ParamArg(slot=2, dtype=dtypes.fp8e4m3fnuz)
203 Ops.INDEX dtypes.fp8e4m3fnuz [202, 2]
204 Ops.LOAD dtypes.fp8e4m3fnuz [203]
205 Ops.MUL dtypes.fp8e4m3fnuz [201, 204]
206 Ops.STORE dtypes.void [187, 205]
207 Ops.CONST dtypes.weakint [] 0
208 Ops.CAST dtypes.fp8e4m3 [207] dtypes.fp8e4m3
209 Ops.ADD dtypes.fp8e4m3 [172, 208]
210 Ops.CAST dtypes.fp8e4m3 [25] dtypes.fp8e4m3
211 Ops.ADD dtypes.fp8e4m3 [172, 210]
212 Ops.CONST dtypes.weakint [] 2
213 Ops.CAST dtypes.fp8e4m3 [212] dtypes.fp8e4m3
214 Ops.ADD dtypes.fp8e4m3 [172, 213]
215 Ops.CONST dtypes.weakint [] 3
216 Ops.CAST dtypes.fp8e4m3 [215] dtypes.fp8e4m3
217 Ops.ADD dtypes.fp8e4m3 [172, 216]
218 Ops.CONST dtypes.weakint [] 4
219 Ops.CAST dtypes.fp8e4m3 [218] dtypes.fp8e4m3
220 Ops.ADD dtypes.fp8e4m3 [172, 219]
221 Ops.CONST dtypes.weakint [] 5
222 Ops.CAST dtypes.fp8e4m3 [221] dtypes.fp8e4m3
223 Ops.ADD dtypes.fp8e4m3 [172, 222]
224 Ops.CONST dtypes.weakint [] 6
225 Ops.CAST dtypes.fp8e4m3 [224] dtypes.fp8e4m3
226 Ops.ADD dtypes.fp8e4m3 [172, 225]
227 Ops.CONST dtypes.weakint [] 7
228 Ops.CAST dtypes.fp8e4m3 [227] dtypes.fp8e4m3
229 Ops.ADD dtypes.fp8e4m3 [172, 228]
230 Ops.STACK dtypes.fp8e4m3 [209, 211, 214, 217, 220, 223, 226, 229]
231 Ops.ADD dtypes.fp8e4m3 [177, 208]
232 Ops.ADD dtypes.fp8e4m3 [177, 210]
233 Ops.ADD dtypes.fp8e4m3 [177, 213]
234 Ops.ADD dtypes.fp8e4m3 [177, 216]
235 Ops.ADD dtypes.fp8e4m3 [177, 219]
236 Ops.ADD dtypes.fp8e4m3 [177, 222]
237 Ops.ADD dtypes.fp8e4m3 [177, 225]
238 Ops.ADD dtypes.fp8e4m3 [177, 228]
239 Ops.STACK dtypes.fp8e4m3 [231, 232, 233, 234, 235, 236, 237, 238]
240 Ops.STACK dtypes.f32 [62, 62, 62, 62]
241 Ops.WMMA dtypes.f32 [230, 239, 240] ((16, 16, 32), dtypes.fp8e4m3, 64, None)
242 Ops.PARAM dtypes.fp8e5m2fnuz [0] ParamArg(slot=1, dtype=dtypes.fp8e5m2fnuz)
243 Ops.INDEX dtypes.fp8e5m2fnuz [242, 2]
244 Ops.LOAD dtypes.fp8e5m2fnuz [243]
245 Ops.CAST dtypes.fp8e5m2fnuz [207] dtypes.fp8e5m2fnuz
246 Ops.ADD dtypes.fp8e5m2fnuz [244, 245]
247 Ops.CAST dtypes.fp8e5m2fnuz [25] dtypes.fp8e5m2fnuz
248 Ops.ADD dtypes.fp8e5m2fnuz [244, 247]
249 Ops.CAST dtypes.fp8e5m2fnuz [212] dtypes.fp8e5m2fnuz
250 Ops.ADD dtypes.fp8e5m2fnuz [244, 249]
251 Ops.CAST dtypes.fp8e5m2fnuz [215] dtypes.fp8e5m2fnuz
252 Ops.ADD dtypes.fp8e5m2fnuz [244, 251]
253 Ops.CAST dtypes.fp8e5m2fnuz [218] dtypes.fp8e5m2fnuz
254 Ops.ADD dtypes.fp8e5m2fnuz [244, 253]
255 Ops.CAST dtypes.fp8e5m2fnuz [221] dtypes.fp8e5m2fnuz
256 Ops.ADD dtypes.fp8e5m2fnuz [244, 255]
257 Ops.CAST dtypes.fp8e5m2fnuz [224] dtypes.fp8e5m2fnuz
258 Ops.ADD dtypes.fp8e5m2fnuz [244, 257]
259 Ops.CAST dtypes.fp8e5m2fnuz [227] dtypes.fp8e5m2fnuz
260 Ops.ADD dtypes.fp8e5m2fnuz [244, 259]
261 Ops.STACK dtypes.fp8e5m2fnuz [246, 248, 250, 252, 254, 256, 258, 260]
262 Ops.PARAM dtypes.fp8e5m2fnuz [0] ParamArg(slot=2, dtype=dtypes.fp8e5m2fnuz)
263 Ops.INDEX dtypes.fp8e5m2fnuz [262, 2]
264 Ops.LOAD dtypes.fp8e5m2fnuz [263]
265 Ops.ADD dtypes.fp8e5m2fnuz [264, 245]
266 Ops.ADD dtypes.fp8e5m2fnuz [264, 247]
267 Ops.ADD dtypes.fp8e5m2fnuz [264, 249]
268 Ops.ADD dtypes.fp8e5m2fnuz [264, 251]
269 Ops.ADD dtypes.fp8e5m2fnuz [264, 253]
270 Ops.ADD dtypes.fp8e5m2fnuz [264, 255]
271 Ops.ADD dtypes.fp8e5m2fnuz [264, 257]
272 Ops.ADD dtypes.fp8e5m2fnuz [264, 259]
273 Ops.STACK dtypes.fp8e5m2fnuz [265, 266, 267, 268, 269, 270, 271, 272]
274 Ops.WMMA dtypes.f32 [261, 273, 240] ((16, 16, 32), dtypes.fp8e5m2fnuz, 64, None)
275 Ops.CAST dtypes.f16 [207] dtypes.f16
276 Ops.ADD dtypes.f16 [116, 275]
277 Ops.CAST dtypes.f16 [25] dtypes.f16
278 Ops.ADD dtypes.f16 [116, 277]
279 Ops.CAST dtypes.f16 [212] dtypes.f16
280 Ops.ADD dtypes.f16 [116, 279]
281 Ops.CAST dtypes.f16 [215] dtypes.f16
282 Ops.ADD dtypes.f16 [116, 281]
283 Ops.CAST dtypes.f16 [218] dtypes.f16
284 Ops.ADD dtypes.f16 [116, 283]
285 Ops.CAST dtypes.f16 [221] dtypes.f16
286 Ops.ADD dtypes.f16 [116, 285]
287 Ops.CAST dtypes.f16 [224] dtypes.f16
288 Ops.ADD dtypes.f16 [116, 287]
289 Ops.CAST dtypes.f16 [227] dtypes.f16
290 Ops.ADD dtypes.f16 [116, 289]
291 Ops.STACK dtypes.f16 [276, 278, 280, 282, 284, 286, 288, 290]
292 Ops.ADD dtypes.f16 [165, 275]
293 Ops.ADD dtypes.f16 [165, 277]
294 Ops.ADD dtypes.f16 [165, 279]
295 Ops.ADD dtypes.f16 [165, 281]
296 Ops.ADD dtypes.f16 [165, 283]
297 Ops.ADD dtypes.f16 [165, 285]
298 Ops.ADD dtypes.f16 [165, 287]
299 Ops.ADD dtypes.f16 [165, 289]
300 Ops.STACK dtypes.f16 [292, 293, 294, 295, 296, 297, 298, 299]
301 Ops.WMMA dtypes.f32 [291, 300, 240] ((16, 16, 32), dtypes.f16, 64, None)
302 Ops.ADD dtypes.bf16 [6, 11]
303 Ops.STORE dtypes.void [3, 302]
304 Ops.MUL dtypes.bf16 [6, 37]
305 Ops.STORE dtypes.void [3, 304]
306 Ops.CMPLT dtypes.bool [6, 37]
307 Ops.STORE dtypes.void [57, 306]
308 Ops.CAST dtypes.bf16 [116] dtypes.bf16
309 Ops.STORE dtypes.void [3, 308]
310 Ops.CAST dtypes.i32 [6] dtypes.i32
311 Ops.STORE dtypes.void [132, 310]
312 Ops.CAST dtypes.f16 [139] dtypes.f16
313 Ops.STORE dtypes.void [136, 312]
314 Ops.SQRT dtypes.bf16 [6]
315 Ops.STORE dtypes.void [3, 314]
316 Ops.EXP2 dtypes.bf16 [6]
317 Ops.STORE dtypes.void [3, 316]
318 Ops.CAST dtypes.f32 [6] dtypes.f32
319 Ops.STORE dtypes.void [96, 318]
320 Ops.CAST dtypes.bf16 [100] dtypes.bf16
321 Ops.STORE dtypes.void [3, 320]
322 Ops.CAST dtypes.f32 [172] dtypes.f32
323 Ops.CAST dtypes.fp8e4m3 [11] dtypes.fp8e4m3
324 Ops.CAST dtypes.f32 [323] dtypes.f32
325 Ops.ADD dtypes.f32 [322, 324]
326 Ops.CAST dtypes.fp8e4m3 [325] dtypes.fp8e4m3
327 Ops.STORE dtypes.void [169, 326]
328 Ops.CAST dtypes.f32 [177] dtypes.f32
329 Ops.CMPLT dtypes.bool [322, 328]
330 Ops.STORE dtypes.void [57, 329]
331 Ops.CAST dtypes.fp8e5m2 [322] dtypes.fp8e5m2
332 Ops.STORE dtypes.void [181, 331]
333 Ops.CAST dtypes.fp8e4m3fnuz [322] dtypes.fp8e4m3fnuz
334 Ops.STORE dtypes.void [187, 333]
335 Ops.CAST dtypes.f32 [201] dtypes.f32
336 Ops.CAST dtypes.f32 [204] dtypes.f32
337 Ops.MUL dtypes.f32 [335, 336]
338 Ops.CAST dtypes.fp8e4m3fnuz [337] dtypes.fp8e4m3fnuz
339 Ops.STORE dtypes.void [187, 338]
340 Ops.CAST dtypes.f32 [208] dtypes.f32
341 Ops.ADD dtypes.f32 [322, 340]
342 Ops.CAST dtypes.fp8e4m3 [341] dtypes.fp8e4m3
343 Ops.CAST dtypes.f32 [210] dtypes.f32
344 Ops.ADD dtypes.f32 [322, 343]
345 Ops.CAST dtypes.fp8e4m3 [344] dtypes.fp8e4m3
346 Ops.CAST dtypes.f32 [213] dtypes.f32
347 Ops.ADD dtypes.f32 [322, 346]
348 Ops.CAST dtypes.fp8e4m3 [347] dtypes.fp8e4m3
349 Ops.CAST dtypes.f32 [216] dtypes.f32
350 Ops.ADD dtypes.f32 [322, 349]
351 Ops.CAST dtypes.fp8e4m3 [350] dtypes.fp8e4m3
352 Ops.CAST dtypes.f32 [219] dtypes.f32
353 Ops.ADD dtypes.f32 [322, 352]
354 Ops.CAST dtypes.fp8e4m3 [353] dtypes.fp8e4m3
355 Ops.CAST dtypes.f32 [222] dtypes.f32
356 Ops.ADD dtypes.f32 [322, 355]
357 Ops.CAST dtypes.fp8e4m3 [356] dtypes.fp8e4m3
358 Ops.CAST dtypes.f32 [225] dtypes.f32
359 Ops.ADD dtypes.f32 [322, 358]
360 Ops.CAST dtypes.fp8e4m3 [359] dtypes.fp8e4m3
361 Ops.CAST dtypes.f32 [228] dtypes.f32
362 Ops.ADD dtypes.f32 [322, 361]
363 Ops.CAST dtypes.fp8e4m3 [362] dtypes.fp8e4m3
364 Ops.STACK dtypes.fp8e4m3 [342, 345, 348, 351, 354, 357, 360, 363]
365 Ops.ADD dtypes.f32 [328, 340]
366 Ops.CAST dtypes.fp8e4m3 [365] dtypes.fp8e4m3
367 Ops.ADD dtypes.f32 [328, 343]
368 Ops.CAST dtypes.fp8e4m3 [367] dtypes.fp8e4m3
369 Ops.ADD dtypes.f32 [328, 346]
370 Ops.CAST dtypes.fp8e4m3 [369] dtypes.fp8e4m3
371 Ops.ADD dtypes.f32 [328, 349]
372 Ops.CAST dtypes.fp8e4m3 [371] dtypes.fp8e4m3
373 Ops.ADD dtypes.f32 [328, 352]
374 Ops.CAST dtypes.fp8e4m3 [373] dtypes.fp8e4m3
375 Ops.ADD dtypes.f32 [328, 355]
376 Ops.CAST dtypes.fp8e4m3 [375] dtypes.fp8e4m3
377 Ops.ADD dtypes.f32 [328, 358]
378 Ops.CAST dtypes.fp8e4m3 [377] dtypes.fp8e4m3
379 Ops.ADD dtypes.f32 [328, 361]
380 Ops.CAST dtypes.fp8e4m3 [379] dtypes.fp8e4m3
381 Ops.STACK dtypes.fp8e4m3 [366, 368, 370, 372, 374, 376, 378, 380]
382 Ops.WMMA dtypes.f32 [364, 381, 240] ((16, 16, 32), dtypes.fp8e4m3, 64, None)
383 Ops.CAST dtypes.f32 [244] dtypes.f32
384 Ops.CAST dtypes.f32 [245] dtypes.f32
385 Ops.ADD dtypes.f32 [383, 384]
386 Ops.CAST dtypes.fp8e5m2fnuz [385] dtypes.fp8e5m2fnuz
387 Ops.CAST dtypes.f32 [247] dtypes.f32
388 Ops.ADD dtypes.f32 [383, 387]
389 Ops.CAST dtypes.fp8e5m2fnuz [388] dtypes.fp8e5m2fnuz
390 Ops.CAST dtypes.f32 [249] dtypes.f32
391 Ops.ADD dtypes.f32 [383, 390]
392 Ops.CAST dtypes.fp8e5m2fnuz [391] dtypes.fp8e5m2fnuz
393 Ops.CAST dtypes.f32 [251] dtypes.f32
394 Ops.ADD dtypes.f32 [383, 393]
395 Ops.CAST dtypes.fp8e5m2fnuz [394] dtypes.fp8e5m2fnuz
396 Ops.CAST dtypes.f32 [253] dtypes.f32
397 Ops.ADD dtypes.f32 [383, 396]
398 Ops.CAST dtypes.fp8e5m2fnuz [397] dtypes.fp8e5m2fnuz
399 Ops.CAST dtypes.f32 [255] dtypes.f32
400 Ops.ADD dtypes.f32 [383, 399]
401 Ops.CAST dtypes.fp8e5m2fnuz [400] dtypes.fp8e5m2fnuz
402 Ops.CAST dtypes.f32 [257] dtypes.f32
403 Ops.ADD dtypes.f32 [383, 402]
404 Ops.CAST dtypes.fp8e5m2fnuz [403] dtypes.fp8e5m2fnuz
405 Ops.CAST dtypes.f32 [259] dtypes.f32
406 Ops.ADD dtypes.f32 [383, 405]
407 Ops.CAST dtypes.fp8e5m2fnuz [406] dtypes.fp8e5m2fnuz
408 Ops.STACK dtypes.fp8e5m2fnuz [386, 389, 392, 395, 398, 401, 404, 407]
409 Ops.CAST dtypes.f32 [264] dtypes.f32
410 Ops.ADD dtypes.f32 [409, 384]
411 Ops.CAST dtypes.fp8e5m2fnuz [410] dtypes.fp8e5m2fnuz
412 Ops.ADD dtypes.f32 [409, 387]
413 Ops.CAST dtypes.fp8e5m2fnuz [412] dtypes.fp8e5m2fnuz
414 Ops.ADD dtypes.f32 [409, 390]
415 Ops.CAST dtypes.fp8e5m2fnuz [414] dtypes.fp8e5m2fnuz
416 Ops.ADD dtypes.f32 [409, 393]
417 Ops.CAST dtypes.fp8e5m2fnuz [416] dtypes.fp8e5m2fnuz
418 Ops.ADD dtypes.f32 [409, 396]
419 Ops.CAST dtypes.fp8e5m2fnuz [418] dtypes.fp8e5m2fnuz
420 Ops.ADD dtypes.f32 [409, 399]
421 Ops.CAST dtypes.fp8e5m2fnuz [420] dtypes.fp8e5m2fnuz
422 Ops.ADD dtypes.f32 [409, 402]
423 Ops.CAST dtypes.fp8e5m2fnuz [422] dtypes.fp8e5m2fnuz
424 Ops.ADD dtypes.f32 [409, 405]
425 Ops.CAST dtypes.fp8e5m2fnuz [424] dtypes.fp8e5m2fnuz
426 Ops.STACK dtypes.fp8e5m2fnuz [411, 413, 415, 417, 419, 421, 423, 425]
427 Ops.WMMA dtypes.f32 [408, 426, 240] ((16, 16, 32), dtypes.fp8e5m2fnuz, 64, None)
428 Ops.CAST dtypes.fp8e4m3 [322] dtypes.fp8e4m3
429 Ops.STORE dtypes.void [169, 428]
430 Ops.CAST dtypes.fp8e5m2 [117] dtypes.fp8e5m2
431 Ops.STORE dtypes.void [181, 430]
432 Ops.CAST dtypes.f32 [196] dtypes.f32
433 Ops.CAST dtypes.i32 [432] dtypes.i32
434 Ops.STORE dtypes.void [132, 433]
435 Ops.BITCAST dtypes.u32 [364] dtypes.u32
436 Ops.BITCAST dtypes.u32 [381] dtypes.u32
437 Ops.WMMA dtypes.f32 [435, 436, 240] ((16, 16, 32), dtypes.fp8e4m3, 64, None)
438 Ops.BITCAST dtypes.u32 [408] dtypes.u32
439 Ops.BITCAST dtypes.u32 [426] dtypes.u32
440 Ops.WMMA dtypes.f32 [438, 439, 240] ((16, 16, 32), dtypes.fp8e5m2fnuz, 64, None)
441 Ops.BITCAST dtypes.u64 [364] dtypes.u64
442 Ops.BITCAST dtypes.u64 [381] dtypes.u64
443 Ops.WMMA dtypes.f32 [441, 442, 240] ((16, 16, 32), dtypes.fp8e4m3, 64, None)
444 Ops.BITCAST dtypes.u64 [408] dtypes.u64
445 Ops.BITCAST dtypes.u64 [426] dtypes.u64
446 Ops.WMMA dtypes.f32 [444, 445, 240] ((16, 16, 32), dtypes.fp8e5m2fnuz, 64, None)
447 Ops.SINK dtypes.void [34, 55, 59, 66, 80, 94, 97, 113, 130, 134, 161, 162, 167, 174, 179, 183, 185, 189, 191, 193, 198, 206, 241, 274, 301, 303, 305, 307, 66, 80, 94, 97, 113, 309, 311, 313, 162, 167, 174, 179, 183, 185, 189, 191, 193, 198, 206, 241, 274, 301, 303, 305, 307, 66, 315, 317, 319, 321, 309, 311, 313, 162, 167, 327, 330, 332, 185, 334, 191, 193, 198, 339, 382, 427, 301, 34, 55, 59, 66, 80, 94, 97, 113, 130, 134, 313, 162, 167, 327, 330, 332, 429, 334, 191, 431, 434, 339, 382, 427, 301, 34, 55, 59, 66, 80, 94, 97, 113, 130, 134, 313, 162, 167, 327, 330, 332, 429, 334, 191, 431, 434, 339, 437, 440, 301, 34, 55, 59, 66, 80, 94, 97, 113, 130, 134, 313, 162, 167, 327, 330, 332, 429, 334, 191, 431, 434, 339, 443, 446, 301, 34, 55, 59, 66, 80, 94, 97, 113, 130, 134, 313, 162, 167, 327, 330, 332, 429, 334, 191, 431, 434, 339, 443, 446, 301]

