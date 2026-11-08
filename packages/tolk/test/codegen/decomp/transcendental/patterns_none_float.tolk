# tolk D31 D126 D413 golden=3df05fc6d3b7af1e4523bd66b6f84bd9
0 Ops.STACK dtypes.void []
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CMPEQ dtypes.bool [1, 1]
3 Ops.CONST dtypes.bool [] False
4 Ops.CMPEQ dtypes.bool [2, 3]
5 Ops.CONST dtypes.weakfloat [] nan
6 Ops.CAST dtypes.f32 [5] dtypes.f32
7 Ops.CONST dtypes.weakfloat [] -150.0
8 Ops.CMPLT dtypes.bool [1, 7]
9 Ops.CONST dtypes.weakfloat [] 0.0
10 Ops.CAST dtypes.f32 [9] dtypes.f32
11 Ops.CONST dtypes.weakfloat [] 128.0
12 Ops.CMPLT dtypes.bool [1, 11]
13 Ops.CMPEQ dtypes.bool [12, 3]
14 Ops.CONST dtypes.weakfloat [] inf
15 Ops.CAST dtypes.f32 [14] dtypes.f32
16 Ops.CONST dtypes.weakfloat [] 0.0001535920892
17 Ops.CMPEQ dtypes.bool [1, 14]
18 Ops.CMPEQ dtypes.bool [17, 3]
19 Ops.CONST dtypes.weakfloat [] -inf
20 Ops.CMPEQ dtypes.bool [1, 19]
21 Ops.CMPEQ dtypes.bool [20, 3]
22 Ops.WHERE dtypes.f32 [21, 1, 10]
23 Ops.WHERE dtypes.f32 [4, 10, 22]
24 Ops.WHERE dtypes.f32 [18, 23, 10]
25 Ops.CMPLT dtypes.bool [24, 9]
26 Ops.CONST dtypes.weakfloat [] -0.5
27 Ops.CAST dtypes.f32 [26] dtypes.f32
28 Ops.CONST dtypes.weakfloat [] 0.5
29 Ops.CAST dtypes.f32 [28] dtypes.f32
30 Ops.WHERE dtypes.f32 [25, 27, 29]
31 Ops.ADD dtypes.f32 [24, 30]
32 Ops.CAST dtypes.i32 [31] dtypes.i32
33 Ops.CAST dtypes.f32 [32] dtypes.f32
34 Ops.CONST dtypes.weakfloat [] -1.0
35 Ops.MUL dtypes.f32 [33, 34]
36 Ops.ADD dtypes.f32 [24, 35]
37 Ops.MUL dtypes.f32 [16, 36]
38 Ops.CONST dtypes.weakfloat [] 0.001339262701
39 Ops.ADD dtypes.f32 [37, 38]
40 Ops.MUL dtypes.f32 [39, 36]
41 Ops.CONST dtypes.weakfloat [] 0.009618384764
42 Ops.ADD dtypes.f32 [40, 41]
43 Ops.MUL dtypes.f32 [42, 36]
44 Ops.CONST dtypes.weakfloat [] 0.05550347269
45 Ops.ADD dtypes.f32 [43, 44]
46 Ops.MUL dtypes.f32 [45, 36]
47 Ops.CONST dtypes.weakfloat [] 0.2402264476
48 Ops.ADD dtypes.f32 [46, 47]
49 Ops.MUL dtypes.f32 [48, 36]
50 Ops.CONST dtypes.weakfloat [] 0.6931471825
51 Ops.ADD dtypes.f32 [49, 50]
52 Ops.MUL dtypes.f32 [51, 36]
53 Ops.CONST dtypes.weakfloat [] 1.0
54 Ops.ADD dtypes.f32 [52, 53]
55 Ops.CONST dtypes.weakint [] 2
56 Ops.DIV dtypes.i32 [32, 55]
57 Ops.CONST dtypes.weakint [] 127
58 Ops.ADD dtypes.i32 [56, 57]
59 Ops.CONST dtypes.weakint [] 8388608
60 Ops.MUL dtypes.i32 [58, 59]
61 Ops.BITCAST dtypes.f32 [60] dtypes.f32
62 Ops.MUL dtypes.f32 [54, 61]
63 Ops.CONST dtypes.weakint [] -1
64 Ops.MUL dtypes.i32 [56, 63]
65 Ops.ADD dtypes.i32 [32, 64]
66 Ops.ADD dtypes.i32 [65, 57]
67 Ops.MUL dtypes.i32 [66, 59]
68 Ops.BITCAST dtypes.f32 [67] dtypes.f32
69 Ops.MUL dtypes.f32 [62, 68]
70 Ops.WHERE dtypes.f32 [13, 15, 69]
71 Ops.WHERE dtypes.f32 [8, 10, 70]
72 Ops.WHERE dtypes.f32 [4, 6, 71]
73 Ops.CONST dtypes.weakfloat [] -0.0
74 Ops.CMPLT dtypes.bool [1, 73]
75 Ops.RECIP dtypes.f32 [1]
76 Ops.CMPEQ dtypes.bool [75, 19]
77 Ops.CMPEQ dtypes.bool [76, 3]
78 Ops.CMPEQ dtypes.bool [1, 9]
79 Ops.CMPEQ dtypes.bool [78, 3]
80 Ops.CONST dtypes.weakfloat [] 1.1754943508222875e-38
81 Ops.CMPLT dtypes.bool [1, 80]
82 Ops.CONST dtypes.weakfloat [] 1.8446744073709552e+19
83 Ops.MUL dtypes.f32 [1, 82]
84 Ops.WHERE dtypes.f32 [81, 83, 1]
85 Ops.BITCAST dtypes.i32 [84] dtypes.i32
86 Ops.CONST dtypes.weakfloat [] 1.4142135623730951
87 Ops.MUL dtypes.f32 [84, 86]
88 Ops.BITCAST dtypes.i32 [87] dtypes.i32
89 Ops.DIV dtypes.i32 [88, 59]
90 Ops.CONST dtypes.weakint [] 255
91 Ops.AND dtypes.i32 [89, 90]
92 Ops.MUL dtypes.weakint [57, 63]
93 Ops.ADD dtypes.i32 [91, 92]
94 Ops.CAST dtypes.f32 [93] dtypes.f32
95 Ops.MUL dtypes.f32 [94, 34]
96 Ops.CAST dtypes.i32 [95] dtypes.i32
97 Ops.MUL dtypes.i32 [96, 59]
98 Ops.ADD dtypes.i32 [85, 97]
99 Ops.BITCAST dtypes.f32 [98] dtypes.f32
100 Ops.MUL dtypes.weakfloat [53, 34]
101 Ops.ADD dtypes.f32 [99, 100]
102 Ops.MUL dtypes.f32 [28, 101]
103 Ops.MUL dtypes.f32 [102, 101]
104 Ops.MUL dtypes.f32 [103, 34]
105 Ops.ADD dtypes.f32 [101, 104]
106 Ops.BITCAST dtypes.u32 [105] dtypes.u32
107 Ops.CONST dtypes.weakint [] 4294963200
108 Ops.CAST dtypes.u32 [107] dtypes.u32
109 Ops.AND dtypes.u32 [106, 108]
110 Ops.BITCAST dtypes.f32 [109] dtypes.f32
111 Ops.MUL dtypes.f32 [110, 34]
112 Ops.ADD dtypes.f32 [101, 111]
113 Ops.ADD dtypes.f32 [112, 104]
114 Ops.CONST dtypes.weakfloat [] 2.0
115 Ops.ADD dtypes.f32 [114, 101]
116 Ops.RECIP dtypes.f32 [115]
117 Ops.MUL dtypes.f32 [101, 116]
118 Ops.MUL dtypes.f32 [117, 117]
119 Ops.CONST dtypes.weakfloat [] 0.24279078841209412
120 Ops.MUL dtypes.f32 [119, 118]
121 Ops.CONST dtypes.weakfloat [] 0.2849878668785095
122 Ops.ADD dtypes.f32 [120, 121]
123 Ops.MUL dtypes.f32 [122, 118]
124 Ops.CONST dtypes.weakfloat [] 0.40000972151756287
125 Ops.ADD dtypes.f32 [123, 124]
126 Ops.MUL dtypes.f32 [125, 118]
127 Ops.CONST dtypes.weakfloat [] 0.6666666269302368
128 Ops.ADD dtypes.f32 [126, 127]
129 Ops.MUL dtypes.f32 [118, 128]
130 Ops.ADD dtypes.f32 [103, 129]
131 Ops.MUL dtypes.f32 [117, 130]
132 Ops.ADD dtypes.f32 [113, 131]
133 Ops.ADD dtypes.f32 [132, 110]
134 Ops.CONST dtypes.weakfloat [] -0.00017605285393
135 Ops.MUL dtypes.f32 [133, 134]
136 Ops.CONST dtypes.weakfloat [] 1.4428710938
137 Ops.MUL dtypes.f32 [132, 136]
138 Ops.ADD dtypes.f32 [135, 137]
139 Ops.CONST dtypes.weakfloat [] 64.0
140 Ops.MUL dtypes.weakfloat [139, 34]
141 Ops.ADD dtypes.f32 [94, 140]
142 Ops.WHERE dtypes.f32 [81, 141, 94]
143 Ops.MUL dtypes.f32 [110, 136]
144 Ops.ADD dtypes.f32 [142, 143]
145 Ops.MUL dtypes.f32 [144, 34]
146 Ops.ADD dtypes.f32 [142, 145]
147 Ops.ADD dtypes.f32 [146, 143]
148 Ops.ADD dtypes.f32 [138, 147]
149 Ops.ADD dtypes.f32 [148, 144]
150 Ops.WHERE dtypes.f32 [18, 149, 15]
151 Ops.CAST dtypes.f32 [19] dtypes.f32
152 Ops.WHERE dtypes.f32 [79, 150, 151]
153 Ops.WHERE dtypes.f32 [77, 152, 151]
154 Ops.WHERE dtypes.f32 [74, 6, 153]
155 Ops.WHERE dtypes.f32 [4, 6, 154]
156 Ops.CMPEQ dtypes.bool [24, 9]
157 Ops.CMPEQ dtypes.bool [156, 3]
158 Ops.CAST dtypes.f32 [34] dtypes.f32
159 Ops.CAST dtypes.f32 [53] dtypes.f32
160 Ops.WHERE dtypes.f32 [25, 158, 159]
161 Ops.MUL dtypes.f32 [24, 160]
162 Ops.CONST dtypes.weakfloat [] 30.0
163 Ops.CMPLT dtypes.bool [161, 162]
164 Ops.CONST dtypes.weakfloat [] 0.6366197723675814
165 Ops.MUL dtypes.f32 [161, 164]
166 Ops.CMPLT dtypes.bool [165, 9]
167 Ops.WHERE dtypes.f32 [166, 27, 29]
168 Ops.ADD dtypes.f32 [165, 167]
169 Ops.CAST dtypes.i32 [168] dtypes.i32
170 Ops.BITCAST dtypes.u32 [161] dtypes.u32
171 Ops.CONST dtypes.weakint [] 2155872255
172 Ops.AND dtypes.u32 [170, 171]
173 Ops.CONST dtypes.weakint [] 1056964608
174 Ops.OR dtypes.u32 [172, 173]
175 Ops.BITCAST dtypes.f32 [174] dtypes.f32
176 Ops.CONST dtypes.weakfloat [] 16777216.0
177 Ops.MUL dtypes.f32 [175, 176]
178 Ops.CAST dtypes.u64 [177] dtypes.u64
179 Ops.CONST dtypes.weakint [] 4294967295
180 Ops.AND dtypes.u64 [178, 179]
181 Ops.DIV dtypes.u32 [170, 59]
182 Ops.AND dtypes.u32 [181, 90]
183 Ops.ADD dtypes.u32 [182, 92]
184 Ops.CONST dtypes.weakint [] 1
185 Ops.ADD dtypes.u32 [183, 184]
186 Ops.CAST dtypes.i32 [185] dtypes.i32
187 Ops.CONST dtypes.weakint [] 104
188 Ops.ADD dtypes.i32 [186, 187]
189 Ops.CONST dtypes.weakint [] 0
190 Ops.MAX dtypes.i32 [188, 189]
191 Ops.CONST dtypes.weakint [] 32
192 Ops.DIV dtypes.i32 [190, 191]
193 Ops.AND dtypes.i32 [192, 184]
194 Ops.CMPEQ dtypes.bool [193, 189]
195 Ops.CMPEQ dtypes.bool [194, 3]
196 Ops.AND dtypes.i32 [192, 55]
197 Ops.CMPEQ dtypes.bool [196, 189]
198 Ops.CMPEQ dtypes.bool [197, 3]
199 Ops.CONST dtypes.weakint [] 4
200 Ops.AND dtypes.i32 [192, 199]
201 Ops.CMPEQ dtypes.bool [200, 189]
202 Ops.CMPEQ dtypes.bool [201, 3]
203 Ops.CONST dtypes.weakint [] 2102212464
204 Ops.CAST dtypes.u64 [203] dtypes.u64
205 Ops.CAST dtypes.u64 [189] dtypes.u64
206 Ops.WHERE dtypes.u64 [202, 204, 205]
207 Ops.CONST dtypes.weakint [] 2475754826
208 Ops.CAST dtypes.u64 [207] dtypes.u64
209 Ops.WHERE dtypes.u64 [202, 208, 205]
210 Ops.WHERE dtypes.u64 [198, 206, 209]
211 Ops.CONST dtypes.weakint [] 2131351028
212 Ops.CAST dtypes.u64 [211] dtypes.u64
213 Ops.WHERE dtypes.u64 [202, 212, 205]
214 Ops.CONST dtypes.weakint [] 683565275
215 Ops.CAST dtypes.u64 [214] dtypes.u64
216 Ops.WHERE dtypes.u64 [202, 215, 205]
217 Ops.WHERE dtypes.u64 [198, 213, 216]
218 Ops.WHERE dtypes.u64 [195, 210, 217]
219 Ops.CONST dtypes.weakint [] 31
220 Ops.AND dtypes.i32 [190, 219]
221 Ops.CAST dtypes.u64 [220] dtypes.u64
222 Ops.SHL dtypes.u64 [218, 221]
223 Ops.CONST dtypes.weakint [] 920167782
224 Ops.CAST dtypes.u64 [223] dtypes.u64
225 Ops.WHERE dtypes.u64 [202, 224, 215]
226 Ops.WHERE dtypes.u64 [198, 225, 213]
227 Ops.WHERE dtypes.u64 [195, 226, 210]
228 Ops.CONST dtypes.weakint [] 18446744073709551615
229 Ops.MUL dtypes.u64 [221, 228]
230 Ops.ADD dtypes.u64 [191, 229]
231 Ops.SHR dtypes.u64 [227, 230]
232 Ops.OR dtypes.u64 [222, 231]
233 Ops.AND dtypes.u64 [232, 179]
234 Ops.MUL dtypes.u64 [180, 233]
235 Ops.AND dtypes.u64 [234, 179]
236 Ops.SHL dtypes.u64 [227, 221]
237 Ops.CONST dtypes.weakint [] 1326507024
238 Ops.CAST dtypes.u64 [237] dtypes.u64
239 Ops.WHERE dtypes.u64 [202, 238, 208]
240 Ops.WHERE dtypes.u64 [198, 239, 206]
241 Ops.WHERE dtypes.u64 [195, 240, 226]
242 Ops.SHR dtypes.u64 [241, 230]
243 Ops.OR dtypes.u64 [236, 242]
244 Ops.AND dtypes.u64 [243, 179]
245 Ops.MUL dtypes.u64 [180, 244]
246 Ops.CONST dtypes.weakint [] 4294967296
247 Ops.DIV dtypes.u64 [245, 246]
248 Ops.ADD dtypes.u64 [235, 247]
249 Ops.AND dtypes.u64 [245, 179]
250 Ops.SHL dtypes.u64 [241, 221]
251 Ops.CONST dtypes.weakint [] 2140428522
252 Ops.CAST dtypes.u64 [251] dtypes.u64
253 Ops.WHERE dtypes.u64 [202, 252, 212]
254 Ops.WHERE dtypes.u64 [198, 253, 225]
255 Ops.WHERE dtypes.u64 [195, 254, 240]
256 Ops.SHR dtypes.u64 [255, 230]
257 Ops.OR dtypes.u64 [250, 256]
258 Ops.AND dtypes.u64 [257, 179]
259 Ops.MUL dtypes.u64 [180, 258]
260 Ops.DIV dtypes.u64 [259, 246]
261 Ops.ADD dtypes.u64 [249, 260]
262 Ops.AND dtypes.u64 [259, 179]
263 Ops.SHL dtypes.u64 [255, 221]
264 Ops.CONST dtypes.weakint [] 4155437400
265 Ops.CAST dtypes.u64 [264] dtypes.u64
266 Ops.WHERE dtypes.u64 [202, 265, 204]
267 Ops.WHERE dtypes.u64 [198, 266, 239]
268 Ops.WHERE dtypes.u64 [195, 267, 254]
269 Ops.SHR dtypes.u64 [268, 230]
270 Ops.OR dtypes.u64 [263, 269]
271 Ops.AND dtypes.u64 [270, 179]
272 Ops.MUL dtypes.u64 [180, 271]
273 Ops.DIV dtypes.u64 [272, 246]
274 Ops.ADD dtypes.u64 [262, 273]
275 Ops.AND dtypes.u64 [272, 179]
276 Ops.SHL dtypes.u64 [268, 221]
277 Ops.CONST dtypes.weakint [] 1841896334
278 Ops.CAST dtypes.u64 [277] dtypes.u64
279 Ops.WHERE dtypes.u64 [202, 278, 224]
280 Ops.WHERE dtypes.u64 [198, 279, 253]
281 Ops.WHERE dtypes.u64 [195, 280, 267]
282 Ops.SHR dtypes.u64 [281, 230]
283 Ops.OR dtypes.u64 [276, 282]
284 Ops.AND dtypes.u64 [283, 179]
285 Ops.MUL dtypes.u64 [180, 284]
286 Ops.DIV dtypes.u64 [285, 246]
287 Ops.ADD dtypes.u64 [275, 286]
288 Ops.AND dtypes.u64 [285, 179]
289 Ops.SHL dtypes.u64 [281, 221]
290 Ops.CONST dtypes.weakint [] 2425582776
291 Ops.CAST dtypes.u64 [290] dtypes.u64
292 Ops.WHERE dtypes.u64 [202, 291, 238]
293 Ops.WHERE dtypes.u64 [198, 292, 266]
294 Ops.WHERE dtypes.u64 [195, 293, 280]
295 Ops.SHR dtypes.u64 [294, 230]
296 Ops.OR dtypes.u64 [289, 295]
297 Ops.AND dtypes.u64 [296, 179]
298 Ops.MUL dtypes.u64 [180, 297]
299 Ops.DIV dtypes.u64 [298, 246]
300 Ops.ADD dtypes.u64 [288, 299]
301 Ops.AND dtypes.u64 [298, 179]
302 Ops.SHL dtypes.u64 [294, 221]
303 Ops.CONST dtypes.weakint [] 26364858
304 Ops.CAST dtypes.u64 [303] dtypes.u64
305 Ops.WHERE dtypes.u64 [202, 304, 252]
306 Ops.WHERE dtypes.u64 [198, 305, 279]
307 Ops.WHERE dtypes.u64 [195, 306, 293]
308 Ops.SHR dtypes.u64 [307, 230]
309 Ops.OR dtypes.u64 [302, 308]
310 Ops.AND dtypes.u64 [309, 179]
311 Ops.MUL dtypes.u64 [180, 310]
312 Ops.DIV dtypes.u64 [311, 246]
313 Ops.ADD dtypes.u64 [301, 312]
314 Ops.AND dtypes.u64 [311, 179]
315 Ops.DIV dtypes.u64 [314, 246]
316 Ops.ADD dtypes.u64 [313, 315]
317 Ops.DIV dtypes.u64 [316, 246]
318 Ops.ADD dtypes.u64 [300, 317]
319 Ops.DIV dtypes.u64 [318, 246]
320 Ops.ADD dtypes.u64 [287, 319]
321 Ops.DIV dtypes.u64 [320, 246]
322 Ops.ADD dtypes.u64 [274, 321]
323 Ops.DIV dtypes.u64 [322, 246]
324 Ops.ADD dtypes.u64 [261, 323]
325 Ops.DIV dtypes.u64 [324, 246]
326 Ops.ADD dtypes.u64 [248, 325]
327 Ops.AND dtypes.u64 [326, 179]
328 Ops.MUL dtypes.u64 [327, 246]
329 Ops.AND dtypes.u64 [324, 179]
330 Ops.OR dtypes.u64 [328, 329]
331 Ops.CONST dtypes.weakint [] 4611686018427387904
332 Ops.DIV dtypes.u64 [330, 331]
333 Ops.CONST dtypes.weakint [] 2305843009213693952
334 Ops.DIV dtypes.u64 [330, 333]
335 Ops.AND dtypes.u64 [334, 184]
336 Ops.ADD dtypes.u64 [332, 335]
337 Ops.CAST dtypes.i32 [336] dtypes.i32
338 Ops.WHERE dtypes.i32 [163, 169, 337]
339 Ops.AND dtypes.i32 [338, 55]
340 Ops.CMPEQ dtypes.bool [339, 189]
341 Ops.CMPEQ dtypes.bool [340, 3]
342 Ops.AND dtypes.i32 [338, 184]
343 Ops.CMPEQ dtypes.bool [342, 189]
344 Ops.CMPEQ dtypes.bool [343, 3]
345 Ops.CAST dtypes.f32 [169] dtypes.f32
346 Ops.CONST dtypes.weakfloat [] -0.00048351287841796875
347 Ops.MUL dtypes.f32 [345, 346]
348 Ops.CONST dtypes.weakfloat [] -1.5703125
349 Ops.MUL dtypes.f32 [345, 348]
350 Ops.ADD dtypes.f32 [349, 161]
351 Ops.ADD dtypes.f32 [347, 350]
352 Ops.CONST dtypes.weakfloat [] -3.1385570764541626e-07
353 Ops.MUL dtypes.f32 [345, 352]
354 Ops.ADD dtypes.f32 [351, 353]
355 Ops.MUL dtypes.f32 [351, 34]
356 Ops.ADD dtypes.f32 [354, 355]
357 Ops.MUL dtypes.f32 [356, 34]
358 Ops.ADD dtypes.f32 [353, 357]
359 Ops.CONST dtypes.weakfloat [] -6.07710062827671e-11
360 Ops.MUL dtypes.f32 [345, 359]
361 Ops.ADD dtypes.f32 [358, 360]
362 Ops.ADD dtypes.f32 [354, 361]
363 Ops.CONST dtypes.weakint [] 4611686018427387903
364 Ops.AND dtypes.u64 [330, 363]
365 Ops.CAST dtypes.i64 [364] dtypes.i64
366 Ops.CAST dtypes.i64 [335] dtypes.i64
367 Ops.MUL dtypes.i64 [366, 331]
368 Ops.MUL dtypes.i64 [367, 63]
369 Ops.ADD dtypes.i64 [365, 368]
370 Ops.CONST dtypes.weakint [] 4398046511104
371 Ops.DIV dtypes.i64 [369, 370]
372 Ops.CAST dtypes.f32 [371] dtypes.f32
373 Ops.CONST dtypes.weakfloat [] 4398046511104.0
374 Ops.MUL dtypes.f32 [372, 373]
375 Ops.CONST dtypes.weakint [] 2097152
376 Ops.DIV dtypes.i64 [369, 375]
377 Ops.CONST dtypes.weakint [] 2097151
378 Ops.AND dtypes.i64 [376, 377]
379 Ops.CAST dtypes.f32 [378] dtypes.f32
380 Ops.CONST dtypes.weakfloat [] 2097152.0
381 Ops.MUL dtypes.f32 [379, 380]
382 Ops.ADD dtypes.f32 [374, 381]
383 Ops.AND dtypes.i64 [369, 377]
384 Ops.CAST dtypes.f32 [383] dtypes.f32
385 Ops.MUL dtypes.f32 [384, 53]
386 Ops.ADD dtypes.f32 [382, 385]
387 Ops.AND dtypes.u64 [322, 179]
388 Ops.MUL dtypes.u64 [387, 246]
389 Ops.AND dtypes.u64 [320, 179]
390 Ops.OR dtypes.u64 [388, 389]
391 Ops.CONST dtypes.weakint [] 8796093022208
392 Ops.DIV dtypes.u64 [390, 391]
393 Ops.CAST dtypes.f32 [392] dtypes.f32
394 Ops.CONST dtypes.weakfloat [] 1.1368683772161603e-13
395 Ops.MUL dtypes.f32 [393, 394]
396 Ops.ADD dtypes.f32 [386, 395]
397 Ops.CONST dtypes.weakfloat [] 3.4061216748705226e-19
398 Ops.MUL dtypes.f32 [396, 397]
399 Ops.CONST dtypes.weakfloat [] 4097.0
400 Ops.MUL dtypes.f32 [396, 399]
401 Ops.MUL dtypes.f32 [396, 34]
402 Ops.ADD dtypes.f32 [400, 401]
403 Ops.MUL dtypes.f32 [402, 34]
404 Ops.ADD dtypes.f32 [400, 403]
405 Ops.CONST dtypes.weakfloat [] 3.406131239146355e-19
406 Ops.MUL dtypes.f32 [404, 405]
407 Ops.MUL dtypes.f32 [398, 34]
408 Ops.ADD dtypes.f32 [406, 407]
409 Ops.CONST dtypes.weakfloat [] -9.564275832644382e-25
410 Ops.MUL dtypes.f32 [404, 409]
411 Ops.ADD dtypes.f32 [408, 410]
412 Ops.MUL dtypes.f32 [404, 34]
413 Ops.ADD dtypes.f32 [396, 412]
414 Ops.MUL dtypes.f32 [413, 405]
415 Ops.ADD dtypes.f32 [411, 414]
416 Ops.MUL dtypes.f32 [413, 409]
417 Ops.ADD dtypes.f32 [415, 416]
418 Ops.CONST dtypes.weakfloat [] -9.47839680047608e-27
419 Ops.MUL dtypes.f32 [396, 418]
420 Ops.ADD dtypes.f32 [417, 419]
421 Ops.MUL dtypes.f32 [374, 34]
422 Ops.ADD dtypes.f32 [382, 421]
423 Ops.MUL dtypes.f32 [422, 34]
424 Ops.ADD dtypes.f32 [381, 423]
425 Ops.ADD dtypes.f32 [10, 424]
426 Ops.MUL dtypes.f32 [382, 34]
427 Ops.ADD dtypes.f32 [386, 426]
428 Ops.MUL dtypes.f32 [427, 34]
429 Ops.ADD dtypes.f32 [385, 428]
430 Ops.ADD dtypes.f32 [425, 429]
431 Ops.MUL dtypes.f32 [386, 34]
432 Ops.ADD dtypes.f32 [396, 431]
433 Ops.MUL dtypes.f32 [432, 34]
434 Ops.ADD dtypes.f32 [395, 433]
435 Ops.ADD dtypes.f32 [430, 434]
436 Ops.MUL dtypes.f32 [435, 397]
437 Ops.ADD dtypes.f32 [420, 436]
438 Ops.ADD dtypes.f32 [398, 437]
439 Ops.WHERE dtypes.f32 [163, 362, 438]
440 Ops.MUL dtypes.f32 [439, 439]
441 Ops.MUL dtypes.f32 [28, 440]
442 Ops.MUL dtypes.f32 [441, 34]
443 Ops.ADD dtypes.f32 [53, 442]
444 Ops.MUL dtypes.f32 [443, 34]
445 Ops.ADD dtypes.f32 [53, 444]
446 Ops.ADD dtypes.f32 [445, 442]
447 Ops.MUL dtypes.f32 [440, 440]
448 Ops.CONST dtypes.weakfloat [] -2.7557314351390663e-07
449 Ops.MUL dtypes.f32 [448, 440]
450 Ops.CONST dtypes.weakfloat [] 2.480158728947673e-05
451 Ops.ADD dtypes.f32 [449, 450]
452 Ops.MUL dtypes.f32 [451, 440]
453 Ops.CONST dtypes.weakfloat [] -0.001388888888887411
454 Ops.ADD dtypes.f32 [452, 453]
455 Ops.MUL dtypes.f32 [454, 440]
456 Ops.CONST dtypes.weakfloat [] 0.0416666666666666
457 Ops.ADD dtypes.f32 [455, 456]
458 Ops.MUL dtypes.f32 [447, 457]
459 Ops.MUL dtypes.f32 [354, 34]
460 Ops.ADD dtypes.f32 [362, 459]
461 Ops.MUL dtypes.f32 [460, 34]
462 Ops.ADD dtypes.f32 [361, 461]
463 Ops.ADD dtypes.f32 [438, 407]
464 Ops.MUL dtypes.f32 [463, 34]
465 Ops.ADD dtypes.f32 [437, 464]
466 Ops.WHERE dtypes.f32 [163, 462, 465]
467 Ops.MUL dtypes.f32 [439, 466]
468 Ops.MUL dtypes.f32 [467, 34]
469 Ops.ADD dtypes.f32 [458, 468]
470 Ops.ADD dtypes.f32 [446, 469]
471 Ops.ADD dtypes.f32 [443, 470]
472 Ops.MUL dtypes.f32 [28, 466]
473 Ops.MUL dtypes.f32 [440, 439]
474 Ops.CONST dtypes.weakfloat [] 2.7557313707070068e-06
475 Ops.MUL dtypes.f32 [474, 440]
476 Ops.CONST dtypes.weakfloat [] -0.0001984126982985795
477 Ops.ADD dtypes.f32 [475, 476]
478 Ops.MUL dtypes.f32 [477, 440]
479 Ops.CONST dtypes.weakfloat [] 0.00833333333332249
480 Ops.ADD dtypes.f32 [478, 479]
481 Ops.MUL dtypes.f32 [473, 480]
482 Ops.MUL dtypes.f32 [481, 34]
483 Ops.ADD dtypes.f32 [472, 482]
484 Ops.MUL dtypes.f32 [440, 483]
485 Ops.MUL dtypes.f32 [466, 34]
486 Ops.ADD dtypes.f32 [484, 485]
487 Ops.CONST dtypes.weakfloat [] -0.16666666666666632
488 Ops.MUL dtypes.f32 [473, 487]
489 Ops.MUL dtypes.f32 [488, 34]
490 Ops.ADD dtypes.f32 [486, 489]
491 Ops.MUL dtypes.f32 [490, 34]
492 Ops.ADD dtypes.f32 [439, 491]
493 Ops.WHERE dtypes.f32 [344, 471, 492]
494 Ops.MUL dtypes.f32 [493, 160]
495 Ops.MUL dtypes.f32 [494, 34]
496 Ops.WHERE dtypes.f32 [341, 495, 494]
497 Ops.WHERE dtypes.f32 [157, 496, 24]
498 Ops.WHERE dtypes.f32 [21, 497, 6]
499 Ops.WHERE dtypes.f32 [4, 6, 498]
500 Ops.WHERE dtypes.f32 [18, 499, 6]
501 Ops.CMPEQ dtypes.bool [29, 9]
502 Ops.CMPLT dtypes.bool [1, 9]
503 Ops.CAST dtypes.i32 [29] dtypes.i32
504 Ops.CAST dtypes.f32 [503] dtypes.f32
505 Ops.CMPEQ dtypes.bool [29, 504]
506 Ops.CMPEQ dtypes.bool [505, 3]
507 Ops.MUL dtypes.f32 [1, 34]
508 Ops.WHERE dtypes.f32 [502, 507, 1]
509 Ops.CMPEQ dtypes.bool [508, 508]
510 Ops.CMPEQ dtypes.bool [509, 3]
511 Ops.CMPLT dtypes.bool [508, 73]
512 Ops.RECIP dtypes.f32 [508]
513 Ops.CMPEQ dtypes.bool [512, 19]
514 Ops.CMPEQ dtypes.bool [513, 3]
515 Ops.CMPEQ dtypes.bool [508, 9]
516 Ops.CMPEQ dtypes.bool [515, 3]
517 Ops.CMPEQ dtypes.bool [508, 14]
518 Ops.CMPEQ dtypes.bool [517, 3]
519 Ops.CMPLT dtypes.bool [508, 80]
520 Ops.MUL dtypes.f32 [508, 82]
521 Ops.WHERE dtypes.f32 [519, 520, 508]
522 Ops.BITCAST dtypes.i32 [521] dtypes.i32
523 Ops.MUL dtypes.f32 [521, 86]
524 Ops.BITCAST dtypes.i32 [523] dtypes.i32
525 Ops.DIV dtypes.i32 [524, 59]
526 Ops.AND dtypes.i32 [525, 90]
527 Ops.ADD dtypes.i32 [526, 92]
528 Ops.CAST dtypes.f32 [527] dtypes.f32
529 Ops.MUL dtypes.f32 [528, 34]
530 Ops.CAST dtypes.i32 [529] dtypes.i32
531 Ops.MUL dtypes.i32 [530, 59]
532 Ops.ADD dtypes.i32 [522, 531]
533 Ops.BITCAST dtypes.f32 [532] dtypes.f32
534 Ops.ADD dtypes.f32 [533, 100]
535 Ops.MUL dtypes.f32 [28, 534]
536 Ops.MUL dtypes.f32 [535, 534]
537 Ops.MUL dtypes.f32 [536, 34]
538 Ops.ADD dtypes.f32 [534, 537]
539 Ops.BITCAST dtypes.u32 [538] dtypes.u32
540 Ops.AND dtypes.u32 [539, 108]
541 Ops.BITCAST dtypes.f32 [540] dtypes.f32
542 Ops.MUL dtypes.f32 [541, 34]
543 Ops.ADD dtypes.f32 [534, 542]
544 Ops.ADD dtypes.f32 [543, 537]
545 Ops.ADD dtypes.f32 [114, 534]
546 Ops.RECIP dtypes.f32 [545]
547 Ops.MUL dtypes.f32 [534, 546]
548 Ops.MUL dtypes.f32 [547, 547]
549 Ops.MUL dtypes.f32 [119, 548]
550 Ops.ADD dtypes.f32 [549, 121]
551 Ops.MUL dtypes.f32 [550, 548]
552 Ops.ADD dtypes.f32 [551, 124]
553 Ops.MUL dtypes.f32 [552, 548]
554 Ops.ADD dtypes.f32 [553, 127]
555 Ops.MUL dtypes.f32 [548, 554]
556 Ops.ADD dtypes.f32 [536, 555]
557 Ops.MUL dtypes.f32 [547, 556]
558 Ops.ADD dtypes.f32 [544, 557]
559 Ops.ADD dtypes.f32 [558, 541]
560 Ops.MUL dtypes.f32 [559, 134]
561 Ops.MUL dtypes.f32 [558, 136]
562 Ops.ADD dtypes.f32 [560, 561]
563 Ops.ADD dtypes.f32 [528, 140]
564 Ops.WHERE dtypes.f32 [519, 563, 528]
565 Ops.MUL dtypes.f32 [541, 136]
566 Ops.ADD dtypes.f32 [564, 565]
567 Ops.MUL dtypes.f32 [566, 34]
568 Ops.ADD dtypes.f32 [564, 567]
569 Ops.ADD dtypes.f32 [568, 565]
570 Ops.ADD dtypes.f32 [562, 569]
571 Ops.ADD dtypes.f32 [570, 566]
572 Ops.WHERE dtypes.f32 [518, 571, 15]
573 Ops.WHERE dtypes.f32 [516, 572, 151]
574 Ops.WHERE dtypes.f32 [514, 573, 151]
575 Ops.WHERE dtypes.f32 [511, 6, 574]
576 Ops.WHERE dtypes.f32 [510, 6, 575]
577 Ops.MUL dtypes.f32 [576, 29]
578 Ops.CMPEQ dtypes.bool [577, 577]
579 Ops.CMPEQ dtypes.bool [578, 3]
580 Ops.CMPLT dtypes.bool [577, 7]
581 Ops.CMPLT dtypes.bool [577, 11]
582 Ops.CMPEQ dtypes.bool [581, 3]
583 Ops.CMPEQ dtypes.bool [577, 14]
584 Ops.CMPEQ dtypes.bool [583, 3]
585 Ops.CMPEQ dtypes.bool [577, 19]
586 Ops.CMPEQ dtypes.bool [585, 3]
587 Ops.WHERE dtypes.f32 [586, 577, 10]
588 Ops.WHERE dtypes.f32 [579, 10, 587]
589 Ops.WHERE dtypes.f32 [584, 588, 10]
590 Ops.CMPLT dtypes.bool [589, 9]
591 Ops.WHERE dtypes.f32 [590, 27, 29]
592 Ops.ADD dtypes.f32 [589, 591]
593 Ops.CAST dtypes.i32 [592] dtypes.i32
594 Ops.CAST dtypes.f32 [593] dtypes.f32
595 Ops.MUL dtypes.f32 [594, 34]
596 Ops.ADD dtypes.f32 [589, 595]
597 Ops.MUL dtypes.f32 [16, 596]
598 Ops.ADD dtypes.f32 [597, 38]
599 Ops.MUL dtypes.f32 [598, 596]
600 Ops.ADD dtypes.f32 [599, 41]
601 Ops.MUL dtypes.f32 [600, 596]
602 Ops.ADD dtypes.f32 [601, 44]
603 Ops.MUL dtypes.f32 [602, 596]
604 Ops.ADD dtypes.f32 [603, 47]
605 Ops.MUL dtypes.f32 [604, 596]
606 Ops.ADD dtypes.f32 [605, 50]
607 Ops.MUL dtypes.f32 [606, 596]
608 Ops.ADD dtypes.f32 [607, 53]
609 Ops.DIV dtypes.i32 [593, 55]
610 Ops.ADD dtypes.i32 [609, 57]
611 Ops.MUL dtypes.i32 [610, 59]
612 Ops.BITCAST dtypes.f32 [611] dtypes.f32
613 Ops.MUL dtypes.f32 [608, 612]
614 Ops.MUL dtypes.i32 [609, 63]
615 Ops.ADD dtypes.i32 [593, 614]
616 Ops.ADD dtypes.i32 [615, 57]
617 Ops.MUL dtypes.i32 [616, 59]
618 Ops.BITCAST dtypes.f32 [617] dtypes.f32
619 Ops.MUL dtypes.f32 [613, 618]
620 Ops.WHERE dtypes.f32 [582, 15, 619]
621 Ops.WHERE dtypes.f32 [580, 10, 620]
622 Ops.WHERE dtypes.f32 [579, 6, 621]
623 Ops.WHERE dtypes.f32 [21, 6, 622]
624 Ops.CMPLT dtypes.bool [29, 9]
625 Ops.MUL dtypes.f32 [29, 34]
626 Ops.WHERE dtypes.f32 [624, 625, 29]
627 Ops.CAST dtypes.i32 [626] dtypes.i32
628 Ops.MOD dtypes.i32 [627, 55]
629 Ops.CAST dtypes.bool [628] dtypes.bool
630 Ops.MUL dtypes.f32 [622, 34]
631 Ops.WHERE dtypes.f32 [629, 630, 622]
632 Ops.WHERE dtypes.f32 [506, 623, 631]
633 Ops.WHERE dtypes.f32 [502, 632, 622]
634 Ops.WHERE dtypes.f32 [501, 159, 633]
635 Ops.SINK dtypes.void [72, 155, 500, 634]

