# tolk D31 D126 D413 golden=b56a4cdfe5f0f3ac9cd6ec5950f88435
0 Ops.STACK dtypes.void []
1 Ops.PARAM dtypes.fp8e4m3 [0] ParamArg(slot=0, dtype=dtypes.fp8e4m3)
2 Ops.CAST dtypes.f32 [1] dtypes.f32
3 Ops.CMPEQ dtypes.bool [2, 2]
4 Ops.CONST dtypes.bool [] False
5 Ops.CMPEQ dtypes.bool [3, 4]
6 Ops.CONST dtypes.weakfloat [] nan
7 Ops.CAST dtypes.f32 [6] dtypes.f32
8 Ops.CONST dtypes.weakfloat [] -150.0
9 Ops.CMPLT dtypes.bool [2, 8]
10 Ops.CONST dtypes.weakfloat [] 0.0
11 Ops.CAST dtypes.f32 [10] dtypes.f32
12 Ops.CONST dtypes.weakfloat [] 128.0
13 Ops.CMPLT dtypes.bool [2, 12]
14 Ops.CMPEQ dtypes.bool [13, 4]
15 Ops.CONST dtypes.weakfloat [] inf
16 Ops.CAST dtypes.f32 [15] dtypes.f32
17 Ops.CONST dtypes.weakfloat [] 0.0001535920892
18 Ops.CMPEQ dtypes.bool [2, 15]
19 Ops.CMPEQ dtypes.bool [18, 4]
20 Ops.CONST dtypes.weakfloat [] -inf
21 Ops.CMPEQ dtypes.bool [2, 20]
22 Ops.CMPEQ dtypes.bool [21, 4]
23 Ops.WHERE dtypes.f32 [22, 2, 11]
24 Ops.WHERE dtypes.f32 [5, 11, 23]
25 Ops.WHERE dtypes.f32 [19, 24, 11]
26 Ops.CMPLT dtypes.bool [25, 10]
27 Ops.CONST dtypes.weakfloat [] -0.5
28 Ops.CAST dtypes.f32 [27] dtypes.f32
29 Ops.CONST dtypes.weakfloat [] 0.5
30 Ops.CAST dtypes.f32 [29] dtypes.f32
31 Ops.WHERE dtypes.f32 [26, 28, 30]
32 Ops.ADD dtypes.f32 [25, 31]
33 Ops.CAST dtypes.i32 [32] dtypes.i32
34 Ops.CAST dtypes.f32 [33] dtypes.f32
35 Ops.CONST dtypes.weakfloat [] -1.0
36 Ops.MUL dtypes.f32 [34, 35]
37 Ops.ADD dtypes.f32 [25, 36]
38 Ops.MUL dtypes.f32 [17, 37]
39 Ops.CONST dtypes.weakfloat [] 0.001339262701
40 Ops.ADD dtypes.f32 [38, 39]
41 Ops.MUL dtypes.f32 [40, 37]
42 Ops.CONST dtypes.weakfloat [] 0.009618384764
43 Ops.ADD dtypes.f32 [41, 42]
44 Ops.MUL dtypes.f32 [43, 37]
45 Ops.CONST dtypes.weakfloat [] 0.05550347269
46 Ops.ADD dtypes.f32 [44, 45]
47 Ops.MUL dtypes.f32 [46, 37]
48 Ops.CONST dtypes.weakfloat [] 0.2402264476
49 Ops.ADD dtypes.f32 [47, 48]
50 Ops.MUL dtypes.f32 [49, 37]
51 Ops.CONST dtypes.weakfloat [] 0.6931471825
52 Ops.ADD dtypes.f32 [50, 51]
53 Ops.MUL dtypes.f32 [52, 37]
54 Ops.CONST dtypes.weakfloat [] 1.0
55 Ops.ADD dtypes.f32 [53, 54]
56 Ops.CONST dtypes.weakint [] 2
57 Ops.DIV dtypes.i32 [33, 56]
58 Ops.CONST dtypes.weakint [] 127
59 Ops.ADD dtypes.i32 [57, 58]
60 Ops.CONST dtypes.weakint [] 8388608
61 Ops.MUL dtypes.i32 [59, 60]
62 Ops.BITCAST dtypes.f32 [61] dtypes.f32
63 Ops.MUL dtypes.f32 [55, 62]
64 Ops.CONST dtypes.weakint [] -1
65 Ops.MUL dtypes.i32 [57, 64]
66 Ops.ADD dtypes.i32 [33, 65]
67 Ops.ADD dtypes.i32 [66, 58]
68 Ops.MUL dtypes.i32 [67, 60]
69 Ops.BITCAST dtypes.f32 [68] dtypes.f32
70 Ops.MUL dtypes.f32 [63, 69]
71 Ops.WHERE dtypes.f32 [14, 16, 70]
72 Ops.WHERE dtypes.f32 [9, 11, 71]
73 Ops.WHERE dtypes.f32 [5, 7, 72]
74 Ops.CAST dtypes.fp8e4m3 [73] dtypes.fp8e4m3
75 Ops.CONST dtypes.weakfloat [] -0.0
76 Ops.CMPLT dtypes.bool [2, 75]
77 Ops.RECIP dtypes.f32 [2]
78 Ops.CMPEQ dtypes.bool [77, 20]
79 Ops.CMPEQ dtypes.bool [78, 4]
80 Ops.CMPEQ dtypes.bool [2, 10]
81 Ops.CMPEQ dtypes.bool [80, 4]
82 Ops.CONST dtypes.weakfloat [] 1.1754943508222875e-38
83 Ops.CMPLT dtypes.bool [2, 82]
84 Ops.CONST dtypes.weakfloat [] 1.8446744073709552e+19
85 Ops.MUL dtypes.f32 [2, 84]
86 Ops.WHERE dtypes.f32 [83, 85, 2]
87 Ops.BITCAST dtypes.i32 [86] dtypes.i32
88 Ops.CONST dtypes.weakfloat [] 1.4142135623730951
89 Ops.MUL dtypes.f32 [86, 88]
90 Ops.BITCAST dtypes.i32 [89] dtypes.i32
91 Ops.DIV dtypes.i32 [90, 60]
92 Ops.CONST dtypes.weakint [] 255
93 Ops.AND dtypes.i32 [91, 92]
94 Ops.MUL dtypes.weakint [58, 64]
95 Ops.ADD dtypes.i32 [93, 94]
96 Ops.CAST dtypes.f32 [95] dtypes.f32
97 Ops.MUL dtypes.f32 [96, 35]
98 Ops.CAST dtypes.i32 [97] dtypes.i32
99 Ops.MUL dtypes.i32 [98, 60]
100 Ops.ADD dtypes.i32 [87, 99]
101 Ops.BITCAST dtypes.f32 [100] dtypes.f32
102 Ops.MUL dtypes.weakfloat [54, 35]
103 Ops.ADD dtypes.f32 [101, 102]
104 Ops.MUL dtypes.f32 [29, 103]
105 Ops.MUL dtypes.f32 [104, 103]
106 Ops.MUL dtypes.f32 [105, 35]
107 Ops.ADD dtypes.f32 [103, 106]
108 Ops.BITCAST dtypes.u32 [107] dtypes.u32
109 Ops.CONST dtypes.weakint [] 4294963200
110 Ops.CAST dtypes.u32 [109] dtypes.u32
111 Ops.AND dtypes.u32 [108, 110]
112 Ops.BITCAST dtypes.f32 [111] dtypes.f32
113 Ops.MUL dtypes.f32 [112, 35]
114 Ops.ADD dtypes.f32 [103, 113]
115 Ops.ADD dtypes.f32 [114, 106]
116 Ops.CONST dtypes.weakfloat [] 2.0
117 Ops.ADD dtypes.f32 [116, 103]
118 Ops.RECIP dtypes.f32 [117]
119 Ops.MUL dtypes.f32 [103, 118]
120 Ops.MUL dtypes.f32 [119, 119]
121 Ops.CONST dtypes.weakfloat [] 0.24279078841209412
122 Ops.MUL dtypes.f32 [121, 120]
123 Ops.CONST dtypes.weakfloat [] 0.2849878668785095
124 Ops.ADD dtypes.f32 [122, 123]
125 Ops.MUL dtypes.f32 [124, 120]
126 Ops.CONST dtypes.weakfloat [] 0.40000972151756287
127 Ops.ADD dtypes.f32 [125, 126]
128 Ops.MUL dtypes.f32 [127, 120]
129 Ops.CONST dtypes.weakfloat [] 0.6666666269302368
130 Ops.ADD dtypes.f32 [128, 129]
131 Ops.MUL dtypes.f32 [120, 130]
132 Ops.ADD dtypes.f32 [105, 131]
133 Ops.MUL dtypes.f32 [119, 132]
134 Ops.ADD dtypes.f32 [115, 133]
135 Ops.ADD dtypes.f32 [134, 112]
136 Ops.CONST dtypes.weakfloat [] -0.00017605285393
137 Ops.MUL dtypes.f32 [135, 136]
138 Ops.CONST dtypes.weakfloat [] 1.4428710938
139 Ops.MUL dtypes.f32 [134, 138]
140 Ops.ADD dtypes.f32 [137, 139]
141 Ops.CONST dtypes.weakfloat [] 64.0
142 Ops.MUL dtypes.weakfloat [141, 35]
143 Ops.ADD dtypes.f32 [96, 142]
144 Ops.WHERE dtypes.f32 [83, 143, 96]
145 Ops.MUL dtypes.f32 [112, 138]
146 Ops.ADD dtypes.f32 [144, 145]
147 Ops.MUL dtypes.f32 [146, 35]
148 Ops.ADD dtypes.f32 [144, 147]
149 Ops.ADD dtypes.f32 [148, 145]
150 Ops.ADD dtypes.f32 [140, 149]
151 Ops.ADD dtypes.f32 [150, 146]
152 Ops.WHERE dtypes.f32 [19, 151, 16]
153 Ops.CAST dtypes.f32 [20] dtypes.f32
154 Ops.WHERE dtypes.f32 [81, 152, 153]
155 Ops.WHERE dtypes.f32 [79, 154, 153]
156 Ops.WHERE dtypes.f32 [76, 7, 155]
157 Ops.WHERE dtypes.f32 [5, 7, 156]
158 Ops.CAST dtypes.fp8e4m3 [157] dtypes.fp8e4m3
159 Ops.CMPEQ dtypes.bool [25, 10]
160 Ops.CMPEQ dtypes.bool [159, 4]
161 Ops.CAST dtypes.f32 [35] dtypes.f32
162 Ops.CAST dtypes.f32 [54] dtypes.f32
163 Ops.WHERE dtypes.f32 [26, 161, 162]
164 Ops.MUL dtypes.f32 [25, 163]
165 Ops.CONST dtypes.weakfloat [] 30.0
166 Ops.CMPLT dtypes.bool [164, 165]
167 Ops.CONST dtypes.weakfloat [] 0.6366197723675814
168 Ops.MUL dtypes.f32 [164, 167]
169 Ops.CMPLT dtypes.bool [168, 10]
170 Ops.WHERE dtypes.f32 [169, 28, 30]
171 Ops.ADD dtypes.f32 [168, 170]
172 Ops.CAST dtypes.i32 [171] dtypes.i32
173 Ops.BITCAST dtypes.u32 [164] dtypes.u32
174 Ops.CONST dtypes.weakint [] 2155872255
175 Ops.AND dtypes.u32 [173, 174]
176 Ops.CONST dtypes.weakint [] 1056964608
177 Ops.OR dtypes.u32 [175, 176]
178 Ops.BITCAST dtypes.f32 [177] dtypes.f32
179 Ops.CONST dtypes.weakfloat [] 16777216.0
180 Ops.MUL dtypes.f32 [178, 179]
181 Ops.CAST dtypes.u64 [180] dtypes.u64
182 Ops.CONST dtypes.weakint [] 4294967295
183 Ops.AND dtypes.u64 [181, 182]
184 Ops.DIV dtypes.u32 [173, 60]
185 Ops.AND dtypes.u32 [184, 92]
186 Ops.ADD dtypes.u32 [185, 94]
187 Ops.CONST dtypes.weakint [] 1
188 Ops.ADD dtypes.u32 [186, 187]
189 Ops.CAST dtypes.i32 [188] dtypes.i32
190 Ops.CONST dtypes.weakint [] 104
191 Ops.ADD dtypes.i32 [189, 190]
192 Ops.CONST dtypes.weakint [] 0
193 Ops.MAX dtypes.i32 [191, 192]
194 Ops.CONST dtypes.weakint [] 32
195 Ops.DIV dtypes.i32 [193, 194]
196 Ops.AND dtypes.i32 [195, 187]
197 Ops.CMPEQ dtypes.bool [196, 192]
198 Ops.CMPEQ dtypes.bool [197, 4]
199 Ops.AND dtypes.i32 [195, 56]
200 Ops.CMPEQ dtypes.bool [199, 192]
201 Ops.CMPEQ dtypes.bool [200, 4]
202 Ops.CONST dtypes.weakint [] 4
203 Ops.AND dtypes.i32 [195, 202]
204 Ops.CMPEQ dtypes.bool [203, 192]
205 Ops.CMPEQ dtypes.bool [204, 4]
206 Ops.CONST dtypes.weakint [] 2102212464
207 Ops.CAST dtypes.u64 [206] dtypes.u64
208 Ops.CAST dtypes.u64 [192] dtypes.u64
209 Ops.WHERE dtypes.u64 [205, 207, 208]
210 Ops.CONST dtypes.weakint [] 2475754826
211 Ops.CAST dtypes.u64 [210] dtypes.u64
212 Ops.WHERE dtypes.u64 [205, 211, 208]
213 Ops.WHERE dtypes.u64 [201, 209, 212]
214 Ops.CONST dtypes.weakint [] 2131351028
215 Ops.CAST dtypes.u64 [214] dtypes.u64
216 Ops.WHERE dtypes.u64 [205, 215, 208]
217 Ops.CONST dtypes.weakint [] 683565275
218 Ops.CAST dtypes.u64 [217] dtypes.u64
219 Ops.WHERE dtypes.u64 [205, 218, 208]
220 Ops.WHERE dtypes.u64 [201, 216, 219]
221 Ops.WHERE dtypes.u64 [198, 213, 220]
222 Ops.CONST dtypes.weakint [] 31
223 Ops.AND dtypes.i32 [193, 222]
224 Ops.CAST dtypes.u64 [223] dtypes.u64
225 Ops.SHL dtypes.u64 [221, 224]
226 Ops.CONST dtypes.weakint [] 920167782
227 Ops.CAST dtypes.u64 [226] dtypes.u64
228 Ops.WHERE dtypes.u64 [205, 227, 218]
229 Ops.WHERE dtypes.u64 [201, 228, 216]
230 Ops.WHERE dtypes.u64 [198, 229, 213]
231 Ops.CONST dtypes.weakint [] 18446744073709551615
232 Ops.MUL dtypes.u64 [224, 231]
233 Ops.ADD dtypes.u64 [194, 232]
234 Ops.SHR dtypes.u64 [230, 233]
235 Ops.OR dtypes.u64 [225, 234]
236 Ops.AND dtypes.u64 [235, 182]
237 Ops.MUL dtypes.u64 [183, 236]
238 Ops.AND dtypes.u64 [237, 182]
239 Ops.SHL dtypes.u64 [230, 224]
240 Ops.CONST dtypes.weakint [] 1326507024
241 Ops.CAST dtypes.u64 [240] dtypes.u64
242 Ops.WHERE dtypes.u64 [205, 241, 211]
243 Ops.WHERE dtypes.u64 [201, 242, 209]
244 Ops.WHERE dtypes.u64 [198, 243, 229]
245 Ops.SHR dtypes.u64 [244, 233]
246 Ops.OR dtypes.u64 [239, 245]
247 Ops.AND dtypes.u64 [246, 182]
248 Ops.MUL dtypes.u64 [183, 247]
249 Ops.CONST dtypes.weakint [] 4294967296
250 Ops.DIV dtypes.u64 [248, 249]
251 Ops.ADD dtypes.u64 [238, 250]
252 Ops.AND dtypes.u64 [248, 182]
253 Ops.SHL dtypes.u64 [244, 224]
254 Ops.CONST dtypes.weakint [] 2140428522
255 Ops.CAST dtypes.u64 [254] dtypes.u64
256 Ops.WHERE dtypes.u64 [205, 255, 215]
257 Ops.WHERE dtypes.u64 [201, 256, 228]
258 Ops.WHERE dtypes.u64 [198, 257, 243]
259 Ops.SHR dtypes.u64 [258, 233]
260 Ops.OR dtypes.u64 [253, 259]
261 Ops.AND dtypes.u64 [260, 182]
262 Ops.MUL dtypes.u64 [183, 261]
263 Ops.DIV dtypes.u64 [262, 249]
264 Ops.ADD dtypes.u64 [252, 263]
265 Ops.AND dtypes.u64 [262, 182]
266 Ops.SHL dtypes.u64 [258, 224]
267 Ops.CONST dtypes.weakint [] 4155437400
268 Ops.CAST dtypes.u64 [267] dtypes.u64
269 Ops.WHERE dtypes.u64 [205, 268, 207]
270 Ops.WHERE dtypes.u64 [201, 269, 242]
271 Ops.WHERE dtypes.u64 [198, 270, 257]
272 Ops.SHR dtypes.u64 [271, 233]
273 Ops.OR dtypes.u64 [266, 272]
274 Ops.AND dtypes.u64 [273, 182]
275 Ops.MUL dtypes.u64 [183, 274]
276 Ops.DIV dtypes.u64 [275, 249]
277 Ops.ADD dtypes.u64 [265, 276]
278 Ops.AND dtypes.u64 [275, 182]
279 Ops.SHL dtypes.u64 [271, 224]
280 Ops.CONST dtypes.weakint [] 1841896334
281 Ops.CAST dtypes.u64 [280] dtypes.u64
282 Ops.WHERE dtypes.u64 [205, 281, 227]
283 Ops.WHERE dtypes.u64 [201, 282, 256]
284 Ops.WHERE dtypes.u64 [198, 283, 270]
285 Ops.SHR dtypes.u64 [284, 233]
286 Ops.OR dtypes.u64 [279, 285]
287 Ops.AND dtypes.u64 [286, 182]
288 Ops.MUL dtypes.u64 [183, 287]
289 Ops.DIV dtypes.u64 [288, 249]
290 Ops.ADD dtypes.u64 [278, 289]
291 Ops.AND dtypes.u64 [288, 182]
292 Ops.SHL dtypes.u64 [284, 224]
293 Ops.CONST dtypes.weakint [] 2425582776
294 Ops.CAST dtypes.u64 [293] dtypes.u64
295 Ops.WHERE dtypes.u64 [205, 294, 241]
296 Ops.WHERE dtypes.u64 [201, 295, 269]
297 Ops.WHERE dtypes.u64 [198, 296, 283]
298 Ops.SHR dtypes.u64 [297, 233]
299 Ops.OR dtypes.u64 [292, 298]
300 Ops.AND dtypes.u64 [299, 182]
301 Ops.MUL dtypes.u64 [183, 300]
302 Ops.DIV dtypes.u64 [301, 249]
303 Ops.ADD dtypes.u64 [291, 302]
304 Ops.AND dtypes.u64 [301, 182]
305 Ops.SHL dtypes.u64 [297, 224]
306 Ops.CONST dtypes.weakint [] 26364858
307 Ops.CAST dtypes.u64 [306] dtypes.u64
308 Ops.WHERE dtypes.u64 [205, 307, 255]
309 Ops.WHERE dtypes.u64 [201, 308, 282]
310 Ops.WHERE dtypes.u64 [198, 309, 296]
311 Ops.SHR dtypes.u64 [310, 233]
312 Ops.OR dtypes.u64 [305, 311]
313 Ops.AND dtypes.u64 [312, 182]
314 Ops.MUL dtypes.u64 [183, 313]
315 Ops.DIV dtypes.u64 [314, 249]
316 Ops.ADD dtypes.u64 [304, 315]
317 Ops.AND dtypes.u64 [314, 182]
318 Ops.DIV dtypes.u64 [317, 249]
319 Ops.ADD dtypes.u64 [316, 318]
320 Ops.DIV dtypes.u64 [319, 249]
321 Ops.ADD dtypes.u64 [303, 320]
322 Ops.DIV dtypes.u64 [321, 249]
323 Ops.ADD dtypes.u64 [290, 322]
324 Ops.DIV dtypes.u64 [323, 249]
325 Ops.ADD dtypes.u64 [277, 324]
326 Ops.DIV dtypes.u64 [325, 249]
327 Ops.ADD dtypes.u64 [264, 326]
328 Ops.DIV dtypes.u64 [327, 249]
329 Ops.ADD dtypes.u64 [251, 328]
330 Ops.AND dtypes.u64 [329, 182]
331 Ops.MUL dtypes.u64 [330, 249]
332 Ops.AND dtypes.u64 [327, 182]
333 Ops.OR dtypes.u64 [331, 332]
334 Ops.CONST dtypes.weakint [] 4611686018427387904
335 Ops.DIV dtypes.u64 [333, 334]
336 Ops.CONST dtypes.weakint [] 2305843009213693952
337 Ops.DIV dtypes.u64 [333, 336]
338 Ops.AND dtypes.u64 [337, 187]
339 Ops.ADD dtypes.u64 [335, 338]
340 Ops.CAST dtypes.i32 [339] dtypes.i32
341 Ops.WHERE dtypes.i32 [166, 172, 340]
342 Ops.AND dtypes.i32 [341, 56]
343 Ops.CMPEQ dtypes.bool [342, 192]
344 Ops.CMPEQ dtypes.bool [343, 4]
345 Ops.AND dtypes.i32 [341, 187]
346 Ops.CMPEQ dtypes.bool [345, 192]
347 Ops.CMPEQ dtypes.bool [346, 4]
348 Ops.CAST dtypes.f32 [172] dtypes.f32
349 Ops.CONST dtypes.weakfloat [] -0.00048351287841796875
350 Ops.MUL dtypes.f32 [348, 349]
351 Ops.CONST dtypes.weakfloat [] -1.5703125
352 Ops.MUL dtypes.f32 [348, 351]
353 Ops.ADD dtypes.f32 [352, 164]
354 Ops.ADD dtypes.f32 [350, 353]
355 Ops.CONST dtypes.weakfloat [] -3.1385570764541626e-07
356 Ops.MUL dtypes.f32 [348, 355]
357 Ops.ADD dtypes.f32 [354, 356]
358 Ops.MUL dtypes.f32 [354, 35]
359 Ops.ADD dtypes.f32 [357, 358]
360 Ops.MUL dtypes.f32 [359, 35]
361 Ops.ADD dtypes.f32 [356, 360]
362 Ops.CONST dtypes.weakfloat [] -6.07710062827671e-11
363 Ops.MUL dtypes.f32 [348, 362]
364 Ops.ADD dtypes.f32 [361, 363]
365 Ops.ADD dtypes.f32 [357, 364]
366 Ops.CONST dtypes.weakint [] 4611686018427387903
367 Ops.AND dtypes.u64 [333, 366]
368 Ops.CAST dtypes.i64 [367] dtypes.i64
369 Ops.CAST dtypes.i64 [338] dtypes.i64
370 Ops.MUL dtypes.i64 [369, 334]
371 Ops.MUL dtypes.i64 [370, 64]
372 Ops.ADD dtypes.i64 [368, 371]
373 Ops.CONST dtypes.weakint [] 4398046511104
374 Ops.DIV dtypes.i64 [372, 373]
375 Ops.CAST dtypes.f32 [374] dtypes.f32
376 Ops.CONST dtypes.weakfloat [] 4398046511104.0
377 Ops.MUL dtypes.f32 [375, 376]
378 Ops.CONST dtypes.weakint [] 2097152
379 Ops.DIV dtypes.i64 [372, 378]
380 Ops.CONST dtypes.weakint [] 2097151
381 Ops.AND dtypes.i64 [379, 380]
382 Ops.CAST dtypes.f32 [381] dtypes.f32
383 Ops.CONST dtypes.weakfloat [] 2097152.0
384 Ops.MUL dtypes.f32 [382, 383]
385 Ops.ADD dtypes.f32 [377, 384]
386 Ops.AND dtypes.i64 [372, 380]
387 Ops.CAST dtypes.f32 [386] dtypes.f32
388 Ops.MUL dtypes.f32 [387, 54]
389 Ops.ADD dtypes.f32 [385, 388]
390 Ops.AND dtypes.u64 [325, 182]
391 Ops.MUL dtypes.u64 [390, 249]
392 Ops.AND dtypes.u64 [323, 182]
393 Ops.OR dtypes.u64 [391, 392]
394 Ops.CONST dtypes.weakint [] 8796093022208
395 Ops.DIV dtypes.u64 [393, 394]
396 Ops.CAST dtypes.f32 [395] dtypes.f32
397 Ops.CONST dtypes.weakfloat [] 1.1368683772161603e-13
398 Ops.MUL dtypes.f32 [396, 397]
399 Ops.ADD dtypes.f32 [389, 398]
400 Ops.CONST dtypes.weakfloat [] 3.4061216748705226e-19
401 Ops.MUL dtypes.f32 [399, 400]
402 Ops.CONST dtypes.weakfloat [] 4097.0
403 Ops.MUL dtypes.f32 [399, 402]
404 Ops.MUL dtypes.f32 [399, 35]
405 Ops.ADD dtypes.f32 [403, 404]
406 Ops.MUL dtypes.f32 [405, 35]
407 Ops.ADD dtypes.f32 [403, 406]
408 Ops.CONST dtypes.weakfloat [] 3.406131239146355e-19
409 Ops.MUL dtypes.f32 [407, 408]
410 Ops.MUL dtypes.f32 [401, 35]
411 Ops.ADD dtypes.f32 [409, 410]
412 Ops.CONST dtypes.weakfloat [] -9.564275832644382e-25
413 Ops.MUL dtypes.f32 [407, 412]
414 Ops.ADD dtypes.f32 [411, 413]
415 Ops.MUL dtypes.f32 [407, 35]
416 Ops.ADD dtypes.f32 [399, 415]
417 Ops.MUL dtypes.f32 [416, 408]
418 Ops.ADD dtypes.f32 [414, 417]
419 Ops.MUL dtypes.f32 [416, 412]
420 Ops.ADD dtypes.f32 [418, 419]
421 Ops.CONST dtypes.weakfloat [] -9.47839680047608e-27
422 Ops.MUL dtypes.f32 [399, 421]
423 Ops.ADD dtypes.f32 [420, 422]
424 Ops.MUL dtypes.f32 [377, 35]
425 Ops.ADD dtypes.f32 [385, 424]
426 Ops.MUL dtypes.f32 [425, 35]
427 Ops.ADD dtypes.f32 [384, 426]
428 Ops.ADD dtypes.f32 [11, 427]
429 Ops.MUL dtypes.f32 [385, 35]
430 Ops.ADD dtypes.f32 [389, 429]
431 Ops.MUL dtypes.f32 [430, 35]
432 Ops.ADD dtypes.f32 [388, 431]
433 Ops.ADD dtypes.f32 [428, 432]
434 Ops.MUL dtypes.f32 [389, 35]
435 Ops.ADD dtypes.f32 [399, 434]
436 Ops.MUL dtypes.f32 [435, 35]
437 Ops.ADD dtypes.f32 [398, 436]
438 Ops.ADD dtypes.f32 [433, 437]
439 Ops.MUL dtypes.f32 [438, 400]
440 Ops.ADD dtypes.f32 [423, 439]
441 Ops.ADD dtypes.f32 [401, 440]
442 Ops.WHERE dtypes.f32 [166, 365, 441]
443 Ops.MUL dtypes.f32 [442, 442]
444 Ops.MUL dtypes.f32 [29, 443]
445 Ops.MUL dtypes.f32 [444, 35]
446 Ops.ADD dtypes.f32 [54, 445]
447 Ops.MUL dtypes.f32 [446, 35]
448 Ops.ADD dtypes.f32 [54, 447]
449 Ops.ADD dtypes.f32 [448, 445]
450 Ops.MUL dtypes.f32 [443, 443]
451 Ops.CONST dtypes.weakfloat [] -2.7557314351390663e-07
452 Ops.MUL dtypes.f32 [451, 443]
453 Ops.CONST dtypes.weakfloat [] 2.480158728947673e-05
454 Ops.ADD dtypes.f32 [452, 453]
455 Ops.MUL dtypes.f32 [454, 443]
456 Ops.CONST dtypes.weakfloat [] -0.001388888888887411
457 Ops.ADD dtypes.f32 [455, 456]
458 Ops.MUL dtypes.f32 [457, 443]
459 Ops.CONST dtypes.weakfloat [] 0.0416666666666666
460 Ops.ADD dtypes.f32 [458, 459]
461 Ops.MUL dtypes.f32 [450, 460]
462 Ops.MUL dtypes.f32 [357, 35]
463 Ops.ADD dtypes.f32 [365, 462]
464 Ops.MUL dtypes.f32 [463, 35]
465 Ops.ADD dtypes.f32 [364, 464]
466 Ops.ADD dtypes.f32 [441, 410]
467 Ops.MUL dtypes.f32 [466, 35]
468 Ops.ADD dtypes.f32 [440, 467]
469 Ops.WHERE dtypes.f32 [166, 465, 468]
470 Ops.MUL dtypes.f32 [442, 469]
471 Ops.MUL dtypes.f32 [470, 35]
472 Ops.ADD dtypes.f32 [461, 471]
473 Ops.ADD dtypes.f32 [449, 472]
474 Ops.ADD dtypes.f32 [446, 473]
475 Ops.MUL dtypes.f32 [29, 469]
476 Ops.MUL dtypes.f32 [443, 442]
477 Ops.CONST dtypes.weakfloat [] 2.7557313707070068e-06
478 Ops.MUL dtypes.f32 [477, 443]
479 Ops.CONST dtypes.weakfloat [] -0.0001984126982985795
480 Ops.ADD dtypes.f32 [478, 479]
481 Ops.MUL dtypes.f32 [480, 443]
482 Ops.CONST dtypes.weakfloat [] 0.00833333333332249
483 Ops.ADD dtypes.f32 [481, 482]
484 Ops.MUL dtypes.f32 [476, 483]
485 Ops.MUL dtypes.f32 [484, 35]
486 Ops.ADD dtypes.f32 [475, 485]
487 Ops.MUL dtypes.f32 [443, 486]
488 Ops.MUL dtypes.f32 [469, 35]
489 Ops.ADD dtypes.f32 [487, 488]
490 Ops.CONST dtypes.weakfloat [] -0.16666666666666632
491 Ops.MUL dtypes.f32 [476, 490]
492 Ops.MUL dtypes.f32 [491, 35]
493 Ops.ADD dtypes.f32 [489, 492]
494 Ops.MUL dtypes.f32 [493, 35]
495 Ops.ADD dtypes.f32 [442, 494]
496 Ops.WHERE dtypes.f32 [347, 474, 495]
497 Ops.MUL dtypes.f32 [496, 163]
498 Ops.MUL dtypes.f32 [497, 35]
499 Ops.WHERE dtypes.f32 [344, 498, 497]
500 Ops.WHERE dtypes.f32 [160, 499, 25]
501 Ops.WHERE dtypes.f32 [22, 500, 7]
502 Ops.WHERE dtypes.f32 [5, 7, 501]
503 Ops.WHERE dtypes.f32 [19, 502, 7]
504 Ops.CAST dtypes.fp8e4m3 [503] dtypes.fp8e4m3
505 Ops.CAST dtypes.fp8e4m3 [29] dtypes.fp8e4m3
506 Ops.CMPEQ dtypes.bool [505, 10]
507 Ops.CAST dtypes.fp8e4m3 [54] dtypes.fp8e4m3
508 Ops.CMPLT dtypes.bool [1, 10]
509 Ops.CAST dtypes.i32 [505] dtypes.i32
510 Ops.CAST dtypes.fp8e4m3 [509] dtypes.fp8e4m3
511 Ops.CMPEQ dtypes.bool [505, 510]
512 Ops.CMPEQ dtypes.bool [511, 4]
513 Ops.CMPEQ dtypes.bool [1, 20]
514 Ops.CMPEQ dtypes.bool [513, 4]
515 Ops.CAST dtypes.fp8e4m3 [6] dtypes.fp8e4m3
516 Ops.MUL dtypes.fp8e4m3 [1, 35]
517 Ops.WHERE dtypes.fp8e4m3 [508, 516, 1]
518 Ops.CAST dtypes.f32 [517] dtypes.f32
519 Ops.CMPEQ dtypes.bool [518, 518]
520 Ops.CMPEQ dtypes.bool [519, 4]
521 Ops.CMPLT dtypes.bool [518, 75]
522 Ops.RECIP dtypes.f32 [518]
523 Ops.CMPEQ dtypes.bool [522, 20]
524 Ops.CMPEQ dtypes.bool [523, 4]
525 Ops.CMPEQ dtypes.bool [518, 10]
526 Ops.CMPEQ dtypes.bool [525, 4]
527 Ops.CMPEQ dtypes.bool [518, 15]
528 Ops.CMPEQ dtypes.bool [527, 4]
529 Ops.CMPLT dtypes.bool [518, 82]
530 Ops.MUL dtypes.f32 [518, 84]
531 Ops.WHERE dtypes.f32 [529, 530, 518]
532 Ops.BITCAST dtypes.i32 [531] dtypes.i32
533 Ops.MUL dtypes.f32 [531, 88]
534 Ops.BITCAST dtypes.i32 [533] dtypes.i32
535 Ops.DIV dtypes.i32 [534, 60]
536 Ops.AND dtypes.i32 [535, 92]
537 Ops.ADD dtypes.i32 [536, 94]
538 Ops.CAST dtypes.f32 [537] dtypes.f32
539 Ops.MUL dtypes.f32 [538, 35]
540 Ops.CAST dtypes.i32 [539] dtypes.i32
541 Ops.MUL dtypes.i32 [540, 60]
542 Ops.ADD dtypes.i32 [532, 541]
543 Ops.BITCAST dtypes.f32 [542] dtypes.f32
544 Ops.ADD dtypes.f32 [543, 102]
545 Ops.MUL dtypes.f32 [29, 544]
546 Ops.MUL dtypes.f32 [545, 544]
547 Ops.MUL dtypes.f32 [546, 35]
548 Ops.ADD dtypes.f32 [544, 547]
549 Ops.BITCAST dtypes.u32 [548] dtypes.u32
550 Ops.AND dtypes.u32 [549, 110]
551 Ops.BITCAST dtypes.f32 [550] dtypes.f32
552 Ops.MUL dtypes.f32 [551, 35]
553 Ops.ADD dtypes.f32 [544, 552]
554 Ops.ADD dtypes.f32 [553, 547]
555 Ops.ADD dtypes.f32 [116, 544]
556 Ops.RECIP dtypes.f32 [555]
557 Ops.MUL dtypes.f32 [544, 556]
558 Ops.MUL dtypes.f32 [557, 557]
559 Ops.MUL dtypes.f32 [121, 558]
560 Ops.ADD dtypes.f32 [559, 123]
561 Ops.MUL dtypes.f32 [560, 558]
562 Ops.ADD dtypes.f32 [561, 126]
563 Ops.MUL dtypes.f32 [562, 558]
564 Ops.ADD dtypes.f32 [563, 129]
565 Ops.MUL dtypes.f32 [558, 564]
566 Ops.ADD dtypes.f32 [546, 565]
567 Ops.MUL dtypes.f32 [557, 566]
568 Ops.ADD dtypes.f32 [554, 567]
569 Ops.ADD dtypes.f32 [568, 551]
570 Ops.MUL dtypes.f32 [569, 136]
571 Ops.MUL dtypes.f32 [568, 138]
572 Ops.ADD dtypes.f32 [570, 571]
573 Ops.ADD dtypes.f32 [538, 142]
574 Ops.WHERE dtypes.f32 [529, 573, 538]
575 Ops.MUL dtypes.f32 [551, 138]
576 Ops.ADD dtypes.f32 [574, 575]
577 Ops.MUL dtypes.f32 [576, 35]
578 Ops.ADD dtypes.f32 [574, 577]
579 Ops.ADD dtypes.f32 [578, 575]
580 Ops.ADD dtypes.f32 [572, 579]
581 Ops.ADD dtypes.f32 [580, 576]
582 Ops.WHERE dtypes.f32 [528, 581, 16]
583 Ops.WHERE dtypes.f32 [526, 582, 153]
584 Ops.WHERE dtypes.f32 [524, 583, 153]
585 Ops.WHERE dtypes.f32 [521, 7, 584]
586 Ops.WHERE dtypes.f32 [520, 7, 585]
587 Ops.CAST dtypes.fp8e4m3 [586] dtypes.fp8e4m3
588 Ops.MUL dtypes.fp8e4m3 [587, 505]
589 Ops.CAST dtypes.f32 [588] dtypes.f32
590 Ops.CMPEQ dtypes.bool [589, 589]
591 Ops.CMPEQ dtypes.bool [590, 4]
592 Ops.CMPLT dtypes.bool [589, 8]
593 Ops.CMPLT dtypes.bool [589, 12]
594 Ops.CMPEQ dtypes.bool [593, 4]
595 Ops.CMPEQ dtypes.bool [589, 15]
596 Ops.CMPEQ dtypes.bool [595, 4]
597 Ops.CMPEQ dtypes.bool [589, 20]
598 Ops.CMPEQ dtypes.bool [597, 4]
599 Ops.WHERE dtypes.f32 [598, 589, 11]
600 Ops.WHERE dtypes.f32 [591, 11, 599]
601 Ops.WHERE dtypes.f32 [596, 600, 11]
602 Ops.CMPLT dtypes.bool [601, 10]
603 Ops.WHERE dtypes.f32 [602, 28, 30]
604 Ops.ADD dtypes.f32 [601, 603]
605 Ops.CAST dtypes.i32 [604] dtypes.i32
606 Ops.CAST dtypes.f32 [605] dtypes.f32
607 Ops.MUL dtypes.f32 [606, 35]
608 Ops.ADD dtypes.f32 [601, 607]
609 Ops.MUL dtypes.f32 [17, 608]
610 Ops.ADD dtypes.f32 [609, 39]
611 Ops.MUL dtypes.f32 [610, 608]
612 Ops.ADD dtypes.f32 [611, 42]
613 Ops.MUL dtypes.f32 [612, 608]
614 Ops.ADD dtypes.f32 [613, 45]
615 Ops.MUL dtypes.f32 [614, 608]
616 Ops.ADD dtypes.f32 [615, 48]
617 Ops.MUL dtypes.f32 [616, 608]
618 Ops.ADD dtypes.f32 [617, 51]
619 Ops.MUL dtypes.f32 [618, 608]
620 Ops.ADD dtypes.f32 [619, 54]
621 Ops.DIV dtypes.i32 [605, 56]
622 Ops.ADD dtypes.i32 [621, 58]
623 Ops.MUL dtypes.i32 [622, 60]
624 Ops.BITCAST dtypes.f32 [623] dtypes.f32
625 Ops.MUL dtypes.f32 [620, 624]
626 Ops.MUL dtypes.i32 [621, 64]
627 Ops.ADD dtypes.i32 [605, 626]
628 Ops.ADD dtypes.i32 [627, 58]
629 Ops.MUL dtypes.i32 [628, 60]
630 Ops.BITCAST dtypes.f32 [629] dtypes.f32
631 Ops.MUL dtypes.f32 [625, 630]
632 Ops.WHERE dtypes.f32 [594, 16, 631]
633 Ops.WHERE dtypes.f32 [592, 11, 632]
634 Ops.WHERE dtypes.f32 [591, 7, 633]
635 Ops.CAST dtypes.fp8e4m3 [634] dtypes.fp8e4m3
636 Ops.WHERE dtypes.fp8e4m3 [514, 515, 635]
637 Ops.CMPLT dtypes.bool [505, 10]
638 Ops.MUL dtypes.fp8e4m3 [505, 35]
639 Ops.WHERE dtypes.fp8e4m3 [637, 638, 505]
640 Ops.CAST dtypes.i32 [639] dtypes.i32
641 Ops.MOD dtypes.i32 [640, 56]
642 Ops.CAST dtypes.bool [641] dtypes.bool
643 Ops.MUL dtypes.fp8e4m3 [635, 35]
644 Ops.WHERE dtypes.fp8e4m3 [642, 643, 635]
645 Ops.WHERE dtypes.fp8e4m3 [512, 636, 644]
646 Ops.WHERE dtypes.fp8e4m3 [508, 645, 635]
647 Ops.WHERE dtypes.fp8e4m3 [506, 507, 646]
648 Ops.SINK dtypes.void [74, 158, 504, 647]

