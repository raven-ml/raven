# tolk D31 D126 D413 golden=0c639216faca3436b6fdec34861ad18f
0 Ops.STACK dtypes.void []
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.EXP2 dtypes.f32 [1]
3 Ops.LOG2 dtypes.f32 [1]
4 Ops.CONST dtypes.weakfloat [] inf
5 Ops.CMPEQ dtypes.bool [1, 4]
6 Ops.CONST dtypes.bool [] False
7 Ops.CMPEQ dtypes.bool [5, 6]
8 Ops.CMPEQ dtypes.bool [1, 1]
9 Ops.CMPEQ dtypes.bool [8, 6]
10 Ops.CONST dtypes.weakfloat [] nan
11 Ops.CAST dtypes.f32 [10] dtypes.f32
12 Ops.CONST dtypes.weakfloat [] -inf
13 Ops.CMPEQ dtypes.bool [1, 12]
14 Ops.CMPEQ dtypes.bool [13, 6]
15 Ops.CONST dtypes.weakfloat [] 0.0
16 Ops.CAST dtypes.f32 [15] dtypes.f32
17 Ops.WHERE dtypes.f32 [14, 1, 16]
18 Ops.WHERE dtypes.f32 [9, 16, 17]
19 Ops.WHERE dtypes.f32 [7, 18, 16]
20 Ops.CMPEQ dtypes.bool [19, 15]
21 Ops.CMPEQ dtypes.bool [20, 6]
22 Ops.CMPLT dtypes.bool [19, 15]
23 Ops.CONST dtypes.weakfloat [] -1.0
24 Ops.CAST dtypes.f32 [23] dtypes.f32
25 Ops.CONST dtypes.weakfloat [] 1.0
26 Ops.CAST dtypes.f32 [25] dtypes.f32
27 Ops.WHERE dtypes.f32 [22, 24, 26]
28 Ops.MUL dtypes.f32 [19, 27]
29 Ops.CONST dtypes.weakfloat [] 30.0
30 Ops.CMPLT dtypes.bool [28, 29]
31 Ops.CONST dtypes.weakfloat [] 0.6366197723675814
32 Ops.MUL dtypes.f32 [28, 31]
33 Ops.CMPLT dtypes.bool [32, 15]
34 Ops.CONST dtypes.weakfloat [] -0.5
35 Ops.CAST dtypes.f32 [34] dtypes.f32
36 Ops.CONST dtypes.weakfloat [] 0.5
37 Ops.CAST dtypes.f32 [36] dtypes.f32
38 Ops.WHERE dtypes.f32 [33, 35, 37]
39 Ops.ADD dtypes.f32 [32, 38]
40 Ops.CAST dtypes.i32 [39] dtypes.i32
41 Ops.BITCAST dtypes.u32 [28] dtypes.u32
42 Ops.CONST dtypes.weakint [] 2155872255
43 Ops.AND dtypes.u32 [41, 42]
44 Ops.CONST dtypes.weakint [] 1056964608
45 Ops.OR dtypes.u32 [43, 44]
46 Ops.BITCAST dtypes.f32 [45] dtypes.f32
47 Ops.CONST dtypes.weakfloat [] 16777216.0
48 Ops.MUL dtypes.f32 [46, 47]
49 Ops.CAST dtypes.u64 [48] dtypes.u64
50 Ops.CONST dtypes.weakint [] 4294967295
51 Ops.AND dtypes.u64 [49, 50]
52 Ops.CONST dtypes.weakint [] 8388608
53 Ops.DIV dtypes.u32 [41, 52]
54 Ops.CONST dtypes.weakint [] 255
55 Ops.AND dtypes.u32 [53, 54]
56 Ops.CONST dtypes.weakint [] 127
57 Ops.CONST dtypes.weakint [] -1
58 Ops.MUL dtypes.weakint [56, 57]
59 Ops.ADD dtypes.u32 [55, 58]
60 Ops.CONST dtypes.weakint [] 1
61 Ops.ADD dtypes.u32 [59, 60]
62 Ops.CAST dtypes.i32 [61] dtypes.i32
63 Ops.CONST dtypes.weakint [] 104
64 Ops.ADD dtypes.i32 [62, 63]
65 Ops.CONST dtypes.weakint [] 0
66 Ops.MAX dtypes.i32 [64, 65]
67 Ops.CONST dtypes.weakint [] 32
68 Ops.DIV dtypes.i32 [66, 67]
69 Ops.AND dtypes.i32 [68, 60]
70 Ops.CMPEQ dtypes.bool [69, 65]
71 Ops.CMPEQ dtypes.bool [70, 6]
72 Ops.CONST dtypes.weakint [] 2
73 Ops.AND dtypes.i32 [68, 72]
74 Ops.CMPEQ dtypes.bool [73, 65]
75 Ops.CMPEQ dtypes.bool [74, 6]
76 Ops.CONST dtypes.weakint [] 4
77 Ops.AND dtypes.i32 [68, 76]
78 Ops.CMPEQ dtypes.bool [77, 65]
79 Ops.CMPEQ dtypes.bool [78, 6]
80 Ops.CONST dtypes.weakint [] 2102212464
81 Ops.CAST dtypes.u64 [80] dtypes.u64
82 Ops.CAST dtypes.u64 [65] dtypes.u64
83 Ops.WHERE dtypes.u64 [79, 81, 82]
84 Ops.CONST dtypes.weakint [] 2475754826
85 Ops.CAST dtypes.u64 [84] dtypes.u64
86 Ops.WHERE dtypes.u64 [79, 85, 82]
87 Ops.WHERE dtypes.u64 [75, 83, 86]
88 Ops.CONST dtypes.weakint [] 2131351028
89 Ops.CAST dtypes.u64 [88] dtypes.u64
90 Ops.WHERE dtypes.u64 [79, 89, 82]
91 Ops.CONST dtypes.weakint [] 683565275
92 Ops.CAST dtypes.u64 [91] dtypes.u64
93 Ops.WHERE dtypes.u64 [79, 92, 82]
94 Ops.WHERE dtypes.u64 [75, 90, 93]
95 Ops.WHERE dtypes.u64 [71, 87, 94]
96 Ops.CONST dtypes.weakint [] 31
97 Ops.AND dtypes.i32 [66, 96]
98 Ops.CAST dtypes.u64 [97] dtypes.u64
99 Ops.SHL dtypes.u64 [95, 98]
100 Ops.CONST dtypes.weakint [] 920167782
101 Ops.CAST dtypes.u64 [100] dtypes.u64
102 Ops.WHERE dtypes.u64 [79, 101, 92]
103 Ops.WHERE dtypes.u64 [75, 102, 90]
104 Ops.WHERE dtypes.u64 [71, 103, 87]
105 Ops.CONST dtypes.weakint [] 18446744073709551615
106 Ops.MUL dtypes.u64 [98, 105]
107 Ops.ADD dtypes.u64 [67, 106]
108 Ops.SHR dtypes.u64 [104, 107]
109 Ops.OR dtypes.u64 [99, 108]
110 Ops.AND dtypes.u64 [109, 50]
111 Ops.MUL dtypes.u64 [51, 110]
112 Ops.AND dtypes.u64 [111, 50]
113 Ops.SHL dtypes.u64 [104, 98]
114 Ops.CONST dtypes.weakint [] 1326507024
115 Ops.CAST dtypes.u64 [114] dtypes.u64
116 Ops.WHERE dtypes.u64 [79, 115, 85]
117 Ops.WHERE dtypes.u64 [75, 116, 83]
118 Ops.WHERE dtypes.u64 [71, 117, 103]
119 Ops.SHR dtypes.u64 [118, 107]
120 Ops.OR dtypes.u64 [113, 119]
121 Ops.AND dtypes.u64 [120, 50]
122 Ops.MUL dtypes.u64 [51, 121]
123 Ops.CONST dtypes.weakint [] 4294967296
124 Ops.DIV dtypes.u64 [122, 123]
125 Ops.ADD dtypes.u64 [112, 124]
126 Ops.AND dtypes.u64 [122, 50]
127 Ops.SHL dtypes.u64 [118, 98]
128 Ops.CONST dtypes.weakint [] 2140428522
129 Ops.CAST dtypes.u64 [128] dtypes.u64
130 Ops.WHERE dtypes.u64 [79, 129, 89]
131 Ops.WHERE dtypes.u64 [75, 130, 102]
132 Ops.WHERE dtypes.u64 [71, 131, 117]
133 Ops.SHR dtypes.u64 [132, 107]
134 Ops.OR dtypes.u64 [127, 133]
135 Ops.AND dtypes.u64 [134, 50]
136 Ops.MUL dtypes.u64 [51, 135]
137 Ops.DIV dtypes.u64 [136, 123]
138 Ops.ADD dtypes.u64 [126, 137]
139 Ops.AND dtypes.u64 [136, 50]
140 Ops.SHL dtypes.u64 [132, 98]
141 Ops.CONST dtypes.weakint [] 4155437400
142 Ops.CAST dtypes.u64 [141] dtypes.u64
143 Ops.WHERE dtypes.u64 [79, 142, 81]
144 Ops.WHERE dtypes.u64 [75, 143, 116]
145 Ops.WHERE dtypes.u64 [71, 144, 131]
146 Ops.SHR dtypes.u64 [145, 107]
147 Ops.OR dtypes.u64 [140, 146]
148 Ops.AND dtypes.u64 [147, 50]
149 Ops.MUL dtypes.u64 [51, 148]
150 Ops.DIV dtypes.u64 [149, 123]
151 Ops.ADD dtypes.u64 [139, 150]
152 Ops.AND dtypes.u64 [149, 50]
153 Ops.SHL dtypes.u64 [145, 98]
154 Ops.CONST dtypes.weakint [] 1841896334
155 Ops.CAST dtypes.u64 [154] dtypes.u64
156 Ops.WHERE dtypes.u64 [79, 155, 101]
157 Ops.WHERE dtypes.u64 [75, 156, 130]
158 Ops.WHERE dtypes.u64 [71, 157, 144]
159 Ops.SHR dtypes.u64 [158, 107]
160 Ops.OR dtypes.u64 [153, 159]
161 Ops.AND dtypes.u64 [160, 50]
162 Ops.MUL dtypes.u64 [51, 161]
163 Ops.DIV dtypes.u64 [162, 123]
164 Ops.ADD dtypes.u64 [152, 163]
165 Ops.AND dtypes.u64 [162, 50]
166 Ops.SHL dtypes.u64 [158, 98]
167 Ops.CONST dtypes.weakint [] 2425582776
168 Ops.CAST dtypes.u64 [167] dtypes.u64
169 Ops.WHERE dtypes.u64 [79, 168, 115]
170 Ops.WHERE dtypes.u64 [75, 169, 143]
171 Ops.WHERE dtypes.u64 [71, 170, 157]
172 Ops.SHR dtypes.u64 [171, 107]
173 Ops.OR dtypes.u64 [166, 172]
174 Ops.AND dtypes.u64 [173, 50]
175 Ops.MUL dtypes.u64 [51, 174]
176 Ops.DIV dtypes.u64 [175, 123]
177 Ops.ADD dtypes.u64 [165, 176]
178 Ops.AND dtypes.u64 [175, 50]
179 Ops.SHL dtypes.u64 [171, 98]
180 Ops.CONST dtypes.weakint [] 26364858
181 Ops.CAST dtypes.u64 [180] dtypes.u64
182 Ops.WHERE dtypes.u64 [79, 181, 129]
183 Ops.WHERE dtypes.u64 [75, 182, 156]
184 Ops.WHERE dtypes.u64 [71, 183, 170]
185 Ops.SHR dtypes.u64 [184, 107]
186 Ops.OR dtypes.u64 [179, 185]
187 Ops.AND dtypes.u64 [186, 50]
188 Ops.MUL dtypes.u64 [51, 187]
189 Ops.DIV dtypes.u64 [188, 123]
190 Ops.ADD dtypes.u64 [178, 189]
191 Ops.AND dtypes.u64 [188, 50]
192 Ops.DIV dtypes.u64 [191, 123]
193 Ops.ADD dtypes.u64 [190, 192]
194 Ops.DIV dtypes.u64 [193, 123]
195 Ops.ADD dtypes.u64 [177, 194]
196 Ops.DIV dtypes.u64 [195, 123]
197 Ops.ADD dtypes.u64 [164, 196]
198 Ops.DIV dtypes.u64 [197, 123]
199 Ops.ADD dtypes.u64 [151, 198]
200 Ops.DIV dtypes.u64 [199, 123]
201 Ops.ADD dtypes.u64 [138, 200]
202 Ops.DIV dtypes.u64 [201, 123]
203 Ops.ADD dtypes.u64 [125, 202]
204 Ops.AND dtypes.u64 [203, 50]
205 Ops.MUL dtypes.u64 [204, 123]
206 Ops.AND dtypes.u64 [201, 50]
207 Ops.OR dtypes.u64 [205, 206]
208 Ops.CONST dtypes.weakint [] 4611686018427387904
209 Ops.DIV dtypes.u64 [207, 208]
210 Ops.CONST dtypes.weakint [] 2305843009213693952
211 Ops.DIV dtypes.u64 [207, 210]
212 Ops.AND dtypes.u64 [211, 60]
213 Ops.ADD dtypes.u64 [209, 212]
214 Ops.CAST dtypes.i32 [213] dtypes.i32
215 Ops.WHERE dtypes.i32 [30, 40, 214]
216 Ops.AND dtypes.i32 [215, 72]
217 Ops.CMPEQ dtypes.bool [216, 65]
218 Ops.CMPEQ dtypes.bool [217, 6]
219 Ops.AND dtypes.i32 [215, 60]
220 Ops.CMPEQ dtypes.bool [219, 65]
221 Ops.CMPEQ dtypes.bool [220, 6]
222 Ops.CAST dtypes.f32 [40] dtypes.f32
223 Ops.CONST dtypes.weakfloat [] -0.00048351287841796875
224 Ops.MUL dtypes.f32 [222, 223]
225 Ops.CONST dtypes.weakfloat [] -1.5703125
226 Ops.MUL dtypes.f32 [222, 225]
227 Ops.ADD dtypes.f32 [226, 28]
228 Ops.ADD dtypes.f32 [224, 227]
229 Ops.CONST dtypes.weakfloat [] -3.1385570764541626e-07
230 Ops.MUL dtypes.f32 [222, 229]
231 Ops.ADD dtypes.f32 [228, 230]
232 Ops.MUL dtypes.f32 [228, 23]
233 Ops.ADD dtypes.f32 [231, 232]
234 Ops.MUL dtypes.f32 [233, 23]
235 Ops.ADD dtypes.f32 [230, 234]
236 Ops.CONST dtypes.weakfloat [] -6.07710062827671e-11
237 Ops.MUL dtypes.f32 [222, 236]
238 Ops.ADD dtypes.f32 [235, 237]
239 Ops.ADD dtypes.f32 [231, 238]
240 Ops.CONST dtypes.weakint [] 4611686018427387903
241 Ops.AND dtypes.u64 [207, 240]
242 Ops.CAST dtypes.i64 [241] dtypes.i64
243 Ops.CAST dtypes.i64 [212] dtypes.i64
244 Ops.MUL dtypes.i64 [243, 208]
245 Ops.MUL dtypes.i64 [244, 57]
246 Ops.ADD dtypes.i64 [242, 245]
247 Ops.CONST dtypes.weakint [] 4398046511104
248 Ops.DIV dtypes.i64 [246, 247]
249 Ops.CAST dtypes.f32 [248] dtypes.f32
250 Ops.CONST dtypes.weakfloat [] 4398046511104.0
251 Ops.MUL dtypes.f32 [249, 250]
252 Ops.CONST dtypes.weakint [] 2097152
253 Ops.DIV dtypes.i64 [246, 252]
254 Ops.CONST dtypes.weakint [] 2097151
255 Ops.AND dtypes.i64 [253, 254]
256 Ops.CAST dtypes.f32 [255] dtypes.f32
257 Ops.CONST dtypes.weakfloat [] 2097152.0
258 Ops.MUL dtypes.f32 [256, 257]
259 Ops.ADD dtypes.f32 [251, 258]
260 Ops.AND dtypes.i64 [246, 254]
261 Ops.CAST dtypes.f32 [260] dtypes.f32
262 Ops.MUL dtypes.f32 [261, 25]
263 Ops.ADD dtypes.f32 [259, 262]
264 Ops.AND dtypes.u64 [199, 50]
265 Ops.MUL dtypes.u64 [264, 123]
266 Ops.AND dtypes.u64 [197, 50]
267 Ops.OR dtypes.u64 [265, 266]
268 Ops.CONST dtypes.weakint [] 8796093022208
269 Ops.DIV dtypes.u64 [267, 268]
270 Ops.CAST dtypes.f32 [269] dtypes.f32
271 Ops.CONST dtypes.weakfloat [] 1.1368683772161603e-13
272 Ops.MUL dtypes.f32 [270, 271]
273 Ops.ADD dtypes.f32 [263, 272]
274 Ops.CONST dtypes.weakfloat [] 3.4061216748705226e-19
275 Ops.MUL dtypes.f32 [273, 274]
276 Ops.CONST dtypes.weakfloat [] 4097.0
277 Ops.MUL dtypes.f32 [273, 276]
278 Ops.MUL dtypes.f32 [273, 23]
279 Ops.ADD dtypes.f32 [277, 278]
280 Ops.MUL dtypes.f32 [279, 23]
281 Ops.ADD dtypes.f32 [277, 280]
282 Ops.CONST dtypes.weakfloat [] 3.406131239146355e-19
283 Ops.MUL dtypes.f32 [281, 282]
284 Ops.MUL dtypes.f32 [275, 23]
285 Ops.ADD dtypes.f32 [283, 284]
286 Ops.CONST dtypes.weakfloat [] -9.564275832644382e-25
287 Ops.MUL dtypes.f32 [281, 286]
288 Ops.ADD dtypes.f32 [285, 287]
289 Ops.MUL dtypes.f32 [281, 23]
290 Ops.ADD dtypes.f32 [273, 289]
291 Ops.MUL dtypes.f32 [290, 282]
292 Ops.ADD dtypes.f32 [288, 291]
293 Ops.MUL dtypes.f32 [290, 286]
294 Ops.ADD dtypes.f32 [292, 293]
295 Ops.CONST dtypes.weakfloat [] -9.47839680047608e-27
296 Ops.MUL dtypes.f32 [273, 295]
297 Ops.ADD dtypes.f32 [294, 296]
298 Ops.MUL dtypes.f32 [251, 23]
299 Ops.ADD dtypes.f32 [259, 298]
300 Ops.MUL dtypes.f32 [299, 23]
301 Ops.ADD dtypes.f32 [258, 300]
302 Ops.ADD dtypes.f32 [16, 301]
303 Ops.MUL dtypes.f32 [259, 23]
304 Ops.ADD dtypes.f32 [263, 303]
305 Ops.MUL dtypes.f32 [304, 23]
306 Ops.ADD dtypes.f32 [262, 305]
307 Ops.ADD dtypes.f32 [302, 306]
308 Ops.MUL dtypes.f32 [263, 23]
309 Ops.ADD dtypes.f32 [273, 308]
310 Ops.MUL dtypes.f32 [309, 23]
311 Ops.ADD dtypes.f32 [272, 310]
312 Ops.ADD dtypes.f32 [307, 311]
313 Ops.MUL dtypes.f32 [312, 274]
314 Ops.ADD dtypes.f32 [297, 313]
315 Ops.ADD dtypes.f32 [275, 314]
316 Ops.WHERE dtypes.f32 [30, 239, 315]
317 Ops.MUL dtypes.f32 [316, 316]
318 Ops.MUL dtypes.f32 [36, 317]
319 Ops.MUL dtypes.f32 [318, 23]
320 Ops.ADD dtypes.f32 [25, 319]
321 Ops.MUL dtypes.f32 [320, 23]
322 Ops.ADD dtypes.f32 [25, 321]
323 Ops.ADD dtypes.f32 [322, 319]
324 Ops.MUL dtypes.f32 [317, 317]
325 Ops.CONST dtypes.weakfloat [] -2.7557314351390663e-07
326 Ops.MUL dtypes.f32 [325, 317]
327 Ops.CONST dtypes.weakfloat [] 2.480158728947673e-05
328 Ops.ADD dtypes.f32 [326, 327]
329 Ops.MUL dtypes.f32 [328, 317]
330 Ops.CONST dtypes.weakfloat [] -0.001388888888887411
331 Ops.ADD dtypes.f32 [329, 330]
332 Ops.MUL dtypes.f32 [331, 317]
333 Ops.CONST dtypes.weakfloat [] 0.0416666666666666
334 Ops.ADD dtypes.f32 [332, 333]
335 Ops.MUL dtypes.f32 [324, 334]
336 Ops.MUL dtypes.f32 [231, 23]
337 Ops.ADD dtypes.f32 [239, 336]
338 Ops.MUL dtypes.f32 [337, 23]
339 Ops.ADD dtypes.f32 [238, 338]
340 Ops.ADD dtypes.f32 [315, 284]
341 Ops.MUL dtypes.f32 [340, 23]
342 Ops.ADD dtypes.f32 [314, 341]
343 Ops.WHERE dtypes.f32 [30, 339, 342]
344 Ops.MUL dtypes.f32 [316, 343]
345 Ops.MUL dtypes.f32 [344, 23]
346 Ops.ADD dtypes.f32 [335, 345]
347 Ops.ADD dtypes.f32 [323, 346]
348 Ops.ADD dtypes.f32 [320, 347]
349 Ops.MUL dtypes.f32 [36, 343]
350 Ops.MUL dtypes.f32 [317, 316]
351 Ops.CONST dtypes.weakfloat [] 2.7557313707070068e-06
352 Ops.MUL dtypes.f32 [351, 317]
353 Ops.CONST dtypes.weakfloat [] -0.0001984126982985795
354 Ops.ADD dtypes.f32 [352, 353]
355 Ops.MUL dtypes.f32 [354, 317]
356 Ops.CONST dtypes.weakfloat [] 0.00833333333332249
357 Ops.ADD dtypes.f32 [355, 356]
358 Ops.MUL dtypes.f32 [350, 357]
359 Ops.MUL dtypes.f32 [358, 23]
360 Ops.ADD dtypes.f32 [349, 359]
361 Ops.MUL dtypes.f32 [317, 360]
362 Ops.MUL dtypes.f32 [343, 23]
363 Ops.ADD dtypes.f32 [361, 362]
364 Ops.CONST dtypes.weakfloat [] -0.16666666666666632
365 Ops.MUL dtypes.f32 [350, 364]
366 Ops.MUL dtypes.f32 [365, 23]
367 Ops.ADD dtypes.f32 [363, 366]
368 Ops.MUL dtypes.f32 [367, 23]
369 Ops.ADD dtypes.f32 [316, 368]
370 Ops.WHERE dtypes.f32 [221, 348, 369]
371 Ops.MUL dtypes.f32 [370, 27]
372 Ops.MUL dtypes.f32 [371, 23]
373 Ops.WHERE dtypes.f32 [218, 372, 371]
374 Ops.WHERE dtypes.f32 [21, 373, 19]
375 Ops.WHERE dtypes.f32 [14, 374, 11]
376 Ops.WHERE dtypes.f32 [9, 11, 375]
377 Ops.WHERE dtypes.f32 [7, 376, 11]
378 Ops.CMPEQ dtypes.bool [37, 15]
379 Ops.CMPLT dtypes.bool [1, 15]
380 Ops.CAST dtypes.i32 [37] dtypes.i32
381 Ops.CAST dtypes.f32 [380] dtypes.f32
382 Ops.CMPEQ dtypes.bool [37, 381]
383 Ops.CMPEQ dtypes.bool [382, 6]
384 Ops.MUL dtypes.f32 [1, 23]
385 Ops.WHERE dtypes.f32 [379, 384, 1]
386 Ops.LOG2 dtypes.f32 [385]
387 Ops.MUL dtypes.f32 [386, 37]
388 Ops.EXP2 dtypes.f32 [387]
389 Ops.WHERE dtypes.f32 [14, 11, 388]
390 Ops.CMPLT dtypes.bool [37, 15]
391 Ops.MUL dtypes.f32 [37, 23]
392 Ops.WHERE dtypes.f32 [390, 391, 37]
393 Ops.CAST dtypes.i32 [392] dtypes.i32
394 Ops.MOD dtypes.i32 [393, 72]
395 Ops.CAST dtypes.bool [394] dtypes.bool
396 Ops.MUL dtypes.f32 [388, 23]
397 Ops.WHERE dtypes.f32 [395, 396, 388]
398 Ops.WHERE dtypes.f32 [383, 389, 397]
399 Ops.WHERE dtypes.f32 [379, 398, 388]
400 Ops.WHERE dtypes.f32 [378, 26, 399]
401 Ops.SINK dtypes.void [2, 3, 377, 400]

