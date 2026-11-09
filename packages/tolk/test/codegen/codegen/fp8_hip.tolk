# tolk D9 golden=b93f06ef02dccdd0b62084bac7490775
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 16
2 Ops.PARAM dtypes.u8 [1] ParamArg(slot=0, dtype=dtypes.u8)
3 Ops.CONST dtypes.weakint [] 4
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.SPECIAL dtypes.i32 [4] "lidx0"
6 Ops.CONST dtypes.weakint [] 2
7 Ops.CAST dtypes.i32 [6] dtypes.i32
8 Ops.SHL dtypes.i32 [5, 7]
9 Ops.INDEX dtypes.u8 [2, 8]
10 Ops.PARAM dtypes.u8 [1] ParamArg(slot=1, dtype=dtypes.u8)
11 Ops.INDEX dtypes.u8 [10, 8]
12 Ops.LOAD dtypes.u8 [11]
13 Ops.CONST dtypes.weakint [] 128
14 Ops.CAST dtypes.u8 [13] dtypes.u8
15 Ops.AND dtypes.u8 [12, 14]
16 Ops.CAST dtypes.u32 [15] dtypes.u32
17 Ops.CONST dtypes.weakint [] 24
18 Ops.CAST dtypes.u32 [17] dtypes.u32
19 Ops.SHL dtypes.u32 [16, 18]
20 Ops.CONST dtypes.weakint [] 127
21 Ops.CAST dtypes.u8 [20] dtypes.u8
22 Ops.AND dtypes.u8 [12, 21]
23 Ops.CAST dtypes.u32 [22] dtypes.u32
24 Ops.CAST dtypes.u32 [20] dtypes.u32
25 Ops.CMPEQ dtypes.bool [23, 24]
26 Ops.CONST dtypes.weakint [] 20
27 Ops.CAST dtypes.u32 [26] dtypes.u32
28 Ops.SHL dtypes.u32 [23, 27]
29 Ops.CONST dtypes.weakint [] 2139095040
30 Ops.CAST dtypes.u32 [29] dtypes.u32
31 Ops.OR dtypes.u32 [28, 30]
32 Ops.CONST dtypes.weakint [] 3
33 Ops.CAST dtypes.u32 [32] dtypes.u32
34 Ops.SHR dtypes.u32 [23, 33]
35 Ops.CONST dtypes.weakint [] 0
36 Ops.CAST dtypes.u32 [35] dtypes.u32
37 Ops.CMPEQ dtypes.bool [34, 36]
38 Ops.CMPEQ dtypes.bool [23, 36]
39 Ops.CAST dtypes.f32 [23] dtypes.f32
40 Ops.BITCAST dtypes.u32 [39] dtypes.u32
41 Ops.CONST dtypes.weakint [] 4219469824
42 Ops.CAST dtypes.u32 [41] dtypes.u32
43 Ops.ADD dtypes.u32 [40, 42]
44 Ops.WHERE dtypes.u32 [38, 36, 43]
45 Ops.CONST dtypes.weakint [] 1006632960
46 Ops.CAST dtypes.u32 [45] dtypes.u32
47 Ops.ADD dtypes.u32 [28, 46]
48 Ops.WHERE dtypes.u32 [37, 44, 47]
49 Ops.WHERE dtypes.u32 [25, 31, 48]
50 Ops.OR dtypes.u32 [19, 49]
51 Ops.BITCAST dtypes.f32 [50] dtypes.f32
52 Ops.CONST dtypes.weakint [] 1
53 Ops.CAST dtypes.i32 [52] dtypes.i32
54 Ops.ADD dtypes.i32 [8, 53]
55 Ops.INDEX dtypes.u8 [10, 54]
56 Ops.LOAD dtypes.u8 [55]
57 Ops.AND dtypes.u8 [56, 14]
58 Ops.CAST dtypes.u32 [57] dtypes.u32
59 Ops.SHL dtypes.u32 [58, 18]
60 Ops.AND dtypes.u8 [56, 21]
61 Ops.CAST dtypes.u32 [60] dtypes.u32
62 Ops.CMPEQ dtypes.bool [61, 24]
63 Ops.SHL dtypes.u32 [61, 27]
64 Ops.OR dtypes.u32 [63, 30]
65 Ops.SHR dtypes.u32 [61, 33]
66 Ops.CMPEQ dtypes.bool [65, 36]
67 Ops.CMPEQ dtypes.bool [61, 36]
68 Ops.CAST dtypes.f32 [61] dtypes.f32
69 Ops.BITCAST dtypes.u32 [68] dtypes.u32
70 Ops.ADD dtypes.u32 [69, 42]
71 Ops.WHERE dtypes.u32 [67, 36, 70]
72 Ops.ADD dtypes.u32 [63, 46]
73 Ops.WHERE dtypes.u32 [66, 71, 72]
74 Ops.WHERE dtypes.u32 [62, 64, 73]
75 Ops.OR dtypes.u32 [59, 74]
76 Ops.BITCAST dtypes.f32 [75] dtypes.f32
77 Ops.ADD dtypes.i32 [8, 7]
78 Ops.INDEX dtypes.u8 [10, 77]
79 Ops.LOAD dtypes.u8 [78]
80 Ops.AND dtypes.u8 [79, 14]
81 Ops.CAST dtypes.u32 [80] dtypes.u32
82 Ops.SHL dtypes.u32 [81, 18]
83 Ops.AND dtypes.u8 [79, 21]
84 Ops.CAST dtypes.u32 [83] dtypes.u32
85 Ops.CMPEQ dtypes.bool [84, 24]
86 Ops.SHL dtypes.u32 [84, 27]
87 Ops.OR dtypes.u32 [86, 30]
88 Ops.SHR dtypes.u32 [84, 33]
89 Ops.CMPEQ dtypes.bool [88, 36]
90 Ops.CMPEQ dtypes.bool [84, 36]
91 Ops.CAST dtypes.f32 [84] dtypes.f32
92 Ops.BITCAST dtypes.u32 [91] dtypes.u32
93 Ops.ADD dtypes.u32 [92, 42]
94 Ops.WHERE dtypes.u32 [90, 36, 93]
95 Ops.ADD dtypes.u32 [86, 46]
96 Ops.WHERE dtypes.u32 [89, 94, 95]
97 Ops.WHERE dtypes.u32 [85, 87, 96]
98 Ops.OR dtypes.u32 [82, 97]
99 Ops.BITCAST dtypes.f32 [98] dtypes.f32
100 Ops.CAST dtypes.i32 [32] dtypes.i32
101 Ops.ADD dtypes.i32 [8, 100]
102 Ops.INDEX dtypes.u8 [10, 101]
103 Ops.LOAD dtypes.u8 [102]
104 Ops.AND dtypes.u8 [103, 14]
105 Ops.CAST dtypes.u32 [104] dtypes.u32
106 Ops.SHL dtypes.u32 [105, 18]
107 Ops.AND dtypes.u8 [103, 21]
108 Ops.CAST dtypes.u32 [107] dtypes.u32
109 Ops.CMPEQ dtypes.bool [108, 24]
110 Ops.SHL dtypes.u32 [108, 27]
111 Ops.OR dtypes.u32 [110, 30]
112 Ops.SHR dtypes.u32 [108, 33]
113 Ops.CMPEQ dtypes.bool [112, 36]
114 Ops.CMPEQ dtypes.bool [108, 36]
115 Ops.CAST dtypes.f32 [108] dtypes.f32
116 Ops.BITCAST dtypes.u32 [115] dtypes.u32
117 Ops.ADD dtypes.u32 [116, 42]
118 Ops.WHERE dtypes.u32 [114, 36, 117]
119 Ops.ADD dtypes.u32 [110, 46]
120 Ops.WHERE dtypes.u32 [113, 118, 119]
121 Ops.WHERE dtypes.u32 [109, 111, 120]
122 Ops.OR dtypes.u32 [106, 121]
123 Ops.BITCAST dtypes.f32 [122] dtypes.f32
124 Ops.STACK dtypes.f32 [51, 76, 99, 123]
125 Ops.CAST dtypes.i32 [35] dtypes.i32
126 Ops.INDEX dtypes.f32 [124, 125]
127 Ops.CONST dtypes.weakfloat [] 2.0
128 Ops.CAST dtypes.f32 [127] dtypes.f32
129 Ops.MUL dtypes.f32 [126, 128]
130 Ops.BITCAST dtypes.u32 [129] dtypes.u32
131 Ops.SHR dtypes.u32 [130, 18]
132 Ops.CAST dtypes.u32 [13] dtypes.u32
133 Ops.AND dtypes.u32 [131, 132]
134 Ops.CONST dtypes.weakint [] 2147483647
135 Ops.CAST dtypes.u32 [134] dtypes.u32
136 Ops.AND dtypes.u32 [130, 135]
137 Ops.CMPLT dtypes.bool [30, 136]
138 Ops.CMPEQ dtypes.bool [136, 30]
139 Ops.CONST dtypes.weakint [] 23
140 Ops.CAST dtypes.u32 [139] dtypes.u32
141 Ops.SHR dtypes.u32 [130, 140]
142 Ops.CONST dtypes.weakint [] 255
143 Ops.CAST dtypes.u32 [142] dtypes.u32
144 Ops.AND dtypes.u32 [141, 143]
145 Ops.CONST dtypes.weakint [] 121
146 Ops.CAST dtypes.u32 [145] dtypes.u32
147 Ops.CMPLT dtypes.bool [144, 146]
148 Ops.CONST dtypes.weakint [] 8388607
149 Ops.CAST dtypes.u32 [148] dtypes.u32
150 Ops.AND dtypes.u32 [130, 149]
151 Ops.CONST dtypes.weakint [] 8388608
152 Ops.CAST dtypes.u32 [151] dtypes.u32
153 Ops.OR dtypes.u32 [150, 152]
154 Ops.CONST dtypes.weakint [] 141
155 Ops.CAST dtypes.u32 [154] dtypes.u32
156 Ops.SUB dtypes.u32 [155, 144]
157 Ops.CONST dtypes.weakint [] 25
158 Ops.CAST dtypes.u32 [157] dtypes.u32
159 Ops.CMPLT dtypes.bool [156, 158]
160 Ops.WHERE dtypes.u32 [159, 156, 158]
161 Ops.CAST dtypes.u32 [52] dtypes.u32
162 Ops.WHERE dtypes.u32 [147, 160, 161]
163 Ops.SHR dtypes.u32 [153, 162]
164 Ops.CONST dtypes.weakint [] 4294967295
165 Ops.CAST dtypes.u32 [164] dtypes.u32
166 Ops.ADD dtypes.u32 [162, 165]
167 Ops.CONST dtypes.weakint [] 32
168 Ops.CAST dtypes.u32 [167] dtypes.u32
169 Ops.CMPLT dtypes.bool [166, 168]
170 Ops.CONST dtypes.weakint [] 31
171 Ops.CAST dtypes.u32 [170] dtypes.u32
172 Ops.AND dtypes.u32 [166, 171]
173 Ops.SHR dtypes.u32 [153, 172]
174 Ops.WHERE dtypes.u32 [169, 173, 36]
175 Ops.AND dtypes.u32 [174, 161]
176 Ops.SHL dtypes.u32 [161, 172]
177 Ops.WHERE dtypes.u32 [169, 176, 36]
178 Ops.ADD dtypes.u32 [177, 165]
179 Ops.AND dtypes.u32 [153, 178]
180 Ops.CMPNE dtypes.bool [179, 36]
181 Ops.CAST dtypes.u32 [180] dtypes.u32
182 Ops.AND dtypes.u32 [163, 161]
183 Ops.OR dtypes.u32 [181, 182]
184 Ops.AND dtypes.u32 [175, 183]
185 Ops.ADD dtypes.u32 [163, 184]
186 Ops.SHR dtypes.u32 [136, 27]
187 Ops.CONST dtypes.weakint [] 19
188 Ops.CAST dtypes.u32 [187] dtypes.u32
189 Ops.SHR dtypes.u32 [136, 188]
190 Ops.AND dtypes.u32 [189, 161]
191 Ops.CONST dtypes.weakint [] 524287
192 Ops.CAST dtypes.u32 [191] dtypes.u32
193 Ops.AND dtypes.u32 [130, 192]
194 Ops.CMPNE dtypes.bool [193, 36]
195 Ops.CAST dtypes.u32 [194] dtypes.u32
196 Ops.AND dtypes.u32 [186, 161]
197 Ops.OR dtypes.u32 [195, 196]
198 Ops.AND dtypes.u32 [190, 197]
199 Ops.ADD dtypes.u32 [186, 198]
200 Ops.CONST dtypes.weakint [] 4294966336
201 Ops.CAST dtypes.u32 [200] dtypes.u32
202 Ops.ADD dtypes.u32 [199, 201]
203 Ops.XOR dtypes.u32 [202, 165]
204 Ops.CONST dtypes.weakint [] 4294967169
205 Ops.CAST dtypes.u32 [204] dtypes.u32
206 Ops.CMPLT dtypes.bool [203, 205]
207 Ops.WHERE dtypes.u32 [206, 205, 203]
208 Ops.XOR dtypes.u32 [207, 165]
209 Ops.WHERE dtypes.u32 [147, 185, 208]
210 Ops.WHERE dtypes.u32 [138, 24, 209]
211 Ops.WHERE dtypes.u32 [137, 24, 210]
212 Ops.OR dtypes.u32 [133, 211]
213 Ops.CAST dtypes.u8 [212] dtypes.u8
214 Ops.AND dtypes.u8 [213, 14]
215 Ops.CAST dtypes.u32 [214] dtypes.u32
216 Ops.SHL dtypes.u32 [215, 18]
217 Ops.AND dtypes.u8 [213, 21]
218 Ops.CAST dtypes.u32 [217] dtypes.u32
219 Ops.CMPEQ dtypes.bool [218, 24]
220 Ops.SHL dtypes.u32 [218, 27]
221 Ops.OR dtypes.u32 [220, 30]
222 Ops.SHR dtypes.u32 [218, 33]
223 Ops.CMPEQ dtypes.bool [222, 36]
224 Ops.CMPEQ dtypes.bool [218, 36]
225 Ops.CAST dtypes.f32 [218] dtypes.f32
226 Ops.BITCAST dtypes.u32 [225] dtypes.u32
227 Ops.ADD dtypes.u32 [226, 42]
228 Ops.WHERE dtypes.u32 [224, 36, 227]
229 Ops.ADD dtypes.u32 [220, 46]
230 Ops.WHERE dtypes.u32 [223, 228, 229]
231 Ops.WHERE dtypes.u32 [219, 221, 230]
232 Ops.OR dtypes.u32 [216, 231]
233 Ops.SHR dtypes.u32 [232, 18]
234 Ops.AND dtypes.u32 [233, 132]
235 Ops.AND dtypes.u32 [232, 135]
236 Ops.CMPLT dtypes.bool [30, 235]
237 Ops.CMPEQ dtypes.bool [235, 30]
238 Ops.SHR dtypes.u32 [232, 140]
239 Ops.AND dtypes.u32 [238, 143]
240 Ops.CMPLT dtypes.bool [239, 146]
241 Ops.AND dtypes.u32 [232, 149]
242 Ops.OR dtypes.u32 [241, 152]
243 Ops.SUB dtypes.u32 [155, 239]
244 Ops.CMPLT dtypes.bool [243, 158]
245 Ops.WHERE dtypes.u32 [244, 243, 158]
246 Ops.WHERE dtypes.u32 [240, 245, 161]
247 Ops.SHR dtypes.u32 [242, 246]
248 Ops.ADD dtypes.u32 [246, 165]
249 Ops.CMPLT dtypes.bool [248, 168]
250 Ops.AND dtypes.u32 [248, 171]
251 Ops.SHR dtypes.u32 [242, 250]
252 Ops.WHERE dtypes.u32 [249, 251, 36]
253 Ops.AND dtypes.u32 [252, 161]
254 Ops.SHL dtypes.u32 [161, 250]
255 Ops.WHERE dtypes.u32 [249, 254, 36]
256 Ops.ADD dtypes.u32 [255, 165]
257 Ops.AND dtypes.u32 [242, 256]
258 Ops.CMPNE dtypes.bool [257, 36]
259 Ops.CAST dtypes.u32 [258] dtypes.u32
260 Ops.AND dtypes.u32 [247, 161]
261 Ops.OR dtypes.u32 [259, 260]
262 Ops.AND dtypes.u32 [253, 261]
263 Ops.ADD dtypes.u32 [247, 262]
264 Ops.SHR dtypes.u32 [235, 27]
265 Ops.SHR dtypes.u32 [235, 188]
266 Ops.AND dtypes.u32 [265, 161]
267 Ops.AND dtypes.u32 [232, 192]
268 Ops.CMPNE dtypes.bool [267, 36]
269 Ops.CAST dtypes.u32 [268] dtypes.u32
270 Ops.AND dtypes.u32 [264, 161]
271 Ops.OR dtypes.u32 [269, 270]
272 Ops.AND dtypes.u32 [266, 271]
273 Ops.ADD dtypes.u32 [264, 272]
274 Ops.ADD dtypes.u32 [273, 201]
275 Ops.XOR dtypes.u32 [274, 165]
276 Ops.CMPLT dtypes.bool [275, 205]
277 Ops.WHERE dtypes.u32 [276, 205, 275]
278 Ops.XOR dtypes.u32 [277, 165]
279 Ops.WHERE dtypes.u32 [240, 263, 278]
280 Ops.WHERE dtypes.u32 [237, 24, 279]
281 Ops.WHERE dtypes.u32 [236, 24, 280]
282 Ops.OR dtypes.u32 [234, 281]
283 Ops.CAST dtypes.u8 [282] dtypes.u8
284 Ops.STORE dtypes.void [9, 283]
285 Ops.INDEX dtypes.u8 [2, 54]
286 Ops.INDEX dtypes.f32 [124, 53]
287 Ops.MUL dtypes.f32 [286, 128]
288 Ops.BITCAST dtypes.u32 [287] dtypes.u32
289 Ops.SHR dtypes.u32 [288, 18]
290 Ops.AND dtypes.u32 [289, 132]
291 Ops.AND dtypes.u32 [288, 135]
292 Ops.CMPLT dtypes.bool [30, 291]
293 Ops.CMPEQ dtypes.bool [291, 30]
294 Ops.SHR dtypes.u32 [288, 140]
295 Ops.AND dtypes.u32 [294, 143]
296 Ops.CMPLT dtypes.bool [295, 146]
297 Ops.AND dtypes.u32 [288, 149]
298 Ops.OR dtypes.u32 [297, 152]
299 Ops.SUB dtypes.u32 [155, 295]
300 Ops.CMPLT dtypes.bool [299, 158]
301 Ops.WHERE dtypes.u32 [300, 299, 158]
302 Ops.WHERE dtypes.u32 [296, 301, 161]
303 Ops.SHR dtypes.u32 [298, 302]
304 Ops.ADD dtypes.u32 [302, 165]
305 Ops.CMPLT dtypes.bool [304, 168]
306 Ops.AND dtypes.u32 [304, 171]
307 Ops.SHR dtypes.u32 [298, 306]
308 Ops.WHERE dtypes.u32 [305, 307, 36]
309 Ops.AND dtypes.u32 [308, 161]
310 Ops.SHL dtypes.u32 [161, 306]
311 Ops.WHERE dtypes.u32 [305, 310, 36]
312 Ops.ADD dtypes.u32 [311, 165]
313 Ops.AND dtypes.u32 [298, 312]
314 Ops.CMPNE dtypes.bool [313, 36]
315 Ops.CAST dtypes.u32 [314] dtypes.u32
316 Ops.AND dtypes.u32 [303, 161]
317 Ops.OR dtypes.u32 [315, 316]
318 Ops.AND dtypes.u32 [309, 317]
319 Ops.ADD dtypes.u32 [303, 318]
320 Ops.SHR dtypes.u32 [291, 27]
321 Ops.SHR dtypes.u32 [291, 188]
322 Ops.AND dtypes.u32 [321, 161]
323 Ops.AND dtypes.u32 [288, 192]
324 Ops.CMPNE dtypes.bool [323, 36]
325 Ops.CAST dtypes.u32 [324] dtypes.u32
326 Ops.AND dtypes.u32 [320, 161]
327 Ops.OR dtypes.u32 [325, 326]
328 Ops.AND dtypes.u32 [322, 327]
329 Ops.ADD dtypes.u32 [320, 328]
330 Ops.ADD dtypes.u32 [329, 201]
331 Ops.XOR dtypes.u32 [330, 165]
332 Ops.CMPLT dtypes.bool [331, 205]
333 Ops.WHERE dtypes.u32 [332, 205, 331]
334 Ops.XOR dtypes.u32 [333, 165]
335 Ops.WHERE dtypes.u32 [296, 319, 334]
336 Ops.WHERE dtypes.u32 [293, 24, 335]
337 Ops.WHERE dtypes.u32 [292, 24, 336]
338 Ops.OR dtypes.u32 [290, 337]
339 Ops.CAST dtypes.u8 [338] dtypes.u8
340 Ops.AND dtypes.u8 [339, 14]
341 Ops.CAST dtypes.u32 [340] dtypes.u32
342 Ops.SHL dtypes.u32 [341, 18]
343 Ops.AND dtypes.u8 [339, 21]
344 Ops.CAST dtypes.u32 [343] dtypes.u32
345 Ops.CMPEQ dtypes.bool [344, 24]
346 Ops.SHL dtypes.u32 [344, 27]
347 Ops.OR dtypes.u32 [346, 30]
348 Ops.SHR dtypes.u32 [344, 33]
349 Ops.CMPEQ dtypes.bool [348, 36]
350 Ops.CMPEQ dtypes.bool [344, 36]
351 Ops.CAST dtypes.f32 [344] dtypes.f32
352 Ops.BITCAST dtypes.u32 [351] dtypes.u32
353 Ops.ADD dtypes.u32 [352, 42]
354 Ops.WHERE dtypes.u32 [350, 36, 353]
355 Ops.ADD dtypes.u32 [346, 46]
356 Ops.WHERE dtypes.u32 [349, 354, 355]
357 Ops.WHERE dtypes.u32 [345, 347, 356]
358 Ops.OR dtypes.u32 [342, 357]
359 Ops.SHR dtypes.u32 [358, 18]
360 Ops.AND dtypes.u32 [359, 132]
361 Ops.AND dtypes.u32 [358, 135]
362 Ops.CMPLT dtypes.bool [30, 361]
363 Ops.CMPEQ dtypes.bool [361, 30]
364 Ops.SHR dtypes.u32 [358, 140]
365 Ops.AND dtypes.u32 [364, 143]
366 Ops.CMPLT dtypes.bool [365, 146]
367 Ops.AND dtypes.u32 [358, 149]
368 Ops.OR dtypes.u32 [367, 152]
369 Ops.SUB dtypes.u32 [155, 365]
370 Ops.CMPLT dtypes.bool [369, 158]
371 Ops.WHERE dtypes.u32 [370, 369, 158]
372 Ops.WHERE dtypes.u32 [366, 371, 161]
373 Ops.SHR dtypes.u32 [368, 372]
374 Ops.ADD dtypes.u32 [372, 165]
375 Ops.CMPLT dtypes.bool [374, 168]
376 Ops.AND dtypes.u32 [374, 171]
377 Ops.SHR dtypes.u32 [368, 376]
378 Ops.WHERE dtypes.u32 [375, 377, 36]
379 Ops.AND dtypes.u32 [378, 161]
380 Ops.SHL dtypes.u32 [161, 376]
381 Ops.WHERE dtypes.u32 [375, 380, 36]
382 Ops.ADD dtypes.u32 [381, 165]
383 Ops.AND dtypes.u32 [368, 382]
384 Ops.CMPNE dtypes.bool [383, 36]
385 Ops.CAST dtypes.u32 [384] dtypes.u32
386 Ops.AND dtypes.u32 [373, 161]
387 Ops.OR dtypes.u32 [385, 386]
388 Ops.AND dtypes.u32 [379, 387]
389 Ops.ADD dtypes.u32 [373, 388]
390 Ops.SHR dtypes.u32 [361, 27]
391 Ops.SHR dtypes.u32 [361, 188]
392 Ops.AND dtypes.u32 [391, 161]
393 Ops.AND dtypes.u32 [358, 192]
394 Ops.CMPNE dtypes.bool [393, 36]
395 Ops.CAST dtypes.u32 [394] dtypes.u32
396 Ops.AND dtypes.u32 [390, 161]
397 Ops.OR dtypes.u32 [395, 396]
398 Ops.AND dtypes.u32 [392, 397]
399 Ops.ADD dtypes.u32 [390, 398]
400 Ops.ADD dtypes.u32 [399, 201]
401 Ops.XOR dtypes.u32 [400, 165]
402 Ops.CMPLT dtypes.bool [401, 205]
403 Ops.WHERE dtypes.u32 [402, 205, 401]
404 Ops.XOR dtypes.u32 [403, 165]
405 Ops.WHERE dtypes.u32 [366, 389, 404]
406 Ops.WHERE dtypes.u32 [363, 24, 405]
407 Ops.WHERE dtypes.u32 [362, 24, 406]
408 Ops.OR dtypes.u32 [360, 407]
409 Ops.CAST dtypes.u8 [408] dtypes.u8
410 Ops.STORE dtypes.void [285, 409]
411 Ops.INDEX dtypes.u8 [2, 77]
412 Ops.INDEX dtypes.f32 [124, 7]
413 Ops.MUL dtypes.f32 [412, 128]
414 Ops.BITCAST dtypes.u32 [413] dtypes.u32
415 Ops.SHR dtypes.u32 [414, 18]
416 Ops.AND dtypes.u32 [415, 132]
417 Ops.AND dtypes.u32 [414, 135]
418 Ops.CMPLT dtypes.bool [30, 417]
419 Ops.CMPEQ dtypes.bool [417, 30]
420 Ops.SHR dtypes.u32 [414, 140]
421 Ops.AND dtypes.u32 [420, 143]
422 Ops.CMPLT dtypes.bool [421, 146]
423 Ops.AND dtypes.u32 [414, 149]
424 Ops.OR dtypes.u32 [423, 152]
425 Ops.SUB dtypes.u32 [155, 421]
426 Ops.CMPLT dtypes.bool [425, 158]
427 Ops.WHERE dtypes.u32 [426, 425, 158]
428 Ops.WHERE dtypes.u32 [422, 427, 161]
429 Ops.SHR dtypes.u32 [424, 428]
430 Ops.ADD dtypes.u32 [428, 165]
431 Ops.CMPLT dtypes.bool [430, 168]
432 Ops.AND dtypes.u32 [430, 171]
433 Ops.SHR dtypes.u32 [424, 432]
434 Ops.WHERE dtypes.u32 [431, 433, 36]
435 Ops.AND dtypes.u32 [434, 161]
436 Ops.SHL dtypes.u32 [161, 432]
437 Ops.WHERE dtypes.u32 [431, 436, 36]
438 Ops.ADD dtypes.u32 [437, 165]
439 Ops.AND dtypes.u32 [424, 438]
440 Ops.CMPNE dtypes.bool [439, 36]
441 Ops.CAST dtypes.u32 [440] dtypes.u32
442 Ops.AND dtypes.u32 [429, 161]
443 Ops.OR dtypes.u32 [441, 442]
444 Ops.AND dtypes.u32 [435, 443]
445 Ops.ADD dtypes.u32 [429, 444]
446 Ops.SHR dtypes.u32 [417, 27]
447 Ops.SHR dtypes.u32 [417, 188]
448 Ops.AND dtypes.u32 [447, 161]
449 Ops.AND dtypes.u32 [414, 192]
450 Ops.CMPNE dtypes.bool [449, 36]
451 Ops.CAST dtypes.u32 [450] dtypes.u32
452 Ops.AND dtypes.u32 [446, 161]
453 Ops.OR dtypes.u32 [451, 452]
454 Ops.AND dtypes.u32 [448, 453]
455 Ops.ADD dtypes.u32 [446, 454]
456 Ops.ADD dtypes.u32 [455, 201]
457 Ops.XOR dtypes.u32 [456, 165]
458 Ops.CMPLT dtypes.bool [457, 205]
459 Ops.WHERE dtypes.u32 [458, 205, 457]
460 Ops.XOR dtypes.u32 [459, 165]
461 Ops.WHERE dtypes.u32 [422, 445, 460]
462 Ops.WHERE dtypes.u32 [419, 24, 461]
463 Ops.WHERE dtypes.u32 [418, 24, 462]
464 Ops.OR dtypes.u32 [416, 463]
465 Ops.CAST dtypes.u8 [464] dtypes.u8
466 Ops.AND dtypes.u8 [465, 14]
467 Ops.CAST dtypes.u32 [466] dtypes.u32
468 Ops.SHL dtypes.u32 [467, 18]
469 Ops.AND dtypes.u8 [465, 21]
470 Ops.CAST dtypes.u32 [469] dtypes.u32
471 Ops.CMPEQ dtypes.bool [470, 24]
472 Ops.SHL dtypes.u32 [470, 27]
473 Ops.OR dtypes.u32 [472, 30]
474 Ops.SHR dtypes.u32 [470, 33]
475 Ops.CMPEQ dtypes.bool [474, 36]
476 Ops.CMPEQ dtypes.bool [470, 36]
477 Ops.CAST dtypes.f32 [470] dtypes.f32
478 Ops.BITCAST dtypes.u32 [477] dtypes.u32
479 Ops.ADD dtypes.u32 [478, 42]
480 Ops.WHERE dtypes.u32 [476, 36, 479]
481 Ops.ADD dtypes.u32 [472, 46]
482 Ops.WHERE dtypes.u32 [475, 480, 481]
483 Ops.WHERE dtypes.u32 [471, 473, 482]
484 Ops.OR dtypes.u32 [468, 483]
485 Ops.SHR dtypes.u32 [484, 18]
486 Ops.AND dtypes.u32 [485, 132]
487 Ops.AND dtypes.u32 [484, 135]
488 Ops.CMPLT dtypes.bool [30, 487]
489 Ops.CMPEQ dtypes.bool [487, 30]
490 Ops.SHR dtypes.u32 [484, 140]
491 Ops.AND dtypes.u32 [490, 143]
492 Ops.CMPLT dtypes.bool [491, 146]
493 Ops.AND dtypes.u32 [484, 149]
494 Ops.OR dtypes.u32 [493, 152]
495 Ops.SUB dtypes.u32 [155, 491]
496 Ops.CMPLT dtypes.bool [495, 158]
497 Ops.WHERE dtypes.u32 [496, 495, 158]
498 Ops.WHERE dtypes.u32 [492, 497, 161]
499 Ops.SHR dtypes.u32 [494, 498]
500 Ops.ADD dtypes.u32 [498, 165]
501 Ops.CMPLT dtypes.bool [500, 168]
502 Ops.AND dtypes.u32 [500, 171]
503 Ops.SHR dtypes.u32 [494, 502]
504 Ops.WHERE dtypes.u32 [501, 503, 36]
505 Ops.AND dtypes.u32 [504, 161]
506 Ops.SHL dtypes.u32 [161, 502]
507 Ops.WHERE dtypes.u32 [501, 506, 36]
508 Ops.ADD dtypes.u32 [507, 165]
509 Ops.AND dtypes.u32 [494, 508]
510 Ops.CMPNE dtypes.bool [509, 36]
511 Ops.CAST dtypes.u32 [510] dtypes.u32
512 Ops.AND dtypes.u32 [499, 161]
513 Ops.OR dtypes.u32 [511, 512]
514 Ops.AND dtypes.u32 [505, 513]
515 Ops.ADD dtypes.u32 [499, 514]
516 Ops.SHR dtypes.u32 [487, 27]
517 Ops.SHR dtypes.u32 [487, 188]
518 Ops.AND dtypes.u32 [517, 161]
519 Ops.AND dtypes.u32 [484, 192]
520 Ops.CMPNE dtypes.bool [519, 36]
521 Ops.CAST dtypes.u32 [520] dtypes.u32
522 Ops.AND dtypes.u32 [516, 161]
523 Ops.OR dtypes.u32 [521, 522]
524 Ops.AND dtypes.u32 [518, 523]
525 Ops.ADD dtypes.u32 [516, 524]
526 Ops.ADD dtypes.u32 [525, 201]
527 Ops.XOR dtypes.u32 [526, 165]
528 Ops.CMPLT dtypes.bool [527, 205]
529 Ops.WHERE dtypes.u32 [528, 205, 527]
530 Ops.XOR dtypes.u32 [529, 165]
531 Ops.WHERE dtypes.u32 [492, 515, 530]
532 Ops.WHERE dtypes.u32 [489, 24, 531]
533 Ops.WHERE dtypes.u32 [488, 24, 532]
534 Ops.OR dtypes.u32 [486, 533]
535 Ops.CAST dtypes.u8 [534] dtypes.u8
536 Ops.STORE dtypes.void [411, 535]
537 Ops.INDEX dtypes.u8 [2, 101]
538 Ops.INDEX dtypes.f32 [124, 100]
539 Ops.MUL dtypes.f32 [538, 128]
540 Ops.BITCAST dtypes.u32 [539] dtypes.u32
541 Ops.SHR dtypes.u32 [540, 18]
542 Ops.AND dtypes.u32 [541, 132]
543 Ops.AND dtypes.u32 [540, 135]
544 Ops.CMPLT dtypes.bool [30, 543]
545 Ops.CMPEQ dtypes.bool [543, 30]
546 Ops.SHR dtypes.u32 [540, 140]
547 Ops.AND dtypes.u32 [546, 143]
548 Ops.CMPLT dtypes.bool [547, 146]
549 Ops.AND dtypes.u32 [540, 149]
550 Ops.OR dtypes.u32 [549, 152]
551 Ops.SUB dtypes.u32 [155, 547]
552 Ops.CMPLT dtypes.bool [551, 158]
553 Ops.WHERE dtypes.u32 [552, 551, 158]
554 Ops.WHERE dtypes.u32 [548, 553, 161]
555 Ops.SHR dtypes.u32 [550, 554]
556 Ops.ADD dtypes.u32 [554, 165]
557 Ops.CMPLT dtypes.bool [556, 168]
558 Ops.AND dtypes.u32 [556, 171]
559 Ops.SHR dtypes.u32 [550, 558]
560 Ops.WHERE dtypes.u32 [557, 559, 36]
561 Ops.AND dtypes.u32 [560, 161]
562 Ops.SHL dtypes.u32 [161, 558]
563 Ops.WHERE dtypes.u32 [557, 562, 36]
564 Ops.ADD dtypes.u32 [563, 165]
565 Ops.AND dtypes.u32 [550, 564]
566 Ops.CMPNE dtypes.bool [565, 36]
567 Ops.CAST dtypes.u32 [566] dtypes.u32
568 Ops.AND dtypes.u32 [555, 161]
569 Ops.OR dtypes.u32 [567, 568]
570 Ops.AND dtypes.u32 [561, 569]
571 Ops.ADD dtypes.u32 [555, 570]
572 Ops.SHR dtypes.u32 [543, 27]
573 Ops.SHR dtypes.u32 [543, 188]
574 Ops.AND dtypes.u32 [573, 161]
575 Ops.AND dtypes.u32 [540, 192]
576 Ops.CMPNE dtypes.bool [575, 36]
577 Ops.CAST dtypes.u32 [576] dtypes.u32
578 Ops.AND dtypes.u32 [572, 161]
579 Ops.OR dtypes.u32 [577, 578]
580 Ops.AND dtypes.u32 [574, 579]
581 Ops.ADD dtypes.u32 [572, 580]
582 Ops.ADD dtypes.u32 [581, 201]
583 Ops.XOR dtypes.u32 [582, 165]
584 Ops.CMPLT dtypes.bool [583, 205]
585 Ops.WHERE dtypes.u32 [584, 205, 583]
586 Ops.XOR dtypes.u32 [585, 165]
587 Ops.WHERE dtypes.u32 [548, 571, 586]
588 Ops.WHERE dtypes.u32 [545, 24, 587]
589 Ops.WHERE dtypes.u32 [544, 24, 588]
590 Ops.OR dtypes.u32 [542, 589]
591 Ops.CAST dtypes.u8 [590] dtypes.u8
592 Ops.AND dtypes.u8 [591, 14]
593 Ops.CAST dtypes.u32 [592] dtypes.u32
594 Ops.SHL dtypes.u32 [593, 18]
595 Ops.AND dtypes.u8 [591, 21]
596 Ops.CAST dtypes.u32 [595] dtypes.u32
597 Ops.CMPEQ dtypes.bool [596, 24]
598 Ops.SHL dtypes.u32 [596, 27]
599 Ops.OR dtypes.u32 [598, 30]
600 Ops.SHR dtypes.u32 [596, 33]
601 Ops.CMPEQ dtypes.bool [600, 36]
602 Ops.CMPEQ dtypes.bool [596, 36]
603 Ops.CAST dtypes.f32 [596] dtypes.f32
604 Ops.BITCAST dtypes.u32 [603] dtypes.u32
605 Ops.ADD dtypes.u32 [604, 42]
606 Ops.WHERE dtypes.u32 [602, 36, 605]
607 Ops.ADD dtypes.u32 [598, 46]
608 Ops.WHERE dtypes.u32 [601, 606, 607]
609 Ops.WHERE dtypes.u32 [597, 599, 608]
610 Ops.OR dtypes.u32 [594, 609]
611 Ops.SHR dtypes.u32 [610, 18]
612 Ops.AND dtypes.u32 [611, 132]
613 Ops.AND dtypes.u32 [610, 135]
614 Ops.CMPLT dtypes.bool [30, 613]
615 Ops.CMPEQ dtypes.bool [613, 30]
616 Ops.SHR dtypes.u32 [610, 140]
617 Ops.AND dtypes.u32 [616, 143]
618 Ops.CMPLT dtypes.bool [617, 146]
619 Ops.AND dtypes.u32 [610, 149]
620 Ops.OR dtypes.u32 [619, 152]
621 Ops.SUB dtypes.u32 [155, 617]
622 Ops.CMPLT dtypes.bool [621, 158]
623 Ops.WHERE dtypes.u32 [622, 621, 158]
624 Ops.WHERE dtypes.u32 [618, 623, 161]
625 Ops.SHR dtypes.u32 [620, 624]
626 Ops.ADD dtypes.u32 [624, 165]
627 Ops.CMPLT dtypes.bool [626, 168]
628 Ops.AND dtypes.u32 [626, 171]
629 Ops.SHR dtypes.u32 [620, 628]
630 Ops.WHERE dtypes.u32 [627, 629, 36]
631 Ops.AND dtypes.u32 [630, 161]
632 Ops.SHL dtypes.u32 [161, 628]
633 Ops.WHERE dtypes.u32 [627, 632, 36]
634 Ops.ADD dtypes.u32 [633, 165]
635 Ops.AND dtypes.u32 [620, 634]
636 Ops.CMPNE dtypes.bool [635, 36]
637 Ops.CAST dtypes.u32 [636] dtypes.u32
638 Ops.AND dtypes.u32 [625, 161]
639 Ops.OR dtypes.u32 [637, 638]
640 Ops.AND dtypes.u32 [631, 639]
641 Ops.ADD dtypes.u32 [625, 640]
642 Ops.SHR dtypes.u32 [613, 27]
643 Ops.SHR dtypes.u32 [613, 188]
644 Ops.AND dtypes.u32 [643, 161]
645 Ops.AND dtypes.u32 [610, 192]
646 Ops.CMPNE dtypes.bool [645, 36]
647 Ops.CAST dtypes.u32 [646] dtypes.u32
648 Ops.AND dtypes.u32 [642, 161]
649 Ops.OR dtypes.u32 [647, 648]
650 Ops.AND dtypes.u32 [644, 649]
651 Ops.ADD dtypes.u32 [642, 650]
652 Ops.ADD dtypes.u32 [651, 201]
653 Ops.XOR dtypes.u32 [652, 165]
654 Ops.CMPLT dtypes.bool [653, 205]
655 Ops.WHERE dtypes.u32 [654, 205, 653]
656 Ops.XOR dtypes.u32 [655, 165]
657 Ops.WHERE dtypes.u32 [618, 641, 656]
658 Ops.WHERE dtypes.u32 [615, 24, 657]
659 Ops.WHERE dtypes.u32 [614, 24, 658]
660 Ops.OR dtypes.u32 [612, 659]
661 Ops.CAST dtypes.u8 [660] dtypes.u8
662 Ops.STORE dtypes.void [537, 661]
663 Ops.AFTER dtypes.void [0, 284, 410, 536, 662]
664 Ops.AFTER dtypes.void [0, 663]
665 Ops.SINK dtypes.void [664] KernelInfo(name="E_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.LOCAL))), optimized=True)
666 Ops.SINK dtypes.void [664] KernelInfo(name="E_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.LOCAL))), estimates=Estimates(ops=2032, lds=32, mem=32), optimized=True)
667 Ops.LINEAR dtypes.void [1, 2, 10, 3, 4, 5, 52, 53, 6, 7, 8, 54, 55, 56, 77, 78, 79, 32, 100, 101, 102, 103, 11, 12, 0, 20, 21, 60, 61, 68, 83, 84, 91, 107, 108, 115, 22, 23, 39, 13, 14, 57, 58, 80, 81, 104, 105, 15, 16, 35, 36, 45, 46, 24, 26, 27, 29, 30, 17, 18, 33, 41, 42, 69, 70, 63, 72, 59, 65, 67, 62, 66, 64, 71, 73, 74, 75, 76, 92, 93, 86, 95, 82, 88, 90, 85, 89, 87, 94, 96, 97, 98, 99, 116, 117, 110, 119, 106, 112, 114, 109, 113, 111, 118, 120, 121, 122, 123, 40, 43, 28, 47, 19, 34, 38, 25, 37, 31, 44, 48, 49, 50, 51, 124, 285, 411, 537, 9, 125, 126, 286, 412, 538, 127, 128, 191, 192, 129, 130, 193, 194, 195, 161, 145, 146, 154, 155, 139, 140, 157, 158, 142, 143, 170, 171, 167, 168, 164, 165, 148, 149, 151, 152, 141, 144, 147, 156, 159, 160, 162, 166, 172, 176, 169, 177, 178, 150, 153, 179, 180, 181, 132, 187, 188, 134, 135, 200, 201, 204, 205, 136, 189, 186, 196, 197, 190, 198, 199, 202, 173, 163, 182, 183, 174, 175, 184, 185, 131, 137, 203, 206, 138, 207, 208, 133, 209, 210, 211, 212, 213, 217, 218, 225, 287, 288, 323, 324, 325, 294, 295, 296, 299, 300, 301, 302, 304, 306, 310, 305, 311, 312, 297, 298, 313, 314, 315, 291, 321, 320, 326, 327, 322, 328, 329, 330, 307, 303, 316, 317, 308, 309, 318, 319, 289, 292, 331, 332, 293, 333, 334, 290, 335, 336, 337, 338, 339, 343, 344, 351, 413, 414, 449, 450, 451, 420, 421, 422, 425, 426, 427, 428, 430, 432, 436, 431, 437, 438, 423, 424, 439, 440, 441, 417, 447, 446, 452, 453, 448, 454, 455, 456, 433, 429, 442, 443, 434, 435, 444, 445, 415, 418, 457, 458, 419, 459, 460, 416, 461, 462, 463, 464, 465, 469, 470, 477, 539, 540, 575, 576, 577, 546, 547, 548, 551, 552, 553, 554, 556, 558, 562, 557, 563, 564, 549, 550, 565, 566, 567, 543, 573, 572, 578, 579, 574, 580, 581, 582, 559, 555, 568, 569, 560, 561, 570, 571, 541, 544, 583, 584, 545, 585, 586, 542, 587, 588, 589, 590, 591, 595, 596, 603, 214, 215, 226, 227, 220, 229, 216, 222, 224, 219, 223, 221, 228, 230, 231, 232, 267, 268, 269, 340, 341, 352, 353, 346, 355, 342, 348, 350, 345, 349, 347, 354, 356, 357, 358, 393, 394, 395, 466, 467, 478, 479, 472, 481, 468, 474, 476, 471, 475, 473, 480, 482, 483, 484, 519, 520, 521, 592, 593, 604, 605, 598, 607, 594, 600, 602, 597, 601, 599, 606, 608, 609, 610, 645, 646, 647, 238, 239, 240, 243, 244, 245, 246, 248, 250, 254, 249, 255, 256, 241, 242, 257, 258, 259, 364, 365, 366, 369, 370, 371, 372, 374, 376, 380, 375, 381, 382, 367, 368, 383, 384, 385, 490, 491, 492, 495, 496, 497, 498, 500, 502, 506, 501, 507, 508, 493, 494, 509, 510, 511, 616, 617, 618, 621, 622, 623, 624, 626, 628, 632, 627, 633, 634, 619, 620, 635, 636, 637, 235, 265, 264, 270, 271, 266, 272, 273, 274, 251, 247, 260, 261, 252, 253, 262, 263, 233, 236, 275, 276, 237, 277, 278, 234, 279, 280, 281, 282, 283, 361, 391, 390, 396, 397, 392, 398, 399, 400, 377, 373, 386, 387, 378, 379, 388, 389, 359, 362, 401, 402, 363, 403, 404, 360, 405, 406, 407, 408, 409, 487, 517, 516, 522, 523, 518, 524, 525, 526, 503, 499, 512, 513, 504, 505, 514, 515, 485, 488, 527, 528, 489, 529, 530, 486, 531, 532, 533, 534, 535, 613, 643, 642, 648, 649, 644, 650, 651, 652, 629, 625, 638, 639, 630, 631, 640, 641, 611, 614, 653, 654, 615, 655, 656, 612, 657, 658, 659, 660, 661, 410, 536, 662, 284, 663, 664, 666]
668 Ops.SOURCE dtypes.void [] "typedef long unsigned int size_t;\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_local_size(unsigned int);\ntypedef float float4 __attribute__((ext_vector_type(4)));\nstatic inline __attribute__((device)) float4 make_float4(float x, float y, float z, float w) { return { x, y, z, w }; }\nextern \"C\" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 4))) E_4_4(unsigned char* data0_16, unsigned char* data1_16) {\n  int lidx0 = __ockl_get_local_id(0); /* 4 */\n  int alu0 = (lidx0<<2);\n  int alu1 = (alu0+1);\n  unsigned char val0 = (*(data1_16+alu1));\n  int alu2 = (alu0+2);\n  unsigned char val1 = (*(data1_16+alu2));\n  int alu3 = (alu0+3);\n  unsigned char val2 = (*(data1_16+alu3));\n  unsigned char val3 = (*(data1_16+alu0));\n  unsigned int cast0 = ((unsigned int)((val0&((unsigned char)(127u)))));\n  unsigned int cast1 = ((unsigned int)((val1&((unsigned char)(127u)))));\n  unsigned int cast2 = ((unsigned int)((val2&((unsigned char)(127u)))));\n  unsigned int cast3 = ((unsigned int)((val3&((unsigned char)(127u)))));\n  unsigned int alu4 = (cast0<<20u);\n  unsigned int alu5 = ((cast0==0u)?0u:(__builtin_bit_cast(unsigned int, (float)(((float)(cast0))))+4219469824u));\n  unsigned int alu6 = (((cast0>>3u)==0u)?alu5:(alu4+1006632960u));\n  unsigned int alu7 = ((cast0==127u)?(alu4|2139095040u):alu6);\n  unsigned int alu8 = (cast1<<20u);\n  unsigned int alu9 = ((cast1==0u)?0u:(__builtin_bit_cast(unsigned int, (float)(((float)(cast1))))+4219469824u));\n  unsigned int alu10 = (((cast1>>3u)==0u)?alu9:(alu8+1006632960u));\n  unsigned int alu11 = ((cast1==127u)?(alu8|2139095040u):alu10);\n  unsigned int alu12 = (cast2<<20u);\n  unsigned int alu13 = ((cast2==0u)?0u:(__builtin_bit_cast(unsigned int, (float)(((float)(cast2))))+4219469824u));\n  unsigned int alu14 = (((cast2>>3u)==0u)?alu13:(alu12+1006632960u));\n  unsigned int alu15 = ((cast2==127u)?(alu12|2139095040u):alu14);\n  unsigned int alu16 = (cast3<<20u);\n  unsigned int alu17 = ((cast3==0u)?0u:(__builtin_bit_cast(unsigned int, (float)(((float)(cast3))))+4219469824u));\n  unsigned int alu18 = (((cast3>>3u)==0u)?alu17:(alu16+1006632960u));\n  unsigned int alu19 = ((cast3==127u)?(alu16|2139095040u):alu18);\n  float4 cast4 = make_float4(__builtin_bit_cast(float, (unsigned int)(((((unsigned int)((val3&((unsigned char)(128u)))))<<24u)|alu19))),__builtin_bit_cast(float, (unsigned int)(((((unsigned int)((val0&((unsigned char)(128u)))))<<24u)|alu7))),__builtin_bit_cast(float, (unsigned int)(((((unsigned int)((val1&((unsigned char)(128u)))))<<24u)|alu11))),__builtin_bit_cast(float, (unsigned int)(((((unsigned int)((val2&((unsigned char)(128u)))))<<24u)|alu15))));\n  unsigned int cast5 = __builtin_bit_cast(unsigned int, (float)((cast4.x*2.0f)));\n  unsigned int alu20 = ((cast5>>23u)&255u);\n  bool alu21 = (alu20<121u);\n  unsigned int alu22 = (141u-alu20);\n  unsigned int alu23 = ((alu22<25u)?alu22:25u);\n  unsigned int alu24 = (alu21?alu23:1u);\n  unsigned int alu25 = (alu24+4294967295u);\n  unsigned int alu26 = (alu25&31u);\n  bool alu27 = (alu25<32u);\n  unsigned int alu28 = (alu27?(1u<<alu26):0u);\n  unsigned int alu29 = ((cast5&8388607u)|8388608u);\n  unsigned int alu30 = (cast5&2147483647u);\n  unsigned int alu31 = (alu30>>20u);\n  unsigned int alu32 = (alu29>>alu24);\n  unsigned int alu33 = (alu27?(alu29>>alu26):0u);\n  unsigned int alu34 = ((alu31+((alu30>>19u)&1u&(((unsigned int)(((cast5&524287u)!=0u)))|(alu31&1u)))+4294966336u)^4294967295u);\n  unsigned int alu35 = ((alu34<4294967169u)?4294967169u:alu34);\n  unsigned int alu36 = (alu21?(alu32+(alu33&1u&(((unsigned int)(((alu29&(alu28+4294967295u))!=0u)))|(alu32&1u)))):(alu35^4294967295u));\n  unsigned int alu37 = ((alu30==2139095040u)?127u:alu36);\n  unsigned int alu38 = ((2139095040u<alu30)?127u:alu37);\n  unsigned char cast6 = ((unsigned char)((((cast5>>24u)&128u)|alu38)));\n  unsigned int cast7 = ((unsigned int)((cast6&((unsigned char)(127u)))));\n  unsigned int cast8 = __builtin_bit_cast(unsigned int, (float)((cast4.y*2.0f)));\n  unsigned int alu39 = ((cast8>>23u)&255u);\n  bool alu40 = (alu39<121u);\n  unsigned int alu41 = (141u-alu39);\n  unsigned int alu42 = ((alu41<25u)?alu41:25u);\n  unsigned int alu43 = (alu40?alu42:1u);\n  unsigned int alu44 = (alu43+4294967295u);\n  unsigned int alu45 = (alu44&31u);\n  bool alu46 = (alu44<32u);\n  unsigned int alu47 = (alu46?(1u<<alu45):0u);\n  unsigned int alu48 = ((cast8&8388607u)|8388608u);\n  unsigned int alu49 = (cast8&2147483647u);\n  unsigned int alu50 = (alu49>>20u);\n  unsigned int alu51 = (alu48>>alu43);\n  unsigned int alu52 = (alu46?(alu48>>alu45):0u);\n  unsigned int alu53 = ((alu50+((alu49>>19u)&1u&(((unsigned int)(((cast8&524287u)!=0u)))|(alu50&1u)))+4294966336u)^4294967295u);\n  unsigned int alu54 = ((alu53<4294967169u)?4294967169u:alu53);\n  unsigned int alu55 = (alu40?(alu51+(alu52&1u&(((unsigned int)(((alu48&(alu47+4294967295u))!=0u)))|(alu51&1u)))):(alu54^4294967295u));\n  unsigned int alu56 = ((alu49==2139095040u)?127u:alu55);\n  unsigned int alu57 = ((2139095040u<alu49)?127u:alu56);\n  unsigned char cast9 = ((unsigned char)((((cast8>>24u)&128u)|alu57)));\n  unsigned int cast10 = ((unsigned int)((cast9&((unsigned char)(127u)))));\n  unsigned int cast11 = __builtin_bit_cast(unsigned int, (float)((cast4.z*2.0f)));\n  unsigned int alu58 = ((cast11>>23u)&255u);\n  bool alu59 = (alu58<121u);\n  unsigned int alu60 = (141u-alu58);\n  unsigned int alu61 = ((alu60<25u)?alu60:25u);\n  unsigned int alu62 = (alu59?alu61:1u);\n  unsigned int alu63 = (alu62+4294967295u);\n  unsigned int alu64 = (alu63&31u);\n  bool alu65 = (alu63<32u);\n  unsigned int alu66 = (alu65?(1u<<alu64):0u);\n  unsigned int alu67 = ((cast11&8388607u)|8388608u);\n  unsigned int alu68 = (cast11&2147483647u);\n  unsigned int alu69 = (alu68>>20u);\n  unsigned int alu70 = (alu67>>alu62);\n  unsigned int alu71 = (alu65?(alu67>>alu64):0u);\n  unsigned int alu72 = ((alu69+((alu68>>19u)&1u&(((unsigned int)(((cast11&524287u)!=0u)))|(alu69&1u)))+4294966336u)^4294967295u);\n  unsigned int alu73 = ((alu72<4294967169u)?4294967169u:alu72);\n  unsigned int alu74 = (alu59?(alu70+(alu71&1u&(((unsigned int)(((alu67&(alu66+4294967295u))!=0u)))|(alu70&1u)))):(alu73^4294967295u));\n  unsigned int alu75 = ((alu68==2139095040u)?127u:alu74);\n  unsigned int alu76 = ((2139095040u<alu68)?127u:alu75);\n  unsigned char cast12 = ((unsigned char)((((cast11>>24u)&128u)|alu76)));\n  unsigned int cast13 = ((unsigned int)((cast12&((unsigned char)(127u)))));\n  unsigned int cast14 = __builtin_bit_cast(unsigned int, (float)((cast4.w*2.0f)));\n  unsigned int alu77 = ((cast14>>23u)&255u);\n  bool alu78 = (alu77<121u);\n  unsigned int alu79 = (141u-alu77);\n  unsigned int alu80 = ((alu79<25u)?alu79:25u);\n  unsigned int alu81 = (alu78?alu80:1u);\n  unsigned int alu82 = (alu81+4294967295u);\n  unsigned int alu83 = (alu82&31u);\n  bool alu84 = (alu82<32u);\n  unsigned int alu85 = (alu84?(1u<<alu83):0u);\n  unsigned int alu86 = ((cast14&8388607u)|8388608u);\n  unsigned int alu87 = (cast14&2147483647u);\n  unsigned int alu88 = (alu87>>20u);\n  unsigned int alu89 = (alu86>>alu81);\n  unsigned int alu90 = (alu84?(alu86>>alu83):0u);\n  unsigned int alu91 = ((alu88+((alu87>>19u)&1u&(((unsigned int)(((cast14&524287u)!=0u)))|(alu88&1u)))+4294966336u)^4294967295u);\n  unsigned int alu92 = ((alu91<4294967169u)?4294967169u:alu91);\n  unsigned int alu93 = (alu78?(alu89+(alu90&1u&(((unsigned int)(((alu86&(alu85+4294967295u))!=0u)))|(alu89&1u)))):(alu92^4294967295u));\n  unsigned int alu94 = ((alu87==2139095040u)?127u:alu93);\n  unsigned int alu95 = ((2139095040u<alu87)?127u:alu94);\n  unsigned char cast15 = ((unsigned char)((((cast14>>24u)&128u)|alu95)));\n  unsigned int cast16 = ((unsigned int)((cast15&((unsigned char)(127u)))));\n  unsigned int alu96 = (cast7<<20u);\n  unsigned int alu97 = ((cast7==0u)?0u:(__builtin_bit_cast(unsigned int, (float)(((float)(cast7))))+4219469824u));\n  unsigned int alu98 = (((cast7>>3u)==0u)?alu97:(alu96+1006632960u));\n  unsigned int alu99 = ((cast7==127u)?(alu96|2139095040u):alu98);\n  unsigned int alu100 = ((((unsigned int)((cast6&((unsigned char)(128u)))))<<24u)|alu99);\n  unsigned int alu101 = (cast10<<20u);\n  unsigned int alu102 = ((cast10==0u)?0u:(__builtin_bit_cast(unsigned int, (float)(((float)(cast10))))+4219469824u));\n  unsigned int alu103 = (((cast10>>3u)==0u)?alu102:(alu101+1006632960u));\n  unsigned int alu104 = ((cast10==127u)?(alu101|2139095040u):alu103);\n  unsigned int alu105 = ((((unsigned int)((cast9&((unsigned char)(128u)))))<<24u)|alu104);\n  unsigned int alu106 = (cast13<<20u);\n  unsigned int alu107 = ((cast13==0u)?0u:(__builtin_bit_cast(unsigned int, (float)(((float)(cast13))))+4219469824u));\n  unsigned int alu108 = (((cast13>>3u)==0u)?alu107:(alu106+1006632960u));\n  unsigned int alu109 = ((cast13==127u)?(alu106|2139095040u):alu108);\n  unsigned int alu110 = ((((unsigned int)((cast12&((unsigned char)(128u)))))<<24u)|alu109);\n  unsigned int alu111 = (cast16<<20u);\n  unsigned int alu112 = ((cast16==0u)?0u:(__builtin_bit_cast(unsigned int, (float)(((float)(cast16))))+4219469824u));\n  unsigned int alu113 = (((cast16>>3u)==0u)?alu112:(alu111+1006632960u));\n  unsigned int alu114 = ((cast16==127u)?(alu111|2139095040u):alu113);\n  unsigned int alu115 = ((((unsigned int)((cast15&((unsigned char)(128u)))))<<24u)|alu114);\n  unsigned int alu116 = ((alu100>>23u)&255u);\n  bool alu117 = (alu116<121u);\n  unsigned int alu118 = (141u-alu116);\n  unsigned int alu119 = ((alu118<25u)?alu118:25u);\n  unsigned int alu120 = (alu117?alu119:1u);\n  unsigned int alu121 = (alu120+4294967295u);\n  unsigned int alu122 = (alu121&31u);\n  bool alu123 = (alu121<32u);\n  unsigned int alu124 = (alu123?(1u<<alu122):0u);\n  unsigned int alu125 = ((alu100&8388607u)|8388608u);\n  unsigned int alu126 = ((alu105>>23u)&255u);\n  bool alu127 = (alu126<121u);\n  unsigned int alu128 = (141u-alu126);\n  unsigned int alu129 = ((alu128<25u)?alu128:25u);\n  unsigned int alu130 = (alu127?alu129:1u);\n  unsigned int alu131 = (alu130+4294967295u);\n  unsigned int alu132 = (alu131&31u);\n  bool alu133 = (alu131<32u);\n  unsigned int alu134 = (alu133?(1u<<alu132):0u);\n  unsigned int alu135 = ((alu105&8388607u)|8388608u);\n  unsigned int alu136 = ((alu110>>23u)&255u);\n  bool alu137 = (alu136<121u);\n  unsigned int alu138 = (141u-alu136);\n  unsigned int alu139 = ((alu138<25u)?alu138:25u);\n  unsigned int alu140 = (alu137?alu139:1u);\n  unsigned int alu141 = (alu140+4294967295u);\n  unsigned int alu142 = (alu141&31u);\n  bool alu143 = (alu141<32u);\n  unsigned int alu144 = (alu143?(1u<<alu142):0u);\n  unsigned int alu145 = ((alu110&8388607u)|8388608u);\n  unsigned int alu146 = ((alu115>>23u)&255u);\n  bool alu147 = (alu146<121u);\n  unsigned int alu148 = (141u-alu146);\n  unsigned int alu149 = ((alu148<25u)?alu148:25u);\n  unsigned int alu150 = (alu147?alu149:1u);\n  unsigned int alu151 = (alu150+4294967295u);\n  unsigned int alu152 = (alu151&31u);\n  bool alu153 = (alu151<32u);\n  unsigned int alu154 = (alu153?(1u<<alu152):0u);\n  unsigned int alu155 = ((alu115&8388607u)|8388608u);\n  unsigned int alu156 = (alu100&2147483647u);\n  unsigned int alu157 = (alu156>>20u);\n  unsigned int alu158 = (alu125>>alu120);\n  unsigned int alu159 = (alu123?(alu125>>alu122):0u);\n  unsigned int alu160 = ((alu157+((alu156>>19u)&1u&(((unsigned int)(((alu100&524287u)!=0u)))|(alu157&1u)))+4294966336u)^4294967295u);\n  unsigned int alu161 = ((alu160<4294967169u)?4294967169u:alu160);\n  unsigned int alu162 = (alu117?(alu158+(alu159&1u&(((unsigned int)(((alu125&(alu124+4294967295u))!=0u)))|(alu158&1u)))):(alu161^4294967295u));\n  unsigned int alu163 = ((alu156==2139095040u)?127u:alu162);\n  unsigned int alu164 = ((2139095040u<alu156)?127u:alu163);\n  unsigned int alu165 = (alu105&2147483647u);\n  unsigned int alu166 = (alu165>>20u);\n  unsigned int alu167 = (alu135>>alu130);\n  unsigned int alu168 = (alu133?(alu135>>alu132):0u);\n  unsigned int alu169 = ((alu166+((alu165>>19u)&1u&(((unsigned int)(((alu105&524287u)!=0u)))|(alu166&1u)))+4294966336u)^4294967295u);\n  unsigned int alu170 = ((alu169<4294967169u)?4294967169u:alu169);\n  unsigned int alu171 = (alu127?(alu167+(alu168&1u&(((unsigned int)(((alu135&(alu134+4294967295u))!=0u)))|(alu167&1u)))):(alu170^4294967295u));\n  unsigned int alu172 = ((alu165==2139095040u)?127u:alu171);\n  unsigned int alu173 = ((2139095040u<alu165)?127u:alu172);\n  unsigned int alu174 = (alu110&2147483647u);\n  unsigned int alu175 = (alu174>>20u);\n  unsigned int alu176 = (alu145>>alu140);\n  unsigned int alu177 = (alu143?(alu145>>alu142):0u);\n  unsigned int alu178 = ((alu175+((alu174>>19u)&1u&(((unsigned int)(((alu110&524287u)!=0u)))|(alu175&1u)))+4294966336u)^4294967295u);\n  unsigned int alu179 = ((alu178<4294967169u)?4294967169u:alu178);\n  unsigned int alu180 = (alu137?(alu176+(alu177&1u&(((unsigned int)(((alu145&(alu144+4294967295u))!=0u)))|(alu176&1u)))):(alu179^4294967295u));\n  unsigned int alu181 = ((alu174==2139095040u)?127u:alu180);\n  unsigned int alu182 = ((2139095040u<alu174)?127u:alu181);\n  unsigned int alu183 = (alu115&2147483647u);\n  unsigned int alu184 = (alu183>>20u);\n  unsigned int alu185 = (alu155>>alu150);\n  unsigned int alu186 = (alu153?(alu155>>alu152):0u);\n  unsigned int alu187 = ((alu184+((alu183>>19u)&1u&(((unsigned int)(((alu115&524287u)!=0u)))|(alu184&1u)))+4294966336u)^4294967295u);\n  unsigned int alu188 = ((alu187<4294967169u)?4294967169u:alu187);\n  unsigned int alu189 = (alu147?(alu185+(alu186&1u&(((unsigned int)(((alu155&(alu154+4294967295u))!=0u)))|(alu185&1u)))):(alu188^4294967295u));\n  unsigned int alu190 = ((alu183==2139095040u)?127u:alu189);\n  unsigned int alu191 = ((2139095040u<alu183)?127u:alu190);\n  *(data0_16+alu1) = ((unsigned char)((((alu105>>24u)&128u)|alu173)));\n  *(data0_16+alu2) = ((unsigned char)((((alu110>>24u)&128u)|alu182)));\n  *(data0_16+alu3) = ((unsigned char)((((alu115>>24u)&128u)|alu191)));\n  *(data0_16+alu0) = ((unsigned char)((((alu100>>24u)&128u)|alu164)));\n}"
669 Ops.PROGRAM dtypes.void [666, 667, 668] ProgramInfo(local_size=(4, 1, 1), params=(%2, %10), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="AMD", renderer="HIP", arch="gfx1100"))
670 Ops.SINK dtypes.void [665, 669]

