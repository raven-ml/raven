# tolk D89 D430 golden=0f24a71d6209c8c8f2518b577f62ca71
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 4096
2 Ops.PARAM dtypes.bf16 [1] ParamArg(slot=0, dtype=dtypes.bf16)
3 Ops.CONST dtypes.weakint [] 32
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.SPECIAL dtypes.i32 [4] "lidx0"
6 Ops.CONST dtypes.weakint [] 4
7 Ops.CAST dtypes.i32 [6] dtypes.i32
8 Ops.SHR dtypes.i32 [5, 7]
9 Ops.CONST dtypes.weakint [] 6
10 Ops.CAST dtypes.i32 [9] dtypes.i32
11 Ops.SHL dtypes.i32 [8, 10]
12 Ops.SPECIAL dtypes.i32 [7] "gidx1"
13 Ops.CONST dtypes.weakint [] 10
14 Ops.CAST dtypes.i32 [13] dtypes.i32
15 Ops.SHL dtypes.i32 [12, 14]
16 Ops.ADD dtypes.i32 [11, 15]
17 Ops.SPECIAL dtypes.i32 [7] "gidx0"
18 Ops.SHL dtypes.i32 [17, 7]
19 Ops.CONST dtypes.weakint [] 15
20 Ops.CAST dtypes.i32 [19] dtypes.i32
21 Ops.AND dtypes.i32 [5, 20]
22 Ops.ADD dtypes.i32 [18, 21]
23 Ops.ADD dtypes.i32 [16, 22]
24 Ops.INDEX dtypes.bf16 [2, 23]
25 Ops.CONST dtypes.weakint [] 8
26 Ops.ALLOC dtypes.f32 [25] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
27 Ops.AFTER dtypes.f32 [26, 5, 12, 17]
28 Ops.CONST dtypes.weakint [] 0
29 Ops.CAST dtypes.i32 [28] dtypes.i32
30 Ops.INDEX dtypes.f32 [27, 29]
31 Ops.CONST dtypes.weakfloat [] 0.0
32 Ops.CAST dtypes.f32 [31] dtypes.f32
33 Ops.STORE dtypes.void [30, 32]
34 Ops.CONST dtypes.weakint [] 1
35 Ops.CAST dtypes.i32 [34] dtypes.i32
36 Ops.INDEX dtypes.f32 [27, 35]
37 Ops.STORE dtypes.void [36, 32]
38 Ops.CONST dtypes.weakint [] 2
39 Ops.CAST dtypes.i32 [38] dtypes.i32
40 Ops.INDEX dtypes.f32 [27, 39]
41 Ops.STORE dtypes.void [40, 32]
42 Ops.CONST dtypes.weakint [] 3
43 Ops.CAST dtypes.i32 [42] dtypes.i32
44 Ops.INDEX dtypes.f32 [27, 43]
45 Ops.STORE dtypes.void [44, 32]
46 Ops.INDEX dtypes.f32 [27, 7]
47 Ops.STORE dtypes.void [46, 32]
48 Ops.CONST dtypes.weakint [] 5
49 Ops.CAST dtypes.i32 [48] dtypes.i32
50 Ops.INDEX dtypes.f32 [27, 49]
51 Ops.STORE dtypes.void [50, 32]
52 Ops.INDEX dtypes.f32 [27, 10]
53 Ops.STORE dtypes.void [52, 32]
54 Ops.CONST dtypes.weakint [] 7
55 Ops.CAST dtypes.i32 [54] dtypes.i32
56 Ops.INDEX dtypes.f32 [27, 55]
57 Ops.STORE dtypes.void [56, 32]
58 Ops.RANGE dtypes.i32 [7] (AxisType.WEAK, 0)
59 Ops.AFTER dtypes.f32 [26, 33, 37, 41, 45, 47, 51, 53, 57, 58]
60 Ops.INDEX dtypes.f32 [59, 29]
61 Ops.PARAM dtypes.bf16 [1] ParamArg(slot=1, dtype=dtypes.bf16)
62 Ops.SHL dtypes.i32 [21, 10]
63 Ops.ADD dtypes.i32 [62, 15]
64 Ops.SHL dtypes.i32 [58, 7]
65 Ops.ADD dtypes.i32 [63, 64]
66 Ops.SHRINK dtypes.bf16 [61, 65, 7]
67 Ops.LOAD dtypes.bf16 [66]
68 Ops.INDEX dtypes.bf16 [67, 29]
69 Ops.INDEX dtypes.bf16 [67, 35]
70 Ops.INDEX dtypes.bf16 [67, 39]
71 Ops.INDEX dtypes.bf16 [67, 43]
72 Ops.ADD dtypes.i32 [65, 7]
73 Ops.SHRINK dtypes.bf16 [61, 72, 7]
74 Ops.LOAD dtypes.bf16 [73]
75 Ops.INDEX dtypes.bf16 [74, 29]
76 Ops.INDEX dtypes.bf16 [74, 35]
77 Ops.INDEX dtypes.bf16 [74, 39]
78 Ops.INDEX dtypes.bf16 [74, 43]
79 Ops.CAST dtypes.i32 [25] dtypes.i32
80 Ops.ADD dtypes.i32 [65, 79]
81 Ops.SHRINK dtypes.bf16 [61, 80, 7]
82 Ops.LOAD dtypes.bf16 [81]
83 Ops.INDEX dtypes.bf16 [82, 29]
84 Ops.INDEX dtypes.bf16 [82, 35]
85 Ops.INDEX dtypes.bf16 [82, 39]
86 Ops.INDEX dtypes.bf16 [82, 43]
87 Ops.CONST dtypes.weakint [] 12
88 Ops.CAST dtypes.i32 [87] dtypes.i32
89 Ops.ADD dtypes.i32 [65, 88]
90 Ops.SHRINK dtypes.bf16 [61, 89, 7]
91 Ops.LOAD dtypes.bf16 [90]
92 Ops.INDEX dtypes.bf16 [91, 29]
93 Ops.INDEX dtypes.bf16 [91, 35]
94 Ops.INDEX dtypes.bf16 [91, 39]
95 Ops.INDEX dtypes.bf16 [91, 43]
96 Ops.STACK dtypes.bf16 [68, 69, 70, 71, 75, 76, 77, 78, 83, 84, 85, 86, 92, 93, 94, 95]
97 Ops.PARAM dtypes.bf16 [1] ParamArg(slot=2, dtype=dtypes.bf16)
98 Ops.SHL dtypes.i32 [58, 14]
99 Ops.ADD dtypes.i32 [22, 98]
100 Ops.INDEX dtypes.bf16 [97, 99]
101 Ops.LOAD dtypes.bf16 [100]
102 Ops.CONST dtypes.weakint [] 64
103 Ops.CAST dtypes.i32 [102] dtypes.i32
104 Ops.ADD dtypes.i32 [99, 103]
105 Ops.INDEX dtypes.bf16 [97, 104]
106 Ops.LOAD dtypes.bf16 [105]
107 Ops.CONST dtypes.weakint [] 128
108 Ops.CAST dtypes.i32 [107] dtypes.i32
109 Ops.ADD dtypes.i32 [99, 108]
110 Ops.INDEX dtypes.bf16 [97, 109]
111 Ops.LOAD dtypes.bf16 [110]
112 Ops.CONST dtypes.weakint [] 192
113 Ops.CAST dtypes.i32 [112] dtypes.i32
114 Ops.ADD dtypes.i32 [99, 113]
115 Ops.INDEX dtypes.bf16 [97, 114]
116 Ops.LOAD dtypes.bf16 [115]
117 Ops.CONST dtypes.weakint [] 256
118 Ops.CAST dtypes.i32 [117] dtypes.i32
119 Ops.ADD dtypes.i32 [99, 118]
120 Ops.INDEX dtypes.bf16 [97, 119]
121 Ops.LOAD dtypes.bf16 [120]
122 Ops.CONST dtypes.weakint [] 320
123 Ops.CAST dtypes.i32 [122] dtypes.i32
124 Ops.ADD dtypes.i32 [99, 123]
125 Ops.INDEX dtypes.bf16 [97, 124]
126 Ops.LOAD dtypes.bf16 [125]
127 Ops.CONST dtypes.weakint [] 384
128 Ops.CAST dtypes.i32 [127] dtypes.i32
129 Ops.ADD dtypes.i32 [99, 128]
130 Ops.INDEX dtypes.bf16 [97, 129]
131 Ops.LOAD dtypes.bf16 [130]
132 Ops.CONST dtypes.weakint [] 448
133 Ops.CAST dtypes.i32 [132] dtypes.i32
134 Ops.ADD dtypes.i32 [99, 133]
135 Ops.INDEX dtypes.bf16 [97, 134]
136 Ops.LOAD dtypes.bf16 [135]
137 Ops.CONST dtypes.weakint [] 512
138 Ops.CAST dtypes.i32 [137] dtypes.i32
139 Ops.ADD dtypes.i32 [99, 138]
140 Ops.INDEX dtypes.bf16 [97, 139]
141 Ops.LOAD dtypes.bf16 [140]
142 Ops.CONST dtypes.weakint [] 576
143 Ops.CAST dtypes.i32 [142] dtypes.i32
144 Ops.ADD dtypes.i32 [99, 143]
145 Ops.INDEX dtypes.bf16 [97, 144]
146 Ops.LOAD dtypes.bf16 [145]
147 Ops.CONST dtypes.weakint [] 640
148 Ops.CAST dtypes.i32 [147] dtypes.i32
149 Ops.ADD dtypes.i32 [99, 148]
150 Ops.INDEX dtypes.bf16 [97, 149]
151 Ops.LOAD dtypes.bf16 [150]
152 Ops.CONST dtypes.weakint [] 704
153 Ops.CAST dtypes.i32 [152] dtypes.i32
154 Ops.ADD dtypes.i32 [99, 153]
155 Ops.INDEX dtypes.bf16 [97, 154]
156 Ops.LOAD dtypes.bf16 [155]
157 Ops.CONST dtypes.weakint [] 768
158 Ops.CAST dtypes.i32 [157] dtypes.i32
159 Ops.ADD dtypes.i32 [99, 158]
160 Ops.INDEX dtypes.bf16 [97, 159]
161 Ops.LOAD dtypes.bf16 [160]
162 Ops.CONST dtypes.weakint [] 832
163 Ops.CAST dtypes.i32 [162] dtypes.i32
164 Ops.ADD dtypes.i32 [99, 163]
165 Ops.INDEX dtypes.bf16 [97, 164]
166 Ops.LOAD dtypes.bf16 [165]
167 Ops.CONST dtypes.weakint [] 896
168 Ops.CAST dtypes.i32 [167] dtypes.i32
169 Ops.ADD dtypes.i32 [99, 168]
170 Ops.INDEX dtypes.bf16 [97, 169]
171 Ops.LOAD dtypes.bf16 [170]
172 Ops.CONST dtypes.weakint [] 960
173 Ops.CAST dtypes.i32 [172] dtypes.i32
174 Ops.ADD dtypes.i32 [99, 173]
175 Ops.INDEX dtypes.bf16 [97, 174]
176 Ops.LOAD dtypes.bf16 [175]
177 Ops.STACK dtypes.bf16 [101, 106, 111, 116, 121, 126, 131, 136, 141, 146, 151, 156, 161, 166, 171, 176]
178 Ops.LOAD dtypes.f32 [60]
179 Ops.INDEX dtypes.f32 [59, 35]
180 Ops.LOAD dtypes.f32 [179]
181 Ops.INDEX dtypes.f32 [59, 39]
182 Ops.LOAD dtypes.f32 [181]
183 Ops.INDEX dtypes.f32 [59, 43]
184 Ops.LOAD dtypes.f32 [183]
185 Ops.INDEX dtypes.f32 [59, 7]
186 Ops.LOAD dtypes.f32 [185]
187 Ops.INDEX dtypes.f32 [59, 49]
188 Ops.LOAD dtypes.f32 [187]
189 Ops.INDEX dtypes.f32 [59, 10]
190 Ops.LOAD dtypes.f32 [189]
191 Ops.INDEX dtypes.f32 [59, 55]
192 Ops.LOAD dtypes.f32 [191]
193 Ops.STACK dtypes.f32 [178, 180, 182, 184, 186, 188, 190, 192]
194 Ops.WMMA dtypes.f32 [96, 177, 193] ((16, 16, 16), dtypes.bf16, 32, None)
195 Ops.INDEX dtypes.f32 [194, 29]
196 Ops.STORE dtypes.void [60, 195]
197 Ops.INDEX dtypes.f32 [194, 35]
198 Ops.STORE dtypes.void [179, 197]
199 Ops.INDEX dtypes.f32 [194, 39]
200 Ops.STORE dtypes.void [181, 199]
201 Ops.INDEX dtypes.f32 [194, 43]
202 Ops.STORE dtypes.void [183, 201]
203 Ops.INDEX dtypes.f32 [194, 7]
204 Ops.STORE dtypes.void [185, 203]
205 Ops.INDEX dtypes.f32 [194, 49]
206 Ops.STORE dtypes.void [187, 205]
207 Ops.INDEX dtypes.f32 [194, 10]
208 Ops.STORE dtypes.void [189, 207]
209 Ops.INDEX dtypes.f32 [194, 55]
210 Ops.STORE dtypes.void [191, 209]
211 Ops.AFTER dtypes.void [0, 196, 198, 200, 202, 204, 206, 208, 210]
212 Ops.END dtypes.void [211, 58]
213 Ops.AFTER dtypes.f32 [26, 212]
214 Ops.INDEX dtypes.f32 [213, 29]
215 Ops.LOAD dtypes.f32 [214]
216 Ops.CMPNE dtypes.bool [215, 215]
217 Ops.BITCAST dtypes.u32 [215] dtypes.u32
218 Ops.CONST dtypes.weakint [] 4194304
219 Ops.CAST dtypes.u32 [218] dtypes.u32
220 Ops.OR dtypes.u32 [217, 219]
221 Ops.CONST dtypes.weakint [] 16
222 Ops.CAST dtypes.u32 [221] dtypes.u32
223 Ops.SHR dtypes.u32 [217, 222]
224 Ops.CAST dtypes.u32 [34] dtypes.u32
225 Ops.AND dtypes.u32 [223, 224]
226 Ops.ADD dtypes.u32 [217, 225]
227 Ops.CONST dtypes.weakint [] 32767
228 Ops.CAST dtypes.u32 [227] dtypes.u32
229 Ops.ADD dtypes.u32 [226, 228]
230 Ops.WHERE dtypes.u32 [216, 220, 229]
231 Ops.SHR dtypes.u32 [230, 222]
232 Ops.CAST dtypes.u16 [231] dtypes.u16
233 Ops.BITCAST dtypes.bf16 [232] dtypes.bf16
234 Ops.STORE dtypes.void [24, 233]
235 Ops.ADD dtypes.i32 [23, 108]
236 Ops.INDEX dtypes.bf16 [2, 235]
237 Ops.INDEX dtypes.f32 [213, 35]
238 Ops.LOAD dtypes.f32 [237]
239 Ops.CMPNE dtypes.bool [238, 238]
240 Ops.BITCAST dtypes.u32 [238] dtypes.u32
241 Ops.OR dtypes.u32 [240, 219]
242 Ops.SHR dtypes.u32 [240, 222]
243 Ops.AND dtypes.u32 [242, 224]
244 Ops.ADD dtypes.u32 [240, 243]
245 Ops.ADD dtypes.u32 [244, 228]
246 Ops.WHERE dtypes.u32 [239, 241, 245]
247 Ops.SHR dtypes.u32 [246, 222]
248 Ops.CAST dtypes.u16 [247] dtypes.u16
249 Ops.BITCAST dtypes.bf16 [248] dtypes.bf16
250 Ops.STORE dtypes.void [236, 249]
251 Ops.ADD dtypes.i32 [23, 118]
252 Ops.INDEX dtypes.bf16 [2, 251]
253 Ops.INDEX dtypes.f32 [213, 39]
254 Ops.LOAD dtypes.f32 [253]
255 Ops.CMPNE dtypes.bool [254, 254]
256 Ops.BITCAST dtypes.u32 [254] dtypes.u32
257 Ops.OR dtypes.u32 [256, 219]
258 Ops.SHR dtypes.u32 [256, 222]
259 Ops.AND dtypes.u32 [258, 224]
260 Ops.ADD dtypes.u32 [256, 259]
261 Ops.ADD dtypes.u32 [260, 228]
262 Ops.WHERE dtypes.u32 [255, 257, 261]
263 Ops.SHR dtypes.u32 [262, 222]
264 Ops.CAST dtypes.u16 [263] dtypes.u16
265 Ops.BITCAST dtypes.bf16 [264] dtypes.bf16
266 Ops.STORE dtypes.void [252, 265]
267 Ops.ADD dtypes.i32 [23, 128]
268 Ops.INDEX dtypes.bf16 [2, 267]
269 Ops.INDEX dtypes.f32 [213, 43]
270 Ops.LOAD dtypes.f32 [269]
271 Ops.CMPNE dtypes.bool [270, 270]
272 Ops.BITCAST dtypes.u32 [270] dtypes.u32
273 Ops.OR dtypes.u32 [272, 219]
274 Ops.SHR dtypes.u32 [272, 222]
275 Ops.AND dtypes.u32 [274, 224]
276 Ops.ADD dtypes.u32 [272, 275]
277 Ops.ADD dtypes.u32 [276, 228]
278 Ops.WHERE dtypes.u32 [271, 273, 277]
279 Ops.SHR dtypes.u32 [278, 222]
280 Ops.CAST dtypes.u16 [279] dtypes.u16
281 Ops.BITCAST dtypes.bf16 [280] dtypes.bf16
282 Ops.STORE dtypes.void [268, 281]
283 Ops.ADD dtypes.i32 [23, 138]
284 Ops.INDEX dtypes.bf16 [2, 283]
285 Ops.INDEX dtypes.f32 [213, 7]
286 Ops.LOAD dtypes.f32 [285]
287 Ops.CMPNE dtypes.bool [286, 286]
288 Ops.BITCAST dtypes.u32 [286] dtypes.u32
289 Ops.OR dtypes.u32 [288, 219]
290 Ops.SHR dtypes.u32 [288, 222]
291 Ops.AND dtypes.u32 [290, 224]
292 Ops.ADD dtypes.u32 [288, 291]
293 Ops.ADD dtypes.u32 [292, 228]
294 Ops.WHERE dtypes.u32 [287, 289, 293]
295 Ops.SHR dtypes.u32 [294, 222]
296 Ops.CAST dtypes.u16 [295] dtypes.u16
297 Ops.BITCAST dtypes.bf16 [296] dtypes.bf16
298 Ops.STORE dtypes.void [284, 297]
299 Ops.ADD dtypes.i32 [23, 148]
300 Ops.INDEX dtypes.bf16 [2, 299]
301 Ops.INDEX dtypes.f32 [213, 49]
302 Ops.LOAD dtypes.f32 [301]
303 Ops.CMPNE dtypes.bool [302, 302]
304 Ops.BITCAST dtypes.u32 [302] dtypes.u32
305 Ops.OR dtypes.u32 [304, 219]
306 Ops.SHR dtypes.u32 [304, 222]
307 Ops.AND dtypes.u32 [306, 224]
308 Ops.ADD dtypes.u32 [304, 307]
309 Ops.ADD dtypes.u32 [308, 228]
310 Ops.WHERE dtypes.u32 [303, 305, 309]
311 Ops.SHR dtypes.u32 [310, 222]
312 Ops.CAST dtypes.u16 [311] dtypes.u16
313 Ops.BITCAST dtypes.bf16 [312] dtypes.bf16
314 Ops.STORE dtypes.void [300, 313]
315 Ops.ADD dtypes.i32 [23, 158]
316 Ops.INDEX dtypes.bf16 [2, 315]
317 Ops.INDEX dtypes.f32 [213, 10]
318 Ops.LOAD dtypes.f32 [317]
319 Ops.CMPNE dtypes.bool [318, 318]
320 Ops.BITCAST dtypes.u32 [318] dtypes.u32
321 Ops.OR dtypes.u32 [320, 219]
322 Ops.SHR dtypes.u32 [320, 222]
323 Ops.AND dtypes.u32 [322, 224]
324 Ops.ADD dtypes.u32 [320, 323]
325 Ops.ADD dtypes.u32 [324, 228]
326 Ops.WHERE dtypes.u32 [319, 321, 325]
327 Ops.SHR dtypes.u32 [326, 222]
328 Ops.CAST dtypes.u16 [327] dtypes.u16
329 Ops.BITCAST dtypes.bf16 [328] dtypes.bf16
330 Ops.STORE dtypes.void [316, 329]
331 Ops.ADD dtypes.i32 [23, 168]
332 Ops.INDEX dtypes.bf16 [2, 331]
333 Ops.INDEX dtypes.f32 [213, 55]
334 Ops.LOAD dtypes.f32 [333]
335 Ops.CMPNE dtypes.bool [334, 334]
336 Ops.BITCAST dtypes.u32 [334] dtypes.u32
337 Ops.OR dtypes.u32 [336, 219]
338 Ops.SHR dtypes.u32 [336, 222]
339 Ops.AND dtypes.u32 [338, 224]
340 Ops.ADD dtypes.u32 [336, 339]
341 Ops.ADD dtypes.u32 [340, 228]
342 Ops.WHERE dtypes.u32 [335, 337, 341]
343 Ops.SHR dtypes.u32 [342, 222]
344 Ops.CAST dtypes.u16 [343] dtypes.u16
345 Ops.BITCAST dtypes.bf16 [344] dtypes.bf16
346 Ops.STORE dtypes.void [332, 345]
347 Ops.AFTER dtypes.void [0, 234, 250, 266, 282, 298, 314, 330, 346]
348 Ops.SINK dtypes.void [347] KernelInfo(name="r_4_4_32_2_2_2_2_4", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), optimized=True)
349 Ops.BUFFER dtypes.f32 [25] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
350 Ops.AFTER dtypes.f32 [349, 5, 12, 17]
351 Ops.INDEX dtypes.f32 [350, 29]
352 Ops.STORE dtypes.void [351, 32]
353 Ops.INDEX dtypes.f32 [350, 35]
354 Ops.STORE dtypes.void [353, 32]
355 Ops.INDEX dtypes.f32 [350, 39]
356 Ops.STORE dtypes.void [355, 32]
357 Ops.INDEX dtypes.f32 [350, 43]
358 Ops.STORE dtypes.void [357, 32]
359 Ops.INDEX dtypes.f32 [350, 7]
360 Ops.STORE dtypes.void [359, 32]
361 Ops.INDEX dtypes.f32 [350, 49]
362 Ops.STORE dtypes.void [361, 32]
363 Ops.INDEX dtypes.f32 [350, 10]
364 Ops.STORE dtypes.void [363, 32]
365 Ops.INDEX dtypes.f32 [350, 55]
366 Ops.STORE dtypes.void [365, 32]
367 Ops.AFTER dtypes.f32 [349, 352, 354, 356, 358, 360, 362, 364, 366, 58]
368 Ops.INDEX dtypes.f32 [367, 29]
369 Ops.LOAD dtypes.f32 [368]
370 Ops.INDEX dtypes.f32 [367, 35]
371 Ops.LOAD dtypes.f32 [370]
372 Ops.INDEX dtypes.f32 [367, 39]
373 Ops.LOAD dtypes.f32 [372]
374 Ops.INDEX dtypes.f32 [367, 43]
375 Ops.LOAD dtypes.f32 [374]
376 Ops.INDEX dtypes.f32 [367, 7]
377 Ops.LOAD dtypes.f32 [376]
378 Ops.INDEX dtypes.f32 [367, 49]
379 Ops.LOAD dtypes.f32 [378]
380 Ops.INDEX dtypes.f32 [367, 10]
381 Ops.LOAD dtypes.f32 [380]
382 Ops.INDEX dtypes.f32 [367, 55]
383 Ops.LOAD dtypes.f32 [382]
384 Ops.STACK dtypes.f32 [369, 371, 373, 375, 377, 379, 381, 383]
385 Ops.WMMA dtypes.f32 [96, 177, 384] ((16, 16, 16), dtypes.bf16, 32, None)
386 Ops.INDEX dtypes.f32 [385, 29]
387 Ops.STORE dtypes.void [368, 386]
388 Ops.INDEX dtypes.f32 [385, 35]
389 Ops.STORE dtypes.void [370, 388]
390 Ops.INDEX dtypes.f32 [385, 39]
391 Ops.STORE dtypes.void [372, 390]
392 Ops.INDEX dtypes.f32 [385, 43]
393 Ops.STORE dtypes.void [374, 392]
394 Ops.INDEX dtypes.f32 [385, 7]
395 Ops.STORE dtypes.void [376, 394]
396 Ops.INDEX dtypes.f32 [385, 49]
397 Ops.STORE dtypes.void [378, 396]
398 Ops.INDEX dtypes.f32 [385, 10]
399 Ops.STORE dtypes.void [380, 398]
400 Ops.INDEX dtypes.f32 [385, 55]
401 Ops.STORE dtypes.void [382, 400]
402 Ops.AFTER dtypes.void [0, 387, 389, 391, 393, 395, 397, 399, 401]
403 Ops.END dtypes.void [402, 58]
404 Ops.AFTER dtypes.f32 [349, 403]
405 Ops.INDEX dtypes.f32 [404, 29]
406 Ops.LOAD dtypes.f32 [405]
407 Ops.CMPNE dtypes.bool [406, 406]
408 Ops.BITCAST dtypes.u32 [406] dtypes.u32
409 Ops.OR dtypes.u32 [408, 219]
410 Ops.SHR dtypes.u32 [408, 222]
411 Ops.AND dtypes.u32 [410, 224]
412 Ops.ADD dtypes.u32 [408, 411]
413 Ops.ADD dtypes.u32 [412, 228]
414 Ops.WHERE dtypes.u32 [407, 409, 413]
415 Ops.SHR dtypes.u32 [414, 222]
416 Ops.CAST dtypes.u16 [415] dtypes.u16
417 Ops.BITCAST dtypes.bf16 [416] dtypes.bf16
418 Ops.STORE dtypes.void [24, 417]
419 Ops.INDEX dtypes.f32 [404, 35]
420 Ops.LOAD dtypes.f32 [419]
421 Ops.CMPNE dtypes.bool [420, 420]
422 Ops.BITCAST dtypes.u32 [420] dtypes.u32
423 Ops.OR dtypes.u32 [422, 219]
424 Ops.SHR dtypes.u32 [422, 222]
425 Ops.AND dtypes.u32 [424, 224]
426 Ops.ADD dtypes.u32 [422, 425]
427 Ops.ADD dtypes.u32 [426, 228]
428 Ops.WHERE dtypes.u32 [421, 423, 427]
429 Ops.SHR dtypes.u32 [428, 222]
430 Ops.CAST dtypes.u16 [429] dtypes.u16
431 Ops.BITCAST dtypes.bf16 [430] dtypes.bf16
432 Ops.STORE dtypes.void [236, 431]
433 Ops.INDEX dtypes.f32 [404, 39]
434 Ops.LOAD dtypes.f32 [433]
435 Ops.CMPNE dtypes.bool [434, 434]
436 Ops.BITCAST dtypes.u32 [434] dtypes.u32
437 Ops.OR dtypes.u32 [436, 219]
438 Ops.SHR dtypes.u32 [436, 222]
439 Ops.AND dtypes.u32 [438, 224]
440 Ops.ADD dtypes.u32 [436, 439]
441 Ops.ADD dtypes.u32 [440, 228]
442 Ops.WHERE dtypes.u32 [435, 437, 441]
443 Ops.SHR dtypes.u32 [442, 222]
444 Ops.CAST dtypes.u16 [443] dtypes.u16
445 Ops.BITCAST dtypes.bf16 [444] dtypes.bf16
446 Ops.STORE dtypes.void [252, 445]
447 Ops.INDEX dtypes.f32 [404, 43]
448 Ops.LOAD dtypes.f32 [447]
449 Ops.CMPNE dtypes.bool [448, 448]
450 Ops.BITCAST dtypes.u32 [448] dtypes.u32
451 Ops.OR dtypes.u32 [450, 219]
452 Ops.SHR dtypes.u32 [450, 222]
453 Ops.AND dtypes.u32 [452, 224]
454 Ops.ADD dtypes.u32 [450, 453]
455 Ops.ADD dtypes.u32 [454, 228]
456 Ops.WHERE dtypes.u32 [449, 451, 455]
457 Ops.SHR dtypes.u32 [456, 222]
458 Ops.CAST dtypes.u16 [457] dtypes.u16
459 Ops.BITCAST dtypes.bf16 [458] dtypes.bf16
460 Ops.STORE dtypes.void [268, 459]
461 Ops.INDEX dtypes.f32 [404, 7]
462 Ops.LOAD dtypes.f32 [461]
463 Ops.CMPNE dtypes.bool [462, 462]
464 Ops.BITCAST dtypes.u32 [462] dtypes.u32
465 Ops.OR dtypes.u32 [464, 219]
466 Ops.SHR dtypes.u32 [464, 222]
467 Ops.AND dtypes.u32 [466, 224]
468 Ops.ADD dtypes.u32 [464, 467]
469 Ops.ADD dtypes.u32 [468, 228]
470 Ops.WHERE dtypes.u32 [463, 465, 469]
471 Ops.SHR dtypes.u32 [470, 222]
472 Ops.CAST dtypes.u16 [471] dtypes.u16
473 Ops.BITCAST dtypes.bf16 [472] dtypes.bf16
474 Ops.STORE dtypes.void [284, 473]
475 Ops.INDEX dtypes.f32 [404, 49]
476 Ops.LOAD dtypes.f32 [475]
477 Ops.CMPNE dtypes.bool [476, 476]
478 Ops.BITCAST dtypes.u32 [476] dtypes.u32
479 Ops.OR dtypes.u32 [478, 219]
480 Ops.SHR dtypes.u32 [478, 222]
481 Ops.AND dtypes.u32 [480, 224]
482 Ops.ADD dtypes.u32 [478, 481]
483 Ops.ADD dtypes.u32 [482, 228]
484 Ops.WHERE dtypes.u32 [477, 479, 483]
485 Ops.SHR dtypes.u32 [484, 222]
486 Ops.CAST dtypes.u16 [485] dtypes.u16
487 Ops.BITCAST dtypes.bf16 [486] dtypes.bf16
488 Ops.STORE dtypes.void [300, 487]
489 Ops.INDEX dtypes.f32 [404, 10]
490 Ops.LOAD dtypes.f32 [489]
491 Ops.CMPNE dtypes.bool [490, 490]
492 Ops.BITCAST dtypes.u32 [490] dtypes.u32
493 Ops.OR dtypes.u32 [492, 219]
494 Ops.SHR dtypes.u32 [492, 222]
495 Ops.AND dtypes.u32 [494, 224]
496 Ops.ADD dtypes.u32 [492, 495]
497 Ops.ADD dtypes.u32 [496, 228]
498 Ops.WHERE dtypes.u32 [491, 493, 497]
499 Ops.SHR dtypes.u32 [498, 222]
500 Ops.CAST dtypes.u16 [499] dtypes.u16
501 Ops.BITCAST dtypes.bf16 [500] dtypes.bf16
502 Ops.STORE dtypes.void [316, 501]
503 Ops.INDEX dtypes.f32 [404, 55]
504 Ops.LOAD dtypes.f32 [503]
505 Ops.CMPNE dtypes.bool [504, 504]
506 Ops.BITCAST dtypes.u32 [504] dtypes.u32
507 Ops.OR dtypes.u32 [506, 219]
508 Ops.SHR dtypes.u32 [506, 222]
509 Ops.AND dtypes.u32 [508, 224]
510 Ops.ADD dtypes.u32 [506, 509]
511 Ops.ADD dtypes.u32 [510, 228]
512 Ops.WHERE dtypes.u32 [505, 507, 511]
513 Ops.SHR dtypes.u32 [512, 222]
514 Ops.CAST dtypes.u16 [513] dtypes.u16
515 Ops.BITCAST dtypes.bf16 [514] dtypes.bf16
516 Ops.STORE dtypes.void [332, 515]
517 Ops.AFTER dtypes.void [0, 418, 432, 446, 460, 474, 488, 502, 516]
518 Ops.SINK dtypes.void [517] KernelInfo(name="r_4_4_32_2_2_2_2_4", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), estimates=Estimates(ops=557056, lds=139264, mem=24576), optimized=True)
519 Ops.LINEAR dtypes.void [1, 2, 61, 97, 25, 349, 6, 7, 17, 12, 3, 4, 5, 350, 0, 28, 29, 351, 34, 35, 353, 38, 39, 355, 42, 43, 357, 359, 48, 49, 361, 9, 10, 363, 54, 55, 365, 31, 32, 13, 14, 87, 88, 107, 108, 19, 20, 112, 113, 117, 118, 122, 123, 127, 128, 132, 133, 137, 138, 142, 143, 102, 103, 147, 148, 152, 153, 157, 158, 79, 162, 163, 167, 168, 172, 173, 18, 21, 22, 15, 62, 63, 352, 354, 356, 358, 360, 362, 364, 366, 58, 98, 99, 109, 110, 111, 114, 115, 116, 119, 120, 121, 124, 125, 126, 129, 130, 131, 134, 135, 136, 139, 140, 141, 144, 145, 146, 104, 105, 106, 149, 150, 151, 154, 155, 156, 159, 160, 161, 164, 165, 166, 169, 170, 171, 174, 175, 176, 100, 101, 64, 65, 89, 90, 91, 72, 73, 74, 80, 81, 82, 66, 67, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 92, 93, 94, 95, 75, 76, 77, 78, 83, 84, 85, 86, 68, 69, 70, 71, 96, 177, 384, 385, 386, 388, 390, 392, 394, 396, 398, 400, 387, 389, 391, 393, 395, 397, 399, 401, 402, 403, 404, 405, 406, 419, 420, 433, 434, 447, 448, 461, 462, 475, 476, 489, 490, 503, 504, 8, 11, 16, 23, 235, 236, 251, 252, 267, 268, 283, 284, 299, 300, 315, 316, 331, 332, 24, 224, 221, 222, 227, 228, 218, 219, 408, 410, 411, 412, 413, 407, 409, 414, 415, 416, 422, 424, 425, 426, 427, 421, 423, 428, 429, 430, 436, 438, 439, 440, 441, 435, 437, 442, 443, 444, 450, 452, 453, 454, 455, 449, 451, 456, 457, 458, 464, 466, 467, 468, 469, 463, 465, 470, 471, 472, 478, 480, 481, 482, 483, 477, 479, 484, 485, 486, 492, 494, 495, 496, 497, 491, 493, 498, 499, 500, 506, 508, 509, 510, 511, 505, 507, 512, 513, 514, 417, 431, 445, 459, 473, 487, 501, 515, 432, 446, 460, 474, 488, 502, 516, 418, 517, 518]
520 Ops.SOURCE dtypes.void [] "typedef long unsigned int size_t;\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_local_size(unsigned int);\ntypedef __bf16 __bf164 __attribute__((ext_vector_type(4)));\nstatic inline __attribute__((device)) __bf164 make___bf164(__bf16 x, __bf16 y, __bf16 z, __bf16 w) { return { x, y, z, w }; }\ntypedef __bf16 __bf1616 __attribute__((ext_vector_type(16)));\nstatic inline __attribute__((device)) __bf1616 make___bf1616(__bf16 x, __bf16 y, __bf16 z, __bf16 w, __bf16 a, __bf16 b, __bf16 c, __bf16 d, __bf16 e, __bf16 f, __bf16 g, __bf16 h, __bf16 i, __bf16 j, __bf16 k, __bf16 l) { return { x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l }; }\ntypedef float float8 __attribute__((ext_vector_type(8)));\nstatic inline __attribute__((device)) float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return { x, y, z, w, a, b, c, d }; }\n#define __WMMA_16_16_16_bf16_f32 __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32\nextern \"C\" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 32))) r_4_4_32_2_2_2_2_4(__bf16* data0_4096, __bf16* data1_4096, __bf16* data2_4096) {\n  float buf0[8];\n  int gidx0 = __ockl_get_group_id(0); /* 4 */\n  int gidx1 = __ockl_get_group_id(1); /* 4 */\n  int lidx0 = __ockl_get_local_id(0); /* 32 */\n  int alu0 = (lidx0&15);\n  int alu1 = ((gidx0<<4)+alu0);\n  int alu2 = (gidx1<<10);\n  *(buf0+0) = 0.0f;\n  *(buf0+1) = 0.0f;\n  *(buf0+2) = 0.0f;\n  *(buf0+3) = 0.0f;\n  *(buf0+4) = 0.0f;\n  *(buf0+5) = 0.0f;\n  *(buf0+6) = 0.0f;\n  *(buf0+7) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 4; Lidx0++) {\n    int alu11 = (alu1+(Lidx0<<10));\n    __bf16 val0 = (*(data2_4096+(alu11+128)));\n    __bf16 val1 = (*(data2_4096+(alu11+192)));\n    __bf16 val2 = (*(data2_4096+(alu11+256)));\n    __bf16 val3 = (*(data2_4096+(alu11+320)));\n    __bf16 val4 = (*(data2_4096+(alu11+384)));\n    __bf16 val5 = (*(data2_4096+(alu11+448)));\n    __bf16 val6 = (*(data2_4096+(alu11+512)));\n    __bf16 val7 = (*(data2_4096+(alu11+576)));\n    __bf16 val8 = (*(data2_4096+(alu11+64)));\n    __bf16 val9 = (*(data2_4096+(alu11+640)));\n    __bf16 val10 = (*(data2_4096+(alu11+704)));\n    __bf16 val11 = (*(data2_4096+(alu11+768)));\n    __bf16 val12 = (*(data2_4096+(alu11+832)));\n    __bf16 val13 = (*(data2_4096+(alu11+896)));\n    __bf16 val14 = (*(data2_4096+(alu11+960)));\n    __bf16 val15 = (*(data2_4096+alu11));\n    int alu12 = ((alu0<<6)+alu2+(Lidx0<<4));\n    __bf164 val16 = (*((__bf164*)((data1_4096+(alu12+12)))));\n    __bf164 val17 = (*((__bf164*)((data1_4096+(alu12+4)))));\n    __bf164 val18 = (*((__bf164*)((data1_4096+(alu12+8)))));\n    __bf164 val19 = (*((__bf164*)((data1_4096+alu12))));\n    float8 wmma0 = __WMMA_16_16_16_bf16_f32(make___bf1616(val19.x,val19.y,val19.z,val19.w,val17.x,val17.y,val17.z,val17.w,val18.x,val18.y,val18.z,val18.w,val16.x,val16.y,val16.z,val16.w), make___bf1616(val15,val8,val0,val1,val2,val3,val4,val5,val6,val7,val9,val10,val11,val12,val13,val14), make_float8((*(buf0+0)),(*(buf0+1)),(*(buf0+2)),(*(buf0+3)),(*(buf0+4)),(*(buf0+5)),(*(buf0+6)),(*(buf0+7))));\n    *(buf0+0) = wmma0[0];\n    *(buf0+1) = wmma0[1];\n    *(buf0+2) = wmma0[2];\n    *(buf0+3) = wmma0[3];\n    *(buf0+4) = wmma0[4];\n    *(buf0+5) = wmma0[5];\n    *(buf0+6) = wmma0[6];\n    *(buf0+7) = wmma0[7];\n  }\n  float val20 = (*(buf0+0));\n  float val21 = (*(buf0+1));\n  float val22 = (*(buf0+2));\n  float val23 = (*(buf0+3));\n  float val24 = (*(buf0+4));\n  float val25 = (*(buf0+5));\n  float val26 = (*(buf0+6));\n  float val27 = (*(buf0+7));\n  int alu22 = (((lidx0>>4)<<6)+alu2+alu1);\n  unsigned int cast0 = __builtin_bit_cast(unsigned int, (float)(val20));\n  unsigned int alu23 = ((val20!=val20)?(cast0|4194304u):(cast0+((cast0>>16u)&1u)+32767u));\n  unsigned int cast1 = __builtin_bit_cast(unsigned int, (float)(val21));\n  unsigned int alu24 = ((val21!=val21)?(cast1|4194304u):(cast1+((cast1>>16u)&1u)+32767u));\n  unsigned int cast2 = __builtin_bit_cast(unsigned int, (float)(val22));\n  unsigned int alu25 = ((val22!=val22)?(cast2|4194304u):(cast2+((cast2>>16u)&1u)+32767u));\n  unsigned int cast3 = __builtin_bit_cast(unsigned int, (float)(val23));\n  unsigned int alu26 = ((val23!=val23)?(cast3|4194304u):(cast3+((cast3>>16u)&1u)+32767u));\n  unsigned int cast4 = __builtin_bit_cast(unsigned int, (float)(val24));\n  unsigned int alu27 = ((val24!=val24)?(cast4|4194304u):(cast4+((cast4>>16u)&1u)+32767u));\n  unsigned int cast5 = __builtin_bit_cast(unsigned int, (float)(val25));\n  unsigned int alu28 = ((val25!=val25)?(cast5|4194304u):(cast5+((cast5>>16u)&1u)+32767u));\n  unsigned int cast6 = __builtin_bit_cast(unsigned int, (float)(val26));\n  unsigned int alu29 = ((val26!=val26)?(cast6|4194304u):(cast6+((cast6>>16u)&1u)+32767u));\n  unsigned int cast7 = __builtin_bit_cast(unsigned int, (float)(val27));\n  unsigned int alu30 = ((val27!=val27)?(cast7|4194304u):(cast7+((cast7>>16u)&1u)+32767u));\n  *(data0_4096+(alu22+128)) = __builtin_bit_cast(__bf16, (unsigned short)(((unsigned short)((alu24>>16u)))));\n  *(data0_4096+(alu22+256)) = __builtin_bit_cast(__bf16, (unsigned short)(((unsigned short)((alu25>>16u)))));\n  *(data0_4096+(alu22+384)) = __builtin_bit_cast(__bf16, (unsigned short)(((unsigned short)((alu26>>16u)))));\n  *(data0_4096+(alu22+512)) = __builtin_bit_cast(__bf16, (unsigned short)(((unsigned short)((alu27>>16u)))));\n  *(data0_4096+(alu22+640)) = __builtin_bit_cast(__bf16, (unsigned short)(((unsigned short)((alu28>>16u)))));\n  *(data0_4096+(alu22+768)) = __builtin_bit_cast(__bf16, (unsigned short)(((unsigned short)((alu29>>16u)))));\n  *(data0_4096+(alu22+896)) = __builtin_bit_cast(__bf16, (unsigned short)(((unsigned short)((alu30>>16u)))));\n  *(data0_4096+alu22) = __builtin_bit_cast(__bf16, (unsigned short)(((unsigned short)((alu23>>16u)))));\n}"
521 Ops.PROGRAM dtypes.void [518, 519, 520] ProgramInfo(global_size=(4, 4, 1), local_size=(32, 1, 1), params=(%2, %61, %97), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="AMD", renderer="HIP", arch="gfx1100"))
522 Ops.SINK dtypes.void [348, 521]

