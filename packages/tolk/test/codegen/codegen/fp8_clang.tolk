# tolk D80 D151 golden=a4b00b5c02369bab1227240b956805bd
0 Ops.CONST dtypes.weakint [] 16
1 Ops.PARAM dtypes.u8 [0] ParamArg(slot=0, dtype=dtypes.u8)
2 Ops.STACK dtypes.void []
3 Ops.PARAM dtypes.i32 [2] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 16), name="block_lo", addrspace=AddrSpace.ALU)
4 Ops.PARAM dtypes.i32 [2] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 16), name="block_hi", addrspace=AddrSpace.ALU)
5 Ops.SUB dtypes.i32 [4, 3]
6 Ops.RANGE dtypes.i32 [5] (AxisType.WEAK, 0)
7 Ops.ADD dtypes.i32 [3, 6]
8 Ops.INDEX dtypes.u8 [1, 7]
9 Ops.PARAM dtypes.u8 [0] ParamArg(slot=1, dtype=dtypes.u8)
10 Ops.INDEX dtypes.u8 [9, 7]
11 Ops.LOAD dtypes.u8 [10]
12 Ops.CONST dtypes.weakint [] 128
13 Ops.CAST dtypes.u8 [12] dtypes.u8
14 Ops.AND dtypes.u8 [11, 13]
15 Ops.CAST dtypes.u32 [14] dtypes.u32
16 Ops.CONST dtypes.weakint [] 24
17 Ops.CAST dtypes.u32 [16] dtypes.u32
18 Ops.SHL dtypes.u32 [15, 17]
19 Ops.CONST dtypes.weakint [] 127
20 Ops.CAST dtypes.u8 [19] dtypes.u8
21 Ops.AND dtypes.u8 [11, 20]
22 Ops.CAST dtypes.u32 [21] dtypes.u32
23 Ops.CAST dtypes.u32 [19] dtypes.u32
24 Ops.CMPEQ dtypes.bool [22, 23]
25 Ops.CONST dtypes.weakint [] 20
26 Ops.CAST dtypes.u32 [25] dtypes.u32
27 Ops.SHL dtypes.u32 [22, 26]
28 Ops.CONST dtypes.weakint [] 2139095040
29 Ops.CAST dtypes.u32 [28] dtypes.u32
30 Ops.OR dtypes.u32 [27, 29]
31 Ops.CONST dtypes.weakint [] 3
32 Ops.CAST dtypes.u32 [31] dtypes.u32
33 Ops.SHR dtypes.u32 [22, 32]
34 Ops.CONST dtypes.weakint [] 0
35 Ops.CAST dtypes.u32 [34] dtypes.u32
36 Ops.CMPEQ dtypes.bool [33, 35]
37 Ops.CMPEQ dtypes.bool [22, 35]
38 Ops.CAST dtypes.f32 [22] dtypes.f32
39 Ops.BITCAST dtypes.u32 [38] dtypes.u32
40 Ops.CONST dtypes.weakint [] 4219469824
41 Ops.CAST dtypes.u32 [40] dtypes.u32
42 Ops.ADD dtypes.u32 [39, 41]
43 Ops.WHERE dtypes.u32 [37, 35, 42]
44 Ops.CONST dtypes.weakint [] 1006632960
45 Ops.CAST dtypes.u32 [44] dtypes.u32
46 Ops.ADD dtypes.u32 [27, 45]
47 Ops.WHERE dtypes.u32 [36, 43, 46]
48 Ops.WHERE dtypes.u32 [24, 30, 47]
49 Ops.OR dtypes.u32 [18, 48]
50 Ops.BITCAST dtypes.f32 [49] dtypes.f32
51 Ops.CONST dtypes.weakfloat [] 2.0
52 Ops.CAST dtypes.f32 [51] dtypes.f32
53 Ops.MUL dtypes.f32 [50, 52]
54 Ops.BITCAST dtypes.u32 [53] dtypes.u32
55 Ops.SHR dtypes.u32 [54, 17]
56 Ops.CAST dtypes.u32 [12] dtypes.u32
57 Ops.AND dtypes.u32 [55, 56]
58 Ops.CONST dtypes.weakint [] 2147483647
59 Ops.CAST dtypes.u32 [58] dtypes.u32
60 Ops.AND dtypes.u32 [54, 59]
61 Ops.CMPLT dtypes.bool [29, 60]
62 Ops.CMPEQ dtypes.bool [60, 29]
63 Ops.CONST dtypes.weakint [] 23
64 Ops.CAST dtypes.u32 [63] dtypes.u32
65 Ops.SHR dtypes.u32 [54, 64]
66 Ops.CONST dtypes.weakint [] 255
67 Ops.CAST dtypes.u32 [66] dtypes.u32
68 Ops.AND dtypes.u32 [65, 67]
69 Ops.CONST dtypes.weakint [] 121
70 Ops.CAST dtypes.u32 [69] dtypes.u32
71 Ops.CMPLT dtypes.bool [68, 70]
72 Ops.CONST dtypes.weakint [] 8388607
73 Ops.CAST dtypes.u32 [72] dtypes.u32
74 Ops.AND dtypes.u32 [54, 73]
75 Ops.CONST dtypes.weakint [] 8388608
76 Ops.CAST dtypes.u32 [75] dtypes.u32
77 Ops.OR dtypes.u32 [74, 76]
78 Ops.CONST dtypes.weakint [] 141
79 Ops.CAST dtypes.u32 [78] dtypes.u32
80 Ops.SUB dtypes.u32 [79, 68]
81 Ops.CONST dtypes.weakint [] 25
82 Ops.CAST dtypes.u32 [81] dtypes.u32
83 Ops.CMPLT dtypes.bool [80, 82]
84 Ops.WHERE dtypes.u32 [83, 80, 82]
85 Ops.CONST dtypes.weakint [] 1
86 Ops.CAST dtypes.u32 [85] dtypes.u32
87 Ops.WHERE dtypes.u32 [71, 84, 86]
88 Ops.SHR dtypes.u32 [77, 87]
89 Ops.CONST dtypes.weakint [] 4294967295
90 Ops.CAST dtypes.u32 [89] dtypes.u32
91 Ops.ADD dtypes.u32 [87, 90]
92 Ops.CONST dtypes.weakint [] 32
93 Ops.CAST dtypes.u32 [92] dtypes.u32
94 Ops.CMPLT dtypes.bool [91, 93]
95 Ops.CONST dtypes.weakint [] 31
96 Ops.CAST dtypes.u32 [95] dtypes.u32
97 Ops.AND dtypes.u32 [91, 96]
98 Ops.SHR dtypes.u32 [77, 97]
99 Ops.WHERE dtypes.u32 [94, 98, 35]
100 Ops.AND dtypes.u32 [99, 86]
101 Ops.SHL dtypes.u32 [86, 97]
102 Ops.WHERE dtypes.u32 [94, 101, 35]
103 Ops.ADD dtypes.u32 [102, 90]
104 Ops.AND dtypes.u32 [77, 103]
105 Ops.CMPNE dtypes.bool [104, 35]
106 Ops.CAST dtypes.u32 [105] dtypes.u32
107 Ops.AND dtypes.u32 [88, 86]
108 Ops.OR dtypes.u32 [106, 107]
109 Ops.AND dtypes.u32 [100, 108]
110 Ops.ADD dtypes.u32 [88, 109]
111 Ops.SHR dtypes.u32 [60, 26]
112 Ops.CONST dtypes.weakint [] 19
113 Ops.CAST dtypes.u32 [112] dtypes.u32
114 Ops.SHR dtypes.u32 [60, 113]
115 Ops.AND dtypes.u32 [114, 86]
116 Ops.CONST dtypes.weakint [] 524287
117 Ops.CAST dtypes.u32 [116] dtypes.u32
118 Ops.AND dtypes.u32 [54, 117]
119 Ops.CMPNE dtypes.bool [118, 35]
120 Ops.CAST dtypes.u32 [119] dtypes.u32
121 Ops.AND dtypes.u32 [111, 86]
122 Ops.OR dtypes.u32 [120, 121]
123 Ops.AND dtypes.u32 [115, 122]
124 Ops.ADD dtypes.u32 [111, 123]
125 Ops.CONST dtypes.weakint [] 4294966336
126 Ops.CAST dtypes.u32 [125] dtypes.u32
127 Ops.ADD dtypes.u32 [124, 126]
128 Ops.XOR dtypes.u32 [127, 90]
129 Ops.CONST dtypes.weakint [] 4294967169
130 Ops.CAST dtypes.u32 [129] dtypes.u32
131 Ops.CMPLT dtypes.bool [128, 130]
132 Ops.WHERE dtypes.u32 [131, 130, 128]
133 Ops.XOR dtypes.u32 [132, 90]
134 Ops.WHERE dtypes.u32 [71, 110, 133]
135 Ops.WHERE dtypes.u32 [62, 23, 134]
136 Ops.WHERE dtypes.u32 [61, 23, 135]
137 Ops.OR dtypes.u32 [57, 136]
138 Ops.CAST dtypes.u8 [137] dtypes.u8
139 Ops.AND dtypes.u8 [138, 13]
140 Ops.CAST dtypes.u32 [139] dtypes.u32
141 Ops.SHL dtypes.u32 [140, 17]
142 Ops.AND dtypes.u8 [138, 20]
143 Ops.CAST dtypes.u32 [142] dtypes.u32
144 Ops.CMPEQ dtypes.bool [143, 23]
145 Ops.SHL dtypes.u32 [143, 26]
146 Ops.OR dtypes.u32 [145, 29]
147 Ops.SHR dtypes.u32 [143, 32]
148 Ops.CMPEQ dtypes.bool [147, 35]
149 Ops.CMPEQ dtypes.bool [143, 35]
150 Ops.CAST dtypes.f32 [143] dtypes.f32
151 Ops.BITCAST dtypes.u32 [150] dtypes.u32
152 Ops.ADD dtypes.u32 [151, 41]
153 Ops.WHERE dtypes.u32 [149, 35, 152]
154 Ops.ADD dtypes.u32 [145, 45]
155 Ops.WHERE dtypes.u32 [148, 153, 154]
156 Ops.WHERE dtypes.u32 [144, 146, 155]
157 Ops.OR dtypes.u32 [141, 156]
158 Ops.SHR dtypes.u32 [157, 17]
159 Ops.AND dtypes.u32 [158, 56]
160 Ops.AND dtypes.u32 [157, 59]
161 Ops.CMPLT dtypes.bool [29, 160]
162 Ops.CMPEQ dtypes.bool [160, 29]
163 Ops.SHR dtypes.u32 [157, 64]
164 Ops.AND dtypes.u32 [163, 67]
165 Ops.CMPLT dtypes.bool [164, 70]
166 Ops.AND dtypes.u32 [157, 73]
167 Ops.OR dtypes.u32 [166, 76]
168 Ops.SUB dtypes.u32 [79, 164]
169 Ops.CMPLT dtypes.bool [168, 82]
170 Ops.WHERE dtypes.u32 [169, 168, 82]
171 Ops.WHERE dtypes.u32 [165, 170, 86]
172 Ops.SHR dtypes.u32 [167, 171]
173 Ops.ADD dtypes.u32 [171, 90]
174 Ops.CMPLT dtypes.bool [173, 93]
175 Ops.AND dtypes.u32 [173, 96]
176 Ops.SHR dtypes.u32 [167, 175]
177 Ops.WHERE dtypes.u32 [174, 176, 35]
178 Ops.AND dtypes.u32 [177, 86]
179 Ops.SHL dtypes.u32 [86, 175]
180 Ops.WHERE dtypes.u32 [174, 179, 35]
181 Ops.ADD dtypes.u32 [180, 90]
182 Ops.AND dtypes.u32 [167, 181]
183 Ops.CMPNE dtypes.bool [182, 35]
184 Ops.CAST dtypes.u32 [183] dtypes.u32
185 Ops.AND dtypes.u32 [172, 86]
186 Ops.OR dtypes.u32 [184, 185]
187 Ops.AND dtypes.u32 [178, 186]
188 Ops.ADD dtypes.u32 [172, 187]
189 Ops.SHR dtypes.u32 [160, 26]
190 Ops.SHR dtypes.u32 [160, 113]
191 Ops.AND dtypes.u32 [190, 86]
192 Ops.AND dtypes.u32 [157, 117]
193 Ops.CMPNE dtypes.bool [192, 35]
194 Ops.CAST dtypes.u32 [193] dtypes.u32
195 Ops.AND dtypes.u32 [189, 86]
196 Ops.OR dtypes.u32 [194, 195]
197 Ops.AND dtypes.u32 [191, 196]
198 Ops.ADD dtypes.u32 [189, 197]
199 Ops.ADD dtypes.u32 [198, 126]
200 Ops.XOR dtypes.u32 [199, 90]
201 Ops.CMPLT dtypes.bool [200, 130]
202 Ops.WHERE dtypes.u32 [201, 130, 200]
203 Ops.XOR dtypes.u32 [202, 90]
204 Ops.WHERE dtypes.u32 [165, 188, 203]
205 Ops.WHERE dtypes.u32 [162, 23, 204]
206 Ops.WHERE dtypes.u32 [161, 23, 205]
207 Ops.OR dtypes.u32 [159, 206]
208 Ops.CAST dtypes.u8 [207] dtypes.u8
209 Ops.STORE dtypes.void [8, 208]
210 Ops.END dtypes.void [209, 6]
211 Ops.SINK dtypes.void [210] KernelInfo(name="E_16", optimized=True, split=(16, 2, 3))
212 Ops.PARAM dtypes.u8 [92] ParamArg(slot=0, dtype=dtypes.u8, name="args")
213 Ops.BITCAST dtypes.u64 [212] dtypes.u64
214 Ops.CAST dtypes.i32 [34] dtypes.i32
215 Ops.INDEX dtypes.u64 [213, 214]
216 Ops.LOAD dtypes.u64 [215]
217 Ops.REFPARAM dtypes.u8 [0, 216] ParamArg(slot=0, dtype=dtypes.u8)
218 Ops.BITCAST dtypes.i32 [212] dtypes.i32
219 Ops.CONST dtypes.weakint [] 4
220 Ops.CAST dtypes.i32 [219] dtypes.i32
221 Ops.INDEX dtypes.i32 [218, 220]
222 Ops.LOAD dtypes.i32 [221]
223 Ops.CONST dtypes.weakint [] 5
224 Ops.CAST dtypes.i32 [223] dtypes.i32
225 Ops.INDEX dtypes.i32 [218, 224]
226 Ops.LOAD dtypes.i32 [225]
227 Ops.SUB dtypes.i32 [226, 222]
228 Ops.RANGE dtypes.i32 [227] (AxisType.WEAK, 0)
229 Ops.ADD dtypes.i32 [222, 228]
230 Ops.INDEX dtypes.u8 [217, 229]
231 Ops.CAST dtypes.i32 [85] dtypes.i32
232 Ops.INDEX dtypes.u64 [213, 231]
233 Ops.LOAD dtypes.u64 [232]
234 Ops.REFPARAM dtypes.u8 [0, 233] ParamArg(slot=1, dtype=dtypes.u8)
235 Ops.INDEX dtypes.u8 [234, 229]
236 Ops.LOAD dtypes.u8 [235]
237 Ops.AND dtypes.u8 [236, 13]
238 Ops.CAST dtypes.u32 [237] dtypes.u32
239 Ops.SHL dtypes.u32 [238, 17]
240 Ops.AND dtypes.u8 [236, 20]
241 Ops.CAST dtypes.u32 [240] dtypes.u32
242 Ops.CMPEQ dtypes.bool [241, 23]
243 Ops.SHL dtypes.u32 [241, 26]
244 Ops.OR dtypes.u32 [243, 29]
245 Ops.SHR dtypes.u32 [241, 32]
246 Ops.CMPEQ dtypes.bool [245, 35]
247 Ops.CMPEQ dtypes.bool [241, 35]
248 Ops.CAST dtypes.f32 [241] dtypes.f32
249 Ops.BITCAST dtypes.u32 [248] dtypes.u32
250 Ops.ADD dtypes.u32 [249, 41]
251 Ops.WHERE dtypes.u32 [247, 35, 250]
252 Ops.ADD dtypes.u32 [243, 45]
253 Ops.WHERE dtypes.u32 [246, 251, 252]
254 Ops.WHERE dtypes.u32 [242, 244, 253]
255 Ops.OR dtypes.u32 [239, 254]
256 Ops.BITCAST dtypes.f32 [255] dtypes.f32
257 Ops.MUL dtypes.f32 [256, 52]
258 Ops.BITCAST dtypes.u32 [257] dtypes.u32
259 Ops.SHR dtypes.u32 [258, 17]
260 Ops.AND dtypes.u32 [259, 56]
261 Ops.AND dtypes.u32 [258, 59]
262 Ops.CMPLT dtypes.bool [29, 261]
263 Ops.CMPEQ dtypes.bool [261, 29]
264 Ops.SHR dtypes.u32 [258, 64]
265 Ops.AND dtypes.u32 [264, 67]
266 Ops.CMPLT dtypes.bool [265, 70]
267 Ops.AND dtypes.u32 [258, 73]
268 Ops.OR dtypes.u32 [267, 76]
269 Ops.SUB dtypes.u32 [79, 265]
270 Ops.CMPLT dtypes.bool [269, 82]
271 Ops.WHERE dtypes.u32 [270, 269, 82]
272 Ops.WHERE dtypes.u32 [266, 271, 86]
273 Ops.SHR dtypes.u32 [268, 272]
274 Ops.ADD dtypes.u32 [272, 90]
275 Ops.CMPLT dtypes.bool [274, 93]
276 Ops.AND dtypes.u32 [274, 96]
277 Ops.SHR dtypes.u32 [268, 276]
278 Ops.WHERE dtypes.u32 [275, 277, 35]
279 Ops.AND dtypes.u32 [278, 86]
280 Ops.SHL dtypes.u32 [86, 276]
281 Ops.WHERE dtypes.u32 [275, 280, 35]
282 Ops.ADD dtypes.u32 [281, 90]
283 Ops.AND dtypes.u32 [268, 282]
284 Ops.CMPNE dtypes.bool [283, 35]
285 Ops.CAST dtypes.u32 [284] dtypes.u32
286 Ops.AND dtypes.u32 [273, 86]
287 Ops.OR dtypes.u32 [285, 286]
288 Ops.AND dtypes.u32 [279, 287]
289 Ops.ADD dtypes.u32 [273, 288]
290 Ops.SHR dtypes.u32 [261, 26]
291 Ops.SHR dtypes.u32 [261, 113]
292 Ops.AND dtypes.u32 [291, 86]
293 Ops.AND dtypes.u32 [258, 117]
294 Ops.CMPNE dtypes.bool [293, 35]
295 Ops.CAST dtypes.u32 [294] dtypes.u32
296 Ops.AND dtypes.u32 [290, 86]
297 Ops.OR dtypes.u32 [295, 296]
298 Ops.AND dtypes.u32 [292, 297]
299 Ops.ADD dtypes.u32 [290, 298]
300 Ops.ADD dtypes.u32 [299, 126]
301 Ops.XOR dtypes.u32 [300, 90]
302 Ops.CMPLT dtypes.bool [301, 130]
303 Ops.WHERE dtypes.u32 [302, 130, 301]
304 Ops.XOR dtypes.u32 [303, 90]
305 Ops.WHERE dtypes.u32 [266, 289, 304]
306 Ops.WHERE dtypes.u32 [263, 23, 305]
307 Ops.WHERE dtypes.u32 [262, 23, 306]
308 Ops.OR dtypes.u32 [260, 307]
309 Ops.CAST dtypes.u8 [308] dtypes.u8
310 Ops.AND dtypes.u8 [309, 13]
311 Ops.CAST dtypes.u32 [310] dtypes.u32
312 Ops.SHL dtypes.u32 [311, 17]
313 Ops.AND dtypes.u8 [309, 20]
314 Ops.CAST dtypes.u32 [313] dtypes.u32
315 Ops.CMPEQ dtypes.bool [314, 23]
316 Ops.SHL dtypes.u32 [314, 26]
317 Ops.OR dtypes.u32 [316, 29]
318 Ops.SHR dtypes.u32 [314, 32]
319 Ops.CMPEQ dtypes.bool [318, 35]
320 Ops.CMPEQ dtypes.bool [314, 35]
321 Ops.CAST dtypes.f32 [314] dtypes.f32
322 Ops.BITCAST dtypes.u32 [321] dtypes.u32
323 Ops.ADD dtypes.u32 [322, 41]
324 Ops.WHERE dtypes.u32 [320, 35, 323]
325 Ops.ADD dtypes.u32 [316, 45]
326 Ops.WHERE dtypes.u32 [319, 324, 325]
327 Ops.WHERE dtypes.u32 [315, 317, 326]
328 Ops.OR dtypes.u32 [312, 327]
329 Ops.SHR dtypes.u32 [328, 17]
330 Ops.AND dtypes.u32 [329, 56]
331 Ops.AND dtypes.u32 [328, 59]
332 Ops.CMPLT dtypes.bool [29, 331]
333 Ops.CMPEQ dtypes.bool [331, 29]
334 Ops.SHR dtypes.u32 [328, 64]
335 Ops.AND dtypes.u32 [334, 67]
336 Ops.CMPLT dtypes.bool [335, 70]
337 Ops.AND dtypes.u32 [328, 73]
338 Ops.OR dtypes.u32 [337, 76]
339 Ops.SUB dtypes.u32 [79, 335]
340 Ops.CMPLT dtypes.bool [339, 82]
341 Ops.WHERE dtypes.u32 [340, 339, 82]
342 Ops.WHERE dtypes.u32 [336, 341, 86]
343 Ops.SHR dtypes.u32 [338, 342]
344 Ops.ADD dtypes.u32 [342, 90]
345 Ops.CMPLT dtypes.bool [344, 93]
346 Ops.AND dtypes.u32 [344, 96]
347 Ops.SHR dtypes.u32 [338, 346]
348 Ops.WHERE dtypes.u32 [345, 347, 35]
349 Ops.AND dtypes.u32 [348, 86]
350 Ops.SHL dtypes.u32 [86, 346]
351 Ops.WHERE dtypes.u32 [345, 350, 35]
352 Ops.ADD dtypes.u32 [351, 90]
353 Ops.AND dtypes.u32 [338, 352]
354 Ops.CMPNE dtypes.bool [353, 35]
355 Ops.CAST dtypes.u32 [354] dtypes.u32
356 Ops.AND dtypes.u32 [343, 86]
357 Ops.OR dtypes.u32 [355, 356]
358 Ops.AND dtypes.u32 [349, 357]
359 Ops.ADD dtypes.u32 [343, 358]
360 Ops.SHR dtypes.u32 [331, 26]
361 Ops.SHR dtypes.u32 [331, 113]
362 Ops.AND dtypes.u32 [361, 86]
363 Ops.AND dtypes.u32 [328, 117]
364 Ops.CMPNE dtypes.bool [363, 35]
365 Ops.CAST dtypes.u32 [364] dtypes.u32
366 Ops.AND dtypes.u32 [360, 86]
367 Ops.OR dtypes.u32 [365, 366]
368 Ops.AND dtypes.u32 [362, 367]
369 Ops.ADD dtypes.u32 [360, 368]
370 Ops.ADD dtypes.u32 [369, 126]
371 Ops.XOR dtypes.u32 [370, 90]
372 Ops.CMPLT dtypes.bool [371, 130]
373 Ops.WHERE dtypes.u32 [372, 130, 371]
374 Ops.XOR dtypes.u32 [373, 90]
375 Ops.WHERE dtypes.u32 [336, 359, 374]
376 Ops.WHERE dtypes.u32 [333, 23, 375]
377 Ops.WHERE dtypes.u32 [332, 23, 376]
378 Ops.OR dtypes.u32 [330, 377]
379 Ops.CAST dtypes.u8 [378] dtypes.u8
380 Ops.STORE dtypes.void [230, 379]
381 Ops.END dtypes.void [380, 228]
382 Ops.SINK dtypes.void [381] KernelInfo(name="E_16", estimates=Estimates(ops=2032, lds=32, mem=32), optimized=True, split=(16, 2, 3))
383 Ops.LINEAR dtypes.void [92, 212, 219, 220, 218, 221, 222, 223, 224, 225, 226, 34, 214, 213, 215, 216, 85, 231, 232, 233, 0, 217, 234, 51, 52, 35, 86, 44, 45, 69, 70, 19, 23, 12, 56, 78, 79, 112, 113, 25, 26, 28, 29, 58, 59, 63, 64, 16, 17, 81, 82, 66, 67, 31, 32, 95, 96, 93, 40, 41, 125, 126, 129, 130, 89, 90, 116, 117, 72, 73, 75, 76, 20, 13, 227, 228, 229, 235, 236, 230, 240, 241, 248, 237, 238, 249, 250, 243, 252, 239, 245, 247, 242, 246, 244, 251, 253, 254, 255, 256, 257, 258, 293, 294, 295, 264, 265, 266, 269, 270, 271, 272, 274, 276, 280, 275, 281, 282, 267, 268, 283, 284, 285, 261, 291, 290, 296, 297, 292, 298, 299, 300, 277, 273, 286, 287, 278, 279, 288, 289, 259, 262, 301, 302, 263, 303, 304, 260, 305, 306, 307, 308, 309, 313, 314, 321, 310, 311, 322, 323, 316, 325, 312, 318, 320, 315, 319, 317, 324, 326, 327, 328, 363, 364, 365, 334, 335, 336, 339, 340, 341, 342, 344, 346, 350, 345, 351, 352, 337, 338, 353, 354, 355, 331, 361, 360, 366, 367, 362, 368, 369, 370, 347, 343, 356, 357, 348, 349, 358, 359, 329, 332, 371, 372, 333, 373, 374, 330, 375, 376, 377, 378, 379, 380, 381, 382]
384 Ops.SOURCE dtypes.void [] "\n__attribute__((min_vector_width(512))) void E_16(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = (val0+Lidx0);\n    unsigned char val4 = (*(((unsigned char*)(val3))+alu0));\n    unsigned int cast0 = ((unsigned int)((val4&((unsigned char)(127u)))));\n    unsigned int alu1 = (cast0<<20u);\n    unsigned int alu2 = ((cast0==0u)?0u:(__builtin_bit_cast(unsigned int, (float)(((float)(cast0))))+4219469824u));\n    unsigned int alu3 = (((cast0>>3u)==0u)?alu2:(alu1+1006632960u));\n    unsigned int alu4 = ((cast0==127u)?(alu1|2139095040u):alu3);\n    unsigned int cast1 = __builtin_bit_cast(unsigned int, (float)((__builtin_bit_cast(float, (unsigned int)(((((unsigned int)((val4&((unsigned char)(128u)))))<<24u)|alu4)))*2.0f)));\n    unsigned int alu5 = ((cast1>>23u)&255u);\n    _Bool alu6 = (alu5<121u);\n    unsigned int alu7 = (141u-alu5);\n    unsigned int alu8 = ((alu7<25u)?alu7:25u);\n    unsigned int alu9 = (alu6?alu8:1u);\n    unsigned int alu10 = (alu9+4294967295u);\n    unsigned int alu11 = (alu10&31u);\n    _Bool alu12 = (alu10<32u);\n    unsigned int alu13 = (alu12?(1u<<alu11):0u);\n    unsigned int alu14 = ((cast1&8388607u)|8388608u);\n    unsigned int alu15 = (cast1&2147483647u);\n    unsigned int alu16 = (alu15>>20u);\n    unsigned int alu17 = (alu14>>alu9);\n    unsigned int alu18 = (alu12?(alu14>>alu11):0u);\n    unsigned int alu19 = ((alu16+((alu15>>19u)&1u&(((unsigned int)(((cast1&524287u)!=0u)))|(alu16&1u)))+4294966336u)^4294967295u);\n    unsigned int alu20 = ((alu19<4294967169u)?4294967169u:alu19);\n    unsigned int alu21 = (alu6?(alu17+(alu18&1u&(((unsigned int)(((alu14&(alu13+4294967295u))!=0u)))|(alu17&1u)))):(alu20^4294967295u));\n    unsigned int alu22 = ((alu15==2139095040u)?127u:alu21);\n    unsigned int alu23 = ((2139095040u<alu15)?127u:alu22);\n    unsigned char cast2 = ((unsigned char)((((cast1>>24u)&128u)|alu23)));\n    unsigned int cast3 = ((unsigned int)((cast2&((unsigned char)(127u)))));\n    unsigned int alu24 = (cast3<<20u);\n    unsigned int alu25 = ((cast3==0u)?0u:(__builtin_bit_cast(unsigned int, (float)(((float)(cast3))))+4219469824u));\n    unsigned int alu26 = (((cast3>>3u)==0u)?alu25:(alu24+1006632960u));\n    unsigned int alu27 = ((cast3==127u)?(alu24|2139095040u):alu26);\n    unsigned int alu28 = ((((unsigned int)((cast2&((unsigned char)(128u)))))<<24u)|alu27);\n    unsigned int alu29 = ((alu28>>23u)&255u);\n    _Bool alu30 = (alu29<121u);\n    unsigned int alu31 = (141u-alu29);\n    unsigned int alu32 = ((alu31<25u)?alu31:25u);\n    unsigned int alu33 = (alu30?alu32:1u);\n    unsigned int alu34 = (alu33+4294967295u);\n    unsigned int alu35 = (alu34&31u);\n    _Bool alu36 = (alu34<32u);\n    unsigned int alu37 = (alu36?(1u<<alu35):0u);\n    unsigned int alu38 = ((alu28&8388607u)|8388608u);\n    unsigned int alu39 = (alu28&2147483647u);\n    unsigned int alu40 = (alu39>>20u);\n    unsigned int alu41 = (alu38>>alu33);\n    unsigned int alu42 = (alu36?(alu38>>alu35):0u);\n    unsigned int alu43 = ((alu40+((alu39>>19u)&1u&(((unsigned int)(((alu28&524287u)!=0u)))|(alu40&1u)))+4294966336u)^4294967295u);\n    unsigned int alu44 = ((alu43<4294967169u)?4294967169u:alu43);\n    unsigned int alu45 = (alu30?(alu41+(alu42&1u&(((unsigned int)(((alu38&(alu37+4294967295u))!=0u)))|(alu41&1u)))):(alu44^4294967295u));\n    unsigned int alu46 = ((alu39==2139095040u)?127u:alu45);\n    unsigned int alu47 = ((2139095040u<alu39)?127u:alu46);\n    *(((unsigned char*)(val2))+alu0) = ((unsigned char)((((alu28>>24u)&128u)|alu47)));\n  }\n}\n"
385 Ops.PROGRAM dtypes.void [382, 383, 384] ProgramInfo(global_size=(16, 1, 1), params=(%1, %9, %3, %4), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
386 Ops.SINK dtypes.void [211, 385]

