# tolk D31 D126 D413 golden=fe372a949d5e7b5e162435042d4febe1
0 Ops.STACK dtypes.void []
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakfloat [] inf
3 Ops.CMPEQ dtypes.bool [1, 2]
4 Ops.CONST dtypes.bool [] False
5 Ops.CMPEQ dtypes.bool [3, 4]
6 Ops.CMPEQ dtypes.bool [1, 1]
7 Ops.CMPEQ dtypes.bool [6, 4]
8 Ops.CONST dtypes.weakfloat [] nan
9 Ops.CAST dtypes.f32 [8] dtypes.f32
10 Ops.CONST dtypes.weakfloat [] -inf
11 Ops.CMPEQ dtypes.bool [1, 10]
12 Ops.CMPEQ dtypes.bool [11, 4]
13 Ops.CONST dtypes.weakfloat [] 0.0
14 Ops.CAST dtypes.f32 [13] dtypes.f32
15 Ops.WHERE dtypes.f32 [12, 1, 14]
16 Ops.WHERE dtypes.f32 [7, 14, 15]
17 Ops.WHERE dtypes.f32 [5, 16, 14]
18 Ops.CMPEQ dtypes.bool [17, 13]
19 Ops.CMPEQ dtypes.bool [18, 4]
20 Ops.CMPLT dtypes.bool [17, 13]
21 Ops.CONST dtypes.weakfloat [] -1.0
22 Ops.CAST dtypes.f32 [21] dtypes.f32
23 Ops.CONST dtypes.weakfloat [] 1.0
24 Ops.CAST dtypes.f32 [23] dtypes.f32
25 Ops.WHERE dtypes.f32 [20, 22, 24]
26 Ops.MUL dtypes.f32 [17, 25]
27 Ops.CONST dtypes.weakfloat [] 100.0
28 Ops.CMPLT dtypes.bool [26, 27]
29 Ops.CONST dtypes.weakfloat [] 0.6366197723675814
30 Ops.MUL dtypes.f32 [26, 29]
31 Ops.CMPLT dtypes.bool [30, 13]
32 Ops.CONST dtypes.weakfloat [] -0.5
33 Ops.CAST dtypes.f32 [32] dtypes.f32
34 Ops.CONST dtypes.weakfloat [] 0.5
35 Ops.CAST dtypes.f32 [34] dtypes.f32
36 Ops.WHERE dtypes.f32 [31, 33, 35]
37 Ops.ADD dtypes.f32 [30, 36]
38 Ops.CAST dtypes.i32 [37] dtypes.i32
39 Ops.BITCAST dtypes.u32 [26] dtypes.u32
40 Ops.CONST dtypes.weakint [] 2155872255
41 Ops.AND dtypes.u32 [39, 40]
42 Ops.CONST dtypes.weakint [] 1056964608
43 Ops.OR dtypes.u32 [41, 42]
44 Ops.BITCAST dtypes.f32 [43] dtypes.f32
45 Ops.CONST dtypes.weakfloat [] 16777216.0
46 Ops.MUL dtypes.f32 [44, 45]
47 Ops.CAST dtypes.u64 [46] dtypes.u64
48 Ops.CONST dtypes.weakint [] 4294967295
49 Ops.AND dtypes.u64 [47, 48]
50 Ops.CONST dtypes.weakint [] 8388608
51 Ops.DIV dtypes.u32 [39, 50]
52 Ops.CONST dtypes.weakint [] 255
53 Ops.AND dtypes.u32 [51, 52]
54 Ops.CONST dtypes.weakint [] 127
55 Ops.CONST dtypes.weakint [] -1
56 Ops.MUL dtypes.weakint [54, 55]
57 Ops.ADD dtypes.u32 [53, 56]
58 Ops.CONST dtypes.weakint [] 1
59 Ops.ADD dtypes.u32 [57, 58]
60 Ops.CAST dtypes.i32 [59] dtypes.i32
61 Ops.CONST dtypes.weakint [] 104
62 Ops.ADD dtypes.i32 [60, 61]
63 Ops.CONST dtypes.weakint [] 0
64 Ops.MAX dtypes.i32 [62, 63]
65 Ops.CONST dtypes.weakint [] 32
66 Ops.DIV dtypes.i32 [64, 65]
67 Ops.AND dtypes.i32 [66, 58]
68 Ops.CMPEQ dtypes.bool [67, 63]
69 Ops.CMPEQ dtypes.bool [68, 4]
70 Ops.CONST dtypes.weakint [] 2
71 Ops.AND dtypes.i32 [66, 70]
72 Ops.CMPEQ dtypes.bool [71, 63]
73 Ops.CMPEQ dtypes.bool [72, 4]
74 Ops.CONST dtypes.weakint [] 4
75 Ops.AND dtypes.i32 [66, 74]
76 Ops.CMPEQ dtypes.bool [75, 63]
77 Ops.CMPEQ dtypes.bool [76, 4]
78 Ops.CONST dtypes.weakint [] 2102212464
79 Ops.CAST dtypes.u64 [78] dtypes.u64
80 Ops.CAST dtypes.u64 [63] dtypes.u64
81 Ops.WHERE dtypes.u64 [77, 79, 80]
82 Ops.CONST dtypes.weakint [] 2475754826
83 Ops.CAST dtypes.u64 [82] dtypes.u64
84 Ops.WHERE dtypes.u64 [77, 83, 80]
85 Ops.WHERE dtypes.u64 [73, 81, 84]
86 Ops.CONST dtypes.weakint [] 2131351028
87 Ops.CAST dtypes.u64 [86] dtypes.u64
88 Ops.WHERE dtypes.u64 [77, 87, 80]
89 Ops.CONST dtypes.weakint [] 683565275
90 Ops.CAST dtypes.u64 [89] dtypes.u64
91 Ops.WHERE dtypes.u64 [77, 90, 80]
92 Ops.WHERE dtypes.u64 [73, 88, 91]
93 Ops.WHERE dtypes.u64 [69, 85, 92]
94 Ops.CONST dtypes.weakint [] 31
95 Ops.AND dtypes.i32 [64, 94]
96 Ops.CAST dtypes.u64 [95] dtypes.u64
97 Ops.SHL dtypes.u64 [93, 96]
98 Ops.CONST dtypes.weakint [] 920167782
99 Ops.CAST dtypes.u64 [98] dtypes.u64
100 Ops.WHERE dtypes.u64 [77, 99, 90]
101 Ops.WHERE dtypes.u64 [73, 100, 88]
102 Ops.WHERE dtypes.u64 [69, 101, 85]
103 Ops.CONST dtypes.weakint [] 18446744073709551615
104 Ops.MUL dtypes.u64 [96, 103]
105 Ops.ADD dtypes.u64 [65, 104]
106 Ops.SHR dtypes.u64 [102, 105]
107 Ops.OR dtypes.u64 [97, 106]
108 Ops.AND dtypes.u64 [107, 48]
109 Ops.MUL dtypes.u64 [49, 108]
110 Ops.AND dtypes.u64 [109, 48]
111 Ops.SHL dtypes.u64 [102, 96]
112 Ops.CONST dtypes.weakint [] 1326507024
113 Ops.CAST dtypes.u64 [112] dtypes.u64
114 Ops.WHERE dtypes.u64 [77, 113, 83]
115 Ops.WHERE dtypes.u64 [73, 114, 81]
116 Ops.WHERE dtypes.u64 [69, 115, 101]
117 Ops.SHR dtypes.u64 [116, 105]
118 Ops.OR dtypes.u64 [111, 117]
119 Ops.AND dtypes.u64 [118, 48]
120 Ops.MUL dtypes.u64 [49, 119]
121 Ops.CONST dtypes.weakint [] 4294967296
122 Ops.DIV dtypes.u64 [120, 121]
123 Ops.ADD dtypes.u64 [110, 122]
124 Ops.AND dtypes.u64 [120, 48]
125 Ops.SHL dtypes.u64 [116, 96]
126 Ops.CONST dtypes.weakint [] 2140428522
127 Ops.CAST dtypes.u64 [126] dtypes.u64
128 Ops.WHERE dtypes.u64 [77, 127, 87]
129 Ops.WHERE dtypes.u64 [73, 128, 100]
130 Ops.WHERE dtypes.u64 [69, 129, 115]
131 Ops.SHR dtypes.u64 [130, 105]
132 Ops.OR dtypes.u64 [125, 131]
133 Ops.AND dtypes.u64 [132, 48]
134 Ops.MUL dtypes.u64 [49, 133]
135 Ops.DIV dtypes.u64 [134, 121]
136 Ops.ADD dtypes.u64 [124, 135]
137 Ops.AND dtypes.u64 [134, 48]
138 Ops.SHL dtypes.u64 [130, 96]
139 Ops.CONST dtypes.weakint [] 4155437400
140 Ops.CAST dtypes.u64 [139] dtypes.u64
141 Ops.WHERE dtypes.u64 [77, 140, 79]
142 Ops.WHERE dtypes.u64 [73, 141, 114]
143 Ops.WHERE dtypes.u64 [69, 142, 129]
144 Ops.SHR dtypes.u64 [143, 105]
145 Ops.OR dtypes.u64 [138, 144]
146 Ops.AND dtypes.u64 [145, 48]
147 Ops.MUL dtypes.u64 [49, 146]
148 Ops.DIV dtypes.u64 [147, 121]
149 Ops.ADD dtypes.u64 [137, 148]
150 Ops.AND dtypes.u64 [147, 48]
151 Ops.SHL dtypes.u64 [143, 96]
152 Ops.CONST dtypes.weakint [] 1841896334
153 Ops.CAST dtypes.u64 [152] dtypes.u64
154 Ops.WHERE dtypes.u64 [77, 153, 99]
155 Ops.WHERE dtypes.u64 [73, 154, 128]
156 Ops.WHERE dtypes.u64 [69, 155, 142]
157 Ops.SHR dtypes.u64 [156, 105]
158 Ops.OR dtypes.u64 [151, 157]
159 Ops.AND dtypes.u64 [158, 48]
160 Ops.MUL dtypes.u64 [49, 159]
161 Ops.DIV dtypes.u64 [160, 121]
162 Ops.ADD dtypes.u64 [150, 161]
163 Ops.AND dtypes.u64 [160, 48]
164 Ops.SHL dtypes.u64 [156, 96]
165 Ops.CONST dtypes.weakint [] 2425582776
166 Ops.CAST dtypes.u64 [165] dtypes.u64
167 Ops.WHERE dtypes.u64 [77, 166, 113]
168 Ops.WHERE dtypes.u64 [73, 167, 141]
169 Ops.WHERE dtypes.u64 [69, 168, 155]
170 Ops.SHR dtypes.u64 [169, 105]
171 Ops.OR dtypes.u64 [164, 170]
172 Ops.AND dtypes.u64 [171, 48]
173 Ops.MUL dtypes.u64 [49, 172]
174 Ops.DIV dtypes.u64 [173, 121]
175 Ops.ADD dtypes.u64 [163, 174]
176 Ops.AND dtypes.u64 [173, 48]
177 Ops.SHL dtypes.u64 [169, 96]
178 Ops.CONST dtypes.weakint [] 26364858
179 Ops.CAST dtypes.u64 [178] dtypes.u64
180 Ops.WHERE dtypes.u64 [77, 179, 127]
181 Ops.WHERE dtypes.u64 [73, 180, 154]
182 Ops.WHERE dtypes.u64 [69, 181, 168]
183 Ops.SHR dtypes.u64 [182, 105]
184 Ops.OR dtypes.u64 [177, 183]
185 Ops.AND dtypes.u64 [184, 48]
186 Ops.MUL dtypes.u64 [49, 185]
187 Ops.DIV dtypes.u64 [186, 121]
188 Ops.ADD dtypes.u64 [176, 187]
189 Ops.AND dtypes.u64 [186, 48]
190 Ops.DIV dtypes.u64 [189, 121]
191 Ops.ADD dtypes.u64 [188, 190]
192 Ops.DIV dtypes.u64 [191, 121]
193 Ops.ADD dtypes.u64 [175, 192]
194 Ops.DIV dtypes.u64 [193, 121]
195 Ops.ADD dtypes.u64 [162, 194]
196 Ops.DIV dtypes.u64 [195, 121]
197 Ops.ADD dtypes.u64 [149, 196]
198 Ops.DIV dtypes.u64 [197, 121]
199 Ops.ADD dtypes.u64 [136, 198]
200 Ops.DIV dtypes.u64 [199, 121]
201 Ops.ADD dtypes.u64 [123, 200]
202 Ops.AND dtypes.u64 [201, 48]
203 Ops.MUL dtypes.u64 [202, 121]
204 Ops.AND dtypes.u64 [199, 48]
205 Ops.OR dtypes.u64 [203, 204]
206 Ops.CONST dtypes.weakint [] 4611686018427387904
207 Ops.DIV dtypes.u64 [205, 206]
208 Ops.CONST dtypes.weakint [] 2305843009213693952
209 Ops.DIV dtypes.u64 [205, 208]
210 Ops.AND dtypes.u64 [209, 58]
211 Ops.ADD dtypes.u64 [207, 210]
212 Ops.CAST dtypes.i32 [211] dtypes.i32
213 Ops.WHERE dtypes.i32 [28, 38, 212]
214 Ops.AND dtypes.i32 [213, 70]
215 Ops.CMPEQ dtypes.bool [214, 63]
216 Ops.CMPEQ dtypes.bool [215, 4]
217 Ops.AND dtypes.i32 [213, 58]
218 Ops.CMPEQ dtypes.bool [217, 63]
219 Ops.CMPEQ dtypes.bool [218, 4]
220 Ops.CAST dtypes.f32 [38] dtypes.f32
221 Ops.CONST dtypes.weakfloat [] -0.00048351287841796875
222 Ops.MUL dtypes.f32 [220, 221]
223 Ops.CONST dtypes.weakfloat [] -1.5703125
224 Ops.MUL dtypes.f32 [220, 223]
225 Ops.ADD dtypes.f32 [224, 26]
226 Ops.ADD dtypes.f32 [222, 225]
227 Ops.CONST dtypes.weakfloat [] -3.1385570764541626e-07
228 Ops.MUL dtypes.f32 [220, 227]
229 Ops.ADD dtypes.f32 [226, 228]
230 Ops.MUL dtypes.f32 [226, 21]
231 Ops.ADD dtypes.f32 [229, 230]
232 Ops.MUL dtypes.f32 [231, 21]
233 Ops.ADD dtypes.f32 [228, 232]
234 Ops.CONST dtypes.weakfloat [] -6.07710062827671e-11
235 Ops.MUL dtypes.f32 [220, 234]
236 Ops.ADD dtypes.f32 [233, 235]
237 Ops.ADD dtypes.f32 [229, 236]
238 Ops.CONST dtypes.weakint [] 4611686018427387903
239 Ops.AND dtypes.u64 [205, 238]
240 Ops.CAST dtypes.i64 [239] dtypes.i64
241 Ops.CAST dtypes.i64 [210] dtypes.i64
242 Ops.MUL dtypes.i64 [241, 206]
243 Ops.MUL dtypes.i64 [242, 55]
244 Ops.ADD dtypes.i64 [240, 243]
245 Ops.CONST dtypes.weakint [] 4398046511104
246 Ops.DIV dtypes.i64 [244, 245]
247 Ops.CAST dtypes.f32 [246] dtypes.f32
248 Ops.CONST dtypes.weakfloat [] 4398046511104.0
249 Ops.MUL dtypes.f32 [247, 248]
250 Ops.CONST dtypes.weakint [] 2097152
251 Ops.DIV dtypes.i64 [244, 250]
252 Ops.CONST dtypes.weakint [] 2097151
253 Ops.AND dtypes.i64 [251, 252]
254 Ops.CAST dtypes.f32 [253] dtypes.f32
255 Ops.CONST dtypes.weakfloat [] 2097152.0
256 Ops.MUL dtypes.f32 [254, 255]
257 Ops.ADD dtypes.f32 [249, 256]
258 Ops.AND dtypes.i64 [244, 252]
259 Ops.CAST dtypes.f32 [258] dtypes.f32
260 Ops.MUL dtypes.f32 [259, 23]
261 Ops.ADD dtypes.f32 [257, 260]
262 Ops.AND dtypes.u64 [197, 48]
263 Ops.MUL dtypes.u64 [262, 121]
264 Ops.AND dtypes.u64 [195, 48]
265 Ops.OR dtypes.u64 [263, 264]
266 Ops.CONST dtypes.weakint [] 8796093022208
267 Ops.DIV dtypes.u64 [265, 266]
268 Ops.CAST dtypes.f32 [267] dtypes.f32
269 Ops.CONST dtypes.weakfloat [] 1.1368683772161603e-13
270 Ops.MUL dtypes.f32 [268, 269]
271 Ops.ADD dtypes.f32 [261, 270]
272 Ops.CONST dtypes.weakfloat [] 3.4061216748705226e-19
273 Ops.MUL dtypes.f32 [271, 272]
274 Ops.CONST dtypes.weakfloat [] 4097.0
275 Ops.MUL dtypes.f32 [271, 274]
276 Ops.MUL dtypes.f32 [271, 21]
277 Ops.ADD dtypes.f32 [275, 276]
278 Ops.MUL dtypes.f32 [277, 21]
279 Ops.ADD dtypes.f32 [275, 278]
280 Ops.CONST dtypes.weakfloat [] 3.406131239146355e-19
281 Ops.MUL dtypes.f32 [279, 280]
282 Ops.MUL dtypes.f32 [273, 21]
283 Ops.ADD dtypes.f32 [281, 282]
284 Ops.CONST dtypes.weakfloat [] -9.564275832644382e-25
285 Ops.MUL dtypes.f32 [279, 284]
286 Ops.ADD dtypes.f32 [283, 285]
287 Ops.MUL dtypes.f32 [279, 21]
288 Ops.ADD dtypes.f32 [271, 287]
289 Ops.MUL dtypes.f32 [288, 280]
290 Ops.ADD dtypes.f32 [286, 289]
291 Ops.MUL dtypes.f32 [288, 284]
292 Ops.ADD dtypes.f32 [290, 291]
293 Ops.CONST dtypes.weakfloat [] -9.47839680047608e-27
294 Ops.MUL dtypes.f32 [271, 293]
295 Ops.ADD dtypes.f32 [292, 294]
296 Ops.MUL dtypes.f32 [249, 21]
297 Ops.ADD dtypes.f32 [257, 296]
298 Ops.MUL dtypes.f32 [297, 21]
299 Ops.ADD dtypes.f32 [256, 298]
300 Ops.ADD dtypes.f32 [14, 299]
301 Ops.MUL dtypes.f32 [257, 21]
302 Ops.ADD dtypes.f32 [261, 301]
303 Ops.MUL dtypes.f32 [302, 21]
304 Ops.ADD dtypes.f32 [260, 303]
305 Ops.ADD dtypes.f32 [300, 304]
306 Ops.MUL dtypes.f32 [261, 21]
307 Ops.ADD dtypes.f32 [271, 306]
308 Ops.MUL dtypes.f32 [307, 21]
309 Ops.ADD dtypes.f32 [270, 308]
310 Ops.ADD dtypes.f32 [305, 309]
311 Ops.MUL dtypes.f32 [310, 272]
312 Ops.ADD dtypes.f32 [295, 311]
313 Ops.ADD dtypes.f32 [273, 312]
314 Ops.WHERE dtypes.f32 [28, 237, 313]
315 Ops.MUL dtypes.f32 [314, 314]
316 Ops.MUL dtypes.f32 [34, 315]
317 Ops.MUL dtypes.f32 [316, 21]
318 Ops.ADD dtypes.f32 [23, 317]
319 Ops.MUL dtypes.f32 [318, 21]
320 Ops.ADD dtypes.f32 [23, 319]
321 Ops.ADD dtypes.f32 [320, 317]
322 Ops.MUL dtypes.f32 [315, 315]
323 Ops.CONST dtypes.weakfloat [] -2.7557314351390663e-07
324 Ops.MUL dtypes.f32 [323, 315]
325 Ops.CONST dtypes.weakfloat [] 2.480158728947673e-05
326 Ops.ADD dtypes.f32 [324, 325]
327 Ops.MUL dtypes.f32 [326, 315]
328 Ops.CONST dtypes.weakfloat [] -0.001388888888887411
329 Ops.ADD dtypes.f32 [327, 328]
330 Ops.MUL dtypes.f32 [329, 315]
331 Ops.CONST dtypes.weakfloat [] 0.0416666666666666
332 Ops.ADD dtypes.f32 [330, 331]
333 Ops.MUL dtypes.f32 [322, 332]
334 Ops.MUL dtypes.f32 [229, 21]
335 Ops.ADD dtypes.f32 [237, 334]
336 Ops.MUL dtypes.f32 [335, 21]
337 Ops.ADD dtypes.f32 [236, 336]
338 Ops.ADD dtypes.f32 [313, 282]
339 Ops.MUL dtypes.f32 [338, 21]
340 Ops.ADD dtypes.f32 [312, 339]
341 Ops.WHERE dtypes.f32 [28, 337, 340]
342 Ops.MUL dtypes.f32 [314, 341]
343 Ops.MUL dtypes.f32 [342, 21]
344 Ops.ADD dtypes.f32 [333, 343]
345 Ops.ADD dtypes.f32 [321, 344]
346 Ops.ADD dtypes.f32 [318, 345]
347 Ops.MUL dtypes.f32 [34, 341]
348 Ops.MUL dtypes.f32 [315, 314]
349 Ops.CONST dtypes.weakfloat [] 2.7557313707070068e-06
350 Ops.MUL dtypes.f32 [349, 315]
351 Ops.CONST dtypes.weakfloat [] -0.0001984126982985795
352 Ops.ADD dtypes.f32 [350, 351]
353 Ops.MUL dtypes.f32 [352, 315]
354 Ops.CONST dtypes.weakfloat [] 0.00833333333332249
355 Ops.ADD dtypes.f32 [353, 354]
356 Ops.MUL dtypes.f32 [348, 355]
357 Ops.MUL dtypes.f32 [356, 21]
358 Ops.ADD dtypes.f32 [347, 357]
359 Ops.MUL dtypes.f32 [315, 358]
360 Ops.MUL dtypes.f32 [341, 21]
361 Ops.ADD dtypes.f32 [359, 360]
362 Ops.CONST dtypes.weakfloat [] -0.16666666666666632
363 Ops.MUL dtypes.f32 [348, 362]
364 Ops.MUL dtypes.f32 [363, 21]
365 Ops.ADD dtypes.f32 [361, 364]
366 Ops.MUL dtypes.f32 [365, 21]
367 Ops.ADD dtypes.f32 [314, 366]
368 Ops.WHERE dtypes.f32 [219, 346, 367]
369 Ops.MUL dtypes.f32 [368, 25]
370 Ops.MUL dtypes.f32 [369, 21]
371 Ops.WHERE dtypes.f32 [216, 370, 369]
372 Ops.WHERE dtypes.f32 [19, 371, 17]
373 Ops.WHERE dtypes.f32 [12, 372, 9]
374 Ops.WHERE dtypes.f32 [7, 9, 373]
375 Ops.WHERE dtypes.f32 [5, 374, 9]
376 Ops.SINK dtypes.void [375]

