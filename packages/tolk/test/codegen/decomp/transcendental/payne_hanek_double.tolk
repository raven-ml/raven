# tolk D31 D126 D413 golden=b9fec216c21fe6dadfeec016a5033ef8
0 Ops.STACK dtypes.void []
1 Ops.PARAM dtypes.f64 [0] ParamArg(slot=0, dtype=dtypes.f64)
2 Ops.BITCAST dtypes.u64 [1] dtypes.u64
3 Ops.CONST dtypes.weakint [] 4503599627370495
4 Ops.AND dtypes.u64 [2, 3]
5 Ops.CONST dtypes.weakint [] 4602678819172646912
6 Ops.OR dtypes.u64 [4, 5]
7 Ops.BITCAST dtypes.f64 [6] dtypes.f64
8 Ops.CONST dtypes.weakfloat [] 9007199254740992.0
9 Ops.MUL dtypes.f64 [7, 8]
10 Ops.CAST dtypes.u64 [9] dtypes.u64
11 Ops.CONST dtypes.weakint [] 4294967295
12 Ops.AND dtypes.u64 [10, 11]
13 Ops.CONST dtypes.weakint [] 4503599627370496
14 Ops.DIV dtypes.u64 [2, 13]
15 Ops.CONST dtypes.weakint [] 2047
16 Ops.AND dtypes.u64 [14, 15]
17 Ops.CONST dtypes.weakint [] 1023
18 Ops.CONST dtypes.weakint [] -1
19 Ops.MUL dtypes.weakint [17, 18]
20 Ops.ADD dtypes.u64 [16, 19]
21 Ops.CONST dtypes.weakint [] 1
22 Ops.ADD dtypes.u64 [20, 21]
23 Ops.CAST dtypes.i32 [22] dtypes.i32
24 Ops.CONST dtypes.weakint [] 75
25 Ops.ADD dtypes.i32 [23, 24]
26 Ops.CONST dtypes.weakint [] 0
27 Ops.MAX dtypes.i32 [25, 26]
28 Ops.CONST dtypes.weakint [] 32
29 Ops.DIV dtypes.i32 [27, 28]
30 Ops.AND dtypes.i32 [29, 21]
31 Ops.CMPEQ dtypes.bool [30, 26]
32 Ops.CONST dtypes.bool [] False
33 Ops.CMPEQ dtypes.bool [31, 32]
34 Ops.CONST dtypes.weakint [] 2
35 Ops.AND dtypes.i32 [29, 34]
36 Ops.CMPEQ dtypes.bool [35, 26]
37 Ops.CMPEQ dtypes.bool [36, 32]
38 Ops.CONST dtypes.weakint [] 4
39 Ops.AND dtypes.i32 [29, 38]
40 Ops.CMPEQ dtypes.bool [39, 26]
41 Ops.CMPEQ dtypes.bool [40, 32]
42 Ops.CONST dtypes.weakint [] 8
43 Ops.AND dtypes.i32 [29, 42]
44 Ops.CMPEQ dtypes.bool [43, 26]
45 Ops.CMPEQ dtypes.bool [44, 32]
46 Ops.CONST dtypes.weakint [] 16
47 Ops.AND dtypes.i32 [29, 46]
48 Ops.CMPEQ dtypes.bool [47, 26]
49 Ops.CMPEQ dtypes.bool [48, 32]
50 Ops.AND dtypes.i32 [29, 28]
51 Ops.CMPEQ dtypes.bool [50, 26]
52 Ops.CMPEQ dtypes.bool [51, 32]
53 Ops.CAST dtypes.u64 [26] dtypes.u64
54 Ops.CONST dtypes.weakint [] 3388129901
55 Ops.CAST dtypes.u64 [54] dtypes.u64
56 Ops.WHERE dtypes.u64 [52, 53, 55]
57 Ops.CONST dtypes.weakint [] 2188665991
58 Ops.CAST dtypes.u64 [57] dtypes.u64
59 Ops.WHERE dtypes.u64 [52, 53, 58]
60 Ops.WHERE dtypes.u64 [49, 56, 59]
61 Ops.CONST dtypes.weakint [] 1624559229
62 Ops.CAST dtypes.u64 [61] dtypes.u64
63 Ops.WHERE dtypes.u64 [52, 53, 62]
64 Ops.CONST dtypes.weakint [] 1206081346
65 Ops.CAST dtypes.u64 [64] dtypes.u64
66 Ops.CONST dtypes.weakint [] 2102212464
67 Ops.CAST dtypes.u64 [66] dtypes.u64
68 Ops.WHERE dtypes.u64 [52, 65, 67]
69 Ops.WHERE dtypes.u64 [49, 63, 68]
70 Ops.WHERE dtypes.u64 [45, 60, 69]
71 Ops.CONST dtypes.weakint [] 4152452611
72 Ops.CAST dtypes.u64 [71] dtypes.u64
73 Ops.WHERE dtypes.u64 [52, 53, 72]
74 Ops.CONST dtypes.weakint [] 4155437400
75 Ops.CAST dtypes.u64 [74] dtypes.u64
76 Ops.WHERE dtypes.u64 [52, 53, 75]
77 Ops.WHERE dtypes.u64 [49, 73, 76]
78 Ops.CONST dtypes.weakint [] 1269748001
79 Ops.CAST dtypes.u64 [78] dtypes.u64
80 Ops.WHERE dtypes.u64 [52, 53, 79]
81 Ops.CONST dtypes.weakint [] 4210654302
82 Ops.CAST dtypes.u64 [81] dtypes.u64
83 Ops.WHERE dtypes.u64 [52, 82, 53]
84 Ops.WHERE dtypes.u64 [49, 80, 83]
85 Ops.WHERE dtypes.u64 [45, 77, 84]
86 Ops.WHERE dtypes.u64 [41, 70, 85]
87 Ops.CONST dtypes.weakint [] 3598883655
88 Ops.CAST dtypes.u64 [87] dtypes.u64
89 Ops.WHERE dtypes.u64 [52, 53, 88]
90 Ops.CONST dtypes.weakint [] 2425582776
91 Ops.CAST dtypes.u64 [90] dtypes.u64
92 Ops.WHERE dtypes.u64 [52, 53, 91]
93 Ops.WHERE dtypes.u64 [49, 89, 92]
94 Ops.CONST dtypes.weakint [] 2904022928
95 Ops.CAST dtypes.u64 [94] dtypes.u64
96 Ops.WHERE dtypes.u64 [52, 53, 95]
97 Ops.CONST dtypes.weakint [] 3801392314
98 Ops.CAST dtypes.u64 [97] dtypes.u64
99 Ops.CONST dtypes.weakint [] 2475754826
100 Ops.CAST dtypes.u64 [99] dtypes.u64
101 Ops.WHERE dtypes.u64 [52, 98, 100]
102 Ops.WHERE dtypes.u64 [49, 96, 101]
103 Ops.WHERE dtypes.u64 [45, 93, 102]
104 Ops.CONST dtypes.weakint [] 4018031118
105 Ops.CAST dtypes.u64 [104] dtypes.u64
106 Ops.WHERE dtypes.u64 [52, 53, 105]
107 Ops.CONST dtypes.weakint [] 3206470378
108 Ops.CAST dtypes.u64 [107] dtypes.u64
109 Ops.CONST dtypes.weakint [] 1326507024
110 Ops.CAST dtypes.u64 [109] dtypes.u64
111 Ops.WHERE dtypes.u64 [52, 108, 110]
112 Ops.WHERE dtypes.u64 [49, 106, 111]
113 Ops.CONST dtypes.weakint [] 743074255
114 Ops.CAST dtypes.u64 [113] dtypes.u64
115 Ops.WHERE dtypes.u64 [52, 53, 114]
116 Ops.CONST dtypes.weakint [] 2811755147
117 Ops.CAST dtypes.u64 [116] dtypes.u64
118 Ops.WHERE dtypes.u64 [52, 117, 53]
119 Ops.WHERE dtypes.u64 [49, 115, 118]
120 Ops.WHERE dtypes.u64 [45, 112, 119]
121 Ops.WHERE dtypes.u64 [41, 103, 120]
122 Ops.WHERE dtypes.u64 [37, 86, 121]
123 Ops.CONST dtypes.weakint [] 3679297459
124 Ops.CAST dtypes.u64 [123] dtypes.u64
125 Ops.WHERE dtypes.u64 [52, 53, 124]
126 Ops.CONST dtypes.weakint [] 26364858
127 Ops.CAST dtypes.u64 [126] dtypes.u64
128 Ops.WHERE dtypes.u64 [52, 53, 127]
129 Ops.WHERE dtypes.u64 [49, 125, 128]
130 Ops.CONST dtypes.weakint [] 1315206542
131 Ops.CAST dtypes.u64 [130] dtypes.u64
132 Ops.WHERE dtypes.u64 [52, 53, 131]
133 Ops.CONST dtypes.weakint [] 2600392684
134 Ops.CAST dtypes.u64 [133] dtypes.u64
135 Ops.CONST dtypes.weakint [] 2131351028
136 Ops.CAST dtypes.u64 [135] dtypes.u64
137 Ops.WHERE dtypes.u64 [52, 134, 136]
138 Ops.WHERE dtypes.u64 [49, 132, 137]
139 Ops.WHERE dtypes.u64 [45, 129, 138]
140 Ops.CONST dtypes.weakint [] 3355321855
141 Ops.CAST dtypes.u64 [140] dtypes.u64
142 Ops.WHERE dtypes.u64 [52, 53, 141]
143 Ops.CONST dtypes.weakint [] 4231261960
144 Ops.CAST dtypes.u64 [143] dtypes.u64
145 Ops.CONST dtypes.weakint [] 2140428522
146 Ops.CAST dtypes.u64 [145] dtypes.u64
147 Ops.WHERE dtypes.u64 [52, 144, 146]
148 Ops.WHERE dtypes.u64 [49, 142, 147]
149 Ops.CONST dtypes.weakint [] 3122695549
150 Ops.CAST dtypes.u64 [149] dtypes.u64
151 Ops.WHERE dtypes.u64 [52, 53, 150]
152 Ops.CONST dtypes.weakint [] 1565126321
153 Ops.CAST dtypes.u64 [152] dtypes.u64
154 Ops.WHERE dtypes.u64 [52, 153, 53]
155 Ops.WHERE dtypes.u64 [49, 151, 154]
156 Ops.WHERE dtypes.u64 [45, 148, 155]
157 Ops.WHERE dtypes.u64 [41, 139, 156]
158 Ops.CONST dtypes.weakint [] 4224435298
159 Ops.CAST dtypes.u64 [158] dtypes.u64
160 Ops.WHERE dtypes.u64 [52, 53, 159]
161 Ops.CONST dtypes.weakint [] 1841896334
162 Ops.CAST dtypes.u64 [161] dtypes.u64
163 Ops.WHERE dtypes.u64 [52, 53, 162]
164 Ops.WHERE dtypes.u64 [49, 160, 163]
165 Ops.CONST dtypes.weakint [] 979835913
166 Ops.CAST dtypes.u64 [165] dtypes.u64
167 Ops.WHERE dtypes.u64 [52, 53, 166]
168 Ops.CONST dtypes.weakint [] 3477196413
169 Ops.CAST dtypes.u64 [168] dtypes.u64
170 Ops.CONST dtypes.weakint [] 683565275
171 Ops.CAST dtypes.u64 [170] dtypes.u64
172 Ops.WHERE dtypes.u64 [52, 169, 171]
173 Ops.WHERE dtypes.u64 [49, 167, 172]
174 Ops.WHERE dtypes.u64 [45, 164, 173]
175 Ops.CONST dtypes.weakint [] 656480226
176 Ops.CAST dtypes.u64 [175] dtypes.u64
177 Ops.WHERE dtypes.u64 [52, 53, 176]
178 Ops.CONST dtypes.weakint [] 360762385
179 Ops.CAST dtypes.u64 [178] dtypes.u64
180 Ops.CONST dtypes.weakint [] 920167782
181 Ops.CAST dtypes.u64 [180] dtypes.u64
182 Ops.WHERE dtypes.u64 [52, 179, 181]
183 Ops.WHERE dtypes.u64 [49, 177, 182]
184 Ops.CONST dtypes.weakint [] 1065843399
185 Ops.CAST dtypes.u64 [184] dtypes.u64
186 Ops.WHERE dtypes.u64 [52, 53, 185]
187 Ops.CONST dtypes.weakint [] 3553726425
188 Ops.CAST dtypes.u64 [187] dtypes.u64
189 Ops.WHERE dtypes.u64 [52, 188, 53]
190 Ops.WHERE dtypes.u64 [49, 186, 189]
191 Ops.WHERE dtypes.u64 [45, 183, 190]
192 Ops.WHERE dtypes.u64 [41, 174, 191]
193 Ops.WHERE dtypes.u64 [37, 157, 192]
194 Ops.WHERE dtypes.u64 [33, 122, 193]
195 Ops.CONST dtypes.weakint [] 31
196 Ops.AND dtypes.i32 [27, 195]
197 Ops.CAST dtypes.u64 [196] dtypes.u64
198 Ops.SHL dtypes.u64 [194, 197]
199 Ops.WHERE dtypes.u64 [52, 53, 188]
200 Ops.WHERE dtypes.u64 [49, 199, 186]
201 Ops.WHERE dtypes.u64 [45, 200, 183]
202 Ops.WHERE dtypes.u64 [41, 201, 174]
203 Ops.WHERE dtypes.u64 [37, 202, 157]
204 Ops.WHERE dtypes.u64 [33, 203, 122]
205 Ops.CONST dtypes.weakint [] 18446744073709551615
206 Ops.MUL dtypes.u64 [197, 205]
207 Ops.ADD dtypes.u64 [28, 206]
208 Ops.SHR dtypes.u64 [204, 207]
209 Ops.OR dtypes.u64 [198, 208]
210 Ops.AND dtypes.u64 [209, 11]
211 Ops.MUL dtypes.u64 [12, 210]
212 Ops.AND dtypes.u64 [211, 11]
213 Ops.SHL dtypes.u64 [204, 197]
214 Ops.WHERE dtypes.u64 [52, 53, 117]
215 Ops.WHERE dtypes.u64 [49, 214, 115]
216 Ops.WHERE dtypes.u64 [45, 215, 112]
217 Ops.WHERE dtypes.u64 [41, 216, 103]
218 Ops.WHERE dtypes.u64 [37, 217, 86]
219 Ops.WHERE dtypes.u64 [33, 218, 203]
220 Ops.SHR dtypes.u64 [219, 207]
221 Ops.OR dtypes.u64 [213, 220]
222 Ops.AND dtypes.u64 [221, 11]
223 Ops.MUL dtypes.u64 [12, 222]
224 Ops.CONST dtypes.weakint [] 4294967296
225 Ops.DIV dtypes.u64 [223, 224]
226 Ops.ADD dtypes.u64 [212, 225]
227 Ops.DIV dtypes.u64 [10, 224]
228 Ops.MUL dtypes.u64 [227, 222]
229 Ops.AND dtypes.u64 [228, 11]
230 Ops.ADD dtypes.u64 [226, 229]
231 Ops.SHL dtypes.u64 [219, 197]
232 Ops.WHERE dtypes.u64 [52, 53, 153]
233 Ops.WHERE dtypes.u64 [49, 232, 151]
234 Ops.WHERE dtypes.u64 [45, 233, 148]
235 Ops.WHERE dtypes.u64 [41, 234, 139]
236 Ops.WHERE dtypes.u64 [37, 235, 202]
237 Ops.WHERE dtypes.u64 [33, 236, 218]
238 Ops.SHR dtypes.u64 [237, 207]
239 Ops.OR dtypes.u64 [231, 238]
240 Ops.AND dtypes.u64 [239, 11]
241 Ops.MUL dtypes.u64 [227, 240]
242 Ops.DIV dtypes.u64 [241, 224]
243 Ops.ADD dtypes.u64 [230, 242]
244 Ops.AND dtypes.u64 [223, 11]
245 Ops.MUL dtypes.u64 [12, 240]
246 Ops.DIV dtypes.u64 [245, 224]
247 Ops.ADD dtypes.u64 [244, 246]
248 Ops.AND dtypes.u64 [241, 11]
249 Ops.ADD dtypes.u64 [247, 248]
250 Ops.SHL dtypes.u64 [237, 197]
251 Ops.WHERE dtypes.u64 [52, 53, 82]
252 Ops.WHERE dtypes.u64 [49, 251, 80]
253 Ops.WHERE dtypes.u64 [45, 252, 77]
254 Ops.WHERE dtypes.u64 [41, 253, 70]
255 Ops.WHERE dtypes.u64 [37, 254, 217]
256 Ops.WHERE dtypes.u64 [33, 255, 236]
257 Ops.SHR dtypes.u64 [256, 207]
258 Ops.OR dtypes.u64 [250, 257]
259 Ops.AND dtypes.u64 [258, 11]
260 Ops.MUL dtypes.u64 [227, 259]
261 Ops.DIV dtypes.u64 [260, 224]
262 Ops.ADD dtypes.u64 [249, 261]
263 Ops.AND dtypes.u64 [245, 11]
264 Ops.MUL dtypes.u64 [12, 259]
265 Ops.DIV dtypes.u64 [264, 224]
266 Ops.ADD dtypes.u64 [263, 265]
267 Ops.AND dtypes.u64 [260, 11]
268 Ops.ADD dtypes.u64 [266, 267]
269 Ops.SHL dtypes.u64 [256, 197]
270 Ops.WHERE dtypes.u64 [52, 53, 169]
271 Ops.WHERE dtypes.u64 [49, 270, 167]
272 Ops.WHERE dtypes.u64 [45, 271, 164]
273 Ops.WHERE dtypes.u64 [41, 272, 201]
274 Ops.WHERE dtypes.u64 [37, 273, 235]
275 Ops.WHERE dtypes.u64 [33, 274, 255]
276 Ops.SHR dtypes.u64 [275, 207]
277 Ops.OR dtypes.u64 [269, 276]
278 Ops.AND dtypes.u64 [277, 11]
279 Ops.MUL dtypes.u64 [227, 278]
280 Ops.DIV dtypes.u64 [279, 224]
281 Ops.ADD dtypes.u64 [268, 280]
282 Ops.AND dtypes.u64 [264, 11]
283 Ops.MUL dtypes.u64 [12, 278]
284 Ops.DIV dtypes.u64 [283, 224]
285 Ops.ADD dtypes.u64 [282, 284]
286 Ops.AND dtypes.u64 [279, 11]
287 Ops.ADD dtypes.u64 [285, 286]
288 Ops.SHL dtypes.u64 [275, 197]
289 Ops.WHERE dtypes.u64 [52, 53, 98]
290 Ops.WHERE dtypes.u64 [49, 289, 96]
291 Ops.WHERE dtypes.u64 [45, 290, 93]
292 Ops.WHERE dtypes.u64 [41, 291, 216]
293 Ops.WHERE dtypes.u64 [37, 292, 254]
294 Ops.WHERE dtypes.u64 [33, 293, 274]
295 Ops.SHR dtypes.u64 [294, 207]
296 Ops.OR dtypes.u64 [288, 295]
297 Ops.AND dtypes.u64 [296, 11]
298 Ops.MUL dtypes.u64 [227, 297]
299 Ops.DIV dtypes.u64 [298, 224]
300 Ops.ADD dtypes.u64 [287, 299]
301 Ops.AND dtypes.u64 [283, 11]
302 Ops.MUL dtypes.u64 [12, 297]
303 Ops.DIV dtypes.u64 [302, 224]
304 Ops.ADD dtypes.u64 [301, 303]
305 Ops.AND dtypes.u64 [298, 11]
306 Ops.ADD dtypes.u64 [304, 305]
307 Ops.SHL dtypes.u64 [294, 197]
308 Ops.WHERE dtypes.u64 [52, 53, 134]
309 Ops.WHERE dtypes.u64 [49, 308, 132]
310 Ops.WHERE dtypes.u64 [45, 309, 129]
311 Ops.WHERE dtypes.u64 [41, 310, 234]
312 Ops.WHERE dtypes.u64 [37, 311, 273]
313 Ops.WHERE dtypes.u64 [33, 312, 293]
314 Ops.SHR dtypes.u64 [313, 207]
315 Ops.OR dtypes.u64 [307, 314]
316 Ops.AND dtypes.u64 [315, 11]
317 Ops.MUL dtypes.u64 [227, 316]
318 Ops.DIV dtypes.u64 [317, 224]
319 Ops.ADD dtypes.u64 [306, 318]
320 Ops.AND dtypes.u64 [302, 11]
321 Ops.MUL dtypes.u64 [12, 316]
322 Ops.DIV dtypes.u64 [321, 224]
323 Ops.ADD dtypes.u64 [320, 322]
324 Ops.AND dtypes.u64 [317, 11]
325 Ops.ADD dtypes.u64 [323, 324]
326 Ops.SHL dtypes.u64 [313, 197]
327 Ops.WHERE dtypes.u64 [52, 53, 65]
328 Ops.WHERE dtypes.u64 [49, 327, 63]
329 Ops.WHERE dtypes.u64 [45, 328, 60]
330 Ops.WHERE dtypes.u64 [41, 329, 253]
331 Ops.WHERE dtypes.u64 [37, 330, 292]
332 Ops.WHERE dtypes.u64 [33, 331, 312]
333 Ops.SHR dtypes.u64 [332, 207]
334 Ops.OR dtypes.u64 [326, 333]
335 Ops.AND dtypes.u64 [334, 11]
336 Ops.MUL dtypes.u64 [227, 335]
337 Ops.DIV dtypes.u64 [336, 224]
338 Ops.ADD dtypes.u64 [325, 337]
339 Ops.AND dtypes.u64 [321, 11]
340 Ops.MUL dtypes.u64 [12, 335]
341 Ops.DIV dtypes.u64 [340, 224]
342 Ops.ADD dtypes.u64 [339, 341]
343 Ops.AND dtypes.u64 [336, 11]
344 Ops.ADD dtypes.u64 [342, 343]
345 Ops.DIV dtypes.u64 [344, 224]
346 Ops.ADD dtypes.u64 [338, 345]
347 Ops.DIV dtypes.u64 [346, 224]
348 Ops.ADD dtypes.u64 [319, 347]
349 Ops.DIV dtypes.u64 [348, 224]
350 Ops.ADD dtypes.u64 [300, 349]
351 Ops.DIV dtypes.u64 [350, 224]
352 Ops.ADD dtypes.u64 [281, 351]
353 Ops.DIV dtypes.u64 [352, 224]
354 Ops.ADD dtypes.u64 [262, 353]
355 Ops.DIV dtypes.u64 [354, 224]
356 Ops.ADD dtypes.u64 [243, 355]
357 Ops.AND dtypes.u64 [356, 11]
358 Ops.MUL dtypes.u64 [357, 224]
359 Ops.AND dtypes.u64 [354, 11]
360 Ops.OR dtypes.u64 [358, 359]
361 Ops.CONST dtypes.weakint [] 4611686018427387903
362 Ops.AND dtypes.u64 [360, 361]
363 Ops.CAST dtypes.i64 [362] dtypes.i64
364 Ops.CONST dtypes.weakint [] 2305843009213693952
365 Ops.DIV dtypes.u64 [360, 364]
366 Ops.AND dtypes.u64 [365, 21]
367 Ops.CAST dtypes.i64 [366] dtypes.i64
368 Ops.CONST dtypes.weakint [] 4611686018427387904
369 Ops.MUL dtypes.i64 [367, 368]
370 Ops.MUL dtypes.i64 [369, 18]
371 Ops.ADD dtypes.i64 [363, 370]
372 Ops.DIV dtypes.i64 [371, 224]
373 Ops.CAST dtypes.f64 [372] dtypes.f64
374 Ops.CONST dtypes.weakfloat [] 4294967296.0
375 Ops.MUL dtypes.f64 [373, 374]
376 Ops.AND dtypes.i64 [371, 11]
377 Ops.CAST dtypes.f64 [376] dtypes.f64
378 Ops.CONST dtypes.weakfloat [] 1.0
379 Ops.MUL dtypes.f64 [377, 378]
380 Ops.ADD dtypes.f64 [375, 379]
381 Ops.AND dtypes.u64 [352, 11]
382 Ops.MUL dtypes.u64 [381, 224]
383 Ops.AND dtypes.u64 [350, 11]
384 Ops.OR dtypes.u64 [382, 383]
385 Ops.CONST dtypes.weakint [] 2048
386 Ops.DIV dtypes.u64 [384, 385]
387 Ops.CAST dtypes.f64 [386] dtypes.f64
388 Ops.CONST dtypes.weakfloat [] 1.1102230246251565e-16
389 Ops.MUL dtypes.f64 [387, 388]
390 Ops.ADD dtypes.f64 [380, 389]
391 Ops.CONST dtypes.weakfloat [] 3.4061215800865545e-19
392 Ops.MUL dtypes.f64 [390, 391]
393 Ops.CONST dtypes.weakfloat [] 134217729.0
394 Ops.MUL dtypes.f64 [390, 393]
395 Ops.CONST dtypes.weakfloat [] -1.0
396 Ops.MUL dtypes.f64 [390, 395]
397 Ops.ADD dtypes.f64 [394, 396]
398 Ops.MUL dtypes.f64 [397, 395]
399 Ops.ADD dtypes.f64 [394, 398]
400 Ops.CONST dtypes.weakfloat [] 3.406121610247037e-19
401 Ops.MUL dtypes.f64 [399, 400]
402 Ops.MUL dtypes.f64 [392, 395]
403 Ops.ADD dtypes.f64 [401, 402]
404 Ops.CONST dtypes.weakfloat [] -3.016048278183199e-27
405 Ops.MUL dtypes.f64 [399, 404]
406 Ops.ADD dtypes.f64 [403, 405]
407 Ops.MUL dtypes.f64 [399, 395]
408 Ops.ADD dtypes.f64 [390, 407]
409 Ops.MUL dtypes.f64 [408, 400]
410 Ops.ADD dtypes.f64 [406, 409]
411 Ops.MUL dtypes.f64 [408, 404]
412 Ops.ADD dtypes.f64 [410, 411]
413 Ops.CONST dtypes.weakfloat [] 1.3277647201629795e-35
414 Ops.MUL dtypes.f64 [390, 413]
415 Ops.ADD dtypes.f64 [412, 414]
416 Ops.CONST dtypes.weakfloat [] 0.0
417 Ops.CAST dtypes.f64 [416] dtypes.f64
418 Ops.MUL dtypes.f64 [375, 395]
419 Ops.ADD dtypes.f64 [380, 418]
420 Ops.MUL dtypes.f64 [419, 395]
421 Ops.ADD dtypes.f64 [379, 420]
422 Ops.ADD dtypes.f64 [417, 421]
423 Ops.MUL dtypes.f64 [380, 395]
424 Ops.ADD dtypes.f64 [390, 423]
425 Ops.MUL dtypes.f64 [424, 395]
426 Ops.ADD dtypes.f64 [389, 425]
427 Ops.ADD dtypes.f64 [422, 426]
428 Ops.MUL dtypes.f64 [427, 391]
429 Ops.ADD dtypes.f64 [415, 428]
430 Ops.ADD dtypes.f64 [392, 429]
431 Ops.DIV dtypes.u64 [360, 368]
432 Ops.ADD dtypes.u64 [431, 366]
433 Ops.CAST dtypes.i32 [432] dtypes.i32
434 Ops.SINK dtypes.void [430, 433]

