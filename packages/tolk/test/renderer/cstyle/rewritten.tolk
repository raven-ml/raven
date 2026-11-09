# tolk D9 D413 golden=96b396bd38ca09c804d3782aa2f5cac7
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
18 Ops.BITCAST dtypes.u32 [17] dtypes.u32
19 Ops.CONST dtypes.weakint [] 4294967295
20 Ops.MUL dtypes.u32 [18, 19]
21 Ops.CONST dtypes.weakint [] 2139095040
22 Ops.AND dtypes.u32 [20, 21]
23 Ops.CONST dtypes.weakint [] 0
24 Ops.CMPEQ dtypes.bool [22, 23]
25 Ops.CONST dtypes.bool [] False
26 Ops.CMPEQ dtypes.bool [24, 25]
27 Ops.SHR dtypes.u32 [18, 0]
28 Ops.CONST dtypes.weakint [] 1
29 Ops.AND dtypes.u32 [27, 28]
30 Ops.ADD dtypes.u32 [18, 29]
31 Ops.CONST dtypes.weakint [] 32767
32 Ops.ADD dtypes.u32 [30, 31]
33 Ops.CONST dtypes.weakint [] 65535
34 Ops.AND dtypes.u32 [18, 33]
35 Ops.CMPEQ dtypes.bool [34, 23]
36 Ops.CMPEQ dtypes.bool [35, 25]
37 Ops.CONST dtypes.weakint [] 65536
38 Ops.OR dtypes.u32 [18, 37]
39 Ops.WHERE dtypes.u32 [36, 38, 18]
40 Ops.WHERE dtypes.u32 [26, 32, 39]
41 Ops.SHR dtypes.u32 [40, 0]
42 Ops.CAST dtypes.u16 [41] dtypes.u16
43 Ops.BITCAST dtypes.bf16 [42] dtypes.bf16
44 Ops.STORE dtypes.void [3, 43]
45 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=2, dtype=dtypes.bf16)
46 Ops.INDEX dtypes.bf16 [45, 2]
47 Ops.LOAD dtypes.bf16 [46]
48 Ops.BITCAST dtypes.u16 [47] dtypes.u16
49 Ops.CAST dtypes.u32 [48] dtypes.u32
50 Ops.SHL dtypes.u32 [49, 0]
51 Ops.BITCAST dtypes.f32 [50] dtypes.f32
52 Ops.MUL dtypes.f32 [10, 51]
53 Ops.BITCAST dtypes.u32 [52] dtypes.u32
54 Ops.MUL dtypes.u32 [53, 19]
55 Ops.AND dtypes.u32 [54, 21]
56 Ops.CMPEQ dtypes.bool [55, 23]
57 Ops.CMPEQ dtypes.bool [56, 25]
58 Ops.SHR dtypes.u32 [53, 0]
59 Ops.AND dtypes.u32 [58, 28]
60 Ops.ADD dtypes.u32 [53, 59]
61 Ops.ADD dtypes.u32 [60, 31]
62 Ops.AND dtypes.u32 [53, 33]
63 Ops.CMPEQ dtypes.bool [62, 23]
64 Ops.CMPEQ dtypes.bool [63, 25]
65 Ops.OR dtypes.u32 [53, 37]
66 Ops.WHERE dtypes.u32 [64, 65, 53]
67 Ops.WHERE dtypes.u32 [57, 61, 66]
68 Ops.SHR dtypes.u32 [67, 0]
69 Ops.CAST dtypes.u16 [68] dtypes.u16
70 Ops.BITCAST dtypes.bf16 [69] dtypes.bf16
71 Ops.STORE dtypes.void [3, 70]
72 Ops.PARAM dtypes.bool [0] ParamArg(slot=0, dtype=dtypes.bool)
73 Ops.INDEX dtypes.bool [72, 2]
74 Ops.CMPLT dtypes.bool [10, 51]
75 Ops.STORE dtypes.void [73, 74]
76 Ops.PARAM dtypes.f32 [0] ParamArg(slot=3, dtype=dtypes.f32)
77 Ops.INDEX dtypes.f32 [76, 2]
78 Ops.LOAD dtypes.f32 [77]
79 Ops.CONST dtypes.weakfloat [] 0.0
80 Ops.CMPLT dtypes.bool [78, 79]
81 Ops.WHERE dtypes.bf16 [80, 6, 47]
82 Ops.STORE dtypes.void [3, 81]
83 Ops.SQRT dtypes.f32 [10]
84 Ops.BITCAST dtypes.u32 [83] dtypes.u32
85 Ops.MUL dtypes.u32 [84, 19]
86 Ops.AND dtypes.u32 [85, 21]
87 Ops.CMPEQ dtypes.bool [86, 23]
88 Ops.CMPEQ dtypes.bool [87, 25]
89 Ops.SHR dtypes.u32 [84, 0]
90 Ops.AND dtypes.u32 [89, 28]
91 Ops.ADD dtypes.u32 [84, 90]
92 Ops.ADD dtypes.u32 [91, 31]
93 Ops.AND dtypes.u32 [84, 33]
94 Ops.CMPEQ dtypes.bool [93, 23]
95 Ops.CMPEQ dtypes.bool [94, 25]
96 Ops.OR dtypes.u32 [84, 37]
97 Ops.WHERE dtypes.u32 [95, 96, 84]
98 Ops.WHERE dtypes.u32 [88, 92, 97]
99 Ops.SHR dtypes.u32 [98, 0]
100 Ops.CAST dtypes.u16 [99] dtypes.u16
101 Ops.BITCAST dtypes.bf16 [100] dtypes.bf16
102 Ops.STORE dtypes.void [3, 101]
103 Ops.EXP2 dtypes.f32 [10]
104 Ops.BITCAST dtypes.u32 [103] dtypes.u32
105 Ops.MUL dtypes.u32 [104, 19]
106 Ops.AND dtypes.u32 [105, 21]
107 Ops.CMPEQ dtypes.bool [106, 23]
108 Ops.CMPEQ dtypes.bool [107, 25]
109 Ops.SHR dtypes.u32 [104, 0]
110 Ops.AND dtypes.u32 [109, 28]
111 Ops.ADD dtypes.u32 [104, 110]
112 Ops.ADD dtypes.u32 [111, 31]
113 Ops.AND dtypes.u32 [104, 33]
114 Ops.CMPEQ dtypes.bool [113, 23]
115 Ops.CMPEQ dtypes.bool [114, 25]
116 Ops.OR dtypes.u32 [104, 37]
117 Ops.WHERE dtypes.u32 [115, 116, 104]
118 Ops.WHERE dtypes.u32 [108, 112, 117]
119 Ops.SHR dtypes.u32 [118, 0]
120 Ops.CAST dtypes.u16 [119] dtypes.u16
121 Ops.BITCAST dtypes.bf16 [120] dtypes.bf16
122 Ops.STORE dtypes.void [3, 121]
123 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
124 Ops.INDEX dtypes.f32 [123, 2]
125 Ops.STORE dtypes.void [124, 10]
126 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
127 Ops.INDEX dtypes.f32 [126, 2]
128 Ops.LOAD dtypes.f32 [127]
129 Ops.BITCAST dtypes.u32 [128] dtypes.u32
130 Ops.MUL dtypes.u32 [129, 19]
131 Ops.AND dtypes.u32 [130, 21]
132 Ops.CMPEQ dtypes.bool [131, 23]
133 Ops.CMPEQ dtypes.bool [132, 25]
134 Ops.SHR dtypes.u32 [129, 0]
135 Ops.AND dtypes.u32 [134, 28]
136 Ops.ADD dtypes.u32 [129, 135]
137 Ops.ADD dtypes.u32 [136, 31]
138 Ops.AND dtypes.u32 [129, 33]
139 Ops.CMPEQ dtypes.bool [138, 23]
140 Ops.CMPEQ dtypes.bool [139, 25]
141 Ops.OR dtypes.u32 [129, 37]
142 Ops.WHERE dtypes.u32 [140, 141, 129]
143 Ops.WHERE dtypes.u32 [133, 137, 142]
144 Ops.SHR dtypes.u32 [143, 0]
145 Ops.CAST dtypes.u16 [144] dtypes.u16
146 Ops.BITCAST dtypes.bf16 [145] dtypes.bf16
147 Ops.STORE dtypes.void [3, 146]
148 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
149 Ops.INDEX dtypes.f16 [148, 2]
150 Ops.LOAD dtypes.f16 [149]
151 Ops.CAST dtypes.f32 [150] dtypes.f32
152 Ops.BITCAST dtypes.u32 [151] dtypes.u32
153 Ops.MUL dtypes.u32 [152, 19]
154 Ops.AND dtypes.u32 [153, 21]
155 Ops.CMPEQ dtypes.bool [154, 23]
156 Ops.CMPEQ dtypes.bool [155, 25]
157 Ops.SHR dtypes.u32 [152, 0]
158 Ops.AND dtypes.u32 [157, 28]
159 Ops.ADD dtypes.u32 [152, 158]
160 Ops.ADD dtypes.u32 [159, 31]
161 Ops.AND dtypes.u32 [152, 33]
162 Ops.CMPEQ dtypes.bool [161, 23]
163 Ops.CMPEQ dtypes.bool [162, 25]
164 Ops.OR dtypes.u32 [152, 37]
165 Ops.WHERE dtypes.u32 [163, 164, 152]
166 Ops.WHERE dtypes.u32 [156, 160, 165]
167 Ops.SHR dtypes.u32 [166, 0]
168 Ops.CAST dtypes.u16 [167] dtypes.u16
169 Ops.BITCAST dtypes.bf16 [168] dtypes.bf16
170 Ops.STORE dtypes.void [3, 169]
171 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
172 Ops.INDEX dtypes.i32 [171, 2]
173 Ops.CAST dtypes.i32 [10] dtypes.i32
174 Ops.STORE dtypes.void [172, 173]
175 Ops.PARAM dtypes.f16 [0] ParamArg(slot=0, dtype=dtypes.f16)
176 Ops.INDEX dtypes.f16 [175, 2]
177 Ops.PARAM dtypes.f64 [0] ParamArg(slot=1, dtype=dtypes.f64)
178 Ops.INDEX dtypes.f64 [177, 2]
179 Ops.LOAD dtypes.f64 [178]
180 Ops.CAST dtypes.f32 [179] dtypes.f32
181 Ops.CAST dtypes.f64 [180] dtypes.f64
182 Ops.CMPLT dtypes.bool [179, 181]
183 Ops.CAST dtypes.f64 [79] dtypes.f64
184 Ops.CMPLT dtypes.bool [183, 179]
185 Ops.AND dtypes.bool [182, 184]
186 Ops.CMPLT dtypes.bool [181, 179]
187 Ops.CMPLT dtypes.bool [179, 183]
188 Ops.AND dtypes.bool [186, 187]
189 Ops.OR dtypes.bool [185, 188]
190 Ops.BITCAST dtypes.u32 [180] dtypes.u32
191 Ops.CONST dtypes.weakint [] -1
192 Ops.MUL dtypes.weakint [28, 191]
193 Ops.ADD dtypes.u32 [190, 192]
194 Ops.WHERE dtypes.u32 [189, 193, 190]
195 Ops.CMPEQ dtypes.bool [181, 179]
196 Ops.CMPEQ dtypes.bool [195, 25]
197 Ops.CAST dtypes.u32 [196] dtypes.u32
198 Ops.OR dtypes.u32 [194, 197]
199 Ops.BITCAST dtypes.f32 [198] dtypes.f32
200 Ops.CAST dtypes.f16 [199] dtypes.f16
201 Ops.STORE dtypes.void [176, 200]
202 Ops.STORE dtypes.void [124, 180]
203 Ops.PARAM dtypes.f16 [0] ParamArg(slot=2, dtype=dtypes.f16)
204 Ops.INDEX dtypes.f16 [203, 2]
205 Ops.LOAD dtypes.f16 [204]
206 Ops.ADD dtypes.f16 [150, 205]
207 Ops.STORE dtypes.void [176, 206]
208 Ops.PARAM dtypes.fp8e4m3 [0] ParamArg(slot=0, dtype=dtypes.fp8e4m3)
209 Ops.INDEX dtypes.fp8e4m3 [208, 2]
210 Ops.PARAM dtypes.fp8e4m3 [0] ParamArg(slot=1, dtype=dtypes.fp8e4m3)
211 Ops.INDEX dtypes.fp8e4m3 [210, 2]
212 Ops.LOAD dtypes.fp8e4m3 [211]
213 Ops.ADD dtypes.fp8e4m3 [212, 11]
214 Ops.STORE dtypes.void [209, 213]
215 Ops.PARAM dtypes.fp8e4m3 [0] ParamArg(slot=2, dtype=dtypes.fp8e4m3)
216 Ops.INDEX dtypes.fp8e4m3 [215, 2]
217 Ops.LOAD dtypes.fp8e4m3 [216]
218 Ops.CMPLT dtypes.bool [212, 217]
219 Ops.STORE dtypes.void [73, 218]
220 Ops.PARAM dtypes.fp8e5m2 [0] ParamArg(slot=0, dtype=dtypes.fp8e5m2)
221 Ops.INDEX dtypes.fp8e5m2 [220, 2]
222 Ops.CAST dtypes.fp8e5m2 [212] dtypes.fp8e5m2
223 Ops.STORE dtypes.void [221, 222]
224 Ops.CAST dtypes.fp8e4m3 [212] dtypes.fp8e4m3
225 Ops.STORE dtypes.void [209, 224]
226 Ops.PARAM dtypes.fp8e4m3fnuz [0] ParamArg(slot=0, dtype=dtypes.fp8e4m3fnuz)
227 Ops.INDEX dtypes.fp8e4m3fnuz [226, 2]
228 Ops.CAST dtypes.fp8e4m3fnuz [212] dtypes.fp8e4m3fnuz
229 Ops.STORE dtypes.void [227, 228]
230 Ops.CAST dtypes.fp8e4m3 [128] dtypes.fp8e4m3
231 Ops.STORE dtypes.void [209, 230]
232 Ops.CAST dtypes.fp8e5m2 [150] dtypes.fp8e5m2
233 Ops.STORE dtypes.void [221, 232]
234 Ops.PARAM dtypes.fp8e5m2 [0] ParamArg(slot=1, dtype=dtypes.fp8e5m2)
235 Ops.INDEX dtypes.fp8e5m2 [234, 2]
236 Ops.LOAD dtypes.fp8e5m2 [235]
237 Ops.CAST dtypes.i32 [236] dtypes.i32
238 Ops.STORE dtypes.void [172, 237]
239 Ops.PARAM dtypes.fp8e4m3fnuz [0] ParamArg(slot=1, dtype=dtypes.fp8e4m3fnuz)
240 Ops.INDEX dtypes.fp8e4m3fnuz [239, 2]
241 Ops.LOAD dtypes.fp8e4m3fnuz [240]
242 Ops.PARAM dtypes.fp8e4m3fnuz [0] ParamArg(slot=2, dtype=dtypes.fp8e4m3fnuz)
243 Ops.INDEX dtypes.fp8e4m3fnuz [242, 2]
244 Ops.LOAD dtypes.fp8e4m3fnuz [243]
245 Ops.MUL dtypes.fp8e4m3fnuz [241, 244]
246 Ops.STORE dtypes.void [227, 245]
247 Ops.CAST dtypes.fp8e4m3 [23] dtypes.fp8e4m3
248 Ops.ADD dtypes.fp8e4m3 [212, 247]
249 Ops.CAST dtypes.fp8e4m3 [28] dtypes.fp8e4m3
250 Ops.ADD dtypes.fp8e4m3 [212, 249]
251 Ops.CONST dtypes.weakint [] 2
252 Ops.CAST dtypes.fp8e4m3 [251] dtypes.fp8e4m3
253 Ops.ADD dtypes.fp8e4m3 [212, 252]
254 Ops.CONST dtypes.weakint [] 3
255 Ops.CAST dtypes.fp8e4m3 [254] dtypes.fp8e4m3
256 Ops.ADD dtypes.fp8e4m3 [212, 255]
257 Ops.CONST dtypes.weakint [] 4
258 Ops.CAST dtypes.fp8e4m3 [257] dtypes.fp8e4m3
259 Ops.ADD dtypes.fp8e4m3 [212, 258]
260 Ops.CONST dtypes.weakint [] 5
261 Ops.CAST dtypes.fp8e4m3 [260] dtypes.fp8e4m3
262 Ops.ADD dtypes.fp8e4m3 [212, 261]
263 Ops.CONST dtypes.weakint [] 6
264 Ops.CAST dtypes.fp8e4m3 [263] dtypes.fp8e4m3
265 Ops.ADD dtypes.fp8e4m3 [212, 264]
266 Ops.CONST dtypes.weakint [] 7
267 Ops.CAST dtypes.fp8e4m3 [266] dtypes.fp8e4m3
268 Ops.ADD dtypes.fp8e4m3 [212, 267]
269 Ops.STACK dtypes.fp8e4m3 [248, 250, 253, 256, 259, 262, 265, 268]
270 Ops.ADD dtypes.fp8e4m3 [217, 247]
271 Ops.ADD dtypes.fp8e4m3 [217, 249]
272 Ops.ADD dtypes.fp8e4m3 [217, 252]
273 Ops.ADD dtypes.fp8e4m3 [217, 255]
274 Ops.ADD dtypes.fp8e4m3 [217, 258]
275 Ops.ADD dtypes.fp8e4m3 [217, 261]
276 Ops.ADD dtypes.fp8e4m3 [217, 264]
277 Ops.ADD dtypes.fp8e4m3 [217, 267]
278 Ops.STACK dtypes.fp8e4m3 [270, 271, 272, 273, 274, 275, 276, 277]
279 Ops.STACK dtypes.f32 [78, 78, 78, 78]
280 Ops.WMMA dtypes.f32 [269, 278, 279] ((16, 16, 32), dtypes.fp8e4m3, 64, None)
281 Ops.PARAM dtypes.fp8e5m2fnuz [0] ParamArg(slot=1, dtype=dtypes.fp8e5m2fnuz)
282 Ops.INDEX dtypes.fp8e5m2fnuz [281, 2]
283 Ops.LOAD dtypes.fp8e5m2fnuz [282]
284 Ops.CAST dtypes.fp8e5m2fnuz [23] dtypes.fp8e5m2fnuz
285 Ops.ADD dtypes.fp8e5m2fnuz [283, 284]
286 Ops.CAST dtypes.fp8e5m2fnuz [28] dtypes.fp8e5m2fnuz
287 Ops.ADD dtypes.fp8e5m2fnuz [283, 286]
288 Ops.CAST dtypes.fp8e5m2fnuz [251] dtypes.fp8e5m2fnuz
289 Ops.ADD dtypes.fp8e5m2fnuz [283, 288]
290 Ops.CAST dtypes.fp8e5m2fnuz [254] dtypes.fp8e5m2fnuz
291 Ops.ADD dtypes.fp8e5m2fnuz [283, 290]
292 Ops.CAST dtypes.fp8e5m2fnuz [257] dtypes.fp8e5m2fnuz
293 Ops.ADD dtypes.fp8e5m2fnuz [283, 292]
294 Ops.CAST dtypes.fp8e5m2fnuz [260] dtypes.fp8e5m2fnuz
295 Ops.ADD dtypes.fp8e5m2fnuz [283, 294]
296 Ops.CAST dtypes.fp8e5m2fnuz [263] dtypes.fp8e5m2fnuz
297 Ops.ADD dtypes.fp8e5m2fnuz [283, 296]
298 Ops.CAST dtypes.fp8e5m2fnuz [266] dtypes.fp8e5m2fnuz
299 Ops.ADD dtypes.fp8e5m2fnuz [283, 298]
300 Ops.STACK dtypes.fp8e5m2fnuz [285, 287, 289, 291, 293, 295, 297, 299]
301 Ops.PARAM dtypes.fp8e5m2fnuz [0] ParamArg(slot=2, dtype=dtypes.fp8e5m2fnuz)
302 Ops.INDEX dtypes.fp8e5m2fnuz [301, 2]
303 Ops.LOAD dtypes.fp8e5m2fnuz [302]
304 Ops.ADD dtypes.fp8e5m2fnuz [303, 284]
305 Ops.ADD dtypes.fp8e5m2fnuz [303, 286]
306 Ops.ADD dtypes.fp8e5m2fnuz [303, 288]
307 Ops.ADD dtypes.fp8e5m2fnuz [303, 290]
308 Ops.ADD dtypes.fp8e5m2fnuz [303, 292]
309 Ops.ADD dtypes.fp8e5m2fnuz [303, 294]
310 Ops.ADD dtypes.fp8e5m2fnuz [303, 296]
311 Ops.ADD dtypes.fp8e5m2fnuz [303, 298]
312 Ops.STACK dtypes.fp8e5m2fnuz [304, 305, 306, 307, 308, 309, 310, 311]
313 Ops.WMMA dtypes.f32 [300, 312, 279] ((16, 16, 32), dtypes.fp8e5m2fnuz, 64, None)
314 Ops.CAST dtypes.f16 [23] dtypes.f16
315 Ops.ADD dtypes.f16 [150, 314]
316 Ops.CAST dtypes.f16 [28] dtypes.f16
317 Ops.ADD dtypes.f16 [150, 316]
318 Ops.CAST dtypes.f16 [251] dtypes.f16
319 Ops.ADD dtypes.f16 [150, 318]
320 Ops.CAST dtypes.f16 [254] dtypes.f16
321 Ops.ADD dtypes.f16 [150, 320]
322 Ops.CAST dtypes.f16 [257] dtypes.f16
323 Ops.ADD dtypes.f16 [150, 322]
324 Ops.CAST dtypes.f16 [260] dtypes.f16
325 Ops.ADD dtypes.f16 [150, 324]
326 Ops.CAST dtypes.f16 [263] dtypes.f16
327 Ops.ADD dtypes.f16 [150, 326]
328 Ops.CAST dtypes.f16 [266] dtypes.f16
329 Ops.ADD dtypes.f16 [150, 328]
330 Ops.STACK dtypes.f16 [315, 317, 319, 321, 323, 325, 327, 329]
331 Ops.ADD dtypes.f16 [205, 314]
332 Ops.ADD dtypes.f16 [205, 316]
333 Ops.ADD dtypes.f16 [205, 318]
334 Ops.ADD dtypes.f16 [205, 320]
335 Ops.ADD dtypes.f16 [205, 322]
336 Ops.ADD dtypes.f16 [205, 324]
337 Ops.ADD dtypes.f16 [205, 326]
338 Ops.ADD dtypes.f16 [205, 328]
339 Ops.STACK dtypes.f16 [331, 332, 333, 334, 335, 336, 337, 338]
340 Ops.WMMA dtypes.f32 [330, 339, 279] ((16, 16, 32), dtypes.f16, 64, None)
341 Ops.ADD dtypes.bf16 [6, 11]
342 Ops.STORE dtypes.void [3, 341]
343 Ops.MUL dtypes.bf16 [6, 47]
344 Ops.STORE dtypes.void [3, 343]
345 Ops.CMPLT dtypes.bool [6, 47]
346 Ops.STORE dtypes.void [73, 345]
347 Ops.CAST dtypes.bf16 [150] dtypes.bf16
348 Ops.STORE dtypes.void [3, 347]
349 Ops.CAST dtypes.i32 [6] dtypes.i32
350 Ops.STORE dtypes.void [172, 349]
351 Ops.CAST dtypes.f16 [179] dtypes.f16
352 Ops.STORE dtypes.void [176, 351]
353 Ops.SQRT dtypes.bf16 [6]
354 Ops.STORE dtypes.void [3, 353]
355 Ops.EXP2 dtypes.bf16 [6]
356 Ops.STORE dtypes.void [3, 355]
357 Ops.CAST dtypes.f32 [6] dtypes.f32
358 Ops.STORE dtypes.void [124, 357]
359 Ops.CAST dtypes.bf16 [128] dtypes.bf16
360 Ops.STORE dtypes.void [3, 359]
361 Ops.CAST dtypes.f32 [212] dtypes.f32
362 Ops.CAST dtypes.fp8e4m3 [11] dtypes.fp8e4m3
363 Ops.CAST dtypes.f32 [362] dtypes.f32
364 Ops.ADD dtypes.f32 [361, 363]
365 Ops.CAST dtypes.fp8e4m3 [364] dtypes.fp8e4m3
366 Ops.STORE dtypes.void [209, 365]
367 Ops.CAST dtypes.f32 [217] dtypes.f32
368 Ops.CMPLT dtypes.bool [361, 367]
369 Ops.STORE dtypes.void [73, 368]
370 Ops.CAST dtypes.fp8e5m2 [361] dtypes.fp8e5m2
371 Ops.STORE dtypes.void [221, 370]
372 Ops.CAST dtypes.fp8e4m3fnuz [361] dtypes.fp8e4m3fnuz
373 Ops.STORE dtypes.void [227, 372]
374 Ops.CAST dtypes.f32 [241] dtypes.f32
375 Ops.CAST dtypes.f32 [244] dtypes.f32
376 Ops.MUL dtypes.f32 [374, 375]
377 Ops.CAST dtypes.fp8e4m3fnuz [376] dtypes.fp8e4m3fnuz
378 Ops.STORE dtypes.void [227, 377]
379 Ops.CAST dtypes.f32 [247] dtypes.f32
380 Ops.ADD dtypes.f32 [361, 379]
381 Ops.CAST dtypes.fp8e4m3 [380] dtypes.fp8e4m3
382 Ops.CAST dtypes.f32 [249] dtypes.f32
383 Ops.ADD dtypes.f32 [361, 382]
384 Ops.CAST dtypes.fp8e4m3 [383] dtypes.fp8e4m3
385 Ops.CAST dtypes.f32 [252] dtypes.f32
386 Ops.ADD dtypes.f32 [361, 385]
387 Ops.CAST dtypes.fp8e4m3 [386] dtypes.fp8e4m3
388 Ops.CAST dtypes.f32 [255] dtypes.f32
389 Ops.ADD dtypes.f32 [361, 388]
390 Ops.CAST dtypes.fp8e4m3 [389] dtypes.fp8e4m3
391 Ops.CAST dtypes.f32 [258] dtypes.f32
392 Ops.ADD dtypes.f32 [361, 391]
393 Ops.CAST dtypes.fp8e4m3 [392] dtypes.fp8e4m3
394 Ops.CAST dtypes.f32 [261] dtypes.f32
395 Ops.ADD dtypes.f32 [361, 394]
396 Ops.CAST dtypes.fp8e4m3 [395] dtypes.fp8e4m3
397 Ops.CAST dtypes.f32 [264] dtypes.f32
398 Ops.ADD dtypes.f32 [361, 397]
399 Ops.CAST dtypes.fp8e4m3 [398] dtypes.fp8e4m3
400 Ops.CAST dtypes.f32 [267] dtypes.f32
401 Ops.ADD dtypes.f32 [361, 400]
402 Ops.CAST dtypes.fp8e4m3 [401] dtypes.fp8e4m3
403 Ops.STACK dtypes.fp8e4m3 [381, 384, 387, 390, 393, 396, 399, 402]
404 Ops.ADD dtypes.f32 [367, 379]
405 Ops.CAST dtypes.fp8e4m3 [404] dtypes.fp8e4m3
406 Ops.ADD dtypes.f32 [367, 382]
407 Ops.CAST dtypes.fp8e4m3 [406] dtypes.fp8e4m3
408 Ops.ADD dtypes.f32 [367, 385]
409 Ops.CAST dtypes.fp8e4m3 [408] dtypes.fp8e4m3
410 Ops.ADD dtypes.f32 [367, 388]
411 Ops.CAST dtypes.fp8e4m3 [410] dtypes.fp8e4m3
412 Ops.ADD dtypes.f32 [367, 391]
413 Ops.CAST dtypes.fp8e4m3 [412] dtypes.fp8e4m3
414 Ops.ADD dtypes.f32 [367, 394]
415 Ops.CAST dtypes.fp8e4m3 [414] dtypes.fp8e4m3
416 Ops.ADD dtypes.f32 [367, 397]
417 Ops.CAST dtypes.fp8e4m3 [416] dtypes.fp8e4m3
418 Ops.ADD dtypes.f32 [367, 400]
419 Ops.CAST dtypes.fp8e4m3 [418] dtypes.fp8e4m3
420 Ops.STACK dtypes.fp8e4m3 [405, 407, 409, 411, 413, 415, 417, 419]
421 Ops.WMMA dtypes.f32 [403, 420, 279] ((16, 16, 32), dtypes.fp8e4m3, 64, None)
422 Ops.CAST dtypes.f32 [283] dtypes.f32
423 Ops.CAST dtypes.f32 [284] dtypes.f32
424 Ops.ADD dtypes.f32 [422, 423]
425 Ops.CAST dtypes.fp8e5m2fnuz [424] dtypes.fp8e5m2fnuz
426 Ops.CAST dtypes.f32 [286] dtypes.f32
427 Ops.ADD dtypes.f32 [422, 426]
428 Ops.CAST dtypes.fp8e5m2fnuz [427] dtypes.fp8e5m2fnuz
429 Ops.CAST dtypes.f32 [288] dtypes.f32
430 Ops.ADD dtypes.f32 [422, 429]
431 Ops.CAST dtypes.fp8e5m2fnuz [430] dtypes.fp8e5m2fnuz
432 Ops.CAST dtypes.f32 [290] dtypes.f32
433 Ops.ADD dtypes.f32 [422, 432]
434 Ops.CAST dtypes.fp8e5m2fnuz [433] dtypes.fp8e5m2fnuz
435 Ops.CAST dtypes.f32 [292] dtypes.f32
436 Ops.ADD dtypes.f32 [422, 435]
437 Ops.CAST dtypes.fp8e5m2fnuz [436] dtypes.fp8e5m2fnuz
438 Ops.CAST dtypes.f32 [294] dtypes.f32
439 Ops.ADD dtypes.f32 [422, 438]
440 Ops.CAST dtypes.fp8e5m2fnuz [439] dtypes.fp8e5m2fnuz
441 Ops.CAST dtypes.f32 [296] dtypes.f32
442 Ops.ADD dtypes.f32 [422, 441]
443 Ops.CAST dtypes.fp8e5m2fnuz [442] dtypes.fp8e5m2fnuz
444 Ops.CAST dtypes.f32 [298] dtypes.f32
445 Ops.ADD dtypes.f32 [422, 444]
446 Ops.CAST dtypes.fp8e5m2fnuz [445] dtypes.fp8e5m2fnuz
447 Ops.STACK dtypes.fp8e5m2fnuz [425, 428, 431, 434, 437, 440, 443, 446]
448 Ops.CAST dtypes.f32 [303] dtypes.f32
449 Ops.ADD dtypes.f32 [448, 423]
450 Ops.CAST dtypes.fp8e5m2fnuz [449] dtypes.fp8e5m2fnuz
451 Ops.ADD dtypes.f32 [448, 426]
452 Ops.CAST dtypes.fp8e5m2fnuz [451] dtypes.fp8e5m2fnuz
453 Ops.ADD dtypes.f32 [448, 429]
454 Ops.CAST dtypes.fp8e5m2fnuz [453] dtypes.fp8e5m2fnuz
455 Ops.ADD dtypes.f32 [448, 432]
456 Ops.CAST dtypes.fp8e5m2fnuz [455] dtypes.fp8e5m2fnuz
457 Ops.ADD dtypes.f32 [448, 435]
458 Ops.CAST dtypes.fp8e5m2fnuz [457] dtypes.fp8e5m2fnuz
459 Ops.ADD dtypes.f32 [448, 438]
460 Ops.CAST dtypes.fp8e5m2fnuz [459] dtypes.fp8e5m2fnuz
461 Ops.ADD dtypes.f32 [448, 441]
462 Ops.CAST dtypes.fp8e5m2fnuz [461] dtypes.fp8e5m2fnuz
463 Ops.ADD dtypes.f32 [448, 444]
464 Ops.CAST dtypes.fp8e5m2fnuz [463] dtypes.fp8e5m2fnuz
465 Ops.STACK dtypes.fp8e5m2fnuz [450, 452, 454, 456, 458, 460, 462, 464]
466 Ops.WMMA dtypes.f32 [447, 465, 279] ((16, 16, 32), dtypes.fp8e5m2fnuz, 64, None)
467 Ops.CAST dtypes.f32 [12] dtypes.f32
468 Ops.ADD dtypes.f32 [357, 467]
469 Ops.CAST dtypes.bf16 [468] dtypes.bf16
470 Ops.STORE dtypes.void [3, 469]
471 Ops.CAST dtypes.f32 [47] dtypes.f32
472 Ops.MUL dtypes.f32 [357, 471]
473 Ops.CAST dtypes.bf16 [472] dtypes.bf16
474 Ops.STORE dtypes.void [3, 473]
475 Ops.CMPLT dtypes.bool [357, 471]
476 Ops.STORE dtypes.void [73, 475]
477 Ops.SQRT dtypes.f32 [357]
478 Ops.CAST dtypes.bf16 [477] dtypes.bf16
479 Ops.STORE dtypes.void [3, 478]
480 Ops.EXP2 dtypes.f32 [357]
481 Ops.CAST dtypes.bf16 [480] dtypes.bf16
482 Ops.STORE dtypes.void [3, 481]
483 Ops.CAST dtypes.bf16 [151] dtypes.bf16
484 Ops.STORE dtypes.void [3, 483]
485 Ops.CAST dtypes.i32 [357] dtypes.i32
486 Ops.STORE dtypes.void [172, 485]
487 Ops.CAST dtypes.fp8e4m3 [361] dtypes.fp8e4m3
488 Ops.STORE dtypes.void [209, 487]
489 Ops.CAST dtypes.fp8e5m2 [151] dtypes.fp8e5m2
490 Ops.STORE dtypes.void [221, 489]
491 Ops.CAST dtypes.f32 [236] dtypes.f32
492 Ops.CAST dtypes.i32 [491] dtypes.i32
493 Ops.STORE dtypes.void [172, 492]
494 Ops.BITCAST dtypes.u32 [403] dtypes.u32
495 Ops.BITCAST dtypes.u32 [420] dtypes.u32
496 Ops.WMMA dtypes.f32 [494, 495, 279] ((16, 16, 32), dtypes.fp8e4m3, 64, None)
497 Ops.BITCAST dtypes.u32 [447] dtypes.u32
498 Ops.BITCAST dtypes.u32 [465] dtypes.u32
499 Ops.WMMA dtypes.f32 [497, 498, 279] ((16, 16, 32), dtypes.fp8e5m2fnuz, 64, None)
500 Ops.BITCAST dtypes.u64 [403] dtypes.u64
501 Ops.BITCAST dtypes.u64 [420] dtypes.u64
502 Ops.WMMA dtypes.f32 [500, 501, 279] ((16, 16, 32), dtypes.fp8e4m3, 64, None)
503 Ops.BITCAST dtypes.u64 [447] dtypes.u64
504 Ops.BITCAST dtypes.u64 [465] dtypes.u64
505 Ops.WMMA dtypes.f32 [503, 504, 279] ((16, 16, 32), dtypes.fp8e5m2fnuz, 64, None)
506 Ops.SINK dtypes.void [44, 71, 75, 82, 102, 122, 125, 147, 170, 174, 201, 202, 207, 214, 219, 223, 225, 229, 231, 233, 238, 246, 280, 313, 340, 342, 344, 346, 82, 102, 122, 125, 147, 348, 350, 352, 202, 207, 214, 219, 223, 225, 229, 231, 233, 238, 246, 280, 313, 340, 342, 344, 346, 82, 354, 356, 358, 360, 348, 350, 352, 202, 207, 366, 369, 371, 225, 373, 231, 233, 238, 378, 421, 466, 340, 470, 474, 476, 82, 479, 482, 358, 360, 484, 486, 352, 202, 207, 366, 369, 371, 488, 373, 231, 490, 493, 378, 421, 466, 340, 470, 474, 476, 82, 479, 482, 358, 360, 484, 486, 352, 202, 207, 366, 369, 371, 488, 373, 231, 490, 493, 378, 496, 499, 340, 470, 474, 476, 82, 479, 482, 358, 360, 484, 486, 352, 202, 207, 366, 369, 371, 488, 373, 231, 490, 493, 378, 502, 505, 340, 470, 474, 476, 82, 479, 482, 358, 360, 484, 486, 352, 202, 207, 366, 369, 371, 488, 373, 231, 490, 493, 378, 502, 505, 340]

