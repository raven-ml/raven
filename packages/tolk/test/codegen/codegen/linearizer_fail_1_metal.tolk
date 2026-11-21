# tolk D25 D89 golden=1f7ccc826a92f457650eaf2cbfdf1e96
0 Ops.CONST dtypes.weakint [] 64
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.SPECIAL dtypes.i32 [3] "gidx0"
5 Ops.CONST dtypes.weakint [] 2
6 Ops.CAST dtypes.i32 [5] dtypes.i32
7 Ops.SHL dtypes.i32 [4, 6]
8 Ops.CONST dtypes.weakint [] 4
9 Ops.CAST dtypes.i32 [8] dtypes.i32
10 Ops.SPECIAL dtypes.i32 [9] "lidx1"
11 Ops.ADD dtypes.i32 [7, 10]
12 Ops.INDEX dtypes.f32 [1, 11]
13 Ops.CONST dtypes.weakint [] 1
14 Ops.ALLOC dtypes.f32 [13] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
15 Ops.AFTER dtypes.f32 [14, 4, 10]
16 Ops.CONST dtypes.weakint [] 0
17 Ops.CAST dtypes.i32 [16] dtypes.i32
18 Ops.INDEX dtypes.f32 [15, 17]
19 Ops.CONST dtypes.weakfloat [] 0.0
20 Ops.CAST dtypes.f32 [19] dtypes.f32
21 Ops.STORE dtypes.void [18, 20]
22 Ops.CONST dtypes.weakint [] 32
23 Ops.CAST dtypes.i32 [22] dtypes.i32
24 Ops.ALLOC dtypes.f32 [13] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
25 Ops.SPECIAL dtypes.i32 [23] "lidx0"
26 Ops.AFTER dtypes.f32 [24, 25, 4, 10]
27 Ops.INDEX dtypes.f32 [26, 17]
28 Ops.STORE dtypes.void [27, 20]
29 Ops.RANGE dtypes.i32 [9] (AxisType.WEAK, 0, 0, 0)
30 Ops.AFTER dtypes.f32 [24, 28, 29]
31 Ops.INDEX dtypes.f32 [30, 17]
32 Ops.LOAD dtypes.f32 [31]
33 Ops.CONST dtypes.weakint [] 163840
34 Ops.PARAM dtypes.f32 [33] ParamArg(slot=1, dtype=dtypes.f32)
35 Ops.CONST dtypes.weakint [] 10240
36 Ops.CAST dtypes.i32 [35] dtypes.i32
37 Ops.MUL dtypes.i32 [4, 36]
38 Ops.CONST dtypes.weakint [] 2560
39 Ops.CAST dtypes.i32 [38] dtypes.i32
40 Ops.MUL dtypes.i32 [10, 39]
41 Ops.ADD dtypes.i32 [37, 40]
42 Ops.CONST dtypes.weakint [] 640
43 Ops.CAST dtypes.i32 [42] dtypes.i32
44 Ops.MUL dtypes.i32 [29, 43]
45 Ops.ADD dtypes.i32 [44, 25]
46 Ops.ADD dtypes.i32 [41, 45]
47 Ops.CONST dtypes.weakint [] 608
48 Ops.CAST dtypes.i32 [47] dtypes.i32
49 Ops.ADD dtypes.i32 [46, 48]
50 Ops.INDEX dtypes.f32 [34, 49]
51 Ops.LOAD dtypes.f32 [50]
52 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
53 Ops.INDEX dtypes.f32 [52, 11]
54 Ops.LOAD dtypes.f32 [53]
55 Ops.SUB dtypes.f32 [51, 54]
56 Ops.CONST dtypes.weakint [] 576
57 Ops.CAST dtypes.i32 [56] dtypes.i32
58 Ops.ADD dtypes.i32 [46, 57]
59 Ops.INDEX dtypes.f32 [34, 58]
60 Ops.LOAD dtypes.f32 [59]
61 Ops.SUB dtypes.f32 [60, 54]
62 Ops.CONST dtypes.weakint [] 544
63 Ops.CAST dtypes.i32 [62] dtypes.i32
64 Ops.ADD dtypes.i32 [46, 63]
65 Ops.INDEX dtypes.f32 [34, 64]
66 Ops.LOAD dtypes.f32 [65]
67 Ops.SUB dtypes.f32 [66, 54]
68 Ops.CONST dtypes.weakint [] 512
69 Ops.CAST dtypes.i32 [68] dtypes.i32
70 Ops.ADD dtypes.i32 [46, 69]
71 Ops.INDEX dtypes.f32 [34, 70]
72 Ops.LOAD dtypes.f32 [71]
73 Ops.SUB dtypes.f32 [72, 54]
74 Ops.CONST dtypes.weakint [] 480
75 Ops.CAST dtypes.i32 [74] dtypes.i32
76 Ops.ADD dtypes.i32 [46, 75]
77 Ops.INDEX dtypes.f32 [34, 76]
78 Ops.LOAD dtypes.f32 [77]
79 Ops.SUB dtypes.f32 [78, 54]
80 Ops.CONST dtypes.weakint [] 448
81 Ops.CAST dtypes.i32 [80] dtypes.i32
82 Ops.ADD dtypes.i32 [46, 81]
83 Ops.INDEX dtypes.f32 [34, 82]
84 Ops.LOAD dtypes.f32 [83]
85 Ops.SUB dtypes.f32 [84, 54]
86 Ops.CONST dtypes.weakint [] 416
87 Ops.CAST dtypes.i32 [86] dtypes.i32
88 Ops.ADD dtypes.i32 [46, 87]
89 Ops.INDEX dtypes.f32 [34, 88]
90 Ops.LOAD dtypes.f32 [89]
91 Ops.SUB dtypes.f32 [90, 54]
92 Ops.CONST dtypes.weakint [] 384
93 Ops.CAST dtypes.i32 [92] dtypes.i32
94 Ops.ADD dtypes.i32 [46, 93]
95 Ops.INDEX dtypes.f32 [34, 94]
96 Ops.LOAD dtypes.f32 [95]
97 Ops.SUB dtypes.f32 [96, 54]
98 Ops.CONST dtypes.weakint [] 352
99 Ops.CAST dtypes.i32 [98] dtypes.i32
100 Ops.ADD dtypes.i32 [46, 99]
101 Ops.INDEX dtypes.f32 [34, 100]
102 Ops.LOAD dtypes.f32 [101]
103 Ops.SUB dtypes.f32 [102, 54]
104 Ops.CONST dtypes.weakint [] 320
105 Ops.CAST dtypes.i32 [104] dtypes.i32
106 Ops.ADD dtypes.i32 [46, 105]
107 Ops.INDEX dtypes.f32 [34, 106]
108 Ops.LOAD dtypes.f32 [107]
109 Ops.SUB dtypes.f32 [108, 54]
110 Ops.CONST dtypes.weakint [] 288
111 Ops.CAST dtypes.i32 [110] dtypes.i32
112 Ops.ADD dtypes.i32 [46, 111]
113 Ops.INDEX dtypes.f32 [34, 112]
114 Ops.LOAD dtypes.f32 [113]
115 Ops.SUB dtypes.f32 [114, 54]
116 Ops.CONST dtypes.weakint [] 256
117 Ops.CAST dtypes.i32 [116] dtypes.i32
118 Ops.ADD dtypes.i32 [46, 117]
119 Ops.INDEX dtypes.f32 [34, 118]
120 Ops.LOAD dtypes.f32 [119]
121 Ops.SUB dtypes.f32 [120, 54]
122 Ops.CONST dtypes.weakint [] 224
123 Ops.CAST dtypes.i32 [122] dtypes.i32
124 Ops.ADD dtypes.i32 [46, 123]
125 Ops.INDEX dtypes.f32 [34, 124]
126 Ops.LOAD dtypes.f32 [125]
127 Ops.SUB dtypes.f32 [126, 54]
128 Ops.CONST dtypes.weakint [] 192
129 Ops.CAST dtypes.i32 [128] dtypes.i32
130 Ops.ADD dtypes.i32 [46, 129]
131 Ops.INDEX dtypes.f32 [34, 130]
132 Ops.LOAD dtypes.f32 [131]
133 Ops.SUB dtypes.f32 [132, 54]
134 Ops.CONST dtypes.weakint [] 160
135 Ops.CAST dtypes.i32 [134] dtypes.i32
136 Ops.ADD dtypes.i32 [46, 135]
137 Ops.INDEX dtypes.f32 [34, 136]
138 Ops.LOAD dtypes.f32 [137]
139 Ops.SUB dtypes.f32 [138, 54]
140 Ops.CONST dtypes.weakint [] 128
141 Ops.CAST dtypes.i32 [140] dtypes.i32
142 Ops.ADD dtypes.i32 [46, 141]
143 Ops.INDEX dtypes.f32 [34, 142]
144 Ops.LOAD dtypes.f32 [143]
145 Ops.SUB dtypes.f32 [144, 54]
146 Ops.CONST dtypes.weakint [] 96
147 Ops.CAST dtypes.i32 [146] dtypes.i32
148 Ops.ADD dtypes.i32 [46, 147]
149 Ops.INDEX dtypes.f32 [34, 148]
150 Ops.LOAD dtypes.f32 [149]
151 Ops.SUB dtypes.f32 [150, 54]
152 Ops.CAST dtypes.i32 [0] dtypes.i32
153 Ops.ADD dtypes.i32 [46, 152]
154 Ops.INDEX dtypes.f32 [34, 153]
155 Ops.LOAD dtypes.f32 [154]
156 Ops.SUB dtypes.f32 [155, 54]
157 Ops.ADD dtypes.i32 [46, 23]
158 Ops.INDEX dtypes.f32 [34, 157]
159 Ops.LOAD dtypes.f32 [158]
160 Ops.SUB dtypes.f32 [159, 54]
161 Ops.INDEX dtypes.f32 [34, 46]
162 Ops.LOAD dtypes.f32 [161]
163 Ops.SUB dtypes.f32 [162, 54]
164 Ops.MUL dtypes.f32 [163, 163]
165 Ops.MULACC dtypes.f32 [160, 160, 164]
166 Ops.MULACC dtypes.f32 [156, 156, 165]
167 Ops.MULACC dtypes.f32 [151, 151, 166]
168 Ops.MULACC dtypes.f32 [145, 145, 167]
169 Ops.MULACC dtypes.f32 [139, 139, 168]
170 Ops.MULACC dtypes.f32 [133, 133, 169]
171 Ops.MULACC dtypes.f32 [127, 127, 170]
172 Ops.MULACC dtypes.f32 [121, 121, 171]
173 Ops.MULACC dtypes.f32 [115, 115, 172]
174 Ops.MULACC dtypes.f32 [109, 109, 173]
175 Ops.MULACC dtypes.f32 [103, 103, 174]
176 Ops.MULACC dtypes.f32 [97, 97, 175]
177 Ops.MULACC dtypes.f32 [91, 91, 176]
178 Ops.MULACC dtypes.f32 [85, 85, 177]
179 Ops.MULACC dtypes.f32 [79, 79, 178]
180 Ops.MULACC dtypes.f32 [73, 73, 179]
181 Ops.MULACC dtypes.f32 [67, 67, 180]
182 Ops.MULACC dtypes.f32 [61, 61, 181]
183 Ops.MULACC dtypes.f32 [55, 55, 182]
184 Ops.ADD dtypes.f32 [32, 183]
185 Ops.STORE dtypes.void [31, 184]
186 Ops.END dtypes.void [185, 29]
187 Ops.AFTER dtypes.i32 [23, 186]
188 Ops.RANGE dtypes.i32 [187] (AxisType.WEAK, 102)
189 Ops.AFTER dtypes.f32 [14, 21, 188]
190 Ops.INDEX dtypes.f32 [189, 17]
191 Ops.LOAD dtypes.f32 [190]
192 Ops.ALLOC dtypes.f32 [140] ParamArg(slot=2, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
193 Ops.CONST dtypes.weakint [] 5
194 Ops.CAST dtypes.i32 [193] dtypes.i32
195 Ops.SHL dtypes.i32 [10, 194]
196 Ops.ADD dtypes.i32 [195, 25]
197 Ops.INDEX dtypes.f32 [192, 196]
198 Ops.AFTER dtypes.f32 [24, 186]
199 Ops.INDEX dtypes.f32 [198, 17]
200 Ops.LOAD dtypes.f32 [199]
201 Ops.STORE dtypes.void [197, 200]
202 Ops.BARRIER dtypes.void [201]
203 Ops.AFTER dtypes.f32 [192, 202]
204 Ops.ADD dtypes.i32 [195, 188]
205 Ops.INDEX dtypes.f32 [203, 204]
206 Ops.LOAD dtypes.f32 [205]
207 Ops.ADD dtypes.f32 [191, 206]
208 Ops.STORE dtypes.void [190, 207]
209 Ops.END dtypes.void [208, 188]
210 Ops.AFTER dtypes.f32 [14, 209]
211 Ops.INDEX dtypes.f32 [210, 17]
212 Ops.LOAD dtypes.f32 [211]
213 Ops.CONST dtypes.weakfloat [] 0.0003906250058207661
214 Ops.CAST dtypes.f32 [213] dtypes.f32
215 Ops.MUL dtypes.f32 [212, 214]
216 Ops.CONST dtypes.weakfloat [] 9.999999747378752e-06
217 Ops.CAST dtypes.f32 [216] dtypes.f32
218 Ops.ADD dtypes.f32 [215, 217]
219 Ops.SQRT dtypes.f32 [218]
220 Ops.RECIP dtypes.f32 [219]
221 Ops.CMPEQ dtypes.bool [25, 17]
222 Ops.STORE dtypes.void [12, 220, 221]
223 Ops.SINK dtypes.void [222] KernelInfo(name="r_16_32_4_4_20", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(32, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=3, arg=(20, AxisType.UPCAST))), optimized=True)
224 Ops.CONST dtypes.weakint [] 24
225 Ops.PARAM dtypes.u8 [224] ParamArg(slot=0, dtype=dtypes.u8, name="args")
226 Ops.BITCAST dtypes.u64 [225] dtypes.u64
227 Ops.INDEX dtypes.u64 [226, 17]
228 Ops.LOAD dtypes.u64 [227]
229 Ops.REFPARAM dtypes.f32 [0, 228] ParamArg(slot=0, dtype=dtypes.f32)
230 Ops.INDEX dtypes.f32 [229, 11]
231 Ops.BUFFER dtypes.f32 [13] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
232 Ops.AFTER dtypes.f32 [231, 4, 10]
233 Ops.INDEX dtypes.f32 [232, 17]
234 Ops.STORE dtypes.void [233, 20]
235 Ops.BUFFER dtypes.f32 [13] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
236 Ops.AFTER dtypes.f32 [235, 25, 4, 10]
237 Ops.INDEX dtypes.f32 [236, 17]
238 Ops.STORE dtypes.void [237, 20]
239 Ops.AFTER dtypes.f32 [235, 238, 29]
240 Ops.INDEX dtypes.f32 [239, 17]
241 Ops.LOAD dtypes.f32 [240]
242 Ops.CAST dtypes.i32 [13] dtypes.i32
243 Ops.INDEX dtypes.u64 [226, 242]
244 Ops.LOAD dtypes.u64 [243]
245 Ops.REFPARAM dtypes.f32 [33, 244] ParamArg(slot=1, dtype=dtypes.f32)
246 Ops.INDEX dtypes.f32 [245, 49]
247 Ops.LOAD dtypes.f32 [246]
248 Ops.INDEX dtypes.u64 [226, 6]
249 Ops.LOAD dtypes.u64 [248]
250 Ops.REFPARAM dtypes.f32 [0, 249] ParamArg(slot=2, dtype=dtypes.f32)
251 Ops.INDEX dtypes.f32 [250, 11]
252 Ops.LOAD dtypes.f32 [251]
253 Ops.SUB dtypes.f32 [247, 252]
254 Ops.INDEX dtypes.f32 [245, 58]
255 Ops.LOAD dtypes.f32 [254]
256 Ops.SUB dtypes.f32 [255, 252]
257 Ops.INDEX dtypes.f32 [245, 64]
258 Ops.LOAD dtypes.f32 [257]
259 Ops.SUB dtypes.f32 [258, 252]
260 Ops.INDEX dtypes.f32 [245, 70]
261 Ops.LOAD dtypes.f32 [260]
262 Ops.SUB dtypes.f32 [261, 252]
263 Ops.INDEX dtypes.f32 [245, 76]
264 Ops.LOAD dtypes.f32 [263]
265 Ops.SUB dtypes.f32 [264, 252]
266 Ops.INDEX dtypes.f32 [245, 82]
267 Ops.LOAD dtypes.f32 [266]
268 Ops.SUB dtypes.f32 [267, 252]
269 Ops.INDEX dtypes.f32 [245, 88]
270 Ops.LOAD dtypes.f32 [269]
271 Ops.SUB dtypes.f32 [270, 252]
272 Ops.INDEX dtypes.f32 [245, 94]
273 Ops.LOAD dtypes.f32 [272]
274 Ops.SUB dtypes.f32 [273, 252]
275 Ops.INDEX dtypes.f32 [245, 100]
276 Ops.LOAD dtypes.f32 [275]
277 Ops.SUB dtypes.f32 [276, 252]
278 Ops.INDEX dtypes.f32 [245, 106]
279 Ops.LOAD dtypes.f32 [278]
280 Ops.SUB dtypes.f32 [279, 252]
281 Ops.INDEX dtypes.f32 [245, 112]
282 Ops.LOAD dtypes.f32 [281]
283 Ops.SUB dtypes.f32 [282, 252]
284 Ops.INDEX dtypes.f32 [245, 118]
285 Ops.LOAD dtypes.f32 [284]
286 Ops.SUB dtypes.f32 [285, 252]
287 Ops.INDEX dtypes.f32 [245, 124]
288 Ops.LOAD dtypes.f32 [287]
289 Ops.SUB dtypes.f32 [288, 252]
290 Ops.INDEX dtypes.f32 [245, 130]
291 Ops.LOAD dtypes.f32 [290]
292 Ops.SUB dtypes.f32 [291, 252]
293 Ops.INDEX dtypes.f32 [245, 136]
294 Ops.LOAD dtypes.f32 [293]
295 Ops.SUB dtypes.f32 [294, 252]
296 Ops.INDEX dtypes.f32 [245, 142]
297 Ops.LOAD dtypes.f32 [296]
298 Ops.SUB dtypes.f32 [297, 252]
299 Ops.INDEX dtypes.f32 [245, 148]
300 Ops.LOAD dtypes.f32 [299]
301 Ops.SUB dtypes.f32 [300, 252]
302 Ops.INDEX dtypes.f32 [245, 153]
303 Ops.LOAD dtypes.f32 [302]
304 Ops.SUB dtypes.f32 [303, 252]
305 Ops.INDEX dtypes.f32 [245, 157]
306 Ops.LOAD dtypes.f32 [305]
307 Ops.SUB dtypes.f32 [306, 252]
308 Ops.INDEX dtypes.f32 [245, 46]
309 Ops.LOAD dtypes.f32 [308]
310 Ops.SUB dtypes.f32 [309, 252]
311 Ops.MUL dtypes.f32 [310, 310]
312 Ops.MULACC dtypes.f32 [307, 307, 311]
313 Ops.MULACC dtypes.f32 [304, 304, 312]
314 Ops.MULACC dtypes.f32 [301, 301, 313]
315 Ops.MULACC dtypes.f32 [298, 298, 314]
316 Ops.MULACC dtypes.f32 [295, 295, 315]
317 Ops.MULACC dtypes.f32 [292, 292, 316]
318 Ops.MULACC dtypes.f32 [289, 289, 317]
319 Ops.MULACC dtypes.f32 [286, 286, 318]
320 Ops.MULACC dtypes.f32 [283, 283, 319]
321 Ops.MULACC dtypes.f32 [280, 280, 320]
322 Ops.MULACC dtypes.f32 [277, 277, 321]
323 Ops.MULACC dtypes.f32 [274, 274, 322]
324 Ops.MULACC dtypes.f32 [271, 271, 323]
325 Ops.MULACC dtypes.f32 [268, 268, 324]
326 Ops.MULACC dtypes.f32 [265, 265, 325]
327 Ops.MULACC dtypes.f32 [262, 262, 326]
328 Ops.MULACC dtypes.f32 [259, 259, 327]
329 Ops.MULACC dtypes.f32 [256, 256, 328]
330 Ops.MULACC dtypes.f32 [253, 253, 329]
331 Ops.ADD dtypes.f32 [241, 330]
332 Ops.STORE dtypes.void [240, 331]
333 Ops.END dtypes.void [332, 29]
334 Ops.AFTER dtypes.i32 [23, 333]
335 Ops.RANGE dtypes.i32 [334] (AxisType.WEAK, 102)
336 Ops.AFTER dtypes.f32 [231, 234, 335]
337 Ops.INDEX dtypes.f32 [336, 17]
338 Ops.LOAD dtypes.f32 [337]
339 Ops.BUFFER dtypes.f32 [140] ParamArg(slot=2, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
340 Ops.INDEX dtypes.f32 [339, 196]
341 Ops.AFTER dtypes.f32 [235, 333]
342 Ops.INDEX dtypes.f32 [341, 17]
343 Ops.LOAD dtypes.f32 [342]
344 Ops.STORE dtypes.void [340, 343]
345 Ops.BARRIER dtypes.void [344]
346 Ops.AFTER dtypes.f32 [339, 345]
347 Ops.ADD dtypes.i32 [195, 335]
348 Ops.INDEX dtypes.f32 [346, 347]
349 Ops.LOAD dtypes.f32 [348]
350 Ops.ADD dtypes.f32 [338, 349]
351 Ops.STORE dtypes.void [337, 350]
352 Ops.END dtypes.void [351, 335]
353 Ops.AFTER dtypes.f32 [231, 352]
354 Ops.INDEX dtypes.f32 [353, 17]
355 Ops.LOAD dtypes.f32 [354]
356 Ops.MUL dtypes.f32 [355, 214]
357 Ops.ADD dtypes.f32 [356, 217]
358 Ops.SQRT dtypes.f32 [357]
359 Ops.RECIP dtypes.f32 [358]
360 Ops.STORE dtypes.void [230, 359]
361 Ops.SINK dtypes.void [360] KernelInfo(name="r_16_32_4_4_20", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(32, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=3, arg=(20, AxisType.UPCAST))), estimates=Estimates(ops=567296, lds=934144, mem=655872), optimized=True)
362 Ops.IF dtypes.void [221, 230]
363 Ops.ENDIF dtypes.void [362]
364 Ops.LINEAR dtypes.void [224, 225, 13, 235, 231, 140, 339, 242, 226, 243, 244, 5, 6, 248, 249, 2, 3, 4, 8, 9, 10, 0, 250, 7, 11, 251, 252, 22, 23, 25, 33, 245, 236, 16, 17, 237, 19, 20, 35, 36, 141, 134, 135, 128, 129, 122, 123, 116, 117, 38, 39, 110, 111, 104, 105, 98, 99, 92, 93, 86, 87, 80, 81, 74, 75, 68, 69, 62, 63, 56, 57, 47, 48, 152, 42, 43, 146, 147, 37, 40, 41, 238, 29, 44, 45, 46, 142, 296, 297, 136, 293, 294, 130, 290, 291, 124, 287, 288, 118, 284, 285, 112, 281, 282, 157, 305, 306, 106, 278, 279, 100, 275, 276, 94, 272, 273, 88, 269, 270, 82, 266, 267, 76, 263, 264, 70, 260, 261, 64, 257, 258, 58, 254, 255, 49, 246, 247, 153, 302, 303, 148, 299, 300, 308, 309, 239, 240, 241, 310, 311, 298, 295, 292, 289, 286, 283, 307, 280, 277, 274, 271, 268, 265, 262, 259, 256, 253, 304, 301, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 341, 342, 343, 334, 232, 193, 194, 195, 196, 340, 344, 345, 346, 233, 234, 335, 336, 337, 338, 347, 348, 349, 350, 351, 352, 227, 228, 353, 354, 355, 229, 230, 213, 214, 216, 217, 356, 357, 358, 359, 221, 362, 360, 363, 361]
365 Ops.SOURCE dtypes.void [] "#include <metal_stdlib>\nusing namespace metal;\nkernel void r_16_32_4_4_20(device unsigned char* args_24, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n  float buf0[1];\n  float buf1[1];\n  threadgroup __attribute__((aligned(16))) float buf2[128];\n  unsigned long val0 = (*(((device unsigned long*)(args_24))+1));\n  unsigned long val1 = (*(((device unsigned long*)(args_24))+2));\n  int gidx0 = gid.x; /* 16 */\n  int lidx1 = lid.y; /* 4 */\n  int alu0 = ((gidx0<<2)+lidx1);\n  float val2 = (*(((device float*)(val1))+alu0));\n  int lidx0 = lid.x; /* 32 */\n  *(buf0+0) = 0.0f;\n  for (int Lidx0_0_0 = 0; Lidx0_0_0 < 4; Lidx0_0_0++) {\n    int alu2 = ((gidx0*10240)+(lidx1*2560)+((Lidx0_0_0*640)+lidx0));\n    float val3 = (*(((device float*)(val0))+(alu2+128)));\n    float val4 = (*(((device float*)(val0))+(alu2+160)));\n    float val5 = (*(((device float*)(val0))+(alu2+192)));\n    float val6 = (*(((device float*)(val0))+(alu2+224)));\n    float val7 = (*(((device float*)(val0))+(alu2+256)));\n    float val8 = (*(((device float*)(val0))+(alu2+288)));\n    float val9 = (*(((device float*)(val0))+(alu2+32)));\n    float val10 = (*(((device float*)(val0))+(alu2+320)));\n    float val11 = (*(((device float*)(val0))+(alu2+352)));\n    float val12 = (*(((device float*)(val0))+(alu2+384)));\n    float val13 = (*(((device float*)(val0))+(alu2+416)));\n    float val14 = (*(((device float*)(val0))+(alu2+448)));\n    float val15 = (*(((device float*)(val0))+(alu2+480)));\n    float val16 = (*(((device float*)(val0))+(alu2+512)));\n    float val17 = (*(((device float*)(val0))+(alu2+544)));\n    float val18 = (*(((device float*)(val0))+(alu2+576)));\n    float val19 = (*(((device float*)(val0))+(alu2+608)));\n    float val20 = (*(((device float*)(val0))+(alu2+64)));\n    float val21 = (*(((device float*)(val0))+(alu2+96)));\n    float val22 = (*(((device float*)(val0))+alu2));\n    float alu3 = (val22-val2);\n    float alu4 = (val3-val2);\n    float alu5 = (val4-val2);\n    float alu6 = (val5-val2);\n    float alu7 = (val6-val2);\n    float alu8 = (val7-val2);\n    float alu9 = (val8-val2);\n    float alu10 = (val9-val2);\n    float alu11 = (val10-val2);\n    float alu12 = (val11-val2);\n    float alu13 = (val12-val2);\n    float alu14 = (val13-val2);\n    float alu15 = (val14-val2);\n    float alu16 = (val15-val2);\n    float alu17 = (val16-val2);\n    float alu18 = (val17-val2);\n    float alu19 = (val18-val2);\n    float alu20 = (val19-val2);\n    float alu21 = (val20-val2);\n    float alu22 = (val21-val2);\n    *(buf0+0) = ((*(buf0+0))+fma(alu20,alu20,fma(alu19,alu19,fma(alu18,alu18,fma(alu17,alu17,fma(alu16,alu16,fma(alu15,alu15,fma(alu14,alu14,fma(alu13,alu13,fma(alu12,alu12,fma(alu11,alu11,fma(alu9,alu9,fma(alu8,alu8,fma(alu7,alu7,fma(alu6,alu6,fma(alu5,alu5,fma(alu4,alu4,fma(alu22,alu22,fma(alu21,alu21,fma(alu10,alu10,(alu3*alu3)))))))))))))))))))));\n  }\n  int alu25 = (lidx1<<5);\n  *(buf2+(alu25+lidx0)) = (*(buf0+0));\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  *(buf1+0) = 0.0f;\n  for (int Lidx102 = 0; Lidx102 < 32; Lidx102++) {\n    float val23 = (*(buf2+(alu25+Lidx102)));\n    *(buf1+0) = ((*(buf1+0))+val23);\n  }\n  unsigned long val24 = (*(((device unsigned long*)(args_24))+0));\n  if ((lidx0==0)) {\n    *(((device float*)(val24))+alu0) = (1/sqrt((((*(buf1+0))*0.0003906250058207661f)+9.999999747378752e-06f)));\n  }\n}"
366 Ops.PROGRAM dtypes.void [361, 364, 365] ProgramInfo(global_size=(16, 1, 1), local_size=(32, 4, 1), params=(%1, %34, %52), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="METAL", renderer="METAL", arch="Apple9"))
367 Ops.SINK dtypes.void [223, 366]

