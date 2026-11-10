# tolk D473 golden=9015e6b4df3d3c65cf0ab37c05cbb901
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f16 [0] ParamArg(slot=7, dtype=dtypes.f16, device="CPU")
2 Ops.CONST dtypes.weakint [] 1
3 Ops.PARAM dtypes.f32 [2] ParamArg(slot=0, dtype=dtypes.f32)
4 Ops.CONST dtypes.weakint [] 0
5 Ops.INDEX dtypes.f32 [3, 4]
6 Ops.CONST dtypes.weakint [] 64
7 Ops.PARAM dtypes.f32 [6] ParamArg(slot=1, dtype=dtypes.f32)
8 Ops.RANGE dtypes.weakint [6] (AxisType.WEAK, 0)
9 Ops.INDEX dtypes.f32 [7, 8]
10 Ops.MUL dtypes.f32 [9, 9]
11 Ops.REDUCE dtypes.f32 [10, 8] (Ops.ADD, 0)
12 Ops.CONST dtypes.weakfloat [] 0.015625
13 Ops.MUL dtypes.f32 [11, 12]
14 Ops.CONST dtypes.weakfloat [] 1e-05
15 Ops.ADD dtypes.f32 [13, 14]
16 Ops.SQRT dtypes.f32 [15]
17 Ops.STORE dtypes.void [5, 16]
18 Ops.SINK dtypes.void [17] KernelInfo()
19 Ops.ALLOC dtypes.f32 [2] ParamArg(slot=79, dtype=dtypes.f32, device="CPU")
20 Ops.PARAM dtypes.f32 [6] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
21 Ops.CALL dtypes.void [18, 19, 20] CallInfo()
22 Ops.CONST dtypes.weakint [] 16
23 Ops.PARAM dtypes.f32 [22] ParamArg(slot=0, dtype=dtypes.f32)
24 Ops.STACK dtypes.void []
25 Ops.PARAM dtypes.weakint [24] ParamArg(slot=5, dtype=dtypes.weakint, vmin_vmax=(0, 15), addrspace=AddrSpace.ALU)
26 Ops.ADD dtypes.weakint [25, 2]
27 Ops.RANGE dtypes.weakint [26] (AxisType.WEAK, 1)
28 Ops.INDEX dtypes.f32 [23, 27]
29 Ops.CAST dtypes.i32 [25] dtypes.i32
30 Ops.CONST dtypes.weakint [] 2
31 Ops.MUL dtypes.weakint [25, 30]
32 Ops.ADD dtypes.weakint [31, 2]
33 Ops.CONST dtypes.weakint [] -2
34 Ops.MUL dtypes.weakint [25, 33]
35 Ops.CONST dtypes.weakint [] -1
36 Ops.ADD dtypes.weakint [34, 35]
37 Ops.DIV dtypes.weakint [25, 36]
38 Ops.MUL dtypes.weakint [37, 35]
39 Ops.MAX dtypes.weakint [38, 2]
40 Ops.MUL dtypes.weakint [32, 39]
41 Ops.ADD dtypes.weakint [40, 2]
42 Ops.RANGE dtypes.weakint [26] (AxisType.WEAK, 0)
43 Ops.MUL dtypes.weakint [41, 42]
44 Ops.ADD dtypes.weakint [43, 27]
45 Ops.MUL dtypes.weakint [26, 41]
46 Ops.MOD dtypes.weakint [44, 45]
47 Ops.MOD dtypes.weakint [46, 32]
48 Ops.CMPLT dtypes.bool [47, 25]
49 Ops.CONST dtypes.bool [] False
50 Ops.CMPEQ dtypes.bool [48, 49]
51 Ops.CMPLT dtypes.bool [47, 32]
52 Ops.AND dtypes.bool [50, 51]
53 Ops.CAST dtypes.i32 [2] dtypes.i32
54 Ops.WHERE dtypes.i32 [52, 53, 4]
55 Ops.REDUCE dtypes.i32 [54, 42] (Ops.ADD, 0)
56 Ops.ADD dtypes.i32 [55, 35]
57 Ops.CMPLT dtypes.bool [29, 56]
58 Ops.CMPEQ dtypes.bool [57, 49]
59 Ops.CONST dtypes.weakint [] -8
60 Ops.ADD dtypes.i32 [29, 59]
61 Ops.CMPLT dtypes.bool [60, 56]
62 Ops.AND dtypes.bool [58, 61]
63 Ops.CONST dtypes.weakfloat [] 0.0
64 Ops.CONST dtypes.weakfloat [] -inf
65 Ops.WHERE dtypes.weakfloat [62, 63, 64]
66 Ops.CAST dtypes.f32 [65] dtypes.f32
67 Ops.STORE dtypes.void [28, 66]
68 Ops.END dtypes.void [67, 27]
69 Ops.SINK dtypes.void [68] KernelInfo()
70 Ops.ALLOC dtypes.f32 [22] ParamArg(slot=81, dtype=dtypes.f32, device="CPU")
71 Ops.CALL dtypes.void [69, 70] CallInfo()
72 Ops.PARAM dtypes.f32 [6] ParamArg(slot=0, dtype=dtypes.f32)
73 Ops.CONST dtypes.weakint [] 4
74 Ops.RANGE dtypes.weakint [73] (AxisType.WEAK, 1)
75 Ops.MUL dtypes.weakint [74, 22]
76 Ops.RANGE dtypes.weakint [22] (AxisType.WEAK, 2)
77 Ops.ADD dtypes.weakint [75, 76]
78 Ops.INDEX dtypes.f32 [72, 77]
79 Ops.PARAM dtypes.f32 [2] ParamArg(slot=2, dtype=dtypes.f32)
80 Ops.INDEX dtypes.f32 [79, 4]
81 Ops.RECIP dtypes.f32 [80]
82 Ops.MUL dtypes.f32 [9, 81]
83 Ops.PARAM dtypes.f32 [6] ParamArg(slot=3, dtype=dtypes.f32)
84 Ops.INDEX dtypes.f32 [83, 8]
85 Ops.MUL dtypes.f32 [82, 84]
86 Ops.CONST dtypes.weakint [] 4096
87 Ops.PARAM dtypes.f32 [86] ParamArg(slot=4, dtype=dtypes.f32)
88 Ops.MUL dtypes.weakint [77, 6]
89 Ops.ADD dtypes.weakint [88, 8]
90 Ops.INDEX dtypes.f32 [87, 89]
91 Ops.MUL dtypes.f32 [85, 90]
92 Ops.REDUCE dtypes.f32 [91, 8] (Ops.ADD, 0)
93 Ops.PARAM dtypes.f32 [6] ParamArg(slot=5, dtype=dtypes.f32)
94 Ops.INDEX dtypes.f32 [93, 77]
95 Ops.ADD dtypes.f32 [92, 94]
96 Ops.STORE dtypes.void [78, 95]
97 Ops.END dtypes.void [96, 74, 76]
98 Ops.SINK dtypes.void [97] KernelInfo()
99 Ops.ALLOC dtypes.f32 [6] ParamArg(slot=78, dtype=dtypes.f32, device="CPU")
100 Ops.PARAM dtypes.f32 [6] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
101 Ops.PARAM dtypes.f32 [86] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
102 Ops.PARAM dtypes.f32 [6] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
103 Ops.CALL dtypes.void [98, 99, 20, 19, 100, 101, 102] CallInfo()
104 Ops.CONST dtypes.weakint [] 32
105 Ops.PARAM dtypes.f32 [104] ParamArg(slot=0, dtype=dtypes.f32)
106 Ops.RANGE dtypes.weakint [30] (AxisType.WEAK, 1)
107 Ops.MUL dtypes.weakint [106, 22]
108 Ops.ADD dtypes.weakint [107, 76]
109 Ops.INDEX dtypes.f32 [105, 108]
110 Ops.CONST dtypes.weakint [] 2048
111 Ops.PARAM dtypes.f32 [110] ParamArg(slot=4, dtype=dtypes.f32)
112 Ops.MUL dtypes.weakint [108, 6]
113 Ops.ADD dtypes.weakint [112, 8]
114 Ops.INDEX dtypes.f32 [111, 113]
115 Ops.MUL dtypes.f32 [85, 114]
116 Ops.REDUCE dtypes.f32 [115, 8] (Ops.ADD, 0)
117 Ops.PARAM dtypes.f32 [104] ParamArg(slot=5, dtype=dtypes.f32)
118 Ops.INDEX dtypes.f32 [117, 108]
119 Ops.ADD dtypes.f32 [116, 118]
120 Ops.STORE dtypes.void [109, 119]
121 Ops.END dtypes.void [120, 106, 76]
122 Ops.SINK dtypes.void [121] KernelInfo()
123 Ops.ALLOC dtypes.f32 [104] ParamArg(slot=80, dtype=dtypes.f32, device="CPU")
124 Ops.PARAM dtypes.f32 [110] ParamArg(slot=7, dtype=dtypes.f32, device="CPU")
125 Ops.PARAM dtypes.f32 [104] ParamArg(slot=8, dtype=dtypes.f32, device="CPU")
126 Ops.CALL dtypes.void [122, 123, 20, 19, 100, 124, 125] CallInfo()
127 Ops.PARAM dtypes.f16 [0] ParamArg(slot=0, dtype=dtypes.f16)
128 Ops.MUL dtypes.weakint [25, 22]
129 Ops.RANGE dtypes.weakint [22] (AxisType.WEAK, 3)
130 Ops.ADD dtypes.weakint [128, 129]
131 Ops.RANGE dtypes.weakint [30] (AxisType.WEAK, 2)
132 Ops.CONST dtypes.weakint [] 256
133 Ops.MUL dtypes.weakint [131, 132]
134 Ops.ADD dtypes.weakint [130, 133]
135 Ops.CONST dtypes.weakint [] 512
136 Ops.MUL dtypes.weakint [106, 135]
137 Ops.ADD dtypes.weakint [134, 136]
138 Ops.INDEX dtypes.f16 [127, 137]
139 Ops.CMPLT dtypes.bool [106, 2]
140 Ops.CONST dtypes.weakint [] 8
141 Ops.CMPLT dtypes.bool [129, 140]
142 Ops.PARAM dtypes.f32 [104] ParamArg(slot=1, dtype=dtypes.f32)
143 Ops.MUL dtypes.weakint [131, 22]
144 Ops.MOD dtypes.weakint [129, 140]
145 Ops.ADD dtypes.weakint [143, 144]
146 Ops.INDEX dtypes.f32 [142, 145]
147 Ops.PARAM dtypes.f32 [132] ParamArg(slot=2, dtype=dtypes.f32)
148 Ops.ADD dtypes.weakint [128, 144]
149 Ops.INDEX dtypes.f32 [147, 148]
150 Ops.MUL dtypes.f32 [146, 149]
151 Ops.ADD dtypes.weakint [145, 140]
152 Ops.INDEX dtypes.f32 [142, 151]
153 Ops.ADD dtypes.weakint [148, 140]
154 Ops.INDEX dtypes.f32 [147, 153]
155 Ops.MUL dtypes.f32 [152, 154]
156 Ops.CONST dtypes.weakfloat [] -1.0
157 Ops.MUL dtypes.f32 [155, 156]
158 Ops.ADD dtypes.f32 [150, 157]
159 Ops.MUL dtypes.f32 [152, 149]
160 Ops.MUL dtypes.f32 [146, 154]
161 Ops.ADD dtypes.f32 [159, 160]
162 Ops.WHERE dtypes.f32 [141, 158, 161]
163 Ops.PARAM dtypes.f32 [2] ParamArg(slot=4, dtype=dtypes.f32)
164 Ops.INDEX dtypes.f32 [163, 4]
165 Ops.RECIP dtypes.f32 [164]
166 Ops.MUL dtypes.f32 [84, 165]
167 Ops.INDEX dtypes.f32 [93, 8]
168 Ops.MUL dtypes.f32 [166, 167]
169 Ops.PARAM dtypes.f32 [110] ParamArg(slot=6, dtype=dtypes.f32)
170 Ops.ADD dtypes.weakint [143, 129]
171 Ops.MUL dtypes.weakint [170, 6]
172 Ops.ADD dtypes.weakint [171, 8]
173 Ops.INDEX dtypes.f32 [169, 172]
174 Ops.MUL dtypes.f32 [168, 173]
175 Ops.REDUCE dtypes.f32 [174, 8] (Ops.ADD, 0)
176 Ops.PARAM dtypes.f32 [104] ParamArg(slot=7, dtype=dtypes.f32)
177 Ops.INDEX dtypes.f32 [176, 170]
178 Ops.ADD dtypes.f32 [175, 177]
179 Ops.WHERE dtypes.f32 [139, 162, 178]
180 Ops.CAST dtypes.f16 [179] dtypes.f16
181 Ops.STORE dtypes.void [138, 180]
182 Ops.END dtypes.void [181, 106, 131, 129]
183 Ops.SINK dtypes.void [182] KernelInfo()
184 Ops.PARAM dtypes.f16 [0] ParamArg(slot=6, dtype=dtypes.f16, device="CPU")
185 Ops.PARAM dtypes.f32 [132] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
186 Ops.PARAM dtypes.f32 [110] ParamArg(slot=9, dtype=dtypes.f32, device="CPU")
187 Ops.PARAM dtypes.f32 [104] ParamArg(slot=10, dtype=dtypes.f32, device="CPU")
188 Ops.CALL dtypes.void [183, 184, 123, 185, 20, 19, 100, 186, 187] CallInfo()
189 Ops.CONST dtypes.weakint [] 68
190 Ops.PARAM dtypes.f32 [189] ParamArg(slot=0, dtype=dtypes.f32)
191 Ops.CONST dtypes.weakint [] 17
192 Ops.MUL dtypes.weakint [74, 191]
193 Ops.ADD dtypes.weakint [25, 30]
194 Ops.RANGE dtypes.weakint [193] (AxisType.WEAK, 2)
195 Ops.ADD dtypes.weakint [192, 194]
196 Ops.INDEX dtypes.f32 [190, 195]
197 Ops.RANGE dtypes.weakint [22] (AxisType.WEAK, 0)
198 Ops.CMPLT dtypes.bool [197, 140]
199 Ops.MOD dtypes.weakint [197, 140]
200 Ops.ADD dtypes.weakint [75, 199]
201 Ops.INDEX dtypes.f32 [7, 200]
202 Ops.ADD dtypes.weakint [128, 199]
203 Ops.INDEX dtypes.f32 [147, 202]
204 Ops.MUL dtypes.f32 [201, 203]
205 Ops.ADD dtypes.weakint [200, 140]
206 Ops.INDEX dtypes.f32 [7, 205]
207 Ops.ADD dtypes.weakint [202, 140]
208 Ops.INDEX dtypes.f32 [147, 207]
209 Ops.MUL dtypes.f32 [206, 208]
210 Ops.MUL dtypes.f32 [209, 156]
211 Ops.ADD dtypes.f32 [204, 210]
212 Ops.MUL dtypes.f32 [206, 203]
213 Ops.MUL dtypes.f32 [201, 208]
214 Ops.ADD dtypes.f32 [212, 213]
215 Ops.WHERE dtypes.f32 [198, 211, 214]
216 Ops.ADD dtypes.weakint [128, 104]
217 Ops.MUL dtypes.weakint [216, 74]
218 Ops.ADD dtypes.weakint [217, 194]
219 Ops.MOD dtypes.weakint [218, 193]
220 Ops.CMPLT dtypes.bool [219, 26]
221 Ops.PARAM dtypes.f16 [0] ParamArg(slot=3, dtype=dtypes.f16)
222 Ops.MOD dtypes.weakint [219, 26]
223 Ops.MUL dtypes.weakint [222, 22]
224 Ops.DIV dtypes.weakint [218, 193]
225 Ops.ADD dtypes.weakint [224, 197]
226 Ops.MOD dtypes.weakint [225, 22]
227 Ops.ADD dtypes.weakint [223, 226]
228 Ops.DIV dtypes.weakint [219, 193]
229 Ops.DIV dtypes.weakint [225, 22]
230 Ops.MOD dtypes.weakint [229, 73]
231 Ops.ADD dtypes.weakint [228, 230]
232 Ops.DIV dtypes.weakint [231, 30]
233 Ops.DIV dtypes.weakint [219, 26]
234 Ops.ADD dtypes.weakint [232, 233]
235 Ops.MOD dtypes.weakint [234, 30]
236 Ops.MUL dtypes.weakint [235, 132]
237 Ops.ADD dtypes.weakint [227, 236]
238 Ops.CONST dtypes.bool [] Invalid
239 Ops.WHERE dtypes.weakint [220, 237, 238]
240 Ops.INDEX dtypes.f16 [221, 239]
241 Ops.WHERE dtypes.f16 [220, 240, 63]
242 Ops.CMPEQ dtypes.bool [220, 49]
243 Ops.CMPLT dtypes.bool [219, 193]
244 Ops.AND dtypes.bool [242, 243]
245 Ops.CAST dtypes.f16 [63] dtypes.f16
246 Ops.WHERE dtypes.f16 [244, 245, 63]
247 Ops.ADD dtypes.f16 [241, 246]
248 Ops.CAST dtypes.f32 [247] dtypes.f32
249 Ops.MUL dtypes.f32 [215, 248]
250 Ops.REDUCE dtypes.f32 [249, 197] (Ops.ADD, 0)
251 Ops.CONST dtypes.weakfloat [] 0.25
252 Ops.MUL dtypes.f32 [250, 251]
253 Ops.CMPLT dtypes.bool [194, 26]
254 Ops.PARAM dtypes.f32 [22] ParamArg(slot=4, dtype=dtypes.f32)
255 Ops.MOD dtypes.weakint [194, 26]
256 Ops.WHERE dtypes.weakint [253, 255, 238]
257 Ops.INDEX dtypes.f32 [254, 256]
258 Ops.WHERE dtypes.f32 [253, 257, 63]
259 Ops.CMPEQ dtypes.bool [253, 49]
260 Ops.CMPLT dtypes.bool [194, 193]
261 Ops.AND dtypes.bool [259, 260]
262 Ops.PARAM dtypes.f32 [73] ParamArg(slot=5, dtype=dtypes.f32)
263 Ops.MUL dtypes.weakint [25, 35]
264 Ops.ADD dtypes.weakint [263, 194]
265 Ops.ADD dtypes.weakint [264, 74]
266 Ops.CONST dtypes.weakint [] 3
267 Ops.ADD dtypes.weakint [265, 266]
268 Ops.MOD dtypes.weakint [267, 73]
269 Ops.WHERE dtypes.weakint [261, 268, 238]
270 Ops.INDEX dtypes.f32 [262, 269]
271 Ops.WHERE dtypes.f32 [261, 270, 63]
272 Ops.ADD dtypes.f32 [258, 271]
273 Ops.ADD dtypes.f32 [252, 272]
274 Ops.STORE dtypes.void [196, 273]
275 Ops.END dtypes.void [274, 74, 194]
276 Ops.SINK dtypes.void [275] KernelInfo()
277 Ops.ALLOC dtypes.f32 [189] ParamArg(slot=77, dtype=dtypes.f32, device="CPU")
278 Ops.PARAM dtypes.f32 [73] ParamArg(slot=11, dtype=dtypes.f32, device="CPU")
279 Ops.CALL dtypes.void [276, 277, 99, 185, 184, 70, 278] CallInfo()
280 Ops.PARAM dtypes.f32 [73] ParamArg(slot=0, dtype=dtypes.f32)
281 Ops.INDEX dtypes.f32 [280, 74]
282 Ops.PARAM dtypes.f32 [189] ParamArg(slot=1, dtype=dtypes.f32)
283 Ops.RANGE dtypes.weakint [193] (AxisType.WEAK, 0)
284 Ops.ADD dtypes.weakint [192, 283]
285 Ops.INDEX dtypes.f32 [282, 284]
286 Ops.REDUCE dtypes.f32 [285, 283] (Ops.MAX, 0)
287 Ops.STORE dtypes.void [281, 286]
288 Ops.END dtypes.void [287, 74]
289 Ops.SINK dtypes.void [288] KernelInfo()
290 Ops.ALLOC dtypes.f32 [73] ParamArg(slot=82, dtype=dtypes.f32, device="CPU")
291 Ops.CALL dtypes.void [289, 290, 277] CallInfo()
292 Ops.PARAM dtypes.f32 [73] ParamArg(slot=2, dtype=dtypes.f32)
293 Ops.INDEX dtypes.f32 [292, 74]
294 Ops.MUL dtypes.f32 [293, 156]
295 Ops.ADD dtypes.f32 [285, 294]
296 Ops.CONST dtypes.weakfloat [] 1.4426950408889634
297 Ops.MUL dtypes.f32 [295, 296]
298 Ops.EXP2 dtypes.f32 [297]
299 Ops.REDUCE dtypes.f32 [298, 283] (Ops.ADD, 0)
300 Ops.STORE dtypes.void [281, 299]
301 Ops.END dtypes.void [300, 74]
302 Ops.SINK dtypes.void [301] KernelInfo()
303 Ops.ALLOC dtypes.f32 [73] ParamArg(slot=83, dtype=dtypes.f32, device="CPU")
304 Ops.CALL dtypes.void [302, 303, 277, 290] CallInfo()
305 Ops.RANGE dtypes.weakint [73] (AxisType.WEAK, 0)
306 Ops.MUL dtypes.weakint [305, 191]
307 Ops.RANGE dtypes.weakint [193] (AxisType.WEAK, 1)
308 Ops.ADD dtypes.weakint [306, 307]
309 Ops.INDEX dtypes.f32 [190, 308]
310 Ops.INDEX dtypes.f32 [282, 308]
311 Ops.INDEX dtypes.f32 [292, 305]
312 Ops.MUL dtypes.f32 [311, 156]
313 Ops.ADD dtypes.f32 [310, 312]
314 Ops.MUL dtypes.f32 [313, 296]
315 Ops.EXP2 dtypes.f32 [314]
316 Ops.PARAM dtypes.f32 [73] ParamArg(slot=3, dtype=dtypes.f32)
317 Ops.INDEX dtypes.f32 [316, 305]
318 Ops.RECIP dtypes.f32 [317]
319 Ops.MUL dtypes.f32 [315, 318]
320 Ops.STORE dtypes.void [309, 319]
321 Ops.END dtypes.void [320, 305, 307]
322 Ops.SINK dtypes.void [321] KernelInfo()
323 Ops.ALLOC dtypes.f32 [189] ParamArg(slot=76, dtype=dtypes.f32, device="CPU")
324 Ops.CALL dtypes.void [322, 323, 277, 290, 303] CallInfo()
325 Ops.CMPLT dtypes.bool [283, 26]
326 Ops.PARAM dtypes.f16 [0] ParamArg(slot=2, dtype=dtypes.f16)
327 Ops.MOD dtypes.weakint [283, 26]
328 Ops.MUL dtypes.weakint [327, 22]
329 Ops.ADD dtypes.weakint [328, 76]
330 Ops.DIV dtypes.weakint [74, 30]
331 Ops.DIV dtypes.weakint [283, 26]
332 Ops.ADD dtypes.weakint [330, 331]
333 Ops.MOD dtypes.weakint [332, 30]
334 Ops.MUL dtypes.weakint [333, 132]
335 Ops.ADD dtypes.weakint [329, 334]
336 Ops.ADD dtypes.weakint [335, 135]
337 Ops.WHERE dtypes.weakint [325, 336, 238]
338 Ops.INDEX dtypes.f16 [326, 337]
339 Ops.WHERE dtypes.f16 [325, 338, 63]
340 Ops.CMPEQ dtypes.bool [325, 49]
341 Ops.CMPLT dtypes.bool [283, 193]
342 Ops.AND dtypes.bool [340, 341]
343 Ops.WHERE dtypes.f16 [342, 245, 63]
344 Ops.ADD dtypes.f16 [339, 343]
345 Ops.CAST dtypes.f32 [344] dtypes.f32
346 Ops.MUL dtypes.f32 [285, 345]
347 Ops.REDUCE dtypes.f32 [346, 283] (Ops.ADD, 0)
348 Ops.STORE dtypes.void [78, 347]
349 Ops.END dtypes.void [348, 74, 76]
350 Ops.SINK dtypes.void [349] KernelInfo()
351 Ops.ALLOC dtypes.f32 [6] ParamArg(slot=75, dtype=dtypes.f32, device="CPU")
352 Ops.CALL dtypes.void [350, 351, 323, 184] CallInfo()
353 Ops.RANGE dtypes.weakint [6] (AxisType.WEAK, 1)
354 Ops.INDEX dtypes.f32 [72, 353]
355 Ops.INDEX dtypes.f32 [7, 353]
356 Ops.PARAM dtypes.f32 [6] ParamArg(slot=2, dtype=dtypes.f32)
357 Ops.INDEX dtypes.f32 [356, 8]
358 Ops.PARAM dtypes.f32 [86] ParamArg(slot=3, dtype=dtypes.f32)
359 Ops.MUL dtypes.weakint [353, 6]
360 Ops.ADD dtypes.weakint [359, 8]
361 Ops.INDEX dtypes.f32 [358, 360]
362 Ops.MUL dtypes.f32 [357, 361]
363 Ops.REDUCE dtypes.f32 [362, 8] (Ops.ADD, 0)
364 Ops.PARAM dtypes.f32 [6] ParamArg(slot=4, dtype=dtypes.f32)
365 Ops.INDEX dtypes.f32 [364, 353]
366 Ops.ADD dtypes.f32 [363, 365]
367 Ops.ADD dtypes.f32 [355, 366]
368 Ops.STORE dtypes.void [354, 367]
369 Ops.END dtypes.void [368, 353]
370 Ops.SINK dtypes.void [369] KernelInfo()
371 Ops.ALLOC dtypes.f32 [6] ParamArg(slot=74, dtype=dtypes.f32, device="CPU")
372 Ops.PARAM dtypes.f32 [86] ParamArg(slot=12, dtype=dtypes.f32, device="CPU")
373 Ops.PARAM dtypes.f32 [6] ParamArg(slot=13, dtype=dtypes.f32, device="CPU")
374 Ops.CALL dtypes.void [370, 371, 20, 351, 372, 373] CallInfo()
375 Ops.ALLOC dtypes.f32 [2] ParamArg(slot=85, dtype=dtypes.f32, device="CPU")
376 Ops.CALL dtypes.void [18, 375, 371] CallInfo()
377 Ops.PARAM dtypes.f32 [132] ParamArg(slot=4, dtype=dtypes.f32)
378 Ops.MUL dtypes.weakint [74, 6]
379 Ops.ADD dtypes.weakint [378, 8]
380 Ops.INDEX dtypes.f32 [377, 379]
381 Ops.MUL dtypes.f32 [85, 380]
382 Ops.REDUCE dtypes.f32 [381, 8] (Ops.ADD, 0)
383 Ops.INDEX dtypes.f32 [262, 74]
384 Ops.ADD dtypes.f32 [382, 383]
385 Ops.STORE dtypes.void [281, 384]
386 Ops.END dtypes.void [385, 74]
387 Ops.SINK dtypes.void [386] KernelInfo()
388 Ops.ALLOC dtypes.f32 [73] ParamArg(slot=87, dtype=dtypes.f32, device="CPU")
389 Ops.PARAM dtypes.f32 [6] ParamArg(slot=14, dtype=dtypes.f32, device="CPU")
390 Ops.PARAM dtypes.f32 [132] ParamArg(slot=15, dtype=dtypes.f32, device="CPU")
391 Ops.PARAM dtypes.f32 [73] ParamArg(slot=16, dtype=dtypes.f32, device="CPU")
392 Ops.CALL dtypes.void [387, 388, 371, 375, 389, 390, 391] CallInfo()
393 Ops.PARAM dtypes.i32 [73] ParamArg(slot=0, dtype=dtypes.i32)
394 Ops.INDEX dtypes.i32 [393, 74]
395 Ops.PARAM dtypes.f32 [73] ParamArg(slot=1, dtype=dtypes.f32)
396 Ops.INDEX dtypes.f32 [395, 305]
397 Ops.INDEX dtypes.f32 [395, 74]
398 Ops.CMPLT dtypes.bool [396, 397]
399 Ops.CMPEQ dtypes.bool [397, 396]
400 Ops.CAST dtypes.i32 [74] dtypes.i32
401 Ops.CAST dtypes.i32 [305] dtypes.i32
402 Ops.CMPLT dtypes.bool [400, 401]
403 Ops.AND dtypes.bool [399, 402]
404 Ops.OR dtypes.bool [398, 403]
405 Ops.CAST dtypes.i32 [404] dtypes.i32
406 Ops.REDUCE dtypes.i32 [405, 305] (Ops.ADD, 0)
407 Ops.STORE dtypes.void [394, 406]
408 Ops.END dtypes.void [407, 74]
409 Ops.SINK dtypes.void [408] KernelInfo()
410 Ops.ALLOC dtypes.i32 [73] ParamArg(slot=86, dtype=dtypes.i32, device="CPU")
411 Ops.CALL dtypes.void [409, 410, 388] CallInfo()
412 Ops.MUL dtypes.weakint [131, 104]
413 Ops.RANGE dtypes.weakint [104] (AxisType.WEAK, 3)
414 Ops.ADD dtypes.weakint [412, 413]
415 Ops.INDEX dtypes.f32 [72, 414]
416 Ops.MUL dtypes.f32 [355, 81]
417 Ops.INDEX dtypes.f32 [83, 353]
418 Ops.MUL dtypes.f32 [416, 417]
419 Ops.PARAM dtypes.i32 [73] ParamArg(slot=4, dtype=dtypes.i32)
420 Ops.INDEX dtypes.i32 [419, 4]
421 Ops.CAST dtypes.i32 [131] dtypes.i32
422 Ops.ADD dtypes.i32 [421, 30]
423 Ops.CMPEQ dtypes.bool [420, 422]
424 Ops.INDEX dtypes.i32 [419, 2]
425 Ops.CMPEQ dtypes.bool [424, 422]
426 Ops.OR dtypes.bool [423, 425]
427 Ops.INDEX dtypes.i32 [419, 30]
428 Ops.CMPEQ dtypes.bool [427, 422]
429 Ops.OR dtypes.bool [426, 428]
430 Ops.INDEX dtypes.i32 [419, 266]
431 Ops.CMPEQ dtypes.bool [430, 422]
432 Ops.OR dtypes.bool [429, 431]
433 Ops.CONST dtypes.weakfloat [] 3.0
434 Ops.CONST dtypes.weakfloat [] 2.0
435 Ops.CONST dtypes.weakfloat [] 1.0
436 Ops.CAST dtypes.f32 [435] dtypes.f32
437 Ops.CAST dtypes.f32 [63] dtypes.f32
438 Ops.WHERE dtypes.f32 [425, 436, 437]
439 Ops.WHERE dtypes.f32 [428, 434, 438]
440 Ops.WHERE dtypes.f32 [431, 433, 439]
441 Ops.WHERE dtypes.f32 [432, 440, 63]
442 Ops.CAST dtypes.i32 [441] dtypes.i32
443 Ops.CMPEQ dtypes.bool [442, 401]
444 Ops.CONST dtypes.weakint [] 8192
445 Ops.PARAM dtypes.f32 [444] ParamArg(slot=5, dtype=dtypes.f32)
446 Ops.MUL dtypes.weakint [413, 6]
447 Ops.ADD dtypes.weakint [446, 353]
448 Ops.MUL dtypes.weakint [305, 110]
449 Ops.ADD dtypes.weakint [447, 448]
450 Ops.INDEX dtypes.f32 [445, 449]
451 Ops.WHERE dtypes.f32 [443, 450, 63]
452 Ops.REDUCE dtypes.f32 [451, 305] (Ops.ADD, 0)
453 Ops.MUL dtypes.f32 [418, 452]
454 Ops.REDUCE dtypes.f32 [453, 353] (Ops.ADD, 0)
455 Ops.STORE dtypes.void [415, 454]
456 Ops.END dtypes.void [455, 131, 413]
457 Ops.SINK dtypes.void [456] KernelInfo()
458 Ops.ALLOC dtypes.f32 [6] ParamArg(slot=71, dtype=dtypes.f32, device="CPU")
459 Ops.PARAM dtypes.f32 [444] ParamArg(slot=17, dtype=dtypes.f32, device="CPU")
460 Ops.CALL dtypes.void [457, 458, 371, 375, 389, 410, 459] CallInfo()
461 Ops.ALLOC dtypes.f32 [6] ParamArg(slot=73, dtype=dtypes.f32, device="CPU")
462 Ops.PARAM dtypes.f32 [444] ParamArg(slot=19, dtype=dtypes.f32, device="CPU")
463 Ops.CALL dtypes.void [457, 461, 371, 375, 389, 410, 462] CallInfo()
464 Ops.PARAM dtypes.f32 [30] ParamArg(slot=0, dtype=dtypes.f32)
465 Ops.INDEX dtypes.f32 [464, 106]
466 Ops.PARAM dtypes.i32 [73] ParamArg(slot=1, dtype=dtypes.i32)
467 Ops.INDEX dtypes.i32 [466, 4]
468 Ops.CAST dtypes.i32 [106] dtypes.i32
469 Ops.ADD dtypes.i32 [468, 30]
470 Ops.CMPEQ dtypes.bool [467, 469]
471 Ops.INDEX dtypes.i32 [466, 2]
472 Ops.CMPEQ dtypes.bool [471, 469]
473 Ops.OR dtypes.bool [470, 472]
474 Ops.INDEX dtypes.i32 [466, 30]
475 Ops.CMPEQ dtypes.bool [474, 469]
476 Ops.OR dtypes.bool [473, 475]
477 Ops.INDEX dtypes.i32 [466, 266]
478 Ops.CMPEQ dtypes.bool [477, 469]
479 Ops.OR dtypes.bool [476, 478]
480 Ops.WHERE dtypes.f32 [472, 436, 437]
481 Ops.WHERE dtypes.f32 [475, 434, 480]
482 Ops.WHERE dtypes.f32 [478, 433, 481]
483 Ops.WHERE dtypes.f32 [479, 482, 63]
484 Ops.CAST dtypes.i32 [483] dtypes.i32
485 Ops.CMPEQ dtypes.bool [484, 401]
486 Ops.WHERE dtypes.f32 [485, 311, 63]
487 Ops.REDUCE dtypes.f32 [486, 305] (Ops.ADD, 0)
488 Ops.STORE dtypes.void [465, 487]
489 Ops.END dtypes.void [488, 106]
490 Ops.SINK dtypes.void [489] KernelInfo()
491 Ops.ALLOC dtypes.f32 [30] ParamArg(slot=88, dtype=dtypes.f32, device="CPU")
492 Ops.CALL dtypes.void [490, 491, 410, 388] CallInfo()
493 Ops.MUL dtypes.weakint [106, 104]
494 Ops.RANGE dtypes.weakint [104] (AxisType.WEAK, 2)
495 Ops.ADD dtypes.weakint [493, 494]
496 Ops.INDEX dtypes.f32 [72, 495]
497 Ops.CONST dtypes.weakfloat [] 7.0
498 Ops.INDEX dtypes.f32 [7, 495]
499 Ops.PARAM dtypes.i32 [73] ParamArg(slot=2, dtype=dtypes.i32)
500 Ops.INDEX dtypes.i32 [499, 4]
501 Ops.CMPEQ dtypes.bool [500, 469]
502 Ops.INDEX dtypes.i32 [499, 2]
503 Ops.CMPEQ dtypes.bool [502, 469]
504 Ops.OR dtypes.bool [501, 503]
505 Ops.INDEX dtypes.i32 [499, 30]
506 Ops.CMPEQ dtypes.bool [505, 469]
507 Ops.OR dtypes.bool [504, 506]
508 Ops.INDEX dtypes.i32 [499, 266]
509 Ops.CMPEQ dtypes.bool [508, 469]
510 Ops.OR dtypes.bool [507, 509]
511 Ops.WHERE dtypes.f32 [503, 436, 437]
512 Ops.WHERE dtypes.f32 [506, 434, 511]
513 Ops.WHERE dtypes.f32 [509, 433, 512]
514 Ops.WHERE dtypes.f32 [510, 513, 63]
515 Ops.CAST dtypes.i32 [514] dtypes.i32
516 Ops.CMPEQ dtypes.bool [515, 401]
517 Ops.CONST dtypes.weakint [] 128
518 Ops.PARAM dtypes.f32 [517] ParamArg(slot=3, dtype=dtypes.f32)
519 Ops.MUL dtypes.weakint [305, 104]
520 Ops.ADD dtypes.weakint [519, 494]
521 Ops.INDEX dtypes.f32 [518, 520]
522 Ops.WHERE dtypes.f32 [516, 521, 63]
523 Ops.REDUCE dtypes.f32 [522, 305] (Ops.ADD, 0)
524 Ops.ADD dtypes.f32 [498, 523]
525 Ops.CMPLT dtypes.bool [497, 524]
526 Ops.WHERE dtypes.f32 [525, 497, 524]
527 Ops.STORE dtypes.void [496, 526]
528 Ops.END dtypes.void [527, 106, 494]
529 Ops.SINK dtypes.void [528] KernelInfo()
530 Ops.ALLOC dtypes.f32 [6] ParamArg(slot=84, dtype=dtypes.f32, device="CPU")
531 Ops.PARAM dtypes.f32 [517] ParamArg(slot=18, dtype=dtypes.f32, device="CPU")
532 Ops.CALL dtypes.void [529, 530, 458, 410, 531] CallInfo()
533 Ops.PARAM dtypes.f32 [30] ParamArg(slot=1, dtype=dtypes.f32)
534 Ops.RANGE dtypes.weakint [30] (AxisType.WEAK, 0)
535 Ops.INDEX dtypes.f32 [533, 534]
536 Ops.REDUCE dtypes.f32 [535, 534] (Ops.MAX, 0)
537 Ops.STORE dtypes.void [5, 536]
538 Ops.SINK dtypes.void [537] KernelInfo()
539 Ops.ALLOC dtypes.f32 [2] ParamArg(slot=89, dtype=dtypes.f32, device="CPU")
540 Ops.CALL dtypes.void [538, 539, 491] CallInfo()
541 Ops.MUL dtypes.weakint [534, 104]
542 Ops.RANGE dtypes.weakint [104] (AxisType.WEAK, 1)
543 Ops.ADD dtypes.weakint [541, 542]
544 Ops.INDEX dtypes.f32 [72, 543]
545 Ops.INDEX dtypes.f32 [7, 543]
546 Ops.CONST dtypes.weakfloat [] -2.4554669595930156
547 Ops.MUL dtypes.f32 [545, 546]
548 Ops.EXP2 dtypes.f32 [547]
549 Ops.ADD dtypes.f32 [435, 548]
550 Ops.RECIP dtypes.f32 [549]
551 Ops.STORE dtypes.void [544, 550]
552 Ops.END dtypes.void [551, 534, 542]
553 Ops.SINK dtypes.void [552] KernelInfo()
554 Ops.ALLOC dtypes.f32 [6] ParamArg(slot=72, dtype=dtypes.f32, device="CPU")
555 Ops.CALL dtypes.void [553, 554, 530] CallInfo()
556 Ops.MUL dtypes.f32 [80, 156]
557 Ops.ADD dtypes.f32 [535, 556]
558 Ops.MUL dtypes.f32 [557, 296]
559 Ops.EXP2 dtypes.f32 [558]
560 Ops.REDUCE dtypes.f32 [559, 534] (Ops.ADD, 0)
561 Ops.STORE dtypes.void [5, 560]
562 Ops.SINK dtypes.void [561] KernelInfo()
563 Ops.ALLOC dtypes.f32 [2] ParamArg(slot=90, dtype=dtypes.f32, device="CPU")
564 Ops.CALL dtypes.void [562, 563, 491, 539] CallInfo()
565 Ops.INDEX dtypes.f32 [356, 495]
566 Ops.MUL dtypes.f32 [498, 565]
567 Ops.INDEX dtypes.f32 [83, 495]
568 Ops.CMPEQ dtypes.bool [420, 469]
569 Ops.CMPEQ dtypes.bool [424, 469]
570 Ops.OR dtypes.bool [568, 569]
571 Ops.CMPEQ dtypes.bool [427, 469]
572 Ops.OR dtypes.bool [570, 571]
573 Ops.CMPEQ dtypes.bool [430, 469]
574 Ops.OR dtypes.bool [572, 573]
575 Ops.WHERE dtypes.f32 [569, 436, 437]
576 Ops.WHERE dtypes.f32 [571, 434, 575]
577 Ops.WHERE dtypes.f32 [573, 433, 576]
578 Ops.WHERE dtypes.f32 [574, 577, 63]
579 Ops.CAST dtypes.i32 [578] dtypes.i32
580 Ops.CMPEQ dtypes.bool [579, 401]
581 Ops.PARAM dtypes.f32 [517] ParamArg(slot=5, dtype=dtypes.f32)
582 Ops.INDEX dtypes.f32 [581, 520]
583 Ops.WHERE dtypes.f32 [580, 582, 63]
584 Ops.REDUCE dtypes.f32 [583, 305] (Ops.ADD, 0)
585 Ops.ADD dtypes.f32 [567, 584]
586 Ops.CONST dtypes.weakfloat [] -7.0
587 Ops.MAX dtypes.f32 [585, 586]
588 Ops.CMPLT dtypes.bool [497, 587]
589 Ops.WHERE dtypes.f32 [588, 497, 587]
590 Ops.ADD dtypes.f32 [589, 435]
591 Ops.MUL dtypes.f32 [566, 590]
592 Ops.STORE dtypes.void [496, 591]
593 Ops.END dtypes.void [592, 106, 494]
594 Ops.SINK dtypes.void [593] KernelInfo()
595 Ops.ALLOC dtypes.f32 [6] ParamArg(slot=70, dtype=dtypes.f32, device="CPU")
596 Ops.PARAM dtypes.f32 [517] ParamArg(slot=20, dtype=dtypes.f32, device="CPU")
597 Ops.CALL dtypes.void [594, 595, 530, 554, 461, 410, 596] CallInfo()
598 Ops.PARAM dtypes.f32 [517] ParamArg(slot=0, dtype=dtypes.f32)
599 Ops.MUL dtypes.weakint [131, 6]
600 Ops.RANGE dtypes.weakint [6] (AxisType.WEAK, 3)
601 Ops.ADD dtypes.weakint [599, 600]
602 Ops.INDEX dtypes.f32 [598, 601]
603 Ops.ADD dtypes.weakint [412, 542]
604 Ops.INDEX dtypes.f32 [7, 603]
605 Ops.CMPEQ dtypes.bool [500, 422]
606 Ops.CMPEQ dtypes.bool [502, 422]
607 Ops.OR dtypes.bool [605, 606]
608 Ops.CMPEQ dtypes.bool [505, 422]
609 Ops.OR dtypes.bool [607, 608]
610 Ops.CMPEQ dtypes.bool [508, 422]
611 Ops.OR dtypes.bool [609, 610]
612 Ops.WHERE dtypes.f32 [606, 436, 437]
613 Ops.WHERE dtypes.f32 [608, 434, 612]
614 Ops.WHERE dtypes.f32 [610, 433, 613]
615 Ops.WHERE dtypes.f32 [611, 614, 63]
616 Ops.CAST dtypes.i32 [615] dtypes.i32
617 Ops.CMPEQ dtypes.bool [616, 401]
618 Ops.PARAM dtypes.f32 [444] ParamArg(slot=3, dtype=dtypes.f32)
619 Ops.MUL dtypes.weakint [600, 104]
620 Ops.ADD dtypes.weakint [619, 542]
621 Ops.ADD dtypes.weakint [620, 448]
622 Ops.INDEX dtypes.f32 [618, 621]
623 Ops.WHERE dtypes.f32 [617, 622, 63]
624 Ops.REDUCE dtypes.f32 [623, 305] (Ops.ADD, 0)
625 Ops.MUL dtypes.f32 [604, 624]
626 Ops.REDUCE dtypes.f32 [625, 542] (Ops.ADD, 0)
627 Ops.STORE dtypes.void [602, 626]
628 Ops.END dtypes.void [627, 131, 600]
629 Ops.SINK dtypes.void [628] KernelInfo()
630 Ops.ALLOC dtypes.f32 [517] ParamArg(slot=69, dtype=dtypes.f32, device="CPU")
631 Ops.PARAM dtypes.f32 [444] ParamArg(slot=21, dtype=dtypes.f32, device="CPU")
632 Ops.CALL dtypes.void [629, 630, 595, 410, 631] CallInfo()
633 Ops.RANGE dtypes.weakint [6] (AxisType.WEAK, 2)
634 Ops.INDEX dtypes.f32 [72, 633]
635 Ops.INDEX dtypes.f32 [7, 633]
636 Ops.PARAM dtypes.f32 [517] ParamArg(slot=2, dtype=dtypes.f32)
637 Ops.MUL dtypes.weakint [106, 6]
638 Ops.ADD dtypes.weakint [637, 633]
639 Ops.INDEX dtypes.f32 [636, 638]
640 Ops.PARAM dtypes.i32 [73] ParamArg(slot=3, dtype=dtypes.i32)
641 Ops.INDEX dtypes.i32 [640, 4]
642 Ops.CMPEQ dtypes.bool [641, 469]
643 Ops.INDEX dtypes.i32 [640, 2]
644 Ops.CMPEQ dtypes.bool [643, 469]
645 Ops.OR dtypes.bool [642, 644]
646 Ops.INDEX dtypes.i32 [640, 30]
647 Ops.CMPEQ dtypes.bool [646, 469]
648 Ops.OR dtypes.bool [645, 647]
649 Ops.INDEX dtypes.i32 [640, 266]
650 Ops.CMPEQ dtypes.bool [649, 469]
651 Ops.OR dtypes.bool [648, 650]
652 Ops.WHERE dtypes.f32 [644, 436, 437]
653 Ops.WHERE dtypes.f32 [647, 434, 652]
654 Ops.WHERE dtypes.f32 [650, 433, 653]
655 Ops.WHERE dtypes.f32 [651, 654, 63]
656 Ops.CAST dtypes.i32 [655] dtypes.i32
657 Ops.CMPEQ dtypes.bool [656, 401]
658 Ops.MUL dtypes.weakint [305, 6]
659 Ops.ADD dtypes.weakint [658, 633]
660 Ops.INDEX dtypes.f32 [377, 659]
661 Ops.WHERE dtypes.f32 [657, 660, 63]
662 Ops.REDUCE dtypes.f32 [661, 305] (Ops.ADD, 0)
663 Ops.ADD dtypes.f32 [639, 662]
664 Ops.PARAM dtypes.f32 [30] ParamArg(slot=5, dtype=dtypes.f32)
665 Ops.INDEX dtypes.f32 [664, 106]
666 Ops.PARAM dtypes.f32 [2] ParamArg(slot=6, dtype=dtypes.f32)
667 Ops.INDEX dtypes.f32 [666, 4]
668 Ops.MUL dtypes.f32 [667, 156]
669 Ops.ADD dtypes.f32 [665, 668]
670 Ops.MUL dtypes.f32 [669, 296]
671 Ops.EXP2 dtypes.f32 [670]
672 Ops.PARAM dtypes.f32 [2] ParamArg(slot=7, dtype=dtypes.f32)
673 Ops.INDEX dtypes.f32 [672, 4]
674 Ops.RECIP dtypes.f32 [673]
675 Ops.MUL dtypes.f32 [671, 674]
676 Ops.MUL dtypes.f32 [663, 675]
677 Ops.REDUCE dtypes.f32 [676, 106] (Ops.ADD, 0)
678 Ops.ADD dtypes.f32 [635, 677]
679 Ops.STORE dtypes.void [634, 678]
680 Ops.END dtypes.void [679, 633]
681 Ops.SINK dtypes.void [680] KernelInfo()
682 Ops.PARAM dtypes.f32 [6] ParamArg(slot=23, dtype=dtypes.f32, device="CPU")
683 Ops.PARAM dtypes.f32 [132] ParamArg(slot=22, dtype=dtypes.f32, device="CPU")
684 Ops.CALL dtypes.void [681, 682, 371, 630, 410, 683, 491, 539, 563] CallInfo()
685 Ops.LINEAR dtypes.void [21, 71, 103, 126, 188, 279, 291, 304, 324, 352, 374, 376, 392, 411, 460, 463, 492, 532, 540, 555, 564, 597, 632, 684]
686 Ops.STACK dtypes.weakint [2, 2, 6]
687 Ops.RESHAPE dtypes.f32 [100, 686]
688 Ops.PARAM dtypes.f32 [6] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
689 Ops.PARAM dtypes.f32 [86] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
690 Ops.PARAM dtypes.f32 [6] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
691 Ops.PARAM dtypes.f32 [132] ParamArg(slot=5, dtype=dtypes.f32, device="CPU")
692 Ops.PARAM dtypes.weakint [24] ParamArg(slot=6, dtype=dtypes.weakint, vmin_vmax=(0, 15), addrspace=AddrSpace.ALU)
693 Ops.PARAM dtypes.f32 [110] ParamArg(slot=8, dtype=dtypes.f32, device="CPU")
694 Ops.PARAM dtypes.f32 [104] ParamArg(slot=9, dtype=dtypes.f32, device="CPU")
695 Ops.PARAM dtypes.f32 [110] ParamArg(slot=10, dtype=dtypes.f32, device="CPU")
696 Ops.PARAM dtypes.f32 [104] ParamArg(slot=11, dtype=dtypes.f32, device="CPU")
697 Ops.PARAM dtypes.f32 [73] ParamArg(slot=12, dtype=dtypes.f32, device="CPU")
698 Ops.PARAM dtypes.f32 [86] ParamArg(slot=13, dtype=dtypes.f32, device="CPU")
699 Ops.PARAM dtypes.f32 [6] ParamArg(slot=15, dtype=dtypes.f32, device="CPU")
700 Ops.PARAM dtypes.f32 [132] ParamArg(slot=16, dtype=dtypes.f32, device="CPU")
701 Ops.PARAM dtypes.f32 [73] ParamArg(slot=17, dtype=dtypes.f32, device="CPU")
702 Ops.PARAM dtypes.f32 [444] ParamArg(slot=18, dtype=dtypes.f32, device="CPU")
703 Ops.PARAM dtypes.f32 [517] ParamArg(slot=19, dtype=dtypes.f32, device="CPU")
704 Ops.PARAM dtypes.f32 [444] ParamArg(slot=20, dtype=dtypes.f32, device="CPU")
705 Ops.PARAM dtypes.f32 [517] ParamArg(slot=21, dtype=dtypes.f32, device="CPU")
706 Ops.PARAM dtypes.f32 [444] ParamArg(slot=22, dtype=dtypes.f32, device="CPU")
707 Ops.PARAM dtypes.f32 [132] ParamArg(slot=23, dtype=dtypes.f32, device="CPU")
708 Ops.RESHAPE dtypes.f32 [20, 686]
709 Ops.CALL dtypes.void [685, 687, 688, 689, 690, 691, 692, 1, 693, 694, 695, 696, 697, 698, 389, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708] CallInfo(name="FFNBlock.__call__.<locals>._run", precompile=True)
710 Ops.AFTER dtypes.f16 [1, 709]
711 Ops.AFTER dtypes.f32 [20, 709]
712 Ops.SINK dtypes.void [710, 711]

