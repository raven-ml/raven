# tolk D473 golden=546b2b6399f0d63ffda0c8b735a52f0d
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f16 [0] ParamArg(slot=6, dtype=dtypes.f16, device="CPU")
2 Ops.CONST dtypes.weakint [] 8
3 Ops.PARAM dtypes.f32 [2] ParamArg(slot=0, dtype=dtypes.f32)
4 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
5 Ops.INDEX dtypes.f32 [3, 4]
6 Ops.CONST dtypes.weakint [] 512
7 Ops.PARAM dtypes.f32 [6] ParamArg(slot=1, dtype=dtypes.f32)
8 Ops.CONST dtypes.weakint [] 64
9 Ops.MUL dtypes.weakint [4, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.INDEX dtypes.f32 [7, 11]
13 Ops.MUL dtypes.f32 [12, 12]
14 Ops.REDUCE dtypes.f32 [13, 10] (Ops.ADD, 0)
15 Ops.CONST dtypes.weakfloat [] 0.015625
16 Ops.MUL dtypes.f32 [14, 15]
17 Ops.CONST dtypes.weakfloat [] 1e-05
18 Ops.ADD dtypes.f32 [16, 17]
19 Ops.SQRT dtypes.f32 [18]
20 Ops.STORE dtypes.void [5, 19]
21 Ops.END dtypes.void [20, 4]
22 Ops.SINK dtypes.void [21] KernelInfo()
23 Ops.ALLOC dtypes.f32 [2] ParamArg(slot=78, dtype=dtypes.f32, device="CPU")
24 Ops.PARAM dtypes.f32 [6] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
25 Ops.CALL dtypes.void [22, 23, 24] CallInfo()
26 Ops.PARAM dtypes.f32 [6] ParamArg(slot=0, dtype=dtypes.f32)
27 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
28 Ops.CONST dtypes.weakint [] 16
29 Ops.MUL dtypes.weakint [27, 28]
30 Ops.RANGE dtypes.weakint [28] (AxisType.WEAK, 3)
31 Ops.ADD dtypes.weakint [29, 30]
32 Ops.CONST dtypes.weakint [] 4
33 Ops.RANGE dtypes.weakint [32] (AxisType.WEAK, 1)
34 Ops.CONST dtypes.weakint [] 128
35 Ops.MUL dtypes.weakint [33, 34]
36 Ops.ADD dtypes.weakint [31, 35]
37 Ops.INDEX dtypes.f32 [26, 36]
38 Ops.MUL dtypes.weakint [27, 8]
39 Ops.ADD dtypes.weakint [38, 10]
40 Ops.INDEX dtypes.f32 [7, 39]
41 Ops.PARAM dtypes.f32 [2] ParamArg(slot=2, dtype=dtypes.f32)
42 Ops.INDEX dtypes.f32 [41, 27]
43 Ops.RECIP dtypes.f32 [42]
44 Ops.MUL dtypes.f32 [40, 43]
45 Ops.PARAM dtypes.f32 [8] ParamArg(slot=3, dtype=dtypes.f32)
46 Ops.INDEX dtypes.f32 [45, 10]
47 Ops.MUL dtypes.f32 [44, 46]
48 Ops.CONST dtypes.weakint [] 4096
49 Ops.PARAM dtypes.f32 [48] ParamArg(slot=4, dtype=dtypes.f32)
50 Ops.MUL dtypes.weakint [33, 28]
51 Ops.ADD dtypes.weakint [50, 30]
52 Ops.MUL dtypes.weakint [51, 8]
53 Ops.ADD dtypes.weakint [52, 10]
54 Ops.INDEX dtypes.f32 [49, 53]
55 Ops.MUL dtypes.f32 [47, 54]
56 Ops.REDUCE dtypes.f32 [55, 10] (Ops.ADD, 0)
57 Ops.PARAM dtypes.f32 [8] ParamArg(slot=5, dtype=dtypes.f32)
58 Ops.INDEX dtypes.f32 [57, 51]
59 Ops.ADD dtypes.f32 [56, 58]
60 Ops.STORE dtypes.void [37, 59]
61 Ops.END dtypes.void [60, 33, 27, 30]
62 Ops.SINK dtypes.void [61] KernelInfo()
63 Ops.ALLOC dtypes.f32 [6] ParamArg(slot=77, dtype=dtypes.f32, device="CPU")
64 Ops.PARAM dtypes.f32 [8] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
65 Ops.PARAM dtypes.f32 [48] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
66 Ops.PARAM dtypes.f32 [8] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
67 Ops.CALL dtypes.void [62, 63, 24, 23, 64, 65, 66] CallInfo()
68 Ops.CONST dtypes.weakint [] 256
69 Ops.PARAM dtypes.f32 [68] ParamArg(slot=0, dtype=dtypes.f32)
70 Ops.CONST dtypes.weakint [] 2
71 Ops.RANGE dtypes.weakint [70] (AxisType.WEAK, 1)
72 Ops.MUL dtypes.weakint [71, 34]
73 Ops.ADD dtypes.weakint [31, 72]
74 Ops.INDEX dtypes.f32 [69, 73]
75 Ops.CONST dtypes.weakint [] 2048
76 Ops.PARAM dtypes.f32 [75] ParamArg(slot=4, dtype=dtypes.f32)
77 Ops.MUL dtypes.weakint [71, 28]
78 Ops.ADD dtypes.weakint [77, 30]
79 Ops.MUL dtypes.weakint [78, 8]
80 Ops.ADD dtypes.weakint [79, 10]
81 Ops.INDEX dtypes.f32 [76, 80]
82 Ops.MUL dtypes.f32 [47, 81]
83 Ops.REDUCE dtypes.f32 [82, 10] (Ops.ADD, 0)
84 Ops.CONST dtypes.weakint [] 32
85 Ops.PARAM dtypes.f32 [84] ParamArg(slot=5, dtype=dtypes.f32)
86 Ops.INDEX dtypes.f32 [85, 78]
87 Ops.ADD dtypes.f32 [83, 86]
88 Ops.STORE dtypes.void [74, 87]
89 Ops.END dtypes.void [88, 71, 27, 30]
90 Ops.SINK dtypes.void [89] KernelInfo()
91 Ops.ALLOC dtypes.f32 [68] ParamArg(slot=79, dtype=dtypes.f32, device="CPU")
92 Ops.PARAM dtypes.f32 [75] ParamArg(slot=6, dtype=dtypes.f32, device="CPU")
93 Ops.PARAM dtypes.f32 [84] ParamArg(slot=7, dtype=dtypes.f32, device="CPU")
94 Ops.CALL dtypes.void [90, 91, 24, 23, 64, 92, 93] CallInfo()
95 Ops.PARAM dtypes.f16 [0] ParamArg(slot=0, dtype=dtypes.f16)
96 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 3)
97 Ops.MUL dtypes.weakint [96, 28]
98 Ops.RANGE dtypes.weakint [28] (AxisType.WEAK, 4)
99 Ops.ADD dtypes.weakint [97, 98]
100 Ops.RANGE dtypes.weakint [70] (AxisType.WEAK, 2)
101 Ops.MUL dtypes.weakint [100, 68]
102 Ops.ADD dtypes.weakint [99, 101]
103 Ops.MUL dtypes.weakint [71, 6]
104 Ops.ADD dtypes.weakint [102, 103]
105 Ops.INDEX dtypes.f16 [95, 104]
106 Ops.CONST dtypes.weakint [] 1
107 Ops.CMPLT dtypes.bool [71, 106]
108 Ops.CMPLT dtypes.bool [98, 2]
109 Ops.PARAM dtypes.f32 [68] ParamArg(slot=1, dtype=dtypes.f32)
110 Ops.MOD dtypes.weakint [98, 2]
111 Ops.ADD dtypes.weakint [97, 110]
112 Ops.MUL dtypes.weakint [100, 34]
113 Ops.ADD dtypes.weakint [111, 112]
114 Ops.INDEX dtypes.f32 [109, 113]
115 Ops.PARAM dtypes.f32 [68] ParamArg(slot=2, dtype=dtypes.f32)
116 Ops.INDEX dtypes.f32 [115, 111]
117 Ops.MUL dtypes.f32 [114, 116]
118 Ops.ADD dtypes.weakint [113, 2]
119 Ops.INDEX dtypes.f32 [109, 118]
120 Ops.ADD dtypes.weakint [111, 2]
121 Ops.INDEX dtypes.f32 [115, 120]
122 Ops.MUL dtypes.f32 [119, 121]
123 Ops.CONST dtypes.weakfloat [] -1.0
124 Ops.MUL dtypes.f32 [122, 123]
125 Ops.ADD dtypes.f32 [117, 124]
126 Ops.MUL dtypes.f32 [119, 116]
127 Ops.MUL dtypes.f32 [114, 121]
128 Ops.ADD dtypes.f32 [126, 127]
129 Ops.WHERE dtypes.f32 [108, 125, 128]
130 Ops.PARAM dtypes.f32 [6] ParamArg(slot=3, dtype=dtypes.f32)
131 Ops.MUL dtypes.weakint [96, 8]
132 Ops.ADD dtypes.weakint [131, 10]
133 Ops.INDEX dtypes.f32 [130, 132]
134 Ops.PARAM dtypes.f32 [2] ParamArg(slot=4, dtype=dtypes.f32)
135 Ops.INDEX dtypes.f32 [134, 96]
136 Ops.RECIP dtypes.f32 [135]
137 Ops.MUL dtypes.f32 [133, 136]
138 Ops.INDEX dtypes.f32 [57, 10]
139 Ops.MUL dtypes.f32 [137, 138]
140 Ops.PARAM dtypes.f32 [75] ParamArg(slot=6, dtype=dtypes.f32)
141 Ops.MUL dtypes.weakint [100, 28]
142 Ops.ADD dtypes.weakint [141, 98]
143 Ops.MUL dtypes.weakint [142, 8]
144 Ops.ADD dtypes.weakint [143, 10]
145 Ops.INDEX dtypes.f32 [140, 144]
146 Ops.MUL dtypes.f32 [139, 145]
147 Ops.REDUCE dtypes.f32 [146, 10] (Ops.ADD, 0)
148 Ops.PARAM dtypes.f32 [84] ParamArg(slot=7, dtype=dtypes.f32)
149 Ops.INDEX dtypes.f32 [148, 142]
150 Ops.ADD dtypes.f32 [147, 149]
151 Ops.WHERE dtypes.f32 [107, 129, 150]
152 Ops.CAST dtypes.f16 [151] dtypes.f16
153 Ops.STORE dtypes.void [105, 152]
154 Ops.END dtypes.void [153, 71, 100, 96, 98]
155 Ops.SINK dtypes.void [154] KernelInfo()
156 Ops.PARAM dtypes.f16 [0] ParamArg(slot=5, dtype=dtypes.f16, device="CPU")
157 Ops.PARAM dtypes.f32 [68] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
158 Ops.PARAM dtypes.f32 [75] ParamArg(slot=8, dtype=dtypes.f32, device="CPU")
159 Ops.PARAM dtypes.f32 [84] ParamArg(slot=9, dtype=dtypes.f32, device="CPU")
160 Ops.CALL dtypes.void [155, 156, 91, 157, 24, 23, 64, 158, 159] CallInfo()
161 Ops.CONST dtypes.weakint [] 288
162 Ops.PARAM dtypes.f32 [161] ParamArg(slot=0, dtype=dtypes.f32)
163 Ops.CONST dtypes.weakint [] 9
164 Ops.MUL dtypes.weakint [27, 163]
165 Ops.RANGE dtypes.weakint [163] (AxisType.WEAK, 3)
166 Ops.ADD dtypes.weakint [164, 165]
167 Ops.CONST dtypes.weakint [] 72
168 Ops.MUL dtypes.weakint [33, 167]
169 Ops.ADD dtypes.weakint [166, 168]
170 Ops.INDEX dtypes.f32 [162, 169]
171 Ops.RANGE dtypes.weakint [28] (AxisType.WEAK, 0)
172 Ops.CMPLT dtypes.bool [171, 2]
173 Ops.MOD dtypes.weakint [171, 2]
174 Ops.ADD dtypes.weakint [29, 173]
175 Ops.ADD dtypes.weakint [174, 35]
176 Ops.INDEX dtypes.f32 [7, 175]
177 Ops.INDEX dtypes.f32 [115, 174]
178 Ops.MUL dtypes.f32 [176, 177]
179 Ops.ADD dtypes.weakint [175, 2]
180 Ops.INDEX dtypes.f32 [7, 179]
181 Ops.ADD dtypes.weakint [174, 2]
182 Ops.INDEX dtypes.f32 [115, 181]
183 Ops.MUL dtypes.f32 [180, 182]
184 Ops.MUL dtypes.f32 [183, 123]
185 Ops.ADD dtypes.f32 [178, 184]
186 Ops.MUL dtypes.f32 [180, 177]
187 Ops.MUL dtypes.f32 [176, 182]
188 Ops.ADD dtypes.f32 [186, 187]
189 Ops.WHERE dtypes.f32 [172, 185, 188]
190 Ops.CMPLT dtypes.bool [165, 2]
191 Ops.PARAM dtypes.f16 [0] ParamArg(slot=3, dtype=dtypes.f16)
192 Ops.MUL dtypes.weakint [165, 28]
193 Ops.ADD dtypes.weakint [192, 171]
194 Ops.DIV dtypes.weakint [33, 70]
195 Ops.MUL dtypes.weakint [194, 68]
196 Ops.ADD dtypes.weakint [193, 195]
197 Ops.CONST dtypes.bool [] Invalid
198 Ops.WHERE dtypes.weakint [190, 196, 197]
199 Ops.INDEX dtypes.f16 [191, 198]
200 Ops.CONST dtypes.weakfloat [] 0.0
201 Ops.WHERE dtypes.f16 [190, 199, 200]
202 Ops.CAST dtypes.f32 [201] dtypes.f32
203 Ops.MUL dtypes.f32 [189, 202]
204 Ops.REDUCE dtypes.f32 [203, 171] (Ops.ADD, 0)
205 Ops.CONST dtypes.weakfloat [] 0.25
206 Ops.MUL dtypes.f32 [204, 205]
207 Ops.CAST dtypes.i32 [27] dtypes.i32
208 Ops.CAST dtypes.i32 [165] dtypes.i32
209 Ops.CMPLT dtypes.bool [207, 208]
210 Ops.CONST dtypes.weakfloat [] -inf
211 Ops.WHERE dtypes.weakfloat [209, 210, 200]
212 Ops.CAST dtypes.f32 [211] dtypes.f32
213 Ops.PARAM dtypes.f32 [32] ParamArg(slot=4, dtype=dtypes.f32)
214 Ops.INDEX dtypes.f32 [213, 33]
215 Ops.WHERE dtypes.f32 [190, 212, 214]
216 Ops.ADD dtypes.f32 [206, 215]
217 Ops.STORE dtypes.void [170, 216]
218 Ops.END dtypes.void [217, 33, 27, 165]
219 Ops.SINK dtypes.void [218] KernelInfo()
220 Ops.ALLOC dtypes.f32 [161] ParamArg(slot=76, dtype=dtypes.f32, device="CPU")
221 Ops.PARAM dtypes.f32 [32] ParamArg(slot=10, dtype=dtypes.f32, device="CPU")
222 Ops.CALL dtypes.void [219, 220, 63, 157, 156, 221] CallInfo()
223 Ops.PARAM dtypes.f32 [84] ParamArg(slot=0, dtype=dtypes.f32)
224 Ops.MUL dtypes.weakint [33, 2]
225 Ops.ADD dtypes.weakint [224, 27]
226 Ops.INDEX dtypes.f32 [223, 225]
227 Ops.PARAM dtypes.f32 [161] ParamArg(slot=1, dtype=dtypes.f32)
228 Ops.RANGE dtypes.weakint [163] (AxisType.WEAK, 0)
229 Ops.ADD dtypes.weakint [164, 228]
230 Ops.ADD dtypes.weakint [229, 168]
231 Ops.INDEX dtypes.f32 [227, 230]
232 Ops.REDUCE dtypes.f32 [231, 228] (Ops.MAX, 0)
233 Ops.STORE dtypes.void [226, 232]
234 Ops.END dtypes.void [233, 33, 27]
235 Ops.SINK dtypes.void [234] KernelInfo()
236 Ops.ALLOC dtypes.f32 [84] ParamArg(slot=80, dtype=dtypes.f32, device="CPU")
237 Ops.CALL dtypes.void [235, 236, 220] CallInfo()
238 Ops.PARAM dtypes.f32 [84] ParamArg(slot=2, dtype=dtypes.f32)
239 Ops.INDEX dtypes.f32 [238, 225]
240 Ops.MUL dtypes.f32 [239, 123]
241 Ops.ADD dtypes.f32 [231, 240]
242 Ops.CONST dtypes.weakfloat [] 1.4426950408889634
243 Ops.MUL dtypes.f32 [241, 242]
244 Ops.EXP2 dtypes.f32 [243]
245 Ops.REDUCE dtypes.f32 [244, 228] (Ops.ADD, 0)
246 Ops.STORE dtypes.void [226, 245]
247 Ops.END dtypes.void [246, 33, 27]
248 Ops.SINK dtypes.void [247] KernelInfo()
249 Ops.ALLOC dtypes.f32 [84] ParamArg(slot=81, dtype=dtypes.f32, device="CPU")
250 Ops.CALL dtypes.void [248, 249, 220, 236] CallInfo()
251 Ops.PARAM dtypes.f32 [84] ParamArg(slot=3, dtype=dtypes.f32)
252 Ops.INDEX dtypes.f32 [251, 225]
253 Ops.RECIP dtypes.f32 [252]
254 Ops.MUL dtypes.f32 [244, 253]
255 Ops.CMPLT dtypes.bool [228, 2]
256 Ops.PARAM dtypes.f16 [0] ParamArg(slot=4, dtype=dtypes.f16)
257 Ops.MUL dtypes.weakint [228, 28]
258 Ops.ADD dtypes.weakint [257, 30]
259 Ops.ADD dtypes.weakint [258, 195]
260 Ops.ADD dtypes.weakint [259, 6]
261 Ops.WHERE dtypes.weakint [255, 260, 197]
262 Ops.INDEX dtypes.f16 [256, 261]
263 Ops.WHERE dtypes.f16 [255, 262, 200]
264 Ops.CAST dtypes.f32 [263] dtypes.f32
265 Ops.MUL dtypes.f32 [254, 264]
266 Ops.REDUCE dtypes.f32 [265, 228] (Ops.ADD, 0)
267 Ops.STORE dtypes.void [37, 266]
268 Ops.END dtypes.void [267, 33, 27, 30]
269 Ops.SINK dtypes.void [268] KernelInfo()
270 Ops.ALLOC dtypes.f32 [6] ParamArg(slot=75, dtype=dtypes.f32, device="CPU")
271 Ops.CALL dtypes.void [269, 270, 220, 236, 249, 156] CallInfo()
272 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 2)
273 Ops.ADD dtypes.weakint [9, 272]
274 Ops.INDEX dtypes.f32 [26, 273]
275 Ops.INDEX dtypes.f32 [7, 273]
276 Ops.PARAM dtypes.f32 [6] ParamArg(slot=2, dtype=dtypes.f32)
277 Ops.MUL dtypes.weakint [4, 28]
278 Ops.MOD dtypes.weakint [10, 28]
279 Ops.ADD dtypes.weakint [277, 278]
280 Ops.DIV dtypes.weakint [10, 28]
281 Ops.MUL dtypes.weakint [280, 34]
282 Ops.ADD dtypes.weakint [279, 281]
283 Ops.INDEX dtypes.f32 [276, 282]
284 Ops.PARAM dtypes.f32 [48] ParamArg(slot=3, dtype=dtypes.f32)
285 Ops.MUL dtypes.weakint [272, 8]
286 Ops.ADD dtypes.weakint [285, 10]
287 Ops.INDEX dtypes.f32 [284, 286]
288 Ops.MUL dtypes.f32 [283, 287]
289 Ops.REDUCE dtypes.f32 [288, 10] (Ops.ADD, 0)
290 Ops.PARAM dtypes.f32 [8] ParamArg(slot=4, dtype=dtypes.f32)
291 Ops.INDEX dtypes.f32 [290, 272]
292 Ops.ADD dtypes.f32 [289, 291]
293 Ops.ADD dtypes.f32 [275, 292]
294 Ops.STORE dtypes.void [274, 293]
295 Ops.END dtypes.void [294, 4, 272]
296 Ops.SINK dtypes.void [295] KernelInfo()
297 Ops.ALLOC dtypes.f32 [6] ParamArg(slot=74, dtype=dtypes.f32, device="CPU")
298 Ops.PARAM dtypes.f32 [48] ParamArg(slot=11, dtype=dtypes.f32, device="CPU")
299 Ops.PARAM dtypes.f32 [8] ParamArg(slot=12, dtype=dtypes.f32, device="CPU")
300 Ops.CALL dtypes.void [296, 297, 24, 270, 298, 299] CallInfo()
301 Ops.ALLOC dtypes.f32 [2] ParamArg(slot=83, dtype=dtypes.f32, device="CPU")
302 Ops.CALL dtypes.void [22, 301, 297] CallInfo()
303 Ops.MUL dtypes.weakint [4, 32]
304 Ops.RANGE dtypes.weakint [32] (AxisType.WEAK, 2)
305 Ops.ADD dtypes.weakint [303, 304]
306 Ops.INDEX dtypes.f32 [223, 305]
307 Ops.INDEX dtypes.f32 [41, 4]
308 Ops.RECIP dtypes.f32 [307]
309 Ops.MUL dtypes.f32 [12, 308]
310 Ops.MUL dtypes.f32 [309, 46]
311 Ops.PARAM dtypes.f32 [68] ParamArg(slot=4, dtype=dtypes.f32)
312 Ops.MUL dtypes.weakint [304, 8]
313 Ops.ADD dtypes.weakint [312, 10]
314 Ops.INDEX dtypes.f32 [311, 313]
315 Ops.MUL dtypes.f32 [310, 314]
316 Ops.REDUCE dtypes.f32 [315, 10] (Ops.ADD, 0)
317 Ops.PARAM dtypes.f32 [32] ParamArg(slot=5, dtype=dtypes.f32)
318 Ops.INDEX dtypes.f32 [317, 304]
319 Ops.ADD dtypes.f32 [316, 318]
320 Ops.STORE dtypes.void [306, 319]
321 Ops.END dtypes.void [320, 4, 304]
322 Ops.SINK dtypes.void [321] KernelInfo()
323 Ops.ALLOC dtypes.f32 [84] ParamArg(slot=85, dtype=dtypes.f32, device="CPU")
324 Ops.PARAM dtypes.f32 [8] ParamArg(slot=13, dtype=dtypes.f32, device="CPU")
325 Ops.PARAM dtypes.f32 [68] ParamArg(slot=14, dtype=dtypes.f32, device="CPU")
326 Ops.PARAM dtypes.f32 [32] ParamArg(slot=15, dtype=dtypes.f32, device="CPU")
327 Ops.CALL dtypes.void [322, 323, 297, 301, 324, 325, 326] CallInfo()
328 Ops.PARAM dtypes.i32 [84] ParamArg(slot=0, dtype=dtypes.i32)
329 Ops.INDEX dtypes.i32 [328, 305]
330 Ops.PARAM dtypes.f32 [84] ParamArg(slot=1, dtype=dtypes.f32)
331 Ops.RANGE dtypes.weakint [32] (AxisType.WEAK, 0)
332 Ops.ADD dtypes.weakint [303, 331]
333 Ops.INDEX dtypes.f32 [330, 332]
334 Ops.INDEX dtypes.f32 [330, 305]
335 Ops.CMPLT dtypes.bool [333, 334]
336 Ops.CMPEQ dtypes.bool [334, 333]
337 Ops.CAST dtypes.i32 [304] dtypes.i32
338 Ops.CAST dtypes.i32 [331] dtypes.i32
339 Ops.CMPLT dtypes.bool [337, 338]
340 Ops.AND dtypes.bool [336, 339]
341 Ops.OR dtypes.bool [335, 340]
342 Ops.CAST dtypes.i32 [341] dtypes.i32
343 Ops.REDUCE dtypes.i32 [342, 331] (Ops.ADD, 0)
344 Ops.STORE dtypes.void [329, 343]
345 Ops.END dtypes.void [344, 4, 304]
346 Ops.SINK dtypes.void [345] KernelInfo()
347 Ops.ALLOC dtypes.i32 [84] ParamArg(slot=84, dtype=dtypes.i32, device="CPU")
348 Ops.CALL dtypes.void [346, 347, 323] CallInfo()
349 Ops.RANGE dtypes.weakint [70] (AxisType.WEAK, 3)
350 Ops.MUL dtypes.weakint [349, 84]
351 Ops.RANGE dtypes.weakint [84] (AxisType.WEAK, 4)
352 Ops.ADD dtypes.weakint [350, 351]
353 Ops.ADD dtypes.weakint [352, 38]
354 Ops.INDEX dtypes.f32 [26, 353]
355 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 1)
356 Ops.ADD dtypes.weakint [38, 355]
357 Ops.INDEX dtypes.f32 [7, 356]
358 Ops.MUL dtypes.f32 [357, 43]
359 Ops.INDEX dtypes.f32 [45, 355]
360 Ops.MUL dtypes.f32 [358, 359]
361 Ops.PARAM dtypes.i32 [84] ParamArg(slot=4, dtype=dtypes.i32)
362 Ops.MUL dtypes.weakint [27, 32]
363 Ops.INDEX dtypes.i32 [361, 362]
364 Ops.CAST dtypes.i32 [349] dtypes.i32
365 Ops.ADD dtypes.i32 [364, 70]
366 Ops.CMPEQ dtypes.bool [363, 365]
367 Ops.ADD dtypes.weakint [362, 106]
368 Ops.INDEX dtypes.i32 [361, 367]
369 Ops.CMPEQ dtypes.bool [368, 365]
370 Ops.OR dtypes.bool [366, 369]
371 Ops.ADD dtypes.weakint [362, 70]
372 Ops.INDEX dtypes.i32 [361, 371]
373 Ops.CMPEQ dtypes.bool [372, 365]
374 Ops.OR dtypes.bool [370, 373]
375 Ops.CONST dtypes.weakint [] 3
376 Ops.ADD dtypes.weakint [362, 375]
377 Ops.INDEX dtypes.i32 [361, 376]
378 Ops.CMPEQ dtypes.bool [377, 365]
379 Ops.OR dtypes.bool [374, 378]
380 Ops.CONST dtypes.weakfloat [] 3.0
381 Ops.CONST dtypes.weakfloat [] 2.0
382 Ops.CONST dtypes.weakfloat [] 1.0
383 Ops.CAST dtypes.f32 [382] dtypes.f32
384 Ops.CAST dtypes.f32 [200] dtypes.f32
385 Ops.WHERE dtypes.f32 [369, 383, 384]
386 Ops.WHERE dtypes.f32 [373, 381, 385]
387 Ops.WHERE dtypes.f32 [378, 380, 386]
388 Ops.WHERE dtypes.f32 [379, 387, 200]
389 Ops.CAST dtypes.i32 [388] dtypes.i32
390 Ops.CMPEQ dtypes.bool [389, 338]
391 Ops.CONST dtypes.weakint [] 8192
392 Ops.PARAM dtypes.f32 [391] ParamArg(slot=5, dtype=dtypes.f32)
393 Ops.MUL dtypes.weakint [351, 8]
394 Ops.ADD dtypes.weakint [393, 355]
395 Ops.MUL dtypes.weakint [331, 75]
396 Ops.ADD dtypes.weakint [394, 395]
397 Ops.INDEX dtypes.f32 [392, 396]
398 Ops.WHERE dtypes.f32 [390, 397, 200]
399 Ops.REDUCE dtypes.f32 [398, 331] (Ops.ADD, 0)
400 Ops.MUL dtypes.f32 [360, 399]
401 Ops.REDUCE dtypes.f32 [400, 355] (Ops.ADD, 0)
402 Ops.STORE dtypes.void [354, 401]
403 Ops.END dtypes.void [402, 27, 349, 351]
404 Ops.SINK dtypes.void [403] KernelInfo()
405 Ops.ALLOC dtypes.f32 [6] ParamArg(slot=71, dtype=dtypes.f32, device="CPU")
406 Ops.PARAM dtypes.f32 [391] ParamArg(slot=16, dtype=dtypes.f32, device="CPU")
407 Ops.CALL dtypes.void [404, 405, 297, 301, 324, 347, 406] CallInfo()
408 Ops.ALLOC dtypes.f32 [6] ParamArg(slot=73, dtype=dtypes.f32, device="CPU")
409 Ops.PARAM dtypes.f32 [391] ParamArg(slot=18, dtype=dtypes.f32, device="CPU")
410 Ops.CALL dtypes.void [404, 408, 297, 301, 324, 347, 409] CallInfo()
411 Ops.PARAM dtypes.f32 [28] ParamArg(slot=0, dtype=dtypes.f32)
412 Ops.MUL dtypes.weakint [4, 70]
413 Ops.ADD dtypes.weakint [412, 100]
414 Ops.INDEX dtypes.f32 [411, 413]
415 Ops.PARAM dtypes.i32 [84] ParamArg(slot=1, dtype=dtypes.i32)
416 Ops.INDEX dtypes.i32 [415, 303]
417 Ops.CAST dtypes.i32 [100] dtypes.i32
418 Ops.ADD dtypes.i32 [417, 70]
419 Ops.CMPEQ dtypes.bool [416, 418]
420 Ops.ADD dtypes.weakint [303, 106]
421 Ops.INDEX dtypes.i32 [415, 420]
422 Ops.CMPEQ dtypes.bool [421, 418]
423 Ops.OR dtypes.bool [419, 422]
424 Ops.ADD dtypes.weakint [303, 70]
425 Ops.INDEX dtypes.i32 [415, 424]
426 Ops.CMPEQ dtypes.bool [425, 418]
427 Ops.OR dtypes.bool [423, 426]
428 Ops.ADD dtypes.weakint [303, 375]
429 Ops.INDEX dtypes.i32 [415, 428]
430 Ops.CMPEQ dtypes.bool [429, 418]
431 Ops.OR dtypes.bool [427, 430]
432 Ops.WHERE dtypes.f32 [422, 383, 384]
433 Ops.WHERE dtypes.f32 [426, 381, 432]
434 Ops.WHERE dtypes.f32 [430, 380, 433]
435 Ops.WHERE dtypes.f32 [431, 434, 200]
436 Ops.CAST dtypes.i32 [435] dtypes.i32
437 Ops.CMPEQ dtypes.bool [436, 338]
438 Ops.INDEX dtypes.f32 [238, 332]
439 Ops.WHERE dtypes.f32 [437, 438, 200]
440 Ops.REDUCE dtypes.f32 [439, 331] (Ops.ADD, 0)
441 Ops.STORE dtypes.void [414, 440]
442 Ops.END dtypes.void [441, 4, 100]
443 Ops.SINK dtypes.void [442] KernelInfo()
444 Ops.ALLOC dtypes.f32 [28] ParamArg(slot=86, dtype=dtypes.f32, device="CPU")
445 Ops.CALL dtypes.void [443, 444, 347, 323] CallInfo()
446 Ops.MUL dtypes.weakint [100, 84]
447 Ops.RANGE dtypes.weakint [84] (AxisType.WEAK, 3)
448 Ops.ADD dtypes.weakint [446, 447]
449 Ops.ADD dtypes.weakint [448, 9]
450 Ops.INDEX dtypes.f32 [26, 449]
451 Ops.CONST dtypes.weakfloat [] 7.0
452 Ops.INDEX dtypes.f32 [7, 449]
453 Ops.PARAM dtypes.i32 [84] ParamArg(slot=2, dtype=dtypes.i32)
454 Ops.INDEX dtypes.i32 [453, 303]
455 Ops.CMPEQ dtypes.bool [454, 418]
456 Ops.INDEX dtypes.i32 [453, 420]
457 Ops.CMPEQ dtypes.bool [456, 418]
458 Ops.OR dtypes.bool [455, 457]
459 Ops.INDEX dtypes.i32 [453, 424]
460 Ops.CMPEQ dtypes.bool [459, 418]
461 Ops.OR dtypes.bool [458, 460]
462 Ops.INDEX dtypes.i32 [453, 428]
463 Ops.CMPEQ dtypes.bool [462, 418]
464 Ops.OR dtypes.bool [461, 463]
465 Ops.WHERE dtypes.f32 [457, 383, 384]
466 Ops.WHERE dtypes.f32 [460, 381, 465]
467 Ops.WHERE dtypes.f32 [463, 380, 466]
468 Ops.WHERE dtypes.f32 [464, 467, 200]
469 Ops.CAST dtypes.i32 [468] dtypes.i32
470 Ops.CMPEQ dtypes.bool [469, 338]
471 Ops.PARAM dtypes.f32 [34] ParamArg(slot=3, dtype=dtypes.f32)
472 Ops.MUL dtypes.weakint [331, 84]
473 Ops.ADD dtypes.weakint [472, 447]
474 Ops.INDEX dtypes.f32 [471, 473]
475 Ops.WHERE dtypes.f32 [470, 474, 200]
476 Ops.REDUCE dtypes.f32 [475, 331] (Ops.ADD, 0)
477 Ops.ADD dtypes.f32 [452, 476]
478 Ops.CMPLT dtypes.bool [451, 477]
479 Ops.WHERE dtypes.f32 [478, 451, 477]
480 Ops.STORE dtypes.void [450, 479]
481 Ops.END dtypes.void [480, 4, 100, 447]
482 Ops.SINK dtypes.void [481] KernelInfo()
483 Ops.ALLOC dtypes.f32 [6] ParamArg(slot=82, dtype=dtypes.f32, device="CPU")
484 Ops.PARAM dtypes.f32 [34] ParamArg(slot=17, dtype=dtypes.f32, device="CPU")
485 Ops.CALL dtypes.void [482, 483, 405, 347, 484] CallInfo()
486 Ops.PARAM dtypes.f32 [28] ParamArg(slot=1, dtype=dtypes.f32)
487 Ops.RANGE dtypes.weakint [70] (AxisType.WEAK, 0)
488 Ops.ADD dtypes.weakint [412, 487]
489 Ops.INDEX dtypes.f32 [486, 488]
490 Ops.REDUCE dtypes.f32 [489, 487] (Ops.MAX, 0)
491 Ops.STORE dtypes.void [5, 490]
492 Ops.END dtypes.void [491, 4]
493 Ops.SINK dtypes.void [492] KernelInfo()
494 Ops.ALLOC dtypes.f32 [2] ParamArg(slot=87, dtype=dtypes.f32, device="CPU")
495 Ops.CALL dtypes.void [493, 494, 444] CallInfo()
496 Ops.MUL dtypes.weakint [71, 84]
497 Ops.RANGE dtypes.weakint [84] (AxisType.WEAK, 2)
498 Ops.ADD dtypes.weakint [496, 497]
499 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
500 Ops.MUL dtypes.weakint [499, 8]
501 Ops.ADD dtypes.weakint [498, 500]
502 Ops.INDEX dtypes.f32 [26, 501]
503 Ops.INDEX dtypes.f32 [7, 501]
504 Ops.CONST dtypes.weakfloat [] -2.4554669595930156
505 Ops.MUL dtypes.f32 [503, 504]
506 Ops.EXP2 dtypes.f32 [505]
507 Ops.ADD dtypes.f32 [382, 506]
508 Ops.RECIP dtypes.f32 [507]
509 Ops.STORE dtypes.void [502, 508]
510 Ops.END dtypes.void [509, 499, 71, 497]
511 Ops.SINK dtypes.void [510] KernelInfo()
512 Ops.ALLOC dtypes.f32 [6] ParamArg(slot=72, dtype=dtypes.f32, device="CPU")
513 Ops.CALL dtypes.void [511, 512, 483] CallInfo()
514 Ops.MUL dtypes.f32 [307, 123]
515 Ops.ADD dtypes.f32 [489, 514]
516 Ops.MUL dtypes.f32 [515, 242]
517 Ops.EXP2 dtypes.f32 [516]
518 Ops.REDUCE dtypes.f32 [517, 487] (Ops.ADD, 0)
519 Ops.STORE dtypes.void [5, 518]
520 Ops.END dtypes.void [519, 4]
521 Ops.SINK dtypes.void [520] KernelInfo()
522 Ops.ALLOC dtypes.f32 [2] ParamArg(slot=88, dtype=dtypes.f32, device="CPU")
523 Ops.CALL dtypes.void [521, 522, 444, 494] CallInfo()
524 Ops.INDEX dtypes.f32 [276, 449]
525 Ops.MUL dtypes.f32 [452, 524]
526 Ops.INDEX dtypes.f32 [130, 449]
527 Ops.INDEX dtypes.i32 [361, 303]
528 Ops.CMPEQ dtypes.bool [527, 418]
529 Ops.INDEX dtypes.i32 [361, 420]
530 Ops.CMPEQ dtypes.bool [529, 418]
531 Ops.OR dtypes.bool [528, 530]
532 Ops.INDEX dtypes.i32 [361, 424]
533 Ops.CMPEQ dtypes.bool [532, 418]
534 Ops.OR dtypes.bool [531, 533]
535 Ops.INDEX dtypes.i32 [361, 428]
536 Ops.CMPEQ dtypes.bool [535, 418]
537 Ops.OR dtypes.bool [534, 536]
538 Ops.WHERE dtypes.f32 [530, 383, 384]
539 Ops.WHERE dtypes.f32 [533, 381, 538]
540 Ops.WHERE dtypes.f32 [536, 380, 539]
541 Ops.WHERE dtypes.f32 [537, 540, 200]
542 Ops.CAST dtypes.i32 [541] dtypes.i32
543 Ops.CMPEQ dtypes.bool [542, 338]
544 Ops.PARAM dtypes.f32 [34] ParamArg(slot=5, dtype=dtypes.f32)
545 Ops.INDEX dtypes.f32 [544, 473]
546 Ops.WHERE dtypes.f32 [543, 545, 200]
547 Ops.REDUCE dtypes.f32 [546, 331] (Ops.ADD, 0)
548 Ops.ADD dtypes.f32 [526, 547]
549 Ops.CONST dtypes.weakfloat [] -7.0
550 Ops.MAX dtypes.f32 [548, 549]
551 Ops.CMPLT dtypes.bool [451, 550]
552 Ops.WHERE dtypes.f32 [551, 451, 550]
553 Ops.ADD dtypes.f32 [552, 382]
554 Ops.MUL dtypes.f32 [525, 553]
555 Ops.STORE dtypes.void [450, 554]
556 Ops.END dtypes.void [555, 4, 100, 447]
557 Ops.SINK dtypes.void [556] KernelInfo()
558 Ops.ALLOC dtypes.f32 [6] ParamArg(slot=70, dtype=dtypes.f32, device="CPU")
559 Ops.PARAM dtypes.f32 [34] ParamArg(slot=19, dtype=dtypes.f32, device="CPU")
560 Ops.CALL dtypes.void [557, 558, 483, 512, 408, 347, 559] CallInfo()
561 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
562 Ops.MUL dtypes.weakint [349, 8]
563 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 4)
564 Ops.ADD dtypes.weakint [562, 563]
565 Ops.MUL dtypes.weakint [27, 34]
566 Ops.ADD dtypes.weakint [564, 565]
567 Ops.INDEX dtypes.f32 [561, 566]
568 Ops.RANGE dtypes.weakint [84] (AxisType.WEAK, 1)
569 Ops.ADD dtypes.weakint [350, 568]
570 Ops.ADD dtypes.weakint [569, 38]
571 Ops.INDEX dtypes.f32 [7, 570]
572 Ops.INDEX dtypes.i32 [453, 362]
573 Ops.CMPEQ dtypes.bool [572, 365]
574 Ops.INDEX dtypes.i32 [453, 367]
575 Ops.CMPEQ dtypes.bool [574, 365]
576 Ops.OR dtypes.bool [573, 575]
577 Ops.INDEX dtypes.i32 [453, 371]
578 Ops.CMPEQ dtypes.bool [577, 365]
579 Ops.OR dtypes.bool [576, 578]
580 Ops.INDEX dtypes.i32 [453, 376]
581 Ops.CMPEQ dtypes.bool [580, 365]
582 Ops.OR dtypes.bool [579, 581]
583 Ops.WHERE dtypes.f32 [575, 383, 384]
584 Ops.WHERE dtypes.f32 [578, 381, 583]
585 Ops.WHERE dtypes.f32 [581, 380, 584]
586 Ops.WHERE dtypes.f32 [582, 585, 200]
587 Ops.CAST dtypes.i32 [586] dtypes.i32
588 Ops.CMPEQ dtypes.bool [587, 338]
589 Ops.PARAM dtypes.f32 [391] ParamArg(slot=3, dtype=dtypes.f32)
590 Ops.MUL dtypes.weakint [563, 84]
591 Ops.ADD dtypes.weakint [590, 568]
592 Ops.ADD dtypes.weakint [591, 395]
593 Ops.INDEX dtypes.f32 [589, 592]
594 Ops.WHERE dtypes.f32 [588, 593, 200]
595 Ops.REDUCE dtypes.f32 [594, 331] (Ops.ADD, 0)
596 Ops.MUL dtypes.f32 [571, 595]
597 Ops.REDUCE dtypes.f32 [596, 568] (Ops.ADD, 0)
598 Ops.STORE dtypes.void [567, 597]
599 Ops.END dtypes.void [598, 27, 349, 563]
600 Ops.SINK dtypes.void [599] KernelInfo()
601 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=69, dtype=dtypes.f32, device="CPU")
602 Ops.PARAM dtypes.f32 [391] ParamArg(slot=20, dtype=dtypes.f32, device="CPU")
603 Ops.CALL dtypes.void [600, 601, 558, 347, 602] CallInfo()
604 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 3)
605 Ops.ADD dtypes.weakint [38, 604]
606 Ops.INDEX dtypes.f32 [26, 605]
607 Ops.INDEX dtypes.f32 [7, 605]
608 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
609 Ops.MUL dtypes.weakint [71, 8]
610 Ops.ADD dtypes.weakint [609, 604]
611 Ops.ADD dtypes.weakint [610, 565]
612 Ops.INDEX dtypes.f32 [608, 611]
613 Ops.PARAM dtypes.i32 [84] ParamArg(slot=3, dtype=dtypes.i32)
614 Ops.INDEX dtypes.i32 [613, 362]
615 Ops.CAST dtypes.i32 [71] dtypes.i32
616 Ops.ADD dtypes.i32 [615, 70]
617 Ops.CMPEQ dtypes.bool [614, 616]
618 Ops.INDEX dtypes.i32 [613, 367]
619 Ops.CMPEQ dtypes.bool [618, 616]
620 Ops.OR dtypes.bool [617, 619]
621 Ops.INDEX dtypes.i32 [613, 371]
622 Ops.CMPEQ dtypes.bool [621, 616]
623 Ops.OR dtypes.bool [620, 622]
624 Ops.INDEX dtypes.i32 [613, 376]
625 Ops.CMPEQ dtypes.bool [624, 616]
626 Ops.OR dtypes.bool [623, 625]
627 Ops.WHERE dtypes.f32 [619, 383, 384]
628 Ops.WHERE dtypes.f32 [622, 381, 627]
629 Ops.WHERE dtypes.f32 [625, 380, 628]
630 Ops.WHERE dtypes.f32 [626, 629, 200]
631 Ops.CAST dtypes.i32 [630] dtypes.i32
632 Ops.CMPEQ dtypes.bool [631, 338]
633 Ops.MUL dtypes.weakint [331, 8]
634 Ops.ADD dtypes.weakint [633, 604]
635 Ops.INDEX dtypes.f32 [311, 634]
636 Ops.WHERE dtypes.f32 [632, 635, 200]
637 Ops.REDUCE dtypes.f32 [636, 331] (Ops.ADD, 0)
638 Ops.ADD dtypes.f32 [612, 637]
639 Ops.PARAM dtypes.f32 [28] ParamArg(slot=5, dtype=dtypes.f32)
640 Ops.MUL dtypes.weakint [27, 70]
641 Ops.ADD dtypes.weakint [640, 71]
642 Ops.INDEX dtypes.f32 [639, 641]
643 Ops.PARAM dtypes.f32 [2] ParamArg(slot=6, dtype=dtypes.f32)
644 Ops.INDEX dtypes.f32 [643, 27]
645 Ops.MUL dtypes.f32 [644, 123]
646 Ops.ADD dtypes.f32 [642, 645]
647 Ops.MUL dtypes.f32 [646, 242]
648 Ops.EXP2 dtypes.f32 [647]
649 Ops.PARAM dtypes.f32 [2] ParamArg(slot=7, dtype=dtypes.f32)
650 Ops.INDEX dtypes.f32 [649, 27]
651 Ops.RECIP dtypes.f32 [650]
652 Ops.MUL dtypes.f32 [648, 651]
653 Ops.MUL dtypes.f32 [638, 652]
654 Ops.REDUCE dtypes.f32 [653, 71] (Ops.ADD, 0)
655 Ops.ADD dtypes.f32 [607, 654]
656 Ops.STORE dtypes.void [606, 655]
657 Ops.END dtypes.void [656, 27, 604]
658 Ops.SINK dtypes.void [657] KernelInfo()
659 Ops.PARAM dtypes.f32 [6] ParamArg(slot=22, dtype=dtypes.f32, device="CPU")
660 Ops.PARAM dtypes.f32 [68] ParamArg(slot=21, dtype=dtypes.f32, device="CPU")
661 Ops.CALL dtypes.void [658, 659, 297, 601, 347, 660, 444, 494, 522] CallInfo()
662 Ops.LINEAR dtypes.void [25, 67, 94, 160, 222, 237, 250, 271, 300, 302, 327, 348, 407, 410, 445, 485, 495, 513, 523, 560, 603, 661]
663 Ops.PARAM dtypes.f32 [6] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
664 Ops.STACK dtypes.weakint [106, 2, 8]
665 Ops.RESHAPE dtypes.f32 [663, 664]
666 Ops.PARAM dtypes.f32 [8] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
667 Ops.PARAM dtypes.f32 [48] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
668 Ops.PARAM dtypes.f32 [8] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
669 Ops.PARAM dtypes.f32 [68] ParamArg(slot=5, dtype=dtypes.f32, device="CPU")
670 Ops.PARAM dtypes.f32 [75] ParamArg(slot=7, dtype=dtypes.f32, device="CPU")
671 Ops.PARAM dtypes.f32 [84] ParamArg(slot=8, dtype=dtypes.f32, device="CPU")
672 Ops.PARAM dtypes.f32 [75] ParamArg(slot=9, dtype=dtypes.f32, device="CPU")
673 Ops.PARAM dtypes.f32 [84] ParamArg(slot=10, dtype=dtypes.f32, device="CPU")
674 Ops.PARAM dtypes.f32 [32] ParamArg(slot=11, dtype=dtypes.f32, device="CPU")
675 Ops.PARAM dtypes.f32 [48] ParamArg(slot=12, dtype=dtypes.f32, device="CPU")
676 Ops.PARAM dtypes.f32 [8] ParamArg(slot=14, dtype=dtypes.f32, device="CPU")
677 Ops.PARAM dtypes.f32 [68] ParamArg(slot=15, dtype=dtypes.f32, device="CPU")
678 Ops.PARAM dtypes.f32 [32] ParamArg(slot=16, dtype=dtypes.f32, device="CPU")
679 Ops.PARAM dtypes.f32 [391] ParamArg(slot=17, dtype=dtypes.f32, device="CPU")
680 Ops.PARAM dtypes.f32 [34] ParamArg(slot=18, dtype=dtypes.f32, device="CPU")
681 Ops.PARAM dtypes.f32 [391] ParamArg(slot=19, dtype=dtypes.f32, device="CPU")
682 Ops.PARAM dtypes.f32 [34] ParamArg(slot=20, dtype=dtypes.f32, device="CPU")
683 Ops.PARAM dtypes.f32 [391] ParamArg(slot=21, dtype=dtypes.f32, device="CPU")
684 Ops.PARAM dtypes.f32 [68] ParamArg(slot=22, dtype=dtypes.f32, device="CPU")
685 Ops.RESHAPE dtypes.f32 [24, 664]
686 Ops.CALL dtypes.void [662, 665, 666, 667, 668, 669, 1, 670, 671, 672, 673, 674, 675, 324, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685] CallInfo(name="FFNBlock.__call__.<locals>._run", precompile=True)
687 Ops.AFTER dtypes.f16 [1, 686]
688 Ops.AFTER dtypes.f32 [24, 686]
689 Ops.SINK dtypes.void [687, 688]

