# tolk D89 golden=9e0fa7982822ff14f6ba1f1b371aac59
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 65536
2 Ops.PARAM dtypes.f16 [1] ParamArg(slot=0, dtype=dtypes.f16)
3 Ops.CONST dtypes.weakint [] 32
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.SPECIAL dtypes.i32 [4] "lidx0"
6 Ops.CONST dtypes.weakint [] 4
7 Ops.CAST dtypes.i32 [6] dtypes.i32
8 Ops.SHR dtypes.i32 [5, 7]
9 Ops.CONST dtypes.weakint [] 8
10 Ops.CAST dtypes.i32 [9] dtypes.i32
11 Ops.SHL dtypes.i32 [8, 10]
12 Ops.CONST dtypes.weakint [] 16
13 Ops.CAST dtypes.i32 [12] dtypes.i32
14 Ops.SPECIAL dtypes.i32 [13] "gidx1"
15 Ops.CONST dtypes.weakint [] 12
16 Ops.CAST dtypes.i32 [15] dtypes.i32
17 Ops.SHL dtypes.i32 [14, 16]
18 Ops.ADD dtypes.i32 [11, 17]
19 Ops.SPECIAL dtypes.i32 [13] "gidx0"
20 Ops.SHL dtypes.i32 [19, 7]
21 Ops.CONST dtypes.weakint [] 15
22 Ops.CAST dtypes.i32 [21] dtypes.i32
23 Ops.AND dtypes.i32 [5, 22]
24 Ops.ADD dtypes.i32 [20, 23]
25 Ops.ADD dtypes.i32 [18, 24]
26 Ops.INDEX dtypes.f16 [2, 25]
27 Ops.ALLOC dtypes.f32 [9] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
28 Ops.AFTER dtypes.f32 [27, 5, 14, 19]
29 Ops.CONST dtypes.weakint [] 0
30 Ops.CAST dtypes.i32 [29] dtypes.i32
31 Ops.INDEX dtypes.f32 [28, 30]
32 Ops.CONST dtypes.weakfloat [] 0.0
33 Ops.CAST dtypes.f32 [32] dtypes.f32
34 Ops.STORE dtypes.void [31, 33]
35 Ops.CONST dtypes.weakint [] 1
36 Ops.CAST dtypes.i32 [35] dtypes.i32
37 Ops.INDEX dtypes.f32 [28, 36]
38 Ops.STORE dtypes.void [37, 33]
39 Ops.CONST dtypes.weakint [] 2
40 Ops.CAST dtypes.i32 [39] dtypes.i32
41 Ops.INDEX dtypes.f32 [28, 40]
42 Ops.STORE dtypes.void [41, 33]
43 Ops.CONST dtypes.weakint [] 3
44 Ops.CAST dtypes.i32 [43] dtypes.i32
45 Ops.INDEX dtypes.f32 [28, 44]
46 Ops.STORE dtypes.void [45, 33]
47 Ops.INDEX dtypes.f32 [28, 7]
48 Ops.STORE dtypes.void [47, 33]
49 Ops.CONST dtypes.weakint [] 5
50 Ops.CAST dtypes.i32 [49] dtypes.i32
51 Ops.INDEX dtypes.f32 [28, 50]
52 Ops.STORE dtypes.void [51, 33]
53 Ops.CONST dtypes.weakint [] 6
54 Ops.CAST dtypes.i32 [53] dtypes.i32
55 Ops.INDEX dtypes.f32 [28, 54]
56 Ops.STORE dtypes.void [55, 33]
57 Ops.CONST dtypes.weakint [] 7
58 Ops.CAST dtypes.i32 [57] dtypes.i32
59 Ops.INDEX dtypes.f32 [28, 58]
60 Ops.STORE dtypes.void [59, 33]
61 Ops.RANGE dtypes.i32 [13] (AxisType.WEAK, 0)
62 Ops.AFTER dtypes.f32 [27, 34, 38, 42, 46, 48, 52, 56, 60, 61]
63 Ops.INDEX dtypes.f32 [62, 30]
64 Ops.PARAM dtypes.f16 [1] ParamArg(slot=1, dtype=dtypes.f16)
65 Ops.SHL dtypes.i32 [23, 10]
66 Ops.ADD dtypes.i32 [65, 17]
67 Ops.SHL dtypes.i32 [61, 7]
68 Ops.ADD dtypes.i32 [66, 67]
69 Ops.SHRINK dtypes.f16 [64, 68, 7]
70 Ops.LOAD dtypes.f16 [69]
71 Ops.INDEX dtypes.f16 [70, 30]
72 Ops.INDEX dtypes.f16 [70, 36]
73 Ops.INDEX dtypes.f16 [70, 40]
74 Ops.INDEX dtypes.f16 [70, 44]
75 Ops.ADD dtypes.i32 [68, 7]
76 Ops.SHRINK dtypes.f16 [64, 75, 7]
77 Ops.LOAD dtypes.f16 [76]
78 Ops.INDEX dtypes.f16 [77, 30]
79 Ops.INDEX dtypes.f16 [77, 36]
80 Ops.INDEX dtypes.f16 [77, 40]
81 Ops.INDEX dtypes.f16 [77, 44]
82 Ops.ADD dtypes.i32 [68, 10]
83 Ops.SHRINK dtypes.f16 [64, 82, 7]
84 Ops.LOAD dtypes.f16 [83]
85 Ops.INDEX dtypes.f16 [84, 30]
86 Ops.INDEX dtypes.f16 [84, 36]
87 Ops.INDEX dtypes.f16 [84, 40]
88 Ops.INDEX dtypes.f16 [84, 44]
89 Ops.ADD dtypes.i32 [68, 16]
90 Ops.SHRINK dtypes.f16 [64, 89, 7]
91 Ops.LOAD dtypes.f16 [90]
92 Ops.INDEX dtypes.f16 [91, 30]
93 Ops.INDEX dtypes.f16 [91, 36]
94 Ops.INDEX dtypes.f16 [91, 40]
95 Ops.INDEX dtypes.f16 [91, 44]
96 Ops.STACK dtypes.f16 [71, 72, 73, 74, 78, 79, 80, 81, 85, 86, 87, 88, 92, 93, 94, 95]
97 Ops.PARAM dtypes.f16 [1] ParamArg(slot=2, dtype=dtypes.f16)
98 Ops.SHL dtypes.i32 [61, 16]
99 Ops.ADD dtypes.i32 [24, 98]
100 Ops.INDEX dtypes.f16 [97, 99]
101 Ops.LOAD dtypes.f16 [100]
102 Ops.CONST dtypes.weakint [] 256
103 Ops.CAST dtypes.i32 [102] dtypes.i32
104 Ops.ADD dtypes.i32 [99, 103]
105 Ops.INDEX dtypes.f16 [97, 104]
106 Ops.LOAD dtypes.f16 [105]
107 Ops.CONST dtypes.weakint [] 512
108 Ops.CAST dtypes.i32 [107] dtypes.i32
109 Ops.ADD dtypes.i32 [99, 108]
110 Ops.INDEX dtypes.f16 [97, 109]
111 Ops.LOAD dtypes.f16 [110]
112 Ops.CONST dtypes.weakint [] 768
113 Ops.CAST dtypes.i32 [112] dtypes.i32
114 Ops.ADD dtypes.i32 [99, 113]
115 Ops.INDEX dtypes.f16 [97, 114]
116 Ops.LOAD dtypes.f16 [115]
117 Ops.CONST dtypes.weakint [] 1024
118 Ops.CAST dtypes.i32 [117] dtypes.i32
119 Ops.ADD dtypes.i32 [99, 118]
120 Ops.INDEX dtypes.f16 [97, 119]
121 Ops.LOAD dtypes.f16 [120]
122 Ops.CONST dtypes.weakint [] 1280
123 Ops.CAST dtypes.i32 [122] dtypes.i32
124 Ops.ADD dtypes.i32 [99, 123]
125 Ops.INDEX dtypes.f16 [97, 124]
126 Ops.LOAD dtypes.f16 [125]
127 Ops.CONST dtypes.weakint [] 1536
128 Ops.CAST dtypes.i32 [127] dtypes.i32
129 Ops.ADD dtypes.i32 [99, 128]
130 Ops.INDEX dtypes.f16 [97, 129]
131 Ops.LOAD dtypes.f16 [130]
132 Ops.CONST dtypes.weakint [] 1792
133 Ops.CAST dtypes.i32 [132] dtypes.i32
134 Ops.ADD dtypes.i32 [99, 133]
135 Ops.INDEX dtypes.f16 [97, 134]
136 Ops.LOAD dtypes.f16 [135]
137 Ops.CONST dtypes.weakint [] 2048
138 Ops.CAST dtypes.i32 [137] dtypes.i32
139 Ops.ADD dtypes.i32 [99, 138]
140 Ops.INDEX dtypes.f16 [97, 139]
141 Ops.LOAD dtypes.f16 [140]
142 Ops.CONST dtypes.weakint [] 2304
143 Ops.CAST dtypes.i32 [142] dtypes.i32
144 Ops.ADD dtypes.i32 [99, 143]
145 Ops.INDEX dtypes.f16 [97, 144]
146 Ops.LOAD dtypes.f16 [145]
147 Ops.CONST dtypes.weakint [] 2560
148 Ops.CAST dtypes.i32 [147] dtypes.i32
149 Ops.ADD dtypes.i32 [99, 148]
150 Ops.INDEX dtypes.f16 [97, 149]
151 Ops.LOAD dtypes.f16 [150]
152 Ops.CONST dtypes.weakint [] 2816
153 Ops.CAST dtypes.i32 [152] dtypes.i32
154 Ops.ADD dtypes.i32 [99, 153]
155 Ops.INDEX dtypes.f16 [97, 154]
156 Ops.LOAD dtypes.f16 [155]
157 Ops.CONST dtypes.weakint [] 3072
158 Ops.CAST dtypes.i32 [157] dtypes.i32
159 Ops.ADD dtypes.i32 [99, 158]
160 Ops.INDEX dtypes.f16 [97, 159]
161 Ops.LOAD dtypes.f16 [160]
162 Ops.CONST dtypes.weakint [] 3328
163 Ops.CAST dtypes.i32 [162] dtypes.i32
164 Ops.ADD dtypes.i32 [99, 163]
165 Ops.INDEX dtypes.f16 [97, 164]
166 Ops.LOAD dtypes.f16 [165]
167 Ops.CONST dtypes.weakint [] 3584
168 Ops.CAST dtypes.i32 [167] dtypes.i32
169 Ops.ADD dtypes.i32 [99, 168]
170 Ops.INDEX dtypes.f16 [97, 169]
171 Ops.LOAD dtypes.f16 [170]
172 Ops.CONST dtypes.weakint [] 3840
173 Ops.CAST dtypes.i32 [172] dtypes.i32
174 Ops.ADD dtypes.i32 [99, 173]
175 Ops.INDEX dtypes.f16 [97, 174]
176 Ops.LOAD dtypes.f16 [175]
177 Ops.STACK dtypes.f16 [101, 106, 111, 116, 121, 126, 131, 136, 141, 146, 151, 156, 161, 166, 171, 176]
178 Ops.LOAD dtypes.f32 [63]
179 Ops.INDEX dtypes.f32 [62, 36]
180 Ops.LOAD dtypes.f32 [179]
181 Ops.INDEX dtypes.f32 [62, 40]
182 Ops.LOAD dtypes.f32 [181]
183 Ops.INDEX dtypes.f32 [62, 44]
184 Ops.LOAD dtypes.f32 [183]
185 Ops.INDEX dtypes.f32 [62, 7]
186 Ops.LOAD dtypes.f32 [185]
187 Ops.INDEX dtypes.f32 [62, 50]
188 Ops.LOAD dtypes.f32 [187]
189 Ops.INDEX dtypes.f32 [62, 54]
190 Ops.LOAD dtypes.f32 [189]
191 Ops.INDEX dtypes.f32 [62, 58]
192 Ops.LOAD dtypes.f32 [191]
193 Ops.STACK dtypes.f32 [178, 180, 182, 184, 186, 188, 190, 192]
194 Ops.WMMA dtypes.f32 [96, 177, 193] ((16, 16, 16), dtypes.f16, 32, None)
195 Ops.INDEX dtypes.f32 [194, 30]
196 Ops.STORE dtypes.void [63, 195]
197 Ops.INDEX dtypes.f32 [194, 36]
198 Ops.STORE dtypes.void [179, 197]
199 Ops.INDEX dtypes.f32 [194, 40]
200 Ops.STORE dtypes.void [181, 199]
201 Ops.INDEX dtypes.f32 [194, 44]
202 Ops.STORE dtypes.void [183, 201]
203 Ops.INDEX dtypes.f32 [194, 7]
204 Ops.STORE dtypes.void [185, 203]
205 Ops.INDEX dtypes.f32 [194, 50]
206 Ops.STORE dtypes.void [187, 205]
207 Ops.INDEX dtypes.f32 [194, 54]
208 Ops.STORE dtypes.void [189, 207]
209 Ops.INDEX dtypes.f32 [194, 58]
210 Ops.STORE dtypes.void [191, 209]
211 Ops.AFTER dtypes.void [0, 196, 198, 200, 202, 204, 206, 208, 210]
212 Ops.END dtypes.void [211, 61]
213 Ops.AFTER dtypes.f32 [27, 212]
214 Ops.INDEX dtypes.f32 [213, 30]
215 Ops.LOAD dtypes.f32 [214]
216 Ops.CAST dtypes.f16 [215] dtypes.f16
217 Ops.STORE dtypes.void [26, 216]
218 Ops.ADD dtypes.i32 [25, 108]
219 Ops.INDEX dtypes.f16 [2, 218]
220 Ops.INDEX dtypes.f32 [213, 36]
221 Ops.LOAD dtypes.f32 [220]
222 Ops.CAST dtypes.f16 [221] dtypes.f16
223 Ops.STORE dtypes.void [219, 222]
224 Ops.ADD dtypes.i32 [25, 118]
225 Ops.INDEX dtypes.f16 [2, 224]
226 Ops.INDEX dtypes.f32 [213, 40]
227 Ops.LOAD dtypes.f32 [226]
228 Ops.CAST dtypes.f16 [227] dtypes.f16
229 Ops.STORE dtypes.void [225, 228]
230 Ops.ADD dtypes.i32 [25, 128]
231 Ops.INDEX dtypes.f16 [2, 230]
232 Ops.INDEX dtypes.f32 [213, 44]
233 Ops.LOAD dtypes.f32 [232]
234 Ops.CAST dtypes.f16 [233] dtypes.f16
235 Ops.STORE dtypes.void [231, 234]
236 Ops.ADD dtypes.i32 [25, 138]
237 Ops.INDEX dtypes.f16 [2, 236]
238 Ops.INDEX dtypes.f32 [213, 7]
239 Ops.LOAD dtypes.f32 [238]
240 Ops.CAST dtypes.f16 [239] dtypes.f16
241 Ops.STORE dtypes.void [237, 240]
242 Ops.ADD dtypes.i32 [25, 148]
243 Ops.INDEX dtypes.f16 [2, 242]
244 Ops.INDEX dtypes.f32 [213, 50]
245 Ops.LOAD dtypes.f32 [244]
246 Ops.CAST dtypes.f16 [245] dtypes.f16
247 Ops.STORE dtypes.void [243, 246]
248 Ops.ADD dtypes.i32 [25, 158]
249 Ops.INDEX dtypes.f16 [2, 248]
250 Ops.INDEX dtypes.f32 [213, 54]
251 Ops.LOAD dtypes.f32 [250]
252 Ops.CAST dtypes.f16 [251] dtypes.f16
253 Ops.STORE dtypes.void [249, 252]
254 Ops.ADD dtypes.i32 [25, 168]
255 Ops.INDEX dtypes.f16 [2, 254]
256 Ops.INDEX dtypes.f32 [213, 58]
257 Ops.LOAD dtypes.f32 [256]
258 Ops.CAST dtypes.f16 [257] dtypes.f16
259 Ops.STORE dtypes.void [255, 258]
260 Ops.AFTER dtypes.void [0, 217, 223, 229, 235, 241, 247, 253, 259]
261 Ops.SINK dtypes.void [260] KernelInfo(name="r_16_16_32_2_2_2_2_16", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), optimized=True)
262 Ops.BUFFER dtypes.f32 [9] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
263 Ops.AFTER dtypes.f32 [262, 5, 14, 19]
264 Ops.INDEX dtypes.f32 [263, 30]
265 Ops.STORE dtypes.void [264, 33]
266 Ops.INDEX dtypes.f32 [263, 36]
267 Ops.STORE dtypes.void [266, 33]
268 Ops.INDEX dtypes.f32 [263, 40]
269 Ops.STORE dtypes.void [268, 33]
270 Ops.INDEX dtypes.f32 [263, 44]
271 Ops.STORE dtypes.void [270, 33]
272 Ops.INDEX dtypes.f32 [263, 7]
273 Ops.STORE dtypes.void [272, 33]
274 Ops.INDEX dtypes.f32 [263, 50]
275 Ops.STORE dtypes.void [274, 33]
276 Ops.INDEX dtypes.f32 [263, 54]
277 Ops.STORE dtypes.void [276, 33]
278 Ops.INDEX dtypes.f32 [263, 58]
279 Ops.STORE dtypes.void [278, 33]
280 Ops.AFTER dtypes.f32 [262, 265, 267, 269, 271, 273, 275, 277, 279, 61]
281 Ops.INDEX dtypes.f32 [280, 30]
282 Ops.LOAD dtypes.f32 [281]
283 Ops.INDEX dtypes.f32 [280, 36]
284 Ops.LOAD dtypes.f32 [283]
285 Ops.INDEX dtypes.f32 [280, 40]
286 Ops.LOAD dtypes.f32 [285]
287 Ops.INDEX dtypes.f32 [280, 44]
288 Ops.LOAD dtypes.f32 [287]
289 Ops.INDEX dtypes.f32 [280, 7]
290 Ops.LOAD dtypes.f32 [289]
291 Ops.INDEX dtypes.f32 [280, 50]
292 Ops.LOAD dtypes.f32 [291]
293 Ops.INDEX dtypes.f32 [280, 54]
294 Ops.LOAD dtypes.f32 [293]
295 Ops.INDEX dtypes.f32 [280, 58]
296 Ops.LOAD dtypes.f32 [295]
297 Ops.STACK dtypes.f32 [282, 284, 286, 288, 290, 292, 294, 296]
298 Ops.WMMA dtypes.f32 [96, 177, 297] ((16, 16, 16), dtypes.f16, 32, None)
299 Ops.INDEX dtypes.f32 [298, 30]
300 Ops.STORE dtypes.void [281, 299]
301 Ops.INDEX dtypes.f32 [298, 36]
302 Ops.STORE dtypes.void [283, 301]
303 Ops.INDEX dtypes.f32 [298, 40]
304 Ops.STORE dtypes.void [285, 303]
305 Ops.INDEX dtypes.f32 [298, 44]
306 Ops.STORE dtypes.void [287, 305]
307 Ops.INDEX dtypes.f32 [298, 7]
308 Ops.STORE dtypes.void [289, 307]
309 Ops.INDEX dtypes.f32 [298, 50]
310 Ops.STORE dtypes.void [291, 309]
311 Ops.INDEX dtypes.f32 [298, 54]
312 Ops.STORE dtypes.void [293, 311]
313 Ops.INDEX dtypes.f32 [298, 58]
314 Ops.STORE dtypes.void [295, 313]
315 Ops.AFTER dtypes.void [0, 300, 302, 304, 306, 308, 310, 312, 314]
316 Ops.END dtypes.void [315, 61]
317 Ops.AFTER dtypes.f32 [262, 316]
318 Ops.INDEX dtypes.f32 [317, 30]
319 Ops.LOAD dtypes.f32 [318]
320 Ops.CAST dtypes.f16 [319] dtypes.f16
321 Ops.STORE dtypes.void [26, 320]
322 Ops.INDEX dtypes.f32 [317, 36]
323 Ops.LOAD dtypes.f32 [322]
324 Ops.CAST dtypes.f16 [323] dtypes.f16
325 Ops.STORE dtypes.void [219, 324]
326 Ops.INDEX dtypes.f32 [317, 40]
327 Ops.LOAD dtypes.f32 [326]
328 Ops.CAST dtypes.f16 [327] dtypes.f16
329 Ops.STORE dtypes.void [225, 328]
330 Ops.INDEX dtypes.f32 [317, 44]
331 Ops.LOAD dtypes.f32 [330]
332 Ops.CAST dtypes.f16 [331] dtypes.f16
333 Ops.STORE dtypes.void [231, 332]
334 Ops.INDEX dtypes.f32 [317, 7]
335 Ops.LOAD dtypes.f32 [334]
336 Ops.CAST dtypes.f16 [335] dtypes.f16
337 Ops.STORE dtypes.void [237, 336]
338 Ops.INDEX dtypes.f32 [317, 50]
339 Ops.LOAD dtypes.f32 [338]
340 Ops.CAST dtypes.f16 [339] dtypes.f16
341 Ops.STORE dtypes.void [243, 340]
342 Ops.INDEX dtypes.f32 [317, 54]
343 Ops.LOAD dtypes.f32 [342]
344 Ops.CAST dtypes.f16 [343] dtypes.f16
345 Ops.STORE dtypes.void [249, 344]
346 Ops.INDEX dtypes.f32 [317, 58]
347 Ops.LOAD dtypes.f32 [346]
348 Ops.CAST dtypes.f16 [347] dtypes.f16
349 Ops.STORE dtypes.void [255, 348]
350 Ops.AFTER dtypes.void [0, 321, 325, 329, 333, 337, 341, 345, 349]
351 Ops.SINK dtypes.void [350] KernelInfo(name="r_16_16_32_2_2_2_2_16", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), estimates=Estimates(ops=33554432, lds=8519680, mem=393216), optimized=True)
352 Ops.LINEAR dtypes.void [1, 2, 64, 97, 9, 262, 12, 13, 19, 14, 3, 4, 5, 263, 0, 29, 30, 264, 35, 36, 266, 39, 40, 268, 43, 44, 270, 6, 7, 272, 49, 50, 274, 53, 54, 276, 57, 58, 278, 32, 33, 117, 118, 15, 16, 122, 123, 21, 22, 127, 128, 132, 133, 137, 138, 142, 143, 102, 103, 147, 148, 152, 153, 157, 158, 162, 163, 167, 168, 172, 173, 107, 108, 112, 113, 10, 20, 23, 24, 17, 65, 66, 265, 267, 269, 271, 273, 275, 277, 279, 61, 98, 99, 119, 120, 121, 124, 125, 126, 129, 130, 131, 134, 135, 136, 139, 140, 141, 144, 145, 146, 104, 105, 106, 149, 150, 151, 154, 155, 156, 159, 160, 161, 164, 165, 166, 169, 170, 171, 174, 175, 176, 109, 110, 111, 114, 115, 116, 100, 101, 67, 68, 89, 90, 91, 75, 76, 77, 82, 83, 84, 69, 70, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 92, 93, 94, 95, 78, 79, 80, 81, 85, 86, 87, 88, 71, 72, 73, 74, 96, 177, 297, 298, 299, 301, 303, 305, 307, 309, 311, 313, 300, 302, 304, 306, 308, 310, 312, 314, 315, 316, 317, 318, 319, 322, 323, 326, 327, 330, 331, 334, 335, 338, 339, 342, 343, 346, 347, 8, 11, 18, 25, 224, 225, 230, 231, 236, 237, 242, 243, 248, 249, 254, 255, 218, 219, 26, 320, 324, 328, 332, 336, 340, 344, 348, 329, 333, 337, 341, 345, 349, 325, 321, 350, 351]
353 Ops.SOURCE dtypes.void [] "typedef long unsigned int size_t;\n#define half _Float16\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_local_size(unsigned int);\ntypedef half half4 __attribute__((ext_vector_type(4)));\nstatic inline __attribute__((device)) half4 make_half4(half x, half y, half z, half w) { return { x, y, z, w }; }\ntypedef half half16 __attribute__((ext_vector_type(16)));\nstatic inline __attribute__((device)) half16 make_half16(half x, half y, half z, half w, half a, half b, half c, half d, half e, half f, half g, half h, half i, half j, half k, half l) { return { x, y, z, w, a, b, c, d, e, f, g, h, i, j, k, l }; }\ntypedef float float8 __attribute__((ext_vector_type(8)));\nstatic inline __attribute__((device)) float8 make_float8(float x, float y, float z, float w, float a, float b, float c, float d) { return { x, y, z, w, a, b, c, d }; }\n#define __WMMA_16_16_16_f16_f32 __builtin_amdgcn_wmma_f32_16x16x16_f16_w32\nextern \"C\" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 32))) r_16_16_32_2_2_2_2_16(half* data0_65536, half* data1_65536, half* data2_65536) {\n  float buf0[8];\n  int gidx0 = __ockl_get_group_id(0); /* 16 */\n  int gidx1 = __ockl_get_group_id(1); /* 16 */\n  int lidx0 = __ockl_get_local_id(0); /* 32 */\n  int alu0 = (lidx0&15);\n  int alu1 = ((gidx0<<4)+alu0);\n  int alu2 = (gidx1<<12);\n  *(buf0+0) = 0.0f;\n  *(buf0+1) = 0.0f;\n  *(buf0+2) = 0.0f;\n  *(buf0+3) = 0.0f;\n  *(buf0+4) = 0.0f;\n  *(buf0+5) = 0.0f;\n  *(buf0+6) = 0.0f;\n  *(buf0+7) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 16; Lidx0++) {\n    int alu11 = (alu1+(Lidx0<<12));\n    half val0 = (*(data2_65536+(alu11+1024)));\n    half val1 = (*(data2_65536+(alu11+1280)));\n    half val2 = (*(data2_65536+(alu11+1536)));\n    half val3 = (*(data2_65536+(alu11+1792)));\n    half val4 = (*(data2_65536+(alu11+2048)));\n    half val5 = (*(data2_65536+(alu11+2304)));\n    half val6 = (*(data2_65536+(alu11+256)));\n    half val7 = (*(data2_65536+(alu11+2560)));\n    half val8 = (*(data2_65536+(alu11+2816)));\n    half val9 = (*(data2_65536+(alu11+3072)));\n    half val10 = (*(data2_65536+(alu11+3328)));\n    half val11 = (*(data2_65536+(alu11+3584)));\n    half val12 = (*(data2_65536+(alu11+3840)));\n    half val13 = (*(data2_65536+(alu11+512)));\n    half val14 = (*(data2_65536+(alu11+768)));\n    half val15 = (*(data2_65536+alu11));\n    int alu12 = ((alu0<<8)+alu2+(Lidx0<<4));\n    half4 val16 = (*((half4*)((data1_65536+(alu12+12)))));\n    half4 val17 = (*((half4*)((data1_65536+(alu12+4)))));\n    half4 val18 = (*((half4*)((data1_65536+(alu12+8)))));\n    half4 val19 = (*((half4*)((data1_65536+alu12))));\n    float8 wmma0 = __WMMA_16_16_16_f16_f32(make_half16(val19.x,val19.y,val19.z,val19.w,val17.x,val17.y,val17.z,val17.w,val18.x,val18.y,val18.z,val18.w,val16.x,val16.y,val16.z,val16.w), make_half16(val15,val6,val13,val14,val0,val1,val2,val3,val4,val5,val7,val8,val9,val10,val11,val12), make_float8((*(buf0+0)),(*(buf0+1)),(*(buf0+2)),(*(buf0+3)),(*(buf0+4)),(*(buf0+5)),(*(buf0+6)),(*(buf0+7))));\n    *(buf0+0) = wmma0[0];\n    *(buf0+1) = wmma0[1];\n    *(buf0+2) = wmma0[2];\n    *(buf0+3) = wmma0[3];\n    *(buf0+4) = wmma0[4];\n    *(buf0+5) = wmma0[5];\n    *(buf0+6) = wmma0[6];\n    *(buf0+7) = wmma0[7];\n  }\n  int alu22 = (((lidx0>>4)<<8)+alu2+alu1);\n  *(data0_65536+(alu22+1024)) = ((half)((*(buf0+2))));\n  *(data0_65536+(alu22+1536)) = ((half)((*(buf0+3))));\n  *(data0_65536+(alu22+2048)) = ((half)((*(buf0+4))));\n  *(data0_65536+(alu22+2560)) = ((half)((*(buf0+5))));\n  *(data0_65536+(alu22+3072)) = ((half)((*(buf0+6))));\n  *(data0_65536+(alu22+3584)) = ((half)((*(buf0+7))));\n  *(data0_65536+(alu22+512)) = ((half)((*(buf0+1))));\n  *(data0_65536+alu22) = ((half)((*(buf0+0))));\n}"
354 Ops.PROGRAM dtypes.void [351, 352, 353] ProgramInfo(global_size=(16, 16, 1), local_size=(32, 1, 1), params=(%2, %64, %97), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="AMD", renderer="HIP", arch="gfx1100"))
355 Ops.SINK dtypes.void [261, 354]

