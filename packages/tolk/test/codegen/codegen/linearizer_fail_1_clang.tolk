# tolk D25 D80 D141 D151 D480 golden=af02fe9945147da462ac35b74cb5f837
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 64
2 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.PARAM dtypes.i32 [0] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 16), name="block_lo", addrspace=AddrSpace.ALU)
4 Ops.PARAM dtypes.i32 [0] ParamArg(slot=4, dtype=dtypes.i32, vmin_vmax=(0, 16), name="block_hi", addrspace=AddrSpace.ALU)
5 Ops.SUB dtypes.i32 [4, 3]
6 Ops.RANGE dtypes.i32 [5] (AxisType.WEAK, 1)
7 Ops.ADD dtypes.i32 [3, 6]
8 Ops.CONST dtypes.weakint [] 2
9 Ops.CAST dtypes.i32 [8] dtypes.i32
10 Ops.SHL dtypes.i32 [7, 9]
11 Ops.CONST dtypes.weakint [] 4
12 Ops.CAST dtypes.i32 [11] dtypes.i32
13 Ops.SHRINK dtypes.f32 [2, 10, 12]
14 Ops.CONST dtypes.weakfloat [] 1.0
15 Ops.CAST dtypes.f32 [14] dtypes.f32
16 Ops.ALLOC dtypes.f32 [11] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
17 Ops.AFTER dtypes.f32 [16, 7]
18 Ops.CONST dtypes.weakint [] 0
19 Ops.CAST dtypes.i32 [18] dtypes.i32
20 Ops.SHRINK dtypes.f32 [17, 19, 12]
21 Ops.CONST dtypes.weakfloat [] 0.0
22 Ops.CAST dtypes.f32 [21] dtypes.f32
23 Ops.STACK dtypes.f32 [22, 22, 22, 22]
24 Ops.STORE dtypes.void [20, 23]
25 Ops.CONST dtypes.weakint [] 10
26 Ops.CAST dtypes.i32 [25] dtypes.i32
27 Ops.AFTER dtypes.i32 [26, 6]
28 Ops.RANGE dtypes.i32 [27] (AxisType.WEAK, 0, 0, 0)
29 Ops.AFTER dtypes.f32 [16, 24, 28]
30 Ops.SHRINK dtypes.f32 [29, 19, 12]
31 Ops.LOAD dtypes.f32 [30]
32 Ops.ALLOC dtypes.f32 [11] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
33 Ops.AFTER dtypes.f32 [32, 7, 28]
34 Ops.SHRINK dtypes.f32 [33, 19, 12]
35 Ops.STORE dtypes.void [34, 23]
36 Ops.CAST dtypes.i32 [1] dtypes.i32
37 Ops.AFTER dtypes.i32 [36, 28]
38 Ops.RANGE dtypes.i32 [37] (AxisType.WEAK, 0, 0, 0, 256)
39 Ops.AFTER dtypes.f32 [32, 35, 38]
40 Ops.SHRINK dtypes.f32 [39, 19, 12]
41 Ops.LOAD dtypes.f32 [40]
42 Ops.INDEX dtypes.f32 [41, 19]
43 Ops.CONST dtypes.weakint [] 163840
44 Ops.PARAM dtypes.f32 [43] ParamArg(slot=1, dtype=dtypes.f32)
45 Ops.CONST dtypes.weakint [] 8
46 Ops.CAST dtypes.i32 [45] dtypes.i32
47 Ops.SHL dtypes.i32 [28, 46]
48 Ops.SHL dtypes.i32 [38, 9]
49 Ops.ADD dtypes.i32 [47, 48]
50 Ops.CONST dtypes.weakint [] 10240
51 Ops.CAST dtypes.i32 [50] dtypes.i32
52 Ops.MUL dtypes.i32 [7, 51]
53 Ops.ADD dtypes.i32 [49, 52]
54 Ops.SHRINK dtypes.f32 [44, 53, 12]
55 Ops.LOAD dtypes.f32 [54]
56 Ops.CONST dtypes.weakint [] 3
57 Ops.CAST dtypes.i32 [56] dtypes.i32
58 Ops.INDEX dtypes.f32 [55, 57]
59 Ops.PARAM dtypes.f32 [1] ParamArg(slot=2, dtype=dtypes.f32)
60 Ops.SHRINK dtypes.f32 [59, 10, 12]
61 Ops.LOAD dtypes.f32 [60]
62 Ops.NEG dtypes.f32 [61]
63 Ops.INDEX dtypes.f32 [62, 19]
64 Ops.ADD dtypes.f32 [58, 63]
65 Ops.INDEX dtypes.f32 [55, 9]
66 Ops.ADD dtypes.f32 [65, 63]
67 Ops.CONST dtypes.weakint [] 1
68 Ops.CAST dtypes.i32 [67] dtypes.i32
69 Ops.INDEX dtypes.f32 [55, 68]
70 Ops.ADD dtypes.f32 [69, 63]
71 Ops.INDEX dtypes.f32 [55, 19]
72 Ops.ADD dtypes.f32 [71, 63]
73 Ops.MUL dtypes.f32 [72, 72]
74 Ops.MULACC dtypes.f32 [70, 70, 73]
75 Ops.MULACC dtypes.f32 [66, 66, 74]
76 Ops.MULACC dtypes.f32 [64, 64, 75]
77 Ops.ADD dtypes.f32 [42, 76]
78 Ops.INDEX dtypes.f32 [41, 68]
79 Ops.CONST dtypes.weakint [] 2560
80 Ops.CAST dtypes.i32 [79] dtypes.i32
81 Ops.ADD dtypes.i32 [53, 80]
82 Ops.SHRINK dtypes.f32 [44, 81, 12]
83 Ops.LOAD dtypes.f32 [82]
84 Ops.INDEX dtypes.f32 [83, 57]
85 Ops.INDEX dtypes.f32 [62, 68]
86 Ops.ADD dtypes.f32 [84, 85]
87 Ops.INDEX dtypes.f32 [83, 9]
88 Ops.ADD dtypes.f32 [87, 85]
89 Ops.INDEX dtypes.f32 [83, 68]
90 Ops.ADD dtypes.f32 [89, 85]
91 Ops.INDEX dtypes.f32 [83, 19]
92 Ops.ADD dtypes.f32 [91, 85]
93 Ops.MUL dtypes.f32 [92, 92]
94 Ops.MULACC dtypes.f32 [90, 90, 93]
95 Ops.MULACC dtypes.f32 [88, 88, 94]
96 Ops.MULACC dtypes.f32 [86, 86, 95]
97 Ops.ADD dtypes.f32 [78, 96]
98 Ops.INDEX dtypes.f32 [41, 9]
99 Ops.CONST dtypes.weakint [] 5120
100 Ops.CAST dtypes.i32 [99] dtypes.i32
101 Ops.ADD dtypes.i32 [53, 100]
102 Ops.SHRINK dtypes.f32 [44, 101, 12]
103 Ops.LOAD dtypes.f32 [102]
104 Ops.INDEX dtypes.f32 [103, 57]
105 Ops.INDEX dtypes.f32 [62, 9]
106 Ops.ADD dtypes.f32 [104, 105]
107 Ops.INDEX dtypes.f32 [103, 9]
108 Ops.ADD dtypes.f32 [107, 105]
109 Ops.INDEX dtypes.f32 [103, 68]
110 Ops.ADD dtypes.f32 [109, 105]
111 Ops.INDEX dtypes.f32 [103, 19]
112 Ops.ADD dtypes.f32 [111, 105]
113 Ops.MUL dtypes.f32 [112, 112]
114 Ops.MULACC dtypes.f32 [110, 110, 113]
115 Ops.MULACC dtypes.f32 [108, 108, 114]
116 Ops.MULACC dtypes.f32 [106, 106, 115]
117 Ops.ADD dtypes.f32 [98, 116]
118 Ops.INDEX dtypes.f32 [41, 57]
119 Ops.CONST dtypes.weakint [] 7680
120 Ops.CAST dtypes.i32 [119] dtypes.i32
121 Ops.ADD dtypes.i32 [53, 120]
122 Ops.SHRINK dtypes.f32 [44, 121, 12]
123 Ops.LOAD dtypes.f32 [122]
124 Ops.INDEX dtypes.f32 [123, 57]
125 Ops.INDEX dtypes.f32 [62, 57]
126 Ops.ADD dtypes.f32 [124, 125]
127 Ops.INDEX dtypes.f32 [123, 9]
128 Ops.ADD dtypes.f32 [127, 125]
129 Ops.INDEX dtypes.f32 [123, 68]
130 Ops.ADD dtypes.f32 [129, 125]
131 Ops.INDEX dtypes.f32 [123, 19]
132 Ops.ADD dtypes.f32 [131, 125]
133 Ops.MUL dtypes.f32 [132, 132]
134 Ops.MULACC dtypes.f32 [130, 130, 133]
135 Ops.MULACC dtypes.f32 [128, 128, 134]
136 Ops.MULACC dtypes.f32 [126, 126, 135]
137 Ops.ADD dtypes.f32 [118, 136]
138 Ops.STACK dtypes.f32 [77, 97, 117, 137]
139 Ops.STORE dtypes.void [40, 138]
140 Ops.AFTER dtypes.void [0, 139]
141 Ops.END dtypes.void [140, 38]
142 Ops.AFTER dtypes.f32 [32, 141]
143 Ops.SHRINK dtypes.f32 [142, 19, 12]
144 Ops.LOAD dtypes.f32 [143]
145 Ops.ADD dtypes.f32 [31, 144]
146 Ops.STORE dtypes.void [30, 145]
147 Ops.AFTER dtypes.void [0, 146]
148 Ops.END dtypes.void [147, 28]
149 Ops.AFTER dtypes.f32 [16, 148]
150 Ops.SHRINK dtypes.f32 [149, 19, 12]
151 Ops.LOAD dtypes.f32 [150]
152 Ops.CONST dtypes.weakfloat [] 0.0003906250058207661
153 Ops.CAST dtypes.f32 [152] dtypes.f32
154 Ops.MUL dtypes.f32 [151, 153]
155 Ops.CONST dtypes.weakfloat [] 9.999999747378752e-06
156 Ops.CAST dtypes.f32 [155] dtypes.f32
157 Ops.ADD dtypes.f32 [154, 156]
158 Ops.SQRT dtypes.f32 [157]
159 Ops.FDIV dtypes.f32 [15, 158]
160 Ops.STORE dtypes.void [13, 159]
161 Ops.AFTER dtypes.void [0, 160]
162 Ops.END dtypes.void [161, 6]
163 Ops.SINK dtypes.void [162] KernelInfo(name="r_16_4_640_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), optimized=True, split=(16, 3, 4))
164 Ops.CONST dtypes.weakint [] 40
165 Ops.PARAM dtypes.u8 [164] ParamArg(slot=0, dtype=dtypes.u8, name="args")
166 Ops.BITCAST dtypes.u64 [165] dtypes.u64
167 Ops.INDEX dtypes.u64 [166, 19]
168 Ops.LOAD dtypes.u64 [167]
169 Ops.REFPARAM dtypes.f32 [1, 168] ParamArg(slot=0, dtype=dtypes.f32)
170 Ops.BITCAST dtypes.i32 [165] dtypes.i32
171 Ops.CONST dtypes.weakint [] 6
172 Ops.CAST dtypes.i32 [171] dtypes.i32
173 Ops.INDEX dtypes.i32 [170, 172]
174 Ops.LOAD dtypes.i32 [173]
175 Ops.CONST dtypes.weakint [] 7
176 Ops.CAST dtypes.i32 [175] dtypes.i32
177 Ops.INDEX dtypes.i32 [170, 176]
178 Ops.LOAD dtypes.i32 [177]
179 Ops.SUB dtypes.i32 [178, 174]
180 Ops.RANGE dtypes.i32 [179] (AxisType.WEAK, 1)
181 Ops.ADD dtypes.i32 [174, 180]
182 Ops.SHL dtypes.i32 [181, 9]
183 Ops.SHRINK dtypes.f32 [169, 182, 12]
184 Ops.BUFFER dtypes.f32 [11] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
185 Ops.AFTER dtypes.f32 [184, 181]
186 Ops.SHRINK dtypes.f32 [185, 19, 12]
187 Ops.STORE dtypes.void [186, 23]
188 Ops.AFTER dtypes.i32 [26, 180]
189 Ops.RANGE dtypes.i32 [188] (AxisType.WEAK, 0, 0, 0)
190 Ops.AFTER dtypes.f32 [184, 187, 189]
191 Ops.SHRINK dtypes.f32 [190, 19, 12]
192 Ops.LOAD dtypes.f32 [191]
193 Ops.BUFFER dtypes.f32 [11] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
194 Ops.AFTER dtypes.f32 [193, 181, 189]
195 Ops.SHRINK dtypes.f32 [194, 19, 12]
196 Ops.STORE dtypes.void [195, 23]
197 Ops.AFTER dtypes.i32 [36, 189]
198 Ops.RANGE dtypes.i32 [197] (AxisType.WEAK, 0, 0, 0, 256)
199 Ops.AFTER dtypes.f32 [193, 196, 198]
200 Ops.SHRINK dtypes.f32 [199, 19, 12]
201 Ops.LOAD dtypes.f32 [200]
202 Ops.INDEX dtypes.f32 [201, 19]
203 Ops.INDEX dtypes.u64 [166, 68]
204 Ops.LOAD dtypes.u64 [203]
205 Ops.REFPARAM dtypes.f32 [43, 204] ParamArg(slot=1, dtype=dtypes.f32)
206 Ops.SHL dtypes.i32 [189, 46]
207 Ops.SHL dtypes.i32 [198, 9]
208 Ops.ADD dtypes.i32 [206, 207]
209 Ops.MUL dtypes.i32 [181, 51]
210 Ops.ADD dtypes.i32 [208, 209]
211 Ops.SHRINK dtypes.f32 [205, 210, 12]
212 Ops.LOAD dtypes.f32 [211]
213 Ops.INDEX dtypes.f32 [212, 57]
214 Ops.INDEX dtypes.u64 [166, 9]
215 Ops.LOAD dtypes.u64 [214]
216 Ops.REFPARAM dtypes.f32 [1, 215] ParamArg(slot=2, dtype=dtypes.f32)
217 Ops.SHRINK dtypes.f32 [216, 182, 12]
218 Ops.LOAD dtypes.f32 [217]
219 Ops.NEG dtypes.f32 [218]
220 Ops.INDEX dtypes.f32 [219, 19]
221 Ops.ADD dtypes.f32 [213, 220]
222 Ops.INDEX dtypes.f32 [212, 9]
223 Ops.ADD dtypes.f32 [222, 220]
224 Ops.INDEX dtypes.f32 [212, 68]
225 Ops.ADD dtypes.f32 [224, 220]
226 Ops.INDEX dtypes.f32 [212, 19]
227 Ops.ADD dtypes.f32 [226, 220]
228 Ops.MUL dtypes.f32 [227, 227]
229 Ops.MULACC dtypes.f32 [225, 225, 228]
230 Ops.MULACC dtypes.f32 [223, 223, 229]
231 Ops.MULACC dtypes.f32 [221, 221, 230]
232 Ops.ADD dtypes.f32 [202, 231]
233 Ops.INDEX dtypes.f32 [201, 68]
234 Ops.ADD dtypes.i32 [210, 80]
235 Ops.SHRINK dtypes.f32 [205, 234, 12]
236 Ops.LOAD dtypes.f32 [235]
237 Ops.INDEX dtypes.f32 [236, 57]
238 Ops.INDEX dtypes.f32 [219, 68]
239 Ops.ADD dtypes.f32 [237, 238]
240 Ops.INDEX dtypes.f32 [236, 9]
241 Ops.ADD dtypes.f32 [240, 238]
242 Ops.INDEX dtypes.f32 [236, 68]
243 Ops.ADD dtypes.f32 [242, 238]
244 Ops.INDEX dtypes.f32 [236, 19]
245 Ops.ADD dtypes.f32 [244, 238]
246 Ops.MUL dtypes.f32 [245, 245]
247 Ops.MULACC dtypes.f32 [243, 243, 246]
248 Ops.MULACC dtypes.f32 [241, 241, 247]
249 Ops.MULACC dtypes.f32 [239, 239, 248]
250 Ops.ADD dtypes.f32 [233, 249]
251 Ops.INDEX dtypes.f32 [201, 9]
252 Ops.ADD dtypes.i32 [210, 100]
253 Ops.SHRINK dtypes.f32 [205, 252, 12]
254 Ops.LOAD dtypes.f32 [253]
255 Ops.INDEX dtypes.f32 [254, 57]
256 Ops.INDEX dtypes.f32 [219, 9]
257 Ops.ADD dtypes.f32 [255, 256]
258 Ops.INDEX dtypes.f32 [254, 9]
259 Ops.ADD dtypes.f32 [258, 256]
260 Ops.INDEX dtypes.f32 [254, 68]
261 Ops.ADD dtypes.f32 [260, 256]
262 Ops.INDEX dtypes.f32 [254, 19]
263 Ops.ADD dtypes.f32 [262, 256]
264 Ops.MUL dtypes.f32 [263, 263]
265 Ops.MULACC dtypes.f32 [261, 261, 264]
266 Ops.MULACC dtypes.f32 [259, 259, 265]
267 Ops.MULACC dtypes.f32 [257, 257, 266]
268 Ops.ADD dtypes.f32 [251, 267]
269 Ops.INDEX dtypes.f32 [201, 57]
270 Ops.ADD dtypes.i32 [210, 120]
271 Ops.SHRINK dtypes.f32 [205, 270, 12]
272 Ops.LOAD dtypes.f32 [271]
273 Ops.INDEX dtypes.f32 [272, 57]
274 Ops.INDEX dtypes.f32 [219, 57]
275 Ops.ADD dtypes.f32 [273, 274]
276 Ops.INDEX dtypes.f32 [272, 9]
277 Ops.ADD dtypes.f32 [276, 274]
278 Ops.INDEX dtypes.f32 [272, 68]
279 Ops.ADD dtypes.f32 [278, 274]
280 Ops.INDEX dtypes.f32 [272, 19]
281 Ops.ADD dtypes.f32 [280, 274]
282 Ops.MUL dtypes.f32 [281, 281]
283 Ops.MULACC dtypes.f32 [279, 279, 282]
284 Ops.MULACC dtypes.f32 [277, 277, 283]
285 Ops.MULACC dtypes.f32 [275, 275, 284]
286 Ops.ADD dtypes.f32 [269, 285]
287 Ops.STACK dtypes.f32 [232, 250, 268, 286]
288 Ops.STORE dtypes.void [200, 287]
289 Ops.AFTER dtypes.void [0, 288]
290 Ops.END dtypes.void [289, 198]
291 Ops.AFTER dtypes.f32 [193, 290]
292 Ops.SHRINK dtypes.f32 [291, 19, 12]
293 Ops.LOAD dtypes.f32 [292]
294 Ops.ADD dtypes.f32 [192, 293]
295 Ops.STORE dtypes.void [191, 294]
296 Ops.AFTER dtypes.void [0, 295]
297 Ops.END dtypes.void [296, 189]
298 Ops.AFTER dtypes.f32 [184, 297]
299 Ops.SHRINK dtypes.f32 [298, 19, 12]
300 Ops.LOAD dtypes.f32 [299]
301 Ops.MUL dtypes.f32 [300, 153]
302 Ops.ADD dtypes.f32 [301, 156]
303 Ops.SQRT dtypes.f32 [302]
304 Ops.FDIV dtypes.f32 [15, 303]
305 Ops.STORE dtypes.void [183, 304]
306 Ops.AFTER dtypes.void [0, 305]
307 Ops.END dtypes.void [306, 180]
308 Ops.SINK dtypes.void [307] KernelInfo(name="r_16_4_640_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), estimates=Estimates(ops=492480, lds=655872, mem=655872), optimized=True, split=(16, 3, 4))
309 Ops.LINEAR dtypes.void [164, 165, 11, 193, 184, 171, 172, 170, 173, 174, 175, 176, 177, 178, 18, 19, 166, 167, 168, 67, 68, 203, 204, 8, 9, 214, 215, 1, 169, 43, 205, 216, 0, 21, 22, 23, 152, 153, 14, 15, 155, 156, 25, 26, 50, 51, 79, 80, 56, 57, 12, 99, 100, 36, 119, 120, 45, 46, 179, 180, 181, 182, 217, 218, 188, 185, 219, 220, 238, 256, 274, 186, 209, 187, 189, 197, 194, 195, 206, 196, 198, 207, 208, 210, 234, 235, 236, 252, 253, 254, 270, 271, 272, 211, 212, 199, 200, 201, 244, 242, 240, 237, 262, 260, 258, 255, 280, 278, 276, 273, 226, 224, 222, 213, 202, 233, 251, 269, 245, 243, 241, 239, 263, 261, 259, 257, 281, 279, 277, 275, 227, 225, 223, 221, 228, 229, 230, 231, 232, 246, 247, 248, 249, 250, 264, 265, 266, 267, 268, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 190, 191, 192, 294, 295, 296, 297, 298, 299, 300, 183, 301, 302, 303, 304, 305, 306, 307, 308]
310 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_16_4_640_4(unsigned char* restrict args_40) {\n  float buf0[4];\n  float buf1[4];\n  int val0 = (*(((int*)(args_40))+6));\n  int val1 = (*(((int*)(args_40))+7));\n  unsigned long val2 = (*(((unsigned long*)(args_40))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_40))+1));\n  unsigned long val4 = (*(((unsigned long*)(args_40))+2));\n  float4 cast0 = (float4){0.0f,0.0f,0.0f,0.0f};\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    int alu1 = (alu0<<2);\n    float4 val5 = (*((float4*)((((float*)(val4))+alu1))));\n    float4 alu2 = -val5;\n    *((float4*)((buf1+0))) = cast0;\n    for (int Lidx0_0_0 = 0; Lidx0_0_0 < 10; Lidx0_0_0++) {\n      *((float4*)((buf0+0))) = cast0;\n      for (int Lidx0_0_0_256 = 0; Lidx0_0_0_256 < 64; Lidx0_0_0_256++) {\n        int alu5 = ((Lidx0_0_0<<8)+(Lidx0_0_0_256<<2)+(alu0*10240));\n        float4 val6 = (*((float4*)((((float*)(val3))+(alu5+2560)))));\n        float4 val7 = (*((float4*)((((float*)(val3))+(alu5+5120)))));\n        float4 val8 = (*((float4*)((((float*)(val3))+(alu5+7680)))));\n        float4 val9 = (*((float4*)((((float*)(val3))+alu5))));\n        float4 val10 = (*((float4*)((buf0+0))));\n        float alu6 = (val6[0]+alu2[1]);\n        float alu7 = (val6[1]+alu2[1]);\n        float alu8 = (val6[2]+alu2[1]);\n        float alu9 = (val6[3]+alu2[1]);\n        float alu10 = (val7[0]+alu2[2]);\n        float alu11 = (val7[1]+alu2[2]);\n        float alu12 = (val7[2]+alu2[2]);\n        float alu13 = (val7[3]+alu2[2]);\n        float alu14 = (val8[0]+alu2[3]);\n        float alu15 = (val8[1]+alu2[3]);\n        float alu16 = (val8[2]+alu2[3]);\n        float alu17 = (val8[3]+alu2[3]);\n        float alu18 = (val9[0]+alu2[0]);\n        float alu19 = (val9[1]+alu2[0]);\n        float alu20 = (val9[2]+alu2[0]);\n        float alu21 = (val9[3]+alu2[0]);\n        *((float4*)((buf0+0))) = (float4){(val10[0]+__builtin_fmaf(alu21,alu21,__builtin_fmaf(alu20,alu20,__builtin_fmaf(alu19,alu19,(alu18*alu18))))),(val10[1]+__builtin_fmaf(alu9,alu9,__builtin_fmaf(alu8,alu8,__builtin_fmaf(alu7,alu7,(alu6*alu6))))),(val10[2]+__builtin_fmaf(alu13,alu13,__builtin_fmaf(alu12,alu12,__builtin_fmaf(alu11,alu11,(alu10*alu10))))),(val10[3]+__builtin_fmaf(alu17,alu17,__builtin_fmaf(alu16,alu16,__builtin_fmaf(alu15,alu15,(alu14*alu14)))))};\n      }\n      *((float4*)((buf1+0))) = ((*((float4*)((buf1+0))))+(*((float4*)((buf0+0)))));\n    }\n    *((float4*)((((float*)(val2))+alu1))) = (1.0f/__builtin_elementwise_sqrt((((*((float4*)((buf1+0))))*0.0003906250058207661f)+9.999999747378752e-06f)));\n  }\n}\n"
311 Ops.PROGRAM dtypes.void [308, 309, 310] ProgramInfo(global_size=(16, 1, 1), params=(%2, %44, %59, %3, %4), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
312 Ops.SINK dtypes.void [163, 311]

