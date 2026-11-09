# tolk D89 golden=39537cf3c4dd7149201a2a9d84646546
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.SPECIAL dtypes.i32 [3] "gidx1"
5 Ops.CONST dtypes.weakint [] 5
6 Ops.CAST dtypes.i32 [5] dtypes.i32
7 Ops.SHL dtypes.i32 [4, 6]
8 Ops.SPECIAL dtypes.i32 [3] "gidx0"
9 Ops.ADD dtypes.i32 [7, 8]
10 Ops.INDEX dtypes.f32 [1, 9]
11 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
12 Ops.INDEX dtypes.f32 [11, 9]
13 Ops.LOAD dtypes.f32 [12]
14 Ops.CONST dtypes.weakint [] 1
15 Ops.ALLOC dtypes.f32 [14] ParamArg(slot=4, dtype=dtypes.f32, addrspace=AddrSpace.REG)
16 Ops.AFTER dtypes.f32 [15, 4]
17 Ops.CONST dtypes.weakint [] 0
18 Ops.CAST dtypes.i32 [17] dtypes.i32
19 Ops.INDEX dtypes.f32 [16, 18]
20 Ops.CONST dtypes.weakfloat [] -inf
21 Ops.CAST dtypes.f32 [20] dtypes.f32
22 Ops.STORE dtypes.void [19, 21]
23 Ops.CONST dtypes.weakint [] 4
24 Ops.CAST dtypes.i32 [23] dtypes.i32
25 Ops.ALLOC dtypes.f32 [14] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
26 Ops.SPECIAL dtypes.i32 [24] "lidx1"
27 Ops.SPECIAL dtypes.i32 [24] "lidx0"
28 Ops.AFTER dtypes.f32 [25, 26, 27, 4]
29 Ops.INDEX dtypes.f32 [28, 18]
30 Ops.STORE dtypes.void [29, 21]
31 Ops.CONST dtypes.weakint [] 2
32 Ops.CAST dtypes.i32 [31] dtypes.i32
33 Ops.ALLOC dtypes.f32 [14] ParamArg(slot=3, dtype=dtypes.f32, addrspace=AddrSpace.REG)
34 Ops.AFTER dtypes.f32 [33, 27, 4]
35 Ops.INDEX dtypes.f32 [34, 18]
36 Ops.STORE dtypes.void [35, 21]
37 Ops.CONST dtypes.weakint [] 8
38 Ops.CAST dtypes.i32 [37] dtypes.i32
39 Ops.RANGE dtypes.i32 [38] (AxisType.WEAK, 0)
40 Ops.AFTER dtypes.f32 [33, 36, 39]
41 Ops.INDEX dtypes.f32 [40, 18]
42 Ops.LOAD dtypes.f32 [41]
43 Ops.SHL dtypes.i32 [39, 32]
44 Ops.ADD dtypes.i32 [43, 27]
45 Ops.ADD dtypes.i32 [44, 7]
46 Ops.INDEX dtypes.f32 [11, 45]
47 Ops.LOAD dtypes.f32 [46]
48 Ops.CMPLT dtypes.bool [42, 47]
49 Ops.WHERE dtypes.f32 [48, 47, 42]
50 Ops.STORE dtypes.void [41, 49]
51 Ops.END dtypes.void [50, 39]
52 Ops.AFTER dtypes.i32 [32, 51]
53 Ops.RANGE dtypes.i32 [52] (AxisType.WEAK, 1)
54 Ops.AFTER dtypes.f32 [25, 30, 53]
55 Ops.INDEX dtypes.f32 [54, 18]
56 Ops.LOAD dtypes.f32 [55]
57 Ops.SHL dtypes.i32 [26, 32]
58 Ops.SHL dtypes.i32 [53, 24]
59 Ops.ADD dtypes.i32 [57, 58]
60 Ops.ADD dtypes.i32 [59, 27]
61 Ops.ADD dtypes.i32 [60, 7]
62 Ops.INDEX dtypes.f32 [11, 61]
63 Ops.LOAD dtypes.f32 [62]
64 Ops.CMPLT dtypes.bool [56, 63]
65 Ops.WHERE dtypes.f32 [64, 63, 56]
66 Ops.STORE dtypes.void [55, 65]
67 Ops.END dtypes.void [66, 53]
68 Ops.AFTER dtypes.i32 [24, 67]
69 Ops.RANGE dtypes.i32 [68] (AxisType.WEAK, 105)
70 Ops.AFTER dtypes.f32 [15, 22, 69]
71 Ops.INDEX dtypes.f32 [70, 18]
72 Ops.LOAD dtypes.f32 [71]
73 Ops.ALLOC dtypes.f32 [23] ParamArg(slot=6, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
74 Ops.INDEX dtypes.f32 [73, 27]
75 Ops.AFTER dtypes.f32 [33, 51]
76 Ops.INDEX dtypes.f32 [75, 18]
77 Ops.LOAD dtypes.f32 [76]
78 Ops.STORE dtypes.void [74, 77]
79 Ops.BARRIER dtypes.void [78]
80 Ops.AFTER dtypes.f32 [73, 79]
81 Ops.INDEX dtypes.f32 [80, 69]
82 Ops.LOAD dtypes.f32 [81]
83 Ops.CMPLT dtypes.bool [72, 82]
84 Ops.WHERE dtypes.f32 [83, 82, 72]
85 Ops.STORE dtypes.void [71, 84]
86 Ops.END dtypes.void [85, 69]
87 Ops.AFTER dtypes.f32 [15, 86]
88 Ops.INDEX dtypes.f32 [87, 18]
89 Ops.LOAD dtypes.f32 [88]
90 Ops.SUB dtypes.f32 [13, 89]
91 Ops.CONST dtypes.weakfloat [] 1.4426950216293335
92 Ops.CAST dtypes.f32 [91] dtypes.f32
93 Ops.MUL dtypes.f32 [90, 92]
94 Ops.EXP2 dtypes.f32 [93]
95 Ops.ALLOC dtypes.f32 [14] ParamArg(slot=2, dtype=dtypes.f32, addrspace=AddrSpace.REG)
96 Ops.AFTER dtypes.f32 [95, 4]
97 Ops.INDEX dtypes.f32 [96, 18]
98 Ops.CONST dtypes.weakfloat [] 0.0
99 Ops.CAST dtypes.f32 [98] dtypes.f32
100 Ops.STORE dtypes.void [97, 99]
101 Ops.ALLOC dtypes.f32 [14] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
102 Ops.AFTER dtypes.f32 [101, 4]
103 Ops.INDEX dtypes.f32 [102, 18]
104 Ops.STORE dtypes.void [103, 21]
105 Ops.AFTER dtypes.i32 [24, 86]
106 Ops.RANGE dtypes.i32 [105] (AxisType.WEAK, 106)
107 Ops.AFTER dtypes.i32 [24, 106]
108 Ops.RANGE dtypes.i32 [107] (AxisType.WEAK, 107)
109 Ops.AFTER dtypes.f32 [101, 104, 106, 108]
110 Ops.INDEX dtypes.f32 [109, 18]
111 Ops.LOAD dtypes.f32 [110]
112 Ops.CONST dtypes.weakint [] 16
113 Ops.ALLOC dtypes.f32 [112] ParamArg(slot=5, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
114 Ops.ADD dtypes.i32 [57, 27]
115 Ops.INDEX dtypes.f32 [113, 114]
116 Ops.AFTER dtypes.f32 [25, 67]
117 Ops.INDEX dtypes.f32 [116, 18]
118 Ops.LOAD dtypes.f32 [117]
119 Ops.STORE dtypes.void [115, 118]
120 Ops.BARRIER dtypes.void [119]
121 Ops.AFTER dtypes.f32 [113, 120]
122 Ops.SHL dtypes.i32 [106, 32]
123 Ops.ADD dtypes.i32 [122, 108]
124 Ops.INDEX dtypes.f32 [121, 123]
125 Ops.LOAD dtypes.f32 [124]
126 Ops.CMPLT dtypes.bool [111, 125]
127 Ops.WHERE dtypes.f32 [126, 125, 111]
128 Ops.STORE dtypes.void [110, 127]
129 Ops.END dtypes.void [128, 108]
130 Ops.END dtypes.void [129, 106]
131 Ops.AFTER dtypes.i32 [3, 130]
132 Ops.RANGE dtypes.i32 [131] (AxisType.WEAK, 2)
133 Ops.AFTER dtypes.f32 [95, 100, 132]
134 Ops.INDEX dtypes.f32 [133, 18]
135 Ops.LOAD dtypes.f32 [134]
136 Ops.ADD dtypes.i32 [7, 132]
137 Ops.INDEX dtypes.f32 [11, 136]
138 Ops.LOAD dtypes.f32 [137]
139 Ops.AFTER dtypes.f32 [101, 130]
140 Ops.INDEX dtypes.f32 [139, 18]
141 Ops.LOAD dtypes.f32 [140]
142 Ops.SUB dtypes.f32 [138, 141]
143 Ops.MUL dtypes.f32 [142, 92]
144 Ops.EXP2 dtypes.f32 [143]
145 Ops.ADD dtypes.f32 [135, 144]
146 Ops.STORE dtypes.void [134, 145]
147 Ops.END dtypes.void [146, 132]
148 Ops.AFTER dtypes.f32 [95, 147]
149 Ops.INDEX dtypes.f32 [148, 18]
150 Ops.LOAD dtypes.f32 [149]
151 Ops.RECIP dtypes.f32 [150]
152 Ops.MUL dtypes.f32 [94, 151]
153 Ops.CMPEQ dtypes.bool [27, 18]
154 Ops.CMPEQ dtypes.bool [26, 18]
155 Ops.AND dtypes.bool [153, 154]
156 Ops.STORE dtypes.void [10, 152, 155]
157 Ops.SINK dtypes.void [156] KernelInfo(name="r_32_32_4_4_8_2_32", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.LOCAL))), optimized=True)
158 Ops.PARAM dtypes.u8 [112] ParamArg(slot=0, dtype=dtypes.u8, name="args")
159 Ops.BITCAST dtypes.u64 [158] dtypes.u64
160 Ops.INDEX dtypes.u64 [159, 18]
161 Ops.LOAD dtypes.u64 [160]
162 Ops.REFPARAM dtypes.f32 [0, 161] ParamArg(slot=0, dtype=dtypes.f32)
163 Ops.INDEX dtypes.f32 [162, 9]
164 Ops.CAST dtypes.i32 [14] dtypes.i32
165 Ops.INDEX dtypes.u64 [159, 164]
166 Ops.LOAD dtypes.u64 [165]
167 Ops.REFPARAM dtypes.f32 [0, 166] ParamArg(slot=1, dtype=dtypes.f32)
168 Ops.INDEX dtypes.f32 [167, 9]
169 Ops.LOAD dtypes.f32 [168]
170 Ops.BUFFER dtypes.f32 [14] ParamArg(slot=4, dtype=dtypes.f32, addrspace=AddrSpace.REG)
171 Ops.AFTER dtypes.f32 [170, 4]
172 Ops.INDEX dtypes.f32 [171, 18]
173 Ops.STORE dtypes.void [172, 21]
174 Ops.BUFFER dtypes.f32 [14] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
175 Ops.AFTER dtypes.f32 [174, 26, 27, 4]
176 Ops.INDEX dtypes.f32 [175, 18]
177 Ops.STORE dtypes.void [176, 21]
178 Ops.BUFFER dtypes.f32 [14] ParamArg(slot=3, dtype=dtypes.f32, addrspace=AddrSpace.REG)
179 Ops.AFTER dtypes.f32 [178, 27, 4]
180 Ops.INDEX dtypes.f32 [179, 18]
181 Ops.STORE dtypes.void [180, 21]
182 Ops.AFTER dtypes.f32 [178, 181, 39]
183 Ops.INDEX dtypes.f32 [182, 18]
184 Ops.LOAD dtypes.f32 [183]
185 Ops.INDEX dtypes.f32 [167, 45]
186 Ops.LOAD dtypes.f32 [185]
187 Ops.CMPLT dtypes.bool [184, 186]
188 Ops.WHERE dtypes.f32 [187, 186, 184]
189 Ops.STORE dtypes.void [183, 188]
190 Ops.END dtypes.void [189, 39]
191 Ops.AFTER dtypes.i32 [32, 190]
192 Ops.RANGE dtypes.i32 [191] (AxisType.WEAK, 1)
193 Ops.AFTER dtypes.f32 [174, 177, 192]
194 Ops.INDEX dtypes.f32 [193, 18]
195 Ops.LOAD dtypes.f32 [194]
196 Ops.SHL dtypes.i32 [192, 24]
197 Ops.ADD dtypes.i32 [57, 196]
198 Ops.ADD dtypes.i32 [197, 27]
199 Ops.ADD dtypes.i32 [198, 7]
200 Ops.INDEX dtypes.f32 [167, 199]
201 Ops.LOAD dtypes.f32 [200]
202 Ops.CMPLT dtypes.bool [195, 201]
203 Ops.WHERE dtypes.f32 [202, 201, 195]
204 Ops.STORE dtypes.void [194, 203]
205 Ops.END dtypes.void [204, 192]
206 Ops.AFTER dtypes.i32 [24, 205]
207 Ops.RANGE dtypes.i32 [206] (AxisType.WEAK, 105)
208 Ops.AFTER dtypes.f32 [170, 173, 207]
209 Ops.INDEX dtypes.f32 [208, 18]
210 Ops.LOAD dtypes.f32 [209]
211 Ops.BUFFER dtypes.f32 [23] ParamArg(slot=6, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
212 Ops.INDEX dtypes.f32 [211, 27]
213 Ops.AFTER dtypes.f32 [178, 190]
214 Ops.INDEX dtypes.f32 [213, 18]
215 Ops.LOAD dtypes.f32 [214]
216 Ops.STORE dtypes.void [212, 215]
217 Ops.BARRIER dtypes.void [216]
218 Ops.AFTER dtypes.f32 [211, 217]
219 Ops.INDEX dtypes.f32 [218, 207]
220 Ops.LOAD dtypes.f32 [219]
221 Ops.CMPLT dtypes.bool [210, 220]
222 Ops.WHERE dtypes.f32 [221, 220, 210]
223 Ops.STORE dtypes.void [209, 222]
224 Ops.END dtypes.void [223, 207]
225 Ops.AFTER dtypes.f32 [170, 224]
226 Ops.INDEX dtypes.f32 [225, 18]
227 Ops.LOAD dtypes.f32 [226]
228 Ops.SUB dtypes.f32 [169, 227]
229 Ops.MUL dtypes.f32 [228, 92]
230 Ops.EXP2 dtypes.f32 [229]
231 Ops.BUFFER dtypes.f32 [14] ParamArg(slot=2, dtype=dtypes.f32, addrspace=AddrSpace.REG)
232 Ops.AFTER dtypes.f32 [231, 4]
233 Ops.INDEX dtypes.f32 [232, 18]
234 Ops.STORE dtypes.void [233, 99]
235 Ops.BUFFER dtypes.f32 [14] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
236 Ops.AFTER dtypes.f32 [235, 4]
237 Ops.INDEX dtypes.f32 [236, 18]
238 Ops.STORE dtypes.void [237, 21]
239 Ops.AFTER dtypes.i32 [24, 224]
240 Ops.RANGE dtypes.i32 [239] (AxisType.WEAK, 106)
241 Ops.AFTER dtypes.i32 [24, 240]
242 Ops.RANGE dtypes.i32 [241] (AxisType.WEAK, 107)
243 Ops.AFTER dtypes.f32 [235, 238, 240, 242]
244 Ops.INDEX dtypes.f32 [243, 18]
245 Ops.LOAD dtypes.f32 [244]
246 Ops.BUFFER dtypes.f32 [112] ParamArg(slot=5, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
247 Ops.INDEX dtypes.f32 [246, 114]
248 Ops.AFTER dtypes.f32 [174, 205]
249 Ops.INDEX dtypes.f32 [248, 18]
250 Ops.LOAD dtypes.f32 [249]
251 Ops.STORE dtypes.void [247, 250]
252 Ops.BARRIER dtypes.void [251]
253 Ops.AFTER dtypes.f32 [246, 252]
254 Ops.SHL dtypes.i32 [240, 32]
255 Ops.ADD dtypes.i32 [254, 242]
256 Ops.INDEX dtypes.f32 [253, 255]
257 Ops.LOAD dtypes.f32 [256]
258 Ops.CMPLT dtypes.bool [245, 257]
259 Ops.WHERE dtypes.f32 [258, 257, 245]
260 Ops.STORE dtypes.void [244, 259]
261 Ops.END dtypes.void [260, 242]
262 Ops.END dtypes.void [261, 240]
263 Ops.AFTER dtypes.i32 [3, 262]
264 Ops.RANGE dtypes.i32 [263] (AxisType.WEAK, 2)
265 Ops.AFTER dtypes.f32 [231, 234, 264]
266 Ops.INDEX dtypes.f32 [265, 18]
267 Ops.LOAD dtypes.f32 [266]
268 Ops.ADD dtypes.i32 [7, 264]
269 Ops.INDEX dtypes.f32 [167, 268]
270 Ops.LOAD dtypes.f32 [269]
271 Ops.AFTER dtypes.f32 [235, 262]
272 Ops.INDEX dtypes.f32 [271, 18]
273 Ops.LOAD dtypes.f32 [272]
274 Ops.SUB dtypes.f32 [270, 273]
275 Ops.MUL dtypes.f32 [274, 92]
276 Ops.EXP2 dtypes.f32 [275]
277 Ops.ADD dtypes.f32 [267, 276]
278 Ops.STORE dtypes.void [266, 277]
279 Ops.END dtypes.void [278, 264]
280 Ops.AFTER dtypes.f32 [231, 279]
281 Ops.INDEX dtypes.f32 [280, 18]
282 Ops.LOAD dtypes.f32 [281]
283 Ops.RECIP dtypes.f32 [282]
284 Ops.MUL dtypes.f32 [230, 283]
285 Ops.STORE dtypes.void [163, 284]
286 Ops.SINK dtypes.void [285] KernelInfo(name="r_32_32_4_4_8_2_32", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.LOCAL))), estimates=Estimates(ops=221184, lds=249856, mem=8192), optimized=True)
287 Ops.IF dtypes.void [155, 163]
288 Ops.ENDIF dtypes.void [287]
289 Ops.LINEAR dtypes.void [112, 158, 14, 174, 235, 231, 178, 170, 246, 23, 211, 164, 159, 165, 166, 2, 3, 4, 24, 27, 0, 167, 179, 17, 18, 180, 20, 21, 31, 32, 5, 6, 37, 38, 7, 181, 39, 43, 44, 45, 185, 186, 182, 183, 184, 187, 188, 189, 190, 26, 191, 175, 176, 57, 177, 192, 196, 197, 198, 199, 200, 201, 193, 194, 195, 202, 203, 204, 205, 213, 214, 215, 206, 171, 212, 216, 217, 218, 172, 173, 207, 208, 209, 210, 219, 220, 221, 222, 223, 224, 248, 249, 250, 239, 236, 114, 247, 251, 252, 253, 237, 238, 240, 241, 254, 242, 243, 244, 245, 255, 256, 257, 258, 259, 260, 261, 262, 271, 272, 273, 263, 232, 233, 98, 99, 91, 92, 234, 264, 268, 269, 270, 265, 266, 267, 274, 275, 276, 277, 278, 279, 160, 161, 8, 9, 168, 169, 280, 281, 282, 225, 226, 227, 162, 163, 228, 229, 230, 283, 284, 153, 154, 155, 287, 285, 288, 286]
290 Ops.SOURCE dtypes.void [] "#include <metal_stdlib>\nusing namespace metal;\nkernel void r_32_32_4_4_8_2_32(device unsigned char* args_16, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n  float buf0[1];\n  float buf1[1];\n  float buf2[1];\n  float buf3[1];\n  float buf4[1];\n  threadgroup __attribute__((aligned(16))) float buf5[16];\n  threadgroup __attribute__((aligned(16))) float buf6[4];\n  unsigned long val0 = (*(((device unsigned long*)(args_16))+1));\n  int gidx1 = gid.y; /* 32 */\n  int lidx0 = lid.x; /* 4 */\n  int alu0 = (gidx1<<5);\n  *(buf3+0) = ((float)(-INFINITY));\n  for (int Lidx0 = 0; Lidx0 < 8; Lidx0++) {\n    float val1 = (*(((device float*)(val0))+((Lidx0<<2)+lidx0+alu0)));\n    float val2 = (*(buf3+0));\n    float alu2 = ((val2<val1)?val1:val2);\n    *(buf3+0) = alu2;\n  }\n  int lidx1 = lid.y; /* 4 */\n  int alu5 = (lidx1<<2);\n  *(buf0+0) = ((float)(-INFINITY));\n  for (int Lidx1 = 0; Lidx1 < 2; Lidx1++) {\n    float val3 = (*(((device float*)(val0))+(alu5+(Lidx1<<4)+lidx0+alu0)));\n    float val4 = (*(buf0+0));\n    float alu7 = ((val4<val3)?val3:val4);\n    *(buf0+0) = alu7;\n  }\n  *(buf6+lidx0) = (*(buf3+0));\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  *(buf4+0) = ((float)(-INFINITY));\n  for (int Lidx105 = 0; Lidx105 < 4; Lidx105++) {\n    float val5 = (*(buf4+0));\n    float val6 = (*(buf6+Lidx105));\n    float alu13 = ((val5<val6)?val6:val5);\n    *(buf4+0) = alu13;\n  }\n  *(buf5+(alu5+lidx0)) = (*(buf0+0));\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  *(buf1+0) = ((float)(-INFINITY));\n  for (int Lidx106 = 0; Lidx106 < 4; Lidx106++) {\n    for (int Lidx107 = 0; Lidx107 < 4; Lidx107++) {\n      float val7 = (*(buf1+0));\n      float val8 = (*(buf5+((Lidx106<<2)+Lidx107)));\n      float alu19 = ((val7<val8)?val8:val7);\n      *(buf1+0) = alu19;\n    }\n  }\n  *(buf2+0) = 0.0f;\n  for (int Lidx2 = 0; Lidx2 < 32; Lidx2++) {\n    float val9 = (*(((device float*)(val0))+(alu0+Lidx2)));\n    *(buf2+0) = ((*(buf2+0))+exp2(((val9-(*(buf1+0)))*1.4426950216293335f)));\n  }\n  unsigned long val10 = (*(((device unsigned long*)(args_16))+0));\n  int gidx0 = gid.x; /* 32 */\n  int alu26 = (alu0+gidx0);\n  float val11 = (*(((device float*)(val0))+alu26));\n  if (((lidx0==0)&(lidx1==0))) {\n    *(((device float*)(val10))+alu26) = (exp2(((val11-(*(buf4+0)))*1.4426950216293335f))*(1/(*(buf2+0))));\n  }\n}"
291 Ops.PROGRAM dtypes.void [286, 289, 290] ProgramInfo(global_size=(32, 32, 1), local_size=(4, 4, 1), params=(%1, %11), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="METAL", renderer="METAL", arch="Apple9"))
292 Ops.SINK dtypes.void [157, 291]

