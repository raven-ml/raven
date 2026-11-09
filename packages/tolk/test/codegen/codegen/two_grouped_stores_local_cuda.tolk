# tolk D89 golden=292a45ae15416d45ac640cb49070f569
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
158 Ops.BUFFER dtypes.f32 [14] ParamArg(slot=4, dtype=dtypes.f32, addrspace=AddrSpace.REG)
159 Ops.AFTER dtypes.f32 [158, 4]
160 Ops.INDEX dtypes.f32 [159, 18]
161 Ops.STORE dtypes.void [160, 21]
162 Ops.BUFFER dtypes.f32 [14] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
163 Ops.AFTER dtypes.f32 [162, 26, 27, 4]
164 Ops.INDEX dtypes.f32 [163, 18]
165 Ops.STORE dtypes.void [164, 21]
166 Ops.BUFFER dtypes.f32 [14] ParamArg(slot=3, dtype=dtypes.f32, addrspace=AddrSpace.REG)
167 Ops.AFTER dtypes.f32 [166, 27, 4]
168 Ops.INDEX dtypes.f32 [167, 18]
169 Ops.STORE dtypes.void [168, 21]
170 Ops.AFTER dtypes.f32 [166, 169, 39]
171 Ops.INDEX dtypes.f32 [170, 18]
172 Ops.LOAD dtypes.f32 [171]
173 Ops.CMPLT dtypes.bool [172, 47]
174 Ops.WHERE dtypes.f32 [173, 47, 172]
175 Ops.STORE dtypes.void [171, 174]
176 Ops.END dtypes.void [175, 39]
177 Ops.AFTER dtypes.i32 [32, 176]
178 Ops.RANGE dtypes.i32 [177] (AxisType.WEAK, 1)
179 Ops.AFTER dtypes.f32 [162, 165, 178]
180 Ops.INDEX dtypes.f32 [179, 18]
181 Ops.LOAD dtypes.f32 [180]
182 Ops.SHL dtypes.i32 [178, 24]
183 Ops.ADD dtypes.i32 [57, 182]
184 Ops.ADD dtypes.i32 [183, 27]
185 Ops.ADD dtypes.i32 [184, 7]
186 Ops.INDEX dtypes.f32 [11, 185]
187 Ops.LOAD dtypes.f32 [186]
188 Ops.CMPLT dtypes.bool [181, 187]
189 Ops.WHERE dtypes.f32 [188, 187, 181]
190 Ops.STORE dtypes.void [180, 189]
191 Ops.END dtypes.void [190, 178]
192 Ops.AFTER dtypes.i32 [24, 191]
193 Ops.RANGE dtypes.i32 [192] (AxisType.WEAK, 105)
194 Ops.AFTER dtypes.f32 [158, 161, 193]
195 Ops.INDEX dtypes.f32 [194, 18]
196 Ops.LOAD dtypes.f32 [195]
197 Ops.BUFFER dtypes.f32 [23] ParamArg(slot=6, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
198 Ops.INDEX dtypes.f32 [197, 27]
199 Ops.AFTER dtypes.f32 [166, 176]
200 Ops.INDEX dtypes.f32 [199, 18]
201 Ops.LOAD dtypes.f32 [200]
202 Ops.STORE dtypes.void [198, 201]
203 Ops.BARRIER dtypes.void [202]
204 Ops.AFTER dtypes.f32 [197, 203]
205 Ops.INDEX dtypes.f32 [204, 193]
206 Ops.LOAD dtypes.f32 [205]
207 Ops.CMPLT dtypes.bool [196, 206]
208 Ops.WHERE dtypes.f32 [207, 206, 196]
209 Ops.STORE dtypes.void [195, 208]
210 Ops.END dtypes.void [209, 193]
211 Ops.AFTER dtypes.f32 [158, 210]
212 Ops.INDEX dtypes.f32 [211, 18]
213 Ops.LOAD dtypes.f32 [212]
214 Ops.SUB dtypes.f32 [13, 213]
215 Ops.MUL dtypes.f32 [214, 92]
216 Ops.EXP2 dtypes.f32 [215]
217 Ops.BUFFER dtypes.f32 [14] ParamArg(slot=2, dtype=dtypes.f32, addrspace=AddrSpace.REG)
218 Ops.AFTER dtypes.f32 [217, 4]
219 Ops.INDEX dtypes.f32 [218, 18]
220 Ops.STORE dtypes.void [219, 99]
221 Ops.BUFFER dtypes.f32 [14] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
222 Ops.AFTER dtypes.f32 [221, 4]
223 Ops.INDEX dtypes.f32 [222, 18]
224 Ops.STORE dtypes.void [223, 21]
225 Ops.AFTER dtypes.i32 [24, 210]
226 Ops.RANGE dtypes.i32 [225] (AxisType.WEAK, 106)
227 Ops.AFTER dtypes.i32 [24, 226]
228 Ops.RANGE dtypes.i32 [227] (AxisType.WEAK, 107)
229 Ops.AFTER dtypes.f32 [221, 224, 226, 228]
230 Ops.INDEX dtypes.f32 [229, 18]
231 Ops.LOAD dtypes.f32 [230]
232 Ops.BUFFER dtypes.f32 [112] ParamArg(slot=5, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
233 Ops.INDEX dtypes.f32 [232, 114]
234 Ops.AFTER dtypes.f32 [162, 191]
235 Ops.INDEX dtypes.f32 [234, 18]
236 Ops.LOAD dtypes.f32 [235]
237 Ops.STORE dtypes.void [233, 236]
238 Ops.BARRIER dtypes.void [237]
239 Ops.AFTER dtypes.f32 [232, 238]
240 Ops.SHL dtypes.i32 [226, 32]
241 Ops.ADD dtypes.i32 [240, 228]
242 Ops.INDEX dtypes.f32 [239, 241]
243 Ops.LOAD dtypes.f32 [242]
244 Ops.CMPLT dtypes.bool [231, 243]
245 Ops.WHERE dtypes.f32 [244, 243, 231]
246 Ops.STORE dtypes.void [230, 245]
247 Ops.END dtypes.void [246, 228]
248 Ops.END dtypes.void [247, 226]
249 Ops.AFTER dtypes.i32 [3, 248]
250 Ops.RANGE dtypes.i32 [249] (AxisType.WEAK, 2)
251 Ops.AFTER dtypes.f32 [217, 220, 250]
252 Ops.INDEX dtypes.f32 [251, 18]
253 Ops.LOAD dtypes.f32 [252]
254 Ops.ADD dtypes.i32 [7, 250]
255 Ops.INDEX dtypes.f32 [11, 254]
256 Ops.LOAD dtypes.f32 [255]
257 Ops.AFTER dtypes.f32 [221, 248]
258 Ops.INDEX dtypes.f32 [257, 18]
259 Ops.LOAD dtypes.f32 [258]
260 Ops.SUB dtypes.f32 [256, 259]
261 Ops.MUL dtypes.f32 [260, 92]
262 Ops.EXP2 dtypes.f32 [261]
263 Ops.ADD dtypes.f32 [253, 262]
264 Ops.STORE dtypes.void [252, 263]
265 Ops.END dtypes.void [264, 250]
266 Ops.AFTER dtypes.f32 [217, 265]
267 Ops.INDEX dtypes.f32 [266, 18]
268 Ops.LOAD dtypes.f32 [267]
269 Ops.RECIP dtypes.f32 [268]
270 Ops.MUL dtypes.f32 [216, 269]
271 Ops.STORE dtypes.void [10, 270]
272 Ops.SINK dtypes.void [271] KernelInfo(name="r_32_32_4_4_8_2_32", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.LOCAL))), estimates=Estimates(ops=221184, lds=249856, mem=8192), optimized=True)
273 Ops.IF dtypes.void [155, 10]
274 Ops.ENDIF dtypes.void [273]
275 Ops.LINEAR dtypes.void [0, 1, 11, 14, 162, 221, 217, 166, 158, 112, 232, 23, 197, 2, 3, 4, 24, 27, 167, 17, 18, 168, 20, 21, 31, 32, 5, 6, 37, 38, 7, 169, 39, 43, 44, 45, 46, 47, 170, 171, 172, 173, 174, 175, 176, 26, 177, 163, 164, 57, 165, 178, 182, 183, 184, 185, 186, 187, 179, 180, 181, 188, 189, 190, 191, 199, 200, 201, 192, 159, 198, 202, 203, 204, 160, 161, 193, 194, 195, 196, 205, 206, 207, 208, 209, 210, 234, 235, 236, 225, 222, 114, 233, 237, 238, 239, 223, 224, 226, 227, 240, 228, 229, 230, 231, 241, 242, 243, 244, 245, 246, 247, 248, 257, 258, 259, 249, 218, 219, 98, 99, 91, 92, 220, 250, 254, 255, 256, 251, 252, 253, 260, 261, 262, 263, 264, 265, 8, 9, 12, 13, 266, 267, 268, 211, 212, 213, 10, 214, 215, 216, 269, 270, 153, 154, 155, 273, 271, 274, 272]
276 Ops.SOURCE dtypes.void [] "typedef unsigned int uint;\n#define INFINITY (__int_as_float(0x7f800000))\n#define NAN (__int_as_float(0x7fffffff))\ntemplate <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }\nextern \"C\" __global__ void __launch_bounds__(16) r_32_32_4_4_8_2_32(float* data0_1024, float* data1_1024) {\n  float buf0[1];\n  float buf1[1];\n  float buf2[1];\n  float buf3[1];\n  float buf4[1];\n  __shared__ __align__(16) float buf5[16];\n  __shared__ __align__(16) float buf6[4];\n  int gidx1 = blockIdx.y; /* 32 */\n  int lidx0 = threadIdx.x; /* 4 */\n  int alu0 = (gidx1<<5);\n  *(buf3+0) = ((float)(-INFINITY));\n  for (int Lidx0 = 0; Lidx0 < 8; Lidx0++) {\n    float val0 = (*(data1_1024+((Lidx0<<2)+lidx0+alu0)));\n    float val1 = (*(buf3+0));\n    float alu2 = ((val1<val0)?val0:val1);\n    *(buf3+0) = alu2;\n  }\n  int lidx1 = threadIdx.y; /* 4 */\n  int alu5 = (lidx1<<2);\n  *(buf0+0) = ((float)(-INFINITY));\n  for (int Lidx1 = 0; Lidx1 < 2; Lidx1++) {\n    float val2 = (*(data1_1024+(alu5+(Lidx1<<4)+lidx0+alu0)));\n    float val3 = (*(buf0+0));\n    float alu7 = ((val3<val2)?val2:val3);\n    *(buf0+0) = alu7;\n  }\n  *(buf6+lidx0) = (*(buf3+0));\n  __syncthreads();\n  *(buf4+0) = ((float)(-INFINITY));\n  for (int Lidx105 = 0; Lidx105 < 4; Lidx105++) {\n    float val4 = (*(buf4+0));\n    float val5 = (*(buf6+Lidx105));\n    float alu13 = ((val4<val5)?val5:val4);\n    *(buf4+0) = alu13;\n  }\n  *(buf5+(alu5+lidx0)) = (*(buf0+0));\n  __syncthreads();\n  *(buf1+0) = ((float)(-INFINITY));\n  for (int Lidx106 = 0; Lidx106 < 4; Lidx106++) {\n    for (int Lidx107 = 0; Lidx107 < 4; Lidx107++) {\n      float val6 = (*(buf1+0));\n      float val7 = (*(buf5+((Lidx106<<2)+Lidx107)));\n      float alu19 = ((val6<val7)?val7:val6);\n      *(buf1+0) = alu19;\n    }\n  }\n  *(buf2+0) = 0.0f;\n  for (int Lidx2 = 0; Lidx2 < 32; Lidx2++) {\n    float val8 = (*(data1_1024+(alu0+Lidx2)));\n    *(buf2+0) = ((*(buf2+0))+exp2(((val8-(*(buf1+0)))*1.4426950216293335f)));\n  }\n  int gidx0 = blockIdx.x; /* 32 */\n  int alu26 = (alu0+gidx0);\n  float val9 = (*(data1_1024+alu26));\n  if (((lidx0==0)&(lidx1==0))) {\n    *(data0_1024+alu26) = (exp2(((val9-(*(buf4+0)))*1.4426950216293335f))*(1/(*(buf2+0))));\n  }\n}"
277 Ops.PROGRAM dtypes.void [272, 275, 276] ProgramInfo(global_size=(32, 32, 1), local_size=(4, 4, 1), params=(%1, %11), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CUDA", renderer="CUDA", arch="sm_89"))
278 Ops.SINK dtypes.void [157, 277]

