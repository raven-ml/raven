# tolk D25 D89 golden=9bf87d3c15fbc585923a32c4ec26605c
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
224 Ops.BUFFER dtypes.f32 [13] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
225 Ops.AFTER dtypes.f32 [224, 4, 10]
226 Ops.INDEX dtypes.f32 [225, 17]
227 Ops.STORE dtypes.void [226, 20]
228 Ops.BUFFER dtypes.f32 [13] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
229 Ops.AFTER dtypes.f32 [228, 25, 4, 10]
230 Ops.INDEX dtypes.f32 [229, 17]
231 Ops.STORE dtypes.void [230, 20]
232 Ops.AFTER dtypes.f32 [228, 231, 29]
233 Ops.INDEX dtypes.f32 [232, 17]
234 Ops.LOAD dtypes.f32 [233]
235 Ops.ADD dtypes.f32 [234, 183]
236 Ops.STORE dtypes.void [233, 235]
237 Ops.END dtypes.void [236, 29]
238 Ops.AFTER dtypes.i32 [23, 237]
239 Ops.RANGE dtypes.i32 [238] (AxisType.WEAK, 102)
240 Ops.AFTER dtypes.f32 [224, 227, 239]
241 Ops.INDEX dtypes.f32 [240, 17]
242 Ops.LOAD dtypes.f32 [241]
243 Ops.BUFFER dtypes.f32 [140] ParamArg(slot=2, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
244 Ops.INDEX dtypes.f32 [243, 196]
245 Ops.AFTER dtypes.f32 [228, 237]
246 Ops.INDEX dtypes.f32 [245, 17]
247 Ops.LOAD dtypes.f32 [246]
248 Ops.STORE dtypes.void [244, 247]
249 Ops.BARRIER dtypes.void [248]
250 Ops.AFTER dtypes.f32 [243, 249]
251 Ops.ADD dtypes.i32 [195, 239]
252 Ops.INDEX dtypes.f32 [250, 251]
253 Ops.LOAD dtypes.f32 [252]
254 Ops.ADD dtypes.f32 [242, 253]
255 Ops.STORE dtypes.void [241, 254]
256 Ops.END dtypes.void [255, 239]
257 Ops.AFTER dtypes.f32 [224, 256]
258 Ops.INDEX dtypes.f32 [257, 17]
259 Ops.LOAD dtypes.f32 [258]
260 Ops.MUL dtypes.f32 [259, 214]
261 Ops.ADD dtypes.f32 [260, 217]
262 Ops.SQRT dtypes.f32 [261]
263 Ops.RECIP dtypes.f32 [262]
264 Ops.STORE dtypes.void [12, 263]
265 Ops.SINK dtypes.void [264] KernelInfo(name="r_16_32_4_4_20", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(32, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=3, arg=(20, AxisType.UPCAST))), estimates=Estimates(ops=567296, lds=934144, mem=655872), optimized=True)
266 Ops.IF dtypes.void [221, 12]
267 Ops.ENDIF dtypes.void [266]
268 Ops.LINEAR dtypes.void [0, 1, 33, 34, 52, 13, 228, 224, 140, 243, 2, 3, 4, 8, 9, 10, 5, 6, 7, 11, 53, 54, 22, 23, 25, 229, 16, 17, 230, 19, 20, 35, 36, 141, 134, 135, 128, 129, 122, 123, 116, 117, 38, 39, 110, 111, 104, 105, 98, 99, 92, 93, 86, 87, 80, 81, 74, 75, 68, 69, 62, 63, 56, 57, 47, 48, 152, 42, 43, 146, 147, 37, 40, 41, 231, 29, 44, 45, 46, 142, 143, 144, 136, 137, 138, 130, 131, 132, 124, 125, 126, 118, 119, 120, 112, 113, 114, 157, 158, 159, 106, 107, 108, 100, 101, 102, 94, 95, 96, 88, 89, 90, 82, 83, 84, 76, 77, 78, 70, 71, 72, 64, 65, 66, 58, 59, 60, 49, 50, 51, 153, 154, 155, 148, 149, 150, 161, 162, 232, 233, 234, 163, 164, 145, 139, 133, 127, 121, 115, 160, 109, 103, 97, 91, 85, 79, 73, 67, 61, 55, 156, 151, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 235, 236, 237, 245, 246, 247, 238, 225, 193, 194, 195, 196, 244, 248, 249, 250, 226, 227, 239, 240, 241, 242, 251, 252, 253, 254, 255, 256, 257, 258, 259, 12, 213, 214, 216, 217, 260, 261, 262, 263, 221, 266, 264, 267, 265]
269 Ops.SOURCE dtypes.void [] "typedef unsigned int uint;\n#define INFINITY (__int_as_float(0x7f800000))\n#define NAN (__int_as_float(0x7fffffff))\ntemplate <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }\nextern \"C\" __global__ void __launch_bounds__(128) r_16_32_4_4_20(float* data0_64, float* data1_163840, float* data2_64) {\n  float buf0[1];\n  float buf1[1];\n  __shared__ __align__(16) float buf2[128];\n  int gidx0 = blockIdx.x; /* 16 */\n  int lidx1 = threadIdx.y; /* 4 */\n  int alu0 = ((gidx0<<2)+lidx1);\n  float val0 = (*(data2_64+alu0));\n  int lidx0 = threadIdx.x; /* 32 */\n  *(buf0+0) = 0.0f;\n  for (int Lidx0_0_0 = 0; Lidx0_0_0 < 4; Lidx0_0_0++) {\n    int alu2 = ((gidx0*10240)+(lidx1*2560)+((Lidx0_0_0*640)+lidx0));\n    float val1 = (*(data1_163840+(alu2+128)));\n    float val2 = (*(data1_163840+(alu2+160)));\n    float val3 = (*(data1_163840+(alu2+192)));\n    float val4 = (*(data1_163840+(alu2+224)));\n    float val5 = (*(data1_163840+(alu2+256)));\n    float val6 = (*(data1_163840+(alu2+288)));\n    float val7 = (*(data1_163840+(alu2+32)));\n    float val8 = (*(data1_163840+(alu2+320)));\n    float val9 = (*(data1_163840+(alu2+352)));\n    float val10 = (*(data1_163840+(alu2+384)));\n    float val11 = (*(data1_163840+(alu2+416)));\n    float val12 = (*(data1_163840+(alu2+448)));\n    float val13 = (*(data1_163840+(alu2+480)));\n    float val14 = (*(data1_163840+(alu2+512)));\n    float val15 = (*(data1_163840+(alu2+544)));\n    float val16 = (*(data1_163840+(alu2+576)));\n    float val17 = (*(data1_163840+(alu2+608)));\n    float val18 = (*(data1_163840+(alu2+64)));\n    float val19 = (*(data1_163840+(alu2+96)));\n    float val20 = (*(data1_163840+alu2));\n    float alu3 = (val20-val0);\n    float alu4 = (val1-val0);\n    float alu5 = (val2-val0);\n    float alu6 = (val3-val0);\n    float alu7 = (val4-val0);\n    float alu8 = (val5-val0);\n    float alu9 = (val6-val0);\n    float alu10 = (val7-val0);\n    float alu11 = (val8-val0);\n    float alu12 = (val9-val0);\n    float alu13 = (val10-val0);\n    float alu14 = (val11-val0);\n    float alu15 = (val12-val0);\n    float alu16 = (val13-val0);\n    float alu17 = (val14-val0);\n    float alu18 = (val15-val0);\n    float alu19 = (val16-val0);\n    float alu20 = (val17-val0);\n    float alu21 = (val18-val0);\n    float alu22 = (val19-val0);\n    *(buf0+0) = ((*(buf0+0))+__fmaf_rn(alu20,alu20,__fmaf_rn(alu19,alu19,__fmaf_rn(alu18,alu18,__fmaf_rn(alu17,alu17,__fmaf_rn(alu16,alu16,__fmaf_rn(alu15,alu15,__fmaf_rn(alu14,alu14,__fmaf_rn(alu13,alu13,__fmaf_rn(alu12,alu12,__fmaf_rn(alu11,alu11,__fmaf_rn(alu9,alu9,__fmaf_rn(alu8,alu8,__fmaf_rn(alu7,alu7,__fmaf_rn(alu6,alu6,__fmaf_rn(alu5,alu5,__fmaf_rn(alu4,alu4,__fmaf_rn(alu22,alu22,__fmaf_rn(alu21,alu21,__fmaf_rn(alu10,alu10,(alu3*alu3)))))))))))))))))))));\n  }\n  int alu25 = (lidx1<<5);\n  *(buf2+(alu25+lidx0)) = (*(buf0+0));\n  __syncthreads();\n  *(buf1+0) = 0.0f;\n  for (int Lidx102 = 0; Lidx102 < 32; Lidx102++) {\n    float val21 = (*(buf2+(alu25+Lidx102)));\n    *(buf1+0) = ((*(buf1+0))+val21);\n  }\n  if ((lidx0==0)) {\n    *(data0_64+alu0) = (1/sqrt((((*(buf1+0))*0.0003906250058207661f)+9.999999747378752e-06f)));\n  }\n}"
270 Ops.PROGRAM dtypes.void [265, 268, 269] ProgramInfo(global_size=(16, 1, 1), local_size=(32, 4, 1), params=(%1, %34, %52), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CUDA", renderer="CUDA", arch="sm_89"))
271 Ops.SINK dtypes.void [223, 270]

