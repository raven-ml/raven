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
42 Ops.CONST dtypes.weakint [] 163840
43 Ops.PARAM dtypes.f32 [42] ParamArg(slot=1, dtype=dtypes.f32)
44 Ops.CONST dtypes.weakint [] 8
45 Ops.CAST dtypes.i32 [44] dtypes.i32
46 Ops.SHL dtypes.i32 [28, 45]
47 Ops.SHL dtypes.i32 [38, 9]
48 Ops.ADD dtypes.i32 [46, 47]
49 Ops.CONST dtypes.weakint [] 10240
50 Ops.CAST dtypes.i32 [49] dtypes.i32
51 Ops.MUL dtypes.i32 [7, 50]
52 Ops.ADD dtypes.i32 [48, 51]
53 Ops.SHRINK dtypes.f32 [43, 52, 12]
54 Ops.LOAD dtypes.f32 [53]
55 Ops.INDEX dtypes.f32 [54, 9]
56 Ops.CONST dtypes.weakint [] 5120
57 Ops.CAST dtypes.i32 [56] dtypes.i32
58 Ops.ADD dtypes.i32 [52, 57]
59 Ops.SHRINK dtypes.f32 [43, 58, 12]
60 Ops.LOAD dtypes.f32 [59]
61 Ops.INDEX dtypes.f32 [60, 9]
62 Ops.CONST dtypes.weakint [] 3
63 Ops.CAST dtypes.i32 [62] dtypes.i32
64 Ops.INDEX dtypes.f32 [54, 63]
65 Ops.INDEX dtypes.f32 [60, 63]
66 Ops.STACK dtypes.f32 [55, 61, 64, 65]
67 Ops.INDEX dtypes.f32 [66, 9]
68 Ops.CONST dtypes.weakint [] 2560
69 Ops.CAST dtypes.i32 [68] dtypes.i32
70 Ops.ADD dtypes.i32 [52, 69]
71 Ops.SHRINK dtypes.f32 [43, 70, 12]
72 Ops.LOAD dtypes.f32 [71]
73 Ops.INDEX dtypes.f32 [72, 9]
74 Ops.CONST dtypes.weakint [] 7680
75 Ops.CAST dtypes.i32 [74] dtypes.i32
76 Ops.ADD dtypes.i32 [52, 75]
77 Ops.SHRINK dtypes.f32 [43, 76, 12]
78 Ops.LOAD dtypes.f32 [77]
79 Ops.INDEX dtypes.f32 [78, 9]
80 Ops.INDEX dtypes.f32 [72, 63]
81 Ops.INDEX dtypes.f32 [78, 63]
82 Ops.STACK dtypes.f32 [73, 79, 80, 81]
83 Ops.INDEX dtypes.f32 [82, 9]
84 Ops.INDEX dtypes.f32 [66, 63]
85 Ops.INDEX dtypes.f32 [82, 63]
86 Ops.STACK dtypes.f32 [67, 83, 84, 85]
87 Ops.PARAM dtypes.f32 [1] ParamArg(slot=2, dtype=dtypes.f32)
88 Ops.SHRINK dtypes.f32 [87, 10, 12]
89 Ops.LOAD dtypes.f32 [88]
90 Ops.SUB dtypes.f32 [86, 89]
91 Ops.INDEX dtypes.f32 [66, 19]
92 Ops.INDEX dtypes.f32 [82, 19]
93 Ops.CONST dtypes.weakint [] 1
94 Ops.CAST dtypes.i32 [93] dtypes.i32
95 Ops.INDEX dtypes.f32 [66, 94]
96 Ops.INDEX dtypes.f32 [82, 94]
97 Ops.STACK dtypes.f32 [91, 92, 95, 96]
98 Ops.SUB dtypes.f32 [97, 89]
99 Ops.INDEX dtypes.f32 [54, 19]
100 Ops.INDEX dtypes.f32 [60, 19]
101 Ops.INDEX dtypes.f32 [54, 94]
102 Ops.INDEX dtypes.f32 [60, 94]
103 Ops.STACK dtypes.f32 [99, 100, 101, 102]
104 Ops.INDEX dtypes.f32 [103, 9]
105 Ops.INDEX dtypes.f32 [72, 19]
106 Ops.INDEX dtypes.f32 [78, 19]
107 Ops.INDEX dtypes.f32 [72, 94]
108 Ops.INDEX dtypes.f32 [78, 94]
109 Ops.STACK dtypes.f32 [105, 106, 107, 108]
110 Ops.INDEX dtypes.f32 [109, 9]
111 Ops.INDEX dtypes.f32 [103, 63]
112 Ops.INDEX dtypes.f32 [109, 63]
113 Ops.STACK dtypes.f32 [104, 110, 111, 112]
114 Ops.SUB dtypes.f32 [113, 89]
115 Ops.INDEX dtypes.f32 [103, 19]
116 Ops.INDEX dtypes.f32 [109, 19]
117 Ops.INDEX dtypes.f32 [103, 94]
118 Ops.INDEX dtypes.f32 [109, 94]
119 Ops.STACK dtypes.f32 [115, 116, 117, 118]
120 Ops.SUB dtypes.f32 [119, 89]
121 Ops.MUL dtypes.f32 [120, 120]
122 Ops.MULACC dtypes.f32 [114, 114, 121]
123 Ops.MULACC dtypes.f32 [98, 98, 122]
124 Ops.MULACC dtypes.f32 [90, 90, 123]
125 Ops.ADD dtypes.f32 [41, 124]
126 Ops.STORE dtypes.void [40, 125]
127 Ops.AFTER dtypes.void [0, 126]
128 Ops.END dtypes.void [127, 38]
129 Ops.AFTER dtypes.f32 [32, 128]
130 Ops.SHRINK dtypes.f32 [129, 19, 12]
131 Ops.LOAD dtypes.f32 [130]
132 Ops.ADD dtypes.f32 [31, 131]
133 Ops.STORE dtypes.void [30, 132]
134 Ops.AFTER dtypes.void [0, 133]
135 Ops.END dtypes.void [134, 28]
136 Ops.AFTER dtypes.f32 [16, 135]
137 Ops.SHRINK dtypes.f32 [136, 19, 12]
138 Ops.LOAD dtypes.f32 [137]
139 Ops.CONST dtypes.weakfloat [] 0.0003906250058207661
140 Ops.CAST dtypes.f32 [139] dtypes.f32
141 Ops.MUL dtypes.f32 [138, 140]
142 Ops.CONST dtypes.weakfloat [] 9.999999747378752e-06
143 Ops.CAST dtypes.f32 [142] dtypes.f32
144 Ops.ADD dtypes.f32 [141, 143]
145 Ops.SQRT dtypes.f32 [144]
146 Ops.FDIV dtypes.f32 [15, 145]
147 Ops.STORE dtypes.void [13, 146]
148 Ops.AFTER dtypes.void [0, 147]
149 Ops.END dtypes.void [148, 6]
150 Ops.SINK dtypes.void [149] KernelInfo(name="r_16_4_640_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), optimized=True, split=(16, 3, 4))
151 Ops.CONST dtypes.weakint [] 40
152 Ops.PARAM dtypes.u8 [151] ParamArg(slot=0, dtype=dtypes.u8, name="args")
153 Ops.BITCAST dtypes.u64 [152] dtypes.u64
154 Ops.INDEX dtypes.u64 [153, 19]
155 Ops.LOAD dtypes.u64 [154]
156 Ops.REFPARAM dtypes.f32 [1, 155] ParamArg(slot=0, dtype=dtypes.f32)
157 Ops.BITCAST dtypes.i32 [152] dtypes.i32
158 Ops.CONST dtypes.weakint [] 6
159 Ops.CAST dtypes.i32 [158] dtypes.i32
160 Ops.INDEX dtypes.i32 [157, 159]
161 Ops.LOAD dtypes.i32 [160]
162 Ops.CONST dtypes.weakint [] 7
163 Ops.CAST dtypes.i32 [162] dtypes.i32
164 Ops.INDEX dtypes.i32 [157, 163]
165 Ops.LOAD dtypes.i32 [164]
166 Ops.SUB dtypes.i32 [165, 161]
167 Ops.RANGE dtypes.i32 [166] (AxisType.WEAK, 1)
168 Ops.ADD dtypes.i32 [161, 167]
169 Ops.SHL dtypes.i32 [168, 9]
170 Ops.SHRINK dtypes.f32 [156, 169, 12]
171 Ops.BUFFER dtypes.f32 [11] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
172 Ops.AFTER dtypes.f32 [171, 168]
173 Ops.SHRINK dtypes.f32 [172, 19, 12]
174 Ops.STORE dtypes.void [173, 23]
175 Ops.AFTER dtypes.i32 [26, 167]
176 Ops.RANGE dtypes.i32 [175] (AxisType.WEAK, 0, 0, 0)
177 Ops.AFTER dtypes.f32 [171, 174, 176]
178 Ops.SHRINK dtypes.f32 [177, 19, 12]
179 Ops.LOAD dtypes.f32 [178]
180 Ops.BUFFER dtypes.f32 [11] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
181 Ops.AFTER dtypes.f32 [180, 168, 176]
182 Ops.SHRINK dtypes.f32 [181, 19, 12]
183 Ops.STORE dtypes.void [182, 23]
184 Ops.AFTER dtypes.i32 [36, 176]
185 Ops.RANGE dtypes.i32 [184] (AxisType.WEAK, 0, 0, 0, 256)
186 Ops.AFTER dtypes.f32 [180, 183, 185]
187 Ops.SHRINK dtypes.f32 [186, 19, 12]
188 Ops.LOAD dtypes.f32 [187]
189 Ops.INDEX dtypes.u64 [153, 94]
190 Ops.LOAD dtypes.u64 [189]
191 Ops.REFPARAM dtypes.f32 [42, 190] ParamArg(slot=1, dtype=dtypes.f32)
192 Ops.SHL dtypes.i32 [176, 45]
193 Ops.SHL dtypes.i32 [185, 9]
194 Ops.ADD dtypes.i32 [192, 193]
195 Ops.MUL dtypes.i32 [168, 50]
196 Ops.ADD dtypes.i32 [194, 195]
197 Ops.SHRINK dtypes.f32 [191, 196, 12]
198 Ops.LOAD dtypes.f32 [197]
199 Ops.INDEX dtypes.f32 [198, 9]
200 Ops.ADD dtypes.i32 [196, 57]
201 Ops.SHRINK dtypes.f32 [191, 200, 12]
202 Ops.LOAD dtypes.f32 [201]
203 Ops.INDEX dtypes.f32 [202, 9]
204 Ops.INDEX dtypes.f32 [198, 63]
205 Ops.INDEX dtypes.f32 [202, 63]
206 Ops.STACK dtypes.f32 [199, 203, 204, 205]
207 Ops.INDEX dtypes.f32 [206, 9]
208 Ops.ADD dtypes.i32 [196, 69]
209 Ops.SHRINK dtypes.f32 [191, 208, 12]
210 Ops.LOAD dtypes.f32 [209]
211 Ops.INDEX dtypes.f32 [210, 9]
212 Ops.ADD dtypes.i32 [196, 75]
213 Ops.SHRINK dtypes.f32 [191, 212, 12]
214 Ops.LOAD dtypes.f32 [213]
215 Ops.INDEX dtypes.f32 [214, 9]
216 Ops.INDEX dtypes.f32 [210, 63]
217 Ops.INDEX dtypes.f32 [214, 63]
218 Ops.STACK dtypes.f32 [211, 215, 216, 217]
219 Ops.INDEX dtypes.f32 [218, 9]
220 Ops.INDEX dtypes.f32 [206, 63]
221 Ops.INDEX dtypes.f32 [218, 63]
222 Ops.STACK dtypes.f32 [207, 219, 220, 221]
223 Ops.INDEX dtypes.u64 [153, 9]
224 Ops.LOAD dtypes.u64 [223]
225 Ops.REFPARAM dtypes.f32 [1, 224] ParamArg(slot=2, dtype=dtypes.f32)
226 Ops.SHRINK dtypes.f32 [225, 169, 12]
227 Ops.LOAD dtypes.f32 [226]
228 Ops.SUB dtypes.f32 [222, 227]
229 Ops.INDEX dtypes.f32 [206, 19]
230 Ops.INDEX dtypes.f32 [218, 19]
231 Ops.INDEX dtypes.f32 [206, 94]
232 Ops.INDEX dtypes.f32 [218, 94]
233 Ops.STACK dtypes.f32 [229, 230, 231, 232]
234 Ops.SUB dtypes.f32 [233, 227]
235 Ops.INDEX dtypes.f32 [198, 19]
236 Ops.INDEX dtypes.f32 [202, 19]
237 Ops.INDEX dtypes.f32 [198, 94]
238 Ops.INDEX dtypes.f32 [202, 94]
239 Ops.STACK dtypes.f32 [235, 236, 237, 238]
240 Ops.INDEX dtypes.f32 [239, 9]
241 Ops.INDEX dtypes.f32 [210, 19]
242 Ops.INDEX dtypes.f32 [214, 19]
243 Ops.INDEX dtypes.f32 [210, 94]
244 Ops.INDEX dtypes.f32 [214, 94]
245 Ops.STACK dtypes.f32 [241, 242, 243, 244]
246 Ops.INDEX dtypes.f32 [245, 9]
247 Ops.INDEX dtypes.f32 [239, 63]
248 Ops.INDEX dtypes.f32 [245, 63]
249 Ops.STACK dtypes.f32 [240, 246, 247, 248]
250 Ops.SUB dtypes.f32 [249, 227]
251 Ops.INDEX dtypes.f32 [239, 19]
252 Ops.INDEX dtypes.f32 [245, 19]
253 Ops.INDEX dtypes.f32 [239, 94]
254 Ops.INDEX dtypes.f32 [245, 94]
255 Ops.STACK dtypes.f32 [251, 252, 253, 254]
256 Ops.SUB dtypes.f32 [255, 227]
257 Ops.MUL dtypes.f32 [256, 256]
258 Ops.MULACC dtypes.f32 [250, 250, 257]
259 Ops.MULACC dtypes.f32 [234, 234, 258]
260 Ops.MULACC dtypes.f32 [228, 228, 259]
261 Ops.ADD dtypes.f32 [188, 260]
262 Ops.STORE dtypes.void [187, 261]
263 Ops.AFTER dtypes.void [0, 262]
264 Ops.END dtypes.void [263, 185]
265 Ops.AFTER dtypes.f32 [180, 264]
266 Ops.SHRINK dtypes.f32 [265, 19, 12]
267 Ops.LOAD dtypes.f32 [266]
268 Ops.ADD dtypes.f32 [179, 267]
269 Ops.STORE dtypes.void [178, 268]
270 Ops.AFTER dtypes.void [0, 269]
271 Ops.END dtypes.void [270, 176]
272 Ops.AFTER dtypes.f32 [171, 271]
273 Ops.SHRINK dtypes.f32 [272, 19, 12]
274 Ops.LOAD dtypes.f32 [273]
275 Ops.MUL dtypes.f32 [274, 140]
276 Ops.ADD dtypes.f32 [275, 143]
277 Ops.SQRT dtypes.f32 [276]
278 Ops.FDIV dtypes.f32 [15, 277]
279 Ops.STORE dtypes.void [170, 278]
280 Ops.AFTER dtypes.void [0, 279]
281 Ops.END dtypes.void [280, 167]
282 Ops.SINK dtypes.void [281] KernelInfo(name="r_16_4_640_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), estimates=Estimates(ops=492416, lds=655872, mem=655872), optimized=True, split=(16, 3, 4))
283 Ops.LINEAR dtypes.void [151, 152, 11, 180, 171, 158, 159, 157, 160, 161, 162, 163, 164, 165, 18, 19, 153, 154, 155, 93, 94, 189, 190, 8, 9, 223, 224, 1, 156, 42, 191, 225, 0, 21, 22, 23, 139, 140, 14, 15, 142, 143, 25, 26, 49, 50, 68, 69, 62, 63, 12, 56, 57, 36, 74, 75, 44, 45, 166, 167, 168, 169, 226, 227, 175, 172, 173, 195, 174, 176, 184, 181, 182, 192, 183, 185, 193, 194, 196, 208, 209, 210, 200, 201, 202, 212, 213, 214, 197, 198, 186, 187, 188, 241, 243, 242, 244, 245, 236, 238, 235, 237, 239, 252, 254, 251, 253, 255, 246, 248, 240, 247, 249, 211, 216, 215, 217, 218, 203, 205, 199, 204, 206, 230, 232, 229, 231, 233, 219, 221, 207, 220, 222, 256, 257, 250, 234, 228, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 177, 178, 179, 268, 269, 270, 271, 272, 273, 274, 170, 275, 276, 277, 278, 279, 280, 281, 282]
284 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_16_4_640_4(unsigned char* restrict args_40) {\n  float buf0[4];\n  float buf1[4];\n  int val0 = (*(((int*)(args_40))+6));\n  int val1 = (*(((int*)(args_40))+7));\n  unsigned long val2 = (*(((unsigned long*)(args_40))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_40))+1));\n  unsigned long val4 = (*(((unsigned long*)(args_40))+2));\n  float4 cast0 = (float4){0.0f,0.0f,0.0f,0.0f};\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    int alu1 = (alu0<<2);\n    float4 val5 = (*((float4*)((((float*)(val4))+alu1))));\n    *((float4*)((buf1+0))) = cast0;\n    for (int Lidx0_0_0 = 0; Lidx0_0_0 < 10; Lidx0_0_0++) {\n      *((float4*)((buf0+0))) = cast0;\n      for (int Lidx0_0_0_256 = 0; Lidx0_0_0_256 < 64; Lidx0_0_0_256++) {\n        int alu4 = ((Lidx0_0_0<<8)+(Lidx0_0_0_256<<2)+(alu0*10240));\n        float4 val6 = (*((float4*)((((float*)(val3))+(alu4+2560)))));\n        float4 val7 = (*((float4*)((((float*)(val3))+(alu4+5120)))));\n        float4 val8 = (*((float4*)((((float*)(val3))+(alu4+7680)))));\n        float4 val9 = (*((float4*)((((float*)(val3))+alu4))));\n        float4 cast1 = ((float4)__builtin_shufflevector(val6, val8, 0, 4, 1, 5));\n        float4 cast2 = ((float4)__builtin_shufflevector(val9, val7, 0, 4, 1, 5));\n        float4 cast3 = ((float4)__builtin_shufflevector(val6, val8, 2, 6, 3, 7));\n        float4 cast4 = ((float4)__builtin_shufflevector(val9, val7, 2, 6, 3, 7));\n        float4 alu5 = (((float4)__builtin_shufflevector(cast2, cast1, 0, 4, 1, 5))-val5);\n        float4 alu6 = (((float4)__builtin_shufflevector(cast2, cast1, 2, 6, 3, 7))-val5);\n        float4 alu7 = (((float4)__builtin_shufflevector(cast4, cast3, 0, 4, 1, 5))-val5);\n        float4 alu8 = (((float4)__builtin_shufflevector(cast4, cast3, 2, 6, 3, 7))-val5);\n        *((float4*)((buf0+0))) = ((*((float4*)((buf0+0))))+__builtin_elementwise_fma(alu8, alu8, __builtin_elementwise_fma(alu7, alu7, __builtin_elementwise_fma(alu6, alu6, (alu5*alu5)))));\n      }\n      *((float4*)((buf1+0))) = ((*((float4*)((buf1+0))))+(*((float4*)((buf0+0)))));\n    }\n    *((float4*)((((float*)(val2))+alu1))) = (1.0f/__builtin_elementwise_sqrt((((*((float4*)((buf1+0))))*0.0003906250058207661f)+9.999999747378752e-06f)));\n  }\n}\n"
285 Ops.PROGRAM dtypes.void [282, 283, 284] ProgramInfo(global_size=(16, 1, 1), params=(%2, %43, %87, %3, %4), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
286 Ops.SINK dtypes.void [150, 285]

