# tolk D25 D80 D141 D151 golden=52bd40bb969e8c45fc56f6bc8321c6a6
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 96
2 Ops.CONST dtypes.weakint [] 32
3 Ops.PARAM dtypes.u8 [2] ParamArg(slot=0, dtype=dtypes.u8, name="args")
4 Ops.BITCAST dtypes.u64 [3] dtypes.u64
5 Ops.CONST dtypes.weakint [] 0
6 Ops.CAST dtypes.i32 [5] dtypes.i32
7 Ops.INDEX dtypes.u64 [4, 6]
8 Ops.LOAD dtypes.u64 [7]
9 Ops.REFPARAM dtypes.f32 [1, 8] ParamArg(slot=0, dtype=dtypes.f32)
10 Ops.CONST dtypes.weakint [] 2
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.RANGE dtypes.i32 [11] (AxisType.WEAK, 0)
13 Ops.CONST dtypes.weakint [] 48
14 Ops.CAST dtypes.i32 [13] dtypes.i32
15 Ops.MUL dtypes.i32 [12, 14]
16 Ops.BITCAST dtypes.i32 [3] dtypes.i32
17 Ops.CONST dtypes.weakint [] 4
18 Ops.CAST dtypes.i32 [17] dtypes.i32
19 Ops.INDEX dtypes.i32 [16, 18]
20 Ops.LOAD dtypes.i32 [19]
21 Ops.CONST dtypes.weakint [] 5
22 Ops.CAST dtypes.i32 [21] dtypes.i32
23 Ops.INDEX dtypes.i32 [16, 22]
24 Ops.LOAD dtypes.i32 [23]
25 Ops.SUB dtypes.i32 [24, 20]
26 Ops.AFTER dtypes.i32 [25, 12]
27 Ops.RANGE dtypes.i32 [26] (AxisType.WEAK, 1)
28 Ops.ADD dtypes.i32 [20, 27]
29 Ops.SHL dtypes.i32 [28, 18]
30 Ops.ADD dtypes.i32 [15, 29]
31 Ops.SHRINK dtypes.f32 [9, 30, 18]
32 Ops.CONST dtypes.weakint [] 1
33 Ops.CAST dtypes.i32 [32] dtypes.i32
34 Ops.INDEX dtypes.u64 [4, 33]
35 Ops.LOAD dtypes.u64 [34]
36 Ops.REFPARAM dtypes.f32 [1, 35] ParamArg(slot=1, dtype=dtypes.f32)
37 Ops.ADD dtypes.i32 [15, 28]
38 Ops.INDEX dtypes.f32 [36, 37]
39 Ops.LOAD dtypes.f32 [38]
40 Ops.CONST dtypes.weakint [] 3
41 Ops.CAST dtypes.i32 [40] dtypes.i32
42 Ops.ADD dtypes.i32 [37, 41]
43 Ops.INDEX dtypes.f32 [36, 42]
44 Ops.LOAD dtypes.f32 [43]
45 Ops.CONST dtypes.weakint [] 6
46 Ops.CAST dtypes.i32 [45] dtypes.i32
47 Ops.ADD dtypes.i32 [37, 46]
48 Ops.INDEX dtypes.f32 [36, 47]
49 Ops.LOAD dtypes.f32 [48]
50 Ops.CONST dtypes.weakint [] 9
51 Ops.CAST dtypes.i32 [50] dtypes.i32
52 Ops.ADD dtypes.i32 [37, 51]
53 Ops.INDEX dtypes.f32 [36, 52]
54 Ops.LOAD dtypes.f32 [53]
55 Ops.STACK dtypes.f32 [39, 44, 49, 54]
56 Ops.STORE dtypes.void [31, 55]
57 Ops.ADD dtypes.i32 [30, 18]
58 Ops.SHRINK dtypes.f32 [9, 57, 18]
59 Ops.CONST dtypes.weakint [] 12
60 Ops.CAST dtypes.i32 [59] dtypes.i32
61 Ops.ADD dtypes.i32 [37, 60]
62 Ops.INDEX dtypes.f32 [36, 61]
63 Ops.LOAD dtypes.f32 [62]
64 Ops.CONST dtypes.weakint [] 15
65 Ops.CAST dtypes.i32 [64] dtypes.i32
66 Ops.ADD dtypes.i32 [37, 65]
67 Ops.INDEX dtypes.f32 [36, 66]
68 Ops.LOAD dtypes.f32 [67]
69 Ops.CONST dtypes.weakint [] 18
70 Ops.CAST dtypes.i32 [69] dtypes.i32
71 Ops.ADD dtypes.i32 [37, 70]
72 Ops.INDEX dtypes.f32 [36, 71]
73 Ops.LOAD dtypes.f32 [72]
74 Ops.CONST dtypes.weakint [] 21
75 Ops.CAST dtypes.i32 [74] dtypes.i32
76 Ops.ADD dtypes.i32 [37, 75]
77 Ops.INDEX dtypes.f32 [36, 76]
78 Ops.LOAD dtypes.f32 [77]
79 Ops.STACK dtypes.f32 [63, 68, 73, 78]
80 Ops.STORE dtypes.void [58, 79]
81 Ops.CONST dtypes.weakint [] 8
82 Ops.CAST dtypes.i32 [81] dtypes.i32
83 Ops.ADD dtypes.i32 [30, 82]
84 Ops.SHRINK dtypes.f32 [9, 83, 18]
85 Ops.CONST dtypes.weakint [] 24
86 Ops.CAST dtypes.i32 [85] dtypes.i32
87 Ops.ADD dtypes.i32 [37, 86]
88 Ops.INDEX dtypes.f32 [36, 87]
89 Ops.LOAD dtypes.f32 [88]
90 Ops.CONST dtypes.weakint [] 27
91 Ops.CAST dtypes.i32 [90] dtypes.i32
92 Ops.ADD dtypes.i32 [37, 91]
93 Ops.INDEX dtypes.f32 [36, 92]
94 Ops.LOAD dtypes.f32 [93]
95 Ops.CONST dtypes.weakint [] 30
96 Ops.CAST dtypes.i32 [95] dtypes.i32
97 Ops.ADD dtypes.i32 [37, 96]
98 Ops.INDEX dtypes.f32 [36, 97]
99 Ops.LOAD dtypes.f32 [98]
100 Ops.CONST dtypes.weakint [] 33
101 Ops.CAST dtypes.i32 [100] dtypes.i32
102 Ops.ADD dtypes.i32 [37, 101]
103 Ops.INDEX dtypes.f32 [36, 102]
104 Ops.LOAD dtypes.f32 [103]
105 Ops.STACK dtypes.f32 [89, 94, 99, 104]
106 Ops.STORE dtypes.void [84, 105]
107 Ops.ADD dtypes.i32 [30, 60]
108 Ops.SHRINK dtypes.f32 [9, 107, 18]
109 Ops.CONST dtypes.weakint [] 36
110 Ops.CAST dtypes.i32 [109] dtypes.i32
111 Ops.ADD dtypes.i32 [37, 110]
112 Ops.INDEX dtypes.f32 [36, 111]
113 Ops.LOAD dtypes.f32 [112]
114 Ops.CONST dtypes.weakint [] 39
115 Ops.CAST dtypes.i32 [114] dtypes.i32
116 Ops.ADD dtypes.i32 [37, 115]
117 Ops.INDEX dtypes.f32 [36, 116]
118 Ops.LOAD dtypes.f32 [117]
119 Ops.CONST dtypes.weakint [] 42
120 Ops.CAST dtypes.i32 [119] dtypes.i32
121 Ops.ADD dtypes.i32 [37, 120]
122 Ops.INDEX dtypes.f32 [36, 121]
123 Ops.LOAD dtypes.f32 [122]
124 Ops.CONST dtypes.weakint [] 45
125 Ops.CAST dtypes.i32 [124] dtypes.i32
126 Ops.ADD dtypes.i32 [37, 125]
127 Ops.INDEX dtypes.f32 [36, 126]
128 Ops.LOAD dtypes.f32 [127]
129 Ops.STACK dtypes.f32 [113, 118, 123, 128]
130 Ops.STORE dtypes.void [108, 129]
131 Ops.AFTER dtypes.void [0, 56, 80, 106, 130]
132 Ops.END dtypes.void [131, 27]
133 Ops.END dtypes.void [132, 12]
134 Ops.SINK dtypes.void [133] KernelInfo(name="E_2_3_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(16, AxisType.UPCAST)),), estimates=Estimates(lds=768, mem=768), optimized=True, split=(3, 2, 3))
135 Ops.LINEAR dtypes.void [2, 3, 17, 18, 16, 19, 20, 21, 22, 23, 24, 5, 6, 4, 7, 8, 32, 33, 34, 35, 1, 9, 36, 0, 59, 60, 64, 65, 69, 70, 10, 11, 74, 75, 85, 86, 90, 91, 40, 41, 95, 96, 100, 101, 109, 110, 114, 115, 119, 120, 124, 125, 13, 14, 45, 46, 81, 82, 50, 51, 25, 12, 26, 15, 27, 28, 37, 61, 62, 63, 66, 67, 68, 71, 72, 73, 76, 77, 78, 87, 88, 89, 92, 93, 94, 42, 43, 44, 97, 98, 99, 102, 103, 104, 111, 112, 113, 116, 117, 118, 121, 122, 123, 126, 127, 128, 47, 48, 49, 52, 53, 54, 38, 39, 79, 105, 129, 55, 29, 30, 107, 108, 57, 58, 83, 84, 31, 130, 80, 106, 56, 131, 132, 133, 134]
136 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void E_2_3_16(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx0 = 0; Lidx0 < 2; Lidx0++) {\n    int alu0 = (Lidx0*48);\n    for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n      int alu1 = (val0+Lidx1);\n      int alu2 = (alu0+alu1);\n      float val4 = (*(((float*)(val3))+(alu2+12)));\n      float val5 = (*(((float*)(val3))+(alu2+15)));\n      float val6 = (*(((float*)(val3))+(alu2+18)));\n      float val7 = (*(((float*)(val3))+(alu2+21)));\n      float val8 = (*(((float*)(val3))+(alu2+24)));\n      float val9 = (*(((float*)(val3))+(alu2+27)));\n      float val10 = (*(((float*)(val3))+(alu2+3)));\n      float val11 = (*(((float*)(val3))+(alu2+30)));\n      float val12 = (*(((float*)(val3))+(alu2+33)));\n      float val13 = (*(((float*)(val3))+(alu2+36)));\n      float val14 = (*(((float*)(val3))+(alu2+39)));\n      float val15 = (*(((float*)(val3))+(alu2+42)));\n      float val16 = (*(((float*)(val3))+(alu2+45)));\n      float val17 = (*(((float*)(val3))+(alu2+6)));\n      float val18 = (*(((float*)(val3))+(alu2+9)));\n      float val19 = (*(((float*)(val3))+alu2));\n      int alu3 = (alu0+(alu1<<4));\n      *((float4*)((((float*)(val2))+(alu3+12)))) = (float4){val13,val14,val15,val16};\n      *((float4*)((((float*)(val2))+(alu3+4)))) = (float4){val4,val5,val6,val7};\n      *((float4*)((((float*)(val2))+(alu3+8)))) = (float4){val8,val9,val11,val12};\n      *((float4*)((((float*)(val2))+alu3))) = (float4){val19,val10,val17,val18};\n    }\n  }\n}\n"
137 Ops.BINARY dtypes.u8 [] b"typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void E_2_3_16(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx0 = 0; Lidx0 < 2; Lidx0++) {\n    int alu0 = (Lidx0*48);\n    for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n      int alu1 = (val0+Lidx1);\n      int alu2 = (alu0+alu1);\n      float val4 = (*(((float*)(val3))+(alu2+12)));\n      float val5 = (*(((float*)(val3))+(alu2+15)));\n      float val6 = (*(((float*)(val3))+(alu2+18)));\n      float val7 = (*(((float*)(val3))+(alu2+21)));\n      float val8 = (*(((float*)(val3))+(alu2+24)));\n      float val9 = (*(((float*)(val3))+(alu2+27)));\n      float val10 = (*(((float*)(val3))+(alu2+3)));\n      float val11 = (*(((float*)(val3))+(alu2+30)));\n      float val12 = (*(((float*)(val3))+(alu2+33)));\n      float val13 = (*(((float*)(val3))+(alu2+36)));\n      float val14 = (*(((float*)(val3))+(alu2+39)));\n      float val15 = (*(((float*)(val3))+(alu2+42)));\n      float val16 = (*(((float*)(val3))+(alu2+45)));\n      float val17 = (*(((float*)(val3))+(alu2+6)));\n      float val18 = (*(((float*)(val3))+(alu2+9)));\n      float val19 = (*(((float*)(val3))+alu2));\n      int alu3 = (alu0+(alu1<<4));\n      *((float4*)((((float*)(val2))+(alu3+12)))) = (float4){val13,val14,val15,val16};\n      *((float4*)((((float*)(val2))+(alu3+4)))) = (float4){val4,val5,val6,val7};\n      *((float4*)((((float*)(val2))+(alu3+8)))) = (float4){val8,val9,val11,val12};\n      *((float4*)((((float*)(val2))+alu3))) = (float4){val19,val10,val17,val18};\n    }\n  }\n}\n"
138 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
139 Ops.PARAM dtypes.f32 [1] ParamArg(slot=1, dtype=dtypes.f32)
140 Ops.PARAM dtypes.i32 [0] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 3), name="block_lo", addrspace=AddrSpace.ALU)
141 Ops.PARAM dtypes.i32 [0] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 3), name="block_hi", addrspace=AddrSpace.ALU)
142 Ops.PROGRAM dtypes.void [134, 135, 136, 137] ProgramInfo(global_size=(3, 1, 1), params=(%138, %139, %140, %141), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
143 Ops.CONST dtypes.weakint [] 512
144 Ops.ALLOC dtypes.i8 [143] ParamArg(slot=25, dtype=dtypes.i8, device="CPU")
145 Ops.CONST dtypes.weakint [] 384
146 Ops.SHRINK dtypes.i8 [144, 5, 145]
147 Ops.BITCAST dtypes.f32 [146] dtypes.f32
148 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
149 Ops.CALL dtypes.void [142, 147, 148] CallInfo()
150 Ops.CONST dtypes.weakint [] 40
151 Ops.PARAM dtypes.u8 [150] ParamArg(slot=0, dtype=dtypes.u8, name="args")
152 Ops.BITCAST dtypes.u64 [151] dtypes.u64
153 Ops.INDEX dtypes.u64 [152, 6]
154 Ops.LOAD dtypes.u64 [153]
155 Ops.REFPARAM dtypes.f32 [32, 154] ParamArg(slot=0, dtype=dtypes.f32)
156 Ops.INDEX dtypes.f32 [155, 6]
157 Ops.BUFFER dtypes.f32 [10] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
158 Ops.SHRINK dtypes.f32 [157, 6, 11]
159 Ops.CONST dtypes.weakfloat [] 0.0
160 Ops.CAST dtypes.f32 [159] dtypes.f32
161 Ops.STACK dtypes.f32 [160, 160]
162 Ops.STORE dtypes.void [158, 161]
163 Ops.RANGE dtypes.i32 [18] (AxisType.WEAK, 3, 0, 0)
164 Ops.AFTER dtypes.i32 [18, 163]
165 Ops.RANGE dtypes.i32 [164] (AxisType.WEAK, 3, 0, 1)
166 Ops.AFTER dtypes.f32 [157, 162, 163, 165]
167 Ops.SHRINK dtypes.f32 [166, 6, 11]
168 Ops.LOAD dtypes.f32 [167]
169 Ops.INDEX dtypes.f32 [168, 6]
170 Ops.BUFFER dtypes.f32 [81] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
171 Ops.AFTER dtypes.f32 [170, 165, 163]
172 Ops.SHRINK dtypes.f32 [171, 6, 18]
173 Ops.STACK dtypes.f32 [160, 160, 160, 160]
174 Ops.STORE dtypes.void [172, 173]
175 Ops.SHRINK dtypes.f32 [171, 18, 18]
176 Ops.STORE dtypes.void [175, 173]
177 Ops.AFTER dtypes.i32 [41, 165]
178 Ops.RANGE dtypes.i32 [177] (AxisType.WEAK, 0)
179 Ops.AFTER dtypes.i32 [41, 178]
180 Ops.RANGE dtypes.i32 [179] (AxisType.WEAK, 1)
181 Ops.AFTER dtypes.i32 [41, 180]
182 Ops.RANGE dtypes.i32 [181] (AxisType.WEAK, 2)
183 Ops.AFTER dtypes.f32 [170, 174, 176, 178, 180, 182]
184 Ops.SHRINK dtypes.f32 [183, 6, 18]
185 Ops.INDEX dtypes.u64 [152, 33]
186 Ops.LOAD dtypes.u64 [185]
187 Ops.REFPARAM dtypes.f32 [1, 186] ParamArg(slot=1, dtype=dtypes.f32)
188 Ops.SHL dtypes.i32 [180, 11]
189 Ops.SHL dtypes.i32 [165, 11]
190 Ops.ADD dtypes.i32 [188, 189]
191 Ops.ADD dtypes.i32 [190, 182]
192 Ops.SHL dtypes.i32 [178, 18]
193 Ops.ADD dtypes.i32 [191, 192]
194 Ops.CONST dtypes.weakint [] -5
195 Ops.CAST dtypes.i32 [194] dtypes.i32
196 Ops.ADD dtypes.i32 [193, 195]
197 Ops.INDEX dtypes.f32 [187, 196]
198 Ops.CMPLT dtypes.bool [6, 182]
199 Ops.ADD dtypes.i32 [180, 165]
200 Ops.CMPLT dtypes.bool [6, 199]
201 Ops.CMPLT dtypes.bool [199, 22]
202 Ops.AND dtypes.bool [200, 201]
203 Ops.AND dtypes.bool [198, 202]
204 Ops.LOAD dtypes.f32 [197, 160, 203]
205 Ops.CONST dtypes.weakint [] 43
206 Ops.CAST dtypes.i32 [205] dtypes.i32
207 Ops.ADD dtypes.i32 [193, 206]
208 Ops.INDEX dtypes.f32 [187, 207]
209 Ops.LOAD dtypes.f32 [208, 160, 203]
210 Ops.STACK dtypes.f32 [204, 209]
211 Ops.CONST dtypes.weakint [] 108
212 Ops.INDEX dtypes.u64 [152, 11]
213 Ops.LOAD dtypes.u64 [212]
214 Ops.REFPARAM dtypes.f32 [211, 213] ParamArg(slot=2, dtype=dtypes.f32)
215 Ops.MUL dtypes.i32 [180, 41]
216 Ops.ADD dtypes.i32 [215, 182]
217 Ops.MUL dtypes.i32 [178, 51]
218 Ops.ADD dtypes.i32 [216, 217]
219 Ops.MUL dtypes.i32 [163, 91]
220 Ops.ADD dtypes.i32 [218, 219]
221 Ops.INDEX dtypes.f32 [214, 220]
222 Ops.LOAD dtypes.f32 [221]
223 Ops.STACK dtypes.f32 [222, 222]
224 Ops.LOAD dtypes.f32 [184]
225 Ops.INDEX dtypes.f32 [224, 6]
226 Ops.INDEX dtypes.f32 [224, 33]
227 Ops.STACK dtypes.f32 [225, 226]
228 Ops.MULACC dtypes.f32 [210, 223, 227]
229 Ops.INDEX dtypes.f32 [228, 6]
230 Ops.INDEX dtypes.f32 [228, 33]
231 Ops.CONST dtypes.weakint [] -4
232 Ops.CAST dtypes.i32 [231] dtypes.i32
233 Ops.ADD dtypes.i32 [193, 232]
234 Ops.INDEX dtypes.f32 [187, 233]
235 Ops.LOAD dtypes.f32 [234, 160, 202]
236 Ops.CONST dtypes.weakint [] 44
237 Ops.CAST dtypes.i32 [236] dtypes.i32
238 Ops.ADD dtypes.i32 [193, 237]
239 Ops.INDEX dtypes.f32 [187, 238]
240 Ops.LOAD dtypes.f32 [239, 160, 202]
241 Ops.STACK dtypes.f32 [235, 240]
242 Ops.INDEX dtypes.f32 [224, 11]
243 Ops.INDEX dtypes.f32 [224, 41]
244 Ops.STACK dtypes.f32 [242, 243]
245 Ops.MULACC dtypes.f32 [241, 223, 244]
246 Ops.INDEX dtypes.f32 [245, 6]
247 Ops.INDEX dtypes.f32 [245, 33]
248 Ops.STACK dtypes.f32 [229, 230, 246, 247]
249 Ops.STORE dtypes.void [184, 248]
250 Ops.SHRINK dtypes.f32 [183, 18, 18]
251 Ops.CONST dtypes.weakint [] -3
252 Ops.CAST dtypes.i32 [251] dtypes.i32
253 Ops.ADD dtypes.i32 [193, 252]
254 Ops.INDEX dtypes.f32 [187, 253]
255 Ops.LOAD dtypes.f32 [254, 160, 202]
256 Ops.ADD dtypes.i32 [193, 125]
257 Ops.INDEX dtypes.f32 [187, 256]
258 Ops.LOAD dtypes.f32 [257, 160, 202]
259 Ops.STACK dtypes.f32 [255, 258]
260 Ops.LOAD dtypes.f32 [250]
261 Ops.INDEX dtypes.f32 [260, 6]
262 Ops.INDEX dtypes.f32 [260, 33]
263 Ops.STACK dtypes.f32 [261, 262]
264 Ops.MULACC dtypes.f32 [259, 223, 263]
265 Ops.INDEX dtypes.f32 [264, 6]
266 Ops.INDEX dtypes.f32 [264, 33]
267 Ops.CONST dtypes.weakint [] -2
268 Ops.CAST dtypes.i32 [267] dtypes.i32
269 Ops.ADD dtypes.i32 [193, 268]
270 Ops.INDEX dtypes.f32 [187, 269]
271 Ops.CMPLT dtypes.bool [182, 11]
272 Ops.AND dtypes.bool [271, 202]
273 Ops.LOAD dtypes.f32 [270, 160, 272]
274 Ops.CONST dtypes.weakint [] 46
275 Ops.CAST dtypes.i32 [274] dtypes.i32
276 Ops.ADD dtypes.i32 [193, 275]
277 Ops.INDEX dtypes.f32 [187, 276]
278 Ops.LOAD dtypes.f32 [277, 160, 272]
279 Ops.STACK dtypes.f32 [273, 278]
280 Ops.INDEX dtypes.f32 [260, 11]
281 Ops.INDEX dtypes.f32 [260, 41]
282 Ops.STACK dtypes.f32 [280, 281]
283 Ops.MULACC dtypes.f32 [279, 223, 282]
284 Ops.INDEX dtypes.f32 [283, 6]
285 Ops.INDEX dtypes.f32 [283, 33]
286 Ops.STACK dtypes.f32 [265, 266, 284, 285]
287 Ops.STORE dtypes.void [250, 286]
288 Ops.AFTER dtypes.void [0, 249, 287]
289 Ops.END dtypes.void [288, 182]
290 Ops.END dtypes.void [289, 180]
291 Ops.END dtypes.void [290, 178]
292 Ops.AFTER dtypes.f32 [170, 291]
293 Ops.SHRINK dtypes.f32 [292, 18, 18]
294 Ops.LOAD dtypes.f32 [293]
295 Ops.INDEX dtypes.f32 [294, 11]
296 Ops.INDEX dtypes.u64 [152, 41]
297 Ops.LOAD dtypes.u64 [296]
298 Ops.REFPARAM dtypes.f32 [17, 297] ParamArg(slot=3, dtype=dtypes.f32)
299 Ops.INDEX dtypes.f32 [298, 163]
300 Ops.LOAD dtypes.f32 [299]
301 Ops.ADD dtypes.f32 [295, 300]
302 Ops.CMPLT dtypes.bool [160, 301]
303 Ops.WHERE dtypes.f32 [302, 301, 160]
304 Ops.CONST dtypes.weakint [] 128
305 Ops.INDEX dtypes.u64 [152, 18]
306 Ops.LOAD dtypes.u64 [305]
307 Ops.REFPARAM dtypes.f32 [304, 306] ParamArg(slot=4, dtype=dtypes.f32)
308 Ops.SHL dtypes.i32 [163, 18]
309 Ops.ADD dtypes.i32 [308, 189]
310 Ops.SHRINK dtypes.f32 [307, 309, 18]
311 Ops.LOAD dtypes.f32 [310]
312 Ops.INDEX dtypes.f32 [311, 41]
313 Ops.INDEX dtypes.f32 [294, 6]
314 Ops.ADD dtypes.f32 [313, 300]
315 Ops.CMPLT dtypes.bool [160, 314]
316 Ops.WHERE dtypes.f32 [315, 314, 160]
317 Ops.INDEX dtypes.f32 [311, 11]
318 Ops.SHRINK dtypes.f32 [292, 6, 18]
319 Ops.LOAD dtypes.f32 [318]
320 Ops.INDEX dtypes.f32 [319, 11]
321 Ops.ADD dtypes.f32 [320, 300]
322 Ops.CMPLT dtypes.bool [160, 321]
323 Ops.WHERE dtypes.f32 [322, 321, 160]
324 Ops.INDEX dtypes.f32 [311, 33]
325 Ops.INDEX dtypes.f32 [319, 6]
326 Ops.ADD dtypes.f32 [325, 300]
327 Ops.CMPLT dtypes.bool [160, 326]
328 Ops.WHERE dtypes.f32 [327, 326, 160]
329 Ops.INDEX dtypes.f32 [311, 6]
330 Ops.MUL dtypes.f32 [328, 329]
331 Ops.MULACC dtypes.f32 [323, 324, 330]
332 Ops.MULACC dtypes.f32 [316, 317, 331]
333 Ops.MULACC dtypes.f32 [303, 312, 332]
334 Ops.ADD dtypes.f32 [169, 333]
335 Ops.INDEX dtypes.f32 [168, 33]
336 Ops.INDEX dtypes.f32 [294, 41]
337 Ops.ADD dtypes.f32 [336, 300]
338 Ops.CMPLT dtypes.bool [160, 337]
339 Ops.WHERE dtypes.f32 [338, 337, 160]
340 Ops.CONST dtypes.weakint [] 64
341 Ops.CAST dtypes.i32 [340] dtypes.i32
342 Ops.ADD dtypes.i32 [309, 341]
343 Ops.SHRINK dtypes.f32 [307, 342, 18]
344 Ops.LOAD dtypes.f32 [343]
345 Ops.INDEX dtypes.f32 [344, 41]
346 Ops.INDEX dtypes.f32 [294, 33]
347 Ops.ADD dtypes.f32 [346, 300]
348 Ops.CMPLT dtypes.bool [160, 347]
349 Ops.WHERE dtypes.f32 [348, 347, 160]
350 Ops.INDEX dtypes.f32 [344, 11]
351 Ops.INDEX dtypes.f32 [319, 41]
352 Ops.ADD dtypes.f32 [351, 300]
353 Ops.CMPLT dtypes.bool [160, 352]
354 Ops.WHERE dtypes.f32 [353, 352, 160]
355 Ops.INDEX dtypes.f32 [344, 33]
356 Ops.INDEX dtypes.f32 [319, 33]
357 Ops.ADD dtypes.f32 [356, 300]
358 Ops.CMPLT dtypes.bool [160, 357]
359 Ops.WHERE dtypes.f32 [358, 357, 160]
360 Ops.INDEX dtypes.f32 [344, 6]
361 Ops.MUL dtypes.f32 [359, 360]
362 Ops.MULACC dtypes.f32 [354, 355, 361]
363 Ops.MULACC dtypes.f32 [349, 350, 362]
364 Ops.MULACC dtypes.f32 [339, 345, 363]
365 Ops.ADD dtypes.f32 [335, 364]
366 Ops.STACK dtypes.f32 [334, 365]
367 Ops.STORE dtypes.void [167, 366]
368 Ops.AFTER dtypes.void [0, 367]
369 Ops.END dtypes.void [368, 165]
370 Ops.END dtypes.void [369, 163]
371 Ops.AFTER dtypes.f32 [157, 370]
372 Ops.SHRINK dtypes.f32 [371, 6, 11]
373 Ops.LOAD dtypes.f32 [372]
374 Ops.INDEX dtypes.f32 [373, 6]
375 Ops.INDEX dtypes.f32 [373, 33]
376 Ops.ADD dtypes.f32 [374, 375]
377 Ops.CONST dtypes.weakfloat [] 0.5
378 Ops.CAST dtypes.f32 [377] dtypes.f32
379 Ops.MUL dtypes.f32 [376, 378]
380 Ops.STORE dtypes.void [156, 379]
381 Ops.SINK dtypes.void [380] KernelInfo(name="r_3_3_3_4_4_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=6, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(4, AxisType.UPCAST))), estimates=Estimates(ops=9858, lds=16084, mem=1348), optimized=True)
382 Ops.LINEAR dtypes.void [150, 151, 81, 170, 10, 157, 32, 33, 152, 185, 186, 11, 212, 213, 40, 41, 296, 297, 17, 18, 305, 306, 1, 187, 211, 214, 298, 304, 307, 0, 159, 160, 161, 173, 5, 6, 158, 267, 268, 251, 252, 231, 232, 194, 195, 90, 91, 205, 206, 236, 237, 124, 125, 274, 275, 21, 22, 340, 341, 50, 51, 162, 163, 299, 300, 164, 219, 308, 165, 177, 171, 172, 175, 189, 174, 176, 178, 179, 217, 192, 180, 181, 188, 190, 199, 215, 200, 201, 202, 182, 191, 193, 269, 270, 271, 272, 273, 253, 254, 255, 233, 234, 235, 196, 197, 198, 203, 204, 207, 208, 209, 238, 239, 240, 256, 257, 258, 276, 277, 278, 216, 218, 220, 221, 222, 183, 184, 224, 250, 260, 225, 226, 227, 242, 243, 244, 261, 262, 263, 280, 281, 282, 279, 259, 223, 283, 284, 285, 264, 265, 266, 286, 241, 210, 245, 246, 247, 228, 229, 230, 248, 249, 287, 288, 289, 290, 291, 309, 342, 343, 344, 310, 311, 292, 318, 319, 293, 294, 166, 167, 168, 360, 355, 350, 345, 329, 324, 317, 312, 325, 356, 320, 351, 313, 346, 295, 336, 169, 335, 326, 357, 321, 352, 314, 347, 301, 337, 327, 328, 330, 322, 315, 302, 323, 316, 303, 331, 332, 333, 334, 358, 359, 361, 353, 348, 338, 354, 349, 339, 362, 363, 364, 365, 366, 367, 368, 369, 370, 153, 154, 371, 372, 373, 155, 156, 374, 375, 377, 378, 376, 379, 380, 381]
383 Ops.SOURCE dtypes.void [] "typedef float float2 __attribute__((aligned(4),ext_vector_type(2)));\ntypedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_3_3_3_4_4_2_4(unsigned char* restrict args_40) {\n  float buf0[8];\n  float buf1[2];\n  unsigned long val0 = (*(((unsigned long*)(args_40))+1));\n  unsigned long val1 = (*(((unsigned long*)(args_40))+2));\n  unsigned long val2 = (*(((unsigned long*)(args_40))+3));\n  unsigned long val3 = (*(((unsigned long*)(args_40))+4));\n  float4 cast0 = (float4){0.0f,0.0f,0.0f,0.0f};\n  *((float2*)((buf1+0))) = (float2){0.0f,0.0f};\n  for (int Lidx3_0_0 = 0; Lidx3_0_0 < 4; Lidx3_0_0++) {\n    float val4 = (*(((float*)(val2))+Lidx3_0_0));\n    for (int Lidx3_0_1 = 0; Lidx3_0_1 < 4; Lidx3_0_1++) {\n      int alu1 = (Lidx3_0_1<<2);\n      *((float4*)((buf0+0))) = cast0;\n      *((float4*)((buf0+4))) = cast0;\n      for (int Lidx0 = 0; Lidx0 < 3; Lidx0++) {\n        for (int Lidx1 = 0; Lidx1 < 3; Lidx1++) {\n          int alu4 = (Lidx1+Lidx3_0_1);\n          _Bool alu5 = ((0<alu4)&(alu4<5));\n          for (int Lidx2 = 0; Lidx2 < 3; Lidx2++) {\n            int alu6 = ((Lidx1<<2)+alu1+Lidx2+(Lidx0<<4));\n            _Bool alu7 = ((Lidx2<2)&alu5);\n            float val5 = (alu7?*(((float*)(val0))+(alu6+-2)):0.0f);\n            float val6 = (alu5?*(((float*)(val0))+(alu6+-3)):0.0f);\n            float val7 = (alu5?*(((float*)(val0))+(alu6+-4)):0.0f);\n            _Bool alu8 = ((0<Lidx2)&alu5);\n            float val8 = (alu8?*(((float*)(val0))+(alu6+-5)):0.0f);\n            float val9 = (alu8?*(((float*)(val0))+(alu6+43)):0.0f);\n            float val10 = (alu5?*(((float*)(val0))+(alu6+44)):0.0f);\n            float val11 = (alu5?*(((float*)(val0))+(alu6+45)):0.0f);\n            float val12 = (alu7?*(((float*)(val0))+(alu6+46)):0.0f);\n            float val13 = (*(((float*)(val1))+((Lidx1*3)+Lidx2+(Lidx0*9)+(Lidx3_0_0*27))));\n            float4 val14 = (*((float4*)((buf0+0))));\n            float4 val15 = (*((float4*)((buf0+4))));\n            float2 cast1 = (float2){val13,val13};\n            float2 alu9 = __builtin_elementwise_fma((float2){val5,val12}, cast1, ((float2)__builtin_shufflevector(val15, val15, 2, 3)));\n            float2 alu10 = __builtin_elementwise_fma((float2){val6,val11}, cast1, ((float2)__builtin_shufflevector(val15, val15, 0, 1)));\n            float2 alu11 = __builtin_elementwise_fma((float2){val7,val10}, cast1, ((float2)__builtin_shufflevector(val14, val14, 2, 3)));\n            float2 alu12 = __builtin_elementwise_fma((float2){val8,val9}, cast1, ((float2)__builtin_shufflevector(val14, val14, 0, 1)));\n            *((float4*)((buf0+0))) = ((float4)__builtin_shufflevector(alu12, alu11, 0, 1, 2, 3));\n            *((float4*)((buf0+4))) = ((float4)__builtin_shufflevector(alu10, alu9, 0, 1, 2, 3));\n          }\n        }\n      }\n      int alu18 = ((Lidx3_0_0<<4)+alu1);\n      float4 val16 = (*((float4*)((((float*)(val3))+(alu18+64)))));\n      float4 val17 = (*((float4*)((((float*)(val3))+alu18))));\n      float4 val18 = (*((float4*)((buf0+0))));\n      float4 val19 = (*((float4*)((buf0+4))));\n      float2 val20 = (*((float2*)((buf1+0))));\n      float alu19 = (val18[0]+val4);\n      float alu20 = (val18[1]+val4);\n      float alu21 = (val18[2]+val4);\n      float alu22 = (val18[3]+val4);\n      float alu23 = (val19[0]+val4);\n      float alu24 = (val19[1]+val4);\n      float alu25 = (val19[2]+val4);\n      float alu26 = (val19[3]+val4);\n      float alu27 = ((0.0f<alu19)?alu19:0.0f);\n      float alu28 = ((0.0f<alu21)?alu21:0.0f);\n      float alu29 = ((0.0f<alu23)?alu23:0.0f);\n      float alu30 = ((0.0f<alu25)?alu25:0.0f);\n      float alu31 = ((0.0f<alu20)?alu20:0.0f);\n      float alu32 = ((0.0f<alu22)?alu22:0.0f);\n      float alu33 = ((0.0f<alu24)?alu24:0.0f);\n      float alu34 = ((0.0f<alu26)?alu26:0.0f);\n      *((float2*)((buf1+0))) = (float2){(val20[0]+__builtin_fmaf(alu30,val17[3],__builtin_fmaf(alu29,val17[2],__builtin_fmaf(alu28,val17[1],(alu27*val17[0]))))),(val20[1]+__builtin_fmaf(alu34,val16[3],__builtin_fmaf(alu33,val16[2],__builtin_fmaf(alu32,val16[1],(alu31*val16[0])))))};\n    }\n  }\n  unsigned long val21 = (*(((unsigned long*)(args_40))+0));\n  float2 val22 = (*((float2*)((buf1+0))));\n  *(((float*)(val21))+0) = ((val22[0]+val22[1])*0.5f);\n}\n"
384 Ops.BINARY dtypes.u8 [] b"typedef float float2 __attribute__((aligned(4),ext_vector_type(2)));\ntypedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_3_3_3_4_4_2_4(unsigned char* restrict args_40) {\n  float buf0[8];\n  float buf1[2];\n  unsigned long val0 = (*(((unsigned long*)(args_40))+1));\n  unsigned long val1 = (*(((unsigned long*)(args_40))+2));\n  unsigned long val2 = (*(((unsigned long*)(args_40))+3));\n  unsigned long val3 = (*(((unsigned long*)(args_40))+4));\n  float4 cast0 = (float4){0.0f,0.0f,0.0f,0.0f};\n  *((float2*)((buf1+0))) = (float2){0.0f,0.0f};\n  for (int Lidx3_0_0 = 0; Lidx3_0_0 < 4; Lidx3_0_0++) {\n    float val4 = (*(((float*)(val2))+Lidx3_0_0));\n    for (int Lidx3_0_1 = 0; Lidx3_0_1 < 4; Lidx3_0_1++) {\n      int alu1 = (Lidx3_0_1<<2);\n      *((float4*)((buf0+0))) = cast0;\n      *((float4*)((buf0+4))) = cast0;\n      for (int Lidx0 = 0; Lidx0 < 3; Lidx0++) {\n        for (int Lidx1 = 0; Lidx1 < 3; Lidx1++) {\n          int alu4 = (Lidx1+Lidx3_0_1);\n          _Bool alu5 = ((0<alu4)&(alu4<5));\n          for (int Lidx2 = 0; Lidx2 < 3; Lidx2++) {\n            int alu6 = ((Lidx1<<2)+alu1+Lidx2+(Lidx0<<4));\n            _Bool alu7 = ((Lidx2<2)&alu5);\n            float val5 = (alu7?*(((float*)(val0))+(alu6+-2)):0.0f);\n            float val6 = (alu5?*(((float*)(val0))+(alu6+-3)):0.0f);\n            float val7 = (alu5?*(((float*)(val0))+(alu6+-4)):0.0f);\n            _Bool alu8 = ((0<Lidx2)&alu5);\n            float val8 = (alu8?*(((float*)(val0))+(alu6+-5)):0.0f);\n            float val9 = (alu8?*(((float*)(val0))+(alu6+43)):0.0f);\n            float val10 = (alu5?*(((float*)(val0))+(alu6+44)):0.0f);\n            float val11 = (alu5?*(((float*)(val0))+(alu6+45)):0.0f);\n            float val12 = (alu7?*(((float*)(val0))+(alu6+46)):0.0f);\n            float val13 = (*(((float*)(val1))+((Lidx1*3)+Lidx2+(Lidx0*9)+(Lidx3_0_0*27))));\n            float4 val14 = (*((float4*)((buf0+0))));\n            float4 val15 = (*((float4*)((buf0+4))));\n            float2 cast1 = (float2){val13,val13};\n            float2 alu9 = __builtin_elementwise_fma((float2){val5,val12}, cast1, ((float2)__builtin_shufflevector(val15, val15, 2, 3)));\n            float2 alu10 = __builtin_elementwise_fma((float2){val6,val11}, cast1, ((float2)__builtin_shufflevector(val15, val15, 0, 1)));\n            float2 alu11 = __builtin_elementwise_fma((float2){val7,val10}, cast1, ((float2)__builtin_shufflevector(val14, val14, 2, 3)));\n            float2 alu12 = __builtin_elementwise_fma((float2){val8,val9}, cast1, ((float2)__builtin_shufflevector(val14, val14, 0, 1)));\n            *((float4*)((buf0+0))) = ((float4)__builtin_shufflevector(alu12, alu11, 0, 1, 2, 3));\n            *((float4*)((buf0+4))) = ((float4)__builtin_shufflevector(alu10, alu9, 0, 1, 2, 3));\n          }\n        }\n      }\n      int alu18 = ((Lidx3_0_0<<4)+alu1);\n      float4 val16 = (*((float4*)((((float*)(val3))+(alu18+64)))));\n      float4 val17 = (*((float4*)((((float*)(val3))+alu18))));\n      float4 val18 = (*((float4*)((buf0+0))));\n      float4 val19 = (*((float4*)((buf0+4))));\n      float2 val20 = (*((float2*)((buf1+0))));\n      float alu19 = (val18[0]+val4);\n      float alu20 = (val18[1]+val4);\n      float alu21 = (val18[2]+val4);\n      float alu22 = (val18[3]+val4);\n      float alu23 = (val19[0]+val4);\n      float alu24 = (val19[1]+val4);\n      float alu25 = (val19[2]+val4);\n      float alu26 = (val19[3]+val4);\n      float alu27 = ((0.0f<alu19)?alu19:0.0f);\n      float alu28 = ((0.0f<alu21)?alu21:0.0f);\n      float alu29 = ((0.0f<alu23)?alu23:0.0f);\n      float alu30 = ((0.0f<alu25)?alu25:0.0f);\n      float alu31 = ((0.0f<alu20)?alu20:0.0f);\n      float alu32 = ((0.0f<alu22)?alu22:0.0f);\n      float alu33 = ((0.0f<alu24)?alu24:0.0f);\n      float alu34 = ((0.0f<alu26)?alu26:0.0f);\n      *((float2*)((buf1+0))) = (float2){(val20[0]+__builtin_fmaf(alu30,val17[3],__builtin_fmaf(alu29,val17[2],__builtin_fmaf(alu28,val17[1],(alu27*val17[0]))))),(val20[1]+__builtin_fmaf(alu34,val16[3],__builtin_fmaf(alu33,val16[2],__builtin_fmaf(alu32,val16[1],(alu31*val16[0])))))};\n    }\n  }\n  unsigned long val21 = (*(((unsigned long*)(args_40))+0));\n  float2 val22 = (*((float2*)((buf1+0))));\n  *(((float*)(val21))+0) = ((val22[0]+val22[1])*0.5f);\n}\n"
385 Ops.PARAM dtypes.f32 [32] ParamArg(slot=0, dtype=dtypes.f32)
386 Ops.PARAM dtypes.f32 [211] ParamArg(slot=2, dtype=dtypes.f32)
387 Ops.PARAM dtypes.f32 [17] ParamArg(slot=3, dtype=dtypes.f32)
388 Ops.PARAM dtypes.f32 [304] ParamArg(slot=4, dtype=dtypes.f32)
389 Ops.PROGRAM dtypes.void [381, 382, 383, 384] ProgramInfo(params=(%385, %139, %386, %387, %388), globals=(0, 1, 2, 3, 4), outs=(0,), ins=(1, 2, 3, 4), target=Target(device="CPU", arch="x86_64,x86-64"))
390 Ops.BUFFER dtypes.f32 [32] ParamArg(slot=21, dtype=dtypes.f32, device="CPU")
391 Ops.BUFFER dtypes.f32 [211] ParamArg(slot=6, dtype=dtypes.f32, device="CPU")
392 Ops.BUFFER dtypes.f32 [17] ParamArg(slot=8, dtype=dtypes.f32, device="CPU")
393 Ops.PARAM dtypes.f32 [304] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
394 Ops.CALL dtypes.void [389, 390, 147, 391, 392, 393] CallInfo()
395 Ops.LINEAR dtypes.void [149, 394]

