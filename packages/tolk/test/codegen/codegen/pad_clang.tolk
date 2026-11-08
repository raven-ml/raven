# tolk D80 D151 golden=5aa7928ef274f05855245eafcc0508e0
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 288
2 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.PARAM dtypes.i32 [0] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 18), name="block_lo", addrspace=AddrSpace.ALU)
4 Ops.PARAM dtypes.i32 [0] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 18), name="block_hi", addrspace=AddrSpace.ALU)
5 Ops.SUB dtypes.i32 [4, 3]
6 Ops.RANGE dtypes.i32 [5] (AxisType.WEAK, 0)
7 Ops.ADD dtypes.i32 [3, 6]
8 Ops.CONST dtypes.weakint [] 4
9 Ops.CAST dtypes.i32 [8] dtypes.i32
10 Ops.SHL dtypes.i32 [7, 9]
11 Ops.SHRINK dtypes.f32 [2, 10, 9]
12 Ops.CONST dtypes.weakint [] 225
13 Ops.PARAM dtypes.f32 [12] ParamArg(slot=1, dtype=dtypes.f32)
14 Ops.CONST dtypes.weakint [] 15
15 Ops.CAST dtypes.i32 [14] dtypes.i32
16 Ops.MUL dtypes.i32 [7, 15]
17 Ops.CONST dtypes.weakint [] -15
18 Ops.CAST dtypes.i32 [17] dtypes.i32
19 Ops.ADD dtypes.i32 [16, 18]
20 Ops.INDEX dtypes.f32 [13, 19]
21 Ops.CONST dtypes.weakfloat [] 0.0
22 Ops.CAST dtypes.f32 [21] dtypes.f32
23 Ops.CONST dtypes.weakint [] 0
24 Ops.CAST dtypes.i32 [23] dtypes.i32
25 Ops.CMPLT dtypes.bool [24, 7]
26 Ops.CONST dtypes.weakint [] 16
27 Ops.CAST dtypes.i32 [26] dtypes.i32
28 Ops.CMPLT dtypes.bool [7, 27]
29 Ops.AND dtypes.bool [25, 28]
30 Ops.LOAD dtypes.f32 [20, 22, 29]
31 Ops.CONST dtypes.weakint [] -14
32 Ops.CAST dtypes.i32 [31] dtypes.i32
33 Ops.ADD dtypes.i32 [16, 32]
34 Ops.INDEX dtypes.f32 [13, 33]
35 Ops.LOAD dtypes.f32 [34, 22, 29]
36 Ops.CONST dtypes.weakint [] -13
37 Ops.CAST dtypes.i32 [36] dtypes.i32
38 Ops.ADD dtypes.i32 [16, 37]
39 Ops.INDEX dtypes.f32 [13, 38]
40 Ops.LOAD dtypes.f32 [39, 22, 29]
41 Ops.CONST dtypes.weakint [] -12
42 Ops.CAST dtypes.i32 [41] dtypes.i32
43 Ops.ADD dtypes.i32 [16, 42]
44 Ops.INDEX dtypes.f32 [13, 43]
45 Ops.LOAD dtypes.f32 [44, 22, 29]
46 Ops.CONST dtypes.weakint [] -11
47 Ops.CAST dtypes.i32 [46] dtypes.i32
48 Ops.ADD dtypes.i32 [16, 47]
49 Ops.INDEX dtypes.f32 [13, 48]
50 Ops.LOAD dtypes.f32 [49, 22, 29]
51 Ops.CONST dtypes.weakint [] -10
52 Ops.CAST dtypes.i32 [51] dtypes.i32
53 Ops.ADD dtypes.i32 [16, 52]
54 Ops.INDEX dtypes.f32 [13, 53]
55 Ops.LOAD dtypes.f32 [54, 22, 29]
56 Ops.CONST dtypes.weakint [] -9
57 Ops.CAST dtypes.i32 [56] dtypes.i32
58 Ops.ADD dtypes.i32 [16, 57]
59 Ops.INDEX dtypes.f32 [13, 58]
60 Ops.LOAD dtypes.f32 [59, 22, 29]
61 Ops.CONST dtypes.weakint [] -8
62 Ops.CAST dtypes.i32 [61] dtypes.i32
63 Ops.ADD dtypes.i32 [16, 62]
64 Ops.INDEX dtypes.f32 [13, 63]
65 Ops.LOAD dtypes.f32 [64, 22, 29]
66 Ops.CONST dtypes.weakint [] -7
67 Ops.CAST dtypes.i32 [66] dtypes.i32
68 Ops.ADD dtypes.i32 [16, 67]
69 Ops.INDEX dtypes.f32 [13, 68]
70 Ops.LOAD dtypes.f32 [69, 22, 29]
71 Ops.CONST dtypes.weakint [] -6
72 Ops.CAST dtypes.i32 [71] dtypes.i32
73 Ops.ADD dtypes.i32 [16, 72]
74 Ops.INDEX dtypes.f32 [13, 73]
75 Ops.LOAD dtypes.f32 [74, 22, 29]
76 Ops.CONST dtypes.weakint [] -5
77 Ops.CAST dtypes.i32 [76] dtypes.i32
78 Ops.ADD dtypes.i32 [16, 77]
79 Ops.INDEX dtypes.f32 [13, 78]
80 Ops.LOAD dtypes.f32 [79, 22, 29]
81 Ops.CONST dtypes.weakint [] -4
82 Ops.CAST dtypes.i32 [81] dtypes.i32
83 Ops.ADD dtypes.i32 [16, 82]
84 Ops.INDEX dtypes.f32 [13, 83]
85 Ops.LOAD dtypes.f32 [84, 22, 29]
86 Ops.CONST dtypes.weakint [] -3
87 Ops.CAST dtypes.i32 [86] dtypes.i32
88 Ops.ADD dtypes.i32 [16, 87]
89 Ops.INDEX dtypes.f32 [13, 88]
90 Ops.LOAD dtypes.f32 [89, 22, 29]
91 Ops.CONST dtypes.weakint [] -2
92 Ops.CAST dtypes.i32 [91] dtypes.i32
93 Ops.ADD dtypes.i32 [16, 92]
94 Ops.INDEX dtypes.f32 [13, 93]
95 Ops.LOAD dtypes.f32 [94, 22, 29]
96 Ops.CONST dtypes.weakint [] -1
97 Ops.CAST dtypes.i32 [96] dtypes.i32
98 Ops.ADD dtypes.i32 [16, 97]
99 Ops.INDEX dtypes.f32 [13, 98]
100 Ops.LOAD dtypes.f32 [99, 22, 29]
101 Ops.STACK dtypes.f32 [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 22]
102 Ops.CONST dtypes.weakfloat [] 2.0
103 Ops.CAST dtypes.f32 [102] dtypes.f32
104 Ops.MUL dtypes.f32 [101, 103]
105 Ops.INDEX dtypes.f32 [104, 24]
106 Ops.CONST dtypes.weakint [] 1
107 Ops.CAST dtypes.i32 [106] dtypes.i32
108 Ops.INDEX dtypes.f32 [104, 107]
109 Ops.CONST dtypes.weakint [] 2
110 Ops.CAST dtypes.i32 [109] dtypes.i32
111 Ops.INDEX dtypes.f32 [104, 110]
112 Ops.CONST dtypes.weakint [] 3
113 Ops.CAST dtypes.i32 [112] dtypes.i32
114 Ops.INDEX dtypes.f32 [104, 113]
115 Ops.STACK dtypes.f32 [105, 108, 111, 114]
116 Ops.STORE dtypes.void [11, 115]
117 Ops.ADD dtypes.i32 [10, 9]
118 Ops.SHRINK dtypes.f32 [2, 117, 9]
119 Ops.INDEX dtypes.f32 [104, 9]
120 Ops.CONST dtypes.weakint [] 5
121 Ops.CAST dtypes.i32 [120] dtypes.i32
122 Ops.INDEX dtypes.f32 [104, 121]
123 Ops.CONST dtypes.weakint [] 6
124 Ops.CAST dtypes.i32 [123] dtypes.i32
125 Ops.INDEX dtypes.f32 [104, 124]
126 Ops.CONST dtypes.weakint [] 7
127 Ops.CAST dtypes.i32 [126] dtypes.i32
128 Ops.INDEX dtypes.f32 [104, 127]
129 Ops.STACK dtypes.f32 [119, 122, 125, 128]
130 Ops.STORE dtypes.void [118, 129]
131 Ops.CONST dtypes.weakint [] 8
132 Ops.CAST dtypes.i32 [131] dtypes.i32
133 Ops.ADD dtypes.i32 [10, 132]
134 Ops.SHRINK dtypes.f32 [2, 133, 9]
135 Ops.INDEX dtypes.f32 [104, 132]
136 Ops.CONST dtypes.weakint [] 9
137 Ops.CAST dtypes.i32 [136] dtypes.i32
138 Ops.INDEX dtypes.f32 [104, 137]
139 Ops.CONST dtypes.weakint [] 10
140 Ops.CAST dtypes.i32 [139] dtypes.i32
141 Ops.INDEX dtypes.f32 [104, 140]
142 Ops.CONST dtypes.weakint [] 11
143 Ops.CAST dtypes.i32 [142] dtypes.i32
144 Ops.INDEX dtypes.f32 [104, 143]
145 Ops.STACK dtypes.f32 [135, 138, 141, 144]
146 Ops.STORE dtypes.void [134, 145]
147 Ops.CONST dtypes.weakint [] 12
148 Ops.CAST dtypes.i32 [147] dtypes.i32
149 Ops.ADD dtypes.i32 [10, 148]
150 Ops.SHRINK dtypes.f32 [2, 149, 9]
151 Ops.INDEX dtypes.f32 [104, 148]
152 Ops.CONST dtypes.weakint [] 13
153 Ops.CAST dtypes.i32 [152] dtypes.i32
154 Ops.INDEX dtypes.f32 [104, 153]
155 Ops.CONST dtypes.weakint [] 14
156 Ops.CAST dtypes.i32 [155] dtypes.i32
157 Ops.INDEX dtypes.f32 [104, 156]
158 Ops.INDEX dtypes.f32 [104, 15]
159 Ops.STACK dtypes.f32 [151, 154, 157, 158]
160 Ops.STORE dtypes.void [150, 159]
161 Ops.AFTER dtypes.void [0, 116, 130, 146, 160]
162 Ops.END dtypes.void [161, 6]
163 Ops.SINK dtypes.void [162] KernelInfo(name="E_18_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)),), optimized=True, split=(18, 2, 3))
164 Ops.CONST dtypes.weakint [] 32
165 Ops.PARAM dtypes.u8 [164] ParamArg(slot=0, dtype=dtypes.u8, name="args")
166 Ops.BITCAST dtypes.u64 [165] dtypes.u64
167 Ops.INDEX dtypes.u64 [166, 24]
168 Ops.LOAD dtypes.u64 [167]
169 Ops.REFPARAM dtypes.f32 [1, 168] ParamArg(slot=0, dtype=dtypes.f32)
170 Ops.BITCAST dtypes.i32 [165] dtypes.i32
171 Ops.INDEX dtypes.i32 [170, 9]
172 Ops.LOAD dtypes.i32 [171]
173 Ops.INDEX dtypes.i32 [170, 121]
174 Ops.LOAD dtypes.i32 [173]
175 Ops.SUB dtypes.i32 [174, 172]
176 Ops.RANGE dtypes.i32 [175] (AxisType.WEAK, 0)
177 Ops.ADD dtypes.i32 [172, 176]
178 Ops.SHL dtypes.i32 [177, 9]
179 Ops.SHRINK dtypes.f32 [169, 178, 9]
180 Ops.INDEX dtypes.u64 [166, 107]
181 Ops.LOAD dtypes.u64 [180]
182 Ops.REFPARAM dtypes.f32 [12, 181] ParamArg(slot=1, dtype=dtypes.f32)
183 Ops.MUL dtypes.i32 [177, 15]
184 Ops.ADD dtypes.i32 [183, 18]
185 Ops.INDEX dtypes.f32 [182, 184]
186 Ops.CMPLT dtypes.bool [24, 177]
187 Ops.CMPLT dtypes.bool [177, 27]
188 Ops.AND dtypes.bool [186, 187]
189 Ops.LOAD dtypes.f32 [185, 22, 188]
190 Ops.ADD dtypes.i32 [183, 32]
191 Ops.INDEX dtypes.f32 [182, 190]
192 Ops.LOAD dtypes.f32 [191, 22, 188]
193 Ops.ADD dtypes.i32 [183, 37]
194 Ops.INDEX dtypes.f32 [182, 193]
195 Ops.LOAD dtypes.f32 [194, 22, 188]
196 Ops.ADD dtypes.i32 [183, 42]
197 Ops.INDEX dtypes.f32 [182, 196]
198 Ops.LOAD dtypes.f32 [197, 22, 188]
199 Ops.ADD dtypes.i32 [183, 47]
200 Ops.INDEX dtypes.f32 [182, 199]
201 Ops.LOAD dtypes.f32 [200, 22, 188]
202 Ops.ADD dtypes.i32 [183, 52]
203 Ops.INDEX dtypes.f32 [182, 202]
204 Ops.LOAD dtypes.f32 [203, 22, 188]
205 Ops.ADD dtypes.i32 [183, 57]
206 Ops.INDEX dtypes.f32 [182, 205]
207 Ops.LOAD dtypes.f32 [206, 22, 188]
208 Ops.ADD dtypes.i32 [183, 62]
209 Ops.INDEX dtypes.f32 [182, 208]
210 Ops.LOAD dtypes.f32 [209, 22, 188]
211 Ops.ADD dtypes.i32 [183, 67]
212 Ops.INDEX dtypes.f32 [182, 211]
213 Ops.LOAD dtypes.f32 [212, 22, 188]
214 Ops.ADD dtypes.i32 [183, 72]
215 Ops.INDEX dtypes.f32 [182, 214]
216 Ops.LOAD dtypes.f32 [215, 22, 188]
217 Ops.ADD dtypes.i32 [183, 77]
218 Ops.INDEX dtypes.f32 [182, 217]
219 Ops.LOAD dtypes.f32 [218, 22, 188]
220 Ops.ADD dtypes.i32 [183, 82]
221 Ops.INDEX dtypes.f32 [182, 220]
222 Ops.LOAD dtypes.f32 [221, 22, 188]
223 Ops.ADD dtypes.i32 [183, 87]
224 Ops.INDEX dtypes.f32 [182, 223]
225 Ops.LOAD dtypes.f32 [224, 22, 188]
226 Ops.ADD dtypes.i32 [183, 92]
227 Ops.INDEX dtypes.f32 [182, 226]
228 Ops.LOAD dtypes.f32 [227, 22, 188]
229 Ops.ADD dtypes.i32 [183, 97]
230 Ops.INDEX dtypes.f32 [182, 229]
231 Ops.LOAD dtypes.f32 [230, 22, 188]
232 Ops.STACK dtypes.f32 [189, 192, 195, 198, 201, 204, 207, 210, 213, 216, 219, 222, 225, 228, 231, 22]
233 Ops.MUL dtypes.f32 [232, 103]
234 Ops.INDEX dtypes.f32 [233, 24]
235 Ops.INDEX dtypes.f32 [233, 107]
236 Ops.INDEX dtypes.f32 [233, 110]
237 Ops.INDEX dtypes.f32 [233, 113]
238 Ops.STACK dtypes.f32 [234, 235, 236, 237]
239 Ops.STORE dtypes.void [179, 238]
240 Ops.ADD dtypes.i32 [178, 9]
241 Ops.SHRINK dtypes.f32 [169, 240, 9]
242 Ops.INDEX dtypes.f32 [233, 9]
243 Ops.INDEX dtypes.f32 [233, 121]
244 Ops.INDEX dtypes.f32 [233, 124]
245 Ops.INDEX dtypes.f32 [233, 127]
246 Ops.STACK dtypes.f32 [242, 243, 244, 245]
247 Ops.STORE dtypes.void [241, 246]
248 Ops.ADD dtypes.i32 [178, 132]
249 Ops.SHRINK dtypes.f32 [169, 248, 9]
250 Ops.INDEX dtypes.f32 [233, 132]
251 Ops.INDEX dtypes.f32 [233, 137]
252 Ops.INDEX dtypes.f32 [233, 140]
253 Ops.INDEX dtypes.f32 [233, 143]
254 Ops.STACK dtypes.f32 [250, 251, 252, 253]
255 Ops.STORE dtypes.void [249, 254]
256 Ops.ADD dtypes.i32 [178, 148]
257 Ops.SHRINK dtypes.f32 [169, 256, 9]
258 Ops.INDEX dtypes.f32 [233, 148]
259 Ops.INDEX dtypes.f32 [233, 153]
260 Ops.INDEX dtypes.f32 [233, 156]
261 Ops.INDEX dtypes.f32 [233, 15]
262 Ops.STACK dtypes.f32 [258, 259, 260, 261]
263 Ops.STORE dtypes.void [257, 262]
264 Ops.AFTER dtypes.void [0, 239, 247, 255, 263]
265 Ops.END dtypes.void [264, 176]
266 Ops.SINK dtypes.void [265] KernelInfo(name="E_18_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)),), estimates=Estimates(ops=342, lds=2232, mem=2052), optimized=True, split=(18, 2, 3))
267 Ops.LINEAR dtypes.void [164, 165, 8, 9, 170, 171, 172, 120, 121, 173, 174, 23, 24, 166, 167, 168, 106, 107, 180, 181, 1, 169, 12, 182, 0, 21, 22, 102, 103, 96, 97, 51, 52, 46, 47, 41, 42, 36, 37, 31, 32, 17, 18, 91, 92, 86, 87, 81, 82, 76, 77, 71, 72, 66, 67, 61, 62, 56, 57, 139, 140, 142, 143, 147, 148, 152, 153, 155, 156, 14, 15, 26, 27, 109, 110, 112, 113, 123, 124, 126, 127, 131, 132, 136, 137, 175, 176, 177, 183, 229, 230, 186, 187, 188, 231, 202, 203, 204, 199, 200, 201, 196, 197, 198, 193, 194, 195, 190, 191, 192, 184, 185, 189, 226, 227, 228, 223, 224, 225, 220, 221, 222, 217, 218, 219, 214, 215, 216, 211, 212, 213, 208, 209, 210, 205, 206, 207, 232, 233, 234, 235, 236, 237, 238, 258, 259, 260, 261, 262, 242, 243, 244, 245, 246, 252, 253, 250, 251, 254, 178, 256, 257, 240, 241, 248, 249, 179, 263, 247, 255, 239, 264, 265, 266]
268 Ops.SOURCE dtypes.void [] "typedef float float16 __attribute__((aligned(4),ext_vector_type(16)));\ntypedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void E_18_16(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = (val0+Lidx0);\n    int alu1 = (alu0*15);\n    _Bool alu2 = ((0<alu0)&(alu0<16));\n    float val4 = (alu2?*(((float*)(val3))+(alu1+-1)):0.0f);\n    float val5 = (alu2?*(((float*)(val3))+(alu1+-10)):0.0f);\n    float val6 = (alu2?*(((float*)(val3))+(alu1+-11)):0.0f);\n    float val7 = (alu2?*(((float*)(val3))+(alu1+-12)):0.0f);\n    float val8 = (alu2?*(((float*)(val3))+(alu1+-13)):0.0f);\n    float val9 = (alu2?*(((float*)(val3))+(alu1+-14)):0.0f);\n    float val10 = (alu2?*(((float*)(val3))+(alu1+-15)):0.0f);\n    float val11 = (alu2?*(((float*)(val3))+(alu1+-2)):0.0f);\n    float val12 = (alu2?*(((float*)(val3))+(alu1+-3)):0.0f);\n    float val13 = (alu2?*(((float*)(val3))+(alu1+-4)):0.0f);\n    float val14 = (alu2?*(((float*)(val3))+(alu1+-5)):0.0f);\n    float val15 = (alu2?*(((float*)(val3))+(alu1+-6)):0.0f);\n    float val16 = (alu2?*(((float*)(val3))+(alu1+-7)):0.0f);\n    float val17 = (alu2?*(((float*)(val3))+(alu1+-8)):0.0f);\n    float val18 = (alu2?*(((float*)(val3))+(alu1+-9)):0.0f);\n    float16 alu3 = ((float16){val10,val9,val8,val7,val6,val5,val18,val17,val16,val15,val14,val13,val12,val11,val4,0.0f}*2.0f);\n    int alu4 = (alu0<<4);\n    *((float4*)((((float*)(val2))+(alu4+12)))) = (float4){alu3[12],alu3[13],alu3[14],alu3[15]};\n    *((float4*)((((float*)(val2))+(alu4+4)))) = (float4){alu3[4],alu3[5],alu3[6],alu3[7]};\n    *((float4*)((((float*)(val2))+(alu4+8)))) = (float4){alu3[8],alu3[9],alu3[10],alu3[11]};\n    *((float4*)((((float*)(val2))+alu4))) = (float4){alu3[0],alu3[1],alu3[2],alu3[3]};\n  }\n}\n"
269 Ops.PROGRAM dtypes.void [266, 267, 268] ProgramInfo(global_size=(18, 1, 1), params=(%2, %13, %3, %4), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
270 Ops.SINK dtypes.void [163, 269]

