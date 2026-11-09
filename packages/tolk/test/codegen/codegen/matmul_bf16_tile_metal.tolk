# tolk D89 golden=4059aa2070c3d605db19373e55b3f2f9
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 4096
2 Ops.PARAM dtypes.bf16 [1] ParamArg(slot=0, dtype=dtypes.bf16)
3 Ops.CONST dtypes.weakint [] 32
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.SPECIAL dtypes.i32 [4] "lidx0"
6 Ops.CONST dtypes.weakint [] 4
7 Ops.CAST dtypes.i32 [6] dtypes.i32
8 Ops.SHR dtypes.i32 [5, 7]
9 Ops.CONST dtypes.weakint [] 8
10 Ops.CAST dtypes.i32 [9] dtypes.i32
11 Ops.SHL dtypes.i32 [8, 10]
12 Ops.CONST dtypes.weakint [] 1
13 Ops.CAST dtypes.i32 [12] dtypes.i32
14 Ops.SHR dtypes.i32 [5, 13]
15 Ops.CONST dtypes.weakint [] 3
16 Ops.CAST dtypes.i32 [15] dtypes.i32
17 Ops.AND dtypes.i32 [14, 16]
18 Ops.CONST dtypes.weakint [] 6
19 Ops.CAST dtypes.i32 [18] dtypes.i32
20 Ops.SHL dtypes.i32 [17, 19]
21 Ops.ADD dtypes.i32 [11, 20]
22 Ops.SPECIAL dtypes.i32 [10] "gidx1"
23 Ops.CONST dtypes.weakint [] 9
24 Ops.CAST dtypes.i32 [23] dtypes.i32
25 Ops.SHL dtypes.i32 [22, 24]
26 Ops.ADD dtypes.i32 [21, 25]
27 Ops.SHR dtypes.i32 [5, 16]
28 Ops.AND dtypes.i32 [27, 13]
29 Ops.CONST dtypes.weakint [] 2
30 Ops.CAST dtypes.i32 [29] dtypes.i32
31 Ops.SHL dtypes.i32 [28, 30]
32 Ops.SPECIAL dtypes.i32 [10] "gidx0"
33 Ops.SHL dtypes.i32 [32, 16]
34 Ops.ADD dtypes.i32 [31, 33]
35 Ops.AND dtypes.i32 [5, 13]
36 Ops.SHL dtypes.i32 [35, 13]
37 Ops.ADD dtypes.i32 [34, 36]
38 Ops.ADD dtypes.i32 [26, 37]
39 Ops.SHRINK dtypes.bf16 [2, 38, 30]
40 Ops.ALLOC dtypes.f32 [29] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
41 Ops.AFTER dtypes.f32 [40, 5, 22, 32]
42 Ops.CONST dtypes.weakint [] 0
43 Ops.CAST dtypes.i32 [42] dtypes.i32
44 Ops.INDEX dtypes.f32 [41, 43]
45 Ops.CONST dtypes.weakfloat [] 0.0
46 Ops.CAST dtypes.f32 [45] dtypes.f32
47 Ops.STORE dtypes.void [44, 46]
48 Ops.INDEX dtypes.f32 [41, 13]
49 Ops.STORE dtypes.void [48, 46]
50 Ops.RANGE dtypes.i32 [10] (AxisType.WEAK, 0)
51 Ops.AFTER dtypes.f32 [40, 47, 49, 50]
52 Ops.INDEX dtypes.f32 [51, 43]
53 Ops.PARAM dtypes.bf16 [1] ParamArg(slot=1, dtype=dtypes.bf16)
54 Ops.SHL dtypes.i32 [50, 16]
55 Ops.ADD dtypes.i32 [31, 54]
56 Ops.ADD dtypes.i32 [55, 36]
57 Ops.ADD dtypes.i32 [26, 56]
58 Ops.SHRINK dtypes.bf16 [53, 57, 30]
59 Ops.LOAD dtypes.bf16 [58]
60 Ops.INDEX dtypes.bf16 [59, 43]
61 Ops.BITCAST dtypes.u16 [60] dtypes.u16
62 Ops.CAST dtypes.u32 [61] dtypes.u32
63 Ops.CONST dtypes.weakint [] 16
64 Ops.CAST dtypes.u32 [63] dtypes.u32
65 Ops.SHL dtypes.u32 [62, 64]
66 Ops.BITCAST dtypes.f32 [65] dtypes.f32
67 Ops.INDEX dtypes.bf16 [59, 13]
68 Ops.BITCAST dtypes.u16 [67] dtypes.u16
69 Ops.CAST dtypes.u32 [68] dtypes.u32
70 Ops.SHL dtypes.u32 [69, 64]
71 Ops.BITCAST dtypes.f32 [70] dtypes.f32
72 Ops.STACK dtypes.f32 [66, 71]
73 Ops.PARAM dtypes.bf16 [1] ParamArg(slot=2, dtype=dtypes.bf16)
74 Ops.SHL dtypes.i32 [50, 24]
75 Ops.ADD dtypes.i32 [21, 74]
76 Ops.ADD dtypes.i32 [75, 37]
77 Ops.SHRINK dtypes.bf16 [73, 76, 30]
78 Ops.LOAD dtypes.bf16 [77]
79 Ops.INDEX dtypes.bf16 [78, 43]
80 Ops.BITCAST dtypes.u16 [79] dtypes.u16
81 Ops.CAST dtypes.u32 [80] dtypes.u32
82 Ops.SHL dtypes.u32 [81, 64]
83 Ops.BITCAST dtypes.f32 [82] dtypes.f32
84 Ops.INDEX dtypes.bf16 [78, 13]
85 Ops.BITCAST dtypes.u16 [84] dtypes.u16
86 Ops.CAST dtypes.u32 [85] dtypes.u32
87 Ops.SHL dtypes.u32 [86, 64]
88 Ops.BITCAST dtypes.f32 [87] dtypes.f32
89 Ops.STACK dtypes.f32 [83, 88]
90 Ops.LOAD dtypes.f32 [52]
91 Ops.INDEX dtypes.f32 [51, 13]
92 Ops.LOAD dtypes.f32 [91]
93 Ops.STACK dtypes.f32 [90, 92]
94 Ops.WMMA dtypes.f32 [72, 89, 93] ((8, 8, 8), dtypes.f32, 32, None)
95 Ops.INDEX dtypes.f32 [94, 43]
96 Ops.STORE dtypes.void [52, 95]
97 Ops.INDEX dtypes.f32 [94, 13]
98 Ops.STORE dtypes.void [91, 97]
99 Ops.AFTER dtypes.void [0, 96, 98]
100 Ops.END dtypes.void [99, 50]
101 Ops.AFTER dtypes.f32 [40, 100]
102 Ops.INDEX dtypes.f32 [101, 43]
103 Ops.LOAD dtypes.f32 [102]
104 Ops.BITCAST dtypes.u32 [103] dtypes.u32
105 Ops.NEG dtypes.u32 [104]
106 Ops.CONST dtypes.weakint [] 2139095040
107 Ops.CAST dtypes.u32 [106] dtypes.u32
108 Ops.AND dtypes.u32 [105, 107]
109 Ops.CAST dtypes.u32 [42] dtypes.u32
110 Ops.CMPNE dtypes.bool [108, 109]
111 Ops.SHR dtypes.u32 [104, 64]
112 Ops.CAST dtypes.u32 [12] dtypes.u32
113 Ops.AND dtypes.u32 [111, 112]
114 Ops.ADD dtypes.u32 [104, 113]
115 Ops.CONST dtypes.weakint [] 32767
116 Ops.CAST dtypes.u32 [115] dtypes.u32
117 Ops.ADD dtypes.u32 [114, 116]
118 Ops.CONST dtypes.weakint [] 65535
119 Ops.CAST dtypes.u32 [118] dtypes.u32
120 Ops.AND dtypes.u32 [104, 119]
121 Ops.CMPNE dtypes.bool [120, 109]
122 Ops.CONST dtypes.weakint [] 65536
123 Ops.CAST dtypes.u32 [122] dtypes.u32
124 Ops.OR dtypes.u32 [104, 123]
125 Ops.WHERE dtypes.u32 [121, 124, 104]
126 Ops.WHERE dtypes.u32 [110, 117, 125]
127 Ops.SHR dtypes.u32 [126, 64]
128 Ops.CAST dtypes.u16 [127] dtypes.u16
129 Ops.BITCAST dtypes.bf16 [128] dtypes.bf16
130 Ops.INDEX dtypes.f32 [101, 13]
131 Ops.LOAD dtypes.f32 [130]
132 Ops.BITCAST dtypes.u32 [131] dtypes.u32
133 Ops.NEG dtypes.u32 [132]
134 Ops.AND dtypes.u32 [133, 107]
135 Ops.CMPNE dtypes.bool [134, 109]
136 Ops.SHR dtypes.u32 [132, 64]
137 Ops.AND dtypes.u32 [136, 112]
138 Ops.ADD dtypes.u32 [132, 137]
139 Ops.ADD dtypes.u32 [138, 116]
140 Ops.AND dtypes.u32 [132, 119]
141 Ops.CMPNE dtypes.bool [140, 109]
142 Ops.OR dtypes.u32 [132, 123]
143 Ops.WHERE dtypes.u32 [141, 142, 132]
144 Ops.WHERE dtypes.u32 [135, 139, 143]
145 Ops.SHR dtypes.u32 [144, 64]
146 Ops.CAST dtypes.u16 [145] dtypes.u16
147 Ops.BITCAST dtypes.bf16 [146] dtypes.bf16
148 Ops.STACK dtypes.bf16 [129, 147]
149 Ops.STORE dtypes.void [39, 148]
150 Ops.AFTER dtypes.void [0, 149]
151 Ops.SINK dtypes.void [150] KernelInfo(name="r_8_8_32_2_8", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), optimized=True)
152 Ops.CONST dtypes.weakint [] 24
153 Ops.PARAM dtypes.u8 [152] ParamArg(slot=0, dtype=dtypes.u8, name="args")
154 Ops.BITCAST dtypes.u64 [153] dtypes.u64
155 Ops.INDEX dtypes.u64 [154, 43]
156 Ops.LOAD dtypes.u64 [155]
157 Ops.REFPARAM dtypes.bf16 [1, 156] ParamArg(slot=0, dtype=dtypes.bf16)
158 Ops.SHRINK dtypes.bf16 [157, 38, 30]
159 Ops.BUFFER dtypes.f32 [29] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
160 Ops.AFTER dtypes.f32 [159, 5, 22, 32]
161 Ops.INDEX dtypes.f32 [160, 43]
162 Ops.STORE dtypes.void [161, 46]
163 Ops.INDEX dtypes.f32 [160, 13]
164 Ops.STORE dtypes.void [163, 46]
165 Ops.AFTER dtypes.f32 [159, 162, 164, 50]
166 Ops.INDEX dtypes.f32 [165, 43]
167 Ops.INDEX dtypes.u64 [154, 13]
168 Ops.LOAD dtypes.u64 [167]
169 Ops.REFPARAM dtypes.bf16 [1, 168] ParamArg(slot=1, dtype=dtypes.bf16)
170 Ops.SHRINK dtypes.bf16 [169, 57, 30]
171 Ops.LOAD dtypes.bf16 [170]
172 Ops.INDEX dtypes.bf16 [171, 43]
173 Ops.BITCAST dtypes.u16 [172] dtypes.u16
174 Ops.CAST dtypes.u32 [173] dtypes.u32
175 Ops.SHL dtypes.u32 [174, 64]
176 Ops.BITCAST dtypes.f32 [175] dtypes.f32
177 Ops.INDEX dtypes.bf16 [171, 13]
178 Ops.BITCAST dtypes.u16 [177] dtypes.u16
179 Ops.CAST dtypes.u32 [178] dtypes.u32
180 Ops.SHL dtypes.u32 [179, 64]
181 Ops.BITCAST dtypes.f32 [180] dtypes.f32
182 Ops.STACK dtypes.f32 [176, 181]
183 Ops.INDEX dtypes.u64 [154, 30]
184 Ops.LOAD dtypes.u64 [183]
185 Ops.REFPARAM dtypes.bf16 [1, 184] ParamArg(slot=2, dtype=dtypes.bf16)
186 Ops.SHRINK dtypes.bf16 [185, 76, 30]
187 Ops.LOAD dtypes.bf16 [186]
188 Ops.INDEX dtypes.bf16 [187, 43]
189 Ops.BITCAST dtypes.u16 [188] dtypes.u16
190 Ops.CAST dtypes.u32 [189] dtypes.u32
191 Ops.SHL dtypes.u32 [190, 64]
192 Ops.BITCAST dtypes.f32 [191] dtypes.f32
193 Ops.INDEX dtypes.bf16 [187, 13]
194 Ops.BITCAST dtypes.u16 [193] dtypes.u16
195 Ops.CAST dtypes.u32 [194] dtypes.u32
196 Ops.SHL dtypes.u32 [195, 64]
197 Ops.BITCAST dtypes.f32 [196] dtypes.f32
198 Ops.STACK dtypes.f32 [192, 197]
199 Ops.LOAD dtypes.f32 [166]
200 Ops.INDEX dtypes.f32 [165, 13]
201 Ops.LOAD dtypes.f32 [200]
202 Ops.STACK dtypes.f32 [199, 201]
203 Ops.WMMA dtypes.f32 [182, 198, 202] ((8, 8, 8), dtypes.f32, 32, None)
204 Ops.INDEX dtypes.f32 [203, 43]
205 Ops.STORE dtypes.void [166, 204]
206 Ops.INDEX dtypes.f32 [203, 13]
207 Ops.STORE dtypes.void [200, 206]
208 Ops.AFTER dtypes.void [0, 205, 207]
209 Ops.END dtypes.void [208, 50]
210 Ops.AFTER dtypes.f32 [159, 209]
211 Ops.INDEX dtypes.f32 [210, 43]
212 Ops.LOAD dtypes.f32 [211]
213 Ops.BITCAST dtypes.u32 [212] dtypes.u32
214 Ops.NEG dtypes.u32 [213]
215 Ops.AND dtypes.u32 [214, 107]
216 Ops.CMPNE dtypes.bool [215, 109]
217 Ops.SHR dtypes.u32 [213, 64]
218 Ops.AND dtypes.u32 [217, 112]
219 Ops.ADD dtypes.u32 [213, 218]
220 Ops.ADD dtypes.u32 [219, 116]
221 Ops.AND dtypes.u32 [213, 119]
222 Ops.CMPNE dtypes.bool [221, 109]
223 Ops.OR dtypes.u32 [213, 123]
224 Ops.WHERE dtypes.u32 [222, 223, 213]
225 Ops.WHERE dtypes.u32 [216, 220, 224]
226 Ops.SHR dtypes.u32 [225, 64]
227 Ops.CAST dtypes.u16 [226] dtypes.u16
228 Ops.BITCAST dtypes.bf16 [227] dtypes.bf16
229 Ops.INDEX dtypes.f32 [210, 13]
230 Ops.LOAD dtypes.f32 [229]
231 Ops.BITCAST dtypes.u32 [230] dtypes.u32
232 Ops.NEG dtypes.u32 [231]
233 Ops.AND dtypes.u32 [232, 107]
234 Ops.CMPNE dtypes.bool [233, 109]
235 Ops.SHR dtypes.u32 [231, 64]
236 Ops.AND dtypes.u32 [235, 112]
237 Ops.ADD dtypes.u32 [231, 236]
238 Ops.ADD dtypes.u32 [237, 116]
239 Ops.AND dtypes.u32 [231, 119]
240 Ops.CMPNE dtypes.bool [239, 109]
241 Ops.OR dtypes.u32 [231, 123]
242 Ops.WHERE dtypes.u32 [240, 241, 231]
243 Ops.WHERE dtypes.u32 [234, 238, 242]
244 Ops.SHR dtypes.u32 [243, 64]
245 Ops.CAST dtypes.u16 [244] dtypes.u16
246 Ops.BITCAST dtypes.bf16 [245] dtypes.bf16
247 Ops.STACK dtypes.bf16 [228, 246]
248 Ops.STORE dtypes.void [158, 247]
249 Ops.AFTER dtypes.void [0, 248]
250 Ops.SINK dtypes.void [249] KernelInfo(name="r_8_8_32_2_8", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), estimates=Estimates(ops=643072, lds=139264, mem=24576), optimized=True)
251 Ops.LINEAR dtypes.void [152, 153, 29, 159, 12, 13, 154, 167, 168, 30, 183, 184, 9, 10, 32, 22, 3, 4, 5, 1, 169, 185, 160, 0, 42, 43, 161, 163, 45, 46, 15, 16, 6, 7, 18, 19, 23, 24, 63, 64, 8, 11, 14, 17, 20, 21, 25, 26, 33, 27, 28, 31, 34, 35, 36, 37, 162, 164, 50, 54, 55, 56, 57, 170, 171, 74, 75, 76, 186, 187, 165, 166, 199, 200, 201, 202, 172, 177, 173, 174, 178, 179, 175, 176, 180, 181, 182, 188, 193, 189, 190, 194, 195, 191, 192, 196, 197, 198, 203, 204, 206, 205, 207, 208, 209, 155, 156, 210, 211, 212, 229, 230, 157, 109, 112, 106, 107, 115, 116, 118, 119, 122, 123, 213, 214, 217, 218, 219, 220, 221, 222, 215, 216, 223, 224, 225, 226, 227, 231, 232, 235, 236, 237, 238, 239, 240, 233, 234, 241, 242, 243, 244, 245, 228, 246, 247, 38, 158, 248, 249, 250]
252 Ops.SOURCE dtypes.void [] "#include <metal_stdlib>\nusing namespace metal;\nfloat2 __WMMA_8_8_8_f32_f32(float2 a, float2 b, float2 c){\n  simdgroup_float8x8 mat_a, mat_b; simdgroup_float8x8 mat_c;\n  mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];\n  mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];\n  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);\n  return float2(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);\n}\nkernel void r_8_8_32_2_8(device unsigned char* args_24, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n  float buf0[2];\n  unsigned long val0 = (*(((device unsigned long*)(args_24))+1));\n  unsigned long val1 = (*(((device unsigned long*)(args_24))+2));\n  int gidx0 = gid.x; /* 8 */\n  int gidx1 = gid.y; /* 8 */\n  int lidx0 = lid.x; /* 32 */\n  int alu0 = (((lidx0>>4)<<8)+(((lidx0>>1)&3)<<6));\n  int alu1 = (alu0+(gidx1<<9));\n  int alu2 = (((lidx0>>3)&1)<<2);\n  int alu3 = ((lidx0&1)<<1);\n  int alu4 = (alu2+(gidx0<<3)+alu3);\n  *(buf0+0) = 0.0f;\n  *(buf0+1) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 8; Lidx0++) {\n    bfloat2 val2 = (*((device bfloat2*)((((device bfloat*)(val0))+(alu1+(alu2+(Lidx0<<3)+alu3))))));\n    bfloat2 val3 = (*((device bfloat2*)((((device bfloat*)(val1))+(alu0+(Lidx0<<9)+alu4)))));\n    float2 wmma0 = __WMMA_8_8_8_f32_f32(float2(as_type<float>((uint)((((uint)(as_type<unsigned short>((bfloat)(val2.x))))<<16u))),as_type<float>((uint)((((uint)(as_type<unsigned short>((bfloat)(val2.y))))<<16u)))), float2(as_type<float>((uint)((((uint)(as_type<unsigned short>((bfloat)(val3.x))))<<16u))),as_type<float>((uint)((((uint)(as_type<unsigned short>((bfloat)(val3.y))))<<16u)))), float2((*(buf0+0)),(*(buf0+1))));\n    *(buf0+0) = wmma0.x;\n    *(buf0+1) = wmma0.y;\n  }\n  unsigned long val4 = (*(((device unsigned long*)(args_24))+0));\n  uint cast0 = as_type<uint>((float)((*(buf0+0))));\n  uint alu10 = (((cast0&65535u)!=0u)?(cast0|65536u):cast0);\n  uint alu11 = (((-cast0&2139095040u)!=0u)?(cast0+((cast0>>16u)&1u)+32767u):alu10);\n  uint cast1 = as_type<uint>((float)((*(buf0+1))));\n  uint alu12 = (((cast1&65535u)!=0u)?(cast1|65536u):cast1);\n  uint alu13 = (((-cast1&2139095040u)!=0u)?(cast1+((cast1>>16u)&1u)+32767u):alu12);\n  *((device bfloat2*)((((device bfloat*)(val4))+(alu1+alu4)))) = bfloat2(as_type<bfloat>((unsigned short)(((unsigned short)((alu11>>16u))))),as_type<bfloat>((unsigned short)(((unsigned short)((alu13>>16u))))));\n}"
253 Ops.PROGRAM dtypes.void [250, 251, 252] ProgramInfo(global_size=(8, 8, 1), local_size=(32, 1, 1), params=(%2, %53, %73), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="METAL", renderer="METAL", arch="Apple9"))
254 Ops.SINK dtypes.void [151, 253]

