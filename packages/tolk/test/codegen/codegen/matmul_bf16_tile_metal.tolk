# tolk D89 D430 golden=4059aa2070c3d605db19373e55b3f2f9
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
104 Ops.CMPNE dtypes.bool [103, 103]
105 Ops.BITCAST dtypes.u32 [103] dtypes.u32
106 Ops.CONST dtypes.weakint [] 4194304
107 Ops.CAST dtypes.u32 [106] dtypes.u32
108 Ops.OR dtypes.u32 [105, 107]
109 Ops.SHR dtypes.u32 [105, 64]
110 Ops.CAST dtypes.u32 [12] dtypes.u32
111 Ops.AND dtypes.u32 [109, 110]
112 Ops.ADD dtypes.u32 [105, 111]
113 Ops.CONST dtypes.weakint [] 32767
114 Ops.CAST dtypes.u32 [113] dtypes.u32
115 Ops.ADD dtypes.u32 [112, 114]
116 Ops.WHERE dtypes.u32 [104, 108, 115]
117 Ops.SHR dtypes.u32 [116, 64]
118 Ops.CAST dtypes.u16 [117] dtypes.u16
119 Ops.BITCAST dtypes.bf16 [118] dtypes.bf16
120 Ops.INDEX dtypes.f32 [101, 13]
121 Ops.LOAD dtypes.f32 [120]
122 Ops.CMPNE dtypes.bool [121, 121]
123 Ops.BITCAST dtypes.u32 [121] dtypes.u32
124 Ops.OR dtypes.u32 [123, 107]
125 Ops.SHR dtypes.u32 [123, 64]
126 Ops.AND dtypes.u32 [125, 110]
127 Ops.ADD dtypes.u32 [123, 126]
128 Ops.ADD dtypes.u32 [127, 114]
129 Ops.WHERE dtypes.u32 [122, 124, 128]
130 Ops.SHR dtypes.u32 [129, 64]
131 Ops.CAST dtypes.u16 [130] dtypes.u16
132 Ops.BITCAST dtypes.bf16 [131] dtypes.bf16
133 Ops.STACK dtypes.bf16 [119, 132]
134 Ops.STORE dtypes.void [39, 133]
135 Ops.AFTER dtypes.void [0, 134]
136 Ops.SINK dtypes.void [135] KernelInfo(name="r_8_8_32_2_8", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), optimized=True)
137 Ops.CONST dtypes.weakint [] 24
138 Ops.PARAM dtypes.u8 [137] ParamArg(slot=0, dtype=dtypes.u8, name="args")
139 Ops.BITCAST dtypes.u64 [138] dtypes.u64
140 Ops.INDEX dtypes.u64 [139, 43]
141 Ops.LOAD dtypes.u64 [140]
142 Ops.REFPARAM dtypes.bf16 [1, 141] ParamArg(slot=0, dtype=dtypes.bf16)
143 Ops.SHRINK dtypes.bf16 [142, 38, 30]
144 Ops.BUFFER dtypes.f32 [29] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
145 Ops.AFTER dtypes.f32 [144, 5, 22, 32]
146 Ops.INDEX dtypes.f32 [145, 43]
147 Ops.STORE dtypes.void [146, 46]
148 Ops.INDEX dtypes.f32 [145, 13]
149 Ops.STORE dtypes.void [148, 46]
150 Ops.AFTER dtypes.f32 [144, 147, 149, 50]
151 Ops.INDEX dtypes.f32 [150, 43]
152 Ops.INDEX dtypes.u64 [139, 13]
153 Ops.LOAD dtypes.u64 [152]
154 Ops.REFPARAM dtypes.bf16 [1, 153] ParamArg(slot=1, dtype=dtypes.bf16)
155 Ops.SHRINK dtypes.bf16 [154, 57, 30]
156 Ops.LOAD dtypes.bf16 [155]
157 Ops.INDEX dtypes.bf16 [156, 43]
158 Ops.BITCAST dtypes.u16 [157] dtypes.u16
159 Ops.CAST dtypes.u32 [158] dtypes.u32
160 Ops.SHL dtypes.u32 [159, 64]
161 Ops.BITCAST dtypes.f32 [160] dtypes.f32
162 Ops.INDEX dtypes.bf16 [156, 13]
163 Ops.BITCAST dtypes.u16 [162] dtypes.u16
164 Ops.CAST dtypes.u32 [163] dtypes.u32
165 Ops.SHL dtypes.u32 [164, 64]
166 Ops.BITCAST dtypes.f32 [165] dtypes.f32
167 Ops.STACK dtypes.f32 [161, 166]
168 Ops.INDEX dtypes.u64 [139, 30]
169 Ops.LOAD dtypes.u64 [168]
170 Ops.REFPARAM dtypes.bf16 [1, 169] ParamArg(slot=2, dtype=dtypes.bf16)
171 Ops.SHRINK dtypes.bf16 [170, 76, 30]
172 Ops.LOAD dtypes.bf16 [171]
173 Ops.INDEX dtypes.bf16 [172, 43]
174 Ops.BITCAST dtypes.u16 [173] dtypes.u16
175 Ops.CAST dtypes.u32 [174] dtypes.u32
176 Ops.SHL dtypes.u32 [175, 64]
177 Ops.BITCAST dtypes.f32 [176] dtypes.f32
178 Ops.INDEX dtypes.bf16 [172, 13]
179 Ops.BITCAST dtypes.u16 [178] dtypes.u16
180 Ops.CAST dtypes.u32 [179] dtypes.u32
181 Ops.SHL dtypes.u32 [180, 64]
182 Ops.BITCAST dtypes.f32 [181] dtypes.f32
183 Ops.STACK dtypes.f32 [177, 182]
184 Ops.LOAD dtypes.f32 [151]
185 Ops.INDEX dtypes.f32 [150, 13]
186 Ops.LOAD dtypes.f32 [185]
187 Ops.STACK dtypes.f32 [184, 186]
188 Ops.WMMA dtypes.f32 [167, 183, 187] ((8, 8, 8), dtypes.f32, 32, None)
189 Ops.INDEX dtypes.f32 [188, 43]
190 Ops.STORE dtypes.void [151, 189]
191 Ops.INDEX dtypes.f32 [188, 13]
192 Ops.STORE dtypes.void [185, 191]
193 Ops.AFTER dtypes.void [0, 190, 192]
194 Ops.END dtypes.void [193, 50]
195 Ops.AFTER dtypes.f32 [144, 194]
196 Ops.INDEX dtypes.f32 [195, 43]
197 Ops.LOAD dtypes.f32 [196]
198 Ops.CMPNE dtypes.bool [197, 197]
199 Ops.BITCAST dtypes.u32 [197] dtypes.u32
200 Ops.OR dtypes.u32 [199, 107]
201 Ops.SHR dtypes.u32 [199, 64]
202 Ops.AND dtypes.u32 [201, 110]
203 Ops.ADD dtypes.u32 [199, 202]
204 Ops.ADD dtypes.u32 [203, 114]
205 Ops.WHERE dtypes.u32 [198, 200, 204]
206 Ops.SHR dtypes.u32 [205, 64]
207 Ops.CAST dtypes.u16 [206] dtypes.u16
208 Ops.BITCAST dtypes.bf16 [207] dtypes.bf16
209 Ops.INDEX dtypes.f32 [195, 13]
210 Ops.LOAD dtypes.f32 [209]
211 Ops.CMPNE dtypes.bool [210, 210]
212 Ops.BITCAST dtypes.u32 [210] dtypes.u32
213 Ops.OR dtypes.u32 [212, 107]
214 Ops.SHR dtypes.u32 [212, 64]
215 Ops.AND dtypes.u32 [214, 110]
216 Ops.ADD dtypes.u32 [212, 215]
217 Ops.ADD dtypes.u32 [216, 114]
218 Ops.WHERE dtypes.u32 [211, 213, 217]
219 Ops.SHR dtypes.u32 [218, 64]
220 Ops.CAST dtypes.u16 [219] dtypes.u16
221 Ops.BITCAST dtypes.bf16 [220] dtypes.bf16
222 Ops.STACK dtypes.bf16 [208, 221]
223 Ops.STORE dtypes.void [143, 222]
224 Ops.AFTER dtypes.void [0, 223]
225 Ops.SINK dtypes.void [224] KernelInfo(name="r_8_8_32_2_8", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), estimates=Estimates(ops=622592, lds=139264, mem=24576), optimized=True)
226 Ops.LINEAR dtypes.void [137, 138, 29, 144, 12, 13, 139, 152, 153, 30, 168, 169, 9, 10, 32, 22, 3, 4, 5, 1, 154, 170, 145, 0, 42, 43, 146, 148, 45, 46, 15, 16, 6, 7, 18, 19, 23, 24, 63, 64, 8, 11, 14, 17, 20, 21, 25, 26, 33, 27, 28, 31, 34, 35, 36, 37, 147, 149, 50, 54, 55, 56, 57, 155, 156, 74, 75, 76, 171, 172, 150, 151, 184, 185, 186, 187, 157, 162, 158, 159, 163, 164, 160, 161, 165, 166, 167, 173, 178, 174, 175, 179, 180, 176, 177, 181, 182, 183, 188, 189, 191, 190, 192, 193, 194, 140, 141, 195, 196, 197, 209, 210, 142, 110, 113, 114, 106, 107, 199, 201, 202, 203, 204, 198, 200, 205, 206, 207, 212, 214, 215, 216, 217, 211, 213, 218, 219, 220, 208, 221, 222, 38, 143, 223, 224, 225]
227 Ops.SOURCE dtypes.void [] "#include <metal_stdlib>\nusing namespace metal;\nfloat2 __WMMA_8_8_8_f32_f32(float2 a, float2 b, float2 c){\n  simdgroup_float8x8 mat_a, mat_b; simdgroup_float8x8 mat_c;\n  mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];\n  mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];\n  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);\n  return float2(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);\n}\nkernel void r_8_8_32_2_8(device unsigned char* args_24, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n  float buf0[2];\n  unsigned long val0 = (*(((device unsigned long*)(args_24))+1));\n  unsigned long val1 = (*(((device unsigned long*)(args_24))+2));\n  int gidx0 = gid.x; /* 8 */\n  int gidx1 = gid.y; /* 8 */\n  int lidx0 = lid.x; /* 32 */\n  int alu0 = (((lidx0>>4)<<8)+(((lidx0>>1)&3)<<6));\n  int alu1 = (alu0+(gidx1<<9));\n  int alu2 = (((lidx0>>3)&1)<<2);\n  int alu3 = ((lidx0&1)<<1);\n  int alu4 = (alu2+(gidx0<<3)+alu3);\n  *(buf0+0) = 0.0f;\n  *(buf0+1) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 8; Lidx0++) {\n    bfloat2 val2 = (*((device bfloat2*)((((device bfloat*)(val0))+(alu1+(alu2+(Lidx0<<3)+alu3))))));\n    bfloat2 val3 = (*((device bfloat2*)((((device bfloat*)(val1))+(alu0+(Lidx0<<9)+alu4)))));\n    float2 wmma0 = __WMMA_8_8_8_f32_f32(float2(as_type<float>((uint)((((uint)(as_type<unsigned short>((bfloat)(val2.x))))<<16u))),as_type<float>((uint)((((uint)(as_type<unsigned short>((bfloat)(val2.y))))<<16u)))), float2(as_type<float>((uint)((((uint)(as_type<unsigned short>((bfloat)(val3.x))))<<16u))),as_type<float>((uint)((((uint)(as_type<unsigned short>((bfloat)(val3.y))))<<16u)))), float2((*(buf0+0)),(*(buf0+1))));\n    *(buf0+0) = wmma0.x;\n    *(buf0+1) = wmma0.y;\n  }\n  unsigned long val4 = (*(((device unsigned long*)(args_24))+0));\n  float val5 = (*(buf0+0));\n  float val6 = (*(buf0+1));\n  uint cast0 = as_type<uint>((float)(val5));\n  uint alu10 = ((val5!=val5)?(cast0|4194304u):(cast0+((cast0>>16u)&1u)+32767u));\n  uint cast1 = as_type<uint>((float)(val6));\n  uint alu11 = ((val6!=val6)?(cast1|4194304u):(cast1+((cast1>>16u)&1u)+32767u));\n  *((device bfloat2*)((((device bfloat*)(val4))+(alu1+alu4)))) = bfloat2(as_type<bfloat>((unsigned short)(((unsigned short)((alu10>>16u))))),as_type<bfloat>((unsigned short)(((unsigned short)((alu11>>16u))))));\n}"
228 Ops.PROGRAM dtypes.void [225, 226, 227] ProgramInfo(global_size=(8, 8, 1), local_size=(32, 1, 1), params=(%2, %53, %73), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="METAL", renderer="METAL", arch="Apple9"))
229 Ops.SINK dtypes.void [136, 228]

