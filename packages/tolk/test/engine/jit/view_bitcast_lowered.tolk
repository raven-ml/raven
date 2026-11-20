# tolk D141 D151 golden=7c6aaaa0ee835e3fb67b5317faa4f85e
0 Ops.CONST dtypes.weakint [] 4
1 Ops.CONST dtypes.weakint [] 16
2 Ops.PARAM dtypes.u8 [1] ParamArg(slot=0, dtype=dtypes.u8, name="args")
3 Ops.BITCAST dtypes.u64 [2] dtypes.u64
4 Ops.CONST dtypes.weakint [] 0
5 Ops.CAST dtypes.i32 [4] dtypes.i32
6 Ops.INDEX dtypes.u64 [3, 5]
7 Ops.LOAD dtypes.u64 [6]
8 Ops.REFPARAM dtypes.i32 [0, 7] ParamArg(slot=0, dtype=dtypes.i32)
9 Ops.CAST dtypes.i32 [0] dtypes.i32
10 Ops.SHRINK dtypes.i32 [8, 5, 9]
11 Ops.CONST dtypes.weakint [] 64
12 Ops.CONST dtypes.weakint [] 1
13 Ops.CAST dtypes.i32 [12] dtypes.i32
14 Ops.INDEX dtypes.u64 [3, 13]
15 Ops.LOAD dtypes.u64 [14]
16 Ops.REFPARAM dtypes.f32 [11, 15] ParamArg(slot=1, dtype=dtypes.f32)
17 Ops.SHRINK dtypes.f32 [16, 5, 9]
18 Ops.LOAD dtypes.f32 [17]
19 Ops.INDEX dtypes.f32 [18, 5]
20 Ops.CONST dtypes.weakint [] 32
21 Ops.CAST dtypes.i32 [20] dtypes.i32
22 Ops.SHRINK dtypes.f32 [16, 21, 9]
23 Ops.LOAD dtypes.f32 [22]
24 Ops.INDEX dtypes.f32 [23, 5]
25 Ops.INDEX dtypes.f32 [18, 13]
26 Ops.INDEX dtypes.f32 [23, 13]
27 Ops.STACK dtypes.f32 [19, 24, 25, 26]
28 Ops.INDEX dtypes.f32 [27, 5]
29 Ops.CAST dtypes.i32 [1] dtypes.i32
30 Ops.SHRINK dtypes.f32 [16, 29, 9]
31 Ops.LOAD dtypes.f32 [30]
32 Ops.INDEX dtypes.f32 [31, 5]
33 Ops.CONST dtypes.weakint [] 48
34 Ops.CAST dtypes.i32 [33] dtypes.i32
35 Ops.SHRINK dtypes.f32 [16, 34, 9]
36 Ops.LOAD dtypes.f32 [35]
37 Ops.INDEX dtypes.f32 [36, 5]
38 Ops.INDEX dtypes.f32 [31, 13]
39 Ops.INDEX dtypes.f32 [36, 13]
40 Ops.STACK dtypes.f32 [32, 37, 38, 39]
41 Ops.INDEX dtypes.f32 [40, 5]
42 Ops.INDEX dtypes.f32 [27, 13]
43 Ops.INDEX dtypes.f32 [40, 13]
44 Ops.STACK dtypes.f32 [28, 41, 42, 43]
45 Ops.CONST dtypes.weakint [] 2
46 Ops.CAST dtypes.i32 [45] dtypes.i32
47 Ops.INDEX dtypes.f32 [27, 46]
48 Ops.INDEX dtypes.f32 [40, 46]
49 Ops.CONST dtypes.weakint [] 3
50 Ops.CAST dtypes.i32 [49] dtypes.i32
51 Ops.INDEX dtypes.f32 [27, 50]
52 Ops.INDEX dtypes.f32 [40, 50]
53 Ops.STACK dtypes.f32 [47, 48, 51, 52]
54 Ops.ADD dtypes.f32 [44, 53]
55 Ops.INDEX dtypes.f32 [18, 46]
56 Ops.INDEX dtypes.f32 [23, 46]
57 Ops.INDEX dtypes.f32 [18, 50]
58 Ops.INDEX dtypes.f32 [23, 50]
59 Ops.STACK dtypes.f32 [55, 56, 57, 58]
60 Ops.INDEX dtypes.f32 [59, 5]
61 Ops.INDEX dtypes.f32 [31, 46]
62 Ops.INDEX dtypes.f32 [36, 46]
63 Ops.INDEX dtypes.f32 [31, 50]
64 Ops.INDEX dtypes.f32 [36, 50]
65 Ops.STACK dtypes.f32 [61, 62, 63, 64]
66 Ops.INDEX dtypes.f32 [65, 5]
67 Ops.INDEX dtypes.f32 [59, 13]
68 Ops.INDEX dtypes.f32 [65, 13]
69 Ops.STACK dtypes.f32 [60, 66, 67, 68]
70 Ops.ADD dtypes.f32 [54, 69]
71 Ops.INDEX dtypes.f32 [59, 46]
72 Ops.INDEX dtypes.f32 [65, 46]
73 Ops.INDEX dtypes.f32 [59, 50]
74 Ops.INDEX dtypes.f32 [65, 50]
75 Ops.STACK dtypes.f32 [71, 72, 73, 74]
76 Ops.ADD dtypes.f32 [70, 75]
77 Ops.SHRINK dtypes.f32 [16, 9, 9]
78 Ops.LOAD dtypes.f32 [77]
79 Ops.INDEX dtypes.f32 [78, 5]
80 Ops.CONST dtypes.weakint [] 36
81 Ops.CAST dtypes.i32 [80] dtypes.i32
82 Ops.SHRINK dtypes.f32 [16, 81, 9]
83 Ops.LOAD dtypes.f32 [82]
84 Ops.INDEX dtypes.f32 [83, 5]
85 Ops.INDEX dtypes.f32 [78, 13]
86 Ops.INDEX dtypes.f32 [83, 13]
87 Ops.STACK dtypes.f32 [79, 84, 85, 86]
88 Ops.INDEX dtypes.f32 [87, 5]
89 Ops.CONST dtypes.weakint [] 20
90 Ops.CAST dtypes.i32 [89] dtypes.i32
91 Ops.SHRINK dtypes.f32 [16, 90, 9]
92 Ops.LOAD dtypes.f32 [91]
93 Ops.INDEX dtypes.f32 [92, 5]
94 Ops.CONST dtypes.weakint [] 52
95 Ops.CAST dtypes.i32 [94] dtypes.i32
96 Ops.SHRINK dtypes.f32 [16, 95, 9]
97 Ops.LOAD dtypes.f32 [96]
98 Ops.INDEX dtypes.f32 [97, 5]
99 Ops.INDEX dtypes.f32 [92, 13]
100 Ops.INDEX dtypes.f32 [97, 13]
101 Ops.STACK dtypes.f32 [93, 98, 99, 100]
102 Ops.INDEX dtypes.f32 [101, 5]
103 Ops.INDEX dtypes.f32 [87, 13]
104 Ops.INDEX dtypes.f32 [101, 13]
105 Ops.STACK dtypes.f32 [88, 102, 103, 104]
106 Ops.ADD dtypes.f32 [76, 105]
107 Ops.INDEX dtypes.f32 [87, 46]
108 Ops.INDEX dtypes.f32 [101, 46]
109 Ops.INDEX dtypes.f32 [87, 50]
110 Ops.INDEX dtypes.f32 [101, 50]
111 Ops.STACK dtypes.f32 [107, 108, 109, 110]
112 Ops.ADD dtypes.f32 [106, 111]
113 Ops.INDEX dtypes.f32 [78, 46]
114 Ops.INDEX dtypes.f32 [83, 46]
115 Ops.INDEX dtypes.f32 [78, 50]
116 Ops.INDEX dtypes.f32 [83, 50]
117 Ops.STACK dtypes.f32 [113, 114, 115, 116]
118 Ops.INDEX dtypes.f32 [117, 5]
119 Ops.INDEX dtypes.f32 [92, 46]
120 Ops.INDEX dtypes.f32 [97, 46]
121 Ops.INDEX dtypes.f32 [92, 50]
122 Ops.INDEX dtypes.f32 [97, 50]
123 Ops.STACK dtypes.f32 [119, 120, 121, 122]
124 Ops.INDEX dtypes.f32 [123, 5]
125 Ops.INDEX dtypes.f32 [117, 13]
126 Ops.INDEX dtypes.f32 [123, 13]
127 Ops.STACK dtypes.f32 [118, 124, 125, 126]
128 Ops.ADD dtypes.f32 [112, 127]
129 Ops.INDEX dtypes.f32 [117, 46]
130 Ops.INDEX dtypes.f32 [123, 46]
131 Ops.INDEX dtypes.f32 [117, 50]
132 Ops.INDEX dtypes.f32 [123, 50]
133 Ops.STACK dtypes.f32 [129, 130, 131, 132]
134 Ops.ADD dtypes.f32 [128, 133]
135 Ops.CONST dtypes.weakint [] 8
136 Ops.CAST dtypes.i32 [135] dtypes.i32
137 Ops.SHRINK dtypes.f32 [16, 136, 9]
138 Ops.LOAD dtypes.f32 [137]
139 Ops.INDEX dtypes.f32 [138, 5]
140 Ops.CONST dtypes.weakint [] 40
141 Ops.CAST dtypes.i32 [140] dtypes.i32
142 Ops.SHRINK dtypes.f32 [16, 141, 9]
143 Ops.LOAD dtypes.f32 [142]
144 Ops.INDEX dtypes.f32 [143, 5]
145 Ops.INDEX dtypes.f32 [138, 13]
146 Ops.INDEX dtypes.f32 [143, 13]
147 Ops.STACK dtypes.f32 [139, 144, 145, 146]
148 Ops.INDEX dtypes.f32 [147, 5]
149 Ops.CONST dtypes.weakint [] 24
150 Ops.CAST dtypes.i32 [149] dtypes.i32
151 Ops.SHRINK dtypes.f32 [16, 150, 9]
152 Ops.LOAD dtypes.f32 [151]
153 Ops.INDEX dtypes.f32 [152, 5]
154 Ops.CONST dtypes.weakint [] 56
155 Ops.CAST dtypes.i32 [154] dtypes.i32
156 Ops.SHRINK dtypes.f32 [16, 155, 9]
157 Ops.LOAD dtypes.f32 [156]
158 Ops.INDEX dtypes.f32 [157, 5]
159 Ops.INDEX dtypes.f32 [152, 13]
160 Ops.INDEX dtypes.f32 [157, 13]
161 Ops.STACK dtypes.f32 [153, 158, 159, 160]
162 Ops.INDEX dtypes.f32 [161, 5]
163 Ops.INDEX dtypes.f32 [147, 13]
164 Ops.INDEX dtypes.f32 [161, 13]
165 Ops.STACK dtypes.f32 [148, 162, 163, 164]
166 Ops.ADD dtypes.f32 [134, 165]
167 Ops.INDEX dtypes.f32 [147, 46]
168 Ops.INDEX dtypes.f32 [161, 46]
169 Ops.INDEX dtypes.f32 [147, 50]
170 Ops.INDEX dtypes.f32 [161, 50]
171 Ops.STACK dtypes.f32 [167, 168, 169, 170]
172 Ops.ADD dtypes.f32 [166, 171]
173 Ops.INDEX dtypes.f32 [138, 46]
174 Ops.INDEX dtypes.f32 [143, 46]
175 Ops.INDEX dtypes.f32 [138, 50]
176 Ops.INDEX dtypes.f32 [143, 50]
177 Ops.STACK dtypes.f32 [173, 174, 175, 176]
178 Ops.INDEX dtypes.f32 [177, 5]
179 Ops.INDEX dtypes.f32 [152, 46]
180 Ops.INDEX dtypes.f32 [157, 46]
181 Ops.INDEX dtypes.f32 [152, 50]
182 Ops.INDEX dtypes.f32 [157, 50]
183 Ops.STACK dtypes.f32 [179, 180, 181, 182]
184 Ops.INDEX dtypes.f32 [183, 5]
185 Ops.INDEX dtypes.f32 [177, 13]
186 Ops.INDEX dtypes.f32 [183, 13]
187 Ops.STACK dtypes.f32 [178, 184, 185, 186]
188 Ops.ADD dtypes.f32 [172, 187]
189 Ops.INDEX dtypes.f32 [177, 46]
190 Ops.INDEX dtypes.f32 [183, 46]
191 Ops.INDEX dtypes.f32 [177, 50]
192 Ops.INDEX dtypes.f32 [183, 50]
193 Ops.STACK dtypes.f32 [189, 190, 191, 192]
194 Ops.ADD dtypes.f32 [188, 193]
195 Ops.CONST dtypes.weakint [] 12
196 Ops.CAST dtypes.i32 [195] dtypes.i32
197 Ops.SHRINK dtypes.f32 [16, 196, 9]
198 Ops.LOAD dtypes.f32 [197]
199 Ops.INDEX dtypes.f32 [198, 5]
200 Ops.CONST dtypes.weakint [] 44
201 Ops.CAST dtypes.i32 [200] dtypes.i32
202 Ops.SHRINK dtypes.f32 [16, 201, 9]
203 Ops.LOAD dtypes.f32 [202]
204 Ops.INDEX dtypes.f32 [203, 5]
205 Ops.INDEX dtypes.f32 [198, 13]
206 Ops.INDEX dtypes.f32 [203, 13]
207 Ops.STACK dtypes.f32 [199, 204, 205, 206]
208 Ops.INDEX dtypes.f32 [207, 5]
209 Ops.CONST dtypes.weakint [] 28
210 Ops.CAST dtypes.i32 [209] dtypes.i32
211 Ops.SHRINK dtypes.f32 [16, 210, 9]
212 Ops.LOAD dtypes.f32 [211]
213 Ops.INDEX dtypes.f32 [212, 5]
214 Ops.CONST dtypes.weakint [] 60
215 Ops.CAST dtypes.i32 [214] dtypes.i32
216 Ops.SHRINK dtypes.f32 [16, 215, 9]
217 Ops.LOAD dtypes.f32 [216]
218 Ops.INDEX dtypes.f32 [217, 5]
219 Ops.INDEX dtypes.f32 [212, 13]
220 Ops.INDEX dtypes.f32 [217, 13]
221 Ops.STACK dtypes.f32 [213, 218, 219, 220]
222 Ops.INDEX dtypes.f32 [221, 5]
223 Ops.INDEX dtypes.f32 [207, 13]
224 Ops.INDEX dtypes.f32 [221, 13]
225 Ops.STACK dtypes.f32 [208, 222, 223, 224]
226 Ops.ADD dtypes.f32 [194, 225]
227 Ops.INDEX dtypes.f32 [207, 46]
228 Ops.INDEX dtypes.f32 [221, 46]
229 Ops.INDEX dtypes.f32 [207, 50]
230 Ops.INDEX dtypes.f32 [221, 50]
231 Ops.STACK dtypes.f32 [227, 228, 229, 230]
232 Ops.ADD dtypes.f32 [226, 231]
233 Ops.INDEX dtypes.f32 [198, 46]
234 Ops.INDEX dtypes.f32 [203, 46]
235 Ops.INDEX dtypes.f32 [198, 50]
236 Ops.INDEX dtypes.f32 [203, 50]
237 Ops.STACK dtypes.f32 [233, 234, 235, 236]
238 Ops.INDEX dtypes.f32 [237, 5]
239 Ops.INDEX dtypes.f32 [212, 46]
240 Ops.INDEX dtypes.f32 [217, 46]
241 Ops.INDEX dtypes.f32 [212, 50]
242 Ops.INDEX dtypes.f32 [217, 50]
243 Ops.STACK dtypes.f32 [239, 240, 241, 242]
244 Ops.INDEX dtypes.f32 [243, 5]
245 Ops.INDEX dtypes.f32 [237, 13]
246 Ops.INDEX dtypes.f32 [243, 13]
247 Ops.STACK dtypes.f32 [238, 244, 245, 246]
248 Ops.ADD dtypes.f32 [232, 247]
249 Ops.INDEX dtypes.f32 [237, 46]
250 Ops.INDEX dtypes.f32 [243, 46]
251 Ops.INDEX dtypes.f32 [237, 50]
252 Ops.INDEX dtypes.f32 [243, 50]
253 Ops.STACK dtypes.f32 [249, 250, 251, 252]
254 Ops.ADD dtypes.f32 [248, 253]
255 Ops.CONST dtypes.weakfloat [] 5.0
256 Ops.CAST dtypes.f32 [255] dtypes.f32
257 Ops.ADD dtypes.f32 [254, 256]
258 Ops.BITCAST dtypes.i32 [257] dtypes.i32
259 Ops.STORE dtypes.void [10, 258]
260 Ops.SINK dtypes.void [259, 259, 259, 259] KernelInfo(name="r_4_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST))), estimates=Estimates(ops=64, lds=272, mem=272), optimized=True)
261 Ops.LINEAR dtypes.void [1, 2, 4, 5, 3, 6, 7, 12, 13, 14, 15, 11, 16, 0, 9, 17, 18, 195, 196, 197, 198, 29, 30, 31, 89, 90, 91, 92, 149, 150, 151, 152, 209, 210, 211, 212, 20, 21, 22, 23, 80, 81, 82, 83, 77, 78, 140, 141, 142, 143, 200, 201, 202, 203, 33, 34, 35, 36, 94, 95, 96, 97, 154, 155, 156, 157, 214, 215, 216, 217, 135, 136, 137, 138, 8, 19, 25, 24, 26, 27, 32, 38, 37, 39, 40, 28, 42, 41, 43, 44, 45, 46, 47, 49, 50, 51, 48, 52, 53, 55, 57, 56, 58, 59, 61, 63, 62, 64, 65, 60, 67, 66, 68, 69, 71, 73, 72, 74, 75, 199, 205, 204, 206, 207, 213, 219, 218, 220, 221, 208, 223, 222, 224, 225, 227, 229, 228, 230, 231, 233, 235, 234, 236, 237, 239, 241, 240, 242, 243, 238, 245, 244, 246, 247, 249, 251, 250, 252, 253, 93, 99, 98, 100, 101, 84, 86, 79, 85, 87, 102, 104, 88, 103, 105, 108, 110, 107, 109, 111, 119, 121, 120, 122, 123, 114, 116, 113, 115, 117, 124, 126, 118, 125, 127, 130, 132, 129, 131, 133, 153, 159, 158, 160, 161, 144, 146, 139, 145, 147, 162, 164, 148, 163, 165, 168, 170, 167, 169, 171, 179, 181, 180, 182, 183, 174, 176, 173, 175, 177, 184, 186, 178, 185, 187, 190, 192, 189, 191, 193, 10, 255, 256, 54, 70, 76, 106, 112, 128, 134, 166, 172, 188, 194, 226, 232, 248, 254, 257, 258, 259, 260]
262 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_4_16(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+0));\n  unsigned long val1 = (*(((unsigned long*)(args_16))+1));\n  float4 val2 = (*((float4*)((((float*)(val1))+0))));\n  float4 val3 = (*((float4*)((((float*)(val1))+12))));\n  float4 val4 = (*((float4*)((((float*)(val1))+16))));\n  float4 val5 = (*((float4*)((((float*)(val1))+20))));\n  float4 val6 = (*((float4*)((((float*)(val1))+24))));\n  float4 val7 = (*((float4*)((((float*)(val1))+28))));\n  float4 val8 = (*((float4*)((((float*)(val1))+32))));\n  float4 val9 = (*((float4*)((((float*)(val1))+36))));\n  float4 val10 = (*((float4*)((((float*)(val1))+4))));\n  float4 val11 = (*((float4*)((((float*)(val1))+40))));\n  float4 val12 = (*((float4*)((((float*)(val1))+44))));\n  float4 val13 = (*((float4*)((((float*)(val1))+48))));\n  float4 val14 = (*((float4*)((((float*)(val1))+52))));\n  float4 val15 = (*((float4*)((((float*)(val1))+56))));\n  float4 val16 = (*((float4*)((((float*)(val1))+60))));\n  float4 val17 = (*((float4*)((((float*)(val1))+8))));\n  float4 cast0 = ((float4)__builtin_shufflevector(val2, val8, 0, 4, 1, 5));\n  float4 cast1 = ((float4)__builtin_shufflevector(val4, val13, 0, 4, 1, 5));\n  float4 cast2 = ((float4)__builtin_shufflevector(val2, val8, 2, 6, 3, 7));\n  float4 cast3 = ((float4)__builtin_shufflevector(val4, val13, 2, 6, 3, 7));\n  float4 cast4 = ((float4)__builtin_shufflevector(val3, val12, 0, 4, 1, 5));\n  float4 cast5 = ((float4)__builtin_shufflevector(val7, val16, 0, 4, 1, 5));\n  float4 cast6 = ((float4)__builtin_shufflevector(val3, val12, 2, 6, 3, 7));\n  float4 cast7 = ((float4)__builtin_shufflevector(val7, val16, 2, 6, 3, 7));\n  float4 cast8 = ((float4)__builtin_shufflevector(val5, val14, 0, 4, 1, 5));\n  float4 cast9 = ((float4)__builtin_shufflevector(val10, val9, 0, 4, 1, 5));\n  float4 cast10 = ((float4)__builtin_shufflevector(val5, val14, 2, 6, 3, 7));\n  float4 cast11 = ((float4)__builtin_shufflevector(val10, val9, 2, 6, 3, 7));\n  float4 cast12 = ((float4)__builtin_shufflevector(val6, val15, 0, 4, 1, 5));\n  float4 cast13 = ((float4)__builtin_shufflevector(val17, val11, 0, 4, 1, 5));\n  float4 cast14 = ((float4)__builtin_shufflevector(val6, val15, 2, 6, 3, 7));\n  float4 cast15 = ((float4)__builtin_shufflevector(val17, val11, 2, 6, 3, 7));\n  *((int4*)((((int*)(val0))+0))) = ((int4)((float4)((((float4)__builtin_shufflevector(cast0, cast1, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast0, cast1, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast2, cast3, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast2, cast3, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast9, cast8, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast9, cast8, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast11, cast10, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast11, cast10, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast13, cast12, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast13, cast12, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast15, cast14, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast15, cast14, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast4, cast5, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast4, cast5, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast6, cast7, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast6, cast7, 2, 6, 3, 7))+5.0f))));\n}\n"
263 Ops.BINARY dtypes.u8 [] b"typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_4_16(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+0));\n  unsigned long val1 = (*(((unsigned long*)(args_16))+1));\n  float4 val2 = (*((float4*)((((float*)(val1))+0))));\n  float4 val3 = (*((float4*)((((float*)(val1))+12))));\n  float4 val4 = (*((float4*)((((float*)(val1))+16))));\n  float4 val5 = (*((float4*)((((float*)(val1))+20))));\n  float4 val6 = (*((float4*)((((float*)(val1))+24))));\n  float4 val7 = (*((float4*)((((float*)(val1))+28))));\n  float4 val8 = (*((float4*)((((float*)(val1))+32))));\n  float4 val9 = (*((float4*)((((float*)(val1))+36))));\n  float4 val10 = (*((float4*)((((float*)(val1))+4))));\n  float4 val11 = (*((float4*)((((float*)(val1))+40))));\n  float4 val12 = (*((float4*)((((float*)(val1))+44))));\n  float4 val13 = (*((float4*)((((float*)(val1))+48))));\n  float4 val14 = (*((float4*)((((float*)(val1))+52))));\n  float4 val15 = (*((float4*)((((float*)(val1))+56))));\n  float4 val16 = (*((float4*)((((float*)(val1))+60))));\n  float4 val17 = (*((float4*)((((float*)(val1))+8))));\n  float4 cast0 = ((float4)__builtin_shufflevector(val2, val8, 0, 4, 1, 5));\n  float4 cast1 = ((float4)__builtin_shufflevector(val4, val13, 0, 4, 1, 5));\n  float4 cast2 = ((float4)__builtin_shufflevector(val2, val8, 2, 6, 3, 7));\n  float4 cast3 = ((float4)__builtin_shufflevector(val4, val13, 2, 6, 3, 7));\n  float4 cast4 = ((float4)__builtin_shufflevector(val3, val12, 0, 4, 1, 5));\n  float4 cast5 = ((float4)__builtin_shufflevector(val7, val16, 0, 4, 1, 5));\n  float4 cast6 = ((float4)__builtin_shufflevector(val3, val12, 2, 6, 3, 7));\n  float4 cast7 = ((float4)__builtin_shufflevector(val7, val16, 2, 6, 3, 7));\n  float4 cast8 = ((float4)__builtin_shufflevector(val5, val14, 0, 4, 1, 5));\n  float4 cast9 = ((float4)__builtin_shufflevector(val10, val9, 0, 4, 1, 5));\n  float4 cast10 = ((float4)__builtin_shufflevector(val5, val14, 2, 6, 3, 7));\n  float4 cast11 = ((float4)__builtin_shufflevector(val10, val9, 2, 6, 3, 7));\n  float4 cast12 = ((float4)__builtin_shufflevector(val6, val15, 0, 4, 1, 5));\n  float4 cast13 = ((float4)__builtin_shufflevector(val17, val11, 0, 4, 1, 5));\n  float4 cast14 = ((float4)__builtin_shufflevector(val6, val15, 2, 6, 3, 7));\n  float4 cast15 = ((float4)__builtin_shufflevector(val17, val11, 2, 6, 3, 7));\n  *((int4*)((((int*)(val0))+0))) = ((int4)((float4)((((float4)__builtin_shufflevector(cast0, cast1, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast0, cast1, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast2, cast3, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast2, cast3, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast9, cast8, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast9, cast8, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast11, cast10, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast11, cast10, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast13, cast12, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast13, cast12, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast15, cast14, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast15, cast14, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast4, cast5, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast4, cast5, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast6, cast7, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast6, cast7, 2, 6, 3, 7))+5.0f))));\n}\n"
264 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
265 Ops.PARAM dtypes.f32 [11] ParamArg(slot=1, dtype=dtypes.f32)
266 Ops.PROGRAM dtypes.void [260, 261, 262, 263] ProgramInfo(params=(%264, %265), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
267 Ops.CONST dtypes.weakint [] 256
268 Ops.ALLOC dtypes.i8 [267] ParamArg(slot=6, dtype=dtypes.i8, device="CPU:1")
269 Ops.SHRINK dtypes.i8 [268, 4, 1]
270 Ops.BITCAST dtypes.i32 [269] dtypes.i32
271 Ops.PARAM dtypes.f32 [11] ParamArg(slot=0, dtype=dtypes.f32, device="CPU:1")
272 Ops.CALL dtypes.void [266, 270, 271] CallInfo()
273 Ops.PARAM dtypes.i32 [0] ParamArg(slot=1, dtype=dtypes.i32)
274 Ops.STORE dtypes.void [264, 273]
275 Ops.BUFFER dtypes.i32 [0] ParamArg(slot=2, dtype=dtypes.i32, device="CPU:2")
276 Ops.CALL dtypes.void [274, 275, 270] CallInfo()
277 Ops.LINEAR dtypes.void [272, 276]

