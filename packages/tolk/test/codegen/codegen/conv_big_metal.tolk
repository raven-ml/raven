# tolk D25 D89 D486 D491 golden=912d5ee364b1222e709a64f4616691cb
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 57600
2 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.CONST dtypes.weakint [] 32
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.SPECIAL dtypes.i32 [4] "lidx0"
6 Ops.CONST dtypes.weakint [] 4
7 Ops.CAST dtypes.i32 [6] dtypes.i32
8 Ops.SHR dtypes.i32 [5, 7]
9 Ops.CONST dtypes.weakint [] 2
10 Ops.CAST dtypes.i32 [9] dtypes.i32
11 Ops.SHL dtypes.i32 [8, 10]
12 Ops.CONST dtypes.weakint [] 1
13 Ops.CAST dtypes.i32 [12] dtypes.i32
14 Ops.SHR dtypes.i32 [5, 13]
15 Ops.CONST dtypes.weakint [] 3
16 Ops.CAST dtypes.i32 [15] dtypes.i32
17 Ops.AND dtypes.i32 [14, 16]
18 Ops.ADD dtypes.i32 [11, 17]
19 Ops.CONST dtypes.weakint [] 120
20 Ops.CAST dtypes.i32 [19] dtypes.i32
21 Ops.SPECIAL dtypes.i32 [20] "gidx0"
22 Ops.CONST dtypes.weakint [] 30
23 Ops.CAST dtypes.i32 [22] dtypes.i32
24 Ops.TRUNCDIV dtypes.i32 [21, 23]
25 Ops.SHL dtypes.i32 [24, 16]
26 Ops.ADD dtypes.i32 [18, 25]
27 Ops.TRUNCMOD dtypes.i32 [21, 23]
28 Ops.MUL dtypes.i32 [27, 23]
29 Ops.ADD dtypes.i32 [26, 28]
30 Ops.SPECIAL dtypes.i32 [10] "gidx1"
31 Ops.CONST dtypes.weakint [] 14400
32 Ops.CAST dtypes.i32 [31] dtypes.i32
33 Ops.MUL dtypes.i32 [30, 32]
34 Ops.SPECIAL dtypes.i32 [10] "lidx1"
35 Ops.CONST dtypes.weakint [] 7200
36 Ops.CAST dtypes.i32 [35] dtypes.i32
37 Ops.MUL dtypes.i32 [34, 36]
38 Ops.ADD dtypes.i32 [33, 37]
39 Ops.SHR dtypes.i32 [5, 16]
40 Ops.AND dtypes.i32 [39, 13]
41 Ops.CONST dtypes.weakint [] 3600
42 Ops.CAST dtypes.i32 [41] dtypes.i32
43 Ops.MUL dtypes.i32 [40, 42]
44 Ops.ADD dtypes.i32 [38, 43]
45 Ops.AND dtypes.i32 [5, 13]
46 Ops.CONST dtypes.weakint [] 1800
47 Ops.CAST dtypes.i32 [46] dtypes.i32
48 Ops.MUL dtypes.i32 [45, 47]
49 Ops.ADD dtypes.i32 [44, 48]
50 Ops.ADD dtypes.i32 [29, 49]
51 Ops.SPECIAL dtypes.i32 [10] "gidx2"
52 Ops.CONST dtypes.weakint [] 28800
53 Ops.CAST dtypes.i32 [52] dtypes.i32
54 Ops.MUL dtypes.i32 [51, 53]
55 Ops.ADD dtypes.i32 [50, 54]
56 Ops.INDEX dtypes.f32 [2, 55]
57 Ops.ALLOC dtypes.f32 [9] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
58 Ops.AFTER dtypes.f32 [57, 5, 21, 51, 30, 34]
59 Ops.CONST dtypes.weakint [] 0
60 Ops.CAST dtypes.i32 [59] dtypes.i32
61 Ops.INDEX dtypes.f32 [58, 60]
62 Ops.CONST dtypes.weakfloat [] 0.0
63 Ops.CAST dtypes.f32 [62] dtypes.f32
64 Ops.STORE dtypes.void [61, 63]
65 Ops.INDEX dtypes.f32 [58, 13]
66 Ops.STORE dtypes.void [65, 63]
67 Ops.RANGE dtypes.i32 [10] (AxisType.WEAK, 0)
68 Ops.AFTER dtypes.i32 [16, 67]
69 Ops.RANGE dtypes.i32 [68] (AxisType.WEAK, 1)
70 Ops.AFTER dtypes.i32 [16, 69]
71 Ops.RANGE dtypes.i32 [70] (AxisType.WEAK, 2)
72 Ops.AFTER dtypes.f32 [57, 64, 66, 67, 69, 71]
73 Ops.INDEX dtypes.f32 [72, 60]
74 Ops.CONST dtypes.weakint [] 32768
75 Ops.PARAM dtypes.f32 [74] ParamArg(slot=1, dtype=dtypes.f32)
76 Ops.ADD dtypes.i32 [26, 71]
77 Ops.CONST dtypes.weakint [] 5
78 Ops.CAST dtypes.i32 [77] dtypes.i32
79 Ops.SHL dtypes.i32 [27, 78]
80 Ops.SHL dtypes.i32 [69, 78]
81 Ops.ADD dtypes.i32 [79, 80]
82 Ops.ADD dtypes.i32 [76, 81]
83 Ops.CONST dtypes.weakint [] 12
84 Ops.CAST dtypes.i32 [83] dtypes.i32
85 Ops.SHL dtypes.i32 [40, 84]
86 Ops.CONST dtypes.weakint [] 13
87 Ops.CAST dtypes.i32 [86] dtypes.i32
88 Ops.SHL dtypes.i32 [67, 87]
89 Ops.ADD dtypes.i32 [85, 88]
90 Ops.CONST dtypes.weakint [] 11
91 Ops.CAST dtypes.i32 [90] dtypes.i32
92 Ops.SHL dtypes.i32 [45, 91]
93 Ops.ADD dtypes.i32 [89, 92]
94 Ops.ADD dtypes.i32 [82, 93]
95 Ops.CONST dtypes.weakint [] 14
96 Ops.CAST dtypes.i32 [95] dtypes.i32
97 Ops.SHL dtypes.i32 [51, 96]
98 Ops.ADD dtypes.i32 [94, 97]
99 Ops.INDEX dtypes.f32 [75, 98]
100 Ops.CMPLT dtypes.bool [26, 23]
101 Ops.LOAD dtypes.f32 [99, 63, 100]
102 Ops.CONST dtypes.weakint [] 1024
103 Ops.CAST dtypes.i32 [102] dtypes.i32
104 Ops.ADD dtypes.i32 [98, 103]
105 Ops.INDEX dtypes.f32 [75, 104]
106 Ops.LOAD dtypes.f32 [105, 63, 100]
107 Ops.STACK dtypes.f32 [101, 106]
108 Ops.CONST dtypes.weakint [] 4608
109 Ops.PARAM dtypes.f32 [108] ParamArg(slot=2, dtype=dtypes.f32)
110 Ops.CONST dtypes.weakint [] 36
111 Ops.CAST dtypes.i32 [110] dtypes.i32
112 Ops.MUL dtypes.i32 [8, 111]
113 Ops.CONST dtypes.weakint [] 9
114 Ops.CAST dtypes.i32 [113] dtypes.i32
115 Ops.MUL dtypes.i32 [17, 114]
116 Ops.ADD dtypes.i32 [112, 115]
117 Ops.CONST dtypes.weakint [] 72
118 Ops.CAST dtypes.i32 [117] dtypes.i32
119 Ops.MUL dtypes.i32 [67, 118]
120 Ops.ADD dtypes.i32 [116, 119]
121 Ops.MUL dtypes.i32 [69, 16]
122 Ops.ADD dtypes.i32 [121, 71]
123 Ops.ADD dtypes.i32 [120, 122]
124 Ops.CONST dtypes.weakint [] 2304
125 Ops.CAST dtypes.i32 [124] dtypes.i32
126 Ops.MUL dtypes.i32 [30, 125]
127 Ops.CONST dtypes.weakint [] 1152
128 Ops.CAST dtypes.i32 [127] dtypes.i32
129 Ops.MUL dtypes.i32 [34, 128]
130 Ops.ADD dtypes.i32 [126, 129]
131 Ops.CONST dtypes.weakint [] 576
132 Ops.CAST dtypes.i32 [131] dtypes.i32
133 Ops.MUL dtypes.i32 [40, 132]
134 Ops.ADD dtypes.i32 [130, 133]
135 Ops.CONST dtypes.weakint [] 288
136 Ops.CAST dtypes.i32 [135] dtypes.i32
137 Ops.MUL dtypes.i32 [45, 136]
138 Ops.ADD dtypes.i32 [134, 137]
139 Ops.ADD dtypes.i32 [123, 138]
140 Ops.INDEX dtypes.f32 [109, 139]
141 Ops.LOAD dtypes.f32 [140]
142 Ops.CONST dtypes.weakint [] 144
143 Ops.CAST dtypes.i32 [142] dtypes.i32
144 Ops.ADD dtypes.i32 [139, 143]
145 Ops.INDEX dtypes.f32 [109, 144]
146 Ops.LOAD dtypes.f32 [145]
147 Ops.STACK dtypes.f32 [141, 146]
148 Ops.LOAD dtypes.f32 [73]
149 Ops.INDEX dtypes.f32 [72, 13]
150 Ops.LOAD dtypes.f32 [149]
151 Ops.STACK dtypes.f32 [148, 150]
152 Ops.WMMA dtypes.f32 [107, 147, 151] ((8, 8, 8), dtypes.f32, 32, None)
153 Ops.INDEX dtypes.f32 [152, 60]
154 Ops.STORE dtypes.void [73, 153]
155 Ops.INDEX dtypes.f32 [152, 13]
156 Ops.STORE dtypes.void [149, 155]
157 Ops.AFTER dtypes.void [0, 154, 156]
158 Ops.END dtypes.void [157, 71]
159 Ops.END dtypes.void [158, 69]
160 Ops.END dtypes.void [159, 67]
161 Ops.AFTER dtypes.f32 [57, 160]
162 Ops.INDEX dtypes.f32 [161, 60]
163 Ops.LOAD dtypes.f32 [162]
164 Ops.STORE dtypes.void [56, 163, 100]
165 Ops.CONST dtypes.weakint [] 900
166 Ops.CAST dtypes.i32 [165] dtypes.i32
167 Ops.ADD dtypes.i32 [55, 166]
168 Ops.INDEX dtypes.f32 [2, 167]
169 Ops.INDEX dtypes.f32 [161, 13]
170 Ops.LOAD dtypes.f32 [169]
171 Ops.STORE dtypes.void [168, 170, 100]
172 Ops.AFTER dtypes.void [0, 164, 171]
173 Ops.SINK dtypes.void [172] KernelInfo(name="r_2_2_30_4_2_32_2_2_3_3", applied_opts=(Opt(op=OptOps.TC, axis=2, arg=(-1, 2, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.LOCAL))), optimized=True)
174 Ops.CONST dtypes.weakint [] 24
175 Ops.PARAM dtypes.u8 [174] ParamArg(slot=0, dtype=dtypes.u8, name="args")
176 Ops.BITCAST dtypes.u64 [175] dtypes.u64
177 Ops.INDEX dtypes.u64 [176, 60]
178 Ops.LOAD dtypes.u64 [177]
179 Ops.REFPARAM dtypes.f32 [1, 178] ParamArg(slot=0, dtype=dtypes.f32)
180 Ops.INDEX dtypes.f32 [179, 55]
181 Ops.BUFFER dtypes.f32 [9] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
182 Ops.AFTER dtypes.f32 [181, 5, 21, 51, 30, 34]
183 Ops.INDEX dtypes.f32 [182, 60]
184 Ops.STORE dtypes.void [183, 63]
185 Ops.INDEX dtypes.f32 [182, 13]
186 Ops.STORE dtypes.void [185, 63]
187 Ops.AFTER dtypes.f32 [181, 184, 186, 67, 69, 71]
188 Ops.INDEX dtypes.f32 [187, 60]
189 Ops.INDEX dtypes.u64 [176, 13]
190 Ops.LOAD dtypes.u64 [189]
191 Ops.REFPARAM dtypes.f32 [74, 190] ParamArg(slot=1, dtype=dtypes.f32)
192 Ops.INDEX dtypes.f32 [191, 98]
193 Ops.LOAD dtypes.f32 [192, 63, 100]
194 Ops.INDEX dtypes.f32 [191, 104]
195 Ops.LOAD dtypes.f32 [194, 63, 100]
196 Ops.STACK dtypes.f32 [193, 195]
197 Ops.INDEX dtypes.u64 [176, 10]
198 Ops.LOAD dtypes.u64 [197]
199 Ops.REFPARAM dtypes.f32 [108, 198] ParamArg(slot=2, dtype=dtypes.f32)
200 Ops.INDEX dtypes.f32 [199, 139]
201 Ops.LOAD dtypes.f32 [200]
202 Ops.INDEX dtypes.f32 [199, 144]
203 Ops.LOAD dtypes.f32 [202]
204 Ops.STACK dtypes.f32 [201, 203]
205 Ops.LOAD dtypes.f32 [188]
206 Ops.INDEX dtypes.f32 [187, 13]
207 Ops.LOAD dtypes.f32 [206]
208 Ops.STACK dtypes.f32 [205, 207]
209 Ops.WMMA dtypes.f32 [196, 204, 208] ((8, 8, 8), dtypes.f32, 32, None)
210 Ops.INDEX dtypes.f32 [209, 60]
211 Ops.STORE dtypes.void [188, 210]
212 Ops.INDEX dtypes.f32 [209, 13]
213 Ops.STORE dtypes.void [206, 212]
214 Ops.AFTER dtypes.void [0, 211, 213]
215 Ops.END dtypes.void [214, 71]
216 Ops.END dtypes.void [215, 69]
217 Ops.END dtypes.void [216, 67]
218 Ops.AFTER dtypes.f32 [181, 217]
219 Ops.INDEX dtypes.f32 [218, 60]
220 Ops.LOAD dtypes.f32 [219]
221 Ops.STORE dtypes.void [180, 220]
222 Ops.INDEX dtypes.f32 [179, 167]
223 Ops.INDEX dtypes.f32 [218, 13]
224 Ops.LOAD dtypes.f32 [223]
225 Ops.STORE dtypes.void [222, 224]
226 Ops.AFTER dtypes.void [0, 221, 225]
227 Ops.SINK dtypes.void [226] KernelInfo(name="r_2_2_30_4_2_32_2_2_3_3", applied_opts=(Opt(op=OptOps.TC, axis=2, arg=(-1, 2, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.LOCAL))), estimates=Estimates(ops=17725440, lds=9093120, mem=379904), optimized=True)
228 Ops.IF dtypes.void [100, 222]
229 Ops.ENDIF dtypes.void [228]
230 Ops.IF dtypes.void [100, 180]
231 Ops.ENDIF dtypes.void [230]
232 Ops.LINEAR dtypes.void [174, 175, 9, 181, 12, 13, 176, 189, 190, 10, 197, 198, 19, 20, 21, 30, 51, 3, 4, 5, 34, 74, 191, 108, 199, 182, 0, 59, 60, 183, 185, 62, 63, 102, 103, 90, 91, 127, 128, 83, 84, 86, 87, 95, 96, 142, 143, 124, 125, 135, 136, 15, 16, 22, 23, 110, 111, 6, 7, 77, 78, 131, 132, 117, 118, 113, 114, 126, 129, 130, 39, 40, 133, 134, 45, 137, 138, 8, 11, 14, 17, 18, 24, 25, 26, 112, 115, 116, 97, 27, 79, 92, 85, 100, 184, 186, 67, 68, 119, 120, 88, 89, 93, 69, 70, 80, 81, 121, 71, 76, 82, 94, 98, 104, 194, 195, 192, 193, 122, 123, 139, 144, 202, 203, 200, 201, 187, 188, 205, 206, 207, 196, 204, 208, 209, 210, 212, 211, 213, 214, 215, 216, 217, 177, 178, 218, 219, 220, 223, 224, 1, 179, 31, 32, 46, 47, 52, 53, 41, 42, 35, 36, 165, 166, 33, 37, 38, 43, 44, 48, 49, 28, 29, 50, 54, 55, 167, 222, 180, 228, 225, 229, 230, 221, 231, 226, 227]
233 Ops.SOURCE dtypes.void [] "#include <metal_stdlib>\nusing namespace metal;\nfloat2 __WMMA_8_8_8_f32_f32(float2 a, float2 b, float2 c){\n  simdgroup_float8x8 mat_a, mat_b; simdgroup_float8x8 mat_c;\n  mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];\n  mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];\n  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);\n  return float2(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);\n}\nkernel void r_2_2_30_4_2_32_2_2_3_3(device unsigned char* args_24, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n  float buf0[2];\n  unsigned long val0 = (*(((device unsigned long*)(args_24))+1));\n  unsigned long val1 = (*(((device unsigned long*)(args_24))+2));\n  int gidx0 = gid.x; /* 120 */\n  int gidx1 = gid.y; /* 2 */\n  int gidx2 = gid.z; /* 2 */\n  int lidx0 = lid.x; /* 32 */\n  int lidx1 = lid.y; /* 2 */\n  int alu0 = ((lidx0>>3)&1);\n  int alu1 = (lidx0&1);\n  int alu2 = (lidx0>>4);\n  int alu3 = ((lidx0>>1)&3);\n  int alu4 = ((alu2<<2)+alu3+((gidx0/30)<<3));\n  int alu5 = (gidx0%30);\n  bool alu6 = (alu4<30);\n  *(buf0+0) = 0.0f;\n  *(buf0+1) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 2; Lidx0++) {\n    for (int Lidx1 = 0; Lidx1 < 3; Lidx1++) {\n      for (int Lidx2 = 0; Lidx2 < 3; Lidx2++) {\n        int alu9 = (alu4+Lidx2+((alu5<<5)+(Lidx1<<5))+((alu0<<12)+(Lidx0<<13)+(alu1<<11))+(gidx2<<14));\n        float val2 = (alu6?*(((device float*)(val0))+(alu9+1024)):0.0f);\n        float val3 = (alu6?*(((device float*)(val0))+alu9):0.0f);\n        int alu10 = ((alu2*36)+(alu3*9)+(Lidx0*72)+((Lidx1*3)+Lidx2)+((gidx1*2304)+(lidx1*1152)+(alu0*576)+(alu1*288)));\n        float val4 = (*(((device float*)(val1))+(alu10+144)));\n        float val5 = (*(((device float*)(val1))+alu10));\n        float2 wmma0 = __WMMA_8_8_8_f32_f32(float2(val3,val2), float2(val5,val4), float2((*(buf0+0)),(*(buf0+1))));\n        *(buf0+0) = wmma0.x;\n        *(buf0+1) = wmma0.y;\n      }\n    }\n  }\n  unsigned long val6 = (*(((device unsigned long*)(args_24))+0));\n  int alu16 = (alu4+(alu5*30)+((gidx1*14400)+(lidx1*7200)+(alu0*3600)+(alu1*1800))+(gidx2*28800));\n  if (alu6) {\n    *(((device float*)(val6))+(alu16+900)) = (*(buf0+1));\n  }\n  if (alu6) {\n    *(((device float*)(val6))+alu16) = (*(buf0+0));\n  }\n}"
234 Ops.PROGRAM dtypes.void [227, 232, 233] ProgramInfo(global_size=(120, 2, 2), local_size=(32, 2, 1), params=(%2, %75, %109), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="METAL", renderer="METAL", arch="Apple9"))
235 Ops.SINK dtypes.void [173, 234]

