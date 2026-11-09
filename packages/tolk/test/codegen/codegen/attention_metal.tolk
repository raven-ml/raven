# tolk D89 golden=b605b01fc4783c71a1212d583a77a167
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 512
2 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.CONST dtypes.weakint [] 32
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.SPECIAL dtypes.i32 [4] "lidx0"
6 Ops.CONST dtypes.weakint [] 4
7 Ops.CAST dtypes.i32 [6] dtypes.i32
8 Ops.SHR dtypes.i32 [5, 7]
9 Ops.CONST dtypes.weakint [] 7
10 Ops.CAST dtypes.i32 [9] dtypes.i32
11 Ops.SHL dtypes.i32 [8, 10]
12 Ops.CONST dtypes.weakint [] 1
13 Ops.CAST dtypes.i32 [12] dtypes.i32
14 Ops.SHR dtypes.i32 [5, 13]
15 Ops.CONST dtypes.weakint [] 3
16 Ops.CAST dtypes.i32 [15] dtypes.i32
17 Ops.AND dtypes.i32 [14, 16]
18 Ops.CONST dtypes.weakint [] 5
19 Ops.CAST dtypes.i32 [18] dtypes.i32
20 Ops.SHL dtypes.i32 [17, 19]
21 Ops.ADD dtypes.i32 [11, 20]
22 Ops.CONST dtypes.weakint [] 2
23 Ops.CAST dtypes.i32 [22] dtypes.i32
24 Ops.SPECIAL dtypes.i32 [23] "gidx1"
25 Ops.CONST dtypes.weakint [] 8
26 Ops.CAST dtypes.i32 [25] dtypes.i32
27 Ops.SHL dtypes.i32 [24, 26]
28 Ops.ADD dtypes.i32 [21, 27]
29 Ops.SHR dtypes.i32 [5, 16]
30 Ops.AND dtypes.i32 [29, 13]
31 Ops.SHL dtypes.i32 [30, 23]
32 Ops.SPECIAL dtypes.i32 [7] "gidx0"
33 Ops.SHL dtypes.i32 [32, 16]
34 Ops.ADD dtypes.i32 [31, 33]
35 Ops.AND dtypes.i32 [5, 13]
36 Ops.SHL dtypes.i32 [35, 13]
37 Ops.ADD dtypes.i32 [34, 36]
38 Ops.ADD dtypes.i32 [28, 37]
39 Ops.SHRINK dtypes.f32 [2, 38, 23]
40 Ops.ALLOC dtypes.f32 [22] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
41 Ops.AFTER dtypes.f32 [40, 5, 24, 32]
42 Ops.CONST dtypes.weakint [] 0
43 Ops.CAST dtypes.i32 [42] dtypes.i32
44 Ops.INDEX dtypes.f32 [41, 43]
45 Ops.CONST dtypes.weakfloat [] 0.0
46 Ops.CAST dtypes.f32 [45] dtypes.f32
47 Ops.STORE dtypes.void [44, 46]
48 Ops.INDEX dtypes.f32 [41, 13]
49 Ops.STORE dtypes.void [48, 46]
50 Ops.RANGE dtypes.i32 [23] (AxisType.WEAK, 0)
51 Ops.AFTER dtypes.f32 [40, 47, 49, 50]
52 Ops.INDEX dtypes.f32 [51, 43]
53 Ops.CONST dtypes.weakint [] 256
54 Ops.PARAM dtypes.f32 [53] ParamArg(slot=1, dtype=dtypes.f32)
55 Ops.CONST dtypes.weakint [] 6
56 Ops.CAST dtypes.i32 [55] dtypes.i32
57 Ops.SHL dtypes.i32 [8, 56]
58 Ops.SHL dtypes.i32 [17, 7]
59 Ops.ADD dtypes.i32 [57, 58]
60 Ops.SHL dtypes.i32 [24, 10]
61 Ops.ADD dtypes.i32 [59, 60]
62 Ops.SHL dtypes.i32 [50, 16]
63 Ops.ADD dtypes.i32 [31, 62]
64 Ops.ADD dtypes.i32 [63, 36]
65 Ops.ADD dtypes.i32 [61, 64]
66 Ops.SHRINK dtypes.f32 [54, 65, 23]
67 Ops.LOAD dtypes.f32 [66]
68 Ops.INDEX dtypes.f32 [67, 43]
69 Ops.CONST dtypes.weakint [] 16
70 Ops.PARAM dtypes.f32 [69] ParamArg(slot=2, dtype=dtypes.f32)
71 Ops.SHL dtypes.i32 [8, 23]
72 Ops.ADD dtypes.i32 [71, 17]
73 Ops.SHL dtypes.i32 [24, 16]
74 Ops.ADD dtypes.i32 [72, 73]
75 Ops.INDEX dtypes.f32 [70, 74]
76 Ops.LOAD dtypes.f32 [75]
77 Ops.SUB dtypes.f32 [68, 76]
78 Ops.CONST dtypes.weakfloat [] 1.4426950216293335
79 Ops.CAST dtypes.f32 [78] dtypes.f32
80 Ops.MUL dtypes.f32 [77, 79]
81 Ops.EXP2 dtypes.f32 [80]
82 Ops.PARAM dtypes.f32 [69] ParamArg(slot=3, dtype=dtypes.f32)
83 Ops.INDEX dtypes.f32 [82, 74]
84 Ops.LOAD dtypes.f32 [83]
85 Ops.RECIP dtypes.f32 [84]
86 Ops.MUL dtypes.f32 [81, 85]
87 Ops.INDEX dtypes.f32 [67, 13]
88 Ops.SUB dtypes.f32 [87, 76]
89 Ops.MUL dtypes.f32 [88, 79]
90 Ops.EXP2 dtypes.f32 [89]
91 Ops.MUL dtypes.f32 [90, 85]
92 Ops.STACK dtypes.f32 [86, 91]
93 Ops.PARAM dtypes.f32 [1] ParamArg(slot=4, dtype=dtypes.f32)
94 Ops.SHL dtypes.i32 [50, 26]
95 Ops.ADD dtypes.i32 [21, 94]
96 Ops.ADD dtypes.i32 [95, 37]
97 Ops.SHRINK dtypes.f32 [93, 96, 23]
98 Ops.LOAD dtypes.f32 [97]
99 Ops.LOAD dtypes.f32 [52]
100 Ops.INDEX dtypes.f32 [51, 13]
101 Ops.LOAD dtypes.f32 [100]
102 Ops.STACK dtypes.f32 [99, 101]
103 Ops.WMMA dtypes.f32 [92, 98, 102] ((8, 8, 8), dtypes.f32, 32, None)
104 Ops.INDEX dtypes.f32 [103, 43]
105 Ops.STORE dtypes.void [52, 104]
106 Ops.INDEX dtypes.f32 [103, 13]
107 Ops.STORE dtypes.void [100, 106]
108 Ops.AFTER dtypes.void [0, 105, 107]
109 Ops.END dtypes.void [108, 50]
110 Ops.AFTER dtypes.f32 [40, 109]
111 Ops.INDEX dtypes.f32 [110, 43]
112 Ops.LOAD dtypes.f32 [111]
113 Ops.INDEX dtypes.f32 [110, 13]
114 Ops.LOAD dtypes.f32 [113]
115 Ops.STACK dtypes.f32 [112, 114]
116 Ops.STORE dtypes.void [39, 115]
117 Ops.AFTER dtypes.void [0, 116]
118 Ops.SINK dtypes.void [117] KernelInfo(name="r_2_4_32_2_2", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), optimized=True)
119 Ops.CONST dtypes.weakint [] 40
120 Ops.PARAM dtypes.u8 [119] ParamArg(slot=0, dtype=dtypes.u8, name="args")
121 Ops.BITCAST dtypes.u64 [120] dtypes.u64
122 Ops.INDEX dtypes.u64 [121, 43]
123 Ops.LOAD dtypes.u64 [122]
124 Ops.REFPARAM dtypes.f32 [1, 123] ParamArg(slot=0, dtype=dtypes.f32)
125 Ops.SHRINK dtypes.f32 [124, 38, 23]
126 Ops.BUFFER dtypes.f32 [22] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
127 Ops.AFTER dtypes.f32 [126, 5, 24, 32]
128 Ops.INDEX dtypes.f32 [127, 43]
129 Ops.STORE dtypes.void [128, 46]
130 Ops.INDEX dtypes.f32 [127, 13]
131 Ops.STORE dtypes.void [130, 46]
132 Ops.AFTER dtypes.f32 [126, 129, 131, 50]
133 Ops.INDEX dtypes.f32 [132, 43]
134 Ops.INDEX dtypes.u64 [121, 13]
135 Ops.LOAD dtypes.u64 [134]
136 Ops.REFPARAM dtypes.f32 [53, 135] ParamArg(slot=1, dtype=dtypes.f32)
137 Ops.SHRINK dtypes.f32 [136, 65, 23]
138 Ops.LOAD dtypes.f32 [137]
139 Ops.INDEX dtypes.f32 [138, 43]
140 Ops.INDEX dtypes.u64 [121, 23]
141 Ops.LOAD dtypes.u64 [140]
142 Ops.REFPARAM dtypes.f32 [69, 141] ParamArg(slot=2, dtype=dtypes.f32)
143 Ops.INDEX dtypes.f32 [142, 74]
144 Ops.LOAD dtypes.f32 [143]
145 Ops.SUB dtypes.f32 [139, 144]
146 Ops.MUL dtypes.f32 [145, 79]
147 Ops.EXP2 dtypes.f32 [146]
148 Ops.INDEX dtypes.u64 [121, 16]
149 Ops.LOAD dtypes.u64 [148]
150 Ops.REFPARAM dtypes.f32 [69, 149] ParamArg(slot=3, dtype=dtypes.f32)
151 Ops.INDEX dtypes.f32 [150, 74]
152 Ops.LOAD dtypes.f32 [151]
153 Ops.RECIP dtypes.f32 [152]
154 Ops.MUL dtypes.f32 [147, 153]
155 Ops.INDEX dtypes.f32 [138, 13]
156 Ops.SUB dtypes.f32 [155, 144]
157 Ops.MUL dtypes.f32 [156, 79]
158 Ops.EXP2 dtypes.f32 [157]
159 Ops.MUL dtypes.f32 [158, 153]
160 Ops.STACK dtypes.f32 [154, 159]
161 Ops.INDEX dtypes.u64 [121, 7]
162 Ops.LOAD dtypes.u64 [161]
163 Ops.REFPARAM dtypes.f32 [1, 162] ParamArg(slot=4, dtype=dtypes.f32)
164 Ops.SHRINK dtypes.f32 [163, 96, 23]
165 Ops.LOAD dtypes.f32 [164]
166 Ops.LOAD dtypes.f32 [133]
167 Ops.INDEX dtypes.f32 [132, 13]
168 Ops.LOAD dtypes.f32 [167]
169 Ops.STACK dtypes.f32 [166, 168]
170 Ops.WMMA dtypes.f32 [160, 165, 169] ((8, 8, 8), dtypes.f32, 32, None)
171 Ops.INDEX dtypes.f32 [170, 43]
172 Ops.STORE dtypes.void [133, 171]
173 Ops.INDEX dtypes.f32 [170, 13]
174 Ops.STORE dtypes.void [167, 173]
175 Ops.AFTER dtypes.void [0, 172, 174]
176 Ops.END dtypes.void [175, 50]
177 Ops.AFTER dtypes.f32 [126, 176]
178 Ops.INDEX dtypes.f32 [177, 43]
179 Ops.LOAD dtypes.f32 [178]
180 Ops.INDEX dtypes.f32 [177, 13]
181 Ops.LOAD dtypes.f32 [180]
182 Ops.STACK dtypes.f32 [179, 181]
183 Ops.STORE dtypes.void [125, 182]
184 Ops.AFTER dtypes.void [0, 183]
185 Ops.SINK dtypes.void [184] KernelInfo(name="r_2_4_32_2_2", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), estimates=Estimates(ops=20736, lds=10752, mem=5248), optimized=True)
186 Ops.LINEAR dtypes.void [119, 120, 22, 126, 12, 13, 121, 134, 135, 23, 140, 141, 15, 16, 148, 149, 6, 7, 161, 162, 24, 3, 4, 5, 69, 142, 8, 71, 14, 17, 72, 73, 74, 143, 144, 150, 151, 152, 32, 53, 136, 1, 163, 127, 0, 42, 43, 128, 130, 45, 46, 78, 79, 18, 19, 55, 56, 9, 10, 25, 26, 153, 57, 58, 59, 60, 61, 33, 29, 30, 31, 34, 35, 36, 37, 11, 20, 21, 129, 131, 50, 132, 133, 166, 167, 168, 62, 63, 64, 65, 137, 138, 94, 95, 96, 164, 165, 169, 139, 155, 145, 146, 147, 156, 157, 158, 154, 159, 160, 170, 171, 173, 172, 174, 175, 176, 122, 123, 177, 178, 179, 180, 181, 124, 182, 27, 28, 38, 125, 183, 184, 185]
187 Ops.SOURCE dtypes.void [] "#include <metal_stdlib>\nusing namespace metal;\nfloat2 __WMMA_8_8_8_f32_f32(float2 a, float2 b, float2 c){\n  simdgroup_float8x8 mat_a, mat_b; simdgroup_float8x8 mat_c;\n  mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];\n  mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];\n  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);\n  return float2(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);\n}\nkernel void r_2_4_32_2_2(device unsigned char* args_40, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n  float buf0[2];\n  unsigned long val0 = (*(((device unsigned long*)(args_40))+1));\n  unsigned long val1 = (*(((device unsigned long*)(args_40))+2));\n  unsigned long val2 = (*(((device unsigned long*)(args_40))+3));\n  unsigned long val3 = (*(((device unsigned long*)(args_40))+4));\n  int gidx1 = gid.y; /* 2 */\n  int lidx0 = lid.x; /* 32 */\n  int alu0 = (lidx0>>4);\n  int alu1 = ((lidx0>>1)&3);\n  int alu2 = ((alu0<<2)+alu1+(gidx1<<3));\n  float val4 = (*(((device float*)(val1))+alu2));\n  float val5 = (*(((device float*)(val2))+alu2));\n  int gidx0 = gid.x; /* 4 */\n  float alu3 = (1/val5);\n  int alu4 = (((lidx0>>3)&1)<<2);\n  int alu5 = ((lidx0&1)<<1);\n  int alu6 = (alu4+(gidx0<<3)+alu5);\n  int alu7 = ((alu0<<7)+(alu1<<5));\n  *(buf0+0) = 0.0f;\n  *(buf0+1) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 2; Lidx0++) {\n    float2 val6 = (*((device float2*)((((device float*)(val0))+((alu0<<6)+(alu1<<4)+(gidx1<<7)+(alu4+(Lidx0<<3)+alu5))))));\n    float2 val7 = (*((device float2*)((((device float*)(val3))+(alu7+(Lidx0<<8)+alu6)))));\n    float2 wmma0 = __WMMA_8_8_8_f32_f32(float2((exp2(((val6.x-val4)*1.4426950216293335f))*alu3),(exp2(((val6.y-val4)*1.4426950216293335f))*alu3)), val7, float2((*(buf0+0)),(*(buf0+1))));\n    *(buf0+0) = wmma0.x;\n    *(buf0+1) = wmma0.y;\n  }\n  unsigned long val8 = (*(((device unsigned long*)(args_40))+0));\n  *((device float2*)((((device float*)(val8))+(alu7+(gidx1<<8)+alu6)))) = float2((*(buf0+0)),(*(buf0+1)));\n}"
188 Ops.PROGRAM dtypes.void [185, 186, 187] ProgramInfo(global_size=(4, 2, 1), local_size=(32, 1, 1), params=(%2, %54, %70, %82, %93), globals=(0, 1, 2, 3, 4), outs=(0,), ins=(1, 2, 3, 4), target=Target(device="METAL", renderer="METAL", arch="Apple9"))
189 Ops.SINK dtypes.void [118, 188]

