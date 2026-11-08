# tolk D80 D151 golden=30ea1f24f54b09ffac466bafcf0c9206
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 1024
2 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.PARAM dtypes.i32 [0] ParamArg(slot=4, dtype=dtypes.i32, vmin_vmax=(0, 8), name="block_lo", addrspace=AddrSpace.ALU)
4 Ops.PARAM dtypes.i32 [0] ParamArg(slot=5, dtype=dtypes.i32, vmin_vmax=(0, 8), name="block_hi", addrspace=AddrSpace.ALU)
5 Ops.SUB dtypes.i32 [4, 3]
6 Ops.RANGE dtypes.i32 [5] (AxisType.WEAK, 0)
7 Ops.ADD dtypes.i32 [3, 6]
8 Ops.CONST dtypes.weakint [] 7
9 Ops.CAST dtypes.i32 [8] dtypes.i32
10 Ops.SHL dtypes.i32 [7, 9]
11 Ops.CONST dtypes.weakint [] 8
12 Ops.CAST dtypes.i32 [11] dtypes.i32
13 Ops.AFTER dtypes.i32 [12, 6]
14 Ops.RANGE dtypes.i32 [13] (AxisType.WEAK, 1)
15 Ops.CONST dtypes.weakint [] 4
16 Ops.CAST dtypes.i32 [15] dtypes.i32
17 Ops.SHL dtypes.i32 [14, 16]
18 Ops.ADD dtypes.i32 [10, 17]
19 Ops.SHRINK dtypes.f32 [2, 18, 16]
20 Ops.PARAM dtypes.f32 [1] ParamArg(slot=1, dtype=dtypes.f32)
21 Ops.SHRINK dtypes.f32 [20, 18, 16]
22 Ops.LOAD dtypes.f32 [21]
23 Ops.CONST dtypes.weakint [] 0
24 Ops.CAST dtypes.i32 [23] dtypes.i32
25 Ops.INDEX dtypes.f32 [22, 24]
26 Ops.CONST dtypes.weakint [] 1
27 Ops.CAST dtypes.i32 [26] dtypes.i32
28 Ops.INDEX dtypes.f32 [22, 27]
29 Ops.CONST dtypes.weakint [] 2
30 Ops.CAST dtypes.i32 [29] dtypes.i32
31 Ops.INDEX dtypes.f32 [22, 30]
32 Ops.CONST dtypes.weakint [] 3
33 Ops.CAST dtypes.i32 [32] dtypes.i32
34 Ops.INDEX dtypes.f32 [22, 33]
35 Ops.ADD dtypes.i32 [18, 16]
36 Ops.SHRINK dtypes.f32 [20, 35, 16]
37 Ops.LOAD dtypes.f32 [36]
38 Ops.INDEX dtypes.f32 [37, 24]
39 Ops.INDEX dtypes.f32 [37, 27]
40 Ops.INDEX dtypes.f32 [37, 30]
41 Ops.INDEX dtypes.f32 [37, 33]
42 Ops.ADD dtypes.i32 [18, 12]
43 Ops.SHRINK dtypes.f32 [20, 42, 16]
44 Ops.LOAD dtypes.f32 [43]
45 Ops.INDEX dtypes.f32 [44, 24]
46 Ops.INDEX dtypes.f32 [44, 27]
47 Ops.INDEX dtypes.f32 [44, 30]
48 Ops.INDEX dtypes.f32 [44, 33]
49 Ops.CONST dtypes.weakint [] 12
50 Ops.CAST dtypes.i32 [49] dtypes.i32
51 Ops.ADD dtypes.i32 [18, 50]
52 Ops.SHRINK dtypes.f32 [20, 51, 16]
53 Ops.LOAD dtypes.f32 [52]
54 Ops.INDEX dtypes.f32 [53, 24]
55 Ops.INDEX dtypes.f32 [53, 27]
56 Ops.INDEX dtypes.f32 [53, 30]
57 Ops.INDEX dtypes.f32 [53, 33]
58 Ops.STACK dtypes.f32 [25, 28, 31, 34, 38, 39, 40, 41, 45, 46, 47, 48, 54, 55, 56, 57]
59 Ops.PARAM dtypes.f32 [11] ParamArg(slot=2, dtype=dtypes.f32)
60 Ops.INDEX dtypes.f32 [59, 7]
61 Ops.LOAD dtypes.f32 [60]
62 Ops.SUB dtypes.f32 [58, 61]
63 Ops.CONST dtypes.weakfloat [] 1.0
64 Ops.CAST dtypes.f32 [63] dtypes.f32
65 Ops.PARAM dtypes.f32 [11] ParamArg(slot=3, dtype=dtypes.f32)
66 Ops.INDEX dtypes.f32 [65, 7]
67 Ops.LOAD dtypes.f32 [66]
68 Ops.FDIV dtypes.f32 [64, 67]
69 Ops.MUL dtypes.f32 [62, 68]
70 Ops.INDEX dtypes.f32 [69, 24]
71 Ops.INDEX dtypes.f32 [69, 27]
72 Ops.INDEX dtypes.f32 [69, 30]
73 Ops.INDEX dtypes.f32 [69, 33]
74 Ops.STACK dtypes.f32 [70, 71, 72, 73]
75 Ops.STORE dtypes.void [19, 74]
76 Ops.SHRINK dtypes.f32 [2, 35, 16]
77 Ops.INDEX dtypes.f32 [69, 16]
78 Ops.CONST dtypes.weakint [] 5
79 Ops.CAST dtypes.i32 [78] dtypes.i32
80 Ops.INDEX dtypes.f32 [69, 79]
81 Ops.CONST dtypes.weakint [] 6
82 Ops.CAST dtypes.i32 [81] dtypes.i32
83 Ops.INDEX dtypes.f32 [69, 82]
84 Ops.INDEX dtypes.f32 [69, 9]
85 Ops.STACK dtypes.f32 [77, 80, 83, 84]
86 Ops.STORE dtypes.void [76, 85]
87 Ops.SHRINK dtypes.f32 [2, 42, 16]
88 Ops.INDEX dtypes.f32 [69, 12]
89 Ops.CONST dtypes.weakint [] 9
90 Ops.CAST dtypes.i32 [89] dtypes.i32
91 Ops.INDEX dtypes.f32 [69, 90]
92 Ops.CONST dtypes.weakint [] 10
93 Ops.CAST dtypes.i32 [92] dtypes.i32
94 Ops.INDEX dtypes.f32 [69, 93]
95 Ops.CONST dtypes.weakint [] 11
96 Ops.CAST dtypes.i32 [95] dtypes.i32
97 Ops.INDEX dtypes.f32 [69, 96]
98 Ops.STACK dtypes.f32 [88, 91, 94, 97]
99 Ops.STORE dtypes.void [87, 98]
100 Ops.SHRINK dtypes.f32 [2, 51, 16]
101 Ops.INDEX dtypes.f32 [69, 50]
102 Ops.CONST dtypes.weakint [] 13
103 Ops.CAST dtypes.i32 [102] dtypes.i32
104 Ops.INDEX dtypes.f32 [69, 103]
105 Ops.CONST dtypes.weakint [] 14
106 Ops.CAST dtypes.i32 [105] dtypes.i32
107 Ops.INDEX dtypes.f32 [69, 106]
108 Ops.CONST dtypes.weakint [] 15
109 Ops.CAST dtypes.i32 [108] dtypes.i32
110 Ops.INDEX dtypes.f32 [69, 109]
111 Ops.STACK dtypes.f32 [101, 104, 107, 110]
112 Ops.STORE dtypes.void [100, 111]
113 Ops.AFTER dtypes.void [0, 75, 86, 99, 112]
114 Ops.END dtypes.void [113, 14]
115 Ops.END dtypes.void [114, 6]
116 Ops.SINK dtypes.void [115] KernelInfo(name="E_8_8_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)),), optimized=True, split=(8, 4, 5))
117 Ops.CONST dtypes.weakint [] 48
118 Ops.PARAM dtypes.u8 [117] ParamArg(slot=0, dtype=dtypes.u8, name="args")
119 Ops.BITCAST dtypes.u64 [118] dtypes.u64
120 Ops.INDEX dtypes.u64 [119, 24]
121 Ops.LOAD dtypes.u64 [120]
122 Ops.REFPARAM dtypes.f32 [1, 121] ParamArg(slot=0, dtype=dtypes.f32)
123 Ops.BITCAST dtypes.i32 [118] dtypes.i32
124 Ops.INDEX dtypes.i32 [123, 12]
125 Ops.LOAD dtypes.i32 [124]
126 Ops.INDEX dtypes.i32 [123, 90]
127 Ops.LOAD dtypes.i32 [126]
128 Ops.SUB dtypes.i32 [127, 125]
129 Ops.RANGE dtypes.i32 [128] (AxisType.WEAK, 0)
130 Ops.ADD dtypes.i32 [125, 129]
131 Ops.SHL dtypes.i32 [130, 9]
132 Ops.AFTER dtypes.i32 [12, 129]
133 Ops.RANGE dtypes.i32 [132] (AxisType.WEAK, 1)
134 Ops.SHL dtypes.i32 [133, 16]
135 Ops.ADD dtypes.i32 [131, 134]
136 Ops.SHRINK dtypes.f32 [122, 135, 16]
137 Ops.INDEX dtypes.u64 [119, 27]
138 Ops.LOAD dtypes.u64 [137]
139 Ops.REFPARAM dtypes.f32 [1, 138] ParamArg(slot=1, dtype=dtypes.f32)
140 Ops.SHRINK dtypes.f32 [139, 135, 16]
141 Ops.LOAD dtypes.f32 [140]
142 Ops.INDEX dtypes.f32 [141, 24]
143 Ops.INDEX dtypes.f32 [141, 27]
144 Ops.INDEX dtypes.f32 [141, 30]
145 Ops.INDEX dtypes.f32 [141, 33]
146 Ops.ADD dtypes.i32 [135, 16]
147 Ops.SHRINK dtypes.f32 [139, 146, 16]
148 Ops.LOAD dtypes.f32 [147]
149 Ops.INDEX dtypes.f32 [148, 24]
150 Ops.INDEX dtypes.f32 [148, 27]
151 Ops.INDEX dtypes.f32 [148, 30]
152 Ops.INDEX dtypes.f32 [148, 33]
153 Ops.ADD dtypes.i32 [135, 12]
154 Ops.SHRINK dtypes.f32 [139, 153, 16]
155 Ops.LOAD dtypes.f32 [154]
156 Ops.INDEX dtypes.f32 [155, 24]
157 Ops.INDEX dtypes.f32 [155, 27]
158 Ops.INDEX dtypes.f32 [155, 30]
159 Ops.INDEX dtypes.f32 [155, 33]
160 Ops.ADD dtypes.i32 [135, 50]
161 Ops.SHRINK dtypes.f32 [139, 160, 16]
162 Ops.LOAD dtypes.f32 [161]
163 Ops.INDEX dtypes.f32 [162, 24]
164 Ops.INDEX dtypes.f32 [162, 27]
165 Ops.INDEX dtypes.f32 [162, 30]
166 Ops.INDEX dtypes.f32 [162, 33]
167 Ops.STACK dtypes.f32 [142, 143, 144, 145, 149, 150, 151, 152, 156, 157, 158, 159, 163, 164, 165, 166]
168 Ops.INDEX dtypes.u64 [119, 30]
169 Ops.LOAD dtypes.u64 [168]
170 Ops.REFPARAM dtypes.f32 [11, 169] ParamArg(slot=2, dtype=dtypes.f32)
171 Ops.INDEX dtypes.f32 [170, 130]
172 Ops.LOAD dtypes.f32 [171]
173 Ops.SUB dtypes.f32 [167, 172]
174 Ops.INDEX dtypes.u64 [119, 33]
175 Ops.LOAD dtypes.u64 [174]
176 Ops.REFPARAM dtypes.f32 [11, 175] ParamArg(slot=3, dtype=dtypes.f32)
177 Ops.INDEX dtypes.f32 [176, 130]
178 Ops.LOAD dtypes.f32 [177]
179 Ops.FDIV dtypes.f32 [64, 178]
180 Ops.MUL dtypes.f32 [173, 179]
181 Ops.INDEX dtypes.f32 [180, 24]
182 Ops.INDEX dtypes.f32 [180, 27]
183 Ops.INDEX dtypes.f32 [180, 30]
184 Ops.INDEX dtypes.f32 [180, 33]
185 Ops.STACK dtypes.f32 [181, 182, 183, 184]
186 Ops.STORE dtypes.void [136, 185]
187 Ops.SHRINK dtypes.f32 [122, 146, 16]
188 Ops.INDEX dtypes.f32 [180, 16]
189 Ops.INDEX dtypes.f32 [180, 79]
190 Ops.INDEX dtypes.f32 [180, 82]
191 Ops.INDEX dtypes.f32 [180, 9]
192 Ops.STACK dtypes.f32 [188, 189, 190, 191]
193 Ops.STORE dtypes.void [187, 192]
194 Ops.SHRINK dtypes.f32 [122, 153, 16]
195 Ops.INDEX dtypes.f32 [180, 12]
196 Ops.INDEX dtypes.f32 [180, 90]
197 Ops.INDEX dtypes.f32 [180, 93]
198 Ops.INDEX dtypes.f32 [180, 96]
199 Ops.STACK dtypes.f32 [195, 196, 197, 198]
200 Ops.STORE dtypes.void [194, 199]
201 Ops.SHRINK dtypes.f32 [122, 160, 16]
202 Ops.INDEX dtypes.f32 [180, 50]
203 Ops.INDEX dtypes.f32 [180, 103]
204 Ops.INDEX dtypes.f32 [180, 106]
205 Ops.INDEX dtypes.f32 [180, 109]
206 Ops.STACK dtypes.f32 [202, 203, 204, 205]
207 Ops.STORE dtypes.void [201, 206]
208 Ops.AFTER dtypes.void [0, 186, 193, 200, 207]
209 Ops.END dtypes.void [208, 133]
210 Ops.END dtypes.void [209, 129]
211 Ops.SINK dtypes.void [210] KernelInfo(name="E_8_8_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)),), estimates=Estimates(ops=2056, lds=8256, mem=8256), optimized=True, split=(8, 4, 5))
212 Ops.LINEAR dtypes.void [117, 118, 11, 12, 123, 124, 125, 89, 90, 126, 127, 23, 24, 119, 120, 121, 26, 27, 137, 138, 29, 30, 168, 169, 32, 33, 174, 175, 1, 122, 139, 170, 176, 0, 63, 64, 92, 93, 95, 96, 49, 50, 102, 103, 105, 106, 108, 109, 15, 16, 78, 79, 81, 82, 8, 9, 128, 129, 130, 171, 172, 177, 178, 132, 131, 179, 133, 134, 135, 160, 161, 162, 146, 147, 148, 153, 154, 155, 140, 141, 163, 164, 165, 166, 149, 150, 151, 152, 156, 157, 158, 159, 142, 143, 144, 145, 167, 173, 180, 181, 182, 183, 184, 185, 202, 203, 204, 205, 206, 188, 189, 190, 191, 192, 197, 198, 195, 196, 199, 201, 187, 194, 136, 207, 193, 200, 186, 208, 209, 210, 211]
213 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef float float16 __attribute__((aligned(4),ext_vector_type(16)));\n__attribute__((min_vector_width(512))) void E_8_8_16(unsigned char* restrict args_48) {\n  int val0 = (*(((int*)(args_48))+8));\n  int val1 = (*(((int*)(args_48))+9));\n  unsigned long val2 = (*(((unsigned long*)(args_48))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_48))+1));\n  unsigned long val4 = (*(((unsigned long*)(args_48))+2));\n  unsigned long val5 = (*(((unsigned long*)(args_48))+3));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = (val0+Lidx0);\n    float val6 = (*(((float*)(val4))+alu0));\n    float val7 = (*(((float*)(val5))+alu0));\n    for (int Lidx1 = 0; Lidx1 < 8; Lidx1++) {\n      int alu1 = ((alu0<<7)+(Lidx1<<4));\n      int alu2 = (alu1+12);\n      float4 val8 = (*((float4*)((((float*)(val3))+alu2))));\n      int alu3 = (alu1+4);\n      float4 val9 = (*((float4*)((((float*)(val3))+alu3))));\n      int alu4 = (alu1+8);\n      float4 val10 = (*((float4*)((((float*)(val3))+alu4))));\n      float4 val11 = (*((float4*)((((float*)(val3))+alu1))));\n      float16 alu5 = (((float16){val11[0],val11[1],val11[2],val11[3],val9[0],val9[1],val9[2],val9[3],val10[0],val10[1],val10[2],val10[3],val8[0],val8[1],val8[2],val8[3]}-val6)*(1.0f/val7));\n      *((float4*)((((float*)(val2))+alu2))) = (float4){alu5[12],alu5[13],alu5[14],alu5[15]};\n      *((float4*)((((float*)(val2))+alu3))) = (float4){alu5[4],alu5[5],alu5[6],alu5[7]};\n      *((float4*)((((float*)(val2))+alu4))) = (float4){alu5[8],alu5[9],alu5[10],alu5[11]};\n      *((float4*)((((float*)(val2))+alu1))) = (float4){alu5[0],alu5[1],alu5[2],alu5[3]};\n    }\n  }\n}\n"
214 Ops.PROGRAM dtypes.void [211, 212, 213] ProgramInfo(global_size=(8, 1, 1), params=(%2, %20, %59, %65, %3, %4), globals=(0, 1, 2, 3), outs=(0,), ins=(1, 2, 3), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
215 Ops.SINK dtypes.void [116, 214]

