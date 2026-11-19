# tolk D25 D141 D151 golden=517bed0be8d278b20d5eeacf8135b1bf
0 Ops.CONST dtypes.weakint [] 3
1 Ops.CONST dtypes.weakint [] 24
2 Ops.PARAM dtypes.u8 [1] ParamArg(slot=0, dtype=dtypes.u8, name="args")
3 Ops.BITCAST dtypes.u64 [2] dtypes.u64
4 Ops.CONST dtypes.weakint [] 0
5 Ops.CAST dtypes.i32 [4] dtypes.i32
6 Ops.INDEX dtypes.u64 [3, 5]
7 Ops.LOAD dtypes.u64 [6]
8 Ops.REFPARAM dtypes.f32 [0, 7] ParamArg(slot=0, dtype=dtypes.f32)
9 Ops.CONST dtypes.weakint [] 2
10 Ops.CAST dtypes.i32 [9] dtypes.i32
11 Ops.SHRINK dtypes.f32 [8, 5, 10]
12 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
13 Ops.STACK dtypes.void []
14 Ops.SHRINK dtypes.f32 [12, 5, 10]
15 Ops.CONST dtypes.weakfloat [] 0.0
16 Ops.CAST dtypes.f32 [15] dtypes.f32
17 Ops.STACK dtypes.f32 [16, 16]
18 Ops.STORE dtypes.void [14, 17]
19 Ops.INDEX dtypes.f32 [12, 10]
20 Ops.STORE dtypes.void [19, 16]
21 Ops.BITCAST dtypes.i32 [2] dtypes.i32
22 Ops.CONST dtypes.weakint [] 4
23 Ops.CAST dtypes.i32 [22] dtypes.i32
24 Ops.INDEX dtypes.i32 [21, 23]
25 Ops.LOAD dtypes.i32 [24]
26 Ops.RANGE dtypes.i32 [25] (AxisType.WEAK, 0)
27 Ops.AFTER dtypes.f32 [12, 18, 20, 26]
28 Ops.SHRINK dtypes.f32 [27, 5, 10]
29 Ops.LOAD dtypes.f32 [28]
30 Ops.INDEX dtypes.f32 [29, 5]
31 Ops.CONST dtypes.weakint [] 1
32 Ops.CAST dtypes.i32 [31] dtypes.i32
33 Ops.INDEX dtypes.f32 [29, 32]
34 Ops.INDEX dtypes.f32 [27, 10]
35 Ops.LOAD dtypes.f32 [34]
36 Ops.STACK dtypes.f32 [30, 33, 35]
37 Ops.CONST dtypes.weakint [] 30
38 Ops.INDEX dtypes.u64 [3, 32]
39 Ops.LOAD dtypes.u64 [38]
40 Ops.REFPARAM dtypes.f32 [37, 39] ParamArg(slot=1, dtype=dtypes.f32)
41 Ops.CAST dtypes.i32 [0] dtypes.i32
42 Ops.MUL dtypes.i32 [26, 41]
43 Ops.INDEX dtypes.f32 [40, 42]
44 Ops.LOAD dtypes.f32 [43]
45 Ops.ADD dtypes.i32 [42, 32]
46 Ops.INDEX dtypes.f32 [40, 45]
47 Ops.LOAD dtypes.f32 [46]
48 Ops.ADD dtypes.i32 [42, 10]
49 Ops.INDEX dtypes.f32 [40, 48]
50 Ops.LOAD dtypes.f32 [49]
51 Ops.STACK dtypes.f32 [44, 47, 50]
52 Ops.ADD dtypes.f32 [36, 51]
53 Ops.INDEX dtypes.f32 [52, 5]
54 Ops.INDEX dtypes.f32 [52, 32]
55 Ops.STACK dtypes.f32 [53, 54]
56 Ops.STORE dtypes.void [28, 55]
57 Ops.INDEX dtypes.f32 [52, 10]
58 Ops.STORE dtypes.void [34, 57]
59 Ops.AFTER dtypes.void [13, 56, 58]
60 Ops.END dtypes.void [59, 26]
61 Ops.AFTER dtypes.f32 [12, 60]
62 Ops.SHRINK dtypes.f32 [61, 5, 10]
63 Ops.LOAD dtypes.f32 [62]
64 Ops.INDEX dtypes.f32 [63, 5]
65 Ops.INDEX dtypes.f32 [63, 32]
66 Ops.INDEX dtypes.f32 [61, 10]
67 Ops.LOAD dtypes.f32 [66]
68 Ops.STACK dtypes.f32 [64, 65, 67]
69 Ops.CONST dtypes.weakfloat [] 1.0
70 Ops.CAST dtypes.f32 [69] dtypes.f32
71 Ops.CAST dtypes.f32 [25] dtypes.f32
72 Ops.FDIV dtypes.f32 [70, 71]
73 Ops.MUL dtypes.f32 [68, 72]
74 Ops.INDEX dtypes.f32 [73, 5]
75 Ops.INDEX dtypes.f32 [73, 32]
76 Ops.STACK dtypes.f32 [74, 75]
77 Ops.STORE dtypes.void [11, 76]
78 Ops.INDEX dtypes.f32 [8, 10]
79 Ops.INDEX dtypes.f32 [73, 10]
80 Ops.STORE dtypes.void [78, 79]
81 Ops.PARAM dtypes.i32 [13] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(1, 10), name="i", addrspace=AddrSpace.ALU)
82 Ops.MUL dtypes.i32 [81, 0]
83 Ops.ADD dtypes.i32 [82, 22]
84 Ops.MUL dtypes.i32 [31, 81]
85 Ops.MUL dtypes.i32 [22, 84]
86 Ops.ADD dtypes.i32 [4, 85]
87 Ops.ADD dtypes.i32 [86, 85]
88 Ops.ADD dtypes.i32 [87, 85]
89 Ops.ADD dtypes.i32 [88, 22]
90 Ops.CONST dtypes.weakint [] 8
91 Ops.ADD dtypes.i32 [89, 90]
92 Ops.CONST dtypes.weakint [] 12
93 Ops.MUL dtypes.i32 [81, 92]
94 Ops.ADD dtypes.i32 [4, 93]
95 Ops.ADD dtypes.i32 [94, 92]
96 Ops.SINK dtypes.void [77, 77, 80] KernelInfo(name="r_3_i", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(3, AxisType.UPCAST)),), estimates=Estimates(ops=%83, lds=%91, mem=%95), optimized=True)
97 Ops.LINEAR dtypes.void [1, 2, 0, 12, 22, 23, 21, 24, 25, 31, 32, 3, 38, 39, 37, 40, 13, 15, 16, 17, 9, 10, 19, 4, 5, 14, 41, 20, 18, 26, 42, 45, 46, 47, 48, 49, 50, 43, 44, 27, 34, 35, 28, 29, 30, 33, 36, 51, 52, 53, 54, 55, 57, 58, 56, 59, 60, 6, 7, 61, 66, 67, 62, 63, 8, 64, 65, 68, 71, 69, 70, 72, 73, 74, 75, 76, 78, 79, 11, 80, 77, 96]
98 Ops.SOURCE dtypes.void [] "typedef float float2 __attribute__((aligned(4),ext_vector_type(2)));\ntypedef float float3 __attribute__((aligned(4),ext_vector_type(3)));\n__attribute__((min_vector_width(512))) void r_3_i(unsigned char* restrict args_24) {\n  float buf0[3];\n  int val0 = (*(((int*)(args_24))+4));\n  unsigned long val1 = (*(((unsigned long*)(args_24))+1));\n  *(buf0+2) = 0.0f;\n  *((float2*)((buf0+0))) = (float2){0.0f,0.0f};\n  for (int Lidx0 = 0; Lidx0 < val0; Lidx0++) {\n    int alu2 = (Lidx0*3);\n    float val2 = (*(((float*)(val1))+(alu2+1)));\n    float val3 = (*(((float*)(val1))+(alu2+2)));\n    float val4 = (*(((float*)(val1))+alu2));\n    float2 val5 = (*((float2*)((buf0+0))));\n    float3 alu3 = ((float3){val5[0],val5[1],(*(buf0+2))}+(float3){val4,val2,val3});\n    *(buf0+2) = alu3[2];\n    *((float2*)((buf0+0))) = (float2){alu3[0],alu3[1]};\n  }\n  unsigned long val6 = (*(((unsigned long*)(args_24))+0));\n  float2 val7 = (*((float2*)((buf0+0))));\n  float3 alu7 = ((float3){val7[0],val7[1],(*(buf0+2))}*(1.0f/((float)(val0))));\n  *(((float*)(val6))+2) = alu7[2];\n  *((float2*)((((float*)(val6))+0))) = (float2){alu7[0],alu7[1]};\n}\n"
99 Ops.BINARY dtypes.u8 [] b"typedef float float2 __attribute__((aligned(4),ext_vector_type(2)));\ntypedef float float3 __attribute__((aligned(4),ext_vector_type(3)));\n__attribute__((min_vector_width(512))) void r_3_i(unsigned char* restrict args_24) {\n  float buf0[3];\n  int val0 = (*(((int*)(args_24))+4));\n  unsigned long val1 = (*(((unsigned long*)(args_24))+1));\n  *(buf0+2) = 0.0f;\n  *((float2*)((buf0+0))) = (float2){0.0f,0.0f};\n  for (int Lidx0 = 0; Lidx0 < val0; Lidx0++) {\n    int alu2 = (Lidx0*3);\n    float val2 = (*(((float*)(val1))+(alu2+1)));\n    float val3 = (*(((float*)(val1))+(alu2+2)));\n    float val4 = (*(((float*)(val1))+alu2));\n    float2 val5 = (*((float2*)((buf0+0))));\n    float3 alu3 = ((float3){val5[0],val5[1],(*(buf0+2))}+(float3){val4,val2,val3});\n    *(buf0+2) = alu3[2];\n    *((float2*)((buf0+0))) = (float2){alu3[0],alu3[1]};\n  }\n  unsigned long val6 = (*(((unsigned long*)(args_24))+0));\n  float2 val7 = (*((float2*)((buf0+0))));\n  float3 alu7 = ((float3){val7[0],val7[1],(*(buf0+2))}*(1.0f/((float)(val0))));\n  *(((float*)(val6))+2) = alu7[2];\n  *((float2*)((((float*)(val6))+0))) = (float2){alu7[0],alu7[1]};\n}\n"
100 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
101 Ops.PARAM dtypes.f32 [37] ParamArg(slot=1, dtype=dtypes.f32)
102 Ops.PROGRAM dtypes.void [96, 97, 98, 99] ProgramInfo(params=(%100, %101, %81), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
103 Ops.CONST dtypes.weakint [] 256
104 Ops.ALLOC dtypes.i8 [103] ParamArg(slot=5, dtype=dtypes.i8, device="CPU")
105 Ops.SHRINK dtypes.i8 [104, 4, 92]
106 Ops.BITCAST dtypes.f32 [105] dtypes.f32
107 Ops.PARAM dtypes.f32 [37] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
108 Ops.CALL dtypes.void [102, 106, 107] CallInfo()
109 Ops.CONST dtypes.weakint [] 32
110 Ops.PARAM dtypes.u8 [109] ParamArg(slot=0, dtype=dtypes.u8, name="args")
111 Ops.BITCAST dtypes.u64 [110] dtypes.u64
112 Ops.INDEX dtypes.u64 [111, 5]
113 Ops.LOAD dtypes.u64 [112]
114 Ops.REFPARAM dtypes.f32 [0, 113] ParamArg(slot=0, dtype=dtypes.f32)
115 Ops.SHRINK dtypes.f32 [114, 5, 10]
116 Ops.BITCAST dtypes.i32 [110] dtypes.i32
117 Ops.CONST dtypes.weakint [] 6
118 Ops.CAST dtypes.i32 [117] dtypes.i32
119 Ops.INDEX dtypes.i32 [116, 118]
120 Ops.LOAD dtypes.i32 [119]
121 Ops.RANGE dtypes.i32 [120] (AxisType.WEAK, 0)
122 Ops.AFTER dtypes.f32 [12, 18, 20, 121]
123 Ops.SHRINK dtypes.f32 [122, 5, 10]
124 Ops.INDEX dtypes.u64 [111, 32]
125 Ops.LOAD dtypes.u64 [124]
126 Ops.REFPARAM dtypes.f32 [37, 125] ParamArg(slot=1, dtype=dtypes.f32)
127 Ops.MUL dtypes.i32 [121, 41]
128 Ops.INDEX dtypes.f32 [126, 127]
129 Ops.LOAD dtypes.f32 [128]
130 Ops.ADD dtypes.i32 [127, 32]
131 Ops.INDEX dtypes.f32 [126, 130]
132 Ops.LOAD dtypes.f32 [131]
133 Ops.ADD dtypes.i32 [127, 10]
134 Ops.INDEX dtypes.f32 [126, 133]
135 Ops.LOAD dtypes.f32 [134]
136 Ops.STACK dtypes.f32 [129, 132, 135]
137 Ops.INDEX dtypes.u64 [111, 10]
138 Ops.LOAD dtypes.u64 [137]
139 Ops.REFPARAM dtypes.f32 [0, 138] ParamArg(slot=2, dtype=dtypes.f32)
140 Ops.SHRINK dtypes.f32 [139, 5, 10]
141 Ops.LOAD dtypes.f32 [140]
142 Ops.INDEX dtypes.f32 [141, 5]
143 Ops.INDEX dtypes.f32 [141, 32]
144 Ops.INDEX dtypes.f32 [139, 10]
145 Ops.LOAD dtypes.f32 [144]
146 Ops.STACK dtypes.f32 [142, 143, 145]
147 Ops.SUB dtypes.f32 [136, 146]
148 Ops.LOAD dtypes.f32 [123]
149 Ops.INDEX dtypes.f32 [148, 5]
150 Ops.INDEX dtypes.f32 [148, 32]
151 Ops.INDEX dtypes.f32 [122, 10]
152 Ops.LOAD dtypes.f32 [151]
153 Ops.STACK dtypes.f32 [149, 150, 152]
154 Ops.MULACC dtypes.f32 [147, 147, 153]
155 Ops.INDEX dtypes.f32 [154, 5]
156 Ops.INDEX dtypes.f32 [154, 32]
157 Ops.STACK dtypes.f32 [155, 156]
158 Ops.STORE dtypes.void [123, 157]
159 Ops.INDEX dtypes.f32 [154, 10]
160 Ops.STORE dtypes.void [151, 159]
161 Ops.AFTER dtypes.void [13, 158, 160]
162 Ops.END dtypes.void [161, 121]
163 Ops.AFTER dtypes.f32 [12, 162]
164 Ops.SHRINK dtypes.f32 [163, 5, 10]
165 Ops.LOAD dtypes.f32 [164]
166 Ops.INDEX dtypes.f32 [165, 5]
167 Ops.INDEX dtypes.f32 [165, 32]
168 Ops.INDEX dtypes.f32 [163, 10]
169 Ops.LOAD dtypes.f32 [168]
170 Ops.STACK dtypes.f32 [166, 167, 169]
171 Ops.CONST dtypes.weakint [] -1
172 Ops.CAST dtypes.i32 [171] dtypes.i32
173 Ops.ADD dtypes.i32 [120, 172]
174 Ops.CAST dtypes.f32 [173] dtypes.f32
175 Ops.FDIV dtypes.f32 [70, 174]
176 Ops.MUL dtypes.f32 [170, 175]
177 Ops.INDEX dtypes.f32 [176, 5]
178 Ops.INDEX dtypes.f32 [176, 32]
179 Ops.STACK dtypes.f32 [177, 178]
180 Ops.STORE dtypes.void [115, 179]
181 Ops.INDEX dtypes.f32 [114, 10]
182 Ops.INDEX dtypes.f32 [176, 10]
183 Ops.STORE dtypes.void [181, 182]
184 Ops.PARAM dtypes.i32 [13] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(1, 10), name="i", addrspace=AddrSpace.ALU)
185 Ops.CONST dtypes.weakint [] 9
186 Ops.MUL dtypes.i32 [184, 185]
187 Ops.CONST dtypes.weakint [] 5
188 Ops.ADD dtypes.i32 [186, 187]
189 Ops.MUL dtypes.i32 [31, 184]
190 Ops.MUL dtypes.i32 [22, 189]
191 Ops.ADD dtypes.i32 [92, 190]
192 Ops.ADD dtypes.i32 [191, 190]
193 Ops.ADD dtypes.i32 [192, 190]
194 Ops.ADD dtypes.i32 [193, 22]
195 Ops.ADD dtypes.i32 [194, 90]
196 Ops.MUL dtypes.i32 [184, 92]
197 Ops.ADD dtypes.i32 [92, 196]
198 Ops.ADD dtypes.i32 [197, 92]
199 Ops.SINK dtypes.void [180, 180, 183] KernelInfo(name="r_3_i", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(3, AxisType.UPCAST)),), estimates=Estimates(ops=%188, lds=%195, mem=%198), optimized=True)
200 Ops.LINEAR dtypes.void [109, 110, 0, 12, 117, 118, 116, 119, 120, 31, 32, 111, 124, 125, 9, 10, 137, 138, 139, 144, 145, 4, 5, 140, 141, 37, 126, 13, 142, 143, 146, 15, 16, 17, 19, 14, 41, 20, 18, 121, 127, 130, 131, 132, 133, 134, 135, 128, 129, 122, 151, 152, 123, 148, 149, 150, 153, 136, 147, 154, 155, 156, 157, 159, 160, 158, 161, 162, 112, 113, 163, 168, 169, 164, 165, 114, 166, 167, 170, 171, 172, 173, 174, 69, 70, 175, 176, 177, 178, 179, 181, 182, 115, 183, 180, 199]
201 Ops.SOURCE dtypes.void [] "typedef float float2 __attribute__((aligned(4),ext_vector_type(2)));\ntypedef float float3 __attribute__((aligned(4),ext_vector_type(3)));\n__attribute__((min_vector_width(512))) void r_3_i(unsigned char* restrict args_32) {\n  float buf0[3];\n  int val0 = (*(((int*)(args_32))+6));\n  unsigned long val1 = (*(((unsigned long*)(args_32))+1));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+2));\n  float val3 = (*(((float*)(val2))+2));\n  float2 val4 = (*((float2*)((((float*)(val2))+0))));\n  *(buf0+2) = 0.0f;\n  *((float2*)((buf0+0))) = (float2){0.0f,0.0f};\n  for (int Lidx0 = 0; Lidx0 < val0; Lidx0++) {\n    int alu2 = (Lidx0*3);\n    float val5 = (*(((float*)(val1))+(alu2+1)));\n    float val6 = (*(((float*)(val1))+(alu2+2)));\n    float val7 = (*(((float*)(val1))+alu2));\n    float2 val8 = (*((float2*)((buf0+0))));\n    float3 alu3 = ((float3){val7,val5,val6}-(float3){val4[0],val4[1],val3});\n    float3 alu4 = __builtin_elementwise_fma(alu3, alu3, (float3){val8[0],val8[1],(*(buf0+2))});\n    *(buf0+2) = alu4[2];\n    *((float2*)((buf0+0))) = (float2){alu4[0],alu4[1]};\n  }\n  unsigned long val9 = (*(((unsigned long*)(args_32))+0));\n  float2 val10 = (*((float2*)((buf0+0))));\n  float3 alu8 = ((float3){val10[0],val10[1],(*(buf0+2))}*(1.0f/((float)((val0+-1)))));\n  *(((float*)(val9))+2) = alu8[2];\n  *((float2*)((((float*)(val9))+0))) = (float2){alu8[0],alu8[1]};\n}\n"
202 Ops.BINARY dtypes.u8 [] b"typedef float float2 __attribute__((aligned(4),ext_vector_type(2)));\ntypedef float float3 __attribute__((aligned(4),ext_vector_type(3)));\n__attribute__((min_vector_width(512))) void r_3_i(unsigned char* restrict args_32) {\n  float buf0[3];\n  int val0 = (*(((int*)(args_32))+6));\n  unsigned long val1 = (*(((unsigned long*)(args_32))+1));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+2));\n  float val3 = (*(((float*)(val2))+2));\n  float2 val4 = (*((float2*)((((float*)(val2))+0))));\n  *(buf0+2) = 0.0f;\n  *((float2*)((buf0+0))) = (float2){0.0f,0.0f};\n  for (int Lidx0 = 0; Lidx0 < val0; Lidx0++) {\n    int alu2 = (Lidx0*3);\n    float val5 = (*(((float*)(val1))+(alu2+1)));\n    float val6 = (*(((float*)(val1))+(alu2+2)));\n    float val7 = (*(((float*)(val1))+alu2));\n    float2 val8 = (*((float2*)((buf0+0))));\n    float3 alu3 = ((float3){val7,val5,val6}-(float3){val4[0],val4[1],val3});\n    float3 alu4 = __builtin_elementwise_fma(alu3, alu3, (float3){val8[0],val8[1],(*(buf0+2))});\n    *(buf0+2) = alu4[2];\n    *((float2*)((buf0+0))) = (float2){alu4[0],alu4[1]};\n  }\n  unsigned long val9 = (*(((unsigned long*)(args_32))+0));\n  float2 val10 = (*((float2*)((buf0+0))));\n  float3 alu8 = ((float3){val10[0],val10[1],(*(buf0+2))}*(1.0f/((float)((val0+-1)))));\n  *(((float*)(val9))+2) = alu8[2];\n  *((float2*)((((float*)(val9))+0))) = (float2){alu8[0],alu8[1]};\n}\n"
203 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
204 Ops.PROGRAM dtypes.void [199, 200, 201, 202] ProgramInfo(params=(%100, %101, %203, %184), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CPU", arch="x86_64,x86-64"))
205 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
206 Ops.CALL dtypes.void [204, 205, 107, 106] CallInfo()
207 Ops.LINEAR dtypes.void [108, 206]

