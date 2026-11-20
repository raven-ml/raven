# tolk D80 D141 D151 golden=429d86eb9aa9bb6293a296cc77d4bf5c
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 64
2 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.PARAM dtypes.i32 [0] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 16), name="block_lo", addrspace=AddrSpace.ALU)
4 Ops.PARAM dtypes.i32 [0] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 16), name="block_hi", addrspace=AddrSpace.ALU)
5 Ops.SUB dtypes.i32 [4, 3]
6 Ops.RANGE dtypes.i32 [5] (AxisType.WEAK, 1)
7 Ops.ADD dtypes.i32 [3, 6]
8 Ops.CONST dtypes.weakint [] 2
9 Ops.CAST dtypes.i32 [8] dtypes.i32
10 Ops.SHL dtypes.i32 [7, 9]
11 Ops.CONST dtypes.weakint [] 4
12 Ops.CAST dtypes.i32 [11] dtypes.i32
13 Ops.SHRINK dtypes.f32 [2, 10, 12]
14 Ops.ALLOC dtypes.f32 [11] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
15 Ops.AFTER dtypes.f32 [14, 7]
16 Ops.CONST dtypes.weakint [] 0
17 Ops.CAST dtypes.i32 [16] dtypes.i32
18 Ops.SHRINK dtypes.f32 [15, 17, 12]
19 Ops.CONST dtypes.weakfloat [] 0.0
20 Ops.CAST dtypes.f32 [19] dtypes.f32
21 Ops.STACK dtypes.f32 [20, 20, 20, 20]
22 Ops.STORE dtypes.void [18, 21]
23 Ops.CONST dtypes.weakint [] 16
24 Ops.CAST dtypes.i32 [23] dtypes.i32
25 Ops.AFTER dtypes.i32 [24, 6]
26 Ops.RANGE dtypes.i32 [25] (AxisType.WEAK, 0)
27 Ops.AFTER dtypes.f32 [14, 22, 26]
28 Ops.SHRINK dtypes.f32 [27, 17, 12]
29 Ops.LOAD dtypes.f32 [28]
30 Ops.CONST dtypes.weakint [] 4096
31 Ops.PARAM dtypes.f32 [30] ParamArg(slot=1, dtype=dtypes.f32)
32 Ops.SHL dtypes.i32 [26, 9]
33 Ops.CONST dtypes.weakint [] 8
34 Ops.CAST dtypes.i32 [33] dtypes.i32
35 Ops.SHL dtypes.i32 [7, 34]
36 Ops.ADD dtypes.i32 [32, 35]
37 Ops.SHRINK dtypes.f32 [31, 36, 12]
38 Ops.LOAD dtypes.f32 [37]
39 Ops.INDEX dtypes.f32 [38, 17]
40 Ops.CONST dtypes.weakint [] 128
41 Ops.CAST dtypes.i32 [40] dtypes.i32
42 Ops.ADD dtypes.i32 [36, 41]
43 Ops.SHRINK dtypes.f32 [31, 42, 12]
44 Ops.LOAD dtypes.f32 [43]
45 Ops.INDEX dtypes.f32 [44, 17]
46 Ops.CONST dtypes.weakint [] 1
47 Ops.CAST dtypes.i32 [46] dtypes.i32
48 Ops.INDEX dtypes.f32 [38, 47]
49 Ops.INDEX dtypes.f32 [44, 47]
50 Ops.STACK dtypes.f32 [39, 45, 48, 49]
51 Ops.INDEX dtypes.f32 [50, 17]
52 Ops.CAST dtypes.i32 [1] dtypes.i32
53 Ops.ADD dtypes.i32 [36, 52]
54 Ops.SHRINK dtypes.f32 [31, 53, 12]
55 Ops.LOAD dtypes.f32 [54]
56 Ops.INDEX dtypes.f32 [55, 17]
57 Ops.CONST dtypes.weakint [] 192
58 Ops.CAST dtypes.i32 [57] dtypes.i32
59 Ops.ADD dtypes.i32 [36, 58]
60 Ops.SHRINK dtypes.f32 [31, 59, 12]
61 Ops.LOAD dtypes.f32 [60]
62 Ops.INDEX dtypes.f32 [61, 17]
63 Ops.INDEX dtypes.f32 [55, 47]
64 Ops.INDEX dtypes.f32 [61, 47]
65 Ops.STACK dtypes.f32 [56, 62, 63, 64]
66 Ops.INDEX dtypes.f32 [65, 17]
67 Ops.INDEX dtypes.f32 [50, 47]
68 Ops.INDEX dtypes.f32 [65, 47]
69 Ops.STACK dtypes.f32 [51, 66, 67, 68]
70 Ops.INDEX dtypes.f32 [50, 9]
71 Ops.INDEX dtypes.f32 [65, 9]
72 Ops.CONST dtypes.weakint [] 3
73 Ops.CAST dtypes.i32 [72] dtypes.i32
74 Ops.INDEX dtypes.f32 [50, 73]
75 Ops.INDEX dtypes.f32 [65, 73]
76 Ops.STACK dtypes.f32 [70, 71, 74, 75]
77 Ops.ADD dtypes.f32 [69, 76]
78 Ops.INDEX dtypes.f32 [38, 9]
79 Ops.INDEX dtypes.f32 [44, 9]
80 Ops.INDEX dtypes.f32 [38, 73]
81 Ops.INDEX dtypes.f32 [44, 73]
82 Ops.STACK dtypes.f32 [78, 79, 80, 81]
83 Ops.INDEX dtypes.f32 [82, 17]
84 Ops.INDEX dtypes.f32 [55, 9]
85 Ops.INDEX dtypes.f32 [61, 9]
86 Ops.INDEX dtypes.f32 [55, 73]
87 Ops.INDEX dtypes.f32 [61, 73]
88 Ops.STACK dtypes.f32 [84, 85, 86, 87]
89 Ops.INDEX dtypes.f32 [88, 17]
90 Ops.INDEX dtypes.f32 [82, 47]
91 Ops.INDEX dtypes.f32 [88, 47]
92 Ops.STACK dtypes.f32 [83, 89, 90, 91]
93 Ops.ADD dtypes.f32 [77, 92]
94 Ops.INDEX dtypes.f32 [82, 9]
95 Ops.INDEX dtypes.f32 [88, 9]
96 Ops.INDEX dtypes.f32 [82, 73]
97 Ops.INDEX dtypes.f32 [88, 73]
98 Ops.STACK dtypes.f32 [94, 95, 96, 97]
99 Ops.ADD dtypes.f32 [93, 98]
100 Ops.ADD dtypes.f32 [29, 99]
101 Ops.STORE dtypes.void [28, 100]
102 Ops.AFTER dtypes.void [0, 101]
103 Ops.END dtypes.void [102, 26]
104 Ops.AFTER dtypes.f32 [14, 103]
105 Ops.SHRINK dtypes.f32 [104, 17, 12]
106 Ops.LOAD dtypes.f32 [105]
107 Ops.STORE dtypes.void [13, 106]
108 Ops.AFTER dtypes.void [0, 107]
109 Ops.END dtypes.void [108, 6]
110 Ops.SINK dtypes.void [109] KernelInfo(name="r_16_4_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), optimized=True, split=(16, 2, 3))
111 Ops.CONST dtypes.weakint [] 32
112 Ops.PARAM dtypes.u8 [111] ParamArg(slot=0, dtype=dtypes.u8, name="args")
113 Ops.BITCAST dtypes.u64 [112] dtypes.u64
114 Ops.INDEX dtypes.u64 [113, 17]
115 Ops.LOAD dtypes.u64 [114]
116 Ops.REFPARAM dtypes.f32 [1, 115] ParamArg(slot=0, dtype=dtypes.f32)
117 Ops.BITCAST dtypes.i32 [112] dtypes.i32
118 Ops.INDEX dtypes.i32 [117, 12]
119 Ops.LOAD dtypes.i32 [118]
120 Ops.CONST dtypes.weakint [] 5
121 Ops.CAST dtypes.i32 [120] dtypes.i32
122 Ops.INDEX dtypes.i32 [117, 121]
123 Ops.LOAD dtypes.i32 [122]
124 Ops.SUB dtypes.i32 [123, 119]
125 Ops.RANGE dtypes.i32 [124] (AxisType.WEAK, 1)
126 Ops.ADD dtypes.i32 [119, 125]
127 Ops.SHL dtypes.i32 [126, 9]
128 Ops.SHRINK dtypes.f32 [116, 127, 12]
129 Ops.BUFFER dtypes.f32 [11] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
130 Ops.AFTER dtypes.f32 [129, 126]
131 Ops.SHRINK dtypes.f32 [130, 17, 12]
132 Ops.STORE dtypes.void [131, 21]
133 Ops.AFTER dtypes.i32 [24, 125]
134 Ops.RANGE dtypes.i32 [133] (AxisType.WEAK, 0)
135 Ops.AFTER dtypes.f32 [129, 132, 134]
136 Ops.SHRINK dtypes.f32 [135, 17, 12]
137 Ops.LOAD dtypes.f32 [136]
138 Ops.INDEX dtypes.u64 [113, 47]
139 Ops.LOAD dtypes.u64 [138]
140 Ops.REFPARAM dtypes.f32 [30, 139] ParamArg(slot=1, dtype=dtypes.f32)
141 Ops.SHL dtypes.i32 [134, 9]
142 Ops.SHL dtypes.i32 [126, 34]
143 Ops.ADD dtypes.i32 [141, 142]
144 Ops.SHRINK dtypes.f32 [140, 143, 12]
145 Ops.LOAD dtypes.f32 [144]
146 Ops.INDEX dtypes.f32 [145, 17]
147 Ops.ADD dtypes.i32 [143, 41]
148 Ops.SHRINK dtypes.f32 [140, 147, 12]
149 Ops.LOAD dtypes.f32 [148]
150 Ops.INDEX dtypes.f32 [149, 17]
151 Ops.INDEX dtypes.f32 [145, 47]
152 Ops.INDEX dtypes.f32 [149, 47]
153 Ops.STACK dtypes.f32 [146, 150, 151, 152]
154 Ops.INDEX dtypes.f32 [153, 17]
155 Ops.ADD dtypes.i32 [143, 52]
156 Ops.SHRINK dtypes.f32 [140, 155, 12]
157 Ops.LOAD dtypes.f32 [156]
158 Ops.INDEX dtypes.f32 [157, 17]
159 Ops.ADD dtypes.i32 [143, 58]
160 Ops.SHRINK dtypes.f32 [140, 159, 12]
161 Ops.LOAD dtypes.f32 [160]
162 Ops.INDEX dtypes.f32 [161, 17]
163 Ops.INDEX dtypes.f32 [157, 47]
164 Ops.INDEX dtypes.f32 [161, 47]
165 Ops.STACK dtypes.f32 [158, 162, 163, 164]
166 Ops.INDEX dtypes.f32 [165, 17]
167 Ops.INDEX dtypes.f32 [153, 47]
168 Ops.INDEX dtypes.f32 [165, 47]
169 Ops.STACK dtypes.f32 [154, 166, 167, 168]
170 Ops.INDEX dtypes.f32 [153, 9]
171 Ops.INDEX dtypes.f32 [165, 9]
172 Ops.INDEX dtypes.f32 [153, 73]
173 Ops.INDEX dtypes.f32 [165, 73]
174 Ops.STACK dtypes.f32 [170, 171, 172, 173]
175 Ops.ADD dtypes.f32 [169, 174]
176 Ops.INDEX dtypes.f32 [145, 9]
177 Ops.INDEX dtypes.f32 [149, 9]
178 Ops.INDEX dtypes.f32 [145, 73]
179 Ops.INDEX dtypes.f32 [149, 73]
180 Ops.STACK dtypes.f32 [176, 177, 178, 179]
181 Ops.INDEX dtypes.f32 [180, 17]
182 Ops.INDEX dtypes.f32 [157, 9]
183 Ops.INDEX dtypes.f32 [161, 9]
184 Ops.INDEX dtypes.f32 [157, 73]
185 Ops.INDEX dtypes.f32 [161, 73]
186 Ops.STACK dtypes.f32 [182, 183, 184, 185]
187 Ops.INDEX dtypes.f32 [186, 17]
188 Ops.INDEX dtypes.f32 [180, 47]
189 Ops.INDEX dtypes.f32 [186, 47]
190 Ops.STACK dtypes.f32 [181, 187, 188, 189]
191 Ops.ADD dtypes.f32 [175, 190]
192 Ops.INDEX dtypes.f32 [180, 9]
193 Ops.INDEX dtypes.f32 [186, 9]
194 Ops.INDEX dtypes.f32 [180, 73]
195 Ops.INDEX dtypes.f32 [186, 73]
196 Ops.STACK dtypes.f32 [192, 193, 194, 195]
197 Ops.ADD dtypes.f32 [191, 196]
198 Ops.ADD dtypes.f32 [137, 197]
199 Ops.STORE dtypes.void [136, 198]
200 Ops.AFTER dtypes.void [0, 199]
201 Ops.END dtypes.void [200, 134]
202 Ops.AFTER dtypes.f32 [129, 201]
203 Ops.SHRINK dtypes.f32 [202, 17, 12]
204 Ops.LOAD dtypes.f32 [203]
205 Ops.STORE dtypes.void [128, 204]
206 Ops.AFTER dtypes.void [0, 205]
207 Ops.END dtypes.void [206, 125]
208 Ops.SINK dtypes.void [207] KernelInfo(name="r_16_4_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), estimates=Estimates(ops=4096, lds=16640, mem=16640), optimized=True, split=(16, 2, 3))
209 Ops.LINEAR dtypes.void [111, 112, 11, 129, 12, 117, 118, 119, 120, 121, 122, 123, 16, 17, 113, 114, 115, 46, 47, 138, 139, 1, 116, 30, 140, 0, 19, 20, 21, 40, 41, 23, 24, 57, 58, 8, 9, 72, 73, 52, 33, 34, 124, 125, 133, 126, 130, 131, 142, 132, 134, 141, 143, 147, 148, 149, 159, 160, 161, 155, 156, 157, 144, 145, 135, 136, 137, 162, 164, 158, 163, 165, 150, 152, 146, 151, 153, 166, 168, 154, 167, 169, 171, 173, 170, 172, 174, 183, 185, 182, 184, 186, 177, 179, 176, 178, 180, 187, 189, 181, 188, 190, 193, 195, 192, 194, 196, 175, 191, 197, 198, 199, 200, 201, 202, 203, 204, 127, 128, 205, 206, 207, 208]
210 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_16_4_16_4(unsigned char* restrict args_32) {\n  float buf0[4];\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    *((float4*)((buf0+0))) = (float4){0.0f,0.0f,0.0f,0.0f};\n    for (int Lidx0 = 0; Lidx0 < 16; Lidx0++) {\n      int alu2 = ((Lidx0<<2)+(alu0<<8));\n      float4 val4 = (*((float4*)((((float*)(val3))+(alu2+128)))));\n      float4 val5 = (*((float4*)((((float*)(val3))+(alu2+192)))));\n      float4 val6 = (*((float4*)((((float*)(val3))+(alu2+64)))));\n      float4 val7 = (*((float4*)((((float*)(val3))+alu2))));\n      float4 cast0 = ((float4)__builtin_shufflevector(val6, val5, 0, 4, 1, 5));\n      float4 cast1 = ((float4)__builtin_shufflevector(val7, val4, 0, 4, 1, 5));\n      float4 cast2 = ((float4)__builtin_shufflevector(val6, val5, 2, 6, 3, 7));\n      float4 cast3 = ((float4)__builtin_shufflevector(val7, val4, 2, 6, 3, 7));\n      *((float4*)((buf0+0))) = ((*((float4*)((buf0+0))))+(((float4)__builtin_shufflevector(cast1, cast0, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast1, cast0, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast3, cast2, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast3, cast2, 2, 6, 3, 7))));\n    }\n    *((float4*)((((float*)(val2))+(alu0<<2)))) = (*((float4*)((buf0+0))));\n  }\n}\n"
211 Ops.PROGRAM dtypes.void [208, 209, 210] ProgramInfo(global_size=(16, 1, 1), params=(%2, %31, %3, %4), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
212 Ops.SINK dtypes.void [110, 211]

