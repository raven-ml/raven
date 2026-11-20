# tolk D25 D80 D141 D151 golden=da6c513849182ee938a3ab783e96b4ac
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
39 Ops.INDEX dtypes.f32 [38, 9]
40 Ops.CONST dtypes.weakint [] 128
41 Ops.CAST dtypes.i32 [40] dtypes.i32
42 Ops.ADD dtypes.i32 [36, 41]
43 Ops.SHRINK dtypes.f32 [31, 42, 12]
44 Ops.LOAD dtypes.f32 [43]
45 Ops.INDEX dtypes.f32 [44, 9]
46 Ops.CONST dtypes.weakint [] 3
47 Ops.CAST dtypes.i32 [46] dtypes.i32
48 Ops.INDEX dtypes.f32 [38, 47]
49 Ops.INDEX dtypes.f32 [44, 47]
50 Ops.STACK dtypes.f32 [39, 45, 48, 49]
51 Ops.INDEX dtypes.f32 [50, 9]
52 Ops.CAST dtypes.i32 [1] dtypes.i32
53 Ops.ADD dtypes.i32 [36, 52]
54 Ops.SHRINK dtypes.f32 [31, 53, 12]
55 Ops.LOAD dtypes.f32 [54]
56 Ops.INDEX dtypes.f32 [55, 9]
57 Ops.CONST dtypes.weakint [] 192
58 Ops.CAST dtypes.i32 [57] dtypes.i32
59 Ops.ADD dtypes.i32 [36, 58]
60 Ops.SHRINK dtypes.f32 [31, 59, 12]
61 Ops.LOAD dtypes.f32 [60]
62 Ops.INDEX dtypes.f32 [61, 9]
63 Ops.INDEX dtypes.f32 [55, 47]
64 Ops.INDEX dtypes.f32 [61, 47]
65 Ops.STACK dtypes.f32 [56, 62, 63, 64]
66 Ops.INDEX dtypes.f32 [65, 9]
67 Ops.INDEX dtypes.f32 [50, 47]
68 Ops.INDEX dtypes.f32 [65, 47]
69 Ops.STACK dtypes.f32 [51, 66, 67, 68]
70 Ops.CONST dtypes.weakfloat [] 2.0
71 Ops.CAST dtypes.f32 [70] dtypes.f32
72 Ops.INDEX dtypes.f32 [50, 17]
73 Ops.INDEX dtypes.f32 [65, 17]
74 Ops.CONST dtypes.weakint [] 1
75 Ops.CAST dtypes.i32 [74] dtypes.i32
76 Ops.INDEX dtypes.f32 [50, 75]
77 Ops.INDEX dtypes.f32 [65, 75]
78 Ops.STACK dtypes.f32 [72, 73, 76, 77]
79 Ops.INDEX dtypes.f32 [38, 17]
80 Ops.INDEX dtypes.f32 [44, 17]
81 Ops.INDEX dtypes.f32 [38, 75]
82 Ops.INDEX dtypes.f32 [44, 75]
83 Ops.STACK dtypes.f32 [79, 80, 81, 82]
84 Ops.INDEX dtypes.f32 [83, 9]
85 Ops.INDEX dtypes.f32 [55, 17]
86 Ops.INDEX dtypes.f32 [61, 17]
87 Ops.INDEX dtypes.f32 [55, 75]
88 Ops.INDEX dtypes.f32 [61, 75]
89 Ops.STACK dtypes.f32 [85, 86, 87, 88]
90 Ops.INDEX dtypes.f32 [89, 9]
91 Ops.INDEX dtypes.f32 [83, 47]
92 Ops.INDEX dtypes.f32 [89, 47]
93 Ops.STACK dtypes.f32 [84, 90, 91, 92]
94 Ops.INDEX dtypes.f32 [83, 17]
95 Ops.INDEX dtypes.f32 [89, 17]
96 Ops.INDEX dtypes.f32 [83, 75]
97 Ops.INDEX dtypes.f32 [89, 75]
98 Ops.STACK dtypes.f32 [94, 95, 96, 97]
99 Ops.MUL dtypes.f32 [98, 71]
100 Ops.MULACC dtypes.f32 [93, 71, 99]
101 Ops.MULACC dtypes.f32 [78, 71, 100]
102 Ops.MULACC dtypes.f32 [69, 71, 101]
103 Ops.ADD dtypes.f32 [29, 102]
104 Ops.STORE dtypes.void [28, 103]
105 Ops.AFTER dtypes.void [0, 104]
106 Ops.END dtypes.void [105, 26]
107 Ops.AFTER dtypes.f32 [14, 106]
108 Ops.SHRINK dtypes.f32 [107, 17, 12]
109 Ops.LOAD dtypes.f32 [108]
110 Ops.STORE dtypes.void [13, 109]
111 Ops.AFTER dtypes.void [0, 110]
112 Ops.END dtypes.void [111, 6]
113 Ops.SINK dtypes.void [112] KernelInfo(name="r_16_4_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), optimized=True, split=(16, 2, 3))
114 Ops.CONST dtypes.weakint [] 32
115 Ops.PARAM dtypes.u8 [114] ParamArg(slot=0, dtype=dtypes.u8, name="args")
116 Ops.BITCAST dtypes.u64 [115] dtypes.u64
117 Ops.INDEX dtypes.u64 [116, 17]
118 Ops.LOAD dtypes.u64 [117]
119 Ops.REFPARAM dtypes.f32 [1, 118] ParamArg(slot=0, dtype=dtypes.f32)
120 Ops.BITCAST dtypes.i32 [115] dtypes.i32
121 Ops.INDEX dtypes.i32 [120, 12]
122 Ops.LOAD dtypes.i32 [121]
123 Ops.CONST dtypes.weakint [] 5
124 Ops.CAST dtypes.i32 [123] dtypes.i32
125 Ops.INDEX dtypes.i32 [120, 124]
126 Ops.LOAD dtypes.i32 [125]
127 Ops.SUB dtypes.i32 [126, 122]
128 Ops.RANGE dtypes.i32 [127] (AxisType.WEAK, 1)
129 Ops.ADD dtypes.i32 [122, 128]
130 Ops.SHL dtypes.i32 [129, 9]
131 Ops.SHRINK dtypes.f32 [119, 130, 12]
132 Ops.BUFFER dtypes.f32 [11] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
133 Ops.AFTER dtypes.f32 [132, 129]
134 Ops.SHRINK dtypes.f32 [133, 17, 12]
135 Ops.STORE dtypes.void [134, 21]
136 Ops.AFTER dtypes.i32 [24, 128]
137 Ops.RANGE dtypes.i32 [136] (AxisType.WEAK, 0)
138 Ops.AFTER dtypes.f32 [132, 135, 137]
139 Ops.SHRINK dtypes.f32 [138, 17, 12]
140 Ops.LOAD dtypes.f32 [139]
141 Ops.INDEX dtypes.u64 [116, 75]
142 Ops.LOAD dtypes.u64 [141]
143 Ops.REFPARAM dtypes.f32 [30, 142] ParamArg(slot=1, dtype=dtypes.f32)
144 Ops.SHL dtypes.i32 [137, 9]
145 Ops.SHL dtypes.i32 [129, 34]
146 Ops.ADD dtypes.i32 [144, 145]
147 Ops.SHRINK dtypes.f32 [143, 146, 12]
148 Ops.LOAD dtypes.f32 [147]
149 Ops.INDEX dtypes.f32 [148, 9]
150 Ops.ADD dtypes.i32 [146, 41]
151 Ops.SHRINK dtypes.f32 [143, 150, 12]
152 Ops.LOAD dtypes.f32 [151]
153 Ops.INDEX dtypes.f32 [152, 9]
154 Ops.INDEX dtypes.f32 [148, 47]
155 Ops.INDEX dtypes.f32 [152, 47]
156 Ops.STACK dtypes.f32 [149, 153, 154, 155]
157 Ops.INDEX dtypes.f32 [156, 9]
158 Ops.ADD dtypes.i32 [146, 52]
159 Ops.SHRINK dtypes.f32 [143, 158, 12]
160 Ops.LOAD dtypes.f32 [159]
161 Ops.INDEX dtypes.f32 [160, 9]
162 Ops.ADD dtypes.i32 [146, 58]
163 Ops.SHRINK dtypes.f32 [143, 162, 12]
164 Ops.LOAD dtypes.f32 [163]
165 Ops.INDEX dtypes.f32 [164, 9]
166 Ops.INDEX dtypes.f32 [160, 47]
167 Ops.INDEX dtypes.f32 [164, 47]
168 Ops.STACK dtypes.f32 [161, 165, 166, 167]
169 Ops.INDEX dtypes.f32 [168, 9]
170 Ops.INDEX dtypes.f32 [156, 47]
171 Ops.INDEX dtypes.f32 [168, 47]
172 Ops.STACK dtypes.f32 [157, 169, 170, 171]
173 Ops.INDEX dtypes.f32 [156, 17]
174 Ops.INDEX dtypes.f32 [168, 17]
175 Ops.INDEX dtypes.f32 [156, 75]
176 Ops.INDEX dtypes.f32 [168, 75]
177 Ops.STACK dtypes.f32 [173, 174, 175, 176]
178 Ops.INDEX dtypes.f32 [148, 17]
179 Ops.INDEX dtypes.f32 [152, 17]
180 Ops.INDEX dtypes.f32 [148, 75]
181 Ops.INDEX dtypes.f32 [152, 75]
182 Ops.STACK dtypes.f32 [178, 179, 180, 181]
183 Ops.INDEX dtypes.f32 [182, 9]
184 Ops.INDEX dtypes.f32 [160, 17]
185 Ops.INDEX dtypes.f32 [164, 17]
186 Ops.INDEX dtypes.f32 [160, 75]
187 Ops.INDEX dtypes.f32 [164, 75]
188 Ops.STACK dtypes.f32 [184, 185, 186, 187]
189 Ops.INDEX dtypes.f32 [188, 9]
190 Ops.INDEX dtypes.f32 [182, 47]
191 Ops.INDEX dtypes.f32 [188, 47]
192 Ops.STACK dtypes.f32 [183, 189, 190, 191]
193 Ops.INDEX dtypes.f32 [182, 17]
194 Ops.INDEX dtypes.f32 [188, 17]
195 Ops.INDEX dtypes.f32 [182, 75]
196 Ops.INDEX dtypes.f32 [188, 75]
197 Ops.STACK dtypes.f32 [193, 194, 195, 196]
198 Ops.MUL dtypes.f32 [197, 71]
199 Ops.MULACC dtypes.f32 [192, 71, 198]
200 Ops.MULACC dtypes.f32 [177, 71, 199]
201 Ops.MULACC dtypes.f32 [172, 71, 200]
202 Ops.ADD dtypes.f32 [140, 201]
203 Ops.STORE dtypes.void [139, 202]
204 Ops.AFTER dtypes.void [0, 203]
205 Ops.END dtypes.void [204, 137]
206 Ops.AFTER dtypes.f32 [132, 205]
207 Ops.SHRINK dtypes.f32 [206, 17, 12]
208 Ops.LOAD dtypes.f32 [207]
209 Ops.STORE dtypes.void [131, 208]
210 Ops.AFTER dtypes.void [0, 209]
211 Ops.END dtypes.void [210, 128]
212 Ops.SINK dtypes.void [211] KernelInfo(name="r_16_4_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), estimates=Estimates(ops=8192, lds=16640, mem=16640), optimized=True, split=(16, 2, 3))
213 Ops.LINEAR dtypes.void [114, 115, 11, 132, 12, 120, 121, 122, 123, 124, 125, 126, 16, 17, 116, 117, 118, 74, 75, 141, 142, 1, 119, 30, 143, 0, 19, 20, 21, 70, 71, 40, 41, 23, 24, 57, 58, 8, 9, 46, 47, 52, 33, 34, 127, 128, 136, 129, 133, 134, 145, 135, 137, 144, 146, 150, 151, 152, 162, 163, 164, 158, 159, 160, 147, 148, 138, 139, 140, 185, 187, 184, 186, 188, 179, 181, 178, 180, 182, 194, 196, 193, 195, 197, 189, 191, 183, 190, 192, 165, 167, 161, 166, 168, 153, 155, 149, 154, 156, 174, 176, 173, 175, 177, 169, 171, 157, 170, 172, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 130, 131, 209, 210, 211, 212]
214 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_16_4_16_4(unsigned char* restrict args_32) {\n  float buf0[4];\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    *((float4*)((buf0+0))) = (float4){0.0f,0.0f,0.0f,0.0f};\n    for (int Lidx0 = 0; Lidx0 < 16; Lidx0++) {\n      int alu2 = ((Lidx0<<2)+(alu0<<8));\n      float4 val4 = (*((float4*)((((float*)(val3))+(alu2+128)))));\n      float4 val5 = (*((float4*)((((float*)(val3))+(alu2+192)))));\n      float4 val6 = (*((float4*)((((float*)(val3))+(alu2+64)))));\n      float4 val7 = (*((float4*)((((float*)(val3))+alu2))));\n      float4 cast0 = ((float4)__builtin_shufflevector(val6, val5, 0, 4, 1, 5));\n      float4 cast1 = ((float4)__builtin_shufflevector(val7, val4, 0, 4, 1, 5));\n      float4 cast2 = ((float4)__builtin_shufflevector(val6, val5, 2, 6, 3, 7));\n      float4 cast3 = ((float4)__builtin_shufflevector(val7, val4, 2, 6, 3, 7));\n      *((float4*)((buf0+0))) = ((*((float4*)((buf0+0))))+__builtin_elementwise_fma(((float4)__builtin_shufflevector(cast3, cast2, 2, 6, 3, 7)), ((float4)(2.0f)), __builtin_elementwise_fma(((float4)__builtin_shufflevector(cast3, cast2, 0, 4, 1, 5)), ((float4)(2.0f)), __builtin_elementwise_fma(((float4)__builtin_shufflevector(cast1, cast0, 2, 6, 3, 7)), ((float4)(2.0f)), (((float4)__builtin_shufflevector(cast1, cast0, 0, 4, 1, 5))*2.0f)))));\n    }\n    *((float4*)((((float*)(val2))+(alu0<<2)))) = (*((float4*)((buf0+0))));\n  }\n}\n"
215 Ops.PROGRAM dtypes.void [212, 213, 214] ProgramInfo(global_size=(16, 1, 1), params=(%2, %31, %3, %4), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
216 Ops.SINK dtypes.void [113, 215]

