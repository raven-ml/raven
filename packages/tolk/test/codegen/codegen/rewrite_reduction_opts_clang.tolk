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
39 Ops.CONST dtypes.weakint [] 3
40 Ops.CAST dtypes.i32 [39] dtypes.i32
41 Ops.INDEX dtypes.f32 [38, 40]
42 Ops.CAST dtypes.i32 [1] dtypes.i32
43 Ops.ADD dtypes.i32 [36, 42]
44 Ops.SHRINK dtypes.f32 [31, 43, 12]
45 Ops.LOAD dtypes.f32 [44]
46 Ops.INDEX dtypes.f32 [45, 40]
47 Ops.CONST dtypes.weakint [] 128
48 Ops.CAST dtypes.i32 [47] dtypes.i32
49 Ops.ADD dtypes.i32 [36, 48]
50 Ops.SHRINK dtypes.f32 [31, 49, 12]
51 Ops.LOAD dtypes.f32 [50]
52 Ops.INDEX dtypes.f32 [51, 40]
53 Ops.CONST dtypes.weakint [] 192
54 Ops.CAST dtypes.i32 [53] dtypes.i32
55 Ops.ADD dtypes.i32 [36, 54]
56 Ops.SHRINK dtypes.f32 [31, 55, 12]
57 Ops.LOAD dtypes.f32 [56]
58 Ops.INDEX dtypes.f32 [57, 40]
59 Ops.STACK dtypes.f32 [41, 46, 52, 58]
60 Ops.CONST dtypes.weakfloat [] 2.0
61 Ops.CAST dtypes.f32 [60] dtypes.f32
62 Ops.INDEX dtypes.f32 [38, 9]
63 Ops.INDEX dtypes.f32 [45, 9]
64 Ops.INDEX dtypes.f32 [51, 9]
65 Ops.INDEX dtypes.f32 [57, 9]
66 Ops.STACK dtypes.f32 [62, 63, 64, 65]
67 Ops.CONST dtypes.weakint [] 1
68 Ops.CAST dtypes.i32 [67] dtypes.i32
69 Ops.INDEX dtypes.f32 [38, 68]
70 Ops.INDEX dtypes.f32 [45, 68]
71 Ops.INDEX dtypes.f32 [51, 68]
72 Ops.INDEX dtypes.f32 [57, 68]
73 Ops.STACK dtypes.f32 [69, 70, 71, 72]
74 Ops.INDEX dtypes.f32 [38, 17]
75 Ops.INDEX dtypes.f32 [45, 17]
76 Ops.INDEX dtypes.f32 [51, 17]
77 Ops.INDEX dtypes.f32 [57, 17]
78 Ops.STACK dtypes.f32 [74, 75, 76, 77]
79 Ops.MUL dtypes.f32 [78, 61]
80 Ops.MULACC dtypes.f32 [73, 61, 79]
81 Ops.MULACC dtypes.f32 [66, 61, 80]
82 Ops.MULACC dtypes.f32 [59, 61, 81]
83 Ops.ADD dtypes.f32 [29, 82]
84 Ops.STORE dtypes.void [28, 83]
85 Ops.AFTER dtypes.void [0, 84]
86 Ops.END dtypes.void [85, 26]
87 Ops.AFTER dtypes.f32 [14, 86]
88 Ops.SHRINK dtypes.f32 [87, 17, 12]
89 Ops.LOAD dtypes.f32 [88]
90 Ops.STORE dtypes.void [13, 89]
91 Ops.AFTER dtypes.void [0, 90]
92 Ops.END dtypes.void [91, 6]
93 Ops.SINK dtypes.void [92] KernelInfo(name="r_16_4_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), optimized=True, split=(16, 2, 3))
94 Ops.CONST dtypes.weakint [] 32
95 Ops.PARAM dtypes.u8 [94] ParamArg(slot=0, dtype=dtypes.u8, name="args")
96 Ops.BITCAST dtypes.u64 [95] dtypes.u64
97 Ops.INDEX dtypes.u64 [96, 17]
98 Ops.LOAD dtypes.u64 [97]
99 Ops.REFPARAM dtypes.f32 [1, 98] ParamArg(slot=0, dtype=dtypes.f32)
100 Ops.BITCAST dtypes.i32 [95] dtypes.i32
101 Ops.INDEX dtypes.i32 [100, 12]
102 Ops.LOAD dtypes.i32 [101]
103 Ops.CONST dtypes.weakint [] 5
104 Ops.CAST dtypes.i32 [103] dtypes.i32
105 Ops.INDEX dtypes.i32 [100, 104]
106 Ops.LOAD dtypes.i32 [105]
107 Ops.SUB dtypes.i32 [106, 102]
108 Ops.RANGE dtypes.i32 [107] (AxisType.WEAK, 1)
109 Ops.ADD dtypes.i32 [102, 108]
110 Ops.SHL dtypes.i32 [109, 9]
111 Ops.SHRINK dtypes.f32 [99, 110, 12]
112 Ops.BUFFER dtypes.f32 [11] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
113 Ops.AFTER dtypes.f32 [112, 109]
114 Ops.SHRINK dtypes.f32 [113, 17, 12]
115 Ops.STORE dtypes.void [114, 21]
116 Ops.AFTER dtypes.i32 [24, 108]
117 Ops.RANGE dtypes.i32 [116] (AxisType.WEAK, 0)
118 Ops.AFTER dtypes.f32 [112, 115, 117]
119 Ops.SHRINK dtypes.f32 [118, 17, 12]
120 Ops.LOAD dtypes.f32 [119]
121 Ops.INDEX dtypes.u64 [96, 68]
122 Ops.LOAD dtypes.u64 [121]
123 Ops.REFPARAM dtypes.f32 [30, 122] ParamArg(slot=1, dtype=dtypes.f32)
124 Ops.SHL dtypes.i32 [117, 9]
125 Ops.SHL dtypes.i32 [109, 34]
126 Ops.ADD dtypes.i32 [124, 125]
127 Ops.SHRINK dtypes.f32 [123, 126, 12]
128 Ops.LOAD dtypes.f32 [127]
129 Ops.INDEX dtypes.f32 [128, 40]
130 Ops.ADD dtypes.i32 [126, 42]
131 Ops.SHRINK dtypes.f32 [123, 130, 12]
132 Ops.LOAD dtypes.f32 [131]
133 Ops.INDEX dtypes.f32 [132, 40]
134 Ops.ADD dtypes.i32 [126, 48]
135 Ops.SHRINK dtypes.f32 [123, 134, 12]
136 Ops.LOAD dtypes.f32 [135]
137 Ops.INDEX dtypes.f32 [136, 40]
138 Ops.ADD dtypes.i32 [126, 54]
139 Ops.SHRINK dtypes.f32 [123, 138, 12]
140 Ops.LOAD dtypes.f32 [139]
141 Ops.INDEX dtypes.f32 [140, 40]
142 Ops.STACK dtypes.f32 [129, 133, 137, 141]
143 Ops.INDEX dtypes.f32 [128, 9]
144 Ops.INDEX dtypes.f32 [132, 9]
145 Ops.INDEX dtypes.f32 [136, 9]
146 Ops.INDEX dtypes.f32 [140, 9]
147 Ops.STACK dtypes.f32 [143, 144, 145, 146]
148 Ops.INDEX dtypes.f32 [128, 68]
149 Ops.INDEX dtypes.f32 [132, 68]
150 Ops.INDEX dtypes.f32 [136, 68]
151 Ops.INDEX dtypes.f32 [140, 68]
152 Ops.STACK dtypes.f32 [148, 149, 150, 151]
153 Ops.INDEX dtypes.f32 [128, 17]
154 Ops.INDEX dtypes.f32 [132, 17]
155 Ops.INDEX dtypes.f32 [136, 17]
156 Ops.INDEX dtypes.f32 [140, 17]
157 Ops.STACK dtypes.f32 [153, 154, 155, 156]
158 Ops.MUL dtypes.f32 [157, 61]
159 Ops.MULACC dtypes.f32 [152, 61, 158]
160 Ops.MULACC dtypes.f32 [147, 61, 159]
161 Ops.MULACC dtypes.f32 [142, 61, 160]
162 Ops.ADD dtypes.f32 [120, 161]
163 Ops.STORE dtypes.void [119, 162]
164 Ops.AFTER dtypes.void [0, 163]
165 Ops.END dtypes.void [164, 117]
166 Ops.AFTER dtypes.f32 [112, 165]
167 Ops.SHRINK dtypes.f32 [166, 17, 12]
168 Ops.LOAD dtypes.f32 [167]
169 Ops.STORE dtypes.void [111, 168]
170 Ops.AFTER dtypes.void [0, 169]
171 Ops.END dtypes.void [170, 108]
172 Ops.SINK dtypes.void [171] KernelInfo(name="r_16_4_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), estimates=Estimates(ops=8192, lds=16640, mem=16640), optimized=True, split=(16, 2, 3))
173 Ops.LINEAR dtypes.void [94, 95, 11, 112, 12, 100, 101, 102, 103, 104, 105, 106, 16, 17, 96, 97, 98, 67, 68, 121, 122, 1, 99, 30, 123, 0, 19, 20, 21, 60, 61, 47, 48, 23, 24, 53, 54, 8, 9, 39, 40, 42, 33, 34, 107, 108, 116, 109, 113, 114, 125, 115, 117, 124, 126, 134, 135, 136, 138, 139, 140, 130, 131, 132, 127, 128, 118, 119, 120, 155, 156, 154, 153, 157, 150, 151, 149, 148, 152, 145, 146, 144, 143, 147, 137, 141, 133, 129, 142, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 110, 111, 169, 170, 171, 172]
174 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_16_4_16_4(unsigned char* restrict args_32) {\n  float buf0[4];\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    *((float4*)((buf0+0))) = (float4){0.0f,0.0f,0.0f,0.0f};\n    for (int Lidx0 = 0; Lidx0 < 16; Lidx0++) {\n      int alu2 = ((Lidx0<<2)+(alu0<<8));\n      float4 val4 = (*((float4*)((((float*)(val3))+(alu2+128)))));\n      float4 val5 = (*((float4*)((((float*)(val3))+(alu2+192)))));\n      float4 val6 = (*((float4*)((((float*)(val3))+(alu2+64)))));\n      float4 val7 = (*((float4*)((((float*)(val3))+alu2))));\n      *((float4*)((buf0+0))) = ((*((float4*)((buf0+0))))+__builtin_elementwise_fma((float4){val7[3],val6[3],val4[3],val5[3]}, ((float4)(2.0f)), __builtin_elementwise_fma((float4){val7[2],val6[2],val4[2],val5[2]}, ((float4)(2.0f)), __builtin_elementwise_fma((float4){val7[1],val6[1],val4[1],val5[1]}, ((float4)(2.0f)), ((float4){val7[0],val6[0],val4[0],val5[0]}*2.0f)))));\n    }\n    *((float4*)((((float*)(val2))+(alu0<<2)))) = (*((float4*)((buf0+0))));\n  }\n}\n"
175 Ops.PROGRAM dtypes.void [172, 173, 174] ProgramInfo(global_size=(16, 1, 1), params=(%2, %31, %3, %4), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
176 Ops.SINK dtypes.void [93, 175]

