# tolk D25 D80 D141 D151 golden=71454be753b90581ed35c41c42bf9943
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 15
2 Ops.CONST dtypes.weakint [] 48
3 Ops.PARAM dtypes.u8 [2] ParamArg(slot=0, dtype=dtypes.u8, name="args")
4 Ops.BITCAST dtypes.u64 [3] dtypes.u64
5 Ops.CONST dtypes.weakint [] 0
6 Ops.CAST dtypes.i32 [5] dtypes.i32
7 Ops.INDEX dtypes.u64 [4, 6]
8 Ops.LOAD dtypes.u64 [7]
9 Ops.REFPARAM dtypes.f32 [1, 8] ParamArg(slot=0, dtype=dtypes.f32)
10 Ops.BITCAST dtypes.i32 [3] dtypes.i32
11 Ops.CONST dtypes.weakint [] 6
12 Ops.CAST dtypes.i32 [11] dtypes.i32
13 Ops.INDEX dtypes.i32 [10, 12]
14 Ops.LOAD dtypes.i32 [13]
15 Ops.CONST dtypes.weakint [] 7
16 Ops.CAST dtypes.i32 [15] dtypes.i32
17 Ops.INDEX dtypes.i32 [10, 16]
18 Ops.LOAD dtypes.i32 [17]
19 Ops.SUB dtypes.i32 [18, 14]
20 Ops.RANGE dtypes.i32 [19] (AxisType.WEAK, 2)
21 Ops.ADD dtypes.i32 [14, 20]
22 Ops.INDEX dtypes.f32 [9, 21]
23 Ops.CONST dtypes.weakint [] 3
24 Ops.BUFFER dtypes.f32 [23] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
25 Ops.AFTER dtypes.f32 [24, 21]
26 Ops.CONST dtypes.weakint [] 2
27 Ops.CAST dtypes.i32 [26] dtypes.i32
28 Ops.SHRINK dtypes.f32 [25, 6, 27]
29 Ops.CONST dtypes.weakfloat [] 0.0
30 Ops.CAST dtypes.f32 [29] dtypes.f32
31 Ops.STACK dtypes.f32 [30, 30]
32 Ops.STORE dtypes.void [28, 31]
33 Ops.INDEX dtypes.f32 [25, 27]
34 Ops.STORE dtypes.void [33, 30]
35 Ops.CONST dtypes.weakint [] 8
36 Ops.CAST dtypes.i32 [35] dtypes.i32
37 Ops.INDEX dtypes.i32 [10, 36]
38 Ops.LOAD dtypes.i32 [37]
39 Ops.AFTER dtypes.i32 [38, 20]
40 Ops.RANGE dtypes.i32 [39] (AxisType.WEAK, 0)
41 Ops.AFTER dtypes.f32 [24, 32, 34, 40]
42 Ops.SHRINK dtypes.f32 [41, 6, 27]
43 Ops.CONST dtypes.weakint [] 30
44 Ops.CONST dtypes.weakint [] 1
45 Ops.CAST dtypes.i32 [44] dtypes.i32
46 Ops.INDEX dtypes.u64 [4, 45]
47 Ops.LOAD dtypes.u64 [46]
48 Ops.REFPARAM dtypes.f32 [43, 47] ParamArg(slot=1, dtype=dtypes.f32)
49 Ops.INDEX dtypes.f32 [48, 40]
50 Ops.LOAD dtypes.f32 [49]
51 Ops.CONST dtypes.weakint [] 10
52 Ops.CAST dtypes.i32 [51] dtypes.i32
53 Ops.ADD dtypes.i32 [40, 52]
54 Ops.INDEX dtypes.f32 [48, 53]
55 Ops.LOAD dtypes.f32 [54]
56 Ops.CONST dtypes.weakint [] 20
57 Ops.CAST dtypes.i32 [56] dtypes.i32
58 Ops.ADD dtypes.i32 [40, 57]
59 Ops.INDEX dtypes.f32 [48, 58]
60 Ops.LOAD dtypes.f32 [59]
61 Ops.STACK dtypes.f32 [50, 55, 60]
62 Ops.CONST dtypes.weakint [] 50
63 Ops.INDEX dtypes.u64 [4, 27]
64 Ops.LOAD dtypes.u64 [63]
65 Ops.REFPARAM dtypes.f32 [62, 64] ParamArg(slot=2, dtype=dtypes.f32)
66 Ops.CONST dtypes.weakint [] 5
67 Ops.CAST dtypes.i32 [66] dtypes.i32
68 Ops.MUL dtypes.i32 [40, 67]
69 Ops.ADD dtypes.i32 [68, 21]
70 Ops.INDEX dtypes.f32 [65, 69]
71 Ops.LOAD dtypes.f32 [70]
72 Ops.STACK dtypes.f32 [71, 71, 71]
73 Ops.LOAD dtypes.f32 [42]
74 Ops.INDEX dtypes.f32 [73, 6]
75 Ops.INDEX dtypes.f32 [73, 45]
76 Ops.INDEX dtypes.f32 [41, 27]
77 Ops.LOAD dtypes.f32 [76]
78 Ops.STACK dtypes.f32 [74, 75, 77]
79 Ops.MULACC dtypes.f32 [61, 72, 78]
80 Ops.INDEX dtypes.f32 [79, 6]
81 Ops.INDEX dtypes.f32 [79, 45]
82 Ops.STACK dtypes.f32 [80, 81]
83 Ops.STORE dtypes.void [42, 82]
84 Ops.INDEX dtypes.f32 [79, 27]
85 Ops.STORE dtypes.void [76, 84]
86 Ops.AFTER dtypes.void [0, 83, 85]
87 Ops.END dtypes.void [86, 40]
88 Ops.AFTER dtypes.f32 [24, 87]
89 Ops.SHRINK dtypes.f32 [88, 6, 27]
90 Ops.LOAD dtypes.f32 [89]
91 Ops.INDEX dtypes.f32 [90, 6]
92 Ops.STORE dtypes.void [22, 91]
93 Ops.ADD dtypes.i32 [21, 67]
94 Ops.INDEX dtypes.f32 [9, 93]
95 Ops.INDEX dtypes.f32 [90, 45]
96 Ops.STORE dtypes.void [94, 95]
97 Ops.ADD dtypes.i32 [21, 52]
98 Ops.INDEX dtypes.f32 [9, 97]
99 Ops.INDEX dtypes.f32 [88, 27]
100 Ops.LOAD dtypes.f32 [99]
101 Ops.STORE dtypes.void [98, 100]
102 Ops.AFTER dtypes.void [0, 92, 96, 101]
103 Ops.END dtypes.void [102, 20]
104 Ops.PARAM dtypes.i32 [0] ParamArg(slot=5, dtype=dtypes.i32, vmin_vmax=(1, 10), name="i", addrspace=AddrSpace.ALU)
105 Ops.MUL dtypes.i32 [104, 43]
106 Ops.CONST dtypes.weakint [] 80
107 Ops.MUL dtypes.i32 [104, 106]
108 Ops.CONST dtypes.weakint [] 60
109 Ops.ADD dtypes.i32 [107, 108]
110 Ops.MUL dtypes.i32 [104, 56]
111 Ops.CONST dtypes.weakint [] -1
112 Ops.XOR dtypes.i32 [110, 111]
113 Ops.CONST dtypes.weakint [] -121
114 Ops.MAX dtypes.i32 [112, 113]
115 Ops.XOR dtypes.i32 [114, 111]
116 Ops.ADD dtypes.i32 [115, 110]
117 Ops.XOR dtypes.i32 [116, 111]
118 Ops.MAX dtypes.i32 [117, 113]
119 Ops.XOR dtypes.i32 [118, 111]
120 Ops.ADD dtypes.i32 [119, 110]
121 Ops.XOR dtypes.i32 [120, 111]
122 Ops.MAX dtypes.i32 [121, 113]
123 Ops.XOR dtypes.i32 [122, 111]
124 Ops.ADD dtypes.i32 [123, 110]
125 Ops.ADD dtypes.i32 [124, 108]
126 Ops.SINK dtypes.void [103] KernelInfo(name="r_5_3_i", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(3, AxisType.UPCAST)),), estimates=Estimates(ops=%105, lds=%109, mem=%125), optimized=True, split=(5, 3, 4))
127 Ops.LINEAR dtypes.void [2, 3, 23, 24, 11, 12, 10, 13, 14, 15, 16, 17, 18, 35, 36, 37, 38, 5, 6, 4, 7, 8, 44, 45, 46, 47, 26, 27, 63, 64, 1, 9, 43, 48, 62, 65, 0, 29, 30, 31, 51, 52, 56, 57, 66, 67, 19, 20, 39, 21, 25, 33, 28, 34, 32, 40, 53, 54, 55, 58, 59, 60, 49, 50, 68, 69, 70, 71, 41, 76, 77, 42, 73, 74, 75, 78, 61, 72, 79, 80, 81, 82, 84, 85, 83, 86, 87, 88, 99, 100, 89, 90, 22, 97, 98, 93, 94, 91, 95, 92, 101, 96, 102, 103, 126]
128 Ops.SOURCE dtypes.void [] "typedef float float2 __attribute__((aligned(4),ext_vector_type(2)));\ntypedef float float3 __attribute__((aligned(4),ext_vector_type(3)));\n__attribute__((min_vector_width(512))) void r_5_3_i(unsigned char* restrict args_48) {\n  float buf0[3];\n  int val0 = (*(((int*)(args_48))+6));\n  int val1 = (*(((int*)(args_48))+7));\n  int val2 = (*(((int*)(args_48))+8));\n  unsigned long val3 = (*(((unsigned long*)(args_48))+0));\n  unsigned long val4 = (*(((unsigned long*)(args_48))+1));\n  unsigned long val5 = (*(((unsigned long*)(args_48))+2));\n  for (int Lidx2 = 0; Lidx2 < (val1-val0); Lidx2++) {\n    int alu0 = (val0+Lidx2);\n    *(buf0+2) = 0.0f;\n    *((float2*)((buf0+0))) = (float2){0.0f,0.0f};\n    for (int Lidx0 = 0; Lidx0 < val2; Lidx0++) {\n      float val6 = (*(((float*)(val4))+(Lidx0+10)));\n      float val7 = (*(((float*)(val4))+(Lidx0+20)));\n      float val8 = (*(((float*)(val4))+Lidx0));\n      float val9 = (*(((float*)(val5))+((Lidx0*5)+alu0)));\n      float2 val10 = (*((float2*)((buf0+0))));\n      float3 alu3 = __builtin_elementwise_fma((float3){val8,val6,val7}, (float3){val9,val9,val9}, (float3){val10[0],val10[1],(*(buf0+2))});\n      *(buf0+2) = alu3[2];\n      *((float2*)((buf0+0))) = ((float2)__builtin_shufflevector(alu3, alu3, 0, 1));\n    }\n    float2 val11 = (*((float2*)((buf0+0))));\n    *(((float*)(val3))+alu0) = val11[0];\n    *(((float*)(val3))+(alu0+10)) = (*(buf0+2));\n    *(((float*)(val3))+(alu0+5)) = val11[1];\n  }\n}\n"
129 Ops.BINARY dtypes.u8 [] b"typedef float float2 __attribute__((aligned(4),ext_vector_type(2)));\ntypedef float float3 __attribute__((aligned(4),ext_vector_type(3)));\n__attribute__((min_vector_width(512))) void r_5_3_i(unsigned char* restrict args_48) {\n  float buf0[3];\n  int val0 = (*(((int*)(args_48))+6));\n  int val1 = (*(((int*)(args_48))+7));\n  int val2 = (*(((int*)(args_48))+8));\n  unsigned long val3 = (*(((unsigned long*)(args_48))+0));\n  unsigned long val4 = (*(((unsigned long*)(args_48))+1));\n  unsigned long val5 = (*(((unsigned long*)(args_48))+2));\n  for (int Lidx2 = 0; Lidx2 < (val1-val0); Lidx2++) {\n    int alu0 = (val0+Lidx2);\n    *(buf0+2) = 0.0f;\n    *((float2*)((buf0+0))) = (float2){0.0f,0.0f};\n    for (int Lidx0 = 0; Lidx0 < val2; Lidx0++) {\n      float val6 = (*(((float*)(val4))+(Lidx0+10)));\n      float val7 = (*(((float*)(val4))+(Lidx0+20)));\n      float val8 = (*(((float*)(val4))+Lidx0));\n      float val9 = (*(((float*)(val5))+((Lidx0*5)+alu0)));\n      float2 val10 = (*((float2*)((buf0+0))));\n      float3 alu3 = __builtin_elementwise_fma((float3){val8,val6,val7}, (float3){val9,val9,val9}, (float3){val10[0],val10[1],(*(buf0+2))});\n      *(buf0+2) = alu3[2];\n      *((float2*)((buf0+0))) = ((float2)__builtin_shufflevector(alu3, alu3, 0, 1));\n    }\n    float2 val11 = (*((float2*)((buf0+0))));\n    *(((float*)(val3))+alu0) = val11[0];\n    *(((float*)(val3))+(alu0+10)) = (*(buf0+2));\n    *(((float*)(val3))+(alu0+5)) = val11[1];\n  }\n}\n"
130 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
131 Ops.PARAM dtypes.f32 [43] ParamArg(slot=1, dtype=dtypes.f32)
132 Ops.PARAM dtypes.f32 [62] ParamArg(slot=2, dtype=dtypes.f32)
133 Ops.PARAM dtypes.i32 [0] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 5), name="block_lo", addrspace=AddrSpace.ALU)
134 Ops.PARAM dtypes.i32 [0] ParamArg(slot=4, dtype=dtypes.i32, vmin_vmax=(0, 5), name="block_hi", addrspace=AddrSpace.ALU)
135 Ops.PROGRAM dtypes.void [126, 127, 128, 129] ProgramInfo(global_size=(5, 1, 1), params=(%130, %131, %132, %133, %134, %104), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CPU", arch="x86_64,x86-64"))
136 Ops.CONST dtypes.weakint [] 256
137 Ops.ALLOC dtypes.i8 [136] ParamArg(slot=7, dtype=dtypes.i8, device="CPU")
138 Ops.SHRINK dtypes.i8 [137, 5, 108]
139 Ops.BITCAST dtypes.f32 [138] dtypes.f32
140 Ops.PARAM dtypes.f32 [43] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
141 Ops.PARAM dtypes.f32 [62] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
142 Ops.CALL dtypes.void [135, 139, 140, 141] CallInfo()
143 Ops.CONST dtypes.weakint [] 32
144 Ops.PARAM dtypes.u8 [143] ParamArg(slot=0, dtype=dtypes.u8, name="args")
145 Ops.BITCAST dtypes.u64 [144] dtypes.u64
146 Ops.INDEX dtypes.u64 [145, 6]
147 Ops.LOAD dtypes.u64 [146]
148 Ops.REFPARAM dtypes.f32 [1, 147] ParamArg(slot=0, dtype=dtypes.f32)
149 Ops.BITCAST dtypes.i32 [144] dtypes.i32
150 Ops.CONST dtypes.weakint [] 4
151 Ops.CAST dtypes.i32 [150] dtypes.i32
152 Ops.INDEX dtypes.i32 [149, 151]
153 Ops.LOAD dtypes.i32 [152]
154 Ops.INDEX dtypes.i32 [149, 67]
155 Ops.LOAD dtypes.i32 [154]
156 Ops.SUB dtypes.i32 [155, 153]
157 Ops.RANGE dtypes.i32 [156] (AxisType.WEAK, 0)
158 Ops.ADD dtypes.i32 [153, 157]
159 Ops.INDEX dtypes.f32 [148, 158]
160 Ops.INDEX dtypes.u64 [145, 45]
161 Ops.LOAD dtypes.u64 [160]
162 Ops.REFPARAM dtypes.f32 [1, 161] ParamArg(slot=1, dtype=dtypes.f32)
163 Ops.INDEX dtypes.f32 [162, 158]
164 Ops.LOAD dtypes.f32 [163]
165 Ops.CONST dtypes.weakfloat [] 2.0
166 Ops.CAST dtypes.f32 [165] dtypes.f32
167 Ops.MUL dtypes.f32 [164, 166]
168 Ops.STORE dtypes.void [159, 167]
169 Ops.END dtypes.void [168, 157]
170 Ops.SINK dtypes.void [169] KernelInfo(name="E_15", estimates=Estimates(ops=15, lds=120, mem=120), optimized=True, split=(15, 2, 3))
171 Ops.LINEAR dtypes.void [143, 144, 150, 151, 149, 152, 153, 66, 67, 154, 155, 5, 6, 145, 146, 147, 44, 45, 160, 161, 1, 148, 162, 165, 166, 156, 157, 158, 163, 164, 159, 167, 168, 169, 170]
172 Ops.SOURCE dtypes.void [] "\n__attribute__((min_vector_width(512))) void E_15(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = (val0+Lidx0);\n    float val4 = (*(((float*)(val3))+alu0));\n    *(((float*)(val2))+alu0) = (val4*2.0f);\n  }\n}\n"
173 Ops.BINARY dtypes.u8 [] b"\n__attribute__((min_vector_width(512))) void E_15(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = (val0+Lidx0);\n    float val4 = (*(((float*)(val3))+alu0));\n    *(((float*)(val2))+alu0) = (val4*2.0f);\n  }\n}\n"
174 Ops.PARAM dtypes.f32 [1] ParamArg(slot=1, dtype=dtypes.f32)
175 Ops.PARAM dtypes.i32 [0] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 15), name="block_lo", addrspace=AddrSpace.ALU)
176 Ops.PARAM dtypes.i32 [0] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 15), name="block_hi", addrspace=AddrSpace.ALU)
177 Ops.PROGRAM dtypes.void [170, 171, 172, 173] ProgramInfo(global_size=(15, 1, 1), params=(%130, %174, %175, %176), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
178 Ops.BUFFER dtypes.f32 [1] ParamArg(slot=5, dtype=dtypes.f32, device="CPU")
179 Ops.CALL dtypes.void [177, 178, 139] CallInfo()
180 Ops.LINEAR dtypes.void [142, 179]

