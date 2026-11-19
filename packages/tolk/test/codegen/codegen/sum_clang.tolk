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
40 Ops.CAST dtypes.i32 [1] dtypes.i32
41 Ops.ADD dtypes.i32 [36, 40]
42 Ops.SHRINK dtypes.f32 [31, 41, 12]
43 Ops.LOAD dtypes.f32 [42]
44 Ops.INDEX dtypes.f32 [43, 17]
45 Ops.CONST dtypes.weakint [] 128
46 Ops.CAST dtypes.i32 [45] dtypes.i32
47 Ops.ADD dtypes.i32 [36, 46]
48 Ops.SHRINK dtypes.f32 [31, 47, 12]
49 Ops.LOAD dtypes.f32 [48]
50 Ops.INDEX dtypes.f32 [49, 17]
51 Ops.CONST dtypes.weakint [] 192
52 Ops.CAST dtypes.i32 [51] dtypes.i32
53 Ops.ADD dtypes.i32 [36, 52]
54 Ops.SHRINK dtypes.f32 [31, 53, 12]
55 Ops.LOAD dtypes.f32 [54]
56 Ops.INDEX dtypes.f32 [55, 17]
57 Ops.STACK dtypes.f32 [39, 44, 50, 56]
58 Ops.CONST dtypes.weakint [] 1
59 Ops.CAST dtypes.i32 [58] dtypes.i32
60 Ops.INDEX dtypes.f32 [38, 59]
61 Ops.INDEX dtypes.f32 [43, 59]
62 Ops.INDEX dtypes.f32 [49, 59]
63 Ops.INDEX dtypes.f32 [55, 59]
64 Ops.STACK dtypes.f32 [60, 61, 62, 63]
65 Ops.ADD dtypes.f32 [57, 64]
66 Ops.INDEX dtypes.f32 [38, 9]
67 Ops.INDEX dtypes.f32 [43, 9]
68 Ops.INDEX dtypes.f32 [49, 9]
69 Ops.INDEX dtypes.f32 [55, 9]
70 Ops.STACK dtypes.f32 [66, 67, 68, 69]
71 Ops.ADD dtypes.f32 [65, 70]
72 Ops.CONST dtypes.weakint [] 3
73 Ops.CAST dtypes.i32 [72] dtypes.i32
74 Ops.INDEX dtypes.f32 [38, 73]
75 Ops.INDEX dtypes.f32 [43, 73]
76 Ops.INDEX dtypes.f32 [49, 73]
77 Ops.INDEX dtypes.f32 [55, 73]
78 Ops.STACK dtypes.f32 [74, 75, 76, 77]
79 Ops.ADD dtypes.f32 [71, 78]
80 Ops.ADD dtypes.f32 [29, 79]
81 Ops.STORE dtypes.void [28, 80]
82 Ops.AFTER dtypes.void [0, 81]
83 Ops.END dtypes.void [82, 26]
84 Ops.AFTER dtypes.f32 [14, 83]
85 Ops.SHRINK dtypes.f32 [84, 17, 12]
86 Ops.LOAD dtypes.f32 [85]
87 Ops.STORE dtypes.void [13, 86]
88 Ops.AFTER dtypes.void [0, 87]
89 Ops.END dtypes.void [88, 6]
90 Ops.SINK dtypes.void [89] KernelInfo(name="r_16_4_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), optimized=True, split=(16, 2, 3))
91 Ops.CONST dtypes.weakint [] 32
92 Ops.PARAM dtypes.u8 [91] ParamArg(slot=0, dtype=dtypes.u8, name="args")
93 Ops.BITCAST dtypes.u64 [92] dtypes.u64
94 Ops.INDEX dtypes.u64 [93, 17]
95 Ops.LOAD dtypes.u64 [94]
96 Ops.REFPARAM dtypes.f32 [1, 95] ParamArg(slot=0, dtype=dtypes.f32)
97 Ops.BITCAST dtypes.i32 [92] dtypes.i32
98 Ops.INDEX dtypes.i32 [97, 12]
99 Ops.LOAD dtypes.i32 [98]
100 Ops.CONST dtypes.weakint [] 5
101 Ops.CAST dtypes.i32 [100] dtypes.i32
102 Ops.INDEX dtypes.i32 [97, 101]
103 Ops.LOAD dtypes.i32 [102]
104 Ops.SUB dtypes.i32 [103, 99]
105 Ops.RANGE dtypes.i32 [104] (AxisType.WEAK, 1)
106 Ops.ADD dtypes.i32 [99, 105]
107 Ops.SHL dtypes.i32 [106, 9]
108 Ops.SHRINK dtypes.f32 [96, 107, 12]
109 Ops.BUFFER dtypes.f32 [11] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
110 Ops.AFTER dtypes.f32 [109, 106]
111 Ops.SHRINK dtypes.f32 [110, 17, 12]
112 Ops.STORE dtypes.void [111, 21]
113 Ops.AFTER dtypes.i32 [24, 105]
114 Ops.RANGE dtypes.i32 [113] (AxisType.WEAK, 0)
115 Ops.AFTER dtypes.f32 [109, 112, 114]
116 Ops.SHRINK dtypes.f32 [115, 17, 12]
117 Ops.LOAD dtypes.f32 [116]
118 Ops.INDEX dtypes.u64 [93, 59]
119 Ops.LOAD dtypes.u64 [118]
120 Ops.REFPARAM dtypes.f32 [30, 119] ParamArg(slot=1, dtype=dtypes.f32)
121 Ops.SHL dtypes.i32 [114, 9]
122 Ops.SHL dtypes.i32 [106, 34]
123 Ops.ADD dtypes.i32 [121, 122]
124 Ops.SHRINK dtypes.f32 [120, 123, 12]
125 Ops.LOAD dtypes.f32 [124]
126 Ops.INDEX dtypes.f32 [125, 17]
127 Ops.ADD dtypes.i32 [123, 40]
128 Ops.SHRINK dtypes.f32 [120, 127, 12]
129 Ops.LOAD dtypes.f32 [128]
130 Ops.INDEX dtypes.f32 [129, 17]
131 Ops.ADD dtypes.i32 [123, 46]
132 Ops.SHRINK dtypes.f32 [120, 131, 12]
133 Ops.LOAD dtypes.f32 [132]
134 Ops.INDEX dtypes.f32 [133, 17]
135 Ops.ADD dtypes.i32 [123, 52]
136 Ops.SHRINK dtypes.f32 [120, 135, 12]
137 Ops.LOAD dtypes.f32 [136]
138 Ops.INDEX dtypes.f32 [137, 17]
139 Ops.STACK dtypes.f32 [126, 130, 134, 138]
140 Ops.INDEX dtypes.f32 [125, 59]
141 Ops.INDEX dtypes.f32 [129, 59]
142 Ops.INDEX dtypes.f32 [133, 59]
143 Ops.INDEX dtypes.f32 [137, 59]
144 Ops.STACK dtypes.f32 [140, 141, 142, 143]
145 Ops.ADD dtypes.f32 [139, 144]
146 Ops.INDEX dtypes.f32 [125, 9]
147 Ops.INDEX dtypes.f32 [129, 9]
148 Ops.INDEX dtypes.f32 [133, 9]
149 Ops.INDEX dtypes.f32 [137, 9]
150 Ops.STACK dtypes.f32 [146, 147, 148, 149]
151 Ops.ADD dtypes.f32 [145, 150]
152 Ops.INDEX dtypes.f32 [125, 73]
153 Ops.INDEX dtypes.f32 [129, 73]
154 Ops.INDEX dtypes.f32 [133, 73]
155 Ops.INDEX dtypes.f32 [137, 73]
156 Ops.STACK dtypes.f32 [152, 153, 154, 155]
157 Ops.ADD dtypes.f32 [151, 156]
158 Ops.ADD dtypes.f32 [117, 157]
159 Ops.STORE dtypes.void [116, 158]
160 Ops.AFTER dtypes.void [0, 159]
161 Ops.END dtypes.void [160, 114]
162 Ops.AFTER dtypes.f32 [109, 161]
163 Ops.SHRINK dtypes.f32 [162, 17, 12]
164 Ops.LOAD dtypes.f32 [163]
165 Ops.STORE dtypes.void [108, 164]
166 Ops.AFTER dtypes.void [0, 165]
167 Ops.END dtypes.void [166, 105]
168 Ops.SINK dtypes.void [167] KernelInfo(name="r_16_4_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), estimates=Estimates(ops=4096, lds=16640, mem=16640), optimized=True, split=(16, 2, 3))
169 Ops.LINEAR dtypes.void [91, 92, 11, 109, 12, 97, 98, 99, 100, 101, 102, 103, 16, 17, 93, 94, 95, 58, 59, 118, 119, 1, 96, 30, 120, 0, 19, 20, 21, 45, 46, 23, 24, 51, 52, 8, 9, 72, 73, 40, 33, 34, 104, 105, 113, 106, 110, 111, 122, 112, 114, 121, 123, 131, 132, 133, 135, 136, 137, 127, 128, 129, 124, 125, 115, 116, 117, 134, 138, 130, 126, 139, 142, 143, 141, 140, 144, 148, 149, 147, 146, 150, 154, 155, 153, 152, 156, 145, 151, 157, 158, 159, 160, 161, 162, 163, 164, 107, 108, 165, 166, 167, 168]
170 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_16_4_16_4(unsigned char* restrict args_32) {\n  float buf0[4];\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    *((float4*)((buf0+0))) = (float4){0.0f,0.0f,0.0f,0.0f};\n    for (int Lidx0 = 0; Lidx0 < 16; Lidx0++) {\n      int alu2 = ((Lidx0<<2)+(alu0<<8));\n      float4 val4 = (*((float4*)((((float*)(val3))+(alu2+128)))));\n      float4 val5 = (*((float4*)((((float*)(val3))+(alu2+192)))));\n      float4 val6 = (*((float4*)((((float*)(val3))+(alu2+64)))));\n      float4 val7 = (*((float4*)((((float*)(val3))+alu2))));\n      *((float4*)((buf0+0))) = ((*((float4*)((buf0+0))))+((float4){val7[0],val6[0],val4[0],val5[0]}+(float4){val7[1],val6[1],val4[1],val5[1]}+(float4){val7[2],val6[2],val4[2],val5[2]}+(float4){val7[3],val6[3],val4[3],val5[3]}));\n    }\n    *((float4*)((((float*)(val2))+(alu0<<2)))) = (*((float4*)((buf0+0))));\n  }\n}\n"
171 Ops.PROGRAM dtypes.void [168, 169, 170] ProgramInfo(global_size=(16, 1, 1), params=(%2, %31, %3, %4), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
172 Ops.SINK dtypes.void [90, 171]

