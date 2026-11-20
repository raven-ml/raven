# tolk D25 D80 D141 D151 D183 golden=15bf2976ef611f2b9aebb1e12b38c3d2
0 Ops.CONST dtypes.weakint [] 80
1 Ops.CONST dtypes.weakint [] 40
2 Ops.PARAM dtypes.u8 [1] ParamArg(slot=0, dtype=dtypes.u8, name="args")
3 Ops.BITCAST dtypes.u64 [2] dtypes.u64
4 Ops.CONST dtypes.weakint [] 0
5 Ops.CAST dtypes.i32 [4] dtypes.i32
6 Ops.INDEX dtypes.u64 [3, 5]
7 Ops.LOAD dtypes.u64 [6]
8 Ops.REFPARAM dtypes.f32 [0, 7] ParamArg(slot=0, dtype=dtypes.f32)
9 Ops.CONST dtypes.weakint [] 4
10 Ops.CAST dtypes.i32 [9] dtypes.i32
11 Ops.CONST dtypes.weakint [] 2
12 Ops.CAST dtypes.i32 [11] dtypes.i32
13 Ops.RANGE dtypes.i32 [12] (AxisType.WEAK, 1)
14 Ops.AFTER dtypes.i32 [10, 13]
15 Ops.RANGE dtypes.i32 [14] (AxisType.WEAK, 2)
16 Ops.CONST dtypes.weakint [] 10
17 Ops.CAST dtypes.i32 [16] dtypes.i32
18 Ops.MUL dtypes.i32 [15, 17]
19 Ops.BITCAST dtypes.i32 [2] dtypes.i32
20 Ops.CONST dtypes.weakint [] 6
21 Ops.CAST dtypes.i32 [20] dtypes.i32
22 Ops.INDEX dtypes.i32 [19, 21]
23 Ops.LOAD dtypes.i32 [22]
24 Ops.CONST dtypes.weakint [] 7
25 Ops.CAST dtypes.i32 [24] dtypes.i32
26 Ops.INDEX dtypes.i32 [19, 25]
27 Ops.LOAD dtypes.i32 [26]
28 Ops.SUB dtypes.i32 [27, 23]
29 Ops.AFTER dtypes.i32 [28, 15]
30 Ops.RANGE dtypes.i32 [29] (AxisType.WEAK, 3)
31 Ops.ADD dtypes.i32 [23, 30]
32 Ops.ADD dtypes.i32 [18, 31]
33 Ops.CAST dtypes.i32 [1] dtypes.i32
34 Ops.MUL dtypes.i32 [13, 33]
35 Ops.ADD dtypes.i32 [32, 34]
36 Ops.INDEX dtypes.f32 [8, 35]
37 Ops.CONST dtypes.weakint [] 64
38 Ops.CONST dtypes.weakint [] 1
39 Ops.CAST dtypes.i32 [38] dtypes.i32
40 Ops.INDEX dtypes.u64 [3, 39]
41 Ops.LOAD dtypes.u64 [40]
42 Ops.REFPARAM dtypes.f32 [37, 41] ParamArg(slot=1, dtype=dtypes.f32)
43 Ops.CONST dtypes.weakint [] 5
44 Ops.CAST dtypes.i32 [43] dtypes.i32
45 Ops.SHL dtypes.i32 [13, 44]
46 Ops.CONST dtypes.weakint [] 3
47 Ops.CAST dtypes.i32 [46] dtypes.i32
48 Ops.SHL dtypes.i32 [15, 47]
49 Ops.ADD dtypes.i32 [45, 48]
50 Ops.ADD dtypes.i32 [49, 10]
51 Ops.SHRINK dtypes.f32 [42, 50, 10]
52 Ops.LOAD dtypes.f32 [51]
53 Ops.INDEX dtypes.f32 [52, 47]
54 Ops.CONST dtypes.weakint [] 640
55 Ops.INDEX dtypes.u64 [3, 12]
56 Ops.LOAD dtypes.u64 [55]
57 Ops.REFPARAM dtypes.f32 [54, 56] ParamArg(slot=2, dtype=dtypes.f32)
58 Ops.SHL dtypes.i32 [31, 44]
59 Ops.ADD dtypes.i32 [48, 58]
60 Ops.CONST dtypes.weakint [] 320
61 Ops.CAST dtypes.i32 [60] dtypes.i32
62 Ops.MUL dtypes.i32 [13, 61]
63 Ops.ADD dtypes.i32 [59, 62]
64 Ops.ADD dtypes.i32 [63, 10]
65 Ops.SHRINK dtypes.f32 [57, 64, 10]
66 Ops.LOAD dtypes.f32 [65]
67 Ops.INDEX dtypes.f32 [66, 47]
68 Ops.INDEX dtypes.f32 [52, 12]
69 Ops.INDEX dtypes.f32 [66, 12]
70 Ops.INDEX dtypes.f32 [52, 39]
71 Ops.INDEX dtypes.f32 [66, 39]
72 Ops.INDEX dtypes.f32 [52, 5]
73 Ops.INDEX dtypes.f32 [66, 5]
74 Ops.SHRINK dtypes.f32 [42, 49, 10]
75 Ops.LOAD dtypes.f32 [74]
76 Ops.INDEX dtypes.f32 [75, 47]
77 Ops.SHRINK dtypes.f32 [57, 63, 10]
78 Ops.LOAD dtypes.f32 [77]
79 Ops.INDEX dtypes.f32 [78, 47]
80 Ops.INDEX dtypes.f32 [75, 12]
81 Ops.INDEX dtypes.f32 [78, 12]
82 Ops.INDEX dtypes.f32 [75, 39]
83 Ops.INDEX dtypes.f32 [78, 39]
84 Ops.INDEX dtypes.f32 [75, 5]
85 Ops.INDEX dtypes.f32 [78, 5]
86 Ops.MUL dtypes.f32 [84, 85]
87 Ops.MULACC dtypes.f32 [82, 83, 86]
88 Ops.MULACC dtypes.f32 [80, 81, 87]
89 Ops.MULACC dtypes.f32 [76, 79, 88]
90 Ops.MULACC dtypes.f32 [72, 73, 89]
91 Ops.MULACC dtypes.f32 [70, 71, 90]
92 Ops.MULACC dtypes.f32 [68, 69, 91]
93 Ops.MULACC dtypes.f32 [53, 67, 92]
94 Ops.CONST dtypes.weakfloat [] 0.3535533845424652
95 Ops.CAST dtypes.f32 [94] dtypes.f32
96 Ops.MUL dtypes.f32 [93, 95]
97 Ops.STORE dtypes.void [36, 96]
98 Ops.END dtypes.void [97, 30]
99 Ops.END dtypes.void [98, 15]
100 Ops.END dtypes.void [99, 13]
101 Ops.STACK dtypes.void []
102 Ops.PARAM dtypes.i32 [101] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(1, 10), name="i", addrspace=AddrSpace.ALU)
103 Ops.SUB dtypes.i32 [102, 4]
104 Ops.CONST dtypes.weakint [] 128
105 Ops.MUL dtypes.i32 [103, 104]
106 Ops.CONST dtypes.weakint [] 256
107 Ops.CONST dtypes.weakint [] 288
108 Ops.MUL dtypes.i32 [103, 107]
109 Ops.ADD dtypes.i32 [106, 108]
110 Ops.SINK dtypes.void [100] KernelInfo(name="r_2_4_i_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=3, arg=(0, AxisType.UPCAST)),), estimates=Estimates(ops=%105, lds=%109, mem=%109), optimized=True, split=(%102, 3, 4))
111 Ops.LINEAR dtypes.void [1, 2, 20, 21, 19, 22, 23, 24, 25, 26, 27, 4, 5, 3, 6, 7, 38, 39, 40, 41, 11, 12, 55, 56, 0, 8, 37, 42, 54, 57, 94, 95, 16, 17, 46, 47, 60, 61, 9, 10, 33, 43, 44, 28, 13, 14, 62, 34, 45, 15, 48, 49, 50, 51, 52, 74, 75, 29, 72, 70, 68, 53, 84, 82, 80, 76, 18, 30, 31, 58, 59, 63, 64, 65, 66, 77, 78, 32, 35, 36, 73, 71, 69, 67, 85, 83, 81, 79, 86, 87, 88, 89, 90, 91, 92, 93, 96, 97, 98, 99, 100, 110]
112 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_2_4_i_8(unsigned char* restrict args_40) {\n  int val0 = (*(((int*)(args_40))+6));\n  int val1 = (*(((int*)(args_40))+7));\n  unsigned long val2 = (*(((unsigned long*)(args_40))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_40))+1));\n  unsigned long val4 = (*(((unsigned long*)(args_40))+2));\n  for (int Lidx1 = 0; Lidx1 < 2; Lidx1++) {\n    for (int Lidx2 = 0; Lidx2 < 4; Lidx2++) {\n      int alu0 = (Lidx2<<3);\n      int alu1 = ((Lidx1<<5)+alu0);\n      float4 val5 = (*((float4*)((((float*)(val3))+(alu1+4)))));\n      float4 val6 = (*((float4*)((((float*)(val3))+alu1))));\n      for (int Lidx3 = 0; Lidx3 < (val1-val0); Lidx3++) {\n        int alu2 = (val0+Lidx3);\n        int alu3 = (alu0+(alu2<<5)+(Lidx1*320));\n        float4 val7 = (*((float4*)((((float*)(val4))+(alu3+4)))));\n        float4 val8 = (*((float4*)((((float*)(val4))+alu3))));\n        *(((float*)(val2))+((Lidx2*10)+alu2+(Lidx1*40))) = (__builtin_fmaf(val5[3],val7[3],__builtin_fmaf(val5[2],val7[2],__builtin_fmaf(val5[1],val7[1],__builtin_fmaf(val5[0],val7[0],__builtin_fmaf(val6[3],val8[3],__builtin_fmaf(val6[2],val8[2],__builtin_fmaf(val6[1],val8[1],(val6[0]*val8[0]))))))))*0.3535533845424652f);\n      }\n    }\n  }\n}\n"
113 Ops.BINARY dtypes.u8 [] b"typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_2_4_i_8(unsigned char* restrict args_40) {\n  int val0 = (*(((int*)(args_40))+6));\n  int val1 = (*(((int*)(args_40))+7));\n  unsigned long val2 = (*(((unsigned long*)(args_40))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_40))+1));\n  unsigned long val4 = (*(((unsigned long*)(args_40))+2));\n  for (int Lidx1 = 0; Lidx1 < 2; Lidx1++) {\n    for (int Lidx2 = 0; Lidx2 < 4; Lidx2++) {\n      int alu0 = (Lidx2<<3);\n      int alu1 = ((Lidx1<<5)+alu0);\n      float4 val5 = (*((float4*)((((float*)(val3))+(alu1+4)))));\n      float4 val6 = (*((float4*)((((float*)(val3))+alu1))));\n      for (int Lidx3 = 0; Lidx3 < (val1-val0); Lidx3++) {\n        int alu2 = (val0+Lidx3);\n        int alu3 = (alu0+(alu2<<5)+(Lidx1*320));\n        float4 val7 = (*((float4*)((((float*)(val4))+(alu3+4)))));\n        float4 val8 = (*((float4*)((((float*)(val4))+alu3))));\n        *(((float*)(val2))+((Lidx2*10)+alu2+(Lidx1*40))) = (__builtin_fmaf(val5[3],val7[3],__builtin_fmaf(val5[2],val7[2],__builtin_fmaf(val5[1],val7[1],__builtin_fmaf(val5[0],val7[0],__builtin_fmaf(val6[3],val8[3],__builtin_fmaf(val6[2],val8[2],__builtin_fmaf(val6[1],val8[1],(val6[0]*val8[0]))))))))*0.3535533845424652f);\n      }\n    }\n  }\n}\n"
114 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
115 Ops.PARAM dtypes.f32 [37] ParamArg(slot=1, dtype=dtypes.f32)
116 Ops.PARAM dtypes.f32 [54] ParamArg(slot=2, dtype=dtypes.f32)
117 Ops.PARAM dtypes.i32 [101] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 10), name="block_lo", addrspace=AddrSpace.ALU)
118 Ops.PARAM dtypes.i32 [101] ParamArg(slot=4, dtype=dtypes.i32, vmin_vmax=(0, 10), name="block_hi", addrspace=AddrSpace.ALU)
119 Ops.PROGRAM dtypes.void [110, 111, 112, 113] ProgramInfo(global_size=(%102, 1, 1), params=(%114, %115, %116, %117, %118), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CPU", arch="x86_64,x86-64"))
120 Ops.CONST dtypes.weakint [] 1536
121 Ops.ALLOC dtypes.i8 [120] ParamArg(slot=285, dtype=dtypes.i8, device="CPU")
122 Ops.SHRINK dtypes.i8 [121, 4, 60]
123 Ops.BITCAST dtypes.f32 [122] dtypes.f32
124 Ops.PARAM dtypes.f32 [37] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
125 Ops.PARAM dtypes.f32 [54] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
126 Ops.CALL dtypes.void [119, 123, 124, 125] CallInfo()
127 Ops.CONST dtypes.weakint [] 8
128 Ops.REFPARAM dtypes.f32 [127, 7] ParamArg(slot=0, dtype=dtypes.f32)
129 Ops.INDEX dtypes.i32 [19, 10]
130 Ops.LOAD dtypes.i32 [129]
131 Ops.INDEX dtypes.i32 [19, 44]
132 Ops.LOAD dtypes.i32 [131]
133 Ops.SUB dtypes.i32 [132, 130]
134 Ops.RANGE dtypes.i32 [133] (AxisType.WEAK, 1)
135 Ops.ADD dtypes.i32 [130, 134]
136 Ops.SHL dtypes.i32 [135, 12]
137 Ops.SHRINK dtypes.f32 [128, 136, 10]
138 Ops.BUFFER dtypes.f32 [9] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
139 Ops.AFTER dtypes.f32 [138, 135]
140 Ops.SHRINK dtypes.f32 [139, 5, 10]
141 Ops.CONST dtypes.weakfloat [] -inf
142 Ops.CAST dtypes.f32 [141] dtypes.f32
143 Ops.STACK dtypes.f32 [142, 142, 142, 142]
144 Ops.STORE dtypes.void [140, 143]
145 Ops.AFTER dtypes.i32 [23, 134]
146 Ops.RANGE dtypes.i32 [145] (AxisType.WEAK, 0)
147 Ops.AFTER dtypes.f32 [138, 144, 146]
148 Ops.SHRINK dtypes.f32 [147, 5, 10]
149 Ops.LOAD dtypes.f32 [148]
150 Ops.REFPARAM dtypes.f32 [0, 41] ParamArg(slot=1, dtype=dtypes.f32)
151 Ops.MUL dtypes.i32 [135, 33]
152 Ops.ADD dtypes.i32 [151, 146]
153 Ops.INDEX dtypes.f32 [150, 152]
154 Ops.LOAD dtypes.f32 [153]
155 Ops.ADD dtypes.i32 [152, 17]
156 Ops.INDEX dtypes.f32 [150, 155]
157 Ops.LOAD dtypes.f32 [156]
158 Ops.CONST dtypes.weakint [] 20
159 Ops.CAST dtypes.i32 [158] dtypes.i32
160 Ops.ADD dtypes.i32 [152, 159]
161 Ops.INDEX dtypes.f32 [150, 160]
162 Ops.LOAD dtypes.f32 [161]
163 Ops.CONST dtypes.weakint [] 30
164 Ops.CAST dtypes.i32 [163] dtypes.i32
165 Ops.ADD dtypes.i32 [152, 164]
166 Ops.INDEX dtypes.f32 [150, 165]
167 Ops.LOAD dtypes.f32 [166]
168 Ops.STACK dtypes.f32 [154, 157, 162, 167]
169 Ops.CMPLT dtypes.bool [149, 168]
170 Ops.WHERE dtypes.f32 [169, 168, 149]
171 Ops.STORE dtypes.void [148, 170]
172 Ops.AFTER dtypes.void [101, 171]
173 Ops.END dtypes.void [172, 146]
174 Ops.AFTER dtypes.f32 [138, 173]
175 Ops.SHRINK dtypes.f32 [174, 5, 10]
176 Ops.LOAD dtypes.f32 [175]
177 Ops.STORE dtypes.void [137, 176]
178 Ops.AFTER dtypes.void [101, 177]
179 Ops.END dtypes.void [178, 134]
180 Ops.PARAM dtypes.i32 [101] ParamArg(slot=4, dtype=dtypes.i32, vmin_vmax=(1, 10), name="i", addrspace=AddrSpace.ALU)
181 Ops.CONST dtypes.weakint [] 16
182 Ops.MUL dtypes.i32 [180, 181]
183 Ops.CONST dtypes.weakint [] 32
184 Ops.MUL dtypes.i32 [180, 183]
185 Ops.ADD dtypes.i32 [184, 183]
186 Ops.SINK dtypes.void [179] KernelInfo(name="r_2_4_i", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), estimates=Estimates(ops=%182, lds=%185, mem=%185), optimized=True, split=(2, 2, 3))
187 Ops.LINEAR dtypes.void [1, 2, 9, 138, 10, 19, 129, 130, 43, 44, 131, 132, 20, 21, 22, 23, 4, 5, 3, 6, 7, 38, 39, 40, 41, 127, 128, 0, 150, 101, 141, 142, 143, 16, 17, 11, 12, 158, 159, 163, 164, 33, 133, 134, 145, 135, 139, 140, 151, 144, 146, 152, 155, 156, 157, 160, 161, 162, 165, 166, 167, 153, 154, 147, 148, 149, 168, 169, 170, 171, 172, 173, 174, 175, 176, 136, 137, 177, 178, 179, 186]
188 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_2_4_i(unsigned char* restrict args_40) {\n  float buf0[4];\n  int val0 = (*(((int*)(args_40))+4));\n  int val1 = (*(((int*)(args_40))+5));\n  int val2 = (*(((int*)(args_40))+6));\n  unsigned long val3 = (*(((unsigned long*)(args_40))+0));\n  unsigned long val4 = (*(((unsigned long*)(args_40))+1));\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    *((float4*)((buf0+0))) = (float4){((float)(-__builtin_inff())),((float)(-__builtin_inff())),((float)(-__builtin_inff())),((float)(-__builtin_inff()))};\n    for (int Lidx0 = 0; Lidx0 < val2; Lidx0++) {\n      int alu2 = ((alu0*40)+Lidx0);\n      float val5 = (*(((float*)(val4))+(alu2+10)));\n      float val6 = (*(((float*)(val4))+(alu2+20)));\n      float val7 = (*(((float*)(val4))+(alu2+30)));\n      float val8 = (*(((float*)(val4))+alu2));\n      float4 val9 = (*((float4*)((buf0+0))));\n      float4 cast0 = (float4){val8,val5,val6,val7};\n      float4 alu3 = ((val9<cast0)?cast0:val9);\n      *((float4*)((buf0+0))) = alu3;\n    }\n    *((float4*)((((float*)(val3))+(alu0<<2)))) = (*((float4*)((buf0+0))));\n  }\n}\n"
189 Ops.BINARY dtypes.u8 [] b"typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_2_4_i(unsigned char* restrict args_40) {\n  float buf0[4];\n  int val0 = (*(((int*)(args_40))+4));\n  int val1 = (*(((int*)(args_40))+5));\n  int val2 = (*(((int*)(args_40))+6));\n  unsigned long val3 = (*(((unsigned long*)(args_40))+0));\n  unsigned long val4 = (*(((unsigned long*)(args_40))+1));\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    *((float4*)((buf0+0))) = (float4){((float)(-__builtin_inff())),((float)(-__builtin_inff())),((float)(-__builtin_inff())),((float)(-__builtin_inff()))};\n    for (int Lidx0 = 0; Lidx0 < val2; Lidx0++) {\n      int alu2 = ((alu0*40)+Lidx0);\n      float val5 = (*(((float*)(val4))+(alu2+10)));\n      float val6 = (*(((float*)(val4))+(alu2+20)));\n      float val7 = (*(((float*)(val4))+(alu2+30)));\n      float val8 = (*(((float*)(val4))+alu2));\n      float4 val9 = (*((float4*)((buf0+0))));\n      float4 cast0 = (float4){val8,val5,val6,val7};\n      float4 alu3 = ((val9<cast0)?cast0:val9);\n      *((float4*)((buf0+0))) = alu3;\n    }\n    *((float4*)((((float*)(val3))+(alu0<<2)))) = (*((float4*)((buf0+0))));\n  }\n}\n"
190 Ops.PARAM dtypes.f32 [127] ParamArg(slot=0, dtype=dtypes.f32)
191 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
192 Ops.PARAM dtypes.i32 [101] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 2), name="block_lo", addrspace=AddrSpace.ALU)
193 Ops.PARAM dtypes.i32 [101] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 2), name="block_hi", addrspace=AddrSpace.ALU)
194 Ops.PROGRAM dtypes.void [186, 187, 188, 189] ProgramInfo(global_size=(2, 1, 1), params=(%190, %191, %192, %193, %180), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
195 Ops.CONST dtypes.weakint [] 1024
196 Ops.SHRINK dtypes.i8 [121, 195, 183]
197 Ops.BITCAST dtypes.f32 [196] dtypes.f32
198 Ops.CALL dtypes.void [194, 197, 123] CallInfo()
199 Ops.CONST dtypes.weakint [] 48
200 Ops.PARAM dtypes.u8 [199] ParamArg(slot=0, dtype=dtypes.u8, name="args")
201 Ops.BITCAST dtypes.u64 [200] dtypes.u64
202 Ops.INDEX dtypes.u64 [201, 5]
203 Ops.LOAD dtypes.u64 [202]
204 Ops.REFPARAM dtypes.f32 [127, 203] ParamArg(slot=0, dtype=dtypes.f32)
205 Ops.BITCAST dtypes.i32 [200] dtypes.i32
206 Ops.INDEX dtypes.i32 [205, 21]
207 Ops.LOAD dtypes.i32 [206]
208 Ops.INDEX dtypes.i32 [205, 25]
209 Ops.LOAD dtypes.i32 [208]
210 Ops.SUB dtypes.i32 [209, 207]
211 Ops.RANGE dtypes.i32 [210] (AxisType.WEAK, 1)
212 Ops.ADD dtypes.i32 [207, 211]
213 Ops.SHL dtypes.i32 [212, 12]
214 Ops.SHRINK dtypes.f32 [204, 213, 10]
215 Ops.AFTER dtypes.f32 [138, 212]
216 Ops.SHRINK dtypes.f32 [215, 5, 10]
217 Ops.CONST dtypes.weakfloat [] 0.0
218 Ops.CAST dtypes.f32 [217] dtypes.f32
219 Ops.STACK dtypes.f32 [218, 218, 218, 218]
220 Ops.STORE dtypes.void [216, 219]
221 Ops.CAST dtypes.i32 [127] dtypes.i32
222 Ops.INDEX dtypes.i32 [205, 221]
223 Ops.LOAD dtypes.i32 [222]
224 Ops.AFTER dtypes.i32 [223, 211]
225 Ops.RANGE dtypes.i32 [224] (AxisType.WEAK, 0)
226 Ops.AFTER dtypes.f32 [138, 220, 225]
227 Ops.SHRINK dtypes.f32 [226, 5, 10]
228 Ops.LOAD dtypes.f32 [227]
229 Ops.INDEX dtypes.u64 [201, 39]
230 Ops.LOAD dtypes.u64 [229]
231 Ops.REFPARAM dtypes.f32 [0, 230] ParamArg(slot=1, dtype=dtypes.f32)
232 Ops.MUL dtypes.i32 [212, 33]
233 Ops.ADD dtypes.i32 [232, 225]
234 Ops.INDEX dtypes.f32 [231, 233]
235 Ops.LOAD dtypes.f32 [234]
236 Ops.ADD dtypes.i32 [233, 17]
237 Ops.INDEX dtypes.f32 [231, 236]
238 Ops.LOAD dtypes.f32 [237]
239 Ops.ADD dtypes.i32 [233, 159]
240 Ops.INDEX dtypes.f32 [231, 239]
241 Ops.LOAD dtypes.f32 [240]
242 Ops.ADD dtypes.i32 [233, 164]
243 Ops.INDEX dtypes.f32 [231, 242]
244 Ops.LOAD dtypes.f32 [243]
245 Ops.STACK dtypes.f32 [235, 238, 241, 244]
246 Ops.INDEX dtypes.u64 [201, 12]
247 Ops.LOAD dtypes.u64 [246]
248 Ops.REFPARAM dtypes.f32 [127, 247] ParamArg(slot=2, dtype=dtypes.f32)
249 Ops.SHRINK dtypes.f32 [248, 213, 10]
250 Ops.LOAD dtypes.f32 [249]
251 Ops.SUB dtypes.f32 [245, 250]
252 Ops.CONST dtypes.weakfloat [] 1.4426950216293335
253 Ops.CAST dtypes.f32 [252] dtypes.f32
254 Ops.MUL dtypes.f32 [251, 253]
255 Ops.CMPNE dtypes.bool [254, 254]
256 Ops.CONST dtypes.weakfloat [] nan
257 Ops.CAST dtypes.f32 [256] dtypes.f32
258 Ops.CONST dtypes.weakfloat [] -150.0
259 Ops.CAST dtypes.f32 [258] dtypes.f32
260 Ops.CMPLT dtypes.bool [254, 259]
261 Ops.CONST dtypes.weakfloat [] 128.0
262 Ops.CAST dtypes.f32 [261] dtypes.f32
263 Ops.CMPLT dtypes.bool [254, 262]
264 Ops.CONST dtypes.bool [] False
265 Ops.CAST dtypes.bool [264] dtypes.bool
266 Ops.CMPEQ dtypes.bool [263, 265]
267 Ops.CONST dtypes.weakfloat [] inf
268 Ops.CAST dtypes.f32 [267] dtypes.f32
269 Ops.CONST dtypes.weakfloat [] 0.00015359208919107914
270 Ops.CAST dtypes.f32 [269] dtypes.f32
271 Ops.CMPNE dtypes.bool [254, 268]
272 Ops.CMPNE dtypes.bool [254, 142]
273 Ops.WHERE dtypes.f32 [272, 254, 218]
274 Ops.WHERE dtypes.f32 [255, 218, 273]
275 Ops.WHERE dtypes.f32 [271, 274, 218]
276 Ops.CMPLT dtypes.bool [275, 218]
277 Ops.CONST dtypes.weakfloat [] -0.5
278 Ops.CAST dtypes.f32 [277] dtypes.f32
279 Ops.CONST dtypes.weakfloat [] 0.5
280 Ops.CAST dtypes.f32 [279] dtypes.f32
281 Ops.WHERE dtypes.f32 [276, 278, 280]
282 Ops.ADD dtypes.f32 [275, 281]
283 Ops.CAST dtypes.i32 [282] dtypes.i32
284 Ops.CAST dtypes.f32 [283] dtypes.f32
285 Ops.SUB dtypes.f32 [275, 284]
286 Ops.MUL dtypes.f32 [270, 285]
287 Ops.CONST dtypes.weakfloat [] 0.0013392627006396651
288 Ops.CAST dtypes.f32 [287] dtypes.f32
289 Ops.ADD dtypes.f32 [286, 288]
290 Ops.MUL dtypes.f32 [289, 285]
291 Ops.CONST dtypes.weakfloat [] 0.009618384763598442
292 Ops.CAST dtypes.f32 [291] dtypes.f32
293 Ops.ADD dtypes.f32 [290, 292]
294 Ops.MUL dtypes.f32 [293, 285]
295 Ops.CONST dtypes.weakfloat [] 0.055503472685813904
296 Ops.CAST dtypes.f32 [295] dtypes.f32
297 Ops.ADD dtypes.f32 [294, 296]
298 Ops.MUL dtypes.f32 [297, 285]
299 Ops.CONST dtypes.weakfloat [] 0.24022644758224487
300 Ops.CAST dtypes.f32 [299] dtypes.f32
301 Ops.ADD dtypes.f32 [298, 300]
302 Ops.MUL dtypes.f32 [301, 285]
303 Ops.CONST dtypes.weakfloat [] 0.6931471824645996
304 Ops.CAST dtypes.f32 [303] dtypes.f32
305 Ops.ADD dtypes.f32 [302, 304]
306 Ops.MUL dtypes.f32 [305, 285]
307 Ops.CONST dtypes.weakfloat [] 1.0
308 Ops.CAST dtypes.f32 [307] dtypes.f32
309 Ops.ADD dtypes.f32 [306, 308]
310 Ops.SHR dtypes.i32 [283, 39]
311 Ops.CONST dtypes.weakint [] 127
312 Ops.CAST dtypes.i32 [311] dtypes.i32
313 Ops.ADD dtypes.i32 [310, 312]
314 Ops.CAST dtypes.u32 [313] dtypes.u32
315 Ops.CONST dtypes.weakint [] 23
316 Ops.CAST dtypes.u32 [315] dtypes.u32
317 Ops.SHL dtypes.u32 [314, 316]
318 Ops.CAST dtypes.i32 [317] dtypes.i32
319 Ops.BITCAST dtypes.f32 [318] dtypes.f32
320 Ops.MUL dtypes.f32 [309, 319]
321 Ops.CAST dtypes.u32 [283] dtypes.u32
322 Ops.CAST dtypes.u32 [310] dtypes.u32
323 Ops.SUB dtypes.u32 [321, 322]
324 Ops.CAST dtypes.i32 [323] dtypes.i32
325 Ops.CAST dtypes.u32 [324] dtypes.u32
326 Ops.CAST dtypes.u32 [311] dtypes.u32
327 Ops.ADD dtypes.u32 [325, 326]
328 Ops.CAST dtypes.i32 [327] dtypes.i32
329 Ops.CAST dtypes.u32 [328] dtypes.u32
330 Ops.SHL dtypes.u32 [329, 316]
331 Ops.CAST dtypes.i32 [330] dtypes.i32
332 Ops.BITCAST dtypes.f32 [331] dtypes.f32
333 Ops.MUL dtypes.f32 [320, 332]
334 Ops.WHERE dtypes.f32 [266, 268, 333]
335 Ops.WHERE dtypes.f32 [260, 218, 334]
336 Ops.WHERE dtypes.f32 [255, 257, 335]
337 Ops.ADD dtypes.f32 [228, 336]
338 Ops.STORE dtypes.void [227, 337]
339 Ops.AFTER dtypes.void [101, 338]
340 Ops.END dtypes.void [339, 225]
341 Ops.AFTER dtypes.f32 [138, 340]
342 Ops.SHRINK dtypes.f32 [341, 5, 10]
343 Ops.LOAD dtypes.f32 [342]
344 Ops.STORE dtypes.void [214, 343]
345 Ops.AFTER dtypes.void [101, 344]
346 Ops.END dtypes.void [345, 211]
347 Ops.PARAM dtypes.i32 [101] ParamArg(slot=5, dtype=dtypes.i32, vmin_vmax=(1, 10), name="i", addrspace=AddrSpace.ALU)
348 Ops.CONST dtypes.weakint [] 312
349 Ops.MUL dtypes.i32 [347, 348]
350 Ops.MUL dtypes.i32 [347, 183]
351 Ops.ADD dtypes.i32 [350, 37]
352 Ops.SINK dtypes.void [346] KernelInfo(name="r_2_4_i", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), estimates=Estimates(ops=%349, lds=%351, mem=%351), optimized=True, split=(2, 3, 4))
353 Ops.LINEAR dtypes.void [199, 200, 9, 138, 20, 21, 205, 206, 207, 24, 25, 208, 209, 127, 221, 222, 223, 4, 5, 201, 202, 203, 38, 39, 229, 230, 11, 12, 246, 247, 204, 0, 231, 248, 101, 217, 218, 219, 264, 265, 277, 278, 258, 259, 141, 142, 269, 270, 287, 288, 291, 292, 295, 296, 299, 300, 279, 280, 303, 304, 307, 308, 252, 253, 261, 262, 267, 268, 256, 257, 16, 17, 311, 312, 158, 159, 163, 164, 10, 1, 33, 326, 315, 316, 210, 211, 212, 213, 249, 250, 224, 215, 216, 232, 220, 225, 233, 236, 237, 238, 239, 240, 241, 242, 243, 244, 234, 235, 226, 227, 228, 245, 251, 254, 272, 271, 255, 273, 274, 275, 276, 281, 282, 283, 284, 321, 310, 322, 323, 324, 325, 327, 328, 329, 330, 331, 313, 314, 317, 318, 332, 319, 285, 286, 289, 290, 293, 294, 297, 298, 301, 302, 305, 306, 309, 320, 333, 260, 263, 266, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 214, 344, 345, 346, 352]
354 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef unsigned int unsigned_int4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_2_4_i(unsigned char* restrict args_48) {\n  float buf0[4];\n  int val0 = (*(((int*)(args_48))+6));\n  int val1 = (*(((int*)(args_48))+7));\n  int val2 = (*(((int*)(args_48))+8));\n  unsigned long val3 = (*(((unsigned long*)(args_48))+0));\n  unsigned long val4 = (*(((unsigned long*)(args_48))+1));\n  unsigned long val5 = (*(((unsigned long*)(args_48))+2));\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    int alu1 = (alu0<<2);\n    float4 val6 = (*((float4*)((((float*)(val5))+alu1))));\n    *((float4*)((buf0+0))) = (float4){0.0f,0.0f,0.0f,0.0f};\n    for (int Lidx0 = 0; Lidx0 < val2; Lidx0++) {\n      int alu3 = ((alu0*40)+Lidx0);\n      float val7 = (*(((float*)(val4))+(alu3+10)));\n      float val8 = (*(((float*)(val4))+(alu3+20)));\n      float val9 = (*(((float*)(val4))+(alu3+30)));\n      float val10 = (*(((float*)(val4))+alu3));\n      float4 alu4 = (((float4){val10,val7,val8,val9}-val6)*1.4426950216293335f);\n      int4 alu5 = (alu4!=alu4);\n      float4 alu6 = ((alu4!=((float)(-__builtin_inff())))?alu4:0.0f);\n      float4 alu7 = (alu5?0.0f:alu6);\n      float4 alu8 = ((alu4!=((float)(__builtin_inff())))?alu7:0.0f);\n      float4 alu9 = ((alu8<0.0f)?((float4)(-0.5f)):0.5f);\n      int4 cast0 = __builtin_convertvector((alu8+alu9), int4);\n      float4 cast1 = __builtin_convertvector(cast0, float4);\n      unsigned_int4 cast2 = __builtin_convertvector(cast0, unsigned_int4);\n      int4 alu10 = (cast0>>1);\n      unsigned_int4 cast3 = __builtin_convertvector(alu10, unsigned_int4);\n      int4 cast4 = __builtin_convertvector((cast2-cast3), int4);\n      unsigned_int4 cast5 = __builtin_convertvector(cast4, unsigned_int4);\n      int4 cast6 = __builtin_convertvector((cast5+127u), int4);\n      unsigned_int4 cast7 = __builtin_convertvector(cast6, unsigned_int4);\n      int4 cast8 = __builtin_convertvector((cast7<<23u), int4);\n      unsigned_int4 cast9 = __builtin_convertvector((alu10+127), unsigned_int4);\n      int4 cast10 = __builtin_convertvector((cast9<<23u), int4);\n      float4 alu11 = (alu8-cast1);\n      float4 alu12 = ((~((alu4<128.0f)^0))?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu11)+0.0013392627006396651f)*alu11)+0.009618384763598442f)*alu11)+0.055503472685813904f)*alu11)+0.24022644758224487f)*alu11)+0.6931471824645996f)*alu11)+1.0f)*((float4)((int4)(cast10)))*((float4)((int4)(cast8)))));\n      float4 alu13 = ((alu4<-150.0f)?0.0f:alu12);\n      float4 alu14 = (alu5?__builtin_bit_cast(float, (unsigned int)(0x7fc00000u)):alu13);\n      *((float4*)((buf0+0))) = ((*((float4*)((buf0+0))))+alu14);\n    }\n    *((float4*)((((float*)(val3))+alu1))) = (*((float4*)((buf0+0))));\n  }\n}\n"
355 Ops.BINARY dtypes.u8 [] b"typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef unsigned int unsigned_int4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_2_4_i(unsigned char* restrict args_48) {\n  float buf0[4];\n  int val0 = (*(((int*)(args_48))+6));\n  int val1 = (*(((int*)(args_48))+7));\n  int val2 = (*(((int*)(args_48))+8));\n  unsigned long val3 = (*(((unsigned long*)(args_48))+0));\n  unsigned long val4 = (*(((unsigned long*)(args_48))+1));\n  unsigned long val5 = (*(((unsigned long*)(args_48))+2));\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    int alu1 = (alu0<<2);\n    float4 val6 = (*((float4*)((((float*)(val5))+alu1))));\n    *((float4*)((buf0+0))) = (float4){0.0f,0.0f,0.0f,0.0f};\n    for (int Lidx0 = 0; Lidx0 < val2; Lidx0++) {\n      int alu3 = ((alu0*40)+Lidx0);\n      float val7 = (*(((float*)(val4))+(alu3+10)));\n      float val8 = (*(((float*)(val4))+(alu3+20)));\n      float val9 = (*(((float*)(val4))+(alu3+30)));\n      float val10 = (*(((float*)(val4))+alu3));\n      float4 alu4 = (((float4){val10,val7,val8,val9}-val6)*1.4426950216293335f);\n      int4 alu5 = (alu4!=alu4);\n      float4 alu6 = ((alu4!=((float)(-__builtin_inff())))?alu4:0.0f);\n      float4 alu7 = (alu5?0.0f:alu6);\n      float4 alu8 = ((alu4!=((float)(__builtin_inff())))?alu7:0.0f);\n      float4 alu9 = ((alu8<0.0f)?((float4)(-0.5f)):0.5f);\n      int4 cast0 = __builtin_convertvector((alu8+alu9), int4);\n      float4 cast1 = __builtin_convertvector(cast0, float4);\n      unsigned_int4 cast2 = __builtin_convertvector(cast0, unsigned_int4);\n      int4 alu10 = (cast0>>1);\n      unsigned_int4 cast3 = __builtin_convertvector(alu10, unsigned_int4);\n      int4 cast4 = __builtin_convertvector((cast2-cast3), int4);\n      unsigned_int4 cast5 = __builtin_convertvector(cast4, unsigned_int4);\n      int4 cast6 = __builtin_convertvector((cast5+127u), int4);\n      unsigned_int4 cast7 = __builtin_convertvector(cast6, unsigned_int4);\n      int4 cast8 = __builtin_convertvector((cast7<<23u), int4);\n      unsigned_int4 cast9 = __builtin_convertvector((alu10+127), unsigned_int4);\n      int4 cast10 = __builtin_convertvector((cast9<<23u), int4);\n      float4 alu11 = (alu8-cast1);\n      float4 alu12 = ((~((alu4<128.0f)^0))?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu11)+0.0013392627006396651f)*alu11)+0.009618384763598442f)*alu11)+0.055503472685813904f)*alu11)+0.24022644758224487f)*alu11)+0.6931471824645996f)*alu11)+1.0f)*((float4)((int4)(cast10)))*((float4)((int4)(cast8)))));\n      float4 alu13 = ((alu4<-150.0f)?0.0f:alu12);\n      float4 alu14 = (alu5?__builtin_bit_cast(float, (unsigned int)(0x7fc00000u)):alu13);\n      *((float4*)((buf0+0))) = ((*((float4*)((buf0+0))))+alu14);\n    }\n    *((float4*)((((float*)(val3))+alu1))) = (*((float4*)((buf0+0))));\n  }\n}\n"
356 Ops.PARAM dtypes.f32 [127] ParamArg(slot=2, dtype=dtypes.f32)
357 Ops.PARAM dtypes.i32 [101] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 2), name="block_lo", addrspace=AddrSpace.ALU)
358 Ops.PARAM dtypes.i32 [101] ParamArg(slot=4, dtype=dtypes.i32, vmin_vmax=(0, 2), name="block_hi", addrspace=AddrSpace.ALU)
359 Ops.PROGRAM dtypes.void [352, 353, 354, 355] ProgramInfo(global_size=(2, 1, 1), params=(%190, %191, %356, %357, %358, %347), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CPU", arch="x86_64,x86-64"))
360 Ops.CONST dtypes.weakint [] 1280
361 Ops.SHRINK dtypes.i8 [121, 360, 183]
362 Ops.BITCAST dtypes.f32 [361] dtypes.f32
363 Ops.CALL dtypes.void [359, 362, 123, 197] CallInfo()
364 Ops.REFPARAM dtypes.f32 [0, 203] ParamArg(slot=0, dtype=dtypes.f32)
365 Ops.CONST dtypes.weakint [] 9
366 Ops.CAST dtypes.i32 [365] dtypes.i32
367 Ops.INDEX dtypes.i32 [205, 366]
368 Ops.LOAD dtypes.i32 [367]
369 Ops.SUB dtypes.i32 [368, 223]
370 Ops.RANGE dtypes.i32 [369] (AxisType.WEAK, 2)
371 Ops.ADD dtypes.i32 [223, 370]
372 Ops.INDEX dtypes.f32 [364, 371]
373 Ops.INDEX dtypes.f32 [231, 371]
374 Ops.LOAD dtypes.f32 [373]
375 Ops.ADD dtypes.i32 [371, 17]
376 Ops.INDEX dtypes.f32 [231, 375]
377 Ops.LOAD dtypes.f32 [376]
378 Ops.ADD dtypes.i32 [371, 159]
379 Ops.INDEX dtypes.f32 [231, 378]
380 Ops.LOAD dtypes.f32 [379]
381 Ops.ADD dtypes.i32 [371, 164]
382 Ops.INDEX dtypes.f32 [231, 381]
383 Ops.LOAD dtypes.f32 [382]
384 Ops.ADD dtypes.i32 [371, 33]
385 Ops.INDEX dtypes.f32 [231, 384]
386 Ops.LOAD dtypes.f32 [385]
387 Ops.CONST dtypes.weakint [] 50
388 Ops.CAST dtypes.i32 [387] dtypes.i32
389 Ops.ADD dtypes.i32 [371, 388]
390 Ops.INDEX dtypes.f32 [231, 389]
391 Ops.LOAD dtypes.f32 [390]
392 Ops.CONST dtypes.weakint [] 60
393 Ops.CAST dtypes.i32 [392] dtypes.i32
394 Ops.ADD dtypes.i32 [371, 393]
395 Ops.INDEX dtypes.f32 [231, 394]
396 Ops.LOAD dtypes.f32 [395]
397 Ops.CONST dtypes.weakint [] 70
398 Ops.CAST dtypes.i32 [397] dtypes.i32
399 Ops.ADD dtypes.i32 [371, 398]
400 Ops.INDEX dtypes.f32 [231, 399]
401 Ops.LOAD dtypes.f32 [400]
402 Ops.STACK dtypes.f32 [374, 377, 380, 383, 386, 391, 396, 401]
403 Ops.SHRINK dtypes.f32 [248, 5, 10]
404 Ops.LOAD dtypes.f32 [403]
405 Ops.INDEX dtypes.f32 [404, 5]
406 Ops.INDEX dtypes.f32 [404, 39]
407 Ops.INDEX dtypes.f32 [404, 12]
408 Ops.INDEX dtypes.f32 [404, 47]
409 Ops.SHRINK dtypes.f32 [248, 10, 10]
410 Ops.LOAD dtypes.f32 [409]
411 Ops.INDEX dtypes.f32 [410, 5]
412 Ops.INDEX dtypes.f32 [410, 39]
413 Ops.INDEX dtypes.f32 [410, 12]
414 Ops.INDEX dtypes.f32 [410, 47]
415 Ops.STACK dtypes.f32 [405, 406, 407, 408, 411, 412, 413, 414]
416 Ops.SUB dtypes.f32 [402, 415]
417 Ops.MUL dtypes.f32 [416, 253]
418 Ops.CMPNE dtypes.bool [417, 417]
419 Ops.CMPLT dtypes.bool [417, 259]
420 Ops.CMPLT dtypes.bool [417, 262]
421 Ops.CMPEQ dtypes.bool [420, 265]
422 Ops.CMPNE dtypes.bool [417, 268]
423 Ops.CMPNE dtypes.bool [417, 142]
424 Ops.WHERE dtypes.f32 [423, 417, 218]
425 Ops.WHERE dtypes.f32 [418, 218, 424]
426 Ops.WHERE dtypes.f32 [422, 425, 218]
427 Ops.CMPLT dtypes.bool [426, 218]
428 Ops.WHERE dtypes.f32 [427, 278, 280]
429 Ops.ADD dtypes.f32 [426, 428]
430 Ops.CAST dtypes.i32 [429] dtypes.i32
431 Ops.CAST dtypes.f32 [430] dtypes.f32
432 Ops.SUB dtypes.f32 [426, 431]
433 Ops.MUL dtypes.f32 [270, 432]
434 Ops.ADD dtypes.f32 [433, 288]
435 Ops.MUL dtypes.f32 [434, 432]
436 Ops.ADD dtypes.f32 [435, 292]
437 Ops.MUL dtypes.f32 [436, 432]
438 Ops.ADD dtypes.f32 [437, 296]
439 Ops.MUL dtypes.f32 [438, 432]
440 Ops.ADD dtypes.f32 [439, 300]
441 Ops.MUL dtypes.f32 [440, 432]
442 Ops.ADD dtypes.f32 [441, 304]
443 Ops.MUL dtypes.f32 [442, 432]
444 Ops.ADD dtypes.f32 [443, 308]
445 Ops.SHR dtypes.i32 [430, 39]
446 Ops.ADD dtypes.i32 [445, 312]
447 Ops.CAST dtypes.u32 [446] dtypes.u32
448 Ops.SHL dtypes.u32 [447, 316]
449 Ops.CAST dtypes.i32 [448] dtypes.i32
450 Ops.BITCAST dtypes.f32 [449] dtypes.f32
451 Ops.MUL dtypes.f32 [444, 450]
452 Ops.CAST dtypes.u32 [430] dtypes.u32
453 Ops.CAST dtypes.u32 [445] dtypes.u32
454 Ops.SUB dtypes.u32 [452, 453]
455 Ops.CAST dtypes.i32 [454] dtypes.i32
456 Ops.CAST dtypes.u32 [455] dtypes.u32
457 Ops.ADD dtypes.u32 [456, 326]
458 Ops.CAST dtypes.i32 [457] dtypes.i32
459 Ops.CAST dtypes.u32 [458] dtypes.u32
460 Ops.SHL dtypes.u32 [459, 316]
461 Ops.CAST dtypes.i32 [460] dtypes.i32
462 Ops.BITCAST dtypes.f32 [461] dtypes.f32
463 Ops.MUL dtypes.f32 [451, 462]
464 Ops.WHERE dtypes.f32 [421, 268, 463]
465 Ops.WHERE dtypes.f32 [419, 218, 464]
466 Ops.WHERE dtypes.f32 [418, 257, 465]
467 Ops.INDEX dtypes.u64 [201, 47]
468 Ops.LOAD dtypes.u64 [467]
469 Ops.REFPARAM dtypes.f32 [127, 468] ParamArg(slot=3, dtype=dtypes.f32)
470 Ops.SHRINK dtypes.f32 [469, 5, 10]
471 Ops.LOAD dtypes.f32 [470]
472 Ops.INDEX dtypes.f32 [471, 5]
473 Ops.INDEX dtypes.f32 [471, 39]
474 Ops.INDEX dtypes.f32 [471, 12]
475 Ops.INDEX dtypes.f32 [471, 47]
476 Ops.SHRINK dtypes.f32 [469, 10, 10]
477 Ops.LOAD dtypes.f32 [476]
478 Ops.INDEX dtypes.f32 [477, 5]
479 Ops.INDEX dtypes.f32 [477, 39]
480 Ops.INDEX dtypes.f32 [477, 12]
481 Ops.INDEX dtypes.f32 [477, 47]
482 Ops.STACK dtypes.f32 [472, 473, 474, 475, 478, 479, 480, 481]
483 Ops.FDIV dtypes.f32 [308, 482]
484 Ops.MUL dtypes.f32 [466, 483]
485 Ops.INDEX dtypes.f32 [484, 5]
486 Ops.STORE dtypes.void [372, 485]
487 Ops.INDEX dtypes.f32 [364, 375]
488 Ops.INDEX dtypes.f32 [484, 39]
489 Ops.STORE dtypes.void [487, 488]
490 Ops.INDEX dtypes.f32 [364, 378]
491 Ops.INDEX dtypes.f32 [484, 12]
492 Ops.STORE dtypes.void [490, 491]
493 Ops.INDEX dtypes.f32 [364, 381]
494 Ops.INDEX dtypes.f32 [484, 47]
495 Ops.STORE dtypes.void [493, 494]
496 Ops.INDEX dtypes.f32 [364, 384]
497 Ops.INDEX dtypes.f32 [484, 10]
498 Ops.STORE dtypes.void [496, 497]
499 Ops.INDEX dtypes.f32 [364, 389]
500 Ops.INDEX dtypes.f32 [484, 44]
501 Ops.STORE dtypes.void [499, 500]
502 Ops.INDEX dtypes.f32 [364, 394]
503 Ops.INDEX dtypes.f32 [484, 21]
504 Ops.STORE dtypes.void [502, 503]
505 Ops.INDEX dtypes.f32 [364, 399]
506 Ops.INDEX dtypes.f32 [484, 25]
507 Ops.STORE dtypes.void [505, 506]
508 Ops.AFTER dtypes.void [101, 486, 489, 492, 495, 498, 501, 504, 507]
509 Ops.END dtypes.void [508, 370]
510 Ops.MUL dtypes.i32 [103, 348]
511 Ops.ADD dtypes.i32 [127, 510]
512 Ops.MUL dtypes.i32 [103, 37]
513 Ops.ADD dtypes.i32 [37, 512]
514 Ops.SINK dtypes.void [509] KernelInfo(name="E_i_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)),), estimates=Estimates(ops=%511, lds=%513, mem=%513), optimized=True, split=(%102, 4, 5))
515 Ops.LINEAR dtypes.void [199, 200, 127, 221, 205, 222, 223, 365, 366, 367, 368, 4, 5, 201, 202, 203, 38, 39, 229, 230, 11, 12, 246, 247, 46, 47, 467, 468, 248, 9, 10, 403, 404, 409, 410, 469, 470, 471, 476, 477, 0, 364, 231, 101, 405, 406, 407, 408, 411, 412, 413, 414, 415, 472, 473, 474, 475, 478, 479, 480, 481, 482, 264, 265, 277, 278, 258, 259, 141, 142, 217, 218, 269, 270, 287, 288, 291, 292, 295, 296, 299, 300, 279, 280, 303, 304, 307, 308, 252, 253, 261, 262, 267, 268, 256, 257, 16, 17, 311, 312, 158, 159, 163, 164, 1, 33, 43, 44, 387, 388, 20, 21, 392, 393, 24, 25, 397, 398, 326, 315, 316, 369, 483, 370, 371, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 389, 390, 391, 394, 395, 396, 399, 400, 401, 402, 372, 487, 490, 493, 496, 499, 502, 505, 416, 417, 423, 422, 418, 424, 425, 426, 427, 428, 429, 430, 431, 452, 445, 453, 454, 455, 456, 457, 458, 459, 460, 461, 446, 447, 448, 449, 462, 450, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 451, 463, 419, 420, 421, 464, 465, 466, 484, 485, 488, 491, 494, 497, 500, 503, 506, 486, 489, 492, 495, 498, 501, 504, 507, 508, 509, 514]
516 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef float float8 __attribute__((aligned(4),ext_vector_type(8)));\ntypedef int int8 __attribute__((aligned(4),ext_vector_type(8)));\ntypedef unsigned int unsigned_int8 __attribute__((aligned(4),ext_vector_type(8)));\n__attribute__((min_vector_width(512))) void E_i_8(unsigned char* restrict args_48) {\n  int val0 = (*(((int*)(args_48))+8));\n  int val1 = (*(((int*)(args_48))+9));\n  unsigned long val2 = (*(((unsigned long*)(args_48))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_48))+1));\n  unsigned long val4 = (*(((unsigned long*)(args_48))+2));\n  unsigned long val5 = (*(((unsigned long*)(args_48))+3));\n  float4 val6 = (*((float4*)((((float*)(val4))+0))));\n  float4 val7 = (*((float4*)((((float*)(val4))+4))));\n  float4 val8 = (*((float4*)((((float*)(val5))+0))));\n  float4 val9 = (*((float4*)((((float*)(val5))+4))));\n  for (int Lidx2 = 0; Lidx2 < (val1-val0); Lidx2++) {\n    int alu0 = (val0+Lidx2);\n    float val10 = (*(((float*)(val3))+alu0));\n    int alu1 = (alu0+10);\n    float val11 = (*(((float*)(val3))+alu1));\n    int alu2 = (alu0+20);\n    float val12 = (*(((float*)(val3))+alu2));\n    int alu3 = (alu0+30);\n    float val13 = (*(((float*)(val3))+alu3));\n    int alu4 = (alu0+40);\n    float val14 = (*(((float*)(val3))+alu4));\n    int alu5 = (alu0+50);\n    float val15 = (*(((float*)(val3))+alu5));\n    int alu6 = (alu0+60);\n    float val16 = (*(((float*)(val3))+alu6));\n    int alu7 = (alu0+70);\n    float val17 = (*(((float*)(val3))+alu7));\n    float8 alu8 = (((float8){val10,val11,val12,val13,val14,val15,val16,val17}-((float8)__builtin_shufflevector(val6, val7, 0, 1, 2, 3, 4, 5, 6, 7)))*1.4426950216293335f);\n    int8 alu9 = (alu8!=alu8);\n    float8 alu10 = ((alu8!=((float)(-__builtin_inff())))?alu8:0.0f);\n    float8 alu11 = (alu9?0.0f:alu10);\n    float8 alu12 = ((alu8!=((float)(__builtin_inff())))?alu11:0.0f);\n    float8 alu13 = ((alu12<0.0f)?((float8)(-0.5f)):0.5f);\n    int8 cast0 = __builtin_convertvector((alu12+alu13), int8);\n    float8 cast1 = __builtin_convertvector(cast0, float8);\n    unsigned_int8 cast2 = __builtin_convertvector(cast0, unsigned_int8);\n    int8 alu14 = (cast0>>1);\n    unsigned_int8 cast3 = __builtin_convertvector(alu14, unsigned_int8);\n    int8 cast4 = __builtin_convertvector((cast2-cast3), int8);\n    unsigned_int8 cast5 = __builtin_convertvector(cast4, unsigned_int8);\n    int8 cast6 = __builtin_convertvector((cast5+127u), int8);\n    unsigned_int8 cast7 = __builtin_convertvector(cast6, unsigned_int8);\n    int8 cast8 = __builtin_convertvector((cast7<<23u), int8);\n    unsigned_int8 cast9 = __builtin_convertvector((alu14+127), unsigned_int8);\n    int8 cast10 = __builtin_convertvector((cast9<<23u), int8);\n    float8 alu15 = (alu12-cast1);\n    float8 alu16 = ((~((alu8<128.0f)^0))?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu15)+0.0013392627006396651f)*alu15)+0.009618384763598442f)*alu15)+0.055503472685813904f)*alu15)+0.24022644758224487f)*alu15)+0.6931471824645996f)*alu15)+1.0f)*((float8)((int8)(cast10)))*((float8)((int8)(cast8)))));\n    float8 alu17 = ((alu8<-150.0f)?0.0f:alu16);\n    float8 alu18 = (alu9?__builtin_bit_cast(float, (unsigned int)(0x7fc00000u)):alu17);\n    float8 alu19 = (alu18*(1.0f/((float8)__builtin_shufflevector(val8, val9, 0, 1, 2, 3, 4, 5, 6, 7))));\n    *(((float*)(val2))+alu0) = alu19[0];\n    *(((float*)(val2))+alu1) = alu19[1];\n    *(((float*)(val2))+alu2) = alu19[2];\n    *(((float*)(val2))+alu3) = alu19[3];\n    *(((float*)(val2))+alu4) = alu19[4];\n    *(((float*)(val2))+alu5) = alu19[5];\n    *(((float*)(val2))+alu6) = alu19[6];\n    *(((float*)(val2))+alu7) = alu19[7];\n  }\n}\n"
517 Ops.BINARY dtypes.u8 [] b"typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef float float8 __attribute__((aligned(4),ext_vector_type(8)));\ntypedef int int8 __attribute__((aligned(4),ext_vector_type(8)));\ntypedef unsigned int unsigned_int8 __attribute__((aligned(4),ext_vector_type(8)));\n__attribute__((min_vector_width(512))) void E_i_8(unsigned char* restrict args_48) {\n  int val0 = (*(((int*)(args_48))+8));\n  int val1 = (*(((int*)(args_48))+9));\n  unsigned long val2 = (*(((unsigned long*)(args_48))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_48))+1));\n  unsigned long val4 = (*(((unsigned long*)(args_48))+2));\n  unsigned long val5 = (*(((unsigned long*)(args_48))+3));\n  float4 val6 = (*((float4*)((((float*)(val4))+0))));\n  float4 val7 = (*((float4*)((((float*)(val4))+4))));\n  float4 val8 = (*((float4*)((((float*)(val5))+0))));\n  float4 val9 = (*((float4*)((((float*)(val5))+4))));\n  for (int Lidx2 = 0; Lidx2 < (val1-val0); Lidx2++) {\n    int alu0 = (val0+Lidx2);\n    float val10 = (*(((float*)(val3))+alu0));\n    int alu1 = (alu0+10);\n    float val11 = (*(((float*)(val3))+alu1));\n    int alu2 = (alu0+20);\n    float val12 = (*(((float*)(val3))+alu2));\n    int alu3 = (alu0+30);\n    float val13 = (*(((float*)(val3))+alu3));\n    int alu4 = (alu0+40);\n    float val14 = (*(((float*)(val3))+alu4));\n    int alu5 = (alu0+50);\n    float val15 = (*(((float*)(val3))+alu5));\n    int alu6 = (alu0+60);\n    float val16 = (*(((float*)(val3))+alu6));\n    int alu7 = (alu0+70);\n    float val17 = (*(((float*)(val3))+alu7));\n    float8 alu8 = (((float8){val10,val11,val12,val13,val14,val15,val16,val17}-((float8)__builtin_shufflevector(val6, val7, 0, 1, 2, 3, 4, 5, 6, 7)))*1.4426950216293335f);\n    int8 alu9 = (alu8!=alu8);\n    float8 alu10 = ((alu8!=((float)(-__builtin_inff())))?alu8:0.0f);\n    float8 alu11 = (alu9?0.0f:alu10);\n    float8 alu12 = ((alu8!=((float)(__builtin_inff())))?alu11:0.0f);\n    float8 alu13 = ((alu12<0.0f)?((float8)(-0.5f)):0.5f);\n    int8 cast0 = __builtin_convertvector((alu12+alu13), int8);\n    float8 cast1 = __builtin_convertvector(cast0, float8);\n    unsigned_int8 cast2 = __builtin_convertvector(cast0, unsigned_int8);\n    int8 alu14 = (cast0>>1);\n    unsigned_int8 cast3 = __builtin_convertvector(alu14, unsigned_int8);\n    int8 cast4 = __builtin_convertvector((cast2-cast3), int8);\n    unsigned_int8 cast5 = __builtin_convertvector(cast4, unsigned_int8);\n    int8 cast6 = __builtin_convertvector((cast5+127u), int8);\n    unsigned_int8 cast7 = __builtin_convertvector(cast6, unsigned_int8);\n    int8 cast8 = __builtin_convertvector((cast7<<23u), int8);\n    unsigned_int8 cast9 = __builtin_convertvector((alu14+127), unsigned_int8);\n    int8 cast10 = __builtin_convertvector((cast9<<23u), int8);\n    float8 alu15 = (alu12-cast1);\n    float8 alu16 = ((~((alu8<128.0f)^0))?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu15)+0.0013392627006396651f)*alu15)+0.009618384763598442f)*alu15)+0.055503472685813904f)*alu15)+0.24022644758224487f)*alu15)+0.6931471824645996f)*alu15)+1.0f)*((float8)((int8)(cast10)))*((float8)((int8)(cast8)))));\n    float8 alu17 = ((alu8<-150.0f)?0.0f:alu16);\n    float8 alu18 = (alu9?__builtin_bit_cast(float, (unsigned int)(0x7fc00000u)):alu17);\n    float8 alu19 = (alu18*(1.0f/((float8)__builtin_shufflevector(val8, val9, 0, 1, 2, 3, 4, 5, 6, 7))));\n    *(((float*)(val2))+alu0) = alu19[0];\n    *(((float*)(val2))+alu1) = alu19[1];\n    *(((float*)(val2))+alu2) = alu19[2];\n    *(((float*)(val2))+alu3) = alu19[3];\n    *(((float*)(val2))+alu4) = alu19[4];\n    *(((float*)(val2))+alu5) = alu19[5];\n    *(((float*)(val2))+alu6) = alu19[6];\n    *(((float*)(val2))+alu7) = alu19[7];\n  }\n}\n"
518 Ops.PARAM dtypes.f32 [127] ParamArg(slot=3, dtype=dtypes.f32)
519 Ops.PARAM dtypes.i32 [101] ParamArg(slot=4, dtype=dtypes.i32, vmin_vmax=(0, 10), name="block_lo", addrspace=AddrSpace.ALU)
520 Ops.PARAM dtypes.i32 [101] ParamArg(slot=5, dtype=dtypes.i32, vmin_vmax=(0, 10), name="block_hi", addrspace=AddrSpace.ALU)
521 Ops.PROGRAM dtypes.void [514, 515, 516, 517] ProgramInfo(global_size=(%102, 1, 1), params=(%114, %191, %356, %518, %519, %520), globals=(0, 1, 2, 3), outs=(0,), ins=(1, 2, 3), target=Target(device="CPU", arch="x86_64,x86-64"))
522 Ops.CONST dtypes.weakint [] 512
523 Ops.SHRINK dtypes.i8 [121, 522, 60]
524 Ops.BITCAST dtypes.f32 [523] dtypes.f32
525 Ops.CALL dtypes.void [521, 524, 123, 197, 362] CallInfo()
526 Ops.REFPARAM dtypes.f32 [37, 203] ParamArg(slot=0, dtype=dtypes.f32)
527 Ops.AFTER dtypes.i32 [210, 13]
528 Ops.RANGE dtypes.i32 [527] (AxisType.WEAK, 2)
529 Ops.ADD dtypes.i32 [207, 528]
530 Ops.SHL dtypes.i32 [529, 47]
531 Ops.ADD dtypes.i32 [45, 530]
532 Ops.SHRINK dtypes.f32 [526, 531, 10]
533 Ops.BUFFER dtypes.f32 [127] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
534 Ops.AFTER dtypes.f32 [533, 529, 13]
535 Ops.SHRINK dtypes.f32 [534, 5, 10]
536 Ops.STORE dtypes.void [535, 219]
537 Ops.SHRINK dtypes.f32 [534, 10, 10]
538 Ops.STORE dtypes.void [537, 219]
539 Ops.AFTER dtypes.i32 [223, 528]
540 Ops.RANGE dtypes.i32 [539] (AxisType.WEAK, 0)
541 Ops.AFTER dtypes.f32 [533, 536, 538, 540]
542 Ops.SHRINK dtypes.f32 [541, 5, 10]
543 Ops.MUL dtypes.i32 [529, 17]
544 Ops.ADD dtypes.i32 [543, 540]
545 Ops.ADD dtypes.i32 [544, 34]
546 Ops.INDEX dtypes.f32 [231, 545]
547 Ops.LOAD dtypes.f32 [546]
548 Ops.STACK dtypes.f32 [547, 547, 547, 547, 547, 547, 547, 547]
549 Ops.REFPARAM dtypes.f32 [54, 247] ParamArg(slot=2, dtype=dtypes.f32)
550 Ops.SHL dtypes.i32 [540, 44]
551 Ops.ADD dtypes.i32 [550, 530]
552 Ops.ADD dtypes.i32 [551, 62]
553 Ops.SHRINK dtypes.f32 [549, 552, 10]
554 Ops.LOAD dtypes.f32 [553]
555 Ops.INDEX dtypes.f32 [554, 5]
556 Ops.INDEX dtypes.f32 [554, 39]
557 Ops.INDEX dtypes.f32 [554, 12]
558 Ops.INDEX dtypes.f32 [554, 47]
559 Ops.ADD dtypes.i32 [552, 10]
560 Ops.SHRINK dtypes.f32 [549, 559, 10]
561 Ops.LOAD dtypes.f32 [560]
562 Ops.INDEX dtypes.f32 [561, 5]
563 Ops.INDEX dtypes.f32 [561, 39]
564 Ops.INDEX dtypes.f32 [561, 12]
565 Ops.INDEX dtypes.f32 [561, 47]
566 Ops.STACK dtypes.f32 [555, 556, 557, 558, 562, 563, 564, 565]
567 Ops.LOAD dtypes.f32 [542]
568 Ops.INDEX dtypes.f32 [567, 5]
569 Ops.INDEX dtypes.f32 [567, 39]
570 Ops.INDEX dtypes.f32 [567, 12]
571 Ops.INDEX dtypes.f32 [567, 47]
572 Ops.SHRINK dtypes.f32 [541, 10, 10]
573 Ops.LOAD dtypes.f32 [572]
574 Ops.INDEX dtypes.f32 [573, 5]
575 Ops.INDEX dtypes.f32 [573, 39]
576 Ops.INDEX dtypes.f32 [573, 12]
577 Ops.INDEX dtypes.f32 [573, 47]
578 Ops.STACK dtypes.f32 [568, 569, 570, 571, 574, 575, 576, 577]
579 Ops.MULACC dtypes.f32 [548, 566, 578]
580 Ops.INDEX dtypes.f32 [579, 5]
581 Ops.INDEX dtypes.f32 [579, 39]
582 Ops.INDEX dtypes.f32 [579, 12]
583 Ops.INDEX dtypes.f32 [579, 47]
584 Ops.STACK dtypes.f32 [580, 581, 582, 583]
585 Ops.STORE dtypes.void [542, 584]
586 Ops.INDEX dtypes.f32 [579, 10]
587 Ops.INDEX dtypes.f32 [579, 44]
588 Ops.INDEX dtypes.f32 [579, 21]
589 Ops.INDEX dtypes.f32 [579, 25]
590 Ops.STACK dtypes.f32 [586, 587, 588, 589]
591 Ops.STORE dtypes.void [572, 590]
592 Ops.AFTER dtypes.void [101, 585, 591]
593 Ops.END dtypes.void [592, 540]
594 Ops.AFTER dtypes.f32 [533, 593]
595 Ops.SHRINK dtypes.f32 [594, 5, 10]
596 Ops.LOAD dtypes.f32 [595]
597 Ops.STORE dtypes.void [532, 596]
598 Ops.ADD dtypes.i32 [531, 10]
599 Ops.SHRINK dtypes.f32 [526, 598, 10]
600 Ops.SHRINK dtypes.f32 [594, 10, 10]
601 Ops.LOAD dtypes.f32 [600]
602 Ops.STORE dtypes.void [599, 601]
603 Ops.AFTER dtypes.void [101, 597, 602]
604 Ops.END dtypes.void [603, 528]
605 Ops.END dtypes.void [604, 13]
606 Ops.MUL dtypes.i32 [347, 104]
607 Ops.MUL dtypes.i32 [347, 107]
608 Ops.ADD dtypes.i32 [607, 106]
609 Ops.SINK dtypes.void [605] KernelInfo(name="r_2_4_8_i", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(8, AxisType.UPCAST)),), estimates=Estimates(ops=%606, lds=%608, mem=%608), optimized=True, split=(4, 3, 4))
610 Ops.LINEAR dtypes.void [199, 200, 127, 533, 20, 21, 205, 206, 207, 24, 25, 208, 209, 221, 222, 223, 4, 5, 201, 202, 203, 38, 39, 229, 230, 11, 12, 246, 247, 37, 526, 0, 231, 54, 549, 101, 217, 218, 219, 16, 17, 46, 47, 60, 61, 9, 10, 1, 33, 43, 44, 210, 13, 527, 62, 34, 45, 528, 539, 529, 534, 535, 537, 543, 530, 536, 538, 540, 544, 545, 546, 547, 550, 551, 552, 559, 560, 561, 553, 554, 541, 542, 567, 572, 573, 562, 563, 564, 565, 555, 556, 557, 558, 566, 568, 569, 570, 571, 574, 575, 576, 577, 578, 548, 579, 580, 581, 582, 583, 584, 586, 587, 588, 589, 590, 585, 591, 592, 593, 594, 595, 596, 600, 601, 531, 598, 599, 532, 602, 597, 603, 604, 605, 609]
611 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef float float8 __attribute__((aligned(4),ext_vector_type(8)));\n__attribute__((min_vector_width(512))) void r_2_4_8_i(unsigned char* restrict args_48) {\n  float buf0[8];\n  int val0 = (*(((int*)(args_48))+6));\n  int val1 = (*(((int*)(args_48))+7));\n  int val2 = (*(((int*)(args_48))+8));\n  unsigned long val3 = (*(((unsigned long*)(args_48))+0));\n  unsigned long val4 = (*(((unsigned long*)(args_48))+1));\n  unsigned long val5 = (*(((unsigned long*)(args_48))+2));\n  float4 cast0 = (float4){0.0f,0.0f,0.0f,0.0f};\n  for (int Lidx1 = 0; Lidx1 < 2; Lidx1++) {\n    for (int Lidx2 = 0; Lidx2 < (val1-val0); Lidx2++) {\n      int alu0 = (val0+Lidx2);\n      int alu1 = (alu0<<3);\n      *((float4*)((buf0+0))) = cast0;\n      *((float4*)((buf0+4))) = cast0;\n      for (int Lidx0 = 0; Lidx0 < val2; Lidx0++) {\n        float val6 = (*(((float*)(val4))+((alu0*10)+Lidx0+(Lidx1*40))));\n        int alu4 = ((Lidx0<<5)+alu1+(Lidx1*320));\n        float4 val7 = (*((float4*)((((float*)(val5))+(alu4+4)))));\n        float4 val8 = (*((float4*)((((float*)(val5))+alu4))));\n        float4 val9 = (*((float4*)((buf0+0))));\n        float4 val10 = (*((float4*)((buf0+4))));\n        float8 alu5 = __builtin_elementwise_fma((float8){val6,val6,val6,val6,val6,val6,val6,val6}, ((float8)__builtin_shufflevector(val8, val7, 0, 1, 2, 3, 4, 5, 6, 7)), ((float8)__builtin_shufflevector(val9, val10, 0, 1, 2, 3, 4, 5, 6, 7)));\n        *((float4*)((buf0+0))) = ((float4)__builtin_shufflevector(alu5, alu5, 0, 1, 2, 3));\n        *((float4*)((buf0+4))) = ((float4)__builtin_shufflevector(alu5, alu5, 4, 5, 6, 7));\n      }\n      int alu9 = ((Lidx1<<5)+alu1);\n      *((float4*)((((float*)(val3))+(alu9+4)))) = (*((float4*)((buf0+4))));\n      *((float4*)((((float*)(val3))+alu9))) = (*((float4*)((buf0+0))));\n    }\n  }\n}\n"
612 Ops.BINARY dtypes.u8 [] b"typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef float float8 __attribute__((aligned(4),ext_vector_type(8)));\n__attribute__((min_vector_width(512))) void r_2_4_8_i(unsigned char* restrict args_48) {\n  float buf0[8];\n  int val0 = (*(((int*)(args_48))+6));\n  int val1 = (*(((int*)(args_48))+7));\n  int val2 = (*(((int*)(args_48))+8));\n  unsigned long val3 = (*(((unsigned long*)(args_48))+0));\n  unsigned long val4 = (*(((unsigned long*)(args_48))+1));\n  unsigned long val5 = (*(((unsigned long*)(args_48))+2));\n  float4 cast0 = (float4){0.0f,0.0f,0.0f,0.0f};\n  for (int Lidx1 = 0; Lidx1 < 2; Lidx1++) {\n    for (int Lidx2 = 0; Lidx2 < (val1-val0); Lidx2++) {\n      int alu0 = (val0+Lidx2);\n      int alu1 = (alu0<<3);\n      *((float4*)((buf0+0))) = cast0;\n      *((float4*)((buf0+4))) = cast0;\n      for (int Lidx0 = 0; Lidx0 < val2; Lidx0++) {\n        float val6 = (*(((float*)(val4))+((alu0*10)+Lidx0+(Lidx1*40))));\n        int alu4 = ((Lidx0<<5)+alu1+(Lidx1*320));\n        float4 val7 = (*((float4*)((((float*)(val5))+(alu4+4)))));\n        float4 val8 = (*((float4*)((((float*)(val5))+alu4))));\n        float4 val9 = (*((float4*)((buf0+0))));\n        float4 val10 = (*((float4*)((buf0+4))));\n        float8 alu5 = __builtin_elementwise_fma((float8){val6,val6,val6,val6,val6,val6,val6,val6}, ((float8)__builtin_shufflevector(val8, val7, 0, 1, 2, 3, 4, 5, 6, 7)), ((float8)__builtin_shufflevector(val9, val10, 0, 1, 2, 3, 4, 5, 6, 7)));\n        *((float4*)((buf0+0))) = ((float4)__builtin_shufflevector(alu5, alu5, 0, 1, 2, 3));\n        *((float4*)((buf0+4))) = ((float4)__builtin_shufflevector(alu5, alu5, 4, 5, 6, 7));\n      }\n      int alu9 = ((Lidx1<<5)+alu1);\n      *((float4*)((((float*)(val3))+(alu9+4)))) = (*((float4*)((buf0+4))));\n      *((float4*)((((float*)(val3))+alu9))) = (*((float4*)((buf0+0))));\n    }\n  }\n}\n"
613 Ops.PARAM dtypes.f32 [37] ParamArg(slot=0, dtype=dtypes.f32)
614 Ops.PARAM dtypes.i32 [101] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 4), name="block_lo", addrspace=AddrSpace.ALU)
615 Ops.PARAM dtypes.i32 [101] ParamArg(slot=4, dtype=dtypes.i32, vmin_vmax=(0, 4), name="block_hi", addrspace=AddrSpace.ALU)
616 Ops.PROGRAM dtypes.void [609, 610, 611, 612] ProgramInfo(global_size=(4, 1, 1), params=(%613, %191, %116, %614, %615, %347), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CPU", arch="x86_64,x86-64"))
617 Ops.BUFFER dtypes.f32 [37] ParamArg(slot=6, dtype=dtypes.f32, device="CPU")
618 Ops.PARAM dtypes.f32 [54] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
619 Ops.CALL dtypes.void [616, 617, 524, 618] CallInfo()
620 Ops.LINEAR dtypes.void [126, 198, 363, 525, 619]

