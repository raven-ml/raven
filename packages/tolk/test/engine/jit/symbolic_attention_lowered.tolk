# tolk D25 D80 D141 D151 golden=15bf2976ef611f2b9aebb1e12b38c3d2
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
140 Ops.INDEX dtypes.f32 [139, 5]
141 Ops.CONST dtypes.weakfloat [] -inf
142 Ops.CAST dtypes.f32 [141] dtypes.f32
143 Ops.STORE dtypes.void [140, 142]
144 Ops.INDEX dtypes.f32 [139, 39]
145 Ops.STORE dtypes.void [144, 142]
146 Ops.INDEX dtypes.f32 [139, 12]
147 Ops.STORE dtypes.void [146, 142]
148 Ops.INDEX dtypes.f32 [139, 47]
149 Ops.STORE dtypes.void [148, 142]
150 Ops.AFTER dtypes.i32 [23, 134]
151 Ops.RANGE dtypes.i32 [150] (AxisType.WEAK, 0)
152 Ops.AFTER dtypes.f32 [138, 143, 145, 147, 149, 151]
153 Ops.INDEX dtypes.f32 [152, 5]
154 Ops.LOAD dtypes.f32 [153]
155 Ops.REFPARAM dtypes.f32 [0, 41] ParamArg(slot=1, dtype=dtypes.f32)
156 Ops.MUL dtypes.i32 [135, 33]
157 Ops.ADD dtypes.i32 [156, 151]
158 Ops.INDEX dtypes.f32 [155, 157]
159 Ops.LOAD dtypes.f32 [158]
160 Ops.CMPLT dtypes.bool [154, 159]
161 Ops.WHERE dtypes.f32 [160, 159, 154]
162 Ops.STORE dtypes.void [153, 161]
163 Ops.INDEX dtypes.f32 [152, 39]
164 Ops.LOAD dtypes.f32 [163]
165 Ops.ADD dtypes.i32 [157, 17]
166 Ops.INDEX dtypes.f32 [155, 165]
167 Ops.LOAD dtypes.f32 [166]
168 Ops.CMPLT dtypes.bool [164, 167]
169 Ops.WHERE dtypes.f32 [168, 167, 164]
170 Ops.STORE dtypes.void [163, 169]
171 Ops.INDEX dtypes.f32 [152, 12]
172 Ops.LOAD dtypes.f32 [171]
173 Ops.CONST dtypes.weakint [] 20
174 Ops.CAST dtypes.i32 [173] dtypes.i32
175 Ops.ADD dtypes.i32 [157, 174]
176 Ops.INDEX dtypes.f32 [155, 175]
177 Ops.LOAD dtypes.f32 [176]
178 Ops.CMPLT dtypes.bool [172, 177]
179 Ops.WHERE dtypes.f32 [178, 177, 172]
180 Ops.STORE dtypes.void [171, 179]
181 Ops.INDEX dtypes.f32 [152, 47]
182 Ops.LOAD dtypes.f32 [181]
183 Ops.CONST dtypes.weakint [] 30
184 Ops.CAST dtypes.i32 [183] dtypes.i32
185 Ops.ADD dtypes.i32 [157, 184]
186 Ops.INDEX dtypes.f32 [155, 185]
187 Ops.LOAD dtypes.f32 [186]
188 Ops.CMPLT dtypes.bool [182, 187]
189 Ops.WHERE dtypes.f32 [188, 187, 182]
190 Ops.STORE dtypes.void [181, 189]
191 Ops.AFTER dtypes.void [101, 162, 170, 180, 190]
192 Ops.END dtypes.void [191, 151]
193 Ops.AFTER dtypes.f32 [138, 192]
194 Ops.INDEX dtypes.f32 [193, 5]
195 Ops.LOAD dtypes.f32 [194]
196 Ops.INDEX dtypes.f32 [193, 39]
197 Ops.LOAD dtypes.f32 [196]
198 Ops.INDEX dtypes.f32 [193, 12]
199 Ops.LOAD dtypes.f32 [198]
200 Ops.INDEX dtypes.f32 [193, 47]
201 Ops.LOAD dtypes.f32 [200]
202 Ops.STACK dtypes.f32 [195, 197, 199, 201]
203 Ops.STORE dtypes.void [137, 202]
204 Ops.AFTER dtypes.void [101, 203]
205 Ops.END dtypes.void [204, 134]
206 Ops.PARAM dtypes.i32 [101] ParamArg(slot=4, dtype=dtypes.i32, vmin_vmax=(1, 10), name="i", addrspace=AddrSpace.ALU)
207 Ops.CONST dtypes.weakint [] 16
208 Ops.MUL dtypes.i32 [206, 207]
209 Ops.CONST dtypes.weakint [] 32
210 Ops.MUL dtypes.i32 [206, 209]
211 Ops.ADD dtypes.i32 [210, 209]
212 Ops.SINK dtypes.void [205] KernelInfo(name="r_2_4_i", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), estimates=Estimates(ops=%208, lds=%211, mem=%211), optimized=True, split=(2, 2, 3))
213 Ops.LINEAR dtypes.void [1, 2, 9, 138, 10, 19, 129, 130, 43, 44, 131, 132, 20, 21, 22, 23, 4, 5, 3, 6, 7, 38, 39, 40, 41, 127, 128, 0, 155, 101, 141, 142, 16, 17, 11, 12, 173, 174, 46, 47, 183, 184, 33, 133, 134, 150, 135, 139, 140, 144, 146, 148, 156, 143, 145, 147, 149, 151, 157, 165, 166, 167, 175, 176, 177, 185, 186, 187, 158, 159, 152, 153, 154, 163, 164, 171, 172, 181, 182, 160, 168, 178, 188, 161, 169, 179, 189, 162, 170, 180, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 136, 137, 203, 204, 205, 212]
214 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_2_4_i(unsigned char* restrict args_40) {\n  float buf0[4];\n  int val0 = (*(((int*)(args_40))+4));\n  int val1 = (*(((int*)(args_40))+5));\n  int val2 = (*(((int*)(args_40))+6));\n  unsigned long val3 = (*(((unsigned long*)(args_40))+0));\n  unsigned long val4 = (*(((unsigned long*)(args_40))+1));\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    *(buf0+0) = ((float)(-__builtin_inff()));\n    *(buf0+1) = ((float)(-__builtin_inff()));\n    *(buf0+2) = ((float)(-__builtin_inff()));\n    *(buf0+3) = ((float)(-__builtin_inff()));\n    for (int Lidx0 = 0; Lidx0 < val2; Lidx0++) {\n      int alu5 = ((alu0*40)+Lidx0);\n      float val5 = (*(((float*)(val4))+(alu5+10)));\n      float val6 = (*(((float*)(val4))+(alu5+20)));\n      float val7 = (*(((float*)(val4))+(alu5+30)));\n      float val8 = (*(((float*)(val4))+alu5));\n      float val9 = (*(buf0+0));\n      float val10 = (*(buf0+1));\n      float val11 = (*(buf0+2));\n      float val12 = (*(buf0+3));\n      float alu6 = ((val9<val8)?val8:val9);\n      float alu7 = ((val10<val5)?val5:val10);\n      float alu8 = ((val11<val6)?val6:val11);\n      float alu9 = ((val12<val7)?val7:val12);\n      *(buf0+0) = alu6;\n      *(buf0+1) = alu7;\n      *(buf0+2) = alu8;\n      *(buf0+3) = alu9;\n    }\n    *((float4*)((((float*)(val3))+(alu0<<2)))) = (float4){(*(buf0+0)),(*(buf0+1)),(*(buf0+2)),(*(buf0+3))};\n  }\n}\n"
215 Ops.BINARY dtypes.u8 [] b"typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_2_4_i(unsigned char* restrict args_40) {\n  float buf0[4];\n  int val0 = (*(((int*)(args_40))+4));\n  int val1 = (*(((int*)(args_40))+5));\n  int val2 = (*(((int*)(args_40))+6));\n  unsigned long val3 = (*(((unsigned long*)(args_40))+0));\n  unsigned long val4 = (*(((unsigned long*)(args_40))+1));\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    *(buf0+0) = ((float)(-__builtin_inff()));\n    *(buf0+1) = ((float)(-__builtin_inff()));\n    *(buf0+2) = ((float)(-__builtin_inff()));\n    *(buf0+3) = ((float)(-__builtin_inff()));\n    for (int Lidx0 = 0; Lidx0 < val2; Lidx0++) {\n      int alu5 = ((alu0*40)+Lidx0);\n      float val5 = (*(((float*)(val4))+(alu5+10)));\n      float val6 = (*(((float*)(val4))+(alu5+20)));\n      float val7 = (*(((float*)(val4))+(alu5+30)));\n      float val8 = (*(((float*)(val4))+alu5));\n      float val9 = (*(buf0+0));\n      float val10 = (*(buf0+1));\n      float val11 = (*(buf0+2));\n      float val12 = (*(buf0+3));\n      float alu6 = ((val9<val8)?val8:val9);\n      float alu7 = ((val10<val5)?val5:val10);\n      float alu8 = ((val11<val6)?val6:val11);\n      float alu9 = ((val12<val7)?val7:val12);\n      *(buf0+0) = alu6;\n      *(buf0+1) = alu7;\n      *(buf0+2) = alu8;\n      *(buf0+3) = alu9;\n    }\n    *((float4*)((((float*)(val3))+(alu0<<2)))) = (float4){(*(buf0+0)),(*(buf0+1)),(*(buf0+2)),(*(buf0+3))};\n  }\n}\n"
216 Ops.PARAM dtypes.f32 [127] ParamArg(slot=0, dtype=dtypes.f32)
217 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
218 Ops.PARAM dtypes.i32 [101] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 2), name="block_lo", addrspace=AddrSpace.ALU)
219 Ops.PARAM dtypes.i32 [101] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 2), name="block_hi", addrspace=AddrSpace.ALU)
220 Ops.PROGRAM dtypes.void [212, 213, 214, 215] ProgramInfo(global_size=(2, 1, 1), params=(%216, %217, %218, %219, %206), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
221 Ops.CONST dtypes.weakint [] 512
222 Ops.SHRINK dtypes.i8 [121, 221, 209]
223 Ops.BITCAST dtypes.f32 [222] dtypes.f32
224 Ops.CALL dtypes.void [220, 223, 123] CallInfo()
225 Ops.CONST dtypes.weakint [] 48
226 Ops.PARAM dtypes.u8 [225] ParamArg(slot=0, dtype=dtypes.u8, name="args")
227 Ops.BITCAST dtypes.u64 [226] dtypes.u64
228 Ops.INDEX dtypes.u64 [227, 5]
229 Ops.LOAD dtypes.u64 [228]
230 Ops.REFPARAM dtypes.f32 [127, 229] ParamArg(slot=0, dtype=dtypes.f32)
231 Ops.BITCAST dtypes.i32 [226] dtypes.i32
232 Ops.INDEX dtypes.i32 [231, 21]
233 Ops.LOAD dtypes.i32 [232]
234 Ops.INDEX dtypes.i32 [231, 25]
235 Ops.LOAD dtypes.i32 [234]
236 Ops.SUB dtypes.i32 [235, 233]
237 Ops.RANGE dtypes.i32 [236] (AxisType.WEAK, 1)
238 Ops.ADD dtypes.i32 [233, 237]
239 Ops.SHL dtypes.i32 [238, 12]
240 Ops.SHRINK dtypes.f32 [230, 239, 10]
241 Ops.AFTER dtypes.f32 [138, 238]
242 Ops.INDEX dtypes.f32 [241, 5]
243 Ops.CONST dtypes.weakfloat [] 0.0
244 Ops.CAST dtypes.f32 [243] dtypes.f32
245 Ops.STORE dtypes.void [242, 244]
246 Ops.INDEX dtypes.f32 [241, 39]
247 Ops.STORE dtypes.void [246, 244]
248 Ops.INDEX dtypes.f32 [241, 12]
249 Ops.STORE dtypes.void [248, 244]
250 Ops.INDEX dtypes.f32 [241, 47]
251 Ops.STORE dtypes.void [250, 244]
252 Ops.CAST dtypes.i32 [127] dtypes.i32
253 Ops.INDEX dtypes.i32 [231, 252]
254 Ops.LOAD dtypes.i32 [253]
255 Ops.AFTER dtypes.i32 [254, 237]
256 Ops.RANGE dtypes.i32 [255] (AxisType.WEAK, 0)
257 Ops.AFTER dtypes.f32 [138, 245, 247, 249, 251, 256]
258 Ops.INDEX dtypes.f32 [257, 5]
259 Ops.LOAD dtypes.f32 [258]
260 Ops.INDEX dtypes.u64 [227, 39]
261 Ops.LOAD dtypes.u64 [260]
262 Ops.REFPARAM dtypes.f32 [0, 261] ParamArg(slot=1, dtype=dtypes.f32)
263 Ops.MUL dtypes.i32 [238, 33]
264 Ops.ADD dtypes.i32 [263, 256]
265 Ops.INDEX dtypes.f32 [262, 264]
266 Ops.LOAD dtypes.f32 [265]
267 Ops.ADD dtypes.i32 [264, 17]
268 Ops.INDEX dtypes.f32 [262, 267]
269 Ops.LOAD dtypes.f32 [268]
270 Ops.ADD dtypes.i32 [264, 174]
271 Ops.INDEX dtypes.f32 [262, 270]
272 Ops.LOAD dtypes.f32 [271]
273 Ops.ADD dtypes.i32 [264, 184]
274 Ops.INDEX dtypes.f32 [262, 273]
275 Ops.LOAD dtypes.f32 [274]
276 Ops.STACK dtypes.f32 [266, 269, 272, 275]
277 Ops.INDEX dtypes.u64 [227, 12]
278 Ops.LOAD dtypes.u64 [277]
279 Ops.REFPARAM dtypes.f32 [127, 278] ParamArg(slot=2, dtype=dtypes.f32)
280 Ops.SHRINK dtypes.f32 [279, 239, 10]
281 Ops.LOAD dtypes.f32 [280]
282 Ops.SUB dtypes.f32 [276, 281]
283 Ops.CONST dtypes.weakfloat [] 1.4426950216293335
284 Ops.CAST dtypes.f32 [283] dtypes.f32
285 Ops.MUL dtypes.f32 [282, 284]
286 Ops.CMPNE dtypes.bool [285, 285]
287 Ops.CONST dtypes.weakfloat [] nan
288 Ops.CAST dtypes.f32 [287] dtypes.f32
289 Ops.CONST dtypes.weakfloat [] -150.0
290 Ops.CAST dtypes.f32 [289] dtypes.f32
291 Ops.CMPLT dtypes.bool [285, 290]
292 Ops.CONST dtypes.weakfloat [] 128.0
293 Ops.CAST dtypes.f32 [292] dtypes.f32
294 Ops.CMPLT dtypes.bool [285, 293]
295 Ops.CONST dtypes.bool [] False
296 Ops.CAST dtypes.bool [295] dtypes.bool
297 Ops.CMPEQ dtypes.bool [294, 296]
298 Ops.CONST dtypes.weakfloat [] inf
299 Ops.CAST dtypes.f32 [298] dtypes.f32
300 Ops.CONST dtypes.weakfloat [] 0.00015359208919107914
301 Ops.CAST dtypes.f32 [300] dtypes.f32
302 Ops.CMPNE dtypes.bool [285, 299]
303 Ops.CMPNE dtypes.bool [285, 142]
304 Ops.WHERE dtypes.f32 [303, 285, 244]
305 Ops.WHERE dtypes.f32 [286, 244, 304]
306 Ops.WHERE dtypes.f32 [302, 305, 244]
307 Ops.CMPLT dtypes.bool [306, 244]
308 Ops.CONST dtypes.weakfloat [] -0.5
309 Ops.CAST dtypes.f32 [308] dtypes.f32
310 Ops.CONST dtypes.weakfloat [] 0.5
311 Ops.CAST dtypes.f32 [310] dtypes.f32
312 Ops.WHERE dtypes.f32 [307, 309, 311]
313 Ops.ADD dtypes.f32 [306, 312]
314 Ops.CAST dtypes.i32 [313] dtypes.i32
315 Ops.CAST dtypes.f32 [314] dtypes.f32
316 Ops.SUB dtypes.f32 [306, 315]
317 Ops.MUL dtypes.f32 [301, 316]
318 Ops.CONST dtypes.weakfloat [] 0.0013392627006396651
319 Ops.CAST dtypes.f32 [318] dtypes.f32
320 Ops.ADD dtypes.f32 [317, 319]
321 Ops.MUL dtypes.f32 [320, 316]
322 Ops.CONST dtypes.weakfloat [] 0.009618384763598442
323 Ops.CAST dtypes.f32 [322] dtypes.f32
324 Ops.ADD dtypes.f32 [321, 323]
325 Ops.MUL dtypes.f32 [324, 316]
326 Ops.CONST dtypes.weakfloat [] 0.055503472685813904
327 Ops.CAST dtypes.f32 [326] dtypes.f32
328 Ops.ADD dtypes.f32 [325, 327]
329 Ops.MUL dtypes.f32 [328, 316]
330 Ops.CONST dtypes.weakfloat [] 0.24022644758224487
331 Ops.CAST dtypes.f32 [330] dtypes.f32
332 Ops.ADD dtypes.f32 [329, 331]
333 Ops.MUL dtypes.f32 [332, 316]
334 Ops.CONST dtypes.weakfloat [] 0.6931471824645996
335 Ops.CAST dtypes.f32 [334] dtypes.f32
336 Ops.ADD dtypes.f32 [333, 335]
337 Ops.MUL dtypes.f32 [336, 316]
338 Ops.CONST dtypes.weakfloat [] 1.0
339 Ops.CAST dtypes.f32 [338] dtypes.f32
340 Ops.ADD dtypes.f32 [337, 339]
341 Ops.SHR dtypes.i32 [314, 39]
342 Ops.CONST dtypes.weakint [] 127
343 Ops.CAST dtypes.i32 [342] dtypes.i32
344 Ops.ADD dtypes.i32 [341, 343]
345 Ops.CAST dtypes.u32 [344] dtypes.u32
346 Ops.CONST dtypes.weakint [] 23
347 Ops.CAST dtypes.u32 [346] dtypes.u32
348 Ops.SHL dtypes.u32 [345, 347]
349 Ops.CAST dtypes.i32 [348] dtypes.i32
350 Ops.BITCAST dtypes.f32 [349] dtypes.f32
351 Ops.MUL dtypes.f32 [340, 350]
352 Ops.CAST dtypes.u32 [314] dtypes.u32
353 Ops.CAST dtypes.u32 [341] dtypes.u32
354 Ops.SUB dtypes.u32 [352, 353]
355 Ops.CAST dtypes.i32 [354] dtypes.i32
356 Ops.CAST dtypes.u32 [355] dtypes.u32
357 Ops.CAST dtypes.u32 [342] dtypes.u32
358 Ops.ADD dtypes.u32 [356, 357]
359 Ops.CAST dtypes.i32 [358] dtypes.i32
360 Ops.CAST dtypes.u32 [359] dtypes.u32
361 Ops.SHL dtypes.u32 [360, 347]
362 Ops.CAST dtypes.i32 [361] dtypes.i32
363 Ops.BITCAST dtypes.f32 [362] dtypes.f32
364 Ops.MUL dtypes.f32 [351, 363]
365 Ops.WHERE dtypes.f32 [297, 299, 364]
366 Ops.WHERE dtypes.f32 [291, 244, 365]
367 Ops.WHERE dtypes.f32 [286, 288, 366]
368 Ops.INDEX dtypes.f32 [367, 5]
369 Ops.ADD dtypes.f32 [259, 368]
370 Ops.STORE dtypes.void [258, 369]
371 Ops.INDEX dtypes.f32 [257, 39]
372 Ops.LOAD dtypes.f32 [371]
373 Ops.INDEX dtypes.f32 [367, 39]
374 Ops.ADD dtypes.f32 [372, 373]
375 Ops.STORE dtypes.void [371, 374]
376 Ops.INDEX dtypes.f32 [257, 12]
377 Ops.LOAD dtypes.f32 [376]
378 Ops.INDEX dtypes.f32 [367, 12]
379 Ops.ADD dtypes.f32 [377, 378]
380 Ops.STORE dtypes.void [376, 379]
381 Ops.INDEX dtypes.f32 [257, 47]
382 Ops.LOAD dtypes.f32 [381]
383 Ops.INDEX dtypes.f32 [367, 47]
384 Ops.ADD dtypes.f32 [382, 383]
385 Ops.STORE dtypes.void [381, 384]
386 Ops.AFTER dtypes.void [101, 370, 375, 380, 385]
387 Ops.END dtypes.void [386, 256]
388 Ops.AFTER dtypes.f32 [138, 387]
389 Ops.INDEX dtypes.f32 [388, 5]
390 Ops.LOAD dtypes.f32 [389]
391 Ops.INDEX dtypes.f32 [388, 39]
392 Ops.LOAD dtypes.f32 [391]
393 Ops.INDEX dtypes.f32 [388, 12]
394 Ops.LOAD dtypes.f32 [393]
395 Ops.INDEX dtypes.f32 [388, 47]
396 Ops.LOAD dtypes.f32 [395]
397 Ops.STACK dtypes.f32 [390, 392, 394, 396]
398 Ops.STORE dtypes.void [240, 397]
399 Ops.AFTER dtypes.void [101, 398]
400 Ops.END dtypes.void [399, 237]
401 Ops.PARAM dtypes.i32 [101] ParamArg(slot=5, dtype=dtypes.i32, vmin_vmax=(1, 10), name="i", addrspace=AddrSpace.ALU)
402 Ops.CONST dtypes.weakint [] 312
403 Ops.MUL dtypes.i32 [401, 402]
404 Ops.MUL dtypes.i32 [401, 209]
405 Ops.ADD dtypes.i32 [404, 37]
406 Ops.SINK dtypes.void [400] KernelInfo(name="r_2_4_i", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), estimates=Estimates(ops=%403, lds=%405, mem=%405), optimized=True, split=(2, 3, 4))
407 Ops.LINEAR dtypes.void [225, 226, 9, 138, 20, 21, 231, 232, 233, 24, 25, 234, 235, 127, 252, 253, 254, 4, 5, 227, 228, 229, 38, 39, 260, 261, 11, 12, 277, 278, 230, 0, 262, 279, 101, 295, 296, 308, 309, 289, 290, 141, 142, 243, 244, 300, 301, 318, 319, 322, 323, 326, 327, 330, 331, 310, 311, 334, 335, 338, 339, 283, 284, 292, 293, 298, 299, 287, 288, 16, 17, 342, 343, 173, 174, 46, 47, 183, 184, 10, 1, 33, 357, 346, 347, 236, 237, 238, 239, 280, 281, 255, 241, 242, 246, 248, 250, 263, 245, 247, 249, 251, 256, 264, 267, 268, 269, 270, 271, 272, 273, 274, 275, 265, 266, 257, 258, 259, 371, 372, 376, 377, 381, 382, 276, 282, 285, 303, 302, 286, 304, 305, 306, 307, 312, 313, 314, 315, 352, 341, 353, 354, 355, 356, 358, 359, 360, 361, 362, 344, 345, 348, 349, 363, 350, 316, 317, 320, 321, 324, 325, 328, 329, 332, 333, 336, 337, 340, 351, 364, 291, 294, 297, 365, 366, 367, 368, 373, 378, 383, 369, 374, 379, 384, 370, 375, 380, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 240, 398, 399, 400, 406]
408 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef unsigned int unsigned_int4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_2_4_i(unsigned char* restrict args_48) {\n  float buf0[4];\n  int val0 = (*(((int*)(args_48))+6));\n  int val1 = (*(((int*)(args_48))+7));\n  int val2 = (*(((int*)(args_48))+8));\n  unsigned long val3 = (*(((unsigned long*)(args_48))+0));\n  unsigned long val4 = (*(((unsigned long*)(args_48))+1));\n  unsigned long val5 = (*(((unsigned long*)(args_48))+2));\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    int alu1 = (alu0<<2);\n    float4 val6 = (*((float4*)((((float*)(val5))+alu1))));\n    *(buf0+0) = 0.0f;\n    *(buf0+1) = 0.0f;\n    *(buf0+2) = 0.0f;\n    *(buf0+3) = 0.0f;\n    for (int Lidx0 = 0; Lidx0 < val2; Lidx0++) {\n      int alu6 = ((alu0*40)+Lidx0);\n      float val7 = (*(((float*)(val4))+(alu6+10)));\n      float val8 = (*(((float*)(val4))+(alu6+20)));\n      float val9 = (*(((float*)(val4))+(alu6+30)));\n      float val10 = (*(((float*)(val4))+alu6));\n      float4 alu7 = (((float4){val10,val7,val8,val9}-val6)*1.4426950216293335f);\n      int4 alu8 = (alu7!=alu7);\n      float4 alu9 = ((alu7!=((float)(-__builtin_inff())))?alu7:0.0f);\n      float4 alu10 = (alu8?0.0f:alu9);\n      float4 alu11 = ((alu7!=((float)(__builtin_inff())))?alu10:0.0f);\n      float4 alu12 = ((alu11<0.0f)?((float4)(-0.5f)):0.5f);\n      int4 cast0 = __builtin_convertvector((alu11+alu12), int4);\n      float4 cast1 = __builtin_convertvector(cast0, float4);\n      unsigned_int4 cast2 = __builtin_convertvector(cast0, unsigned_int4);\n      int4 alu13 = (cast0>>1);\n      unsigned_int4 cast3 = __builtin_convertvector(alu13, unsigned_int4);\n      int4 cast4 = __builtin_convertvector((cast2-cast3), int4);\n      unsigned_int4 cast5 = __builtin_convertvector(cast4, unsigned_int4);\n      int4 cast6 = __builtin_convertvector((cast5+127u), int4);\n      unsigned_int4 cast7 = __builtin_convertvector(cast6, unsigned_int4);\n      int4 cast8 = __builtin_convertvector((cast7<<23u), int4);\n      unsigned_int4 cast9 = __builtin_convertvector((alu13+127), unsigned_int4);\n      int4 cast10 = __builtin_convertvector((cast9<<23u), int4);\n      float4 alu14 = (alu11-cast1);\n      float4 alu15 = ((~((alu7<128.0f)^0))?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu14)+0.0013392627006396651f)*alu14)+0.009618384763598442f)*alu14)+0.055503472685813904f)*alu14)+0.24022644758224487f)*alu14)+0.6931471824645996f)*alu14)+1.0f)*((float4)((int4)(cast10)))*((float4)((int4)(cast8)))));\n      float4 alu16 = ((alu7<-150.0f)?0.0f:alu15);\n      float4 alu17 = (alu8?__builtin_bit_cast(float, (unsigned int)(0x7fc00000u)):alu16);\n      *(buf0+0) = ((*(buf0+0))+alu17[0]);\n      *(buf0+1) = ((*(buf0+1))+alu17[1]);\n      *(buf0+2) = ((*(buf0+2))+alu17[2]);\n      *(buf0+3) = ((*(buf0+3))+alu17[3]);\n    }\n    *((float4*)((((float*)(val3))+alu1))) = (float4){(*(buf0+0)),(*(buf0+1)),(*(buf0+2)),(*(buf0+3))};\n  }\n}\n"
409 Ops.BINARY dtypes.u8 [] b"typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef unsigned int unsigned_int4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_2_4_i(unsigned char* restrict args_48) {\n  float buf0[4];\n  int val0 = (*(((int*)(args_48))+6));\n  int val1 = (*(((int*)(args_48))+7));\n  int val2 = (*(((int*)(args_48))+8));\n  unsigned long val3 = (*(((unsigned long*)(args_48))+0));\n  unsigned long val4 = (*(((unsigned long*)(args_48))+1));\n  unsigned long val5 = (*(((unsigned long*)(args_48))+2));\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    int alu1 = (alu0<<2);\n    float4 val6 = (*((float4*)((((float*)(val5))+alu1))));\n    *(buf0+0) = 0.0f;\n    *(buf0+1) = 0.0f;\n    *(buf0+2) = 0.0f;\n    *(buf0+3) = 0.0f;\n    for (int Lidx0 = 0; Lidx0 < val2; Lidx0++) {\n      int alu6 = ((alu0*40)+Lidx0);\n      float val7 = (*(((float*)(val4))+(alu6+10)));\n      float val8 = (*(((float*)(val4))+(alu6+20)));\n      float val9 = (*(((float*)(val4))+(alu6+30)));\n      float val10 = (*(((float*)(val4))+alu6));\n      float4 alu7 = (((float4){val10,val7,val8,val9}-val6)*1.4426950216293335f);\n      int4 alu8 = (alu7!=alu7);\n      float4 alu9 = ((alu7!=((float)(-__builtin_inff())))?alu7:0.0f);\n      float4 alu10 = (alu8?0.0f:alu9);\n      float4 alu11 = ((alu7!=((float)(__builtin_inff())))?alu10:0.0f);\n      float4 alu12 = ((alu11<0.0f)?((float4)(-0.5f)):0.5f);\n      int4 cast0 = __builtin_convertvector((alu11+alu12), int4);\n      float4 cast1 = __builtin_convertvector(cast0, float4);\n      unsigned_int4 cast2 = __builtin_convertvector(cast0, unsigned_int4);\n      int4 alu13 = (cast0>>1);\n      unsigned_int4 cast3 = __builtin_convertvector(alu13, unsigned_int4);\n      int4 cast4 = __builtin_convertvector((cast2-cast3), int4);\n      unsigned_int4 cast5 = __builtin_convertvector(cast4, unsigned_int4);\n      int4 cast6 = __builtin_convertvector((cast5+127u), int4);\n      unsigned_int4 cast7 = __builtin_convertvector(cast6, unsigned_int4);\n      int4 cast8 = __builtin_convertvector((cast7<<23u), int4);\n      unsigned_int4 cast9 = __builtin_convertvector((alu13+127), unsigned_int4);\n      int4 cast10 = __builtin_convertvector((cast9<<23u), int4);\n      float4 alu14 = (alu11-cast1);\n      float4 alu15 = ((~((alu7<128.0f)^0))?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu14)+0.0013392627006396651f)*alu14)+0.009618384763598442f)*alu14)+0.055503472685813904f)*alu14)+0.24022644758224487f)*alu14)+0.6931471824645996f)*alu14)+1.0f)*((float4)((int4)(cast10)))*((float4)((int4)(cast8)))));\n      float4 alu16 = ((alu7<-150.0f)?0.0f:alu15);\n      float4 alu17 = (alu8?__builtin_bit_cast(float, (unsigned int)(0x7fc00000u)):alu16);\n      *(buf0+0) = ((*(buf0+0))+alu17[0]);\n      *(buf0+1) = ((*(buf0+1))+alu17[1]);\n      *(buf0+2) = ((*(buf0+2))+alu17[2]);\n      *(buf0+3) = ((*(buf0+3))+alu17[3]);\n    }\n    *((float4*)((((float*)(val3))+alu1))) = (float4){(*(buf0+0)),(*(buf0+1)),(*(buf0+2)),(*(buf0+3))};\n  }\n}\n"
410 Ops.PARAM dtypes.f32 [127] ParamArg(slot=2, dtype=dtypes.f32)
411 Ops.PARAM dtypes.i32 [101] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 2), name="block_lo", addrspace=AddrSpace.ALU)
412 Ops.PARAM dtypes.i32 [101] ParamArg(slot=4, dtype=dtypes.i32, vmin_vmax=(0, 2), name="block_hi", addrspace=AddrSpace.ALU)
413 Ops.PROGRAM dtypes.void [406, 407, 408, 409] ProgramInfo(global_size=(2, 1, 1), params=(%216, %217, %410, %411, %412, %401), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CPU", arch="x86_64,x86-64"))
414 Ops.CONST dtypes.weakint [] 768
415 Ops.SHRINK dtypes.i8 [121, 414, 209]
416 Ops.BITCAST dtypes.f32 [415] dtypes.f32
417 Ops.CALL dtypes.void [413, 416, 123, 223] CallInfo()
418 Ops.REFPARAM dtypes.f32 [0, 229] ParamArg(slot=0, dtype=dtypes.f32)
419 Ops.CONST dtypes.weakint [] 9
420 Ops.CAST dtypes.i32 [419] dtypes.i32
421 Ops.INDEX dtypes.i32 [231, 420]
422 Ops.LOAD dtypes.i32 [421]
423 Ops.SUB dtypes.i32 [422, 254]
424 Ops.RANGE dtypes.i32 [423] (AxisType.WEAK, 2)
425 Ops.ADD dtypes.i32 [254, 424]
426 Ops.INDEX dtypes.f32 [418, 425]
427 Ops.INDEX dtypes.f32 [262, 425]
428 Ops.LOAD dtypes.f32 [427]
429 Ops.ADD dtypes.i32 [425, 17]
430 Ops.INDEX dtypes.f32 [262, 429]
431 Ops.LOAD dtypes.f32 [430]
432 Ops.ADD dtypes.i32 [425, 174]
433 Ops.INDEX dtypes.f32 [262, 432]
434 Ops.LOAD dtypes.f32 [433]
435 Ops.ADD dtypes.i32 [425, 184]
436 Ops.INDEX dtypes.f32 [262, 435]
437 Ops.LOAD dtypes.f32 [436]
438 Ops.ADD dtypes.i32 [425, 33]
439 Ops.INDEX dtypes.f32 [262, 438]
440 Ops.LOAD dtypes.f32 [439]
441 Ops.CONST dtypes.weakint [] 50
442 Ops.CAST dtypes.i32 [441] dtypes.i32
443 Ops.ADD dtypes.i32 [425, 442]
444 Ops.INDEX dtypes.f32 [262, 443]
445 Ops.LOAD dtypes.f32 [444]
446 Ops.CONST dtypes.weakint [] 60
447 Ops.CAST dtypes.i32 [446] dtypes.i32
448 Ops.ADD dtypes.i32 [425, 447]
449 Ops.INDEX dtypes.f32 [262, 448]
450 Ops.LOAD dtypes.f32 [449]
451 Ops.CONST dtypes.weakint [] 70
452 Ops.CAST dtypes.i32 [451] dtypes.i32
453 Ops.ADD dtypes.i32 [425, 452]
454 Ops.INDEX dtypes.f32 [262, 453]
455 Ops.LOAD dtypes.f32 [454]
456 Ops.STACK dtypes.f32 [428, 431, 434, 437, 440, 445, 450, 455]
457 Ops.SHRINK dtypes.f32 [279, 5, 10]
458 Ops.LOAD dtypes.f32 [457]
459 Ops.INDEX dtypes.f32 [458, 5]
460 Ops.INDEX dtypes.f32 [458, 39]
461 Ops.INDEX dtypes.f32 [458, 12]
462 Ops.INDEX dtypes.f32 [458, 47]
463 Ops.SHRINK dtypes.f32 [279, 10, 10]
464 Ops.LOAD dtypes.f32 [463]
465 Ops.INDEX dtypes.f32 [464, 5]
466 Ops.INDEX dtypes.f32 [464, 39]
467 Ops.INDEX dtypes.f32 [464, 12]
468 Ops.INDEX dtypes.f32 [464, 47]
469 Ops.STACK dtypes.f32 [459, 460, 461, 462, 465, 466, 467, 468]
470 Ops.SUB dtypes.f32 [456, 469]
471 Ops.MUL dtypes.f32 [470, 284]
472 Ops.CMPNE dtypes.bool [471, 471]
473 Ops.CMPLT dtypes.bool [471, 290]
474 Ops.CMPLT dtypes.bool [471, 293]
475 Ops.CMPEQ dtypes.bool [474, 296]
476 Ops.CMPNE dtypes.bool [471, 299]
477 Ops.CMPNE dtypes.bool [471, 142]
478 Ops.WHERE dtypes.f32 [477, 471, 244]
479 Ops.WHERE dtypes.f32 [472, 244, 478]
480 Ops.WHERE dtypes.f32 [476, 479, 244]
481 Ops.CMPLT dtypes.bool [480, 244]
482 Ops.WHERE dtypes.f32 [481, 309, 311]
483 Ops.ADD dtypes.f32 [480, 482]
484 Ops.CAST dtypes.i32 [483] dtypes.i32
485 Ops.CAST dtypes.f32 [484] dtypes.f32
486 Ops.SUB dtypes.f32 [480, 485]
487 Ops.MUL dtypes.f32 [301, 486]
488 Ops.ADD dtypes.f32 [487, 319]
489 Ops.MUL dtypes.f32 [488, 486]
490 Ops.ADD dtypes.f32 [489, 323]
491 Ops.MUL dtypes.f32 [490, 486]
492 Ops.ADD dtypes.f32 [491, 327]
493 Ops.MUL dtypes.f32 [492, 486]
494 Ops.ADD dtypes.f32 [493, 331]
495 Ops.MUL dtypes.f32 [494, 486]
496 Ops.ADD dtypes.f32 [495, 335]
497 Ops.MUL dtypes.f32 [496, 486]
498 Ops.ADD dtypes.f32 [497, 339]
499 Ops.SHR dtypes.i32 [484, 39]
500 Ops.ADD dtypes.i32 [499, 343]
501 Ops.CAST dtypes.u32 [500] dtypes.u32
502 Ops.SHL dtypes.u32 [501, 347]
503 Ops.CAST dtypes.i32 [502] dtypes.i32
504 Ops.BITCAST dtypes.f32 [503] dtypes.f32
505 Ops.MUL dtypes.f32 [498, 504]
506 Ops.CAST dtypes.u32 [484] dtypes.u32
507 Ops.CAST dtypes.u32 [499] dtypes.u32
508 Ops.SUB dtypes.u32 [506, 507]
509 Ops.CAST dtypes.i32 [508] dtypes.i32
510 Ops.CAST dtypes.u32 [509] dtypes.u32
511 Ops.ADD dtypes.u32 [510, 357]
512 Ops.CAST dtypes.i32 [511] dtypes.i32
513 Ops.CAST dtypes.u32 [512] dtypes.u32
514 Ops.SHL dtypes.u32 [513, 347]
515 Ops.CAST dtypes.i32 [514] dtypes.i32
516 Ops.BITCAST dtypes.f32 [515] dtypes.f32
517 Ops.MUL dtypes.f32 [505, 516]
518 Ops.WHERE dtypes.f32 [475, 299, 517]
519 Ops.WHERE dtypes.f32 [473, 244, 518]
520 Ops.WHERE dtypes.f32 [472, 288, 519]
521 Ops.INDEX dtypes.u64 [227, 47]
522 Ops.LOAD dtypes.u64 [521]
523 Ops.REFPARAM dtypes.f32 [127, 522] ParamArg(slot=3, dtype=dtypes.f32)
524 Ops.SHRINK dtypes.f32 [523, 5, 10]
525 Ops.LOAD dtypes.f32 [524]
526 Ops.INDEX dtypes.f32 [525, 5]
527 Ops.INDEX dtypes.f32 [525, 39]
528 Ops.INDEX dtypes.f32 [525, 12]
529 Ops.INDEX dtypes.f32 [525, 47]
530 Ops.SHRINK dtypes.f32 [523, 10, 10]
531 Ops.LOAD dtypes.f32 [530]
532 Ops.INDEX dtypes.f32 [531, 5]
533 Ops.INDEX dtypes.f32 [531, 39]
534 Ops.INDEX dtypes.f32 [531, 12]
535 Ops.INDEX dtypes.f32 [531, 47]
536 Ops.STACK dtypes.f32 [526, 527, 528, 529, 532, 533, 534, 535]
537 Ops.FDIV dtypes.f32 [339, 536]
538 Ops.MUL dtypes.f32 [520, 537]
539 Ops.INDEX dtypes.f32 [538, 5]
540 Ops.STORE dtypes.void [426, 539]
541 Ops.INDEX dtypes.f32 [418, 429]
542 Ops.INDEX dtypes.f32 [538, 39]
543 Ops.STORE dtypes.void [541, 542]
544 Ops.INDEX dtypes.f32 [418, 432]
545 Ops.INDEX dtypes.f32 [538, 12]
546 Ops.STORE dtypes.void [544, 545]
547 Ops.INDEX dtypes.f32 [418, 435]
548 Ops.INDEX dtypes.f32 [538, 47]
549 Ops.STORE dtypes.void [547, 548]
550 Ops.INDEX dtypes.f32 [418, 438]
551 Ops.INDEX dtypes.f32 [538, 10]
552 Ops.STORE dtypes.void [550, 551]
553 Ops.INDEX dtypes.f32 [418, 443]
554 Ops.INDEX dtypes.f32 [538, 44]
555 Ops.STORE dtypes.void [553, 554]
556 Ops.INDEX dtypes.f32 [418, 448]
557 Ops.INDEX dtypes.f32 [538, 21]
558 Ops.STORE dtypes.void [556, 557]
559 Ops.INDEX dtypes.f32 [418, 453]
560 Ops.INDEX dtypes.f32 [538, 25]
561 Ops.STORE dtypes.void [559, 560]
562 Ops.AFTER dtypes.void [101, 540, 543, 546, 549, 552, 555, 558, 561]
563 Ops.END dtypes.void [562, 424]
564 Ops.MUL dtypes.i32 [103, 402]
565 Ops.ADD dtypes.i32 [127, 564]
566 Ops.MUL dtypes.i32 [103, 37]
567 Ops.ADD dtypes.i32 [37, 566]
568 Ops.SINK dtypes.void [563] KernelInfo(name="E_i_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)),), estimates=Estimates(ops=%565, lds=%567, mem=%567), optimized=True, split=(%102, 4, 5))
569 Ops.LINEAR dtypes.void [225, 226, 127, 252, 231, 253, 254, 419, 420, 421, 422, 4, 5, 227, 228, 229, 38, 39, 260, 261, 11, 12, 277, 278, 46, 47, 521, 522, 279, 9, 10, 457, 458, 463, 464, 523, 524, 525, 530, 531, 0, 418, 262, 101, 459, 460, 461, 462, 465, 466, 467, 468, 469, 526, 527, 528, 529, 532, 533, 534, 535, 536, 295, 296, 308, 309, 289, 290, 141, 142, 243, 244, 300, 301, 318, 319, 322, 323, 326, 327, 330, 331, 310, 311, 334, 335, 338, 339, 283, 284, 292, 293, 298, 299, 287, 288, 16, 17, 342, 343, 173, 174, 183, 184, 1, 33, 43, 44, 441, 442, 20, 21, 446, 447, 24, 25, 451, 452, 357, 346, 347, 423, 537, 424, 425, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 443, 444, 445, 448, 449, 450, 453, 454, 455, 456, 426, 541, 544, 547, 550, 553, 556, 559, 470, 471, 477, 476, 472, 478, 479, 480, 481, 482, 483, 484, 485, 506, 499, 507, 508, 509, 510, 511, 512, 513, 514, 515, 500, 501, 502, 503, 516, 504, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 505, 517, 473, 474, 475, 518, 519, 520, 538, 539, 542, 545, 548, 551, 554, 557, 560, 540, 543, 546, 549, 552, 555, 558, 561, 562, 563, 568]
570 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef float float8 __attribute__((aligned(4),ext_vector_type(8)));\ntypedef int int8 __attribute__((aligned(4),ext_vector_type(8)));\ntypedef unsigned int unsigned_int8 __attribute__((aligned(4),ext_vector_type(8)));\n__attribute__((min_vector_width(512))) void E_i_8(unsigned char* restrict args_48) {\n  int val0 = (*(((int*)(args_48))+8));\n  int val1 = (*(((int*)(args_48))+9));\n  unsigned long val2 = (*(((unsigned long*)(args_48))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_48))+1));\n  unsigned long val4 = (*(((unsigned long*)(args_48))+2));\n  unsigned long val5 = (*(((unsigned long*)(args_48))+3));\n  float4 val6 = (*((float4*)((((float*)(val4))+0))));\n  float4 val7 = (*((float4*)((((float*)(val4))+4))));\n  float4 val8 = (*((float4*)((((float*)(val5))+0))));\n  float4 val9 = (*((float4*)((((float*)(val5))+4))));\n  for (int Lidx2 = 0; Lidx2 < (val1-val0); Lidx2++) {\n    int alu0 = (val0+Lidx2);\n    float val10 = (*(((float*)(val3))+alu0));\n    int alu1 = (alu0+10);\n    float val11 = (*(((float*)(val3))+alu1));\n    int alu2 = (alu0+20);\n    float val12 = (*(((float*)(val3))+alu2));\n    int alu3 = (alu0+30);\n    float val13 = (*(((float*)(val3))+alu3));\n    int alu4 = (alu0+40);\n    float val14 = (*(((float*)(val3))+alu4));\n    int alu5 = (alu0+50);\n    float val15 = (*(((float*)(val3))+alu5));\n    int alu6 = (alu0+60);\n    float val16 = (*(((float*)(val3))+alu6));\n    int alu7 = (alu0+70);\n    float val17 = (*(((float*)(val3))+alu7));\n    float8 alu8 = (((float8){val10,val11,val12,val13,val14,val15,val16,val17}-(float8){val6[0],val6[1],val6[2],val6[3],val7[0],val7[1],val7[2],val7[3]})*1.4426950216293335f);\n    int8 alu9 = (alu8!=alu8);\n    float8 alu10 = ((alu8!=((float)(-__builtin_inff())))?alu8:0.0f);\n    float8 alu11 = (alu9?0.0f:alu10);\n    float8 alu12 = ((alu8!=((float)(__builtin_inff())))?alu11:0.0f);\n    float8 alu13 = ((alu12<0.0f)?((float8)(-0.5f)):0.5f);\n    int8 cast0 = __builtin_convertvector((alu12+alu13), int8);\n    float8 cast1 = __builtin_convertvector(cast0, float8);\n    unsigned_int8 cast2 = __builtin_convertvector(cast0, unsigned_int8);\n    int8 alu14 = (cast0>>1);\n    unsigned_int8 cast3 = __builtin_convertvector(alu14, unsigned_int8);\n    int8 cast4 = __builtin_convertvector((cast2-cast3), int8);\n    unsigned_int8 cast5 = __builtin_convertvector(cast4, unsigned_int8);\n    int8 cast6 = __builtin_convertvector((cast5+127u), int8);\n    unsigned_int8 cast7 = __builtin_convertvector(cast6, unsigned_int8);\n    int8 cast8 = __builtin_convertvector((cast7<<23u), int8);\n    unsigned_int8 cast9 = __builtin_convertvector((alu14+127), unsigned_int8);\n    int8 cast10 = __builtin_convertvector((cast9<<23u), int8);\n    float8 alu15 = (alu12-cast1);\n    float8 alu16 = ((~((alu8<128.0f)^0))?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu15)+0.0013392627006396651f)*alu15)+0.009618384763598442f)*alu15)+0.055503472685813904f)*alu15)+0.24022644758224487f)*alu15)+0.6931471824645996f)*alu15)+1.0f)*((float8)((int8)(cast10)))*((float8)((int8)(cast8)))));\n    float8 alu17 = ((alu8<-150.0f)?0.0f:alu16);\n    float8 alu18 = (alu9?__builtin_bit_cast(float, (unsigned int)(0x7fc00000u)):alu17);\n    float8 alu19 = (alu18*(1.0f/(float8){val8[0],val8[1],val8[2],val8[3],val9[0],val9[1],val9[2],val9[3]}));\n    *(((float*)(val2))+alu0) = alu19[0];\n    *(((float*)(val2))+alu1) = alu19[1];\n    *(((float*)(val2))+alu2) = alu19[2];\n    *(((float*)(val2))+alu3) = alu19[3];\n    *(((float*)(val2))+alu4) = alu19[4];\n    *(((float*)(val2))+alu5) = alu19[5];\n    *(((float*)(val2))+alu6) = alu19[6];\n    *(((float*)(val2))+alu7) = alu19[7];\n  }\n}\n"
571 Ops.BINARY dtypes.u8 [] b"typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef float float8 __attribute__((aligned(4),ext_vector_type(8)));\ntypedef int int8 __attribute__((aligned(4),ext_vector_type(8)));\ntypedef unsigned int unsigned_int8 __attribute__((aligned(4),ext_vector_type(8)));\n__attribute__((min_vector_width(512))) void E_i_8(unsigned char* restrict args_48) {\n  int val0 = (*(((int*)(args_48))+8));\n  int val1 = (*(((int*)(args_48))+9));\n  unsigned long val2 = (*(((unsigned long*)(args_48))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_48))+1));\n  unsigned long val4 = (*(((unsigned long*)(args_48))+2));\n  unsigned long val5 = (*(((unsigned long*)(args_48))+3));\n  float4 val6 = (*((float4*)((((float*)(val4))+0))));\n  float4 val7 = (*((float4*)((((float*)(val4))+4))));\n  float4 val8 = (*((float4*)((((float*)(val5))+0))));\n  float4 val9 = (*((float4*)((((float*)(val5))+4))));\n  for (int Lidx2 = 0; Lidx2 < (val1-val0); Lidx2++) {\n    int alu0 = (val0+Lidx2);\n    float val10 = (*(((float*)(val3))+alu0));\n    int alu1 = (alu0+10);\n    float val11 = (*(((float*)(val3))+alu1));\n    int alu2 = (alu0+20);\n    float val12 = (*(((float*)(val3))+alu2));\n    int alu3 = (alu0+30);\n    float val13 = (*(((float*)(val3))+alu3));\n    int alu4 = (alu0+40);\n    float val14 = (*(((float*)(val3))+alu4));\n    int alu5 = (alu0+50);\n    float val15 = (*(((float*)(val3))+alu5));\n    int alu6 = (alu0+60);\n    float val16 = (*(((float*)(val3))+alu6));\n    int alu7 = (alu0+70);\n    float val17 = (*(((float*)(val3))+alu7));\n    float8 alu8 = (((float8){val10,val11,val12,val13,val14,val15,val16,val17}-(float8){val6[0],val6[1],val6[2],val6[3],val7[0],val7[1],val7[2],val7[3]})*1.4426950216293335f);\n    int8 alu9 = (alu8!=alu8);\n    float8 alu10 = ((alu8!=((float)(-__builtin_inff())))?alu8:0.0f);\n    float8 alu11 = (alu9?0.0f:alu10);\n    float8 alu12 = ((alu8!=((float)(__builtin_inff())))?alu11:0.0f);\n    float8 alu13 = ((alu12<0.0f)?((float8)(-0.5f)):0.5f);\n    int8 cast0 = __builtin_convertvector((alu12+alu13), int8);\n    float8 cast1 = __builtin_convertvector(cast0, float8);\n    unsigned_int8 cast2 = __builtin_convertvector(cast0, unsigned_int8);\n    int8 alu14 = (cast0>>1);\n    unsigned_int8 cast3 = __builtin_convertvector(alu14, unsigned_int8);\n    int8 cast4 = __builtin_convertvector((cast2-cast3), int8);\n    unsigned_int8 cast5 = __builtin_convertvector(cast4, unsigned_int8);\n    int8 cast6 = __builtin_convertvector((cast5+127u), int8);\n    unsigned_int8 cast7 = __builtin_convertvector(cast6, unsigned_int8);\n    int8 cast8 = __builtin_convertvector((cast7<<23u), int8);\n    unsigned_int8 cast9 = __builtin_convertvector((alu14+127), unsigned_int8);\n    int8 cast10 = __builtin_convertvector((cast9<<23u), int8);\n    float8 alu15 = (alu12-cast1);\n    float8 alu16 = ((~((alu8<128.0f)^0))?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu15)+0.0013392627006396651f)*alu15)+0.009618384763598442f)*alu15)+0.055503472685813904f)*alu15)+0.24022644758224487f)*alu15)+0.6931471824645996f)*alu15)+1.0f)*((float8)((int8)(cast10)))*((float8)((int8)(cast8)))));\n    float8 alu17 = ((alu8<-150.0f)?0.0f:alu16);\n    float8 alu18 = (alu9?__builtin_bit_cast(float, (unsigned int)(0x7fc00000u)):alu17);\n    float8 alu19 = (alu18*(1.0f/(float8){val8[0],val8[1],val8[2],val8[3],val9[0],val9[1],val9[2],val9[3]}));\n    *(((float*)(val2))+alu0) = alu19[0];\n    *(((float*)(val2))+alu1) = alu19[1];\n    *(((float*)(val2))+alu2) = alu19[2];\n    *(((float*)(val2))+alu3) = alu19[3];\n    *(((float*)(val2))+alu4) = alu19[4];\n    *(((float*)(val2))+alu5) = alu19[5];\n    *(((float*)(val2))+alu6) = alu19[6];\n    *(((float*)(val2))+alu7) = alu19[7];\n  }\n}\n"
572 Ops.PARAM dtypes.f32 [127] ParamArg(slot=3, dtype=dtypes.f32)
573 Ops.PARAM dtypes.i32 [101] ParamArg(slot=4, dtype=dtypes.i32, vmin_vmax=(0, 10), name="block_lo", addrspace=AddrSpace.ALU)
574 Ops.PARAM dtypes.i32 [101] ParamArg(slot=5, dtype=dtypes.i32, vmin_vmax=(0, 10), name="block_hi", addrspace=AddrSpace.ALU)
575 Ops.PROGRAM dtypes.void [568, 569, 570, 571] ProgramInfo(global_size=(%102, 1, 1), params=(%114, %217, %410, %572, %573, %574), globals=(0, 1, 2, 3), outs=(0,), ins=(1, 2, 3), target=Target(device="CPU", arch="x86_64,x86-64"))
576 Ops.CONST dtypes.weakint [] 1024
577 Ops.SHRINK dtypes.i8 [121, 576, 60]
578 Ops.BITCAST dtypes.f32 [577] dtypes.f32
579 Ops.CALL dtypes.void [575, 578, 123, 223, 416] CallInfo()
580 Ops.REFPARAM dtypes.f32 [37, 229] ParamArg(slot=0, dtype=dtypes.f32)
581 Ops.AFTER dtypes.i32 [236, 13]
582 Ops.RANGE dtypes.i32 [581] (AxisType.WEAK, 2)
583 Ops.ADD dtypes.i32 [233, 582]
584 Ops.SHL dtypes.i32 [583, 47]
585 Ops.ADD dtypes.i32 [45, 584]
586 Ops.SHRINK dtypes.f32 [580, 585, 10]
587 Ops.BUFFER dtypes.f32 [127] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
588 Ops.AFTER dtypes.f32 [587, 583, 13]
589 Ops.INDEX dtypes.f32 [588, 5]
590 Ops.STORE dtypes.void [589, 244]
591 Ops.INDEX dtypes.f32 [588, 39]
592 Ops.STORE dtypes.void [591, 244]
593 Ops.INDEX dtypes.f32 [588, 12]
594 Ops.STORE dtypes.void [593, 244]
595 Ops.INDEX dtypes.f32 [588, 47]
596 Ops.STORE dtypes.void [595, 244]
597 Ops.INDEX dtypes.f32 [588, 10]
598 Ops.STORE dtypes.void [597, 244]
599 Ops.INDEX dtypes.f32 [588, 44]
600 Ops.STORE dtypes.void [599, 244]
601 Ops.INDEX dtypes.f32 [588, 21]
602 Ops.STORE dtypes.void [601, 244]
603 Ops.INDEX dtypes.f32 [588, 25]
604 Ops.STORE dtypes.void [603, 244]
605 Ops.AFTER dtypes.i32 [254, 582]
606 Ops.RANGE dtypes.i32 [605] (AxisType.WEAK, 0)
607 Ops.AFTER dtypes.f32 [587, 590, 592, 594, 596, 598, 600, 602, 604, 606]
608 Ops.INDEX dtypes.f32 [607, 5]
609 Ops.MUL dtypes.i32 [583, 17]
610 Ops.ADD dtypes.i32 [609, 606]
611 Ops.ADD dtypes.i32 [610, 34]
612 Ops.INDEX dtypes.f32 [262, 611]
613 Ops.LOAD dtypes.f32 [612]
614 Ops.REFPARAM dtypes.f32 [54, 278] ParamArg(slot=2, dtype=dtypes.f32)
615 Ops.SHL dtypes.i32 [606, 44]
616 Ops.ADD dtypes.i32 [615, 584]
617 Ops.ADD dtypes.i32 [616, 62]
618 Ops.SHRINK dtypes.f32 [614, 617, 10]
619 Ops.LOAD dtypes.f32 [618]
620 Ops.INDEX dtypes.f32 [619, 5]
621 Ops.LOAD dtypes.f32 [608]
622 Ops.MULACC dtypes.f32 [613, 620, 621]
623 Ops.STORE dtypes.void [608, 622]
624 Ops.INDEX dtypes.f32 [607, 39]
625 Ops.INDEX dtypes.f32 [619, 39]
626 Ops.LOAD dtypes.f32 [624]
627 Ops.MULACC dtypes.f32 [613, 625, 626]
628 Ops.STORE dtypes.void [624, 627]
629 Ops.INDEX dtypes.f32 [607, 12]
630 Ops.INDEX dtypes.f32 [619, 12]
631 Ops.LOAD dtypes.f32 [629]
632 Ops.MULACC dtypes.f32 [613, 630, 631]
633 Ops.STORE dtypes.void [629, 632]
634 Ops.INDEX dtypes.f32 [607, 47]
635 Ops.INDEX dtypes.f32 [619, 47]
636 Ops.LOAD dtypes.f32 [634]
637 Ops.MULACC dtypes.f32 [613, 635, 636]
638 Ops.STORE dtypes.void [634, 637]
639 Ops.INDEX dtypes.f32 [607, 10]
640 Ops.ADD dtypes.i32 [617, 10]
641 Ops.SHRINK dtypes.f32 [614, 640, 10]
642 Ops.LOAD dtypes.f32 [641]
643 Ops.INDEX dtypes.f32 [642, 5]
644 Ops.LOAD dtypes.f32 [639]
645 Ops.MULACC dtypes.f32 [613, 643, 644]
646 Ops.STORE dtypes.void [639, 645]
647 Ops.INDEX dtypes.f32 [607, 44]
648 Ops.INDEX dtypes.f32 [642, 39]
649 Ops.LOAD dtypes.f32 [647]
650 Ops.MULACC dtypes.f32 [613, 648, 649]
651 Ops.STORE dtypes.void [647, 650]
652 Ops.INDEX dtypes.f32 [607, 21]
653 Ops.INDEX dtypes.f32 [642, 12]
654 Ops.LOAD dtypes.f32 [652]
655 Ops.MULACC dtypes.f32 [613, 653, 654]
656 Ops.STORE dtypes.void [652, 655]
657 Ops.INDEX dtypes.f32 [607, 25]
658 Ops.INDEX dtypes.f32 [642, 47]
659 Ops.LOAD dtypes.f32 [657]
660 Ops.MULACC dtypes.f32 [613, 658, 659]
661 Ops.STORE dtypes.void [657, 660]
662 Ops.AFTER dtypes.void [101, 623, 628, 633, 638, 646, 651, 656, 661]
663 Ops.END dtypes.void [662, 606]
664 Ops.AFTER dtypes.f32 [587, 663]
665 Ops.INDEX dtypes.f32 [664, 5]
666 Ops.LOAD dtypes.f32 [665]
667 Ops.INDEX dtypes.f32 [664, 39]
668 Ops.LOAD dtypes.f32 [667]
669 Ops.INDEX dtypes.f32 [664, 12]
670 Ops.LOAD dtypes.f32 [669]
671 Ops.INDEX dtypes.f32 [664, 47]
672 Ops.LOAD dtypes.f32 [671]
673 Ops.STACK dtypes.f32 [666, 668, 670, 672]
674 Ops.STORE dtypes.void [586, 673]
675 Ops.ADD dtypes.i32 [585, 10]
676 Ops.SHRINK dtypes.f32 [580, 675, 10]
677 Ops.INDEX dtypes.f32 [664, 10]
678 Ops.LOAD dtypes.f32 [677]
679 Ops.INDEX dtypes.f32 [664, 44]
680 Ops.LOAD dtypes.f32 [679]
681 Ops.INDEX dtypes.f32 [664, 21]
682 Ops.LOAD dtypes.f32 [681]
683 Ops.INDEX dtypes.f32 [664, 25]
684 Ops.LOAD dtypes.f32 [683]
685 Ops.STACK dtypes.f32 [678, 680, 682, 684]
686 Ops.STORE dtypes.void [676, 685]
687 Ops.AFTER dtypes.void [101, 674, 686]
688 Ops.END dtypes.void [687, 582]
689 Ops.END dtypes.void [688, 13]
690 Ops.MUL dtypes.i32 [401, 104]
691 Ops.MUL dtypes.i32 [401, 107]
692 Ops.ADD dtypes.i32 [691, 106]
693 Ops.SINK dtypes.void [689] KernelInfo(name="r_2_4_8_i", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(8, AxisType.UPCAST)),), estimates=Estimates(ops=%690, lds=%692, mem=%692), optimized=True, split=(4, 3, 4))
694 Ops.LINEAR dtypes.void [225, 226, 127, 587, 20, 21, 231, 232, 233, 24, 25, 234, 235, 252, 253, 254, 4, 5, 227, 228, 229, 38, 39, 260, 261, 11, 12, 277, 278, 37, 580, 0, 262, 54, 614, 101, 243, 244, 16, 17, 46, 47, 60, 61, 9, 10, 1, 33, 43, 44, 236, 13, 581, 62, 34, 45, 582, 605, 583, 588, 589, 591, 593, 595, 597, 599, 601, 603, 609, 584, 590, 592, 594, 596, 598, 600, 602, 604, 606, 610, 611, 612, 613, 607, 608, 621, 624, 626, 629, 631, 634, 636, 639, 644, 647, 649, 652, 654, 657, 659, 615, 616, 617, 640, 641, 642, 618, 619, 643, 648, 653, 658, 620, 625, 630, 635, 645, 650, 655, 660, 622, 627, 632, 637, 623, 628, 633, 638, 646, 651, 656, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 677, 678, 679, 680, 681, 682, 683, 684, 673, 685, 585, 675, 676, 586, 686, 674, 687, 688, 689, 693]
695 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_2_4_8_i(unsigned char* restrict args_48) {\n  float buf0[8];\n  int val0 = (*(((int*)(args_48))+6));\n  int val1 = (*(((int*)(args_48))+7));\n  int val2 = (*(((int*)(args_48))+8));\n  unsigned long val3 = (*(((unsigned long*)(args_48))+0));\n  unsigned long val4 = (*(((unsigned long*)(args_48))+1));\n  unsigned long val5 = (*(((unsigned long*)(args_48))+2));\n  for (int Lidx1 = 0; Lidx1 < 2; Lidx1++) {\n    for (int Lidx2 = 0; Lidx2 < (val1-val0); Lidx2++) {\n      int alu0 = (val0+Lidx2);\n      int alu1 = (alu0<<3);\n      *(buf0+0) = 0.0f;\n      *(buf0+1) = 0.0f;\n      *(buf0+2) = 0.0f;\n      *(buf0+3) = 0.0f;\n      *(buf0+4) = 0.0f;\n      *(buf0+5) = 0.0f;\n      *(buf0+6) = 0.0f;\n      *(buf0+7) = 0.0f;\n      for (int Lidx0 = 0; Lidx0 < val2; Lidx0++) {\n        float val6 = (*(((float*)(val4))+((alu0*10)+Lidx0+(Lidx1*40))));\n        int alu10 = ((Lidx0<<5)+alu1+(Lidx1*320));\n        float4 val7 = (*((float4*)((((float*)(val5))+(alu10+4)))));\n        float4 val8 = (*((float4*)((((float*)(val5))+alu10))));\n        *(buf0+0) = __builtin_fmaf(val6,val8[0],(*(buf0+0)));\n        *(buf0+1) = __builtin_fmaf(val6,val8[1],(*(buf0+1)));\n        *(buf0+2) = __builtin_fmaf(val6,val8[2],(*(buf0+2)));\n        *(buf0+3) = __builtin_fmaf(val6,val8[3],(*(buf0+3)));\n        *(buf0+4) = __builtin_fmaf(val6,val7[0],(*(buf0+4)));\n        *(buf0+5) = __builtin_fmaf(val6,val7[1],(*(buf0+5)));\n        *(buf0+6) = __builtin_fmaf(val6,val7[2],(*(buf0+6)));\n        *(buf0+7) = __builtin_fmaf(val6,val7[3],(*(buf0+7)));\n      }\n      int alu20 = ((Lidx1<<5)+alu1);\n      *((float4*)((((float*)(val3))+(alu20+4)))) = (float4){(*(buf0+4)),(*(buf0+5)),(*(buf0+6)),(*(buf0+7))};\n      *((float4*)((((float*)(val3))+alu20))) = (float4){(*(buf0+0)),(*(buf0+1)),(*(buf0+2)),(*(buf0+3))};\n    }\n  }\n}\n"
696 Ops.BINARY dtypes.u8 [] b"typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_2_4_8_i(unsigned char* restrict args_48) {\n  float buf0[8];\n  int val0 = (*(((int*)(args_48))+6));\n  int val1 = (*(((int*)(args_48))+7));\n  int val2 = (*(((int*)(args_48))+8));\n  unsigned long val3 = (*(((unsigned long*)(args_48))+0));\n  unsigned long val4 = (*(((unsigned long*)(args_48))+1));\n  unsigned long val5 = (*(((unsigned long*)(args_48))+2));\n  for (int Lidx1 = 0; Lidx1 < 2; Lidx1++) {\n    for (int Lidx2 = 0; Lidx2 < (val1-val0); Lidx2++) {\n      int alu0 = (val0+Lidx2);\n      int alu1 = (alu0<<3);\n      *(buf0+0) = 0.0f;\n      *(buf0+1) = 0.0f;\n      *(buf0+2) = 0.0f;\n      *(buf0+3) = 0.0f;\n      *(buf0+4) = 0.0f;\n      *(buf0+5) = 0.0f;\n      *(buf0+6) = 0.0f;\n      *(buf0+7) = 0.0f;\n      for (int Lidx0 = 0; Lidx0 < val2; Lidx0++) {\n        float val6 = (*(((float*)(val4))+((alu0*10)+Lidx0+(Lidx1*40))));\n        int alu10 = ((Lidx0<<5)+alu1+(Lidx1*320));\n        float4 val7 = (*((float4*)((((float*)(val5))+(alu10+4)))));\n        float4 val8 = (*((float4*)((((float*)(val5))+alu10))));\n        *(buf0+0) = __builtin_fmaf(val6,val8[0],(*(buf0+0)));\n        *(buf0+1) = __builtin_fmaf(val6,val8[1],(*(buf0+1)));\n        *(buf0+2) = __builtin_fmaf(val6,val8[2],(*(buf0+2)));\n        *(buf0+3) = __builtin_fmaf(val6,val8[3],(*(buf0+3)));\n        *(buf0+4) = __builtin_fmaf(val6,val7[0],(*(buf0+4)));\n        *(buf0+5) = __builtin_fmaf(val6,val7[1],(*(buf0+5)));\n        *(buf0+6) = __builtin_fmaf(val6,val7[2],(*(buf0+6)));\n        *(buf0+7) = __builtin_fmaf(val6,val7[3],(*(buf0+7)));\n      }\n      int alu20 = ((Lidx1<<5)+alu1);\n      *((float4*)((((float*)(val3))+(alu20+4)))) = (float4){(*(buf0+4)),(*(buf0+5)),(*(buf0+6)),(*(buf0+7))};\n      *((float4*)((((float*)(val3))+alu20))) = (float4){(*(buf0+0)),(*(buf0+1)),(*(buf0+2)),(*(buf0+3))};\n    }\n  }\n}\n"
697 Ops.PARAM dtypes.f32 [37] ParamArg(slot=0, dtype=dtypes.f32)
698 Ops.PARAM dtypes.i32 [101] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 4), name="block_lo", addrspace=AddrSpace.ALU)
699 Ops.PARAM dtypes.i32 [101] ParamArg(slot=4, dtype=dtypes.i32, vmin_vmax=(0, 4), name="block_hi", addrspace=AddrSpace.ALU)
700 Ops.PROGRAM dtypes.void [693, 694, 695, 696] ProgramInfo(global_size=(4, 1, 1), params=(%697, %217, %116, %698, %699, %401), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CPU", arch="x86_64,x86-64"))
701 Ops.BUFFER dtypes.f32 [37] ParamArg(slot=6, dtype=dtypes.f32, device="CPU")
702 Ops.PARAM dtypes.f32 [54] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
703 Ops.CALL dtypes.void [700, 701, 578, 702] CallInfo()
704 Ops.LINEAR dtypes.void [126, 224, 417, 579, 703]

