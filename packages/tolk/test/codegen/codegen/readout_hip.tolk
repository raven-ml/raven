# tolk D89 D480 golden=7d75fc760f17c951fedbfc069cfaaaae
0 Ops.CONST dtypes.weakint [] 768
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.SPECIAL dtypes.i32 [3] "gidx0"
5 Ops.CONST dtypes.weakint [] 4
6 Ops.CAST dtypes.i32 [5] dtypes.i32
7 Ops.SHL dtypes.i32 [4, 6]
8 Ops.SPECIAL dtypes.i32 [3] "lidx0"
9 Ops.ADD dtypes.i32 [7, 8]
10 Ops.CONST dtypes.weakint [] 3
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.SPECIAL dtypes.i32 [11] "gidx1"
13 Ops.CONST dtypes.weakint [] 8
14 Ops.CAST dtypes.i32 [13] dtypes.i32
15 Ops.SHL dtypes.i32 [12, 14]
16 Ops.ADD dtypes.i32 [9, 15]
17 Ops.INDEX dtypes.f32 [1, 16]
18 Ops.CONST dtypes.weakint [] 1
19 Ops.ALLOC dtypes.f32 [18] ParamArg(slot=2, dtype=dtypes.f32, addrspace=AddrSpace.REG)
20 Ops.AFTER dtypes.f32 [19, 12, 4, 8]
21 Ops.CONST dtypes.weakint [] 0
22 Ops.CAST dtypes.i32 [21] dtypes.i32
23 Ops.INDEX dtypes.f32 [20, 22]
24 Ops.CONST dtypes.weakfloat [] 0.0
25 Ops.CAST dtypes.f32 [24] dtypes.f32
26 Ops.STORE dtypes.void [23, 25]
27 Ops.ALLOC dtypes.f32 [18] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
28 Ops.SPECIAL dtypes.i32 [3] "lidx1"
29 Ops.AFTER dtypes.f32 [27, 28, 12, 4, 8]
30 Ops.INDEX dtypes.f32 [29, 22]
31 Ops.STORE dtypes.void [30, 25]
32 Ops.RANGE dtypes.i32 [3] (AxisType.WEAK, 0)
33 Ops.AFTER dtypes.f32 [27, 31, 32]
34 Ops.INDEX dtypes.f32 [33, 22]
35 Ops.LOAD dtypes.f32 [34]
36 Ops.ALLOC dtypes.f32 [18] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
37 Ops.AFTER dtypes.f32 [36, 28, 32, 12, 4, 8]
38 Ops.INDEX dtypes.f32 [37, 22]
39 Ops.STORE dtypes.void [38, 25]
40 Ops.CONST dtypes.weakint [] 64
41 Ops.CAST dtypes.i32 [40] dtypes.i32
42 Ops.AFTER dtypes.i32 [6, 32]
43 Ops.RANGE dtypes.i32 [42] (AxisType.WEAK, 0, 256)
44 Ops.AFTER dtypes.i32 [41, 43]
45 Ops.RANGE dtypes.i32 [44] (AxisType.WEAK, 1)
46 Ops.AFTER dtypes.f32 [36, 39, 45, 43]
47 Ops.INDEX dtypes.f32 [46, 22]
48 Ops.LOAD dtypes.f32 [47]
49 Ops.CONST dtypes.weakint [] 196608
50 Ops.PARAM dtypes.f32 [49] ParamArg(slot=1, dtype=dtypes.f32)
51 Ops.CONST dtypes.weakint [] 192
52 Ops.CAST dtypes.i32 [51] dtypes.i32
53 Ops.MUL dtypes.i32 [32, 52]
54 Ops.CONST dtypes.weakint [] 48
55 Ops.CAST dtypes.i32 [54] dtypes.i32
56 Ops.MUL dtypes.i32 [43, 55]
57 Ops.ADD dtypes.i32 [53, 56]
58 Ops.MUL dtypes.i32 [28, 11]
59 Ops.ADD dtypes.i32 [57, 58]
60 Ops.ADD dtypes.i32 [59, 12]
61 Ops.CONST dtypes.weakint [] 3072
62 Ops.CAST dtypes.i32 [61] dtypes.i32
63 Ops.MUL dtypes.i32 [45, 62]
64 Ops.ADD dtypes.i32 [60, 63]
65 Ops.INDEX dtypes.f32 [50, 64]
66 Ops.LOAD dtypes.f32 [65]
67 Ops.CONST dtypes.weakint [] 16777216
68 Ops.PARAM dtypes.f32 [67] ParamArg(slot=2, dtype=dtypes.f32)
69 Ops.SHL dtypes.i32 [45, 14]
70 Ops.ADD dtypes.i32 [9, 69]
71 Ops.CONST dtypes.weakint [] 20
72 Ops.CAST dtypes.i32 [71] dtypes.i32
73 Ops.SHL dtypes.i32 [32, 72]
74 Ops.CONST dtypes.weakint [] 18
75 Ops.CAST dtypes.i32 [74] dtypes.i32
76 Ops.SHL dtypes.i32 [43, 75]
77 Ops.ADD dtypes.i32 [73, 76]
78 Ops.CONST dtypes.weakint [] 14
79 Ops.CAST dtypes.i32 [78] dtypes.i32
80 Ops.SHL dtypes.i32 [28, 79]
81 Ops.ADD dtypes.i32 [77, 80]
82 Ops.ADD dtypes.i32 [70, 81]
83 Ops.INDEX dtypes.f32 [68, 82]
84 Ops.LOAD dtypes.f32 [83]
85 Ops.ADD dtypes.f32 [66, 84]
86 Ops.CONST dtypes.weakfloat [] 1.4426950216293335
87 Ops.CAST dtypes.f32 [86] dtypes.f32
88 Ops.MUL dtypes.f32 [85, 87]
89 Ops.EXP2 dtypes.f32 [88]
90 Ops.ADD dtypes.f32 [48, 89]
91 Ops.STORE dtypes.void [47, 90]
92 Ops.END dtypes.void [91, 45]
93 Ops.END dtypes.void [92, 43]
94 Ops.AFTER dtypes.f32 [36, 93]
95 Ops.INDEX dtypes.f32 [94, 22]
96 Ops.LOAD dtypes.f32 [95]
97 Ops.ADD dtypes.f32 [35, 96]
98 Ops.STORE dtypes.void [34, 97]
99 Ops.END dtypes.void [98, 32]
100 Ops.AFTER dtypes.i32 [3, 99]
101 Ops.RANGE dtypes.i32 [100] (AxisType.WEAK, 105)
102 Ops.AFTER dtypes.f32 [19, 26, 101]
103 Ops.INDEX dtypes.f32 [102, 22]
104 Ops.LOAD dtypes.f32 [103]
105 Ops.CONST dtypes.weakint [] 256
106 Ops.ALLOC dtypes.f32 [105] ParamArg(slot=3, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
107 Ops.SHL dtypes.i32 [8, 6]
108 Ops.ADD dtypes.i32 [107, 28]
109 Ops.INDEX dtypes.f32 [106, 108]
110 Ops.AFTER dtypes.f32 [27, 99]
111 Ops.INDEX dtypes.f32 [110, 22]
112 Ops.LOAD dtypes.f32 [111]
113 Ops.STORE dtypes.void [109, 112]
114 Ops.BARRIER dtypes.void [113]
115 Ops.AFTER dtypes.f32 [106, 114]
116 Ops.ADD dtypes.i32 [107, 101]
117 Ops.INDEX dtypes.f32 [115, 116]
118 Ops.LOAD dtypes.f32 [117]
119 Ops.ADD dtypes.f32 [104, 118]
120 Ops.STORE dtypes.void [103, 119]
121 Ops.END dtypes.void [120, 101]
122 Ops.AFTER dtypes.f32 [19, 121]
123 Ops.INDEX dtypes.f32 [122, 22]
124 Ops.LOAD dtypes.f32 [123]
125 Ops.CMPEQ dtypes.bool [28, 22]
126 Ops.STORE dtypes.void [17, 124, 125]
127 Ops.SINK dtypes.void [126] KernelInfo(name="r_3_16_16_16_64_64", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=3, arg=(16, AxisType.LOCAL))), optimized=True)
128 Ops.BUFFER dtypes.f32 [18] ParamArg(slot=2, dtype=dtypes.f32, addrspace=AddrSpace.REG)
129 Ops.AFTER dtypes.f32 [128, 12, 4, 8]
130 Ops.INDEX dtypes.f32 [129, 22]
131 Ops.STORE dtypes.void [130, 25]
132 Ops.BUFFER dtypes.f32 [18] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
133 Ops.AFTER dtypes.f32 [132, 28, 12, 4, 8]
134 Ops.INDEX dtypes.f32 [133, 22]
135 Ops.STORE dtypes.void [134, 25]
136 Ops.AFTER dtypes.f32 [132, 135, 32]
137 Ops.INDEX dtypes.f32 [136, 22]
138 Ops.LOAD dtypes.f32 [137]
139 Ops.BUFFER dtypes.f32 [18] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
140 Ops.AFTER dtypes.f32 [139, 28, 32, 12, 4, 8]
141 Ops.INDEX dtypes.f32 [140, 22]
142 Ops.STORE dtypes.void [141, 25]
143 Ops.AFTER dtypes.f32 [139, 142, 45, 43]
144 Ops.INDEX dtypes.f32 [143, 22]
145 Ops.LOAD dtypes.f32 [144]
146 Ops.ADD dtypes.f32 [145, 89]
147 Ops.STORE dtypes.void [144, 146]
148 Ops.END dtypes.void [147, 45]
149 Ops.END dtypes.void [148, 43]
150 Ops.AFTER dtypes.f32 [139, 149]
151 Ops.INDEX dtypes.f32 [150, 22]
152 Ops.LOAD dtypes.f32 [151]
153 Ops.ADD dtypes.f32 [138, 152]
154 Ops.STORE dtypes.void [137, 153]
155 Ops.END dtypes.void [154, 32]
156 Ops.AFTER dtypes.i32 [3, 155]
157 Ops.RANGE dtypes.i32 [156] (AxisType.WEAK, 105)
158 Ops.AFTER dtypes.f32 [128, 131, 157]
159 Ops.INDEX dtypes.f32 [158, 22]
160 Ops.LOAD dtypes.f32 [159]
161 Ops.BUFFER dtypes.f32 [105] ParamArg(slot=3, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
162 Ops.INDEX dtypes.f32 [161, 108]
163 Ops.AFTER dtypes.f32 [132, 155]
164 Ops.INDEX dtypes.f32 [163, 22]
165 Ops.LOAD dtypes.f32 [164]
166 Ops.STORE dtypes.void [162, 165]
167 Ops.BARRIER dtypes.void [166]
168 Ops.AFTER dtypes.f32 [161, 167]
169 Ops.ADD dtypes.i32 [107, 157]
170 Ops.INDEX dtypes.f32 [168, 169]
171 Ops.LOAD dtypes.f32 [170]
172 Ops.ADD dtypes.f32 [160, 171]
173 Ops.STORE dtypes.void [159, 172]
174 Ops.END dtypes.void [173, 157]
175 Ops.AFTER dtypes.f32 [128, 174]
176 Ops.INDEX dtypes.f32 [175, 22]
177 Ops.LOAD dtypes.f32 [176]
178 Ops.STORE dtypes.void [17, 177]
179 Ops.SINK dtypes.void [178] KernelInfo(name="r_3_16_16_16_64_64", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=3, arg=(16, AxisType.LOCAL))), estimates=Estimates(ops=201732096, lds=403537920, mem=67898368), optimized=True)
180 Ops.IF dtypes.void [125, 17]
181 Ops.ENDIF dtypes.void [180]
182 Ops.LINEAR dtypes.void [0, 1, 49, 50, 67, 68, 18, 139, 132, 128, 105, 161, 2, 3, 4, 10, 11, 12, 8, 28, 133, 21, 22, 134, 24, 25, 86, 87, 78, 79, 74, 75, 51, 52, 71, 72, 61, 62, 5, 6, 54, 55, 40, 41, 13, 14, 7, 9, 58, 80, 135, 32, 42, 140, 141, 53, 73, 142, 43, 44, 56, 57, 59, 60, 76, 77, 81, 45, 63, 64, 65, 66, 69, 70, 82, 83, 84, 143, 144, 145, 85, 88, 89, 146, 147, 148, 149, 150, 151, 152, 136, 137, 138, 153, 154, 155, 163, 164, 165, 156, 129, 107, 108, 162, 166, 167, 168, 130, 131, 157, 158, 159, 160, 169, 170, 171, 172, 173, 174, 175, 176, 177, 15, 16, 17, 125, 180, 178, 181, 179]
183 Ops.SOURCE dtypes.void [] "typedef long unsigned int size_t;\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_local_size(unsigned int);\nextern \"C\" __attribute__((device, pure)) float __ocml_exp2_f32(float);\nextern \"C\" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 256))) r_3_16_16_16_64_64(float* data0_768, float* data1_196608, float* data2_16777216) {\n  float buf0[1];\n  float buf1[1];\n  float buf2[1];\n  __attribute__((shared, aligned(16)))float buf3[256];\n  int gidx0 = __ockl_get_group_id(0); /* 16 */\n  int gidx1 = __ockl_get_group_id(1); /* 3 */\n  int lidx0 = __ockl_get_local_id(0); /* 16 */\n  int lidx1 = __ockl_get_local_id(1); /* 16 */\n  int alu0 = ((gidx0<<4)+lidx0);\n  *(buf1+0) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 16; Lidx0++) {\n    *(buf0+0) = 0.0f;\n    for (int Lidx0_256 = 0; Lidx0_256 < 4; Lidx0_256++) {\n      for (int Lidx1 = 0; Lidx1 < 64; Lidx1++) {\n        float val0 = (*(data1_196608+((Lidx0*192)+(Lidx0_256*48)+(lidx1*3)+gidx1+(Lidx1*3072))));\n        float val1 = (*(data2_16777216+(alu0+(Lidx1<<8)+((Lidx0<<20)+(Lidx0_256<<18)+(lidx1<<14)))));\n        *(buf0+0) = ((*(buf0+0))+__ocml_exp2_f32(((val0+val1)*1.4426950216293335f)));\n      }\n    }\n    *(buf1+0) = ((*(buf1+0))+(*(buf0+0)));\n  }\n  int alu8 = (lidx0<<4);\n  *(buf3+(alu8+lidx1)) = (*(buf1+0));\n  __builtin_amdgcn_fence(__ATOMIC_RELEASE, \"workgroup\");__builtin_amdgcn_s_barrier();__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, \"workgroup\");\n  *(buf2+0) = 0.0f;\n  for (int Lidx105 = 0; Lidx105 < 16; Lidx105++) {\n    float val2 = (*(buf3+(alu8+Lidx105)));\n    *(buf2+0) = ((*(buf2+0))+val2);\n  }\n  if ((lidx1==0)) {\n    *(data0_768+(alu0+(gidx1<<8))) = (*(buf2+0));\n  }\n}"
184 Ops.PROGRAM dtypes.void [179, 182, 183] ProgramInfo(global_size=(16, 3, 1), local_size=(16, 16, 1), params=(%1, %50, %68), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="AMD", renderer="HIP", arch="gfx1100"))
185 Ops.SINK dtypes.void [127, 184]

