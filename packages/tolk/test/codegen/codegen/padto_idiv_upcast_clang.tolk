# tolk D80 D141 D151 D502 golden=cdfef6a59edfd6b5a17bbd50d30c1e91
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 10
2 Ops.PARAM dtypes.i32 [1] ParamArg(slot=0, dtype=dtypes.i32)
3 Ops.PARAM dtypes.i32 [0] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 4), name="block_lo", addrspace=AddrSpace.ALU)
4 Ops.PARAM dtypes.i32 [0] ParamArg(slot=4, dtype=dtypes.i32, vmin_vmax=(0, 4), name="block_hi", addrspace=AddrSpace.ALU)
5 Ops.SUB dtypes.i32 [4, 3]
6 Ops.RANGE dtypes.i32 [5] (AxisType.WEAK, 0)
7 Ops.ADD dtypes.i32 [3, 6]
8 Ops.CONST dtypes.weakint [] 2
9 Ops.CAST dtypes.i32 [8] dtypes.i32
10 Ops.SHL dtypes.i32 [7, 9]
11 Ops.SHRINK dtypes.i32 [2, 10, 9]
12 Ops.CONST dtypes.weakint [] 30
13 Ops.PARAM dtypes.i32 [12] ParamArg(slot=2, dtype=dtypes.i32)
14 Ops.CONST dtypes.weakint [] 3
15 Ops.CAST dtypes.i32 [14] dtypes.i32
16 Ops.CMPLT dtypes.bool [7, 15]
17 Ops.CONST dtypes.weakint [] 9
18 Ops.CAST dtypes.i32 [17] dtypes.i32
19 Ops.WHERE dtypes.i32 [16, 10, 18]
20 Ops.MUL dtypes.i32 [19, 15]
21 Ops.CONST dtypes.weakint [] 1
22 Ops.CAST dtypes.i32 [21] dtypes.i32
23 Ops.ADD dtypes.i32 [20, 22]
24 Ops.INDEX dtypes.i32 [13, 23]
25 Ops.LOAD dtypes.i32 [24]
26 Ops.ADD dtypes.i32 [10, 22]
27 Ops.WHERE dtypes.i32 [16, 26, 18]
28 Ops.MUL dtypes.i32 [27, 15]
29 Ops.ADD dtypes.i32 [28, 22]
30 Ops.INDEX dtypes.i32 [13, 29]
31 Ops.LOAD dtypes.i32 [30]
32 Ops.CMPLT dtypes.bool [7, 9]
33 Ops.ADD dtypes.i32 [10, 9]
34 Ops.WHERE dtypes.i32 [32, 33, 18]
35 Ops.MUL dtypes.i32 [34, 15]
36 Ops.ADD dtypes.i32 [35, 22]
37 Ops.INDEX dtypes.i32 [13, 36]
38 Ops.LOAD dtypes.i32 [37]
39 Ops.ADD dtypes.i32 [10, 15]
40 Ops.WHERE dtypes.i32 [32, 39, 18]
41 Ops.MUL dtypes.i32 [40, 15]
42 Ops.ADD dtypes.i32 [41, 22]
43 Ops.INDEX dtypes.i32 [13, 42]
44 Ops.LOAD dtypes.i32 [43]
45 Ops.STACK dtypes.i32 [25, 31, 38, 44]
46 Ops.CONST dtypes.weakint [] 0
47 Ops.CAST dtypes.i32 [46] dtypes.i32
48 Ops.STACK dtypes.i32 [47, 47, 47, 47]
49 Ops.CMPNE dtypes.bool [45, 48]
50 Ops.PARAM dtypes.i32 [1] ParamArg(slot=1, dtype=dtypes.i32)
51 Ops.INDEX dtypes.i32 [50, 19]
52 Ops.LOAD dtypes.i32 [51]
53 Ops.INDEX dtypes.i32 [50, 27]
54 Ops.LOAD dtypes.i32 [53]
55 Ops.INDEX dtypes.i32 [50, 34]
56 Ops.LOAD dtypes.i32 [55]
57 Ops.INDEX dtypes.i32 [50, 40]
58 Ops.LOAD dtypes.i32 [57]
59 Ops.STACK dtypes.i32 [52, 54, 56, 58]
60 Ops.CMPLT dtypes.bool [59, 47]
61 Ops.CMPLT dtypes.bool [45, 47]
62 Ops.CMPNE dtypes.bool [60, 61]
63 Ops.CAST dtypes.u32 [59] dtypes.u32
64 Ops.NEG dtypes.u32 [63]
65 Ops.WHERE dtypes.u32 [60, 64, 63]
66 Ops.CAST dtypes.u32 [45] dtypes.u32
67 Ops.NEG dtypes.u32 [66]
68 Ops.WHERE dtypes.u32 [61, 67, 66]
69 Ops.CAST dtypes.u32 [46] dtypes.u32
70 Ops.STACK dtypes.u32 [69, 69, 69, 69]
71 Ops.CMPNE dtypes.bool [68, 70]
72 Ops.CAST dtypes.u32 [21] dtypes.u32
73 Ops.STACK dtypes.u32 [72, 72, 72, 72]
74 Ops.WHERE dtypes.u32 [71, 68, 73]
75 Ops.TRUNCDIV dtypes.u32 [65, 74]
76 Ops.NEG dtypes.u32 [75]
77 Ops.WHERE dtypes.u32 [62, 76, 75]
78 Ops.CAST dtypes.i32 [77] dtypes.i32
79 Ops.WHERE dtypes.i32 [49, 78, 48]
80 Ops.INDEX dtypes.i32 [79, 47]
81 Ops.INDEX dtypes.i32 [79, 22]
82 Ops.STACK dtypes.i32 [80, 81]
83 Ops.STORE dtypes.void [11, 82, 16]
84 Ops.SHRINK dtypes.i32 [2, 33, 9]
85 Ops.INDEX dtypes.i32 [79, 9]
86 Ops.INDEX dtypes.i32 [79, 15]
87 Ops.STACK dtypes.i32 [85, 86]
88 Ops.STORE dtypes.void [84, 87, 32]
89 Ops.AFTER dtypes.void [0, 83, 88]
90 Ops.END dtypes.void [89, 6]
91 Ops.SINK dtypes.void [90] KernelInfo(name="E_4_4", applied_opts=(Opt(op=OptOps.PADTO, axis=0, arg=16), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST))), optimized=True, split=(4, 3, 4))
92 Ops.CONST dtypes.weakint [] 40
93 Ops.PARAM dtypes.u8 [92] ParamArg(slot=0, dtype=dtypes.u8, name="args")
94 Ops.BITCAST dtypes.u64 [93] dtypes.u64
95 Ops.INDEX dtypes.u64 [94, 47]
96 Ops.LOAD dtypes.u64 [95]
97 Ops.REFPARAM dtypes.i32 [1, 96] ParamArg(slot=0, dtype=dtypes.i32)
98 Ops.BITCAST dtypes.i32 [93] dtypes.i32
99 Ops.CONST dtypes.weakint [] 6
100 Ops.CAST dtypes.i32 [99] dtypes.i32
101 Ops.INDEX dtypes.i32 [98, 100]
102 Ops.LOAD dtypes.i32 [101]
103 Ops.CONST dtypes.weakint [] 7
104 Ops.CAST dtypes.i32 [103] dtypes.i32
105 Ops.INDEX dtypes.i32 [98, 104]
106 Ops.LOAD dtypes.i32 [105]
107 Ops.SUB dtypes.i32 [106, 102]
108 Ops.RANGE dtypes.i32 [107] (AxisType.WEAK, 0)
109 Ops.ADD dtypes.i32 [102, 108]
110 Ops.SHL dtypes.i32 [109, 9]
111 Ops.SHRINK dtypes.i32 [97, 110, 9]
112 Ops.INDEX dtypes.u64 [94, 9]
113 Ops.LOAD dtypes.u64 [112]
114 Ops.REFPARAM dtypes.i32 [12, 113] ParamArg(slot=2, dtype=dtypes.i32)
115 Ops.CMPLT dtypes.bool [109, 15]
116 Ops.WHERE dtypes.i32 [115, 110, 18]
117 Ops.MUL dtypes.i32 [116, 15]
118 Ops.ADD dtypes.i32 [117, 22]
119 Ops.INDEX dtypes.i32 [114, 118]
120 Ops.LOAD dtypes.i32 [119]
121 Ops.ADD dtypes.i32 [110, 22]
122 Ops.WHERE dtypes.i32 [115, 121, 18]
123 Ops.MUL dtypes.i32 [122, 15]
124 Ops.ADD dtypes.i32 [123, 22]
125 Ops.INDEX dtypes.i32 [114, 124]
126 Ops.LOAD dtypes.i32 [125]
127 Ops.CMPLT dtypes.bool [109, 9]
128 Ops.ADD dtypes.i32 [110, 9]
129 Ops.WHERE dtypes.i32 [127, 128, 18]
130 Ops.MUL dtypes.i32 [129, 15]
131 Ops.ADD dtypes.i32 [130, 22]
132 Ops.INDEX dtypes.i32 [114, 131]
133 Ops.LOAD dtypes.i32 [132]
134 Ops.ADD dtypes.i32 [110, 15]
135 Ops.WHERE dtypes.i32 [127, 134, 18]
136 Ops.MUL dtypes.i32 [135, 15]
137 Ops.ADD dtypes.i32 [136, 22]
138 Ops.INDEX dtypes.i32 [114, 137]
139 Ops.LOAD dtypes.i32 [138]
140 Ops.STACK dtypes.i32 [120, 126, 133, 139]
141 Ops.CMPNE dtypes.bool [140, 48]
142 Ops.INDEX dtypes.u64 [94, 22]
143 Ops.LOAD dtypes.u64 [142]
144 Ops.REFPARAM dtypes.i32 [1, 143] ParamArg(slot=1, dtype=dtypes.i32)
145 Ops.INDEX dtypes.i32 [144, 116]
146 Ops.LOAD dtypes.i32 [145]
147 Ops.INDEX dtypes.i32 [144, 122]
148 Ops.LOAD dtypes.i32 [147]
149 Ops.INDEX dtypes.i32 [144, 129]
150 Ops.LOAD dtypes.i32 [149]
151 Ops.INDEX dtypes.i32 [144, 135]
152 Ops.LOAD dtypes.i32 [151]
153 Ops.STACK dtypes.i32 [146, 148, 150, 152]
154 Ops.CMPLT dtypes.bool [153, 47]
155 Ops.CMPLT dtypes.bool [140, 47]
156 Ops.CMPNE dtypes.bool [154, 155]
157 Ops.CAST dtypes.u32 [153] dtypes.u32
158 Ops.NEG dtypes.u32 [157]
159 Ops.WHERE dtypes.u32 [154, 158, 157]
160 Ops.CAST dtypes.u32 [140] dtypes.u32
161 Ops.NEG dtypes.u32 [160]
162 Ops.WHERE dtypes.u32 [155, 161, 160]
163 Ops.CMPNE dtypes.bool [162, 70]
164 Ops.WHERE dtypes.u32 [163, 162, 73]
165 Ops.TRUNCDIV dtypes.u32 [159, 164]
166 Ops.NEG dtypes.u32 [165]
167 Ops.WHERE dtypes.u32 [156, 166, 165]
168 Ops.CAST dtypes.i32 [167] dtypes.i32
169 Ops.WHERE dtypes.i32 [141, 168, 48]
170 Ops.INDEX dtypes.i32 [169, 47]
171 Ops.INDEX dtypes.i32 [169, 22]
172 Ops.STACK dtypes.i32 [170, 171]
173 Ops.STORE dtypes.void [111, 172]
174 Ops.SHRINK dtypes.i32 [97, 128, 9]
175 Ops.INDEX dtypes.i32 [169, 9]
176 Ops.INDEX dtypes.i32 [169, 15]
177 Ops.STACK dtypes.i32 [175, 176]
178 Ops.STORE dtypes.void [174, 177]
179 Ops.AFTER dtypes.void [0, 173, 178]
180 Ops.END dtypes.void [179, 108]
181 Ops.SINK dtypes.void [180] KernelInfo(name="E_4_4", applied_opts=(Opt(op=OptOps.PADTO, axis=0, arg=16), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST))), estimates=Estimates(ops=224, lds=192, mem=144), optimized=True, split=(4, 3, 4))
182 Ops.IF dtypes.void [127, 174]
183 Ops.ENDIF dtypes.void [182]
184 Ops.IF dtypes.void [115, 111]
185 Ops.ENDIF dtypes.void [184]
186 Ops.LINEAR dtypes.void [92, 93, 99, 100, 98, 101, 102, 103, 104, 105, 106, 46, 47, 94, 95, 96, 21, 22, 142, 143, 8, 9, 112, 113, 1, 97, 144, 12, 114, 0, 48, 69, 70, 72, 73, 14, 15, 17, 18, 107, 108, 109, 110, 128, 127, 129, 149, 150, 134, 135, 151, 152, 121, 115, 122, 147, 148, 116, 145, 146, 130, 131, 132, 133, 136, 137, 138, 139, 123, 124, 125, 126, 117, 118, 119, 120, 153, 140, 157, 160, 158, 161, 154, 155, 162, 163, 159, 164, 165, 166, 156, 167, 168, 141, 169, 170, 171, 172, 175, 176, 177, 174, 111, 182, 178, 183, 184, 173, 185, 179, 180, 181]
187 Ops.SOURCE dtypes.void [] "typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef unsigned int unsigned_int4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int2 __attribute__((aligned(4),ext_vector_type(2)));\n__attribute__((min_vector_width(512))) void E_4_4(unsigned char* restrict args_40) {\n  int val0 = (*(((int*)(args_40))+6));\n  int val1 = (*(((int*)(args_40))+7));\n  unsigned long val2 = (*(((unsigned long*)(args_40))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_40))+1));\n  unsigned long val4 = (*(((unsigned long*)(args_40))+2));\n  int4 cast0 = (int4){0,0,0,0};\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = (val0+Lidx0);\n    int alu1 = (alu0<<2);\n    int alu2 = (alu1+2);\n    _Bool alu3 = (alu0<2);\n    int alu4 = (alu3?alu2:9);\n    int val5 = (*(((int*)(val3))+alu4));\n    int alu5 = (alu3?(alu1+3):9);\n    int val6 = (*(((int*)(val3))+alu5));\n    _Bool alu6 = (alu0<3);\n    int alu7 = (alu6?(alu1+1):9);\n    int val7 = (*(((int*)(val3))+alu7));\n    int alu8 = (alu6?alu1:9);\n    int val8 = (*(((int*)(val3))+alu8));\n    int val9 = (*(((int*)(val4))+((alu4*3)+1)));\n    int val10 = (*(((int*)(val4))+((alu5*3)+1)));\n    int val11 = (*(((int*)(val4))+((alu7*3)+1)));\n    int val12 = (*(((int*)(val4))+((alu8*3)+1)));\n    int4 cast1 = (int4){val8,val7,val5,val6};\n    int4 cast2 = (int4){val12,val11,val9,val10};\n    unsigned_int4 cast3 = __builtin_convertvector(cast1, unsigned_int4);\n    unsigned_int4 cast4 = __builtin_convertvector(cast2, unsigned_int4);\n    int4 alu9 = (cast1<0);\n    int4 alu10 = (cast2<0);\n    unsigned_int4 alu11 = (alu10?-cast4:cast4);\n    unsigned_int4 alu12 = (alu9?-cast3:cast3);\n    unsigned_int4 alu13 = ((alu11!=(unsigned_int4){0u,0u,0u,0u})?alu11:(unsigned_int4){1u,1u,1u,1u});\n    unsigned_int4 alu14 = (alu12/alu13);\n    unsigned_int4 alu15 = ((alu9^alu10)?-alu14:alu14);\n    int4 cast5 = __builtin_convertvector(alu15, int4);\n    int4 alu16 = ((cast2!=cast0)?cast5:cast0);\n    if (alu3) {\n      *((int2*)((((int*)(val2))+alu2))) = ((int2)__builtin_shufflevector(alu16, alu16, 2, 3));\n    }\n    if (alu6) {\n      *((int2*)((((int*)(val2))+alu1))) = ((int2)__builtin_shufflevector(alu16, alu16, 0, 1));\n    }\n  }\n}\n"
188 Ops.PROGRAM dtypes.void [181, 186, 187] ProgramInfo(global_size=(4, 1, 1), params=(%2, %50, %13, %3, %4), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
189 Ops.SINK dtypes.void [91, 188]

