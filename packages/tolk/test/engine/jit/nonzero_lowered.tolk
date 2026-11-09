# tolk D141 D151 golden=1a75b350793c35906a92f669bc121d66
0 Ops.CONST dtypes.weakint [] 5
1 Ops.CONST dtypes.weakint [] 16
2 Ops.PARAM dtypes.u8 [1] ParamArg(slot=0, dtype=dtypes.u8, name="args")
3 Ops.BITCAST dtypes.u64 [2] dtypes.u64
4 Ops.CONST dtypes.weakint [] 0
5 Ops.CAST dtypes.i32 [4] dtypes.i32
6 Ops.INDEX dtypes.u64 [3, 5]
7 Ops.LOAD dtypes.u64 [6]
8 Ops.REFPARAM dtypes.i32 [0, 7] ParamArg(slot=0, dtype=dtypes.i32)
9 Ops.CONST dtypes.weakint [] 4
10 Ops.CAST dtypes.i32 [9] dtypes.i32
11 Ops.SHRINK dtypes.i32 [8, 5, 10]
12 Ops.CONST dtypes.weakint [] 1
13 Ops.CAST dtypes.i32 [12] dtypes.i32
14 Ops.INDEX dtypes.u64 [3, 13]
15 Ops.LOAD dtypes.u64 [14]
16 Ops.REFPARAM dtypes.i32 [0, 15] ParamArg(slot=1, dtype=dtypes.i32)
17 Ops.SHRINK dtypes.i32 [16, 5, 10]
18 Ops.LOAD dtypes.i32 [17]
19 Ops.INDEX dtypes.i32 [18, 5]
20 Ops.CMPNE dtypes.bool [19, 5]
21 Ops.CAST dtypes.i32 [20] dtypes.i32
22 Ops.INDEX dtypes.i32 [18, 13]
23 Ops.CMPNE dtypes.bool [22, 5]
24 Ops.CAST dtypes.i32 [23] dtypes.i32
25 Ops.ADD dtypes.i32 [21, 24]
26 Ops.CONST dtypes.weakint [] 2
27 Ops.CAST dtypes.i32 [26] dtypes.i32
28 Ops.INDEX dtypes.i32 [18, 27]
29 Ops.CMPNE dtypes.bool [28, 5]
30 Ops.CAST dtypes.i32 [29] dtypes.i32
31 Ops.ADD dtypes.i32 [25, 30]
32 Ops.CONST dtypes.weakint [] 3
33 Ops.CAST dtypes.i32 [32] dtypes.i32
34 Ops.INDEX dtypes.i32 [18, 33]
35 Ops.CMPNE dtypes.bool [34, 5]
36 Ops.CAST dtypes.i32 [35] dtypes.i32
37 Ops.ADD dtypes.i32 [31, 36]
38 Ops.STACK dtypes.i32 [21, 25, 31, 37]
39 Ops.STORE dtypes.void [11, 38]
40 Ops.INDEX dtypes.i32 [8, 10]
41 Ops.INDEX dtypes.i32 [16, 10]
42 Ops.LOAD dtypes.i32 [41]
43 Ops.CMPNE dtypes.bool [42, 5]
44 Ops.CAST dtypes.i32 [43] dtypes.i32
45 Ops.ADD dtypes.i32 [37, 44]
46 Ops.STORE dtypes.void [40, 45]
47 Ops.SINK dtypes.void [39, 39, 39, 39, 46] KernelInfo(name="r_5_5", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST))), estimates=Estimates(ops=9, lds=40, mem=40), optimized=True)
48 Ops.LINEAR dtypes.void [1, 2, 12, 13, 3, 14, 15, 0, 16, 9, 10, 41, 42, 4, 5, 17, 18, 6, 7, 8, 19, 22, 26, 27, 28, 32, 33, 34, 20, 21, 23, 24, 29, 30, 35, 36, 25, 31, 37, 38, 40, 11, 43, 44, 45, 46, 39, 47]
49 Ops.SOURCE dtypes.void [] "typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_5_5(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+1));\n  int val1 = (*(((int*)(val0))+4));\n  int4 val2 = (*((int4*)((((int*)(val0))+0))));\n  unsigned long val3 = (*(((unsigned long*)(args_16))+0));\n  int cast0 = ((int)((val2[0]!=0)));\n  int alu0 = (cast0+((int)((val2[1]!=0))));\n  int alu1 = (alu0+((int)((val2[2]!=0))));\n  int alu2 = (alu1+((int)((val2[3]!=0))));\n  *(((int*)(val3))+4) = (alu2+((int)((val1!=0))));\n  *((int4*)((((int*)(val3))+0))) = (int4){cast0,alu0,alu1,alu2};\n}\n"
50 Ops.BINARY dtypes.u8 [] b"typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_5_5(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+1));\n  int val1 = (*(((int*)(val0))+4));\n  int4 val2 = (*((int4*)((((int*)(val0))+0))));\n  unsigned long val3 = (*(((unsigned long*)(args_16))+0));\n  int cast0 = ((int)((val2[0]!=0)));\n  int alu0 = (cast0+((int)((val2[1]!=0))));\n  int alu1 = (alu0+((int)((val2[2]!=0))));\n  int alu2 = (alu1+((int)((val2[3]!=0))));\n  *(((int*)(val3))+4) = (alu2+((int)((val1!=0))));\n  *((int4*)((((int*)(val3))+0))) = (int4){cast0,alu0,alu1,alu2};\n}\n"
51 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
52 Ops.PARAM dtypes.i32 [0] ParamArg(slot=1, dtype=dtypes.i32)
53 Ops.PROGRAM dtypes.void [47, 48, 49, 50] ProgramInfo(params=(%51, %52), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
54 Ops.CONST dtypes.weakint [] 512
55 Ops.ALLOC dtypes.i8 [54] ParamArg(slot=7, dtype=dtypes.i8, device="CPU")
56 Ops.CONST dtypes.weakint [] 20
57 Ops.SHRINK dtypes.i8 [55, 4, 56]
58 Ops.BITCAST dtypes.i32 [57] dtypes.i32
59 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32, device="CPU")
60 Ops.CALL dtypes.void [53, 58, 59] CallInfo()
61 Ops.REFPARAM dtypes.i32 [32, 7] ParamArg(slot=0, dtype=dtypes.i32)
62 Ops.SHRINK dtypes.i32 [61, 5, 27]
63 Ops.STACK dtypes.i32 [19, 22, 28, 34, 42, 19, 22, 28, 34, 42, 19, 22, 28, 34, 42]
64 Ops.STACK dtypes.i32 [5, 5, 5, 5, 5, 13, 13, 13, 13, 13, 27, 27, 27, 27, 27]
65 Ops.CMPEQ dtypes.bool [63, 64]
66 Ops.WHERE dtypes.i32 [65, 13, 5]
67 Ops.INDEX dtypes.i32 [66, 5]
68 Ops.INDEX dtypes.i32 [66, 13]
69 Ops.ADD dtypes.i32 [67, 68]
70 Ops.INDEX dtypes.i32 [66, 27]
71 Ops.ADD dtypes.i32 [69, 70]
72 Ops.INDEX dtypes.i32 [66, 33]
73 Ops.ADD dtypes.i32 [71, 72]
74 Ops.INDEX dtypes.i32 [66, 10]
75 Ops.ADD dtypes.i32 [73, 74]
76 Ops.CAST dtypes.i32 [0] dtypes.i32
77 Ops.INDEX dtypes.i32 [66, 76]
78 Ops.CONST dtypes.weakint [] 6
79 Ops.CAST dtypes.i32 [78] dtypes.i32
80 Ops.INDEX dtypes.i32 [66, 79]
81 Ops.ADD dtypes.i32 [77, 80]
82 Ops.CONST dtypes.weakint [] 7
83 Ops.CAST dtypes.i32 [82] dtypes.i32
84 Ops.INDEX dtypes.i32 [66, 83]
85 Ops.ADD dtypes.i32 [81, 84]
86 Ops.CONST dtypes.weakint [] 8
87 Ops.CAST dtypes.i32 [86] dtypes.i32
88 Ops.INDEX dtypes.i32 [66, 87]
89 Ops.ADD dtypes.i32 [85, 88]
90 Ops.CONST dtypes.weakint [] 9
91 Ops.CAST dtypes.i32 [90] dtypes.i32
92 Ops.INDEX dtypes.i32 [66, 91]
93 Ops.ADD dtypes.i32 [89, 92]
94 Ops.STACK dtypes.i32 [75, 93]
95 Ops.STORE dtypes.void [62, 94]
96 Ops.INDEX dtypes.i32 [61, 27]
97 Ops.CONST dtypes.weakint [] 10
98 Ops.CAST dtypes.i32 [97] dtypes.i32
99 Ops.INDEX dtypes.i32 [66, 98]
100 Ops.CONST dtypes.weakint [] 11
101 Ops.CAST dtypes.i32 [100] dtypes.i32
102 Ops.INDEX dtypes.i32 [66, 101]
103 Ops.ADD dtypes.i32 [99, 102]
104 Ops.CONST dtypes.weakint [] 12
105 Ops.CAST dtypes.i32 [104] dtypes.i32
106 Ops.INDEX dtypes.i32 [66, 105]
107 Ops.ADD dtypes.i32 [103, 106]
108 Ops.CONST dtypes.weakint [] 13
109 Ops.CAST dtypes.i32 [108] dtypes.i32
110 Ops.INDEX dtypes.i32 [66, 109]
111 Ops.ADD dtypes.i32 [107, 110]
112 Ops.CONST dtypes.weakint [] 14
113 Ops.CAST dtypes.i32 [112] dtypes.i32
114 Ops.INDEX dtypes.i32 [66, 113]
115 Ops.ADD dtypes.i32 [111, 114]
116 Ops.STORE dtypes.void [96, 115]
117 Ops.SINK dtypes.void [95, 95, 116] KernelInfo(name="r_3_5", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST))), estimates=Estimates(ops=42, lds=32, mem=32), optimized=True)
118 Ops.LINEAR dtypes.void [1, 2, 12, 13, 3, 14, 15, 0, 16, 9, 10, 41, 42, 4, 5, 17, 18, 6, 7, 32, 61, 19, 22, 26, 27, 28, 33, 34, 63, 64, 65, 66, 67, 68, 70, 72, 74, 76, 77, 78, 79, 80, 82, 83, 84, 86, 87, 88, 90, 91, 92, 69, 81, 71, 85, 73, 89, 75, 93, 94, 96, 97, 98, 99, 100, 101, 102, 104, 105, 106, 108, 109, 110, 112, 113, 114, 62, 103, 107, 111, 115, 116, 95, 117]
119 Ops.SOURCE dtypes.void [] "typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int15 __attribute__((aligned(4),ext_vector_type(15)));\ntypedef int int2 __attribute__((aligned(4),ext_vector_type(2)));\n__attribute__((min_vector_width(512))) void r_3_5(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+1));\n  int val1 = (*(((int*)(val0))+4));\n  int4 val2 = (*((int4*)((((int*)(val0))+0))));\n  unsigned long val3 = (*(((unsigned long*)(args_16))+0));\n  int15 alu0 = (((int15){val2[0],val2[1],val2[2],val2[3],val1,val2[0],val2[1],val2[2],val2[3],val1,val2[0],val2[1],val2[2],val2[3],val1}==(int15){0,0,0,0,0,1,1,1,1,1,2,2,2,2,2})?((int15)(1)):0);\n  *(((int*)(val3))+2) = (alu0[10]+alu0[11]+alu0[12]+alu0[13]+alu0[14]);\n  *((int2*)((((int*)(val3))+0))) = (int2){(alu0[0]+alu0[1]+alu0[2]+alu0[3]+alu0[4]),(alu0[5]+alu0[6]+alu0[7]+alu0[8]+alu0[9])};\n}\n"
120 Ops.BINARY dtypes.u8 [] b"typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int15 __attribute__((aligned(4),ext_vector_type(15)));\ntypedef int int2 __attribute__((aligned(4),ext_vector_type(2)));\n__attribute__((min_vector_width(512))) void r_3_5(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+1));\n  int val1 = (*(((int*)(val0))+4));\n  int4 val2 = (*((int4*)((((int*)(val0))+0))));\n  unsigned long val3 = (*(((unsigned long*)(args_16))+0));\n  int15 alu0 = (((int15){val2[0],val2[1],val2[2],val2[3],val1,val2[0],val2[1],val2[2],val2[3],val1,val2[0],val2[1],val2[2],val2[3],val1}==(int15){0,0,0,0,0,1,1,1,1,1,2,2,2,2,2})?((int15)(1)):0);\n  *(((int*)(val3))+2) = (alu0[10]+alu0[11]+alu0[12]+alu0[13]+alu0[14]);\n  *((int2*)((((int*)(val3))+0))) = (int2){(alu0[0]+alu0[1]+alu0[2]+alu0[3]+alu0[4]),(alu0[5]+alu0[6]+alu0[7]+alu0[8]+alu0[9])};\n}\n"
121 Ops.PARAM dtypes.i32 [32] ParamArg(slot=0, dtype=dtypes.i32)
122 Ops.PROGRAM dtypes.void [117, 118, 119, 120] ProgramInfo(params=(%121, %52), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
123 Ops.CONST dtypes.weakint [] 256
124 Ops.SHRINK dtypes.i8 [55, 123, 104]
125 Ops.BITCAST dtypes.i32 [124] dtypes.i32
126 Ops.CALL dtypes.void [122, 125, 58] CallInfo()
127 Ops.REFPARAM dtypes.i32 [32, 15] ParamArg(slot=1, dtype=dtypes.i32)
128 Ops.SHRINK dtypes.i32 [127, 5, 27]
129 Ops.LOAD dtypes.i32 [128]
130 Ops.INDEX dtypes.i32 [129, 5]
131 Ops.STACK dtypes.i32 [5, 5, 130]
132 Ops.CAST dtypes.u32 [131] dtypes.u32
133 Ops.INDEX dtypes.i32 [129, 13]
134 Ops.STACK dtypes.i32 [5, 130, 133]
135 Ops.CAST dtypes.u32 [134] dtypes.u32
136 Ops.ADD dtypes.u32 [132, 135]
137 Ops.CAST dtypes.i32 [136] dtypes.i32
138 Ops.CAST dtypes.u32 [137] dtypes.u32
139 Ops.INDEX dtypes.i32 [127, 27]
140 Ops.LOAD dtypes.i32 [139]
141 Ops.STACK dtypes.i32 [130, 133, 140]
142 Ops.CAST dtypes.u32 [141] dtypes.u32
143 Ops.ADD dtypes.u32 [138, 142]
144 Ops.CAST dtypes.i32 [143] dtypes.i32
145 Ops.CMPLT dtypes.bool [144, 5]
146 Ops.CAST dtypes.u32 [144] dtypes.u32
147 Ops.CAST dtypes.u32 [0] dtypes.u32
148 Ops.ADD dtypes.u32 [146, 147]
149 Ops.CAST dtypes.i32 [148] dtypes.i32
150 Ops.WHERE dtypes.i32 [145, 149, 144]
151 Ops.INDEX dtypes.i32 [150, 5]
152 Ops.INDEX dtypes.i32 [150, 13]
153 Ops.STACK dtypes.i32 [151, 152]
154 Ops.STORE dtypes.void [62, 153]
155 Ops.INDEX dtypes.i32 [150, 27]
156 Ops.STORE dtypes.void [96, 155]
157 Ops.SINK dtypes.void [154, 154, 156] KernelInfo(name="r_3_3", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST))), estimates=Estimates(ops=15, lds=24, mem=24), optimized=True)
158 Ops.LINEAR dtypes.void [1, 2, 12, 13, 3, 14, 15, 32, 127, 26, 27, 139, 140, 4, 5, 128, 129, 6, 7, 61, 130, 133, 141, 134, 131, 135, 132, 136, 137, 142, 138, 143, 144, 146, 0, 147, 148, 149, 145, 150, 151, 152, 153, 96, 155, 62, 156, 154, 157]
159 Ops.SOURCE dtypes.void [] "typedef int int2 __attribute__((aligned(4),ext_vector_type(2)));\ntypedef int int3 __attribute__((aligned(4),ext_vector_type(3)));\ntypedef unsigned int unsigned_int3 __attribute__((aligned(4),ext_vector_type(3)));\n__attribute__((min_vector_width(512))) void r_3_3(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+1));\n  int val1 = (*(((int*)(val0))+2));\n  int2 val2 = (*((int2*)((((int*)(val0))+0))));\n  unsigned long val3 = (*(((unsigned long*)(args_16))+0));\n  unsigned_int3 cast0 = __builtin_convertvector((int3){0,val2[0],val2[1]}, unsigned_int3);\n  unsigned_int3 cast1 = __builtin_convertvector((int3){0,0,val2[0]}, unsigned_int3);\n  int3 cast2 = __builtin_convertvector((cast1+cast0), int3);\n  unsigned_int3 cast3 = __builtin_convertvector((int3){val2[0],val2[1],val1}, unsigned_int3);\n  unsigned_int3 cast4 = __builtin_convertvector(cast2, unsigned_int3);\n  int3 cast5 = __builtin_convertvector((cast4+cast3), int3);\n  unsigned_int3 cast6 = __builtin_convertvector(cast5, unsigned_int3);\n  int3 cast7 = __builtin_convertvector((cast6+5u), int3);\n  int3 alu0 = ((cast5<0)?cast7:cast5);\n  *(((int*)(val3))+2) = alu0[2];\n  *((int2*)((((int*)(val3))+0))) = (int2){alu0[0],alu0[1]};\n}\n"
160 Ops.BINARY dtypes.u8 [] b"typedef int int2 __attribute__((aligned(4),ext_vector_type(2)));\ntypedef int int3 __attribute__((aligned(4),ext_vector_type(3)));\ntypedef unsigned int unsigned_int3 __attribute__((aligned(4),ext_vector_type(3)));\n__attribute__((min_vector_width(512))) void r_3_3(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+1));\n  int val1 = (*(((int*)(val0))+2));\n  int2 val2 = (*((int2*)((((int*)(val0))+0))));\n  unsigned long val3 = (*(((unsigned long*)(args_16))+0));\n  unsigned_int3 cast0 = __builtin_convertvector((int3){0,val2[0],val2[1]}, unsigned_int3);\n  unsigned_int3 cast1 = __builtin_convertvector((int3){0,0,val2[0]}, unsigned_int3);\n  int3 cast2 = __builtin_convertvector((cast1+cast0), int3);\n  unsigned_int3 cast3 = __builtin_convertvector((int3){val2[0],val2[1],val1}, unsigned_int3);\n  unsigned_int3 cast4 = __builtin_convertvector(cast2, unsigned_int3);\n  int3 cast5 = __builtin_convertvector((cast4+cast3), int3);\n  unsigned_int3 cast6 = __builtin_convertvector(cast5, unsigned_int3);\n  int3 cast7 = __builtin_convertvector((cast6+5u), int3);\n  int3 alu0 = ((cast5<0)?cast7:cast5);\n  *(((int*)(val3))+2) = alu0[2];\n  *((int2*)((((int*)(val3))+0))) = (int2){alu0[0],alu0[1]};\n}\n"
161 Ops.PARAM dtypes.i32 [32] ParamArg(slot=1, dtype=dtypes.i32)
162 Ops.PROGRAM dtypes.void [157, 158, 159, 160] ProgramInfo(params=(%121, %161), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
163 Ops.SHRINK dtypes.i8 [55, 4, 104]
164 Ops.BITCAST dtypes.i32 [163] dtypes.i32
165 Ops.CALL dtypes.void [162, 164, 125] CallInfo()
166 Ops.CONST dtypes.weakint [] 24
167 Ops.PARAM dtypes.u8 [166] ParamArg(slot=0, dtype=dtypes.u8, name="args")
168 Ops.BITCAST dtypes.u64 [167] dtypes.u64
169 Ops.INDEX dtypes.u64 [168, 5]
170 Ops.LOAD dtypes.u64 [169]
171 Ops.REFPARAM dtypes.i32 [32, 170] ParamArg(slot=0, dtype=dtypes.i32)
172 Ops.SHRINK dtypes.i32 [171, 5, 27]
173 Ops.INDEX dtypes.u64 [168, 13]
174 Ops.LOAD dtypes.u64 [173]
175 Ops.REFPARAM dtypes.i32 [0, 174] ParamArg(slot=1, dtype=dtypes.i32)
176 Ops.SHRINK dtypes.i32 [175, 5, 10]
177 Ops.LOAD dtypes.i32 [176]
178 Ops.INDEX dtypes.i32 [177, 5]
179 Ops.INDEX dtypes.i32 [177, 13]
180 Ops.INDEX dtypes.i32 [177, 27]
181 Ops.INDEX dtypes.i32 [177, 33]
182 Ops.INDEX dtypes.i32 [175, 10]
183 Ops.LOAD dtypes.i32 [182]
184 Ops.STACK dtypes.i32 [178, 179, 180, 181, 183]
185 Ops.CMPNE dtypes.bool [184, 5]
186 Ops.CAST dtypes.i32 [185] dtypes.i32
187 Ops.INDEX dtypes.i32 [186, 5]
188 Ops.INDEX dtypes.i32 [186, 13]
189 Ops.ADD dtypes.i32 [187, 188]
190 Ops.INDEX dtypes.i32 [186, 27]
191 Ops.ADD dtypes.i32 [189, 190]
192 Ops.INDEX dtypes.i32 [186, 33]
193 Ops.ADD dtypes.i32 [191, 192]
194 Ops.INDEX dtypes.i32 [186, 10]
195 Ops.ADD dtypes.i32 [193, 194]
196 Ops.CMPLT dtypes.bool [5, 195]
197 Ops.CONST dtypes.weakint [] -1
198 Ops.CAST dtypes.i32 [197] dtypes.i32
199 Ops.INDEX dtypes.u64 [168, 27]
200 Ops.LOAD dtypes.u64 [199]
201 Ops.REFPARAM dtypes.i32 [32, 200] ParamArg(slot=2, dtype=dtypes.i32)
202 Ops.SHRINK dtypes.i32 [201, 5, 27]
203 Ops.LOAD dtypes.i32 [202]
204 Ops.INDEX dtypes.i32 [203, 5]
205 Ops.CMPLT dtypes.bool [198, 204]
206 Ops.CMPLT dtypes.bool [204, 76]
207 Ops.AND dtypes.bool [205, 206]
208 Ops.WHERE dtypes.i32 [207, 204, 5]
209 Ops.WHERE dtypes.i32 [196, 208, 198]
210 Ops.CMPLT dtypes.bool [13, 195]
211 Ops.INDEX dtypes.i32 [203, 13]
212 Ops.CMPLT dtypes.bool [198, 211]
213 Ops.CMPLT dtypes.bool [211, 76]
214 Ops.AND dtypes.bool [212, 213]
215 Ops.WHERE dtypes.i32 [214, 211, 5]
216 Ops.WHERE dtypes.i32 [210, 215, 198]
217 Ops.STACK dtypes.i32 [209, 216]
218 Ops.STORE dtypes.void [172, 217]
219 Ops.INDEX dtypes.i32 [171, 27]
220 Ops.CMPLT dtypes.bool [27, 195]
221 Ops.INDEX dtypes.i32 [201, 27]
222 Ops.LOAD dtypes.i32 [221]
223 Ops.CMPLT dtypes.bool [198, 222]
224 Ops.CMPLT dtypes.bool [222, 76]
225 Ops.AND dtypes.bool [223, 224]
226 Ops.WHERE dtypes.i32 [225, 222, 5]
227 Ops.WHERE dtypes.i32 [220, 226, 198]
228 Ops.STORE dtypes.void [219, 227]
229 Ops.SINK dtypes.void [218, 218, 228] KernelInfo(name="r_3_5", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST))), estimates=Estimates(ops=27, lds=44, mem=44), optimized=True)
230 Ops.LINEAR dtypes.void [166, 167, 12, 13, 168, 173, 174, 0, 175, 9, 10, 182, 183, 26, 27, 199, 200, 32, 201, 221, 222, 4, 5, 176, 177, 202, 203, 169, 170, 171, 178, 179, 180, 33, 181, 184, 204, 211, 185, 186, 187, 188, 190, 192, 194, 197, 198, 76, 189, 191, 193, 195, 206, 213, 205, 212, 196, 210, 207, 214, 208, 209, 215, 216, 217, 219, 172, 224, 223, 220, 225, 226, 227, 228, 218, 229]
231 Ops.SOURCE dtypes.void [] "typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int2 __attribute__((aligned(4),ext_vector_type(2)));\ntypedef int int5 __attribute__((aligned(4),ext_vector_type(5)));\n__attribute__((min_vector_width(512))) void r_3_5(unsigned char* restrict args_24) {\n  unsigned long val0 = (*(((unsigned long*)(args_24))+1));\n  int val1 = (*(((int*)(val0))+4));\n  unsigned long val2 = (*(((unsigned long*)(args_24))+2));\n  int val3 = (*(((int*)(val2))+2));\n  int4 val4 = (*((int4*)((((int*)(val0))+0))));\n  int2 val5 = (*((int2*)((((int*)(val2))+0))));\n  unsigned long val6 = (*(((unsigned long*)(args_24))+0));\n  int5 cast0 = __builtin_convertvector((((int5){val4[0],val4[1],val4[2],val4[3],val1}!=0)&1), int5);\n  int alu0 = (cast0[0]+cast0[1]+cast0[2]+cast0[3]+cast0[4]);\n  int alu1 = (((-1<val5[0])&(val5[0]<5))?val5[0]:0);\n  int alu2 = ((0<alu0)?alu1:-1);\n  int alu3 = (((-1<val5[1])&(val5[1]<5))?val5[1]:0);\n  int alu4 = ((1<alu0)?alu3:-1);\n  int alu5 = (((-1<val3)&(val3<5))?val3:0);\n  int alu6 = ((2<alu0)?alu5:-1);\n  *(((int*)(val6))+2) = alu6;\n  *((int2*)((((int*)(val6))+0))) = (int2){alu2,alu4};\n}\n"
232 Ops.BINARY dtypes.u8 [] b"typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int2 __attribute__((aligned(4),ext_vector_type(2)));\ntypedef int int5 __attribute__((aligned(4),ext_vector_type(5)));\n__attribute__((min_vector_width(512))) void r_3_5(unsigned char* restrict args_24) {\n  unsigned long val0 = (*(((unsigned long*)(args_24))+1));\n  int val1 = (*(((int*)(val0))+4));\n  unsigned long val2 = (*(((unsigned long*)(args_24))+2));\n  int val3 = (*(((int*)(val2))+2));\n  int4 val4 = (*((int4*)((((int*)(val0))+0))));\n  int2 val5 = (*((int2*)((((int*)(val2))+0))));\n  unsigned long val6 = (*(((unsigned long*)(args_24))+0));\n  int5 cast0 = __builtin_convertvector((((int5){val4[0],val4[1],val4[2],val4[3],val1}!=0)&1), int5);\n  int alu0 = (cast0[0]+cast0[1]+cast0[2]+cast0[3]+cast0[4]);\n  int alu1 = (((-1<val5[0])&(val5[0]<5))?val5[0]:0);\n  int alu2 = ((0<alu0)?alu1:-1);\n  int alu3 = (((-1<val5[1])&(val5[1]<5))?val5[1]:0);\n  int alu4 = ((1<alu0)?alu3:-1);\n  int alu5 = (((-1<val3)&(val3<5))?val3:0);\n  int alu6 = ((2<alu0)?alu5:-1);\n  *(((int*)(val6))+2) = alu6;\n  *((int2*)((((int*)(val6))+0))) = (int2){alu2,alu4};\n}\n"
233 Ops.PARAM dtypes.i32 [32] ParamArg(slot=2, dtype=dtypes.i32)
234 Ops.PROGRAM dtypes.void [229, 230, 231, 232] ProgramInfo(params=(%121, %52, %233), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CPU", arch="x86_64,x86-64"))
235 Ops.BUFFER dtypes.i32 [32] ParamArg(slot=2, dtype=dtypes.i32, device="CPU")
236 Ops.CALL dtypes.void [234, 235, 59, 164] CallInfo()
237 Ops.LINEAR dtypes.void [60, 126, 165, 236]

