# tolk D141 D151 golden=bc51ba39dde71b62b804d403ddd1ddf0
0 Ops.CONST dtypes.weakint [] 4
1 Ops.CONST dtypes.weakint [] 16
2 Ops.PARAM dtypes.u8 [1] ParamArg(slot=0, dtype=dtypes.u8, name="args")
3 Ops.BITCAST dtypes.u64 [2] dtypes.u64
4 Ops.CONST dtypes.weakint [] 0
5 Ops.CAST dtypes.i32 [4] dtypes.i32
6 Ops.INDEX dtypes.u64 [3, 5]
7 Ops.LOAD dtypes.u64 [6]
8 Ops.REFPARAM dtypes.i32 [0, 7] ParamArg(slot=0, dtype=dtypes.i32)
9 Ops.CAST dtypes.i32 [0] dtypes.i32
10 Ops.SHRINK dtypes.i32 [8, 5, 9]
11 Ops.CONST dtypes.weakint [] 1
12 Ops.CAST dtypes.i32 [11] dtypes.i32
13 Ops.INDEX dtypes.u64 [3, 12]
14 Ops.LOAD dtypes.u64 [13]
15 Ops.REFPARAM dtypes.bool [0, 14] ParamArg(slot=1, dtype=dtypes.bool)
16 Ops.INDEX dtypes.bool [15, 5]
17 Ops.LOAD dtypes.bool [16]
18 Ops.CAST dtypes.i32 [17] dtypes.i32
19 Ops.INDEX dtypes.bool [15, 12]
20 Ops.LOAD dtypes.bool [19]
21 Ops.CAST dtypes.i32 [20] dtypes.i32
22 Ops.ADD dtypes.i32 [18, 21]
23 Ops.CONST dtypes.weakint [] 2
24 Ops.CAST dtypes.i32 [23] dtypes.i32
25 Ops.INDEX dtypes.bool [15, 24]
26 Ops.LOAD dtypes.bool [25]
27 Ops.CAST dtypes.i32 [26] dtypes.i32
28 Ops.ADD dtypes.i32 [22, 27]
29 Ops.CONST dtypes.weakint [] 3
30 Ops.CAST dtypes.i32 [29] dtypes.i32
31 Ops.INDEX dtypes.bool [15, 30]
32 Ops.LOAD dtypes.bool [31]
33 Ops.CAST dtypes.i32 [32] dtypes.i32
34 Ops.ADD dtypes.i32 [28, 33]
35 Ops.STACK dtypes.i32 [18, 22, 28, 34]
36 Ops.STORE dtypes.void [10, 35]
37 Ops.SINK dtypes.void [36, 36, 36, 36] KernelInfo(name="r_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST))), estimates=Estimates(ops=3, lds=20, mem=20), optimized=True)
38 Ops.LINEAR dtypes.void [1, 2, 11, 12, 3, 13, 14, 0, 15, 4, 5, 16, 17, 19, 20, 23, 24, 25, 26, 29, 30, 31, 32, 6, 7, 8, 18, 21, 27, 33, 22, 28, 34, 35, 9, 10, 36, 37]
39 Ops.SOURCE dtypes.void [] "typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_4_4(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+1));\n  _Bool val1 = (*(((_Bool*)(val0))+0));\n  _Bool val2 = (*(((_Bool*)(val0))+1));\n  _Bool val3 = (*(((_Bool*)(val0))+2));\n  _Bool val4 = (*(((_Bool*)(val0))+3));\n  unsigned long val5 = (*(((unsigned long*)(args_16))+0));\n  int cast0 = ((int)(val1));\n  int alu0 = (cast0+((int)(val2)));\n  int alu1 = (alu0+((int)(val3)));\n  *((int4*)((((int*)(val5))+0))) = (int4){cast0,alu0,alu1,(alu1+((int)(val4)))};\n}\n"
40 Ops.BINARY dtypes.u8 [] b"typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_4_4(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+1));\n  _Bool val1 = (*(((_Bool*)(val0))+0));\n  _Bool val2 = (*(((_Bool*)(val0))+1));\n  _Bool val3 = (*(((_Bool*)(val0))+2));\n  _Bool val4 = (*(((_Bool*)(val0))+3));\n  unsigned long val5 = (*(((unsigned long*)(args_16))+0));\n  int cast0 = ((int)(val1));\n  int alu0 = (cast0+((int)(val2)));\n  int alu1 = (alu0+((int)(val3)));\n  *((int4*)((((int*)(val5))+0))) = (int4){cast0,alu0,alu1,(alu1+((int)(val4)))};\n}\n"
41 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
42 Ops.PARAM dtypes.bool [0] ParamArg(slot=1, dtype=dtypes.bool)
43 Ops.PROGRAM dtypes.void [37, 38, 39, 40] ProgramInfo(params=(%41, %42), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
44 Ops.CONST dtypes.weakint [] 512
45 Ops.ALLOC dtypes.i8 [44] ParamArg(slot=9, dtype=dtypes.i8, device="CPU")
46 Ops.SHRINK dtypes.i8 [45, 4, 1]
47 Ops.BITCAST dtypes.i32 [46] dtypes.i32
48 Ops.PARAM dtypes.bool [0] ParamArg(slot=1, dtype=dtypes.bool, device="CPU")
49 Ops.CALL dtypes.void [43, 47, 48] CallInfo()
50 Ops.REFPARAM dtypes.i32 [0, 14] ParamArg(slot=1, dtype=dtypes.i32)
51 Ops.SHRINK dtypes.i32 [50, 5, 9]
52 Ops.LOAD dtypes.i32 [51]
53 Ops.INDEX dtypes.i32 [52, 5]
54 Ops.INDEX dtypes.i32 [52, 12]
55 Ops.INDEX dtypes.i32 [52, 24]
56 Ops.INDEX dtypes.i32 [52, 30]
57 Ops.STACK dtypes.i32 [53, 54, 55, 56, 53, 54, 55, 56, 53, 54, 55, 56, 53, 54, 55, 56]
58 Ops.STACK dtypes.i32 [5, 5, 5, 5, 12, 12, 12, 12, 24, 24, 24, 24, 30, 30, 30, 30]
59 Ops.CMPEQ dtypes.bool [57, 58]
60 Ops.WHERE dtypes.i32 [59, 12, 5]
61 Ops.INDEX dtypes.i32 [60, 5]
62 Ops.INDEX dtypes.i32 [60, 12]
63 Ops.ADD dtypes.i32 [61, 62]
64 Ops.INDEX dtypes.i32 [60, 24]
65 Ops.ADD dtypes.i32 [63, 64]
66 Ops.INDEX dtypes.i32 [60, 30]
67 Ops.ADD dtypes.i32 [65, 66]
68 Ops.INDEX dtypes.i32 [60, 9]
69 Ops.CONST dtypes.weakint [] 5
70 Ops.CAST dtypes.i32 [69] dtypes.i32
71 Ops.INDEX dtypes.i32 [60, 70]
72 Ops.ADD dtypes.i32 [68, 71]
73 Ops.CONST dtypes.weakint [] 6
74 Ops.CAST dtypes.i32 [73] dtypes.i32
75 Ops.INDEX dtypes.i32 [60, 74]
76 Ops.ADD dtypes.i32 [72, 75]
77 Ops.CONST dtypes.weakint [] 7
78 Ops.CAST dtypes.i32 [77] dtypes.i32
79 Ops.INDEX dtypes.i32 [60, 78]
80 Ops.ADD dtypes.i32 [76, 79]
81 Ops.CONST dtypes.weakint [] 8
82 Ops.CAST dtypes.i32 [81] dtypes.i32
83 Ops.INDEX dtypes.i32 [60, 82]
84 Ops.CONST dtypes.weakint [] 9
85 Ops.CAST dtypes.i32 [84] dtypes.i32
86 Ops.INDEX dtypes.i32 [60, 85]
87 Ops.ADD dtypes.i32 [83, 86]
88 Ops.CONST dtypes.weakint [] 10
89 Ops.CAST dtypes.i32 [88] dtypes.i32
90 Ops.INDEX dtypes.i32 [60, 89]
91 Ops.ADD dtypes.i32 [87, 90]
92 Ops.CONST dtypes.weakint [] 11
93 Ops.CAST dtypes.i32 [92] dtypes.i32
94 Ops.INDEX dtypes.i32 [60, 93]
95 Ops.ADD dtypes.i32 [91, 94]
96 Ops.CONST dtypes.weakint [] 12
97 Ops.CAST dtypes.i32 [96] dtypes.i32
98 Ops.INDEX dtypes.i32 [60, 97]
99 Ops.CONST dtypes.weakint [] 13
100 Ops.CAST dtypes.i32 [99] dtypes.i32
101 Ops.INDEX dtypes.i32 [60, 100]
102 Ops.ADD dtypes.i32 [98, 101]
103 Ops.CONST dtypes.weakint [] 14
104 Ops.CAST dtypes.i32 [103] dtypes.i32
105 Ops.INDEX dtypes.i32 [60, 104]
106 Ops.ADD dtypes.i32 [102, 105]
107 Ops.CONST dtypes.weakint [] 15
108 Ops.CAST dtypes.i32 [107] dtypes.i32
109 Ops.INDEX dtypes.i32 [60, 108]
110 Ops.ADD dtypes.i32 [106, 109]
111 Ops.STACK dtypes.i32 [67, 80, 95, 110]
112 Ops.STORE dtypes.void [10, 111]
113 Ops.SINK dtypes.void [112, 112, 112, 112] KernelInfo(name="r_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST))), estimates=Estimates(ops=44, lds=32, mem=32), optimized=True)
114 Ops.LINEAR dtypes.void [1, 2, 11, 12, 3, 13, 14, 0, 50, 4, 5, 9, 51, 52, 6, 7, 8, 53, 54, 23, 24, 55, 29, 30, 56, 57, 58, 59, 60, 61, 62, 88, 89, 90, 92, 93, 94, 96, 97, 98, 99, 100, 101, 103, 104, 105, 107, 108, 109, 64, 66, 68, 69, 70, 71, 73, 74, 75, 77, 78, 79, 81, 82, 83, 84, 85, 86, 63, 102, 72, 87, 65, 106, 76, 91, 67, 110, 80, 95, 111, 10, 112, 113]
115 Ops.SOURCE dtypes.void [] "typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int16 __attribute__((aligned(4),ext_vector_type(16)));\n__attribute__((min_vector_width(512))) void r_4_4(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+1));\n  int4 val1 = (*((int4*)((((int*)(val0))+0))));\n  unsigned long val2 = (*(((unsigned long*)(args_16))+0));\n  int16 alu0 = (((int16){val1[0],val1[1],val1[2],val1[3],val1[0],val1[1],val1[2],val1[3],val1[0],val1[1],val1[2],val1[3],val1[0],val1[1],val1[2],val1[3]}==(int16){0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3})?((int16)(1)):0);\n  *((int4*)((((int*)(val2))+0))) = (int4){(alu0[0]+alu0[1]+alu0[2]+alu0[3]),(alu0[4]+alu0[5]+alu0[6]+alu0[7]),(alu0[8]+alu0[9]+alu0[10]+alu0[11]),(alu0[12]+alu0[13]+alu0[14]+alu0[15])};\n}\n"
116 Ops.BINARY dtypes.u8 [] b"typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int16 __attribute__((aligned(4),ext_vector_type(16)));\n__attribute__((min_vector_width(512))) void r_4_4(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+1));\n  int4 val1 = (*((int4*)((((int*)(val0))+0))));\n  unsigned long val2 = (*(((unsigned long*)(args_16))+0));\n  int16 alu0 = (((int16){val1[0],val1[1],val1[2],val1[3],val1[0],val1[1],val1[2],val1[3],val1[0],val1[1],val1[2],val1[3],val1[0],val1[1],val1[2],val1[3]}==(int16){0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3})?((int16)(1)):0);\n  *((int4*)((((int*)(val2))+0))) = (int4){(alu0[0]+alu0[1]+alu0[2]+alu0[3]),(alu0[4]+alu0[5]+alu0[6]+alu0[7]),(alu0[8]+alu0[9]+alu0[10]+alu0[11]),(alu0[12]+alu0[13]+alu0[14]+alu0[15])};\n}\n"
117 Ops.PARAM dtypes.i32 [0] ParamArg(slot=1, dtype=dtypes.i32)
118 Ops.PROGRAM dtypes.void [113, 114, 115, 116] ProgramInfo(params=(%41, %117), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
119 Ops.CONST dtypes.weakint [] 256
120 Ops.SHRINK dtypes.i8 [45, 119, 1]
121 Ops.BITCAST dtypes.i32 [120] dtypes.i32
122 Ops.CALL dtypes.void [118, 121, 47] CallInfo()
123 Ops.CMPLT dtypes.bool [53, 5]
124 Ops.CAST dtypes.u32 [53] dtypes.u32
125 Ops.CAST dtypes.u32 [0] dtypes.u32
126 Ops.ADD dtypes.u32 [124, 125]
127 Ops.CAST dtypes.i32 [126] dtypes.i32
128 Ops.WHERE dtypes.i32 [123, 127, 53]
129 Ops.CAST dtypes.u32 [54] dtypes.u32
130 Ops.ADD dtypes.u32 [124, 129]
131 Ops.CAST dtypes.i32 [130] dtypes.i32
132 Ops.CMPLT dtypes.bool [131, 5]
133 Ops.CAST dtypes.u32 [131] dtypes.u32
134 Ops.ADD dtypes.u32 [133, 125]
135 Ops.CAST dtypes.i32 [134] dtypes.i32
136 Ops.WHERE dtypes.i32 [132, 135, 131]
137 Ops.CAST dtypes.u32 [55] dtypes.u32
138 Ops.ADD dtypes.u32 [133, 137]
139 Ops.CAST dtypes.i32 [138] dtypes.i32
140 Ops.CMPLT dtypes.bool [139, 5]
141 Ops.CAST dtypes.u32 [139] dtypes.u32
142 Ops.ADD dtypes.u32 [141, 125]
143 Ops.CAST dtypes.i32 [142] dtypes.i32
144 Ops.WHERE dtypes.i32 [140, 143, 139]
145 Ops.CAST dtypes.u32 [56] dtypes.u32
146 Ops.ADD dtypes.u32 [141, 145]
147 Ops.CAST dtypes.i32 [146] dtypes.i32
148 Ops.CMPLT dtypes.bool [147, 5]
149 Ops.CAST dtypes.u32 [147] dtypes.u32
150 Ops.ADD dtypes.u32 [149, 125]
151 Ops.CAST dtypes.i32 [150] dtypes.i32
152 Ops.WHERE dtypes.i32 [148, 151, 147]
153 Ops.STACK dtypes.i32 [128, 136, 144, 152]
154 Ops.STORE dtypes.void [10, 153]
155 Ops.SINK dtypes.void [154, 154, 154, 154] KernelInfo(name="r_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST))), estimates=Estimates(ops=15, lds=32, mem=32), optimized=True)
156 Ops.LINEAR dtypes.void [1, 2, 11, 12, 3, 13, 14, 0, 50, 4, 5, 9, 51, 52, 6, 7, 8, 53, 54, 23, 24, 55, 29, 30, 56, 124, 129, 130, 131, 125, 126, 127, 137, 133, 138, 139, 134, 135, 145, 141, 146, 147, 142, 143, 149, 150, 151, 123, 132, 140, 148, 128, 136, 144, 152, 153, 10, 154, 155]
157 Ops.SOURCE dtypes.void [] "typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_4_4(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+1));\n  int4 val1 = (*((int4*)((((int*)(val0))+0))));\n  unsigned long val2 = (*(((unsigned long*)(args_16))+0));\n  unsigned int cast0 = ((unsigned int)(val1[0]));\n  int cast1 = ((int)((cast0+((unsigned int)(val1[1])))));\n  unsigned int cast2 = ((unsigned int)(cast1));\n  int cast3 = ((int)((cast2+((unsigned int)(val1[2])))));\n  unsigned int cast4 = ((unsigned int)(cast3));\n  int cast5 = ((int)((cast4+((unsigned int)(val1[3])))));\n  int alu0 = ((val1[0]<0)?((int)((cast0+4u))):val1[0]);\n  int alu1 = ((cast1<0)?((int)((cast2+4u))):cast1);\n  int alu2 = ((cast3<0)?((int)((cast4+4u))):cast3);\n  int alu3 = ((cast5<0)?((int)((((unsigned int)(cast5))+4u))):cast5);\n  *((int4*)((((int*)(val2))+0))) = (int4){alu0,alu1,alu2,alu3};\n}\n"
158 Ops.BINARY dtypes.u8 [] b"typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_4_4(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+1));\n  int4 val1 = (*((int4*)((((int*)(val0))+0))));\n  unsigned long val2 = (*(((unsigned long*)(args_16))+0));\n  unsigned int cast0 = ((unsigned int)(val1[0]));\n  int cast1 = ((int)((cast0+((unsigned int)(val1[1])))));\n  unsigned int cast2 = ((unsigned int)(cast1));\n  int cast3 = ((int)((cast2+((unsigned int)(val1[2])))));\n  unsigned int cast4 = ((unsigned int)(cast3));\n  int cast5 = ((int)((cast4+((unsigned int)(val1[3])))));\n  int alu0 = ((val1[0]<0)?((int)((cast0+4u))):val1[0]);\n  int alu1 = ((cast1<0)?((int)((cast2+4u))):cast1);\n  int alu2 = ((cast3<0)?((int)((cast4+4u))):cast3);\n  int alu3 = ((cast5<0)?((int)((((unsigned int)(cast5))+4u))):cast5);\n  *((int4*)((((int*)(val2))+0))) = (int4){alu0,alu1,alu2,alu3};\n}\n"
159 Ops.PROGRAM dtypes.void [155, 156, 157, 158] ProgramInfo(params=(%41, %117), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
160 Ops.CALL dtypes.void [159, 47, 121] CallInfo()
161 Ops.CONST dtypes.weakint [] 32
162 Ops.PARAM dtypes.u8 [161] ParamArg(slot=0, dtype=dtypes.u8, name="args")
163 Ops.BITCAST dtypes.u64 [162] dtypes.u64
164 Ops.INDEX dtypes.u64 [163, 5]
165 Ops.LOAD dtypes.u64 [164]
166 Ops.REFPARAM dtypes.i32 [0, 165] ParamArg(slot=0, dtype=dtypes.i32)
167 Ops.SHRINK dtypes.i32 [166, 5, 9]
168 Ops.INDEX dtypes.u64 [163, 12]
169 Ops.LOAD dtypes.u64 [168]
170 Ops.REFPARAM dtypes.bool [0, 169] ParamArg(slot=1, dtype=dtypes.bool)
171 Ops.INDEX dtypes.bool [170, 5]
172 Ops.LOAD dtypes.bool [171]
173 Ops.INDEX dtypes.bool [170, 12]
174 Ops.LOAD dtypes.bool [173]
175 Ops.INDEX dtypes.bool [170, 24]
176 Ops.LOAD dtypes.bool [175]
177 Ops.INDEX dtypes.bool [170, 30]
178 Ops.LOAD dtypes.bool [177]
179 Ops.STACK dtypes.bool [172, 174, 176, 178]
180 Ops.CAST dtypes.i32 [179] dtypes.i32
181 Ops.INDEX dtypes.i32 [180, 5]
182 Ops.INDEX dtypes.i32 [180, 12]
183 Ops.ADD dtypes.i32 [181, 182]
184 Ops.INDEX dtypes.i32 [180, 24]
185 Ops.ADD dtypes.i32 [183, 184]
186 Ops.INDEX dtypes.i32 [180, 30]
187 Ops.ADD dtypes.i32 [185, 186]
188 Ops.CMPLT dtypes.bool [5, 187]
189 Ops.INDEX dtypes.u64 [163, 30]
190 Ops.LOAD dtypes.u64 [189]
191 Ops.REFPARAM dtypes.i32 [0, 190] ParamArg(slot=3, dtype=dtypes.i32)
192 Ops.INDEX dtypes.u64 [163, 24]
193 Ops.LOAD dtypes.u64 [192]
194 Ops.REFPARAM dtypes.i32 [0, 193] ParamArg(slot=2, dtype=dtypes.i32)
195 Ops.SHRINK dtypes.i32 [194, 5, 9]
196 Ops.LOAD dtypes.i32 [195]
197 Ops.INDEX dtypes.i32 [196, 5]
198 Ops.INDEX dtypes.i32 [191, 197]
199 Ops.CONST dtypes.weakint [] -1
200 Ops.CAST dtypes.i32 [199] dtypes.i32
201 Ops.CMPLT dtypes.bool [200, 197]
202 Ops.CMPLT dtypes.bool [197, 9]
203 Ops.AND dtypes.bool [201, 202]
204 Ops.LOAD dtypes.i32 [198, 5, 203]
205 Ops.WHERE dtypes.i32 [188, 204, 200]
206 Ops.CMPLT dtypes.bool [12, 187]
207 Ops.INDEX dtypes.i32 [196, 12]
208 Ops.INDEX dtypes.i32 [191, 207]
209 Ops.CMPLT dtypes.bool [200, 207]
210 Ops.CMPLT dtypes.bool [207, 9]
211 Ops.AND dtypes.bool [209, 210]
212 Ops.LOAD dtypes.i32 [208, 5, 211]
213 Ops.WHERE dtypes.i32 [206, 212, 200]
214 Ops.CMPLT dtypes.bool [24, 187]
215 Ops.INDEX dtypes.i32 [196, 24]
216 Ops.INDEX dtypes.i32 [191, 215]
217 Ops.CMPLT dtypes.bool [200, 215]
218 Ops.CMPLT dtypes.bool [215, 9]
219 Ops.AND dtypes.bool [217, 218]
220 Ops.LOAD dtypes.i32 [216, 5, 219]
221 Ops.WHERE dtypes.i32 [214, 220, 200]
222 Ops.CMPLT dtypes.bool [30, 187]
223 Ops.INDEX dtypes.i32 [196, 30]
224 Ops.INDEX dtypes.i32 [191, 223]
225 Ops.CMPLT dtypes.bool [200, 223]
226 Ops.CMPLT dtypes.bool [223, 9]
227 Ops.AND dtypes.bool [225, 226]
228 Ops.LOAD dtypes.i32 [224, 5, 227]
229 Ops.WHERE dtypes.i32 [222, 228, 200]
230 Ops.STACK dtypes.i32 [205, 213, 221, 229]
231 Ops.STORE dtypes.void [167, 230]
232 Ops.SINK dtypes.void [231, 231, 231, 231] KernelInfo(name="r_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST))), estimates=Estimates(ops=23, lds=52, mem=52), optimized=True)
233 Ops.LINEAR dtypes.void [161, 162, 11, 12, 163, 168, 169, 0, 170, 4, 5, 171, 172, 173, 174, 23, 24, 175, 176, 29, 30, 177, 178, 192, 193, 194, 9, 195, 196, 189, 190, 191, 197, 198, 199, 200, 202, 201, 203, 204, 207, 208, 210, 209, 211, 212, 215, 216, 218, 217, 219, 220, 223, 224, 226, 225, 227, 228, 164, 165, 166, 179, 180, 181, 182, 184, 186, 183, 185, 187, 188, 206, 214, 222, 205, 213, 221, 229, 230, 167, 231, 232]
234 Ops.SOURCE dtypes.void [] "typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef signed char signed_char4 __attribute__((aligned(1),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_4_4(unsigned char* restrict args_32) {\n  unsigned long val0 = (*(((unsigned long*)(args_32))+1));\n  _Bool val1 = (*(((_Bool*)(val0))+0));\n  _Bool val2 = (*(((_Bool*)(val0))+1));\n  _Bool val3 = (*(((_Bool*)(val0))+2));\n  _Bool val4 = (*(((_Bool*)(val0))+3));\n  unsigned long val5 = (*(((unsigned long*)(args_32))+2));\n  int4 val6 = (*((int4*)((((int*)(val5))+0))));\n  unsigned long val7 = (*(((unsigned long*)(args_32))+3));\n  int val8 = (((-1<val6[0])&(val6[0]<4))?*(((int*)(val7))+val6[0]):0);\n  int val9 = (((-1<val6[1])&(val6[1]<4))?*(((int*)(val7))+val6[1]):0);\n  int val10 = (((-1<val6[2])&(val6[2]<4))?*(((int*)(val7))+val6[2]):0);\n  int val11 = (((-1<val6[3])&(val6[3]<4))?*(((int*)(val7))+val6[3]):0);\n  unsigned long val12 = (*(((unsigned long*)(args_32))+0));\n  int4 cast0 = __builtin_convertvector(((-(signed_char4){val1,val2,val3,val4})&1), int4);\n  int alu0 = (cast0[0]+cast0[1]+cast0[2]+cast0[3]);\n  int alu1 = ((0<alu0)?val8:-1);\n  int alu2 = ((1<alu0)?val9:-1);\n  int alu3 = ((2<alu0)?val10:-1);\n  int alu4 = ((3<alu0)?val11:-1);\n  *((int4*)((((int*)(val12))+0))) = (int4){alu1,alu2,alu3,alu4};\n}\n"
235 Ops.BINARY dtypes.u8 [] b"typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef signed char signed_char4 __attribute__((aligned(1),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_4_4(unsigned char* restrict args_32) {\n  unsigned long val0 = (*(((unsigned long*)(args_32))+1));\n  _Bool val1 = (*(((_Bool*)(val0))+0));\n  _Bool val2 = (*(((_Bool*)(val0))+1));\n  _Bool val3 = (*(((_Bool*)(val0))+2));\n  _Bool val4 = (*(((_Bool*)(val0))+3));\n  unsigned long val5 = (*(((unsigned long*)(args_32))+2));\n  int4 val6 = (*((int4*)((((int*)(val5))+0))));\n  unsigned long val7 = (*(((unsigned long*)(args_32))+3));\n  int val8 = (((-1<val6[0])&(val6[0]<4))?*(((int*)(val7))+val6[0]):0);\n  int val9 = (((-1<val6[1])&(val6[1]<4))?*(((int*)(val7))+val6[1]):0);\n  int val10 = (((-1<val6[2])&(val6[2]<4))?*(((int*)(val7))+val6[2]):0);\n  int val11 = (((-1<val6[3])&(val6[3]<4))?*(((int*)(val7))+val6[3]):0);\n  unsigned long val12 = (*(((unsigned long*)(args_32))+0));\n  int4 cast0 = __builtin_convertvector(((-(signed_char4){val1,val2,val3,val4})&1), int4);\n  int alu0 = (cast0[0]+cast0[1]+cast0[2]+cast0[3]);\n  int alu1 = ((0<alu0)?val8:-1);\n  int alu2 = ((1<alu0)?val9:-1);\n  int alu3 = ((2<alu0)?val10:-1);\n  int alu4 = ((3<alu0)?val11:-1);\n  *((int4*)((((int*)(val12))+0))) = (int4){alu1,alu2,alu3,alu4};\n}\n"
236 Ops.PARAM dtypes.i32 [0] ParamArg(slot=2, dtype=dtypes.i32)
237 Ops.PARAM dtypes.i32 [0] ParamArg(slot=3, dtype=dtypes.i32)
238 Ops.PROGRAM dtypes.void [232, 233, 234, 235] ProgramInfo(params=(%41, %42, %236, %237), globals=(0, 1, 2, 3), outs=(0,), ins=(1, 2, 3), target=Target(device="CPU", arch="x86_64,x86-64"))
239 Ops.BUFFER dtypes.i32 [0] ParamArg(slot=4, dtype=dtypes.i32, device="CPU")
240 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32, device="CPU")
241 Ops.CALL dtypes.void [238, 239, 48, 47, 240] CallInfo()
242 Ops.LINEAR dtypes.void [49, 122, 160, 241]

