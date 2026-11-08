# tolk D80 D141 D151 golden=1bbfe5d3571edb5368b2904c2ad8a366
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 64
2 Ops.PARAM dtypes.i32 [1] ParamArg(slot=0, dtype=dtypes.i32)
3 Ops.PARAM dtypes.i32 [0] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 4), name="block_lo", addrspace=AddrSpace.ALU)
4 Ops.PARAM dtypes.i32 [0] ParamArg(slot=4, dtype=dtypes.i32, vmin_vmax=(0, 4), name="block_hi", addrspace=AddrSpace.ALU)
5 Ops.SUB dtypes.i32 [4, 3]
6 Ops.RANGE dtypes.i32 [5] (AxisType.WEAK, 0)
7 Ops.ADD dtypes.i32 [3, 6]
8 Ops.CONST dtypes.weakint [] 4
9 Ops.CAST dtypes.i32 [8] dtypes.i32
10 Ops.SHL dtypes.i32 [7, 9]
11 Ops.SHRINK dtypes.i32 [2, 10, 9]
12 Ops.PARAM dtypes.i32 [1] ParamArg(slot=1, dtype=dtypes.i32)
13 Ops.SHRINK dtypes.i32 [12, 10, 9]
14 Ops.LOAD dtypes.i32 [13]
15 Ops.CONST dtypes.weakint [] 0
16 Ops.CAST dtypes.i32 [15] dtypes.i32
17 Ops.INDEX dtypes.i32 [14, 16]
18 Ops.CONST dtypes.weakint [] 1
19 Ops.CAST dtypes.i32 [18] dtypes.i32
20 Ops.INDEX dtypes.i32 [14, 19]
21 Ops.CONST dtypes.weakint [] 2
22 Ops.CAST dtypes.i32 [21] dtypes.i32
23 Ops.INDEX dtypes.i32 [14, 22]
24 Ops.CONST dtypes.weakint [] 3
25 Ops.CAST dtypes.i32 [24] dtypes.i32
26 Ops.INDEX dtypes.i32 [14, 25]
27 Ops.ADD dtypes.i32 [10, 9]
28 Ops.SHRINK dtypes.i32 [12, 27, 9]
29 Ops.LOAD dtypes.i32 [28]
30 Ops.INDEX dtypes.i32 [29, 16]
31 Ops.INDEX dtypes.i32 [29, 19]
32 Ops.INDEX dtypes.i32 [29, 22]
33 Ops.INDEX dtypes.i32 [29, 25]
34 Ops.CONST dtypes.weakint [] 8
35 Ops.CAST dtypes.i32 [34] dtypes.i32
36 Ops.ADD dtypes.i32 [10, 35]
37 Ops.SHRINK dtypes.i32 [12, 36, 9]
38 Ops.LOAD dtypes.i32 [37]
39 Ops.INDEX dtypes.i32 [38, 16]
40 Ops.INDEX dtypes.i32 [38, 19]
41 Ops.INDEX dtypes.i32 [38, 22]
42 Ops.INDEX dtypes.i32 [38, 25]
43 Ops.CONST dtypes.weakint [] 12
44 Ops.CAST dtypes.i32 [43] dtypes.i32
45 Ops.ADD dtypes.i32 [10, 44]
46 Ops.SHRINK dtypes.i32 [12, 45, 9]
47 Ops.LOAD dtypes.i32 [46]
48 Ops.INDEX dtypes.i32 [47, 16]
49 Ops.INDEX dtypes.i32 [47, 19]
50 Ops.INDEX dtypes.i32 [47, 22]
51 Ops.INDEX dtypes.i32 [47, 25]
52 Ops.STACK dtypes.i32 [17, 20, 23, 26, 30, 31, 32, 33, 39, 40, 41, 42, 48, 49, 50, 51]
53 Ops.CONST dtypes.weakint [] 7
54 Ops.CAST dtypes.i32 [53] dtypes.i32
55 Ops.TRUNCDIV dtypes.i32 [52, 54]
56 Ops.TRUNCMOD dtypes.i32 [52, 54]
57 Ops.CMPNE dtypes.bool [56, 16]
58 Ops.CMPLT dtypes.bool [52, 16]
59 Ops.AND dtypes.bool [57, 58]
60 Ops.CAST dtypes.i32 [59] dtypes.i32
61 Ops.SUB dtypes.i32 [55, 60]
62 Ops.PARAM dtypes.i32 [1] ParamArg(slot=2, dtype=dtypes.i32)
63 Ops.SHRINK dtypes.i32 [62, 10, 9]
64 Ops.LOAD dtypes.i32 [63]
65 Ops.INDEX dtypes.i32 [64, 16]
66 Ops.INDEX dtypes.i32 [64, 19]
67 Ops.INDEX dtypes.i32 [64, 22]
68 Ops.INDEX dtypes.i32 [64, 25]
69 Ops.SHRINK dtypes.i32 [62, 27, 9]
70 Ops.LOAD dtypes.i32 [69]
71 Ops.INDEX dtypes.i32 [70, 16]
72 Ops.INDEX dtypes.i32 [70, 19]
73 Ops.INDEX dtypes.i32 [70, 22]
74 Ops.INDEX dtypes.i32 [70, 25]
75 Ops.SHRINK dtypes.i32 [62, 36, 9]
76 Ops.LOAD dtypes.i32 [75]
77 Ops.INDEX dtypes.i32 [76, 16]
78 Ops.INDEX dtypes.i32 [76, 19]
79 Ops.INDEX dtypes.i32 [76, 22]
80 Ops.INDEX dtypes.i32 [76, 25]
81 Ops.SHRINK dtypes.i32 [62, 45, 9]
82 Ops.LOAD dtypes.i32 [81]
83 Ops.INDEX dtypes.i32 [82, 16]
84 Ops.INDEX dtypes.i32 [82, 19]
85 Ops.INDEX dtypes.i32 [82, 22]
86 Ops.INDEX dtypes.i32 [82, 25]
87 Ops.STACK dtypes.i32 [65, 66, 67, 68, 71, 72, 73, 74, 77, 78, 79, 80, 83, 84, 85, 86]
88 Ops.CONST dtypes.weakint [] 5
89 Ops.CAST dtypes.i32 [88] dtypes.i32
90 Ops.TRUNCMOD dtypes.i32 [87, 89]
91 Ops.CMPNE dtypes.bool [90, 16]
92 Ops.CMPLT dtypes.bool [87, 16]
93 Ops.AND dtypes.bool [91, 92]
94 Ops.WHERE dtypes.i32 [93, 89, 16]
95 Ops.ADD dtypes.i32 [90, 94]
96 Ops.ADD dtypes.i32 [61, 95]
97 Ops.INDEX dtypes.i32 [96, 16]
98 Ops.INDEX dtypes.i32 [96, 19]
99 Ops.INDEX dtypes.i32 [96, 22]
100 Ops.INDEX dtypes.i32 [96, 25]
101 Ops.STACK dtypes.i32 [97, 98, 99, 100]
102 Ops.STORE dtypes.void [11, 101]
103 Ops.SHRINK dtypes.i32 [2, 27, 9]
104 Ops.INDEX dtypes.i32 [96, 9]
105 Ops.INDEX dtypes.i32 [96, 89]
106 Ops.CONST dtypes.weakint [] 6
107 Ops.CAST dtypes.i32 [106] dtypes.i32
108 Ops.INDEX dtypes.i32 [96, 107]
109 Ops.INDEX dtypes.i32 [96, 54]
110 Ops.STACK dtypes.i32 [104, 105, 108, 109]
111 Ops.STORE dtypes.void [103, 110]
112 Ops.SHRINK dtypes.i32 [2, 36, 9]
113 Ops.INDEX dtypes.i32 [96, 35]
114 Ops.CONST dtypes.weakint [] 9
115 Ops.CAST dtypes.i32 [114] dtypes.i32
116 Ops.INDEX dtypes.i32 [96, 115]
117 Ops.CONST dtypes.weakint [] 10
118 Ops.CAST dtypes.i32 [117] dtypes.i32
119 Ops.INDEX dtypes.i32 [96, 118]
120 Ops.CONST dtypes.weakint [] 11
121 Ops.CAST dtypes.i32 [120] dtypes.i32
122 Ops.INDEX dtypes.i32 [96, 121]
123 Ops.STACK dtypes.i32 [113, 116, 119, 122]
124 Ops.STORE dtypes.void [112, 123]
125 Ops.SHRINK dtypes.i32 [2, 45, 9]
126 Ops.INDEX dtypes.i32 [96, 44]
127 Ops.CONST dtypes.weakint [] 13
128 Ops.CAST dtypes.i32 [127] dtypes.i32
129 Ops.INDEX dtypes.i32 [96, 128]
130 Ops.CONST dtypes.weakint [] 14
131 Ops.CAST dtypes.i32 [130] dtypes.i32
132 Ops.INDEX dtypes.i32 [96, 131]
133 Ops.CONST dtypes.weakint [] 15
134 Ops.CAST dtypes.i32 [133] dtypes.i32
135 Ops.INDEX dtypes.i32 [96, 134]
136 Ops.STACK dtypes.i32 [126, 129, 132, 135]
137 Ops.STORE dtypes.void [125, 136]
138 Ops.AFTER dtypes.void [0, 102, 111, 124, 137]
139 Ops.END dtypes.void [138, 6]
140 Ops.SINK dtypes.void [139] KernelInfo(name="E_4_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.UPCAST)),), optimized=True, split=(4, 3, 4))
141 Ops.CONST dtypes.weakint [] 40
142 Ops.PARAM dtypes.u8 [141] ParamArg(slot=0, dtype=dtypes.u8, name="args")
143 Ops.BITCAST dtypes.u64 [142] dtypes.u64
144 Ops.INDEX dtypes.u64 [143, 16]
145 Ops.LOAD dtypes.u64 [144]
146 Ops.REFPARAM dtypes.i32 [1, 145] ParamArg(slot=0, dtype=dtypes.i32)
147 Ops.BITCAST dtypes.i32 [142] dtypes.i32
148 Ops.INDEX dtypes.i32 [147, 107]
149 Ops.LOAD dtypes.i32 [148]
150 Ops.INDEX dtypes.i32 [147, 54]
151 Ops.LOAD dtypes.i32 [150]
152 Ops.SUB dtypes.i32 [151, 149]
153 Ops.RANGE dtypes.i32 [152] (AxisType.WEAK, 0)
154 Ops.ADD dtypes.i32 [149, 153]
155 Ops.SHL dtypes.i32 [154, 9]
156 Ops.SHRINK dtypes.i32 [146, 155, 9]
157 Ops.INDEX dtypes.u64 [143, 19]
158 Ops.LOAD dtypes.u64 [157]
159 Ops.REFPARAM dtypes.i32 [1, 158] ParamArg(slot=1, dtype=dtypes.i32)
160 Ops.SHRINK dtypes.i32 [159, 155, 9]
161 Ops.LOAD dtypes.i32 [160]
162 Ops.INDEX dtypes.i32 [161, 16]
163 Ops.INDEX dtypes.i32 [161, 19]
164 Ops.INDEX dtypes.i32 [161, 22]
165 Ops.INDEX dtypes.i32 [161, 25]
166 Ops.ADD dtypes.i32 [155, 9]
167 Ops.SHRINK dtypes.i32 [159, 166, 9]
168 Ops.LOAD dtypes.i32 [167]
169 Ops.INDEX dtypes.i32 [168, 16]
170 Ops.INDEX dtypes.i32 [168, 19]
171 Ops.INDEX dtypes.i32 [168, 22]
172 Ops.INDEX dtypes.i32 [168, 25]
173 Ops.ADD dtypes.i32 [155, 35]
174 Ops.SHRINK dtypes.i32 [159, 173, 9]
175 Ops.LOAD dtypes.i32 [174]
176 Ops.INDEX dtypes.i32 [175, 16]
177 Ops.INDEX dtypes.i32 [175, 19]
178 Ops.INDEX dtypes.i32 [175, 22]
179 Ops.INDEX dtypes.i32 [175, 25]
180 Ops.ADD dtypes.i32 [155, 44]
181 Ops.SHRINK dtypes.i32 [159, 180, 9]
182 Ops.LOAD dtypes.i32 [181]
183 Ops.INDEX dtypes.i32 [182, 16]
184 Ops.INDEX dtypes.i32 [182, 19]
185 Ops.INDEX dtypes.i32 [182, 22]
186 Ops.INDEX dtypes.i32 [182, 25]
187 Ops.STACK dtypes.i32 [162, 163, 164, 165, 169, 170, 171, 172, 176, 177, 178, 179, 183, 184, 185, 186]
188 Ops.TRUNCDIV dtypes.i32 [187, 54]
189 Ops.TRUNCMOD dtypes.i32 [187, 54]
190 Ops.CMPNE dtypes.bool [189, 16]
191 Ops.CMPLT dtypes.bool [187, 16]
192 Ops.AND dtypes.bool [190, 191]
193 Ops.CAST dtypes.i32 [192] dtypes.i32
194 Ops.SUB dtypes.i32 [188, 193]
195 Ops.INDEX dtypes.u64 [143, 22]
196 Ops.LOAD dtypes.u64 [195]
197 Ops.REFPARAM dtypes.i32 [1, 196] ParamArg(slot=2, dtype=dtypes.i32)
198 Ops.SHRINK dtypes.i32 [197, 155, 9]
199 Ops.LOAD dtypes.i32 [198]
200 Ops.INDEX dtypes.i32 [199, 16]
201 Ops.INDEX dtypes.i32 [199, 19]
202 Ops.INDEX dtypes.i32 [199, 22]
203 Ops.INDEX dtypes.i32 [199, 25]
204 Ops.SHRINK dtypes.i32 [197, 166, 9]
205 Ops.LOAD dtypes.i32 [204]
206 Ops.INDEX dtypes.i32 [205, 16]
207 Ops.INDEX dtypes.i32 [205, 19]
208 Ops.INDEX dtypes.i32 [205, 22]
209 Ops.INDEX dtypes.i32 [205, 25]
210 Ops.SHRINK dtypes.i32 [197, 173, 9]
211 Ops.LOAD dtypes.i32 [210]
212 Ops.INDEX dtypes.i32 [211, 16]
213 Ops.INDEX dtypes.i32 [211, 19]
214 Ops.INDEX dtypes.i32 [211, 22]
215 Ops.INDEX dtypes.i32 [211, 25]
216 Ops.SHRINK dtypes.i32 [197, 180, 9]
217 Ops.LOAD dtypes.i32 [216]
218 Ops.INDEX dtypes.i32 [217, 16]
219 Ops.INDEX dtypes.i32 [217, 19]
220 Ops.INDEX dtypes.i32 [217, 22]
221 Ops.INDEX dtypes.i32 [217, 25]
222 Ops.STACK dtypes.i32 [200, 201, 202, 203, 206, 207, 208, 209, 212, 213, 214, 215, 218, 219, 220, 221]
223 Ops.TRUNCMOD dtypes.i32 [222, 89]
224 Ops.CMPNE dtypes.bool [223, 16]
225 Ops.CMPLT dtypes.bool [222, 16]
226 Ops.AND dtypes.bool [224, 225]
227 Ops.WHERE dtypes.i32 [226, 89, 16]
228 Ops.ADD dtypes.i32 [223, 227]
229 Ops.ADD dtypes.i32 [194, 228]
230 Ops.INDEX dtypes.i32 [229, 16]
231 Ops.INDEX dtypes.i32 [229, 19]
232 Ops.INDEX dtypes.i32 [229, 22]
233 Ops.INDEX dtypes.i32 [229, 25]
234 Ops.STACK dtypes.i32 [230, 231, 232, 233]
235 Ops.STORE dtypes.void [156, 234]
236 Ops.SHRINK dtypes.i32 [146, 166, 9]
237 Ops.INDEX dtypes.i32 [229, 9]
238 Ops.INDEX dtypes.i32 [229, 89]
239 Ops.INDEX dtypes.i32 [229, 107]
240 Ops.INDEX dtypes.i32 [229, 54]
241 Ops.STACK dtypes.i32 [237, 238, 239, 240]
242 Ops.STORE dtypes.void [236, 241]
243 Ops.SHRINK dtypes.i32 [146, 173, 9]
244 Ops.INDEX dtypes.i32 [229, 35]
245 Ops.INDEX dtypes.i32 [229, 115]
246 Ops.INDEX dtypes.i32 [229, 118]
247 Ops.INDEX dtypes.i32 [229, 121]
248 Ops.STACK dtypes.i32 [244, 245, 246, 247]
249 Ops.STORE dtypes.void [243, 248]
250 Ops.SHRINK dtypes.i32 [146, 180, 9]
251 Ops.INDEX dtypes.i32 [229, 44]
252 Ops.INDEX dtypes.i32 [229, 128]
253 Ops.INDEX dtypes.i32 [229, 131]
254 Ops.INDEX dtypes.i32 [229, 134]
255 Ops.STACK dtypes.i32 [251, 252, 253, 254]
256 Ops.STORE dtypes.void [250, 255]
257 Ops.AFTER dtypes.void [0, 235, 242, 249, 256]
258 Ops.END dtypes.void [257, 153]
259 Ops.SINK dtypes.void [258] KernelInfo(name="E_4_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.UPCAST)),), estimates=Estimates(ops=832, lds=768, mem=768), optimized=True, split=(4, 3, 4))
260 Ops.LINEAR dtypes.void [141, 142, 106, 107, 147, 148, 149, 53, 54, 150, 151, 15, 16, 143, 144, 145, 18, 19, 157, 158, 21, 22, 195, 196, 1, 146, 159, 197, 0, 117, 118, 120, 121, 43, 44, 127, 128, 130, 131, 133, 134, 24, 25, 8, 9, 88, 89, 34, 35, 114, 115, 152, 153, 154, 155, 180, 181, 182, 166, 167, 168, 173, 174, 175, 160, 161, 216, 217, 204, 205, 210, 211, 198, 199, 183, 184, 185, 186, 169, 170, 171, 172, 176, 177, 178, 179, 162, 163, 164, 165, 187, 218, 219, 220, 221, 206, 207, 208, 209, 212, 213, 214, 215, 200, 201, 202, 203, 222, 189, 191, 190, 192, 193, 223, 225, 224, 226, 227, 228, 188, 194, 229, 230, 231, 232, 233, 234, 251, 252, 253, 254, 255, 237, 238, 239, 240, 241, 246, 247, 244, 245, 248, 250, 236, 243, 156, 256, 242, 249, 235, 257, 258, 259]
261 Ops.SOURCE dtypes.void [] "typedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int16 __attribute__((aligned(4),ext_vector_type(16)));\n__attribute__((min_vector_width(512))) void E_4_16(unsigned char* restrict args_40) {\n  int val0 = (*(((int*)(args_40))+6));\n  int val1 = (*(((int*)(args_40))+7));\n  unsigned long val2 = (*(((unsigned long*)(args_40))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_40))+1));\n  unsigned long val4 = (*(((unsigned long*)(args_40))+2));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = ((val0+Lidx0)<<4);\n    int alu1 = (alu0+12);\n    int4 val5 = (*((int4*)((((int*)(val3))+alu1))));\n    int alu2 = (alu0+4);\n    int4 val6 = (*((int4*)((((int*)(val3))+alu2))));\n    int alu3 = (alu0+8);\n    int4 val7 = (*((int4*)((((int*)(val3))+alu3))));\n    int4 val8 = (*((int4*)((((int*)(val3))+alu0))));\n    int4 val9 = (*((int4*)((((int*)(val4))+alu1))));\n    int4 val10 = (*((int4*)((((int*)(val4))+alu2))));\n    int4 val11 = (*((int4*)((((int*)(val4))+alu3))));\n    int4 val12 = (*((int4*)((((int*)(val4))+alu0))));\n    int16 cast0 = (int16){val8[0],val8[1],val8[2],val8[3],val6[0],val6[1],val6[2],val6[3],val7[0],val7[1],val7[2],val7[3],val5[0],val5[1],val5[2],val5[3]};\n    int16 cast1 = (int16){val12[0],val12[1],val12[2],val12[3],val10[0],val10[1],val10[2],val10[3],val11[0],val11[1],val11[2],val11[3],val9[0],val9[1],val9[2],val9[3]};\n    int16 cast2 = __builtin_convertvector(((((cast0%7)!=0)&(cast0<0))&1), int16);\n    int16 alu4 = (cast1%5);\n    int16 alu5 = (((alu4!=0)&(cast1<0))?((int16)(5)):0);\n    int16 alu6 = (((cast0/7)-cast2)+(alu4+alu5));\n    *((int4*)((((int*)(val2))+alu1))) = (int4){alu6[12],alu6[13],alu6[14],alu6[15]};\n    *((int4*)((((int*)(val2))+alu2))) = (int4){alu6[4],alu6[5],alu6[6],alu6[7]};\n    *((int4*)((((int*)(val2))+alu3))) = (int4){alu6[8],alu6[9],alu6[10],alu6[11]};\n    *((int4*)((((int*)(val2))+alu0))) = (int4){alu6[0],alu6[1],alu6[2],alu6[3]};\n  }\n}\n"
262 Ops.PROGRAM dtypes.void [259, 260, 261] ProgramInfo(global_size=(4, 1, 1), params=(%2, %12, %62, %3, %4), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
263 Ops.SINK dtypes.void [140, 262]

