# tolk D141 D151 D430 golden=12d98234bbb39d58aa2ebacec7c4e830
0 Ops.CONST dtypes.weakint [] 16
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.CONST dtypes.weakint [] 4
5 Ops.CAST dtypes.i32 [4] dtypes.i32
6 Ops.SHRINK dtypes.f32 [1, 3, 5]
7 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=1, dtype=dtypes.bf16)
8 Ops.INDEX dtypes.bf16 [7, 3]
9 Ops.LOAD dtypes.bf16 [8]
10 Ops.CONST dtypes.weakint [] 1
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.INDEX dtypes.bf16 [7, 11]
13 Ops.LOAD dtypes.bf16 [12]
14 Ops.CONST dtypes.weakint [] 2
15 Ops.CAST dtypes.i32 [14] dtypes.i32
16 Ops.INDEX dtypes.bf16 [7, 15]
17 Ops.LOAD dtypes.bf16 [16]
18 Ops.CONST dtypes.weakint [] 3
19 Ops.CAST dtypes.i32 [18] dtypes.i32
20 Ops.INDEX dtypes.bf16 [7, 19]
21 Ops.LOAD dtypes.bf16 [20]
22 Ops.INDEX dtypes.bf16 [7, 5]
23 Ops.LOAD dtypes.bf16 [22]
24 Ops.CONST dtypes.weakint [] 5
25 Ops.CAST dtypes.i32 [24] dtypes.i32
26 Ops.INDEX dtypes.bf16 [7, 25]
27 Ops.LOAD dtypes.bf16 [26]
28 Ops.CONST dtypes.weakint [] 6
29 Ops.CAST dtypes.i32 [28] dtypes.i32
30 Ops.INDEX dtypes.bf16 [7, 29]
31 Ops.LOAD dtypes.bf16 [30]
32 Ops.CONST dtypes.weakint [] 7
33 Ops.CAST dtypes.i32 [32] dtypes.i32
34 Ops.INDEX dtypes.bf16 [7, 33]
35 Ops.LOAD dtypes.bf16 [34]
36 Ops.CONST dtypes.weakint [] 8
37 Ops.CAST dtypes.i32 [36] dtypes.i32
38 Ops.INDEX dtypes.bf16 [7, 37]
39 Ops.LOAD dtypes.bf16 [38]
40 Ops.CONST dtypes.weakint [] 9
41 Ops.CAST dtypes.i32 [40] dtypes.i32
42 Ops.INDEX dtypes.bf16 [7, 41]
43 Ops.LOAD dtypes.bf16 [42]
44 Ops.CONST dtypes.weakint [] 10
45 Ops.CAST dtypes.i32 [44] dtypes.i32
46 Ops.INDEX dtypes.bf16 [7, 45]
47 Ops.LOAD dtypes.bf16 [46]
48 Ops.CONST dtypes.weakint [] 11
49 Ops.CAST dtypes.i32 [48] dtypes.i32
50 Ops.INDEX dtypes.bf16 [7, 49]
51 Ops.LOAD dtypes.bf16 [50]
52 Ops.CONST dtypes.weakint [] 12
53 Ops.CAST dtypes.i32 [52] dtypes.i32
54 Ops.INDEX dtypes.bf16 [7, 53]
55 Ops.LOAD dtypes.bf16 [54]
56 Ops.CONST dtypes.weakint [] 13
57 Ops.CAST dtypes.i32 [56] dtypes.i32
58 Ops.INDEX dtypes.bf16 [7, 57]
59 Ops.LOAD dtypes.bf16 [58]
60 Ops.CONST dtypes.weakint [] 14
61 Ops.CAST dtypes.i32 [60] dtypes.i32
62 Ops.INDEX dtypes.bf16 [7, 61]
63 Ops.LOAD dtypes.bf16 [62]
64 Ops.CONST dtypes.weakint [] 15
65 Ops.CAST dtypes.i32 [64] dtypes.i32
66 Ops.INDEX dtypes.bf16 [7, 65]
67 Ops.LOAD dtypes.bf16 [66]
68 Ops.STACK dtypes.bf16 [9, 13, 17, 21, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67]
69 Ops.BITCAST dtypes.u16 [68] dtypes.u16
70 Ops.CAST dtypes.u32 [69] dtypes.u32
71 Ops.CAST dtypes.u32 [0] dtypes.u32
72 Ops.SHL dtypes.u32 [70, 71]
73 Ops.BITCAST dtypes.f32 [72] dtypes.f32
74 Ops.CONST dtypes.weakfloat [] 1.0
75 Ops.CAST dtypes.f32 [74] dtypes.f32
76 Ops.ADD dtypes.f32 [73, 75]
77 Ops.CMPNE dtypes.bool [76, 76]
78 Ops.BITCAST dtypes.u32 [76] dtypes.u32
79 Ops.CONST dtypes.weakint [] 4194304
80 Ops.CAST dtypes.u32 [79] dtypes.u32
81 Ops.OR dtypes.u32 [78, 80]
82 Ops.SHR dtypes.u32 [78, 71]
83 Ops.CAST dtypes.u32 [10] dtypes.u32
84 Ops.AND dtypes.u32 [82, 83]
85 Ops.ADD dtypes.u32 [78, 84]
86 Ops.CONST dtypes.weakint [] 32767
87 Ops.CAST dtypes.u32 [86] dtypes.u32
88 Ops.ADD dtypes.u32 [85, 87]
89 Ops.WHERE dtypes.u32 [77, 81, 88]
90 Ops.SHR dtypes.u32 [89, 71]
91 Ops.CAST dtypes.u16 [90] dtypes.u16
92 Ops.CAST dtypes.u32 [91] dtypes.u32
93 Ops.SHL dtypes.u32 [92, 71]
94 Ops.BITCAST dtypes.f32 [93] dtypes.f32
95 Ops.INDEX dtypes.f32 [94, 3]
96 Ops.INDEX dtypes.f32 [94, 11]
97 Ops.INDEX dtypes.f32 [94, 15]
98 Ops.INDEX dtypes.f32 [94, 19]
99 Ops.STACK dtypes.f32 [95, 96, 97, 98]
100 Ops.STORE dtypes.void [6, 99]
101 Ops.SHRINK dtypes.f32 [1, 5, 5]
102 Ops.INDEX dtypes.f32 [94, 5]
103 Ops.INDEX dtypes.f32 [94, 25]
104 Ops.INDEX dtypes.f32 [94, 29]
105 Ops.INDEX dtypes.f32 [94, 33]
106 Ops.STACK dtypes.f32 [102, 103, 104, 105]
107 Ops.STORE dtypes.void [101, 106]
108 Ops.SHRINK dtypes.f32 [1, 37, 5]
109 Ops.INDEX dtypes.f32 [94, 37]
110 Ops.INDEX dtypes.f32 [94, 41]
111 Ops.INDEX dtypes.f32 [94, 45]
112 Ops.INDEX dtypes.f32 [94, 49]
113 Ops.STACK dtypes.f32 [109, 110, 111, 112]
114 Ops.STORE dtypes.void [108, 113]
115 Ops.SHRINK dtypes.f32 [1, 53, 5]
116 Ops.INDEX dtypes.f32 [94, 53]
117 Ops.INDEX dtypes.f32 [94, 57]
118 Ops.INDEX dtypes.f32 [94, 61]
119 Ops.INDEX dtypes.f32 [94, 65]
120 Ops.STACK dtypes.f32 [116, 117, 118, 119]
121 Ops.STORE dtypes.void [115, 120]
122 Ops.SINK dtypes.void [100, 100, 100, 100, 107, 107, 107, 107, 114, 114, 114, 114, 121, 121, 121, 121] KernelInfo(name="E_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.UPCAST)),), optimized=True)
123 Ops.PARAM dtypes.u8 [0] ParamArg(slot=0, dtype=dtypes.u8, name="args")
124 Ops.BITCAST dtypes.u64 [123] dtypes.u64
125 Ops.INDEX dtypes.u64 [124, 3]
126 Ops.LOAD dtypes.u64 [125]
127 Ops.REFPARAM dtypes.f32 [0, 126] ParamArg(slot=0, dtype=dtypes.f32)
128 Ops.SHRINK dtypes.f32 [127, 3, 5]
129 Ops.INDEX dtypes.u64 [124, 11]
130 Ops.LOAD dtypes.u64 [129]
131 Ops.REFPARAM dtypes.bf16 [0, 130] ParamArg(slot=1, dtype=dtypes.bf16)
132 Ops.INDEX dtypes.bf16 [131, 3]
133 Ops.LOAD dtypes.bf16 [132]
134 Ops.INDEX dtypes.bf16 [131, 11]
135 Ops.LOAD dtypes.bf16 [134]
136 Ops.INDEX dtypes.bf16 [131, 15]
137 Ops.LOAD dtypes.bf16 [136]
138 Ops.INDEX dtypes.bf16 [131, 19]
139 Ops.LOAD dtypes.bf16 [138]
140 Ops.INDEX dtypes.bf16 [131, 5]
141 Ops.LOAD dtypes.bf16 [140]
142 Ops.INDEX dtypes.bf16 [131, 25]
143 Ops.LOAD dtypes.bf16 [142]
144 Ops.INDEX dtypes.bf16 [131, 29]
145 Ops.LOAD dtypes.bf16 [144]
146 Ops.INDEX dtypes.bf16 [131, 33]
147 Ops.LOAD dtypes.bf16 [146]
148 Ops.INDEX dtypes.bf16 [131, 37]
149 Ops.LOAD dtypes.bf16 [148]
150 Ops.INDEX dtypes.bf16 [131, 41]
151 Ops.LOAD dtypes.bf16 [150]
152 Ops.INDEX dtypes.bf16 [131, 45]
153 Ops.LOAD dtypes.bf16 [152]
154 Ops.INDEX dtypes.bf16 [131, 49]
155 Ops.LOAD dtypes.bf16 [154]
156 Ops.INDEX dtypes.bf16 [131, 53]
157 Ops.LOAD dtypes.bf16 [156]
158 Ops.INDEX dtypes.bf16 [131, 57]
159 Ops.LOAD dtypes.bf16 [158]
160 Ops.INDEX dtypes.bf16 [131, 61]
161 Ops.LOAD dtypes.bf16 [160]
162 Ops.INDEX dtypes.bf16 [131, 65]
163 Ops.LOAD dtypes.bf16 [162]
164 Ops.STACK dtypes.bf16 [133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163]
165 Ops.BITCAST dtypes.u16 [164] dtypes.u16
166 Ops.CAST dtypes.u32 [165] dtypes.u32
167 Ops.SHL dtypes.u32 [166, 71]
168 Ops.BITCAST dtypes.f32 [167] dtypes.f32
169 Ops.ADD dtypes.f32 [168, 75]
170 Ops.CMPNE dtypes.bool [169, 169]
171 Ops.BITCAST dtypes.u32 [169] dtypes.u32
172 Ops.OR dtypes.u32 [171, 80]
173 Ops.SHR dtypes.u32 [171, 71]
174 Ops.AND dtypes.u32 [173, 83]
175 Ops.ADD dtypes.u32 [171, 174]
176 Ops.ADD dtypes.u32 [175, 87]
177 Ops.WHERE dtypes.u32 [170, 172, 176]
178 Ops.SHR dtypes.u32 [177, 71]
179 Ops.CAST dtypes.u16 [178] dtypes.u16
180 Ops.CAST dtypes.u32 [179] dtypes.u32
181 Ops.SHL dtypes.u32 [180, 71]
182 Ops.BITCAST dtypes.f32 [181] dtypes.f32
183 Ops.INDEX dtypes.f32 [182, 3]
184 Ops.INDEX dtypes.f32 [182, 11]
185 Ops.INDEX dtypes.f32 [182, 15]
186 Ops.INDEX dtypes.f32 [182, 19]
187 Ops.STACK dtypes.f32 [183, 184, 185, 186]
188 Ops.STORE dtypes.void [128, 187]
189 Ops.SHRINK dtypes.f32 [127, 5, 5]
190 Ops.INDEX dtypes.f32 [182, 5]
191 Ops.INDEX dtypes.f32 [182, 25]
192 Ops.INDEX dtypes.f32 [182, 29]
193 Ops.INDEX dtypes.f32 [182, 33]
194 Ops.STACK dtypes.f32 [190, 191, 192, 193]
195 Ops.STORE dtypes.void [189, 194]
196 Ops.SHRINK dtypes.f32 [127, 37, 5]
197 Ops.INDEX dtypes.f32 [182, 37]
198 Ops.INDEX dtypes.f32 [182, 41]
199 Ops.INDEX dtypes.f32 [182, 45]
200 Ops.INDEX dtypes.f32 [182, 49]
201 Ops.STACK dtypes.f32 [197, 198, 199, 200]
202 Ops.STORE dtypes.void [196, 201]
203 Ops.SHRINK dtypes.f32 [127, 53, 5]
204 Ops.INDEX dtypes.f32 [182, 53]
205 Ops.INDEX dtypes.f32 [182, 57]
206 Ops.INDEX dtypes.f32 [182, 61]
207 Ops.INDEX dtypes.f32 [182, 65]
208 Ops.STACK dtypes.f32 [204, 205, 206, 207]
209 Ops.STORE dtypes.void [203, 208]
210 Ops.SINK dtypes.void [188, 188, 188, 188, 195, 195, 195, 195, 202, 202, 202, 202, 209, 209, 209, 209] KernelInfo(name="E_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.UPCAST)),), estimates=Estimates(ops=176, lds=96, mem=96), optimized=True)
211 Ops.LINEAR dtypes.void [0, 123, 2, 3, 124, 125, 126, 10, 11, 129, 130, 131, 132, 133, 134, 135, 44, 45, 152, 153, 48, 49, 154, 155, 52, 53, 156, 157, 56, 57, 158, 159, 60, 61, 160, 161, 64, 65, 162, 163, 14, 15, 136, 137, 18, 19, 138, 139, 4, 5, 140, 141, 24, 25, 142, 143, 28, 29, 144, 145, 32, 33, 146, 147, 36, 37, 148, 149, 40, 41, 150, 151, 127, 164, 74, 75, 165, 166, 83, 71, 86, 87, 79, 80, 167, 168, 169, 171, 173, 174, 175, 176, 170, 172, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 204, 205, 206, 207, 208, 190, 191, 192, 193, 194, 199, 200, 197, 198, 201, 128, 203, 189, 196, 188, 209, 195, 202, 210]
212 Ops.SOURCE dtypes.void [] "typedef unsigned short unsigned_short16 __attribute__((aligned(2),ext_vector_type(16)));\ntypedef unsigned short unsigned_short16 __attribute__((aligned(2),ext_vector_type(16)));\ntypedef unsigned int unsigned_int16 __attribute__((aligned(4),ext_vector_type(16)));\ntypedef float float16 __attribute__((aligned(4),ext_vector_type(16)));\ntypedef int int16 __attribute__((aligned(4),ext_vector_type(16)));\ntypedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void E_16(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+0));\n  unsigned long val1 = (*(((unsigned long*)(args_16))+1));\n  unsigned short val2 = (*(((unsigned short*)(val1))+0));\n  unsigned short val3 = (*(((unsigned short*)(val1))+1));\n  unsigned short val4 = (*(((unsigned short*)(val1))+10));\n  unsigned short val5 = (*(((unsigned short*)(val1))+11));\n  unsigned short val6 = (*(((unsigned short*)(val1))+12));\n  unsigned short val7 = (*(((unsigned short*)(val1))+13));\n  unsigned short val8 = (*(((unsigned short*)(val1))+14));\n  unsigned short val9 = (*(((unsigned short*)(val1))+15));\n  unsigned short val10 = (*(((unsigned short*)(val1))+2));\n  unsigned short val11 = (*(((unsigned short*)(val1))+3));\n  unsigned short val12 = (*(((unsigned short*)(val1))+4));\n  unsigned short val13 = (*(((unsigned short*)(val1))+5));\n  unsigned short val14 = (*(((unsigned short*)(val1))+6));\n  unsigned short val15 = (*(((unsigned short*)(val1))+7));\n  unsigned short val16 = (*(((unsigned short*)(val1))+8));\n  unsigned short val17 = (*(((unsigned short*)(val1))+9));\n  unsigned_int16 cast0 = __builtin_convertvector(((unsigned_short16)((unsigned_short16)((unsigned_short16){val2,val3,val10,val11,val12,val13,val14,val15,val16,val17,val4,val5,val6,val7,val8,val9}))), unsigned_int16);\n  float16 alu0 = (((float16)((unsigned_int16)((cast0<<16u))))+1.0f);\n  unsigned_int16 cast1 = ((unsigned_int16)((float16)(alu0)));\n  unsigned_int16 alu1 = ((alu0!=alu0)?(cast1|4194304u):(cast1+((cast1>>16u)&1u)+32767u));\n  unsigned_short16 cast2 = __builtin_convertvector((alu1>>16u), unsigned_short16);\n  unsigned_int16 cast3 = __builtin_convertvector(cast2, unsigned_int16);\n  float16 cast4 = ((float16)((unsigned_int16)((cast3<<16u))));\n  *((float4*)((((float*)(val0))+0))) = (float4){cast4[0],cast4[1],cast4[2],cast4[3]};\n  *((float4*)((((float*)(val0))+12))) = (float4){cast4[12],cast4[13],cast4[14],cast4[15]};\n  *((float4*)((((float*)(val0))+4))) = (float4){cast4[4],cast4[5],cast4[6],cast4[7]};\n  *((float4*)((((float*)(val0))+8))) = (float4){cast4[8],cast4[9],cast4[10],cast4[11]};\n}\n"
213 Ops.PROGRAM dtypes.void [210, 211, 212] ProgramInfo(params=(%1, %7), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
214 Ops.SINK dtypes.void [122, 213]

