# tolk D25 D89 golden=43441e971661d6ab74208fc0c5785765
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 1024
2 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.CONST dtypes.weakint [] 16
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.SPECIAL dtypes.i32 [4] "gidx0"
6 Ops.CONST dtypes.weakint [] 6
7 Ops.CAST dtypes.i32 [6] dtypes.i32
8 Ops.SHL dtypes.i32 [5, 7]
9 Ops.CONST dtypes.weakint [] 32
10 Ops.CAST dtypes.i32 [9] dtypes.i32
11 Ops.SPECIAL dtypes.i32 [10] "lidx0"
12 Ops.ADD dtypes.i32 [8, 11]
13 Ops.INDEX dtypes.f32 [2, 12]
14 Ops.CONST dtypes.weakint [] 2
15 Ops.ALLOC dtypes.f32 [14] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
16 Ops.AFTER dtypes.f32 [15, 5, 11]
17 Ops.CONST dtypes.weakint [] 0
18 Ops.CAST dtypes.i32 [17] dtypes.i32
19 Ops.INDEX dtypes.f32 [16, 18]
20 Ops.CONST dtypes.weakfloat [] 0.0
21 Ops.CAST dtypes.f32 [20] dtypes.f32
22 Ops.STORE dtypes.void [19, 21]
23 Ops.CONST dtypes.weakint [] 1
24 Ops.CAST dtypes.i32 [23] dtypes.i32
25 Ops.INDEX dtypes.f32 [16, 24]
26 Ops.STORE dtypes.void [25, 21]
27 Ops.ALLOC dtypes.f32 [14] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
28 Ops.SPECIAL dtypes.i32 [10] "lidx1"
29 Ops.AFTER dtypes.f32 [27, 28, 5, 11]
30 Ops.INDEX dtypes.f32 [29, 18]
31 Ops.STORE dtypes.void [30, 21]
32 Ops.INDEX dtypes.f32 [29, 24]
33 Ops.STORE dtypes.void [32, 21]
34 Ops.RANGE dtypes.i32 [10] (AxisType.WEAK, 0)
35 Ops.AFTER dtypes.f32 [27, 31, 33, 34]
36 Ops.INDEX dtypes.f32 [35, 18]
37 Ops.PARAM dtypes.f32 [1] ParamArg(slot=1, dtype=dtypes.f32)
38 Ops.CONST dtypes.weakint [] 5
39 Ops.CAST dtypes.i32 [38] dtypes.i32
40 Ops.SHL dtypes.i32 [34, 39]
41 Ops.ADD dtypes.i32 [40, 28]
42 Ops.INDEX dtypes.f32 [37, 41]
43 Ops.LOAD dtypes.f32 [42]
44 Ops.CONST dtypes.weakint [] 1048576
45 Ops.PARAM dtypes.f32 [44] ParamArg(slot=2, dtype=dtypes.f32)
46 Ops.CONST dtypes.weakint [] 10
47 Ops.CAST dtypes.i32 [46] dtypes.i32
48 Ops.SHL dtypes.i32 [28, 47]
49 Ops.CONST dtypes.weakint [] 15
50 Ops.CAST dtypes.i32 [49] dtypes.i32
51 Ops.SHL dtypes.i32 [34, 50]
52 Ops.ADD dtypes.i32 [48, 51]
53 Ops.ADD dtypes.i32 [12, 52]
54 Ops.INDEX dtypes.f32 [45, 53]
55 Ops.LOAD dtypes.f32 [54]
56 Ops.LOAD dtypes.f32 [36]
57 Ops.MULACC dtypes.f32 [43, 55, 56]
58 Ops.STORE dtypes.void [36, 57]
59 Ops.INDEX dtypes.f32 [35, 24]
60 Ops.ADD dtypes.i32 [53, 10]
61 Ops.INDEX dtypes.f32 [45, 60]
62 Ops.LOAD dtypes.f32 [61]
63 Ops.LOAD dtypes.f32 [59]
64 Ops.MULACC dtypes.f32 [43, 62, 63]
65 Ops.STORE dtypes.void [59, 64]
66 Ops.AFTER dtypes.void [0, 58, 65]
67 Ops.END dtypes.void [66, 34]
68 Ops.AFTER dtypes.i32 [10, 67]
69 Ops.RANGE dtypes.i32 [68] (AxisType.WEAK, 104)
70 Ops.AFTER dtypes.f32 [15, 22, 26, 69]
71 Ops.INDEX dtypes.f32 [70, 18]
72 Ops.LOAD dtypes.f32 [71]
73 Ops.CONST dtypes.weakint [] 2048
74 Ops.ALLOC dtypes.f32 [73] ParamArg(slot=2, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
75 Ops.SHL dtypes.i32 [11, 7]
76 Ops.SHL dtypes.i32 [28, 24]
77 Ops.ADD dtypes.i32 [75, 76]
78 Ops.CAST dtypes.i32 [14] dtypes.i32
79 Ops.SHRINK dtypes.f32 [74, 77, 78]
80 Ops.AFTER dtypes.f32 [27, 67]
81 Ops.INDEX dtypes.f32 [80, 18]
82 Ops.LOAD dtypes.f32 [81]
83 Ops.INDEX dtypes.f32 [80, 24]
84 Ops.LOAD dtypes.f32 [83]
85 Ops.STACK dtypes.f32 [82, 84]
86 Ops.STORE dtypes.void [79, 85]
87 Ops.AFTER dtypes.void [0, 86]
88 Ops.BARRIER dtypes.void [87]
89 Ops.AFTER dtypes.f32 [74, 88]
90 Ops.SHL dtypes.i32 [69, 24]
91 Ops.ADD dtypes.i32 [75, 90]
92 Ops.SHRINK dtypes.f32 [89, 91, 78]
93 Ops.LOAD dtypes.f32 [92]
94 Ops.INDEX dtypes.f32 [93, 18]
95 Ops.ADD dtypes.f32 [72, 94]
96 Ops.STORE dtypes.void [71, 95]
97 Ops.INDEX dtypes.f32 [70, 24]
98 Ops.LOAD dtypes.f32 [97]
99 Ops.INDEX dtypes.f32 [93, 24]
100 Ops.ADD dtypes.f32 [98, 99]
101 Ops.STORE dtypes.void [97, 100]
102 Ops.AFTER dtypes.void [0, 96, 101]
103 Ops.END dtypes.void [102, 69]
104 Ops.AFTER dtypes.f32 [15, 103]
105 Ops.INDEX dtypes.f32 [104, 18]
106 Ops.LOAD dtypes.f32 [105]
107 Ops.CMPEQ dtypes.bool [28, 18]
108 Ops.STORE dtypes.void [13, 106, 107]
109 Ops.ADD dtypes.i32 [12, 10]
110 Ops.INDEX dtypes.f32 [2, 109]
111 Ops.INDEX dtypes.f32 [104, 24]
112 Ops.LOAD dtypes.f32 [111]
113 Ops.STORE dtypes.void [110, 112, 107]
114 Ops.AFTER dtypes.void [0, 108, 113]
115 Ops.SINK dtypes.void [114] KernelInfo(name="r_16_32_32_2_32", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(32, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(32, AxisType.LOCAL))), optimized=True)
116 Ops.CONST dtypes.weakint [] 24
117 Ops.PARAM dtypes.u8 [116] ParamArg(slot=0, dtype=dtypes.u8, name="args")
118 Ops.BITCAST dtypes.u64 [117] dtypes.u64
119 Ops.INDEX dtypes.u64 [118, 18]
120 Ops.LOAD dtypes.u64 [119]
121 Ops.REFPARAM dtypes.f32 [1, 120] ParamArg(slot=0, dtype=dtypes.f32)
122 Ops.INDEX dtypes.f32 [121, 12]
123 Ops.BUFFER dtypes.f32 [14] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
124 Ops.AFTER dtypes.f32 [123, 5, 11]
125 Ops.INDEX dtypes.f32 [124, 18]
126 Ops.STORE dtypes.void [125, 21]
127 Ops.INDEX dtypes.f32 [124, 24]
128 Ops.STORE dtypes.void [127, 21]
129 Ops.BUFFER dtypes.f32 [14] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
130 Ops.AFTER dtypes.f32 [129, 28, 5, 11]
131 Ops.INDEX dtypes.f32 [130, 18]
132 Ops.STORE dtypes.void [131, 21]
133 Ops.INDEX dtypes.f32 [130, 24]
134 Ops.STORE dtypes.void [133, 21]
135 Ops.AFTER dtypes.f32 [129, 132, 134, 34]
136 Ops.INDEX dtypes.f32 [135, 18]
137 Ops.INDEX dtypes.u64 [118, 24]
138 Ops.LOAD dtypes.u64 [137]
139 Ops.REFPARAM dtypes.f32 [1, 138] ParamArg(slot=1, dtype=dtypes.f32)
140 Ops.INDEX dtypes.f32 [139, 41]
141 Ops.LOAD dtypes.f32 [140]
142 Ops.INDEX dtypes.u64 [118, 78]
143 Ops.LOAD dtypes.u64 [142]
144 Ops.REFPARAM dtypes.f32 [44, 143] ParamArg(slot=2, dtype=dtypes.f32)
145 Ops.INDEX dtypes.f32 [144, 53]
146 Ops.LOAD dtypes.f32 [145]
147 Ops.LOAD dtypes.f32 [136]
148 Ops.MULACC dtypes.f32 [141, 146, 147]
149 Ops.STORE dtypes.void [136, 148]
150 Ops.INDEX dtypes.f32 [135, 24]
151 Ops.INDEX dtypes.f32 [144, 60]
152 Ops.LOAD dtypes.f32 [151]
153 Ops.LOAD dtypes.f32 [150]
154 Ops.MULACC dtypes.f32 [141, 152, 153]
155 Ops.STORE dtypes.void [150, 154]
156 Ops.AFTER dtypes.void [0, 149, 155]
157 Ops.END dtypes.void [156, 34]
158 Ops.AFTER dtypes.i32 [10, 157]
159 Ops.RANGE dtypes.i32 [158] (AxisType.WEAK, 104)
160 Ops.AFTER dtypes.f32 [123, 126, 128, 159]
161 Ops.INDEX dtypes.f32 [160, 18]
162 Ops.LOAD dtypes.f32 [161]
163 Ops.BUFFER dtypes.f32 [73] ParamArg(slot=2, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
164 Ops.SHRINK dtypes.f32 [163, 77, 78]
165 Ops.AFTER dtypes.f32 [129, 157]
166 Ops.INDEX dtypes.f32 [165, 18]
167 Ops.LOAD dtypes.f32 [166]
168 Ops.INDEX dtypes.f32 [165, 24]
169 Ops.LOAD dtypes.f32 [168]
170 Ops.STACK dtypes.f32 [167, 169]
171 Ops.STORE dtypes.void [164, 170]
172 Ops.AFTER dtypes.void [0, 171]
173 Ops.BARRIER dtypes.void [172]
174 Ops.AFTER dtypes.f32 [163, 173]
175 Ops.SHL dtypes.i32 [159, 24]
176 Ops.ADD dtypes.i32 [75, 175]
177 Ops.SHRINK dtypes.f32 [174, 176, 78]
178 Ops.LOAD dtypes.f32 [177]
179 Ops.INDEX dtypes.f32 [178, 18]
180 Ops.ADD dtypes.f32 [162, 179]
181 Ops.STORE dtypes.void [161, 180]
182 Ops.INDEX dtypes.f32 [160, 24]
183 Ops.LOAD dtypes.f32 [182]
184 Ops.INDEX dtypes.f32 [178, 24]
185 Ops.ADD dtypes.f32 [183, 184]
186 Ops.STORE dtypes.void [182, 185]
187 Ops.AFTER dtypes.void [0, 181, 186]
188 Ops.END dtypes.void [187, 159]
189 Ops.AFTER dtypes.f32 [123, 188]
190 Ops.INDEX dtypes.f32 [189, 18]
191 Ops.LOAD dtypes.f32 [190]
192 Ops.STORE dtypes.void [122, 191]
193 Ops.INDEX dtypes.f32 [121, 109]
194 Ops.INDEX dtypes.f32 [189, 24]
195 Ops.LOAD dtypes.f32 [194]
196 Ops.STORE dtypes.void [193, 195]
197 Ops.AFTER dtypes.void [0, 192, 196]
198 Ops.SINK dtypes.void [197] KernelInfo(name="r_16_32_32_2_32", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(32, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(32, AxisType.LOCAL))), estimates=Estimates(ops=3162112, lds=10747904, mem=4202496), optimized=True)
199 Ops.IF dtypes.void [107, 193]
200 Ops.ENDIF dtypes.void [199]
201 Ops.IF dtypes.void [107, 122]
202 Ops.ENDIF dtypes.void [201]
203 Ops.LINEAR dtypes.void [116, 117, 14, 129, 123, 73, 163, 23, 24, 118, 137, 138, 78, 142, 143, 3, 4, 5, 9, 10, 11, 28, 1, 139, 44, 144, 130, 0, 17, 18, 131, 133, 20, 21, 46, 47, 49, 50, 38, 39, 6, 7, 8, 12, 48, 132, 134, 34, 40, 41, 140, 141, 51, 52, 53, 60, 151, 152, 145, 146, 135, 136, 147, 150, 153, 154, 148, 149, 155, 156, 157, 165, 166, 167, 168, 169, 170, 75, 76, 77, 164, 171, 172, 158, 124, 173, 174, 125, 127, 126, 128, 159, 160, 161, 162, 182, 183, 175, 176, 177, 178, 179, 184, 180, 185, 181, 186, 187, 188, 119, 120, 189, 190, 191, 194, 195, 121, 109, 193, 122, 107, 199, 196, 200, 201, 192, 202, 197, 198]
204 Ops.SOURCE dtypes.void [] "#include <metal_stdlib>\nusing namespace metal;\nkernel void r_16_32_32_2_32(device unsigned char* args_24, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n  float buf0[2];\n  float buf1[2];\n  threadgroup __attribute__((aligned(16))) float buf2[2048];\n  unsigned long val0 = (*(((device unsigned long*)(args_24))+1));\n  unsigned long val1 = (*(((device unsigned long*)(args_24))+2));\n  int gidx0 = gid.x; /* 16 */\n  int lidx0 = lid.x; /* 32 */\n  int lidx1 = lid.y; /* 32 */\n  int alu0 = ((gidx0<<6)+lidx0);\n  *(buf0+0) = 0.0f;\n  *(buf0+1) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 32; Lidx0++) {\n    float val2 = (*(((device float*)(val0))+((Lidx0<<5)+lidx1)));\n    int alu3 = (alu0+((lidx1<<10)+(Lidx0<<15)));\n    float val3 = (*(((device float*)(val1))+(alu3+32)));\n    float val4 = (*(((device float*)(val1))+alu3));\n    *(buf0+0) = fma(val2,val4,(*(buf0+0)));\n    *(buf0+1) = fma(val2,val3,(*(buf0+1)));\n  }\n  int alu7 = (lidx0<<6);\n  *((threadgroup __attribute__((aligned(16))) float2*)((buf2+(alu7+(lidx1<<1))))) = float2((*(buf0+0)),(*(buf0+1)));\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  *(buf1+0) = 0.0f;\n  *(buf1+1) = 0.0f;\n  for (int Lidx104 = 0; Lidx104 < 32; Lidx104++) {\n    float2 val5 = (*((threadgroup __attribute__((aligned(16))) float2*)((buf2+(alu7+(Lidx104<<1))))));\n    *(buf1+0) = ((*(buf1+0))+val5.x);\n    *(buf1+1) = ((*(buf1+1))+val5.y);\n  }\n  unsigned long val6 = (*(((device unsigned long*)(args_24))+0));\n  bool alu15 = (lidx1==0);\n  if (alu15) {\n    *(((device float*)(val6))+(alu0+32)) = (*(buf1+1));\n  }\n  if (alu15) {\n    *(((device float*)(val6))+alu0) = (*(buf1+0));\n  }\n}"
205 Ops.PROGRAM dtypes.void [198, 203, 204] ProgramInfo(global_size=(16, 1, 1), local_size=(32, 32, 1), params=(%2, %37, %45), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="METAL", renderer="METAL", arch="Apple9"))
206 Ops.SINK dtypes.void [115, 205]

