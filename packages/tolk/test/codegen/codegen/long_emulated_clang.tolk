# tolk D80 D151 golden=8a720266db4f97adf46799c8239c3ce5
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 32
2 Ops.PARAM dtypes.i32 [1] ParamArg(slot=0, dtype=dtypes.i32)
3 Ops.PARAM dtypes.i32 [0] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 16), name="block_lo", addrspace=AddrSpace.ALU)
4 Ops.PARAM dtypes.i32 [0] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 16), name="block_hi", addrspace=AddrSpace.ALU)
5 Ops.SUB dtypes.i32 [4, 3]
6 Ops.RANGE dtypes.i32 [5] (AxisType.WEAK, 0)
7 Ops.ADD dtypes.i32 [3, 6]
8 Ops.CONST dtypes.weakint [] 1
9 Ops.CAST dtypes.i32 [8] dtypes.i32
10 Ops.SHL dtypes.i32 [7, 9]
11 Ops.INDEX dtypes.i32 [2, 10]
12 Ops.PARAM dtypes.i32 [1] ParamArg(slot=1, dtype=dtypes.i32)
13 Ops.INDEX dtypes.i32 [12, 10]
14 Ops.LOAD dtypes.i32 [13]
15 Ops.BITCAST dtypes.u32 [14] dtypes.u32
16 Ops.CONST dtypes.weakint [] 65535
17 Ops.CAST dtypes.u32 [16] dtypes.u32
18 Ops.AND dtypes.u32 [15, 17]
19 Ops.CONST dtypes.weakint [] 0
20 Ops.CAST dtypes.u32 [19] dtypes.u32
21 Ops.MUL dtypes.u32 [18, 20]
22 Ops.CONST dtypes.weakint [] 16
23 Ops.CAST dtypes.u32 [22] dtypes.u32
24 Ops.SHL dtypes.u32 [21, 23]
25 Ops.BITCAST dtypes.i32 [24] dtypes.i32
26 Ops.CAST dtypes.u32 [25] dtypes.u32
27 Ops.SHR dtypes.u32 [15, 23]
28 Ops.CONST dtypes.weakint [] 3
29 Ops.CAST dtypes.u32 [28] dtypes.u32
30 Ops.MUL dtypes.u32 [27, 29]
31 Ops.SHL dtypes.u32 [30, 23]
32 Ops.BITCAST dtypes.i32 [31] dtypes.i32
33 Ops.CAST dtypes.u32 [32] dtypes.u32
34 Ops.ADD dtypes.u32 [26, 33]
35 Ops.CAST dtypes.i32 [34] dtypes.i32
36 Ops.CAST dtypes.u32 [35] dtypes.u32
37 Ops.MUL dtypes.u32 [18, 29]
38 Ops.BITCAST dtypes.i32 [37] dtypes.i32
39 Ops.CAST dtypes.u32 [38] dtypes.u32
40 Ops.ADD dtypes.u32 [36, 39]
41 Ops.CAST dtypes.i32 [40] dtypes.i32
42 Ops.CAST dtypes.u32 [41] dtypes.u32
43 Ops.CAST dtypes.u32 [8] dtypes.u32
44 Ops.ADD dtypes.u32 [42, 43]
45 Ops.CAST dtypes.i32 [44] dtypes.i32
46 Ops.STORE dtypes.void [11, 45]
47 Ops.ADD dtypes.i32 [10, 9]
48 Ops.INDEX dtypes.i32 [2, 47]
49 Ops.SHR dtypes.u32 [21, 23]
50 Ops.BITCAST dtypes.i32 [49] dtypes.i32
51 Ops.CAST dtypes.u32 [50] dtypes.u32
52 Ops.SHR dtypes.u32 [30, 23]
53 Ops.BITCAST dtypes.i32 [52] dtypes.i32
54 Ops.CAST dtypes.u32 [53] dtypes.u32
55 Ops.ADD dtypes.u32 [51, 54]
56 Ops.CAST dtypes.i32 [55] dtypes.i32
57 Ops.BITCAST dtypes.u32 [35] dtypes.u32
58 Ops.CMPLT dtypes.bool [57, 24]
59 Ops.CAST dtypes.i32 [58] dtypes.i32
60 Ops.ADD dtypes.i32 [56, 59]
61 Ops.CAST dtypes.u32 [60] dtypes.u32
62 Ops.MUL dtypes.u32 [27, 20]
63 Ops.BITCAST dtypes.i32 [62] dtypes.i32
64 Ops.CAST dtypes.i32 [19] dtypes.i32
65 Ops.MUL dtypes.i32 [14, 64]
66 Ops.ADD dtypes.i32 [63, 65]
67 Ops.CAST dtypes.u32 [66] dtypes.u32
68 Ops.INDEX dtypes.i32 [12, 47]
69 Ops.LOAD dtypes.i32 [68]
70 Ops.CAST dtypes.u32 [69] dtypes.u32
71 Ops.MUL dtypes.u32 [70, 29]
72 Ops.CAST dtypes.i32 [71] dtypes.i32
73 Ops.CAST dtypes.u32 [72] dtypes.u32
74 Ops.ADD dtypes.u32 [67, 73]
75 Ops.CAST dtypes.i32 [74] dtypes.i32
76 Ops.CAST dtypes.u32 [75] dtypes.u32
77 Ops.ADD dtypes.u32 [61, 76]
78 Ops.CAST dtypes.i32 [77] dtypes.i32
79 Ops.CAST dtypes.u32 [78] dtypes.u32
80 Ops.BITCAST dtypes.u32 [41] dtypes.u32
81 Ops.CMPLT dtypes.bool [80, 57]
82 Ops.CAST dtypes.i32 [81] dtypes.i32
83 Ops.CAST dtypes.u32 [82] dtypes.u32
84 Ops.ADD dtypes.u32 [79, 83]
85 Ops.CAST dtypes.i32 [84] dtypes.i32
86 Ops.ADD dtypes.i32 [85, 64]
87 Ops.CAST dtypes.u32 [86] dtypes.u32
88 Ops.BITCAST dtypes.u32 [45] dtypes.u32
89 Ops.CMPLT dtypes.bool [88, 80]
90 Ops.CAST dtypes.i32 [89] dtypes.i32
91 Ops.CAST dtypes.u32 [90] dtypes.u32
92 Ops.ADD dtypes.u32 [87, 91]
93 Ops.CAST dtypes.i32 [92] dtypes.i32
94 Ops.STORE dtypes.void [48, 93]
95 Ops.AFTER dtypes.void [0, 46, 94]
96 Ops.END dtypes.void [95, 6]
97 Ops.SINK dtypes.void [96] KernelInfo(name="E_16", optimized=True, split=(16, 2, 3))
98 Ops.PARAM dtypes.u8 [1] ParamArg(slot=0, dtype=dtypes.u8, name="args")
99 Ops.BITCAST dtypes.u64 [98] dtypes.u64
100 Ops.INDEX dtypes.u64 [99, 64]
101 Ops.LOAD dtypes.u64 [100]
102 Ops.REFPARAM dtypes.i32 [1, 101] ParamArg(slot=0, dtype=dtypes.i32)
103 Ops.BITCAST dtypes.i32 [98] dtypes.i32
104 Ops.CONST dtypes.weakint [] 4
105 Ops.CAST dtypes.i32 [104] dtypes.i32
106 Ops.INDEX dtypes.i32 [103, 105]
107 Ops.LOAD dtypes.i32 [106]
108 Ops.CONST dtypes.weakint [] 5
109 Ops.CAST dtypes.i32 [108] dtypes.i32
110 Ops.INDEX dtypes.i32 [103, 109]
111 Ops.LOAD dtypes.i32 [110]
112 Ops.SUB dtypes.i32 [111, 107]
113 Ops.RANGE dtypes.i32 [112] (AxisType.WEAK, 0)
114 Ops.ADD dtypes.i32 [107, 113]
115 Ops.SHL dtypes.i32 [114, 9]
116 Ops.INDEX dtypes.i32 [102, 115]
117 Ops.INDEX dtypes.u64 [99, 9]
118 Ops.LOAD dtypes.u64 [117]
119 Ops.REFPARAM dtypes.i32 [1, 118] ParamArg(slot=1, dtype=dtypes.i32)
120 Ops.INDEX dtypes.i32 [119, 115]
121 Ops.LOAD dtypes.i32 [120]
122 Ops.BITCAST dtypes.u32 [121] dtypes.u32
123 Ops.AND dtypes.u32 [122, 17]
124 Ops.MUL dtypes.u32 [123, 20]
125 Ops.SHL dtypes.u32 [124, 23]
126 Ops.BITCAST dtypes.i32 [125] dtypes.i32
127 Ops.CAST dtypes.u32 [126] dtypes.u32
128 Ops.SHR dtypes.u32 [122, 23]
129 Ops.MUL dtypes.u32 [128, 29]
130 Ops.SHL dtypes.u32 [129, 23]
131 Ops.BITCAST dtypes.i32 [130] dtypes.i32
132 Ops.CAST dtypes.u32 [131] dtypes.u32
133 Ops.ADD dtypes.u32 [127, 132]
134 Ops.CAST dtypes.i32 [133] dtypes.i32
135 Ops.CAST dtypes.u32 [134] dtypes.u32
136 Ops.MUL dtypes.u32 [123, 29]
137 Ops.BITCAST dtypes.i32 [136] dtypes.i32
138 Ops.CAST dtypes.u32 [137] dtypes.u32
139 Ops.ADD dtypes.u32 [135, 138]
140 Ops.CAST dtypes.i32 [139] dtypes.i32
141 Ops.CAST dtypes.u32 [140] dtypes.u32
142 Ops.ADD dtypes.u32 [141, 43]
143 Ops.CAST dtypes.i32 [142] dtypes.i32
144 Ops.STORE dtypes.void [116, 143]
145 Ops.ADD dtypes.i32 [115, 9]
146 Ops.INDEX dtypes.i32 [102, 145]
147 Ops.SHR dtypes.u32 [124, 23]
148 Ops.BITCAST dtypes.i32 [147] dtypes.i32
149 Ops.CAST dtypes.u32 [148] dtypes.u32
150 Ops.SHR dtypes.u32 [129, 23]
151 Ops.BITCAST dtypes.i32 [150] dtypes.i32
152 Ops.CAST dtypes.u32 [151] dtypes.u32
153 Ops.ADD dtypes.u32 [149, 152]
154 Ops.CAST dtypes.i32 [153] dtypes.i32
155 Ops.BITCAST dtypes.u32 [134] dtypes.u32
156 Ops.CMPLT dtypes.bool [155, 125]
157 Ops.CAST dtypes.i32 [156] dtypes.i32
158 Ops.ADD dtypes.i32 [154, 157]
159 Ops.CAST dtypes.u32 [158] dtypes.u32
160 Ops.MUL dtypes.u32 [128, 20]
161 Ops.BITCAST dtypes.i32 [160] dtypes.i32
162 Ops.MUL dtypes.i32 [121, 64]
163 Ops.ADD dtypes.i32 [161, 162]
164 Ops.CAST dtypes.u32 [163] dtypes.u32
165 Ops.INDEX dtypes.i32 [119, 145]
166 Ops.LOAD dtypes.i32 [165]
167 Ops.CAST dtypes.u32 [166] dtypes.u32
168 Ops.MUL dtypes.u32 [167, 29]
169 Ops.CAST dtypes.i32 [168] dtypes.i32
170 Ops.CAST dtypes.u32 [169] dtypes.u32
171 Ops.ADD dtypes.u32 [164, 170]
172 Ops.CAST dtypes.i32 [171] dtypes.i32
173 Ops.CAST dtypes.u32 [172] dtypes.u32
174 Ops.ADD dtypes.u32 [159, 173]
175 Ops.CAST dtypes.i32 [174] dtypes.i32
176 Ops.CAST dtypes.u32 [175] dtypes.u32
177 Ops.BITCAST dtypes.u32 [140] dtypes.u32
178 Ops.CMPLT dtypes.bool [177, 155]
179 Ops.CAST dtypes.i32 [178] dtypes.i32
180 Ops.CAST dtypes.u32 [179] dtypes.u32
181 Ops.ADD dtypes.u32 [176, 180]
182 Ops.CAST dtypes.i32 [181] dtypes.i32
183 Ops.ADD dtypes.i32 [182, 64]
184 Ops.CAST dtypes.u32 [183] dtypes.u32
185 Ops.BITCAST dtypes.u32 [143] dtypes.u32
186 Ops.CMPLT dtypes.bool [185, 177]
187 Ops.CAST dtypes.i32 [186] dtypes.i32
188 Ops.CAST dtypes.u32 [187] dtypes.u32
189 Ops.ADD dtypes.u32 [184, 188]
190 Ops.CAST dtypes.i32 [189] dtypes.i32
191 Ops.STORE dtypes.void [146, 190]
192 Ops.AFTER dtypes.void [0, 144, 191]
193 Ops.END dtypes.void [192, 113]
194 Ops.SINK dtypes.void [193] KernelInfo(name="E_16", estimates=Estimates(ops=416, lds=256, mem=256), optimized=True, split=(16, 2, 3))
195 Ops.LINEAR dtypes.void [1, 98, 104, 105, 103, 106, 107, 108, 109, 110, 111, 19, 64, 99, 100, 101, 8, 9, 117, 118, 102, 119, 0, 20, 43, 22, 23, 28, 29, 16, 17, 112, 113, 114, 115, 145, 165, 166, 120, 121, 146, 116, 122, 128, 129, 130, 131, 132, 123, 124, 125, 126, 127, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 150, 151, 152, 147, 148, 149, 153, 154, 167, 168, 169, 170, 160, 161, 162, 163, 164, 171, 172, 155, 156, 157, 173, 158, 159, 174, 175, 177, 178, 179, 176, 180, 181, 182, 185, 186, 187, 188, 183, 184, 189, 190, 191, 144, 192, 193, 194]
196 Ops.SOURCE dtypes.void [] "\n__attribute__((min_vector_width(512))) void E_16(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = ((val0+Lidx0)<<1);\n    int alu1 = (alu0+1);\n    int val4 = (*(((int*)(val3))+alu1));\n    int val5 = (*(((int*)(val3))+alu0));\n    unsigned int cast0 = __builtin_bit_cast(unsigned int, (int)(val5));\n    unsigned int alu2 = (cast0>>16u);\n    unsigned int alu3 = (alu2*3u);\n    unsigned int alu4 = (cast0&65535u);\n    unsigned int alu5 = (alu4*0u);\n    unsigned int alu6 = (alu5<<16u);\n    int cast1 = ((int)((((unsigned int)(__builtin_bit_cast(int, (unsigned int)(alu6))))+((unsigned int)(__builtin_bit_cast(int, (unsigned int)((alu3<<16u))))))));\n    int cast2 = ((int)((((unsigned int)(cast1))+((unsigned int)(__builtin_bit_cast(int, (unsigned int)((alu4*3u))))))));\n    int cast3 = ((int)((((unsigned int)(cast2))+1u)));\n    unsigned int cast4 = __builtin_bit_cast(unsigned int, (int)(cast1));\n    unsigned int cast5 = __builtin_bit_cast(unsigned int, (int)(cast2));\n    *(((int*)(val2))+alu1) = ((int)((((unsigned int)((((int)((((unsigned int)(((int)((((unsigned int)((((int)((((unsigned int)(__builtin_bit_cast(int, (unsigned int)((alu5>>16u)))))+((unsigned int)(__builtin_bit_cast(int, (unsigned int)((alu3>>16u))))))))+((int)((cast4<alu6))))))+((unsigned int)(((int)((((unsigned int)((__builtin_bit_cast(int, (unsigned int)((alu2*0u)))+(val5*0))))+((unsigned int)(((int)((((unsigned int)(val4))*3u)))))))))))))))+((unsigned int)(((int)((cast5<cast4))))))))+0)))+((unsigned int)(((int)((__builtin_bit_cast(unsigned int, (int)(cast3))<cast5))))))));\n    *(((int*)(val2))+alu0) = cast3;\n  }\n}\n"
197 Ops.PROGRAM dtypes.void [194, 195, 196] ProgramInfo(global_size=(16, 1, 1), params=(%2, %12, %3, %4), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
198 Ops.SINK dtypes.void [97, 197]

