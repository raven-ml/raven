# tolk D89 golden=eb6499f732457c5772deac39b4107eea
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 65536
2 Ops.PARAM dtypes.f16 [1] ParamArg(slot=0, dtype=dtypes.f16)
3 Ops.CONST dtypes.weakint [] 8
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.SPECIAL dtypes.i32 [4] "gidx0"
6 Ops.CONST dtypes.weakint [] 5
7 Ops.CAST dtypes.i32 [6] dtypes.i32
8 Ops.SHL dtypes.i32 [5, 7]
9 Ops.CONST dtypes.weakint [] 4
10 Ops.CAST dtypes.i32 [9] dtypes.i32
11 Ops.SPECIAL dtypes.i32 [10] "lidx1"
12 Ops.CONST dtypes.weakint [] 3
13 Ops.CAST dtypes.i32 [12] dtypes.i32
14 Ops.SHL dtypes.i32 [11, 13]
15 Ops.ADD dtypes.i32 [8, 14]
16 Ops.CONST dtypes.weakint [] 32
17 Ops.CAST dtypes.i32 [16] dtypes.i32
18 Ops.SPECIAL dtypes.i32 [17] "lidx0"
19 Ops.SHR dtypes.i32 [18, 13]
20 Ops.CONST dtypes.weakint [] 1
21 Ops.CAST dtypes.i32 [20] dtypes.i32
22 Ops.AND dtypes.i32 [19, 21]
23 Ops.CONST dtypes.weakint [] 2
24 Ops.CAST dtypes.i32 [23] dtypes.i32
25 Ops.SHL dtypes.i32 [22, 24]
26 Ops.ADD dtypes.i32 [15, 25]
27 Ops.AND dtypes.i32 [18, 21]
28 Ops.SHL dtypes.i32 [27, 21]
29 Ops.ADD dtypes.i32 [26, 28]
30 Ops.SHR dtypes.i32 [18, 10]
31 Ops.CONST dtypes.weakint [] 10
32 Ops.CAST dtypes.i32 [31] dtypes.i32
33 Ops.SHL dtypes.i32 [30, 32]
34 Ops.SHR dtypes.i32 [18, 21]
35 Ops.AND dtypes.i32 [34, 13]
36 Ops.SHL dtypes.i32 [35, 4]
37 Ops.ADD dtypes.i32 [33, 36]
38 Ops.SPECIAL dtypes.i32 [17] "gidx1"
39 Ops.CONST dtypes.weakint [] 11
40 Ops.CAST dtypes.i32 [39] dtypes.i32
41 Ops.SHL dtypes.i32 [38, 40]
42 Ops.ADD dtypes.i32 [37, 41]
43 Ops.ADD dtypes.i32 [29, 42]
44 Ops.SHRINK dtypes.f16 [2, 43, 24]
45 Ops.ALLOC dtypes.f32 [23] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
46 Ops.AFTER dtypes.f32 [45, 18, 38, 5, 11]
47 Ops.CONST dtypes.weakint [] 0
48 Ops.CAST dtypes.i32 [47] dtypes.i32
49 Ops.INDEX dtypes.f32 [46, 48]
50 Ops.CONST dtypes.weakfloat [] 0.0
51 Ops.CAST dtypes.f32 [50] dtypes.f32
52 Ops.STORE dtypes.void [49, 51]
53 Ops.INDEX dtypes.f32 [46, 21]
54 Ops.STORE dtypes.void [53, 51]
55 Ops.RANGE dtypes.i32 [17] (AxisType.WEAK, 0)
56 Ops.AFTER dtypes.f32 [45, 52, 54, 55]
57 Ops.INDEX dtypes.f32 [56, 48]
58 Ops.PARAM dtypes.f16 [1] ParamArg(slot=1, dtype=dtypes.f16)
59 Ops.SHL dtypes.i32 [55, 13]
60 Ops.ADD dtypes.i32 [25, 59]
61 Ops.ADD dtypes.i32 [60, 28]
62 Ops.ADD dtypes.i32 [42, 61]
63 Ops.SHRINK dtypes.f16 [58, 62, 24]
64 Ops.LOAD dtypes.f16 [63]
65 Ops.PARAM dtypes.f16 [1] ParamArg(slot=2, dtype=dtypes.f16)
66 Ops.SHL dtypes.i32 [55, 40]
67 Ops.ADD dtypes.i32 [37, 66]
68 Ops.ADD dtypes.i32 [29, 67]
69 Ops.SHRINK dtypes.f16 [65, 68, 24]
70 Ops.LOAD dtypes.f16 [69]
71 Ops.LOAD dtypes.f32 [57]
72 Ops.INDEX dtypes.f32 [56, 21]
73 Ops.LOAD dtypes.f32 [72]
74 Ops.STACK dtypes.f32 [71, 73]
75 Ops.WMMA dtypes.f32 [64, 70, 74] ((8, 8, 8), dtypes.f16, 32, None)
76 Ops.INDEX dtypes.f32 [75, 48]
77 Ops.STORE dtypes.void [57, 76]
78 Ops.INDEX dtypes.f32 [75, 21]
79 Ops.STORE dtypes.void [72, 78]
80 Ops.AFTER dtypes.void [0, 77, 79]
81 Ops.END dtypes.void [80, 55]
82 Ops.AFTER dtypes.f32 [45, 81]
83 Ops.INDEX dtypes.f32 [82, 48]
84 Ops.LOAD dtypes.f32 [83]
85 Ops.CAST dtypes.f16 [84] dtypes.f16
86 Ops.INDEX dtypes.f32 [82, 21]
87 Ops.LOAD dtypes.f32 [86]
88 Ops.CAST dtypes.f16 [87] dtypes.f16
89 Ops.STACK dtypes.f16 [85, 88]
90 Ops.STORE dtypes.void [44, 89]
91 Ops.AFTER dtypes.void [0, 90]
92 Ops.SINK dtypes.void [91] KernelInfo(name="r_32_8_4_32_2_32", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL))), optimized=True)
93 Ops.CONST dtypes.weakint [] 24
94 Ops.PARAM dtypes.u8 [93] ParamArg(slot=0, dtype=dtypes.u8, name="args")
95 Ops.BITCAST dtypes.u64 [94] dtypes.u64
96 Ops.INDEX dtypes.u64 [95, 48]
97 Ops.LOAD dtypes.u64 [96]
98 Ops.REFPARAM dtypes.f16 [1, 97] ParamArg(slot=0, dtype=dtypes.f16)
99 Ops.SHRINK dtypes.f16 [98, 43, 24]
100 Ops.BUFFER dtypes.f32 [23] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
101 Ops.AFTER dtypes.f32 [100, 18, 38, 5, 11]
102 Ops.INDEX dtypes.f32 [101, 48]
103 Ops.STORE dtypes.void [102, 51]
104 Ops.INDEX dtypes.f32 [101, 21]
105 Ops.STORE dtypes.void [104, 51]
106 Ops.AFTER dtypes.f32 [100, 103, 105, 55]
107 Ops.INDEX dtypes.f32 [106, 48]
108 Ops.INDEX dtypes.u64 [95, 21]
109 Ops.LOAD dtypes.u64 [108]
110 Ops.REFPARAM dtypes.f16 [1, 109] ParamArg(slot=1, dtype=dtypes.f16)
111 Ops.SHRINK dtypes.f16 [110, 62, 24]
112 Ops.LOAD dtypes.f16 [111]
113 Ops.INDEX dtypes.u64 [95, 24]
114 Ops.LOAD dtypes.u64 [113]
115 Ops.REFPARAM dtypes.f16 [1, 114] ParamArg(slot=2, dtype=dtypes.f16)
116 Ops.SHRINK dtypes.f16 [115, 68, 24]
117 Ops.LOAD dtypes.f16 [116]
118 Ops.LOAD dtypes.f32 [107]
119 Ops.INDEX dtypes.f32 [106, 21]
120 Ops.LOAD dtypes.f32 [119]
121 Ops.STACK dtypes.f32 [118, 120]
122 Ops.WMMA dtypes.f32 [112, 117, 121] ((8, 8, 8), dtypes.f16, 32, None)
123 Ops.INDEX dtypes.f32 [122, 48]
124 Ops.STORE dtypes.void [107, 123]
125 Ops.INDEX dtypes.f32 [122, 21]
126 Ops.STORE dtypes.void [119, 125]
127 Ops.AFTER dtypes.void [0, 124, 126]
128 Ops.END dtypes.void [127, 55]
129 Ops.AFTER dtypes.f32 [100, 128]
130 Ops.INDEX dtypes.f32 [129, 48]
131 Ops.LOAD dtypes.f32 [130]
132 Ops.CAST dtypes.f16 [131] dtypes.f16
133 Ops.INDEX dtypes.f32 [129, 21]
134 Ops.LOAD dtypes.f32 [133]
135 Ops.CAST dtypes.f16 [134] dtypes.f16
136 Ops.STACK dtypes.f16 [132, 135]
137 Ops.STORE dtypes.void [99, 136]
138 Ops.AFTER dtypes.void [0, 137]
139 Ops.SINK dtypes.void [138] KernelInfo(name="r_32_8_4_32_2_32", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL))), estimates=Estimates(ops=33554432, lds=8519680, mem=393216), optimized=True)
140 Ops.LINEAR dtypes.void [93, 94, 23, 100, 20, 21, 95, 108, 109, 24, 113, 114, 3, 4, 5, 16, 17, 38, 18, 9, 10, 11, 1, 110, 115, 101, 0, 47, 48, 102, 104, 50, 51, 31, 32, 39, 40, 12, 13, 6, 7, 8, 14, 15, 19, 22, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 41, 42, 103, 105, 55, 59, 60, 61, 62, 111, 112, 66, 67, 68, 116, 117, 106, 107, 118, 119, 120, 121, 122, 123, 125, 124, 126, 127, 128, 96, 97, 129, 130, 131, 133, 134, 98, 132, 135, 136, 43, 99, 137, 138, 139]
141 Ops.SOURCE dtypes.void [] "#include <metal_stdlib>\nusing namespace metal;\nfloat2 __WMMA_8_8_8_f16_f32(half2 a, half2 b, float2 c){\n  simdgroup_half8x8 mat_a, mat_b; simdgroup_float8x8 mat_c;\n  mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];\n  mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];\n  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);\n  return float2(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);\n}\nkernel void r_32_8_4_32_2_32(device unsigned char* args_24, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n  float buf0[2];\n  unsigned long val0 = (*(((device unsigned long*)(args_24))+1));\n  unsigned long val1 = (*(((device unsigned long*)(args_24))+2));\n  int gidx0 = gid.x; /* 8 */\n  int gidx1 = gid.y; /* 32 */\n  int lidx0 = lid.x; /* 32 */\n  int lidx1 = lid.y; /* 4 */\n  int alu0 = (((lidx0>>3)&1)<<2);\n  int alu1 = ((lidx0&1)<<1);\n  int alu2 = ((gidx0<<5)+(lidx1<<3)+alu0+alu1);\n  int alu3 = (((lidx0>>4)<<10)+(((lidx0>>1)&3)<<8));\n  int alu4 = (alu3+(gidx1<<11));\n  *(buf0+0) = 0.0f;\n  *(buf0+1) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 32; Lidx0++) {\n    half2 val2 = (*((device half2*)((((device half*)(val0))+(alu4+(alu0+(Lidx0<<3)+alu1))))));\n    half2 val3 = (*((device half2*)((((device half*)(val1))+(alu2+(alu3+(Lidx0<<11)))))));\n    float2 wmma0 = __WMMA_8_8_8_f16_f32(val2, val3, float2((*(buf0+0)),(*(buf0+1))));\n    *(buf0+0) = wmma0.x;\n    *(buf0+1) = wmma0.y;\n  }\n  unsigned long val4 = (*(((device unsigned long*)(args_24))+0));\n  *((device half2*)((((device half*)(val4))+(alu2+alu4)))) = half2(((half)((*(buf0+0)))),((half)((*(buf0+1)))));\n}"
142 Ops.PROGRAM dtypes.void [139, 140, 141] ProgramInfo(global_size=(8, 32, 1), local_size=(32, 4, 1), params=(%2, %58, %65), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="METAL", renderer="METAL", arch="Apple9"))
143 Ops.SINK dtypes.void [92, 142]

