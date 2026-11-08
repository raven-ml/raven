# tolk D89 golden=17a9d4375ae0de29f9e326608094bb5e
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 65536
2 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
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
44 Ops.SHRINK dtypes.f32 [2, 43, 24]
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
58 Ops.PARAM dtypes.f32 [1] ParamArg(slot=1, dtype=dtypes.f32)
59 Ops.SHL dtypes.i32 [55, 13]
60 Ops.ADD dtypes.i32 [25, 59]
61 Ops.ADD dtypes.i32 [60, 28]
62 Ops.ADD dtypes.i32 [42, 61]
63 Ops.SHRINK dtypes.f32 [58, 62, 24]
64 Ops.LOAD dtypes.f32 [63]
65 Ops.PARAM dtypes.f32 [1] ParamArg(slot=2, dtype=dtypes.f32)
66 Ops.SHL dtypes.i32 [55, 40]
67 Ops.ADD dtypes.i32 [37, 66]
68 Ops.ADD dtypes.i32 [29, 67]
69 Ops.SHRINK dtypes.f32 [65, 68, 24]
70 Ops.LOAD dtypes.f32 [69]
71 Ops.LOAD dtypes.f32 [57]
72 Ops.INDEX dtypes.f32 [56, 21]
73 Ops.LOAD dtypes.f32 [72]
74 Ops.STACK dtypes.f32 [71, 73]
75 Ops.WMMA dtypes.f32 [64, 70, 74] ((8, 8, 8), dtypes.f32, 32, None)
76 Ops.INDEX dtypes.f32 [75, 48]
77 Ops.STORE dtypes.void [57, 76]
78 Ops.INDEX dtypes.f32 [75, 21]
79 Ops.STORE dtypes.void [72, 78]
80 Ops.AFTER dtypes.void [0, 77, 79]
81 Ops.END dtypes.void [80, 55]
82 Ops.AFTER dtypes.f32 [45, 81]
83 Ops.INDEX dtypes.f32 [82, 48]
84 Ops.LOAD dtypes.f32 [83]
85 Ops.INDEX dtypes.f32 [82, 21]
86 Ops.LOAD dtypes.f32 [85]
87 Ops.STACK dtypes.f32 [84, 86]
88 Ops.STORE dtypes.void [44, 87]
89 Ops.AFTER dtypes.void [0, 88]
90 Ops.SINK dtypes.void [89] KernelInfo(name="r_32_8_4_32_2_32", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL))), optimized=True)
91 Ops.CONST dtypes.weakint [] 24
92 Ops.PARAM dtypes.u8 [91] ParamArg(slot=0, dtype=dtypes.u8, name="args")
93 Ops.BITCAST dtypes.u64 [92] dtypes.u64
94 Ops.INDEX dtypes.u64 [93, 48]
95 Ops.LOAD dtypes.u64 [94]
96 Ops.REFPARAM dtypes.f32 [1, 95] ParamArg(slot=0, dtype=dtypes.f32)
97 Ops.SHRINK dtypes.f32 [96, 43, 24]
98 Ops.BUFFER dtypes.f32 [23] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
99 Ops.AFTER dtypes.f32 [98, 18, 38, 5, 11]
100 Ops.INDEX dtypes.f32 [99, 48]
101 Ops.STORE dtypes.void [100, 51]
102 Ops.INDEX dtypes.f32 [99, 21]
103 Ops.STORE dtypes.void [102, 51]
104 Ops.AFTER dtypes.f32 [98, 101, 103, 55]
105 Ops.INDEX dtypes.f32 [104, 48]
106 Ops.INDEX dtypes.u64 [93, 21]
107 Ops.LOAD dtypes.u64 [106]
108 Ops.REFPARAM dtypes.f32 [1, 107] ParamArg(slot=1, dtype=dtypes.f32)
109 Ops.SHRINK dtypes.f32 [108, 62, 24]
110 Ops.LOAD dtypes.f32 [109]
111 Ops.INDEX dtypes.u64 [93, 24]
112 Ops.LOAD dtypes.u64 [111]
113 Ops.REFPARAM dtypes.f32 [1, 112] ParamArg(slot=2, dtype=dtypes.f32)
114 Ops.SHRINK dtypes.f32 [113, 68, 24]
115 Ops.LOAD dtypes.f32 [114]
116 Ops.LOAD dtypes.f32 [105]
117 Ops.INDEX dtypes.f32 [104, 21]
118 Ops.LOAD dtypes.f32 [117]
119 Ops.STACK dtypes.f32 [116, 118]
120 Ops.WMMA dtypes.f32 [110, 115, 119] ((8, 8, 8), dtypes.f32, 32, None)
121 Ops.INDEX dtypes.f32 [120, 48]
122 Ops.STORE dtypes.void [105, 121]
123 Ops.INDEX dtypes.f32 [120, 21]
124 Ops.STORE dtypes.void [117, 123]
125 Ops.AFTER dtypes.void [0, 122, 124]
126 Ops.END dtypes.void [125, 55]
127 Ops.AFTER dtypes.f32 [98, 126]
128 Ops.INDEX dtypes.f32 [127, 48]
129 Ops.LOAD dtypes.f32 [128]
130 Ops.INDEX dtypes.f32 [127, 21]
131 Ops.LOAD dtypes.f32 [130]
132 Ops.STACK dtypes.f32 [129, 131]
133 Ops.STORE dtypes.void [97, 132]
134 Ops.AFTER dtypes.void [0, 133]
135 Ops.SINK dtypes.void [134] KernelInfo(name="r_32_8_4_32_2_32", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL))), estimates=Estimates(ops=33554432, lds=17039360, mem=786432), optimized=True)
136 Ops.LINEAR dtypes.void [91, 92, 23, 98, 20, 21, 93, 106, 107, 24, 111, 112, 3, 4, 5, 16, 17, 38, 18, 9, 10, 11, 1, 108, 113, 99, 0, 47, 48, 100, 102, 50, 51, 31, 32, 39, 40, 12, 13, 6, 7, 8, 14, 15, 19, 22, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 41, 42, 101, 103, 55, 104, 105, 116, 117, 118, 59, 60, 61, 62, 109, 110, 66, 67, 68, 114, 115, 119, 120, 121, 123, 122, 124, 125, 126, 94, 95, 127, 128, 129, 130, 131, 96, 132, 43, 97, 133, 134, 135]
137 Ops.SOURCE dtypes.void [] "#include <metal_stdlib>\nusing namespace metal;\nfloat2 __WMMA_8_8_8_f32_f32(float2 a, float2 b, float2 c){\n  simdgroup_float8x8 mat_a, mat_b; simdgroup_float8x8 mat_c;\n  mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];\n  mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];\n  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);\n  return float2(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);\n}\nkernel void r_32_8_4_32_2_32(device unsigned char* args_24, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n  float buf0[2];\n  unsigned long val0 = (*(((device unsigned long*)(args_24))+1));\n  unsigned long val1 = (*(((device unsigned long*)(args_24))+2));\n  int gidx0 = gid.x; /* 8 */\n  int gidx1 = gid.y; /* 32 */\n  int lidx0 = lid.x; /* 32 */\n  int lidx1 = lid.y; /* 4 */\n  int alu0 = (((lidx0>>3)&1)<<2);\n  int alu1 = ((lidx0&1)<<1);\n  int alu2 = ((gidx0<<5)+(lidx1<<3)+alu0+alu1);\n  int alu3 = (((lidx0>>4)<<10)+(((lidx0>>1)&3)<<8));\n  int alu4 = (alu3+(gidx1<<11));\n  *(buf0+0) = 0.0f;\n  *(buf0+1) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 32; Lidx0++) {\n    float2 val2 = (*((device float2*)((((device float*)(val0))+(alu4+(alu0+(Lidx0<<3)+alu1))))));\n    float2 val3 = (*((device float2*)((((device float*)(val1))+(alu2+(alu3+(Lidx0<<11)))))));\n    float2 wmma0 = __WMMA_8_8_8_f32_f32(val2, val3, float2((*(buf0+0)),(*(buf0+1))));\n    *(buf0+0) = wmma0.x;\n    *(buf0+1) = wmma0.y;\n  }\n  unsigned long val4 = (*(((device unsigned long*)(args_24))+0));\n  *((device float2*)((((device float*)(val4))+(alu2+alu4)))) = float2((*(buf0+0)),(*(buf0+1)));\n}"
138 Ops.PROGRAM dtypes.void [135, 136, 137] ProgramInfo(global_size=(8, 32, 1), local_size=(32, 4, 1), params=(%2, %58, %65), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="METAL", renderer="METAL", arch="Apple9"))
139 Ops.SINK dtypes.void [90, 138]

