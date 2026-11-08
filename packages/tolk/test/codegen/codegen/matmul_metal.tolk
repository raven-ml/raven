# tolk D89 golden=f16a6c070f6a73896c880573f9bb6cbc
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 256
2 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.CONST dtypes.weakint [] 32
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.SPECIAL dtypes.i32 [4] "lidx0"
6 Ops.CONST dtypes.weakint [] 4
7 Ops.CAST dtypes.i32 [6] dtypes.i32
8 Ops.SHR dtypes.i32 [5, 7]
9 Ops.CONST dtypes.weakint [] 6
10 Ops.CAST dtypes.i32 [9] dtypes.i32
11 Ops.SHL dtypes.i32 [8, 10]
12 Ops.CONST dtypes.weakint [] 1
13 Ops.CAST dtypes.i32 [12] dtypes.i32
14 Ops.SHR dtypes.i32 [5, 13]
15 Ops.CONST dtypes.weakint [] 3
16 Ops.CAST dtypes.i32 [15] dtypes.i32
17 Ops.AND dtypes.i32 [14, 16]
18 Ops.SHL dtypes.i32 [17, 7]
19 Ops.ADD dtypes.i32 [11, 18]
20 Ops.CONST dtypes.weakint [] 2
21 Ops.CAST dtypes.i32 [20] dtypes.i32
22 Ops.SPECIAL dtypes.i32 [21] "gidx1"
23 Ops.CONST dtypes.weakint [] 7
24 Ops.CAST dtypes.i32 [23] dtypes.i32
25 Ops.SHL dtypes.i32 [22, 24]
26 Ops.ADD dtypes.i32 [19, 25]
27 Ops.SHR dtypes.i32 [5, 16]
28 Ops.AND dtypes.i32 [27, 13]
29 Ops.SHL dtypes.i32 [28, 21]
30 Ops.SPECIAL dtypes.i32 [21] "gidx0"
31 Ops.SHL dtypes.i32 [30, 16]
32 Ops.ADD dtypes.i32 [29, 31]
33 Ops.AND dtypes.i32 [5, 13]
34 Ops.SHL dtypes.i32 [33, 13]
35 Ops.ADD dtypes.i32 [32, 34]
36 Ops.ADD dtypes.i32 [26, 35]
37 Ops.SHRINK dtypes.f32 [2, 36, 21]
38 Ops.ALLOC dtypes.f32 [20] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
39 Ops.AFTER dtypes.f32 [38, 5, 22, 30]
40 Ops.CONST dtypes.weakint [] 0
41 Ops.CAST dtypes.i32 [40] dtypes.i32
42 Ops.INDEX dtypes.f32 [39, 41]
43 Ops.CONST dtypes.weakfloat [] 0.0
44 Ops.CAST dtypes.f32 [43] dtypes.f32
45 Ops.STORE dtypes.void [42, 44]
46 Ops.INDEX dtypes.f32 [39, 13]
47 Ops.STORE dtypes.void [46, 44]
48 Ops.RANGE dtypes.i32 [21] (AxisType.WEAK, 0)
49 Ops.AFTER dtypes.f32 [38, 45, 47, 48]
50 Ops.INDEX dtypes.f32 [49, 41]
51 Ops.PARAM dtypes.f32 [1] ParamArg(slot=1, dtype=dtypes.f32)
52 Ops.SHL dtypes.i32 [48, 16]
53 Ops.ADD dtypes.i32 [29, 52]
54 Ops.ADD dtypes.i32 [53, 34]
55 Ops.ADD dtypes.i32 [26, 54]
56 Ops.SHRINK dtypes.f32 [51, 55, 21]
57 Ops.LOAD dtypes.f32 [56]
58 Ops.PARAM dtypes.f32 [1] ParamArg(slot=2, dtype=dtypes.f32)
59 Ops.SHL dtypes.i32 [48, 24]
60 Ops.ADD dtypes.i32 [19, 59]
61 Ops.ADD dtypes.i32 [60, 35]
62 Ops.SHRINK dtypes.f32 [58, 61, 21]
63 Ops.LOAD dtypes.f32 [62]
64 Ops.LOAD dtypes.f32 [50]
65 Ops.INDEX dtypes.f32 [49, 13]
66 Ops.LOAD dtypes.f32 [65]
67 Ops.STACK dtypes.f32 [64, 66]
68 Ops.WMMA dtypes.f32 [57, 63, 67] ((8, 8, 8), dtypes.f32, 32, None)
69 Ops.INDEX dtypes.f32 [68, 41]
70 Ops.STORE dtypes.void [50, 69]
71 Ops.INDEX dtypes.f32 [68, 13]
72 Ops.STORE dtypes.void [65, 71]
73 Ops.AFTER dtypes.void [0, 70, 72]
74 Ops.END dtypes.void [73, 48]
75 Ops.AFTER dtypes.f32 [38, 74]
76 Ops.INDEX dtypes.f32 [75, 41]
77 Ops.LOAD dtypes.f32 [76]
78 Ops.INDEX dtypes.f32 [75, 13]
79 Ops.LOAD dtypes.f32 [78]
80 Ops.STACK dtypes.f32 [77, 79]
81 Ops.STORE dtypes.void [37, 80]
82 Ops.AFTER dtypes.void [0, 81]
83 Ops.SINK dtypes.void [82] KernelInfo(name="r_2_2_32_2_2", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), optimized=True)
84 Ops.CONST dtypes.weakint [] 24
85 Ops.PARAM dtypes.u8 [84] ParamArg(slot=0, dtype=dtypes.u8, name="args")
86 Ops.BITCAST dtypes.u64 [85] dtypes.u64
87 Ops.INDEX dtypes.u64 [86, 41]
88 Ops.LOAD dtypes.u64 [87]
89 Ops.REFPARAM dtypes.f32 [1, 88] ParamArg(slot=0, dtype=dtypes.f32)
90 Ops.SHRINK dtypes.f32 [89, 36, 21]
91 Ops.BUFFER dtypes.f32 [20] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
92 Ops.AFTER dtypes.f32 [91, 5, 22, 30]
93 Ops.INDEX dtypes.f32 [92, 41]
94 Ops.STORE dtypes.void [93, 44]
95 Ops.INDEX dtypes.f32 [92, 13]
96 Ops.STORE dtypes.void [95, 44]
97 Ops.AFTER dtypes.f32 [91, 94, 96, 48]
98 Ops.INDEX dtypes.f32 [97, 41]
99 Ops.INDEX dtypes.u64 [86, 13]
100 Ops.LOAD dtypes.u64 [99]
101 Ops.REFPARAM dtypes.f32 [1, 100] ParamArg(slot=1, dtype=dtypes.f32)
102 Ops.SHRINK dtypes.f32 [101, 55, 21]
103 Ops.LOAD dtypes.f32 [102]
104 Ops.INDEX dtypes.u64 [86, 21]
105 Ops.LOAD dtypes.u64 [104]
106 Ops.REFPARAM dtypes.f32 [1, 105] ParamArg(slot=2, dtype=dtypes.f32)
107 Ops.SHRINK dtypes.f32 [106, 61, 21]
108 Ops.LOAD dtypes.f32 [107]
109 Ops.LOAD dtypes.f32 [98]
110 Ops.INDEX dtypes.f32 [97, 13]
111 Ops.LOAD dtypes.f32 [110]
112 Ops.STACK dtypes.f32 [109, 111]
113 Ops.WMMA dtypes.f32 [103, 108, 112] ((8, 8, 8), dtypes.f32, 32, None)
114 Ops.INDEX dtypes.f32 [113, 41]
115 Ops.STORE dtypes.void [98, 114]
116 Ops.INDEX dtypes.f32 [113, 13]
117 Ops.STORE dtypes.void [110, 116]
118 Ops.AFTER dtypes.void [0, 115, 117]
119 Ops.END dtypes.void [118, 48]
120 Ops.AFTER dtypes.f32 [91, 119]
121 Ops.INDEX dtypes.f32 [120, 41]
122 Ops.LOAD dtypes.f32 [121]
123 Ops.INDEX dtypes.f32 [120, 13]
124 Ops.LOAD dtypes.f32 [123]
125 Ops.STACK dtypes.f32 [122, 124]
126 Ops.STORE dtypes.void [90, 125]
127 Ops.AFTER dtypes.void [0, 126]
128 Ops.SINK dtypes.void [127] KernelInfo(name="r_2_2_32_2_2", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), estimates=Estimates(ops=8192, lds=5120, mem=3072), optimized=True)
129 Ops.LINEAR dtypes.void [84, 85, 20, 91, 12, 13, 86, 99, 100, 21, 104, 105, 30, 22, 3, 4, 5, 1, 101, 106, 92, 0, 40, 41, 93, 95, 43, 44, 15, 16, 6, 7, 9, 10, 23, 24, 8, 11, 14, 17, 18, 19, 25, 26, 31, 27, 28, 29, 32, 33, 34, 35, 94, 96, 48, 97, 98, 109, 110, 111, 52, 53, 54, 55, 102, 103, 59, 60, 61, 107, 108, 112, 113, 114, 116, 115, 117, 118, 119, 87, 88, 120, 121, 122, 123, 124, 89, 125, 36, 90, 126, 127, 128]
130 Ops.SOURCE dtypes.void [] "#include <metal_stdlib>\nusing namespace metal;\nfloat2 __WMMA_8_8_8_f32_f32(float2 a, float2 b, float2 c){\n  simdgroup_float8x8 mat_a, mat_b; simdgroup_float8x8 mat_c;\n  mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];\n  mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];\n  simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);\n  return float2(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);\n}\nkernel void r_2_2_32_2_2(device unsigned char* args_24, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n  float buf0[2];\n  unsigned long val0 = (*(((device unsigned long*)(args_24))+1));\n  unsigned long val1 = (*(((device unsigned long*)(args_24))+2));\n  int gidx0 = gid.x; /* 2 */\n  int gidx1 = gid.y; /* 2 */\n  int lidx0 = lid.x; /* 32 */\n  int alu0 = (((lidx0>>4)<<6)+(((lidx0>>1)&3)<<4));\n  int alu1 = (alu0+(gidx1<<7));\n  int alu2 = (((lidx0>>3)&1)<<2);\n  int alu3 = ((lidx0&1)<<1);\n  int alu4 = (alu2+(gidx0<<3)+alu3);\n  *(buf0+0) = 0.0f;\n  *(buf0+1) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 2; Lidx0++) {\n    float2 val2 = (*((device float2*)((((device float*)(val0))+(alu1+(alu2+(Lidx0<<3)+alu3))))));\n    float2 val3 = (*((device float2*)((((device float*)(val1))+(alu0+(Lidx0<<7)+alu4)))));\n    float2 wmma0 = __WMMA_8_8_8_f32_f32(val2, val3, float2((*(buf0+0)),(*(buf0+1))));\n    *(buf0+0) = wmma0.x;\n    *(buf0+1) = wmma0.y;\n  }\n  unsigned long val4 = (*(((device unsigned long*)(args_24))+0));\n  *((device float2*)((((device float*)(val4))+(alu1+alu4)))) = float2((*(buf0+0)),(*(buf0+1)));\n}"
131 Ops.PROGRAM dtypes.void [128, 129, 130] ProgramInfo(global_size=(2, 2, 1), local_size=(32, 1, 1), params=(%2, %51, %58), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="METAL", renderer="METAL", arch="Apple9"))
132 Ops.SINK dtypes.void [83, 131]

