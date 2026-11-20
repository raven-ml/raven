# tolk D141 D151 golden=0c7ad98f033f471b58646e53e10b3bd1
0 Ops.CONST dtypes.weakint [] 4
1 Ops.CONST dtypes.weakint [] 16
2 Ops.PARAM dtypes.u8 [1] ParamArg(slot=0, dtype=dtypes.u8, name="args")
3 Ops.BITCAST dtypes.u64 [2] dtypes.u64
4 Ops.CONST dtypes.weakint [] 0
5 Ops.CAST dtypes.i32 [4] dtypes.i32
6 Ops.INDEX dtypes.u64 [3, 5]
7 Ops.LOAD dtypes.u64 [6]
8 Ops.REFPARAM dtypes.f32 [0, 7] ParamArg(slot=0, dtype=dtypes.f32)
9 Ops.CAST dtypes.i32 [0] dtypes.i32
10 Ops.SHRINK dtypes.f32 [8, 5, 9]
11 Ops.CONST dtypes.weakint [] 32
12 Ops.CONST dtypes.weakint [] 1
13 Ops.CAST dtypes.i32 [12] dtypes.i32
14 Ops.INDEX dtypes.u64 [3, 13]
15 Ops.LOAD dtypes.u64 [14]
16 Ops.REFPARAM dtypes.f32 [11, 15] ParamArg(slot=1, dtype=dtypes.f32)
17 Ops.SHRINK dtypes.f32 [16, 5, 9]
18 Ops.LOAD dtypes.f32 [17]
19 Ops.INDEX dtypes.f32 [18, 5]
20 Ops.CAST dtypes.i32 [1] dtypes.i32
21 Ops.SHRINK dtypes.f32 [16, 20, 9]
22 Ops.LOAD dtypes.f32 [21]
23 Ops.INDEX dtypes.f32 [22, 5]
24 Ops.INDEX dtypes.f32 [18, 13]
25 Ops.INDEX dtypes.f32 [22, 13]
26 Ops.STACK dtypes.f32 [19, 23, 24, 25]
27 Ops.INDEX dtypes.f32 [26, 5]
28 Ops.CONST dtypes.weakint [] 8
29 Ops.CAST dtypes.i32 [28] dtypes.i32
30 Ops.SHRINK dtypes.f32 [16, 29, 9]
31 Ops.LOAD dtypes.f32 [30]
32 Ops.INDEX dtypes.f32 [31, 5]
33 Ops.CONST dtypes.weakint [] 24
34 Ops.CAST dtypes.i32 [33] dtypes.i32
35 Ops.SHRINK dtypes.f32 [16, 34, 9]
36 Ops.LOAD dtypes.f32 [35]
37 Ops.INDEX dtypes.f32 [36, 5]
38 Ops.INDEX dtypes.f32 [31, 13]
39 Ops.INDEX dtypes.f32 [36, 13]
40 Ops.STACK dtypes.f32 [32, 37, 38, 39]
41 Ops.INDEX dtypes.f32 [40, 5]
42 Ops.INDEX dtypes.f32 [26, 13]
43 Ops.INDEX dtypes.f32 [40, 13]
44 Ops.STACK dtypes.f32 [27, 41, 42, 43]
45 Ops.CONST dtypes.weakint [] 2
46 Ops.CAST dtypes.i32 [45] dtypes.i32
47 Ops.INDEX dtypes.f32 [26, 46]
48 Ops.INDEX dtypes.f32 [40, 46]
49 Ops.CONST dtypes.weakint [] 3
50 Ops.CAST dtypes.i32 [49] dtypes.i32
51 Ops.INDEX dtypes.f32 [26, 50]
52 Ops.INDEX dtypes.f32 [40, 50]
53 Ops.STACK dtypes.f32 [47, 48, 51, 52]
54 Ops.ADD dtypes.f32 [44, 53]
55 Ops.INDEX dtypes.f32 [18, 46]
56 Ops.INDEX dtypes.f32 [22, 46]
57 Ops.INDEX dtypes.f32 [18, 50]
58 Ops.INDEX dtypes.f32 [22, 50]
59 Ops.STACK dtypes.f32 [55, 56, 57, 58]
60 Ops.INDEX dtypes.f32 [59, 5]
61 Ops.INDEX dtypes.f32 [31, 46]
62 Ops.INDEX dtypes.f32 [36, 46]
63 Ops.INDEX dtypes.f32 [31, 50]
64 Ops.INDEX dtypes.f32 [36, 50]
65 Ops.STACK dtypes.f32 [61, 62, 63, 64]
66 Ops.INDEX dtypes.f32 [65, 5]
67 Ops.INDEX dtypes.f32 [59, 13]
68 Ops.INDEX dtypes.f32 [65, 13]
69 Ops.STACK dtypes.f32 [60, 66, 67, 68]
70 Ops.ADD dtypes.f32 [54, 69]
71 Ops.INDEX dtypes.f32 [59, 46]
72 Ops.INDEX dtypes.f32 [65, 46]
73 Ops.INDEX dtypes.f32 [59, 50]
74 Ops.INDEX dtypes.f32 [65, 50]
75 Ops.STACK dtypes.f32 [71, 72, 73, 74]
76 Ops.ADD dtypes.f32 [70, 75]
77 Ops.SHRINK dtypes.f32 [16, 9, 9]
78 Ops.LOAD dtypes.f32 [77]
79 Ops.INDEX dtypes.f32 [78, 5]
80 Ops.CONST dtypes.weakint [] 20
81 Ops.CAST dtypes.i32 [80] dtypes.i32
82 Ops.SHRINK dtypes.f32 [16, 81, 9]
83 Ops.LOAD dtypes.f32 [82]
84 Ops.INDEX dtypes.f32 [83, 5]
85 Ops.INDEX dtypes.f32 [78, 13]
86 Ops.INDEX dtypes.f32 [83, 13]
87 Ops.STACK dtypes.f32 [79, 84, 85, 86]
88 Ops.INDEX dtypes.f32 [87, 5]
89 Ops.CONST dtypes.weakint [] 12
90 Ops.CAST dtypes.i32 [89] dtypes.i32
91 Ops.SHRINK dtypes.f32 [16, 90, 9]
92 Ops.LOAD dtypes.f32 [91]
93 Ops.INDEX dtypes.f32 [92, 5]
94 Ops.CONST dtypes.weakint [] 28
95 Ops.CAST dtypes.i32 [94] dtypes.i32
96 Ops.SHRINK dtypes.f32 [16, 95, 9]
97 Ops.LOAD dtypes.f32 [96]
98 Ops.INDEX dtypes.f32 [97, 5]
99 Ops.INDEX dtypes.f32 [92, 13]
100 Ops.INDEX dtypes.f32 [97, 13]
101 Ops.STACK dtypes.f32 [93, 98, 99, 100]
102 Ops.INDEX dtypes.f32 [101, 5]
103 Ops.INDEX dtypes.f32 [87, 13]
104 Ops.INDEX dtypes.f32 [101, 13]
105 Ops.STACK dtypes.f32 [88, 102, 103, 104]
106 Ops.ADD dtypes.f32 [76, 105]
107 Ops.INDEX dtypes.f32 [87, 46]
108 Ops.INDEX dtypes.f32 [101, 46]
109 Ops.INDEX dtypes.f32 [87, 50]
110 Ops.INDEX dtypes.f32 [101, 50]
111 Ops.STACK dtypes.f32 [107, 108, 109, 110]
112 Ops.ADD dtypes.f32 [106, 111]
113 Ops.INDEX dtypes.f32 [78, 46]
114 Ops.INDEX dtypes.f32 [83, 46]
115 Ops.INDEX dtypes.f32 [78, 50]
116 Ops.INDEX dtypes.f32 [83, 50]
117 Ops.STACK dtypes.f32 [113, 114, 115, 116]
118 Ops.INDEX dtypes.f32 [117, 5]
119 Ops.INDEX dtypes.f32 [92, 46]
120 Ops.INDEX dtypes.f32 [97, 46]
121 Ops.INDEX dtypes.f32 [92, 50]
122 Ops.INDEX dtypes.f32 [97, 50]
123 Ops.STACK dtypes.f32 [119, 120, 121, 122]
124 Ops.INDEX dtypes.f32 [123, 5]
125 Ops.INDEX dtypes.f32 [117, 13]
126 Ops.INDEX dtypes.f32 [123, 13]
127 Ops.STACK dtypes.f32 [118, 124, 125, 126]
128 Ops.ADD dtypes.f32 [112, 127]
129 Ops.INDEX dtypes.f32 [117, 46]
130 Ops.INDEX dtypes.f32 [123, 46]
131 Ops.INDEX dtypes.f32 [117, 50]
132 Ops.INDEX dtypes.f32 [123, 50]
133 Ops.STACK dtypes.f32 [129, 130, 131, 132]
134 Ops.ADD dtypes.f32 [128, 133]
135 Ops.STORE dtypes.void [10, 134]
136 Ops.SINK dtypes.void [135, 135, 135, 135] KernelInfo(name="r_4_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST))), estimates=Estimates(ops=28, lds=144, mem=144), optimized=True)
137 Ops.LINEAR dtypes.void [1, 2, 4, 5, 3, 6, 7, 12, 13, 14, 15, 11, 16, 0, 9, 17, 18, 89, 90, 91, 92, 20, 21, 22, 80, 81, 82, 83, 33, 34, 35, 36, 94, 95, 96, 97, 77, 78, 28, 29, 30, 31, 8, 19, 24, 23, 25, 26, 37, 39, 32, 38, 40, 27, 42, 41, 43, 44, 45, 46, 47, 49, 50, 51, 48, 52, 53, 55, 57, 56, 58, 59, 62, 64, 61, 63, 65, 60, 67, 66, 68, 69, 71, 73, 72, 74, 75, 93, 99, 98, 100, 101, 84, 86, 79, 85, 87, 102, 104, 88, 103, 105, 108, 110, 107, 109, 111, 119, 121, 120, 122, 123, 114, 116, 113, 115, 117, 124, 126, 118, 125, 127, 130, 132, 129, 131, 133, 10, 54, 70, 76, 106, 112, 128, 134, 135, 136]
138 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_4_8(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+0));\n  unsigned long val1 = (*(((unsigned long*)(args_16))+1));\n  float4 val2 = (*((float4*)((((float*)(val1))+0))));\n  float4 val3 = (*((float4*)((((float*)(val1))+12))));\n  float4 val4 = (*((float4*)((((float*)(val1))+16))));\n  float4 val5 = (*((float4*)((((float*)(val1))+20))));\n  float4 val6 = (*((float4*)((((float*)(val1))+24))));\n  float4 val7 = (*((float4*)((((float*)(val1))+28))));\n  float4 val8 = (*((float4*)((((float*)(val1))+4))));\n  float4 val9 = (*((float4*)((((float*)(val1))+8))));\n  float4 cast0 = ((float4)__builtin_shufflevector(val2, val4, 0, 4, 1, 5));\n  float4 cast1 = ((float4)__builtin_shufflevector(val9, val6, 0, 4, 1, 5));\n  float4 cast2 = ((float4)__builtin_shufflevector(val2, val4, 2, 6, 3, 7));\n  float4 cast3 = ((float4)__builtin_shufflevector(val9, val6, 2, 6, 3, 7));\n  float4 cast4 = ((float4)__builtin_shufflevector(val3, val7, 0, 4, 1, 5));\n  float4 cast5 = ((float4)__builtin_shufflevector(val8, val5, 0, 4, 1, 5));\n  float4 cast6 = ((float4)__builtin_shufflevector(val3, val7, 2, 6, 3, 7));\n  float4 cast7 = ((float4)__builtin_shufflevector(val8, val5, 2, 6, 3, 7));\n  *((float4*)((((float*)(val0))+0))) = (((float4)__builtin_shufflevector(cast0, cast1, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast0, cast1, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast2, cast3, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast2, cast3, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast5, cast4, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast5, cast4, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast7, cast6, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast7, cast6, 2, 6, 3, 7)));\n}\n"
139 Ops.BINARY dtypes.u8 [] b"typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_4_8(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+0));\n  unsigned long val1 = (*(((unsigned long*)(args_16))+1));\n  float4 val2 = (*((float4*)((((float*)(val1))+0))));\n  float4 val3 = (*((float4*)((((float*)(val1))+12))));\n  float4 val4 = (*((float4*)((((float*)(val1))+16))));\n  float4 val5 = (*((float4*)((((float*)(val1))+20))));\n  float4 val6 = (*((float4*)((((float*)(val1))+24))));\n  float4 val7 = (*((float4*)((((float*)(val1))+28))));\n  float4 val8 = (*((float4*)((((float*)(val1))+4))));\n  float4 val9 = (*((float4*)((((float*)(val1))+8))));\n  float4 cast0 = ((float4)__builtin_shufflevector(val2, val4, 0, 4, 1, 5));\n  float4 cast1 = ((float4)__builtin_shufflevector(val9, val6, 0, 4, 1, 5));\n  float4 cast2 = ((float4)__builtin_shufflevector(val2, val4, 2, 6, 3, 7));\n  float4 cast3 = ((float4)__builtin_shufflevector(val9, val6, 2, 6, 3, 7));\n  float4 cast4 = ((float4)__builtin_shufflevector(val3, val7, 0, 4, 1, 5));\n  float4 cast5 = ((float4)__builtin_shufflevector(val8, val5, 0, 4, 1, 5));\n  float4 cast6 = ((float4)__builtin_shufflevector(val3, val7, 2, 6, 3, 7));\n  float4 cast7 = ((float4)__builtin_shufflevector(val8, val5, 2, 6, 3, 7));\n  *((float4*)((((float*)(val0))+0))) = (((float4)__builtin_shufflevector(cast0, cast1, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast0, cast1, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast2, cast3, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast2, cast3, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast5, cast4, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast5, cast4, 2, 6, 3, 7))+((float4)__builtin_shufflevector(cast7, cast6, 0, 4, 1, 5))+((float4)__builtin_shufflevector(cast7, cast6, 2, 6, 3, 7)));\n}\n"
140 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
141 Ops.PARAM dtypes.f32 [11] ParamArg(slot=1, dtype=dtypes.f32)
142 Ops.PROGRAM dtypes.void [136, 137, 138, 139] ProgramInfo(params=(%140, %141), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
143 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
144 Ops.BUFFER dtypes.f32 [11] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
145 Ops.CALL dtypes.void [142, 143, 144] CallInfo()
146 Ops.LINEAR dtypes.void [145]

