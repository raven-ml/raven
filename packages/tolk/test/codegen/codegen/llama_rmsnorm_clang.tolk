# tolk D25 D80 D141 D151 golden=6223eda0b369c4cb02d4fc015b368cad
0 Ops.CONST dtypes.weakint [] 2
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.CAST dtypes.i32 [0] dtypes.i32
5 Ops.SHRINK dtypes.f32 [1, 3, 4]
6 Ops.CONST dtypes.weakint [] 16
7 Ops.PARAM dtypes.f32 [6] ParamArg(slot=1, dtype=dtypes.f32)
8 Ops.CONST dtypes.weakint [] 4
9 Ops.CAST dtypes.i32 [8] dtypes.i32
10 Ops.SHRINK dtypes.f32 [7, 9, 9]
11 Ops.LOAD dtypes.f32 [10]
12 Ops.CONST dtypes.weakint [] 3
13 Ops.CAST dtypes.i32 [12] dtypes.i32
14 Ops.INDEX dtypes.f32 [11, 13]
15 Ops.CONST dtypes.weakint [] 12
16 Ops.CAST dtypes.i32 [15] dtypes.i32
17 Ops.SHRINK dtypes.f32 [7, 16, 9]
18 Ops.LOAD dtypes.f32 [17]
19 Ops.INDEX dtypes.f32 [18, 13]
20 Ops.STACK dtypes.f32 [14, 19]
21 Ops.INDEX dtypes.f32 [11, 4]
22 Ops.INDEX dtypes.f32 [18, 4]
23 Ops.STACK dtypes.f32 [21, 22]
24 Ops.CONST dtypes.weakint [] 1
25 Ops.CAST dtypes.i32 [24] dtypes.i32
26 Ops.INDEX dtypes.f32 [11, 25]
27 Ops.INDEX dtypes.f32 [18, 25]
28 Ops.STACK dtypes.f32 [26, 27]
29 Ops.INDEX dtypes.f32 [11, 3]
30 Ops.INDEX dtypes.f32 [18, 3]
31 Ops.STACK dtypes.f32 [29, 30]
32 Ops.SHRINK dtypes.f32 [7, 3, 9]
33 Ops.LOAD dtypes.f32 [32]
34 Ops.INDEX dtypes.f32 [33, 13]
35 Ops.CONST dtypes.weakint [] 8
36 Ops.CAST dtypes.i32 [35] dtypes.i32
37 Ops.SHRINK dtypes.f32 [7, 36, 9]
38 Ops.LOAD dtypes.f32 [37]
39 Ops.INDEX dtypes.f32 [38, 13]
40 Ops.STACK dtypes.f32 [34, 39]
41 Ops.INDEX dtypes.f32 [33, 4]
42 Ops.INDEX dtypes.f32 [38, 4]
43 Ops.STACK dtypes.f32 [41, 42]
44 Ops.INDEX dtypes.f32 [33, 25]
45 Ops.INDEX dtypes.f32 [38, 25]
46 Ops.STACK dtypes.f32 [44, 45]
47 Ops.INDEX dtypes.f32 [33, 3]
48 Ops.INDEX dtypes.f32 [38, 3]
49 Ops.STACK dtypes.f32 [47, 48]
50 Ops.MUL dtypes.f32 [49, 49]
51 Ops.MULACC dtypes.f32 [46, 46, 50]
52 Ops.MULACC dtypes.f32 [43, 43, 51]
53 Ops.MULACC dtypes.f32 [40, 40, 52]
54 Ops.MULACC dtypes.f32 [31, 31, 53]
55 Ops.MULACC dtypes.f32 [28, 28, 54]
56 Ops.MULACC dtypes.f32 [23, 23, 55]
57 Ops.MULACC dtypes.f32 [20, 20, 56]
58 Ops.CONST dtypes.weakfloat [] 0.125
59 Ops.CAST dtypes.f32 [58] dtypes.f32
60 Ops.MUL dtypes.f32 [57, 59]
61 Ops.CONST dtypes.weakfloat [] 9.999999747378752e-06
62 Ops.CAST dtypes.f32 [61] dtypes.f32
63 Ops.ADD dtypes.f32 [60, 62]
64 Ops.SQRT dtypes.f32 [63]
65 Ops.STORE dtypes.void [5, 64]
66 Ops.SINK dtypes.void [65, 65] KernelInfo(name="r_2_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST))), optimized=True)
67 Ops.PARAM dtypes.u8 [6] ParamArg(slot=0, dtype=dtypes.u8, name="args")
68 Ops.BITCAST dtypes.u64 [67] dtypes.u64
69 Ops.INDEX dtypes.u64 [68, 3]
70 Ops.LOAD dtypes.u64 [69]
71 Ops.REFPARAM dtypes.f32 [0, 70] ParamArg(slot=0, dtype=dtypes.f32)
72 Ops.SHRINK dtypes.f32 [71, 3, 4]
73 Ops.INDEX dtypes.u64 [68, 25]
74 Ops.LOAD dtypes.u64 [73]
75 Ops.REFPARAM dtypes.f32 [6, 74] ParamArg(slot=1, dtype=dtypes.f32)
76 Ops.SHRINK dtypes.f32 [75, 9, 9]
77 Ops.LOAD dtypes.f32 [76]
78 Ops.INDEX dtypes.f32 [77, 13]
79 Ops.SHRINK dtypes.f32 [75, 16, 9]
80 Ops.LOAD dtypes.f32 [79]
81 Ops.INDEX dtypes.f32 [80, 13]
82 Ops.STACK dtypes.f32 [78, 81]
83 Ops.INDEX dtypes.f32 [77, 4]
84 Ops.INDEX dtypes.f32 [80, 4]
85 Ops.STACK dtypes.f32 [83, 84]
86 Ops.INDEX dtypes.f32 [77, 25]
87 Ops.INDEX dtypes.f32 [80, 25]
88 Ops.STACK dtypes.f32 [86, 87]
89 Ops.INDEX dtypes.f32 [77, 3]
90 Ops.INDEX dtypes.f32 [80, 3]
91 Ops.STACK dtypes.f32 [89, 90]
92 Ops.SHRINK dtypes.f32 [75, 3, 9]
93 Ops.LOAD dtypes.f32 [92]
94 Ops.INDEX dtypes.f32 [93, 13]
95 Ops.SHRINK dtypes.f32 [75, 36, 9]
96 Ops.LOAD dtypes.f32 [95]
97 Ops.INDEX dtypes.f32 [96, 13]
98 Ops.STACK dtypes.f32 [94, 97]
99 Ops.INDEX dtypes.f32 [93, 4]
100 Ops.INDEX dtypes.f32 [96, 4]
101 Ops.STACK dtypes.f32 [99, 100]
102 Ops.INDEX dtypes.f32 [93, 25]
103 Ops.INDEX dtypes.f32 [96, 25]
104 Ops.STACK dtypes.f32 [102, 103]
105 Ops.INDEX dtypes.f32 [93, 3]
106 Ops.INDEX dtypes.f32 [96, 3]
107 Ops.STACK dtypes.f32 [105, 106]
108 Ops.MUL dtypes.f32 [107, 107]
109 Ops.MULACC dtypes.f32 [104, 104, 108]
110 Ops.MULACC dtypes.f32 [101, 101, 109]
111 Ops.MULACC dtypes.f32 [98, 98, 110]
112 Ops.MULACC dtypes.f32 [91, 91, 111]
113 Ops.MULACC dtypes.f32 [88, 88, 112]
114 Ops.MULACC dtypes.f32 [85, 85, 113]
115 Ops.MULACC dtypes.f32 [82, 82, 114]
116 Ops.MUL dtypes.f32 [115, 59]
117 Ops.ADD dtypes.f32 [116, 62]
118 Ops.SQRT dtypes.f32 [117]
119 Ops.STORE dtypes.void [72, 118]
120 Ops.SINK dtypes.void [119, 119] KernelInfo(name="r_2_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST))), estimates=Estimates(ops=36, lds=72, mem=72), optimized=True)
121 Ops.LINEAR dtypes.void [6, 67, 2, 3, 68, 69, 70, 24, 25, 73, 74, 75, 8, 9, 92, 93, 15, 16, 79, 80, 76, 77, 35, 36, 95, 96, 0, 71, 105, 106, 107, 102, 103, 104, 4, 99, 100, 101, 12, 13, 94, 97, 98, 90, 89, 91, 87, 86, 88, 84, 83, 85, 81, 78, 82, 72, 58, 59, 61, 62, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
122 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef float float2 __attribute__((aligned(4),ext_vector_type(2)));\n__attribute__((min_vector_width(512))) void r_2_8(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+0));\n  unsigned long val1 = (*(((unsigned long*)(args_16))+1));\n  float4 val2 = (*((float4*)((((float*)(val1))+0))));\n  float4 val3 = (*((float4*)((((float*)(val1))+12))));\n  float4 val4 = (*((float4*)((((float*)(val1))+4))));\n  float4 val5 = (*((float4*)((((float*)(val1))+8))));\n  float2 cast0 = ((float2)__builtin_shufflevector(val2, val5, 0, 4));\n  float2 cast1 = ((float2)__builtin_shufflevector(val2, val5, 1, 5));\n  float2 cast2 = ((float2)__builtin_shufflevector(val2, val5, 2, 6));\n  float2 cast3 = ((float2)__builtin_shufflevector(val2, val5, 3, 7));\n  float2 cast4 = ((float2)__builtin_shufflevector(val4, val3, 0, 4));\n  float2 cast5 = ((float2)__builtin_shufflevector(val4, val3, 1, 5));\n  float2 cast6 = ((float2)__builtin_shufflevector(val4, val3, 2, 6));\n  float2 cast7 = ((float2)__builtin_shufflevector(val4, val3, 3, 7));\n  *((float2*)((((float*)(val0))+0))) = __builtin_elementwise_sqrt(((__builtin_elementwise_fma(cast7, cast7, __builtin_elementwise_fma(cast6, cast6, __builtin_elementwise_fma(cast5, cast5, __builtin_elementwise_fma(cast4, cast4, __builtin_elementwise_fma(cast3, cast3, __builtin_elementwise_fma(cast2, cast2, __builtin_elementwise_fma(cast1, cast1, (cast0*cast0))))))))*0.125f)+9.999999747378752e-06f));\n}\n"
123 Ops.PROGRAM dtypes.void [120, 121, 122] ProgramInfo(params=(%1, %7), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
124 Ops.SINK dtypes.void [66, 123]

