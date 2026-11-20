# tolk D25 D141 D151 golden=36f33d04252a80e28bc2e43b2e64bfd0
0 Ops.CONST dtypes.weakint [] 4
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.CAST dtypes.i32 [0] dtypes.i32
5 Ops.SHRINK dtypes.f32 [1, 3, 4]
6 Ops.CONST dtypes.weakfloat [] 0.0
7 Ops.CAST dtypes.f32 [6] dtypes.f32
8 Ops.CONST dtypes.weakint [] 3
9 Ops.PARAM dtypes.f32 [8] ParamArg(slot=1, dtype=dtypes.f32)
10 Ops.CONST dtypes.weakint [] 2
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.SHRINK dtypes.f32 [9, 3, 11]
13 Ops.LOAD dtypes.f32 [12]
14 Ops.INDEX dtypes.f32 [13, 3]
15 Ops.CONST dtypes.weakint [] 1
16 Ops.CAST dtypes.i32 [15] dtypes.i32
17 Ops.INDEX dtypes.f32 [13, 16]
18 Ops.INDEX dtypes.f32 [9, 11]
19 Ops.LOAD dtypes.f32 [18]
20 Ops.STACK dtypes.f32 [14, 17, 19, 7]
21 Ops.PARAM dtypes.f32 [10] ParamArg(slot=2, dtype=dtypes.f32)
22 Ops.SHRINK dtypes.f32 [21, 3, 11]
23 Ops.LOAD dtypes.f32 [22]
24 Ops.INDEX dtypes.f32 [23, 16]
25 Ops.STACK dtypes.f32 [24, 24, 24, 24]
26 Ops.STACK dtypes.f32 [7, 14, 17, 19]
27 Ops.INDEX dtypes.f32 [23, 3]
28 Ops.STACK dtypes.f32 [27, 27, 27, 27]
29 Ops.MUL dtypes.f32 [26, 28]
30 Ops.MULACC dtypes.f32 [20, 25, 29]
31 Ops.CMPLT dtypes.bool [7, 30]
32 Ops.WHERE dtypes.f32 [31, 30, 7]
33 Ops.STORE dtypes.void [5, 32]
34 Ops.SINK dtypes.void [33, 33, 33, 33] KernelInfo(name="r_4_2", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST))), optimized=True)
35 Ops.CONST dtypes.weakint [] 24
36 Ops.PARAM dtypes.u8 [35] ParamArg(slot=0, dtype=dtypes.u8, name="args")
37 Ops.BITCAST dtypes.u64 [36] dtypes.u64
38 Ops.INDEX dtypes.u64 [37, 3]
39 Ops.LOAD dtypes.u64 [38]
40 Ops.REFPARAM dtypes.f32 [0, 39] ParamArg(slot=0, dtype=dtypes.f32)
41 Ops.SHRINK dtypes.f32 [40, 3, 4]
42 Ops.INDEX dtypes.u64 [37, 16]
43 Ops.LOAD dtypes.u64 [42]
44 Ops.REFPARAM dtypes.f32 [8, 43] ParamArg(slot=1, dtype=dtypes.f32)
45 Ops.SHRINK dtypes.f32 [44, 3, 11]
46 Ops.LOAD dtypes.f32 [45]
47 Ops.INDEX dtypes.f32 [46, 3]
48 Ops.INDEX dtypes.f32 [46, 16]
49 Ops.INDEX dtypes.f32 [44, 11]
50 Ops.LOAD dtypes.f32 [49]
51 Ops.STACK dtypes.f32 [47, 48, 50, 7]
52 Ops.INDEX dtypes.u64 [37, 11]
53 Ops.LOAD dtypes.u64 [52]
54 Ops.REFPARAM dtypes.f32 [10, 53] ParamArg(slot=2, dtype=dtypes.f32)
55 Ops.SHRINK dtypes.f32 [54, 3, 11]
56 Ops.LOAD dtypes.f32 [55]
57 Ops.INDEX dtypes.f32 [56, 16]
58 Ops.STACK dtypes.f32 [57, 57, 57, 57]
59 Ops.STACK dtypes.f32 [7, 47, 48, 50]
60 Ops.INDEX dtypes.f32 [56, 3]
61 Ops.STACK dtypes.f32 [60, 60, 60, 60]
62 Ops.MUL dtypes.f32 [59, 61]
63 Ops.MULACC dtypes.f32 [51, 58, 62]
64 Ops.CMPLT dtypes.bool [7, 63]
65 Ops.WHERE dtypes.f32 [64, 63, 7]
66 Ops.STORE dtypes.void [41, 65]
67 Ops.SINK dtypes.void [66, 66, 66, 66] KernelInfo(name="r_4_2", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST))), estimates=Estimates(ops=20, lds=36, mem=36), optimized=True)
68 Ops.LINEAR dtypes.void [35, 36, 2, 3, 37, 38, 39, 15, 16, 42, 43, 10, 11, 52, 53, 8, 44, 49, 50, 45, 46, 54, 55, 56, 0, 40, 47, 48, 6, 7, 51, 60, 61, 57, 58, 59, 4, 41, 62, 63, 64, 65, 66, 67]
69 Ops.SOURCE dtypes.void [] "typedef float float2 __attribute__((aligned(4),ext_vector_type(2)));\ntypedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef int int4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_4_2(unsigned char* restrict args_24) {\n  unsigned long val0 = (*(((unsigned long*)(args_24))+0));\n  unsigned long val1 = (*(((unsigned long*)(args_24))+1));\n  unsigned long val2 = (*(((unsigned long*)(args_24))+2));\n  float val3 = (*(((float*)(val1))+2));\n  float2 val4 = (*((float2*)((((float*)(val1))+0))));\n  float2 val5 = (*((float2*)((((float*)(val2))+0))));\n  float4 alu0 = __builtin_elementwise_fma((float4){val4[0],val4[1],val3,0.0f}, ((float4)__builtin_shufflevector(val5, val5, 1, 1, 1, 1)), ((float4){0.0f,val4[0],val4[1],val3}*((float4)__builtin_shufflevector(val5, val5, 0, 0, 0, 0))));\n  float4 alu1 = ((0.0f<alu0)?alu0:0.0f);\n  *((float4*)((((float*)(val0))+0))) = alu1;\n}\n"
70 Ops.PROGRAM dtypes.void [67, 68, 69] ProgramInfo(params=(%1, %9, %21), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
71 Ops.SINK dtypes.void [34, 70]

