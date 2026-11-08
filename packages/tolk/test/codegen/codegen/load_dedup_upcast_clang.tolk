# tolk D151 golden=cb1e622a537bda08325f379bf05f896a
0 Ops.CONST dtypes.weakint [] 3
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.CONST dtypes.weakint [] 2
5 Ops.CAST dtypes.i32 [4] dtypes.i32
6 Ops.SHRINK dtypes.f32 [1, 3, 5]
7 Ops.CONST dtypes.weakint [] 4
8 Ops.PARAM dtypes.f32 [7] ParamArg(slot=1, dtype=dtypes.f32)
9 Ops.CAST dtypes.i32 [7] dtypes.i32
10 Ops.SHRINK dtypes.f32 [8, 3, 9]
11 Ops.LOAD dtypes.f32 [10]
12 Ops.INDEX dtypes.f32 [11, 3]
13 Ops.CONST dtypes.weakint [] 1
14 Ops.CAST dtypes.i32 [13] dtypes.i32
15 Ops.INDEX dtypes.f32 [11, 14]
16 Ops.INDEX dtypes.f32 [11, 5]
17 Ops.STACK dtypes.f32 [12, 15, 16]
18 Ops.CAST dtypes.i32 [0] dtypes.i32
19 Ops.INDEX dtypes.f32 [11, 18]
20 Ops.STACK dtypes.f32 [15, 16, 19]
21 Ops.ADD dtypes.f32 [17, 20]
22 Ops.INDEX dtypes.f32 [21, 3]
23 Ops.INDEX dtypes.f32 [21, 14]
24 Ops.STACK dtypes.f32 [22, 23]
25 Ops.STORE dtypes.void [6, 24]
26 Ops.INDEX dtypes.f32 [1, 5]
27 Ops.INDEX dtypes.f32 [21, 5]
28 Ops.STORE dtypes.void [26, 27]
29 Ops.SINK dtypes.void [25, 25, 28] KernelInfo(name="E_3", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)),), optimized=True)
30 Ops.CONST dtypes.weakint [] 16
31 Ops.PARAM dtypes.u8 [30] ParamArg(slot=0, dtype=dtypes.u8, name="args")
32 Ops.BITCAST dtypes.u64 [31] dtypes.u64
33 Ops.INDEX dtypes.u64 [32, 3]
34 Ops.LOAD dtypes.u64 [33]
35 Ops.REFPARAM dtypes.f32 [0, 34] ParamArg(slot=0, dtype=dtypes.f32)
36 Ops.SHRINK dtypes.f32 [35, 3, 5]
37 Ops.INDEX dtypes.u64 [32, 14]
38 Ops.LOAD dtypes.u64 [37]
39 Ops.REFPARAM dtypes.f32 [7, 38] ParamArg(slot=1, dtype=dtypes.f32)
40 Ops.SHRINK dtypes.f32 [39, 3, 9]
41 Ops.LOAD dtypes.f32 [40]
42 Ops.INDEX dtypes.f32 [41, 3]
43 Ops.INDEX dtypes.f32 [41, 14]
44 Ops.INDEX dtypes.f32 [41, 5]
45 Ops.STACK dtypes.f32 [42, 43, 44]
46 Ops.INDEX dtypes.f32 [41, 18]
47 Ops.STACK dtypes.f32 [43, 44, 46]
48 Ops.ADD dtypes.f32 [45, 47]
49 Ops.INDEX dtypes.f32 [48, 3]
50 Ops.INDEX dtypes.f32 [48, 14]
51 Ops.STACK dtypes.f32 [49, 50]
52 Ops.STORE dtypes.void [36, 51]
53 Ops.INDEX dtypes.f32 [35, 5]
54 Ops.INDEX dtypes.f32 [48, 5]
55 Ops.STORE dtypes.void [53, 54]
56 Ops.SINK dtypes.void [52, 52, 55] KernelInfo(name="E_3", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)),), estimates=Estimates(ops=3, lds=28, mem=28), optimized=True)
57 Ops.LINEAR dtypes.void [30, 31, 2, 3, 32, 33, 34, 13, 14, 37, 38, 7, 39, 9, 40, 41, 0, 35, 42, 43, 4, 5, 44, 45, 18, 46, 47, 48, 49, 50, 51, 53, 54, 36, 55, 52, 56]
58 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\ntypedef float float3 __attribute__((aligned(4),ext_vector_type(3)));\ntypedef float float2 __attribute__((aligned(4),ext_vector_type(2)));\n__attribute__((min_vector_width(512))) void E_3(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+0));\n  unsigned long val1 = (*(((unsigned long*)(args_16))+1));\n  float4 val2 = (*((float4*)((((float*)(val1))+0))));\n  float3 alu0 = ((float3){val2[0],val2[1],val2[2]}+(float3){val2[1],val2[2],val2[3]});\n  *(((float*)(val0))+2) = alu0[2];\n  *((float2*)((((float*)(val0))+0))) = (float2){alu0[0],alu0[1]};\n}\n"
59 Ops.PROGRAM dtypes.void [56, 57, 58] ProgramInfo(params=(%1, %8), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
60 Ops.SINK dtypes.void [29, 59]

