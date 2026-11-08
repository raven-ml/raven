# tolk D151 golden=d0cb39da2a5ebb06d66afdebc1fea7b5
0 Ops.CONST dtypes.weakint [] 2
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.CAST dtypes.i32 [0] dtypes.i32
5 Ops.SHRINK dtypes.f32 [1, 3, 4]
6 Ops.CONST dtypes.weakint [] 1
7 Ops.PARAM dtypes.f32 [6] ParamArg(slot=1, dtype=dtypes.f32)
8 Ops.INDEX dtypes.f32 [7, 3]
9 Ops.LOAD dtypes.f32 [8]
10 Ops.PARAM dtypes.f32 [6] ParamArg(slot=2, dtype=dtypes.f32)
11 Ops.INDEX dtypes.f32 [10, 3]
12 Ops.LOAD dtypes.f32 [11]
13 Ops.STACK dtypes.f32 [9, 12]
14 Ops.STORE dtypes.void [5, 13]
15 Ops.SINK dtypes.void [14, 14] KernelInfo(name="E_2", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)),), optimized=True)
16 Ops.CONST dtypes.weakint [] 24
17 Ops.PARAM dtypes.u8 [16] ParamArg(slot=0, dtype=dtypes.u8, name="args")
18 Ops.BITCAST dtypes.u64 [17] dtypes.u64
19 Ops.INDEX dtypes.u64 [18, 3]
20 Ops.LOAD dtypes.u64 [19]
21 Ops.REFPARAM dtypes.f32 [0, 20] ParamArg(slot=0, dtype=dtypes.f32)
22 Ops.SHRINK dtypes.f32 [21, 3, 4]
23 Ops.CAST dtypes.i32 [6] dtypes.i32
24 Ops.INDEX dtypes.u64 [18, 23]
25 Ops.LOAD dtypes.u64 [24]
26 Ops.REFPARAM dtypes.f32 [6, 25] ParamArg(slot=1, dtype=dtypes.f32)
27 Ops.INDEX dtypes.f32 [26, 3]
28 Ops.LOAD dtypes.f32 [27]
29 Ops.INDEX dtypes.u64 [18, 4]
30 Ops.LOAD dtypes.u64 [29]
31 Ops.REFPARAM dtypes.f32 [6, 30] ParamArg(slot=2, dtype=dtypes.f32)
32 Ops.INDEX dtypes.f32 [31, 3]
33 Ops.LOAD dtypes.f32 [32]
34 Ops.STACK dtypes.f32 [28, 33]
35 Ops.STORE dtypes.void [22, 34]
36 Ops.SINK dtypes.void [35, 35] KernelInfo(name="E_2", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)),), estimates=Estimates(lds=16, mem=16), optimized=True)
37 Ops.LINEAR dtypes.void [16, 17, 2, 3, 18, 19, 20, 6, 23, 24, 25, 0, 4, 29, 30, 26, 27, 28, 31, 32, 33, 21, 34, 22, 35, 36]
38 Ops.SOURCE dtypes.void [] "typedef float float2 __attribute__((aligned(4),ext_vector_type(2)));\n__attribute__((min_vector_width(512))) void E_2(unsigned char* restrict args_24) {\n  unsigned long val0 = (*(((unsigned long*)(args_24))+0));\n  unsigned long val1 = (*(((unsigned long*)(args_24))+1));\n  unsigned long val2 = (*(((unsigned long*)(args_24))+2));\n  float val3 = (*(((float*)(val1))+0));\n  float val4 = (*(((float*)(val2))+0));\n  *((float2*)((((float*)(val0))+0))) = (float2){val3,val4};\n}\n"
39 Ops.PROGRAM dtypes.void [36, 37, 38] ProgramInfo(params=(%1, %7, %10), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
40 Ops.SINK dtypes.void [15, 39]

