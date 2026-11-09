# tolk D141 D151 golden=f0ea96d3b96f54ce4f890b85121f8e8f
0 Ops.CONST dtypes.weakint [] 3
1 Ops.CONST dtypes.weakint [] 24
2 Ops.PARAM dtypes.u8 [1] ParamArg(slot=0, dtype=dtypes.u8, name="args")
3 Ops.BITCAST dtypes.u64 [2] dtypes.u64
4 Ops.CONST dtypes.weakint [] 0
5 Ops.CAST dtypes.i32 [4] dtypes.i32
6 Ops.INDEX dtypes.u64 [3, 5]
7 Ops.LOAD dtypes.u64 [6]
8 Ops.REFPARAM dtypes.f32 [0, 7] ParamArg(slot=0, dtype=dtypes.f32)
9 Ops.CONST dtypes.weakint [] 2
10 Ops.CAST dtypes.i32 [9] dtypes.i32
11 Ops.SHRINK dtypes.f32 [8, 5, 10]
12 Ops.CONST dtypes.weakint [] 1
13 Ops.CAST dtypes.i32 [12] dtypes.i32
14 Ops.INDEX dtypes.u64 [3, 13]
15 Ops.LOAD dtypes.u64 [14]
16 Ops.REFPARAM dtypes.f32 [0, 15] ParamArg(slot=1, dtype=dtypes.f32)
17 Ops.INDEX dtypes.f32 [16, 13]
18 Ops.LOAD dtypes.f32 [17]
19 Ops.INDEX dtypes.f32 [16, 10]
20 Ops.LOAD dtypes.f32 [19]
21 Ops.INDEX dtypes.u64 [3, 10]
22 Ops.LOAD dtypes.u64 [21]
23 Ops.REFPARAM dtypes.f32 [12, 22] ParamArg(slot=2, dtype=dtypes.f32)
24 Ops.INDEX dtypes.f32 [23, 5]
25 Ops.LOAD dtypes.f32 [24]
26 Ops.STACK dtypes.f32 [18, 20, 25]
27 Ops.INDEX dtypes.f32 [26, 5]
28 Ops.INDEX dtypes.f32 [26, 13]
29 Ops.STACK dtypes.f32 [27, 28]
30 Ops.STORE dtypes.void [11, 29]
31 Ops.INDEX dtypes.f32 [8, 10]
32 Ops.INDEX dtypes.f32 [26, 10]
33 Ops.STORE dtypes.void [31, 32]
34 Ops.SINK dtypes.void [30, 30, 33] KernelInfo(name="E_3", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)),), estimates=Estimates(lds=24, mem=24), optimized=True)
35 Ops.LINEAR dtypes.void [1, 2, 4, 5, 3, 6, 7, 12, 13, 14, 15, 9, 10, 21, 22, 0, 16, 17, 18, 19, 20, 23, 24, 25, 8, 26, 27, 28, 29, 31, 32, 11, 33, 30, 34]
36 Ops.SOURCE dtypes.void [] "typedef float float3 __attribute__((aligned(4),ext_vector_type(3)));\ntypedef float float2 __attribute__((aligned(4),ext_vector_type(2)));\n__attribute__((min_vector_width(512))) void E_3(unsigned char* restrict args_24) {\n  unsigned long val0 = (*(((unsigned long*)(args_24))+0));\n  unsigned long val1 = (*(((unsigned long*)(args_24))+1));\n  unsigned long val2 = (*(((unsigned long*)(args_24))+2));\n  float val3 = (*(((float*)(val1))+1));\n  float val4 = (*(((float*)(val1))+2));\n  float val5 = (*(((float*)(val2))+0));\n  float3 cast0 = (float3){val3,val4,val5};\n  *(((float*)(val0))+2) = cast0[2];\n  *((float2*)((((float*)(val0))+0))) = (float2){cast0[0],cast0[1]};\n}\n"
37 Ops.BINARY dtypes.u8 [] b"typedef float float3 __attribute__((aligned(4),ext_vector_type(3)));\ntypedef float float2 __attribute__((aligned(4),ext_vector_type(2)));\n__attribute__((min_vector_width(512))) void E_3(unsigned char* restrict args_24) {\n  unsigned long val0 = (*(((unsigned long*)(args_24))+0));\n  unsigned long val1 = (*(((unsigned long*)(args_24))+1));\n  unsigned long val2 = (*(((unsigned long*)(args_24))+2));\n  float val3 = (*(((float*)(val1))+1));\n  float val4 = (*(((float*)(val1))+2));\n  float val5 = (*(((float*)(val2))+0));\n  float3 cast0 = (float3){val3,val4,val5};\n  *(((float*)(val0))+2) = cast0[2];\n  *((float2*)((((float*)(val0))+0))) = (float2){cast0[0],cast0[1]};\n}\n"
38 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
39 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
40 Ops.PARAM dtypes.f32 [12] ParamArg(slot=2, dtype=dtypes.f32)
41 Ops.PROGRAM dtypes.void [34, 35, 36, 37] ProgramInfo(params=(%38, %39, %40), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CPU", arch="x86_64,x86-64"))
42 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
43 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
44 Ops.PARAM dtypes.f32 [12] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
45 Ops.CALL dtypes.void [41, 42, 43, 44] CallInfo()
46 Ops.CONST dtypes.weakint [] 16
47 Ops.PARAM dtypes.u8 [46] ParamArg(slot=0, dtype=dtypes.u8, name="args")
48 Ops.BITCAST dtypes.u64 [47] dtypes.u64
49 Ops.INDEX dtypes.u64 [48, 5]
50 Ops.LOAD dtypes.u64 [49]
51 Ops.REFPARAM dtypes.f32 [12, 50] ParamArg(slot=0, dtype=dtypes.f32)
52 Ops.INDEX dtypes.f32 [51, 5]
53 Ops.INDEX dtypes.u64 [48, 13]
54 Ops.LOAD dtypes.u64 [53]
55 Ops.REFPARAM dtypes.f32 [0, 54] ParamArg(slot=1, dtype=dtypes.f32)
56 Ops.INDEX dtypes.f32 [55, 13]
57 Ops.LOAD dtypes.f32 [56]
58 Ops.STORE dtypes.void [52, 57]
59 Ops.SINK dtypes.void [58] KernelInfo(name="E", estimates=Estimates(lds=8, mem=8), optimized=True)
60 Ops.LINEAR dtypes.void [46, 47, 4, 5, 48, 49, 50, 12, 13, 53, 54, 0, 55, 56, 57, 51, 52, 58, 59]
61 Ops.SOURCE dtypes.void [] "\n__attribute__((min_vector_width(512))) void E(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+0));\n  unsigned long val1 = (*(((unsigned long*)(args_16))+1));\n  float val2 = (*(((float*)(val1))+1));\n  *(((float*)(val0))+0) = val2;\n}\n"
62 Ops.BINARY dtypes.u8 [] b"\n__attribute__((min_vector_width(512))) void E(unsigned char* restrict args_16) {\n  unsigned long val0 = (*(((unsigned long*)(args_16))+0));\n  unsigned long val1 = (*(((unsigned long*)(args_16))+1));\n  float val2 = (*(((float*)(val1))+1));\n  *(((float*)(val0))+0) = val2;\n}\n"
63 Ops.PARAM dtypes.f32 [12] ParamArg(slot=0, dtype=dtypes.f32)
64 Ops.PROGRAM dtypes.void [59, 60, 61, 62] ProgramInfo(params=(%63, %39), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
65 Ops.BUFFER dtypes.f32 [12] ParamArg(slot=5, dtype=dtypes.f32, device="CPU")
66 Ops.CALL dtypes.void [64, 65, 43] CallInfo()
67 Ops.LINEAR dtypes.void [45, 66]

