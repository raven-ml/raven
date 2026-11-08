# tolk D151 golden=0ba79e95c402828866034843081f79ab
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 256
2 Ops.PARAM dtypes.f32 [1] ParamArg(slot=1, dtype=dtypes.f32)
3 Ops.CAST dtypes.i32 [1] dtypes.i32
4 Ops.RANGE dtypes.i32 [3] (AxisType.LOOP, 0)
5 Ops.INDEX dtypes.f32 [2, 4]
6 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
7 Ops.INDEX dtypes.f32 [6, 4]
8 Ops.LOAD dtypes.f32 [7]
9 Ops.CONST dtypes.weakfloat [] 1.0
10 Ops.CAST dtypes.f32 [9] dtypes.f32
11 Ops.ADD dtypes.f32 [8, 10]
12 Ops.STORE dtypes.void [5, 11]
13 Ops.PARAM dtypes.f32 [1] ParamArg(slot=2, dtype=dtypes.f32)
14 Ops.INDEX dtypes.f32 [13, 4]
15 Ops.CONST dtypes.weakfloat [] 2.0
16 Ops.CAST dtypes.f32 [15] dtypes.f32
17 Ops.MUL dtypes.f32 [8, 16]
18 Ops.STORE dtypes.void [14, 17]
19 Ops.AFTER dtypes.void [0, 12, 18]
20 Ops.END dtypes.void [19, 4]
21 Ops.SINK dtypes.void [20] KernelInfo(name="two_outputs", optimized=True)
22 Ops.CONST dtypes.weakint [] 24
23 Ops.PARAM dtypes.u8 [22] ParamArg(slot=0, dtype=dtypes.u8, name="args")
24 Ops.BITCAST dtypes.u64 [23] dtypes.u64
25 Ops.CONST dtypes.weakint [] 1
26 Ops.CAST dtypes.i32 [25] dtypes.i32
27 Ops.INDEX dtypes.u64 [24, 26]
28 Ops.LOAD dtypes.u64 [27]
29 Ops.REFPARAM dtypes.f32 [1, 28] ParamArg(slot=1, dtype=dtypes.f32)
30 Ops.INDEX dtypes.f32 [29, 4]
31 Ops.CONST dtypes.weakint [] 0
32 Ops.CAST dtypes.i32 [31] dtypes.i32
33 Ops.INDEX dtypes.u64 [24, 32]
34 Ops.LOAD dtypes.u64 [33]
35 Ops.REFPARAM dtypes.f32 [1, 34] ParamArg(slot=0, dtype=dtypes.f32)
36 Ops.INDEX dtypes.f32 [35, 4]
37 Ops.LOAD dtypes.f32 [36]
38 Ops.ADD dtypes.f32 [37, 10]
39 Ops.STORE dtypes.void [30, 38]
40 Ops.CONST dtypes.weakint [] 2
41 Ops.CAST dtypes.i32 [40] dtypes.i32
42 Ops.INDEX dtypes.u64 [24, 41]
43 Ops.LOAD dtypes.u64 [42]
44 Ops.REFPARAM dtypes.f32 [1, 43] ParamArg(slot=2, dtype=dtypes.f32)
45 Ops.INDEX dtypes.f32 [44, 4]
46 Ops.MUL dtypes.f32 [37, 16]
47 Ops.STORE dtypes.void [45, 46]
48 Ops.AFTER dtypes.void [0, 39, 47]
49 Ops.END dtypes.void [48, 4]
50 Ops.SINK dtypes.void [49] KernelInfo(name="two_outputs", estimates=Estimates(ops=512, lds=3072, mem=3072), optimized=True)
51 Ops.LINEAR dtypes.void [22, 23, 31, 32, 24, 33, 34, 25, 26, 27, 28, 40, 41, 42, 43, 1, 35, 29, 44, 0, 9, 10, 15, 16, 3, 4, 36, 37, 30, 45, 38, 46, 39, 47, 48, 49, 50]
52 Ops.SOURCE dtypes.void [] "\n__attribute__((min_vector_width(512))) void two_outputs(unsigned char* restrict args_24) {\n  unsigned long val0 = (*(((unsigned long*)(args_24))+0));\n  unsigned long val1 = (*(((unsigned long*)(args_24))+1));\n  unsigned long val2 = (*(((unsigned long*)(args_24))+2));\n  for (int Lidx0 = 0; Lidx0 < 256; Lidx0++) {\n    float val3 = (*(((float*)(val0))+Lidx0));\n    *(((float*)(val1))+Lidx0) = (val3+1.0f);\n    *(((float*)(val2))+Lidx0) = (val3*2.0f);\n  }\n}\n"
53 Ops.PROGRAM dtypes.void [50, 51, 52] ProgramInfo(params=(%6, %2, %13), globals=(0, 1, 2), outs=(1, 2), ins=(0,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
54 Ops.SINK dtypes.void [21, 53]

