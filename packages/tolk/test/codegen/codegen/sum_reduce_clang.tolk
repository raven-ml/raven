# tolk D151 golden=1a2db27b83b97eece24952ac3a2d2607
0 Ops.CONST dtypes.weakint [] 1
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.INDEX dtypes.f32 [1, 3]
5 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
6 Ops.INDEX dtypes.f32 [5, 3]
7 Ops.CONST dtypes.weakfloat [] 0.0
8 Ops.CAST dtypes.f32 [7] dtypes.f32
9 Ops.STORE dtypes.void [6, 8]
10 Ops.CONST dtypes.weakint [] 64
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.RANGE dtypes.i32 [11] (AxisType.WEAK, 0)
13 Ops.AFTER dtypes.f32 [5, 9, 12]
14 Ops.INDEX dtypes.f32 [13, 3]
15 Ops.LOAD dtypes.f32 [14]
16 Ops.CONST dtypes.weakint [] 256
17 Ops.PARAM dtypes.f32 [16] ParamArg(slot=0, dtype=dtypes.f32)
18 Ops.CONST dtypes.weakint [] 2
19 Ops.CAST dtypes.i32 [18] dtypes.i32
20 Ops.SHL dtypes.i32 [12, 19]
21 Ops.CONST dtypes.weakint [] 4
22 Ops.CAST dtypes.i32 [21] dtypes.i32
23 Ops.SHRINK dtypes.f32 [17, 20, 22]
24 Ops.LOAD dtypes.f32 [23]
25 Ops.INDEX dtypes.f32 [24, 3]
26 Ops.CAST dtypes.i32 [0] dtypes.i32
27 Ops.INDEX dtypes.f32 [24, 26]
28 Ops.ADD dtypes.f32 [25, 27]
29 Ops.INDEX dtypes.f32 [24, 19]
30 Ops.ADD dtypes.f32 [28, 29]
31 Ops.CONST dtypes.weakint [] 3
32 Ops.CAST dtypes.i32 [31] dtypes.i32
33 Ops.INDEX dtypes.f32 [24, 32]
34 Ops.ADD dtypes.f32 [30, 33]
35 Ops.ADD dtypes.f32 [15, 34]
36 Ops.STORE dtypes.void [14, 35]
37 Ops.END dtypes.void [36, 12]
38 Ops.AFTER dtypes.f32 [5, 37]
39 Ops.INDEX dtypes.f32 [38, 3]
40 Ops.LOAD dtypes.f32 [39]
41 Ops.STORE dtypes.void [4, 40]
42 Ops.SINK dtypes.void [41] KernelInfo(name="sum_reduce", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
43 Ops.CONST dtypes.weakint [] 16
44 Ops.PARAM dtypes.u8 [43] ParamArg(slot=0, dtype=dtypes.u8, name="args")
45 Ops.BITCAST dtypes.u64 [44] dtypes.u64
46 Ops.INDEX dtypes.u64 [45, 26]
47 Ops.LOAD dtypes.u64 [46]
48 Ops.REFPARAM dtypes.f32 [0, 47] ParamArg(slot=1, dtype=dtypes.f32)
49 Ops.INDEX dtypes.f32 [48, 3]
50 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
51 Ops.INDEX dtypes.f32 [50, 3]
52 Ops.STORE dtypes.void [51, 8]
53 Ops.AFTER dtypes.f32 [50, 52, 12]
54 Ops.INDEX dtypes.f32 [53, 3]
55 Ops.LOAD dtypes.f32 [54]
56 Ops.INDEX dtypes.u64 [45, 3]
57 Ops.LOAD dtypes.u64 [56]
58 Ops.REFPARAM dtypes.f32 [16, 57] ParamArg(slot=0, dtype=dtypes.f32)
59 Ops.SHRINK dtypes.f32 [58, 20, 22]
60 Ops.LOAD dtypes.f32 [59]
61 Ops.INDEX dtypes.f32 [60, 3]
62 Ops.INDEX dtypes.f32 [60, 26]
63 Ops.ADD dtypes.f32 [61, 62]
64 Ops.INDEX dtypes.f32 [60, 19]
65 Ops.ADD dtypes.f32 [63, 64]
66 Ops.INDEX dtypes.f32 [60, 32]
67 Ops.ADD dtypes.f32 [65, 66]
68 Ops.ADD dtypes.f32 [55, 67]
69 Ops.STORE dtypes.void [54, 68]
70 Ops.END dtypes.void [69, 12]
71 Ops.AFTER dtypes.f32 [50, 70]
72 Ops.INDEX dtypes.f32 [71, 3]
73 Ops.LOAD dtypes.f32 [72]
74 Ops.STORE dtypes.void [49, 73]
75 Ops.SINK dtypes.void [74] KernelInfo(name="sum_reduce", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), estimates=Estimates(ops=256, lds=1028, mem=1028), optimized=True)
76 Ops.LINEAR dtypes.void [43, 44, 0, 50, 2, 3, 45, 56, 57, 16, 58, 51, 7, 8, 26, 18, 19, 31, 32, 21, 22, 10, 11, 52, 12, 53, 54, 55, 20, 59, 60, 61, 62, 64, 66, 63, 65, 67, 68, 69, 70, 46, 47, 71, 72, 73, 48, 49, 74, 75]
77 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void sum_reduce(unsigned char* restrict args_16) {\n  float buf0[1];\n  unsigned long val0 = (*(((unsigned long*)(args_16))+0));\n  *(buf0+0) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 64; Lidx0++) {\n    float4 val1 = (*((float4*)((((float*)(val0))+(Lidx0<<2)))));\n    *(buf0+0) = ((*(buf0+0))+(val1[0]+val1[1]+val1[2]+val1[3]));\n  }\n  unsigned long val2 = (*(((unsigned long*)(args_16))+1));\n  *(((float*)(val2))+0) = (*(buf0+0));\n}\n"
78 Ops.PROGRAM dtypes.void [75, 76, 77] ProgramInfo(params=(%17, %1), globals=(0, 1), outs=(1,), ins=(0,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
79 Ops.SINK dtypes.void [42, 78]

