# tolk D141 D151 golden=d9b23911a76afccf8b918b096b1ddbf6
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
12 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
13 Ops.STACK dtypes.void []
14 Ops.SHRINK dtypes.f32 [12, 5, 10]
15 Ops.CONST dtypes.weakfloat [] 0.0
16 Ops.CAST dtypes.f32 [15] dtypes.f32
17 Ops.STACK dtypes.f32 [16, 16]
18 Ops.STORE dtypes.void [14, 17]
19 Ops.INDEX dtypes.f32 [12, 10]
20 Ops.STORE dtypes.void [19, 16]
21 Ops.BITCAST dtypes.i32 [2] dtypes.i32
22 Ops.CONST dtypes.weakint [] 4
23 Ops.CAST dtypes.i32 [22] dtypes.i32
24 Ops.INDEX dtypes.i32 [21, 23]
25 Ops.LOAD dtypes.i32 [24]
26 Ops.RANGE dtypes.i32 [25] (AxisType.WEAK, 0)
27 Ops.AFTER dtypes.f32 [12, 18, 20, 26]
28 Ops.SHRINK dtypes.f32 [27, 5, 10]
29 Ops.LOAD dtypes.f32 [28]
30 Ops.INDEX dtypes.f32 [29, 5]
31 Ops.CONST dtypes.weakint [] 1
32 Ops.CAST dtypes.i32 [31] dtypes.i32
33 Ops.INDEX dtypes.f32 [29, 32]
34 Ops.INDEX dtypes.f32 [27, 10]
35 Ops.LOAD dtypes.f32 [34]
36 Ops.STACK dtypes.f32 [30, 33, 35]
37 Ops.CONST dtypes.weakint [] 30
38 Ops.INDEX dtypes.u64 [3, 32]
39 Ops.LOAD dtypes.u64 [38]
40 Ops.REFPARAM dtypes.f32 [37, 39] ParamArg(slot=1, dtype=dtypes.f32)
41 Ops.CAST dtypes.i32 [0] dtypes.i32
42 Ops.MUL dtypes.i32 [26, 41]
43 Ops.INDEX dtypes.f32 [40, 42]
44 Ops.LOAD dtypes.f32 [43]
45 Ops.ADD dtypes.i32 [42, 32]
46 Ops.INDEX dtypes.f32 [40, 45]
47 Ops.LOAD dtypes.f32 [46]
48 Ops.ADD dtypes.i32 [42, 10]
49 Ops.INDEX dtypes.f32 [40, 48]
50 Ops.LOAD dtypes.f32 [49]
51 Ops.STACK dtypes.f32 [44, 47, 50]
52 Ops.ADD dtypes.f32 [36, 51]
53 Ops.INDEX dtypes.f32 [52, 5]
54 Ops.INDEX dtypes.f32 [52, 32]
55 Ops.STACK dtypes.f32 [53, 54]
56 Ops.STORE dtypes.void [28, 55]
57 Ops.INDEX dtypes.f32 [52, 10]
58 Ops.STORE dtypes.void [34, 57]
59 Ops.AFTER dtypes.void [13, 56, 58]
60 Ops.END dtypes.void [59, 26]
61 Ops.AFTER dtypes.f32 [12, 60]
62 Ops.SHRINK dtypes.f32 [61, 5, 10]
63 Ops.LOAD dtypes.f32 [62]
64 Ops.INDEX dtypes.f32 [63, 5]
65 Ops.INDEX dtypes.f32 [63, 32]
66 Ops.INDEX dtypes.f32 [61, 10]
67 Ops.LOAD dtypes.f32 [66]
68 Ops.STACK dtypes.f32 [64, 65, 67]
69 Ops.CONST dtypes.weakfloat [] 1.0
70 Ops.CAST dtypes.f32 [69] dtypes.f32
71 Ops.CAST dtypes.f32 [25] dtypes.f32
72 Ops.FDIV dtypes.f32 [70, 71]
73 Ops.MUL dtypes.f32 [68, 72]
74 Ops.INDEX dtypes.f32 [73, 5]
75 Ops.INDEX dtypes.f32 [73, 32]
76 Ops.STACK dtypes.f32 [74, 75]
77 Ops.STORE dtypes.void [11, 76]
78 Ops.INDEX dtypes.f32 [8, 10]
79 Ops.INDEX dtypes.f32 [73, 10]
80 Ops.STORE dtypes.void [78, 79]
81 Ops.PARAM dtypes.i32 [13] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(1, 10), name="i", addrspace=AddrSpace.ALU)
82 Ops.MUL dtypes.i32 [81, 0]
83 Ops.ADD dtypes.i32 [82, 22]
84 Ops.MUL dtypes.i32 [31, 81]
85 Ops.MUL dtypes.i32 [22, 84]
86 Ops.ADD dtypes.i32 [4, 85]
87 Ops.ADD dtypes.i32 [86, 85]
88 Ops.ADD dtypes.i32 [87, 85]
89 Ops.ADD dtypes.i32 [88, 22]
90 Ops.CONST dtypes.weakint [] 8
91 Ops.ADD dtypes.i32 [89, 90]
92 Ops.CONST dtypes.weakint [] 12
93 Ops.MUL dtypes.i32 [81, 92]
94 Ops.ADD dtypes.i32 [4, 93]
95 Ops.ADD dtypes.i32 [94, 92]
96 Ops.SINK dtypes.void [77, 77, 80] KernelInfo(name="r_3_i", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(3, AxisType.UPCAST)),), estimates=Estimates(ops=%83, lds=%91, mem=%95), optimized=True)
97 Ops.LINEAR dtypes.void [1, 2, 0, 12, 22, 23, 21, 24, 25, 31, 32, 3, 38, 39, 37, 40, 13, 15, 16, 17, 9, 10, 19, 4, 5, 14, 41, 20, 18, 26, 42, 45, 46, 47, 48, 49, 50, 43, 44, 27, 34, 35, 28, 29, 30, 33, 36, 51, 52, 53, 54, 55, 57, 58, 56, 59, 60, 6, 7, 61, 66, 67, 62, 63, 8, 64, 65, 68, 71, 69, 70, 72, 73, 74, 75, 76, 78, 79, 11, 80, 77, 96]
98 Ops.SOURCE dtypes.void [] "typedef float float2 __attribute__((aligned(4),ext_vector_type(2)));\ntypedef float float3 __attribute__((aligned(4),ext_vector_type(3)));\n__attribute__((min_vector_width(512))) void r_3_i(unsigned char* restrict args_24) {\n  float buf0[3];\n  int val0 = (*(((int*)(args_24))+4));\n  unsigned long val1 = (*(((unsigned long*)(args_24))+1));\n  *(buf0+2) = 0.0f;\n  *((float2*)((buf0+0))) = (float2){0.0f,0.0f};\n  for (int Lidx0 = 0; Lidx0 < val0; Lidx0++) {\n    int alu2 = (Lidx0*3);\n    float val2 = (*(((float*)(val1))+(alu2+1)));\n    float val3 = (*(((float*)(val1))+(alu2+2)));\n    float val4 = (*(((float*)(val1))+alu2));\n    float2 val5 = (*((float2*)((buf0+0))));\n    float3 alu3 = ((float3){val5[0],val5[1],(*(buf0+2))}+(float3){val4,val2,val3});\n    *(buf0+2) = alu3[2];\n    *((float2*)((buf0+0))) = ((float2)__builtin_shufflevector(alu3, alu3, 0, 1));\n  }\n  unsigned long val6 = (*(((unsigned long*)(args_24))+0));\n  float2 val7 = (*((float2*)((buf0+0))));\n  float3 alu7 = ((float3){val7[0],val7[1],(*(buf0+2))}*(1.0f/((float)(val0))));\n  *(((float*)(val6))+2) = alu7[2];\n  *((float2*)((((float*)(val6))+0))) = ((float2)__builtin_shufflevector(alu7, alu7, 0, 1));\n}\n"
99 Ops.BINARY dtypes.u8 [] b"typedef float float2 __attribute__((aligned(4),ext_vector_type(2)));\ntypedef float float3 __attribute__((aligned(4),ext_vector_type(3)));\n__attribute__((min_vector_width(512))) void r_3_i(unsigned char* restrict args_24) {\n  float buf0[3];\n  int val0 = (*(((int*)(args_24))+4));\n  unsigned long val1 = (*(((unsigned long*)(args_24))+1));\n  *(buf0+2) = 0.0f;\n  *((float2*)((buf0+0))) = (float2){0.0f,0.0f};\n  for (int Lidx0 = 0; Lidx0 < val0; Lidx0++) {\n    int alu2 = (Lidx0*3);\n    float val2 = (*(((float*)(val1))+(alu2+1)));\n    float val3 = (*(((float*)(val1))+(alu2+2)));\n    float val4 = (*(((float*)(val1))+alu2));\n    float2 val5 = (*((float2*)((buf0+0))));\n    float3 alu3 = ((float3){val5[0],val5[1],(*(buf0+2))}+(float3){val4,val2,val3});\n    *(buf0+2) = alu3[2];\n    *((float2*)((buf0+0))) = ((float2)__builtin_shufflevector(alu3, alu3, 0, 1));\n  }\n  unsigned long val6 = (*(((unsigned long*)(args_24))+0));\n  float2 val7 = (*((float2*)((buf0+0))));\n  float3 alu7 = ((float3){val7[0],val7[1],(*(buf0+2))}*(1.0f/((float)(val0))));\n  *(((float*)(val6))+2) = alu7[2];\n  *((float2*)((((float*)(val6))+0))) = ((float2)__builtin_shufflevector(alu7, alu7, 0, 1));\n}\n"
100 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
101 Ops.PARAM dtypes.f32 [37] ParamArg(slot=1, dtype=dtypes.f32)
102 Ops.PROGRAM dtypes.void [96, 97, 98, 99] ProgramInfo(params=(%100, %101, %81), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
103 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
104 Ops.PARAM dtypes.f32 [37] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
105 Ops.CALL dtypes.void [102, 103, 104] CallInfo()
106 Ops.LINEAR dtypes.void [105]

