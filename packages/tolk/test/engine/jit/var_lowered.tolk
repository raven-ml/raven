# tolk D80 D141 D151 golden=9afb1e1b7d958862a35fa78923da1ece
0 Ops.CONST dtypes.weakint [] 30
1 Ops.CONST dtypes.weakint [] 32
2 Ops.PARAM dtypes.u8 [1] ParamArg(slot=0, dtype=dtypes.u8, name="args")
3 Ops.BITCAST dtypes.u64 [2] dtypes.u64
4 Ops.CONST dtypes.weakint [] 0
5 Ops.CAST dtypes.i32 [4] dtypes.i32
6 Ops.INDEX dtypes.u64 [3, 5]
7 Ops.LOAD dtypes.u64 [6]
8 Ops.REFPARAM dtypes.f32 [0, 7] ParamArg(slot=0, dtype=dtypes.f32)
9 Ops.BITCAST dtypes.i32 [2] dtypes.i32
10 Ops.CONST dtypes.weakint [] 4
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.INDEX dtypes.i32 [9, 11]
13 Ops.LOAD dtypes.i32 [12]
14 Ops.CONST dtypes.weakint [] 5
15 Ops.CAST dtypes.i32 [14] dtypes.i32
16 Ops.INDEX dtypes.i32 [9, 15]
17 Ops.LOAD dtypes.i32 [16]
18 Ops.SUB dtypes.i32 [17, 13]
19 Ops.RANGE dtypes.i32 [18] (AxisType.WEAK, 0)
20 Ops.ADD dtypes.i32 [13, 19]
21 Ops.INDEX dtypes.f32 [8, 20]
22 Ops.CONST dtypes.weakint [] 1
23 Ops.CAST dtypes.i32 [22] dtypes.i32
24 Ops.INDEX dtypes.u64 [3, 23]
25 Ops.LOAD dtypes.u64 [24]
26 Ops.REFPARAM dtypes.f32 [0, 25] ParamArg(slot=1, dtype=dtypes.f32)
27 Ops.INDEX dtypes.f32 [26, 20]
28 Ops.LOAD dtypes.f32 [27]
29 Ops.SUB dtypes.f32 [28, 28]
30 Ops.MUL dtypes.f32 [29, 29]
31 Ops.CONST dtypes.weakfloat [] inf
32 Ops.CAST dtypes.f32 [31] dtypes.f32
33 Ops.MUL dtypes.f32 [30, 32]
34 Ops.STORE dtypes.void [21, 33]
35 Ops.END dtypes.void [34, 19]
36 Ops.STACK dtypes.void []
37 Ops.PARAM dtypes.i32 [36] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(1, 10), name="i", addrspace=AddrSpace.ALU)
38 Ops.CONST dtypes.weakint [] 3
39 Ops.MUL dtypes.i32 [37, 38]
40 Ops.SUB dtypes.i32 [39, 4]
41 Ops.MUL dtypes.i32 [40, 38]
42 Ops.CONST dtypes.weakint [] 8
43 Ops.MUL dtypes.i32 [40, 42]
44 Ops.SINK dtypes.void [35] KernelInfo(name="E_(i*3)", estimates=Estimates(ops=%41, lds=%43, mem=%43), optimized=True, split=(%39, 2, 3))
45 Ops.LINEAR dtypes.void [1, 2, 10, 11, 9, 12, 13, 14, 15, 16, 17, 4, 5, 3, 6, 7, 22, 23, 24, 25, 0, 8, 26, 31, 32, 18, 19, 20, 27, 28, 21, 29, 30, 33, 34, 35, 44]
46 Ops.SOURCE dtypes.void [] "\n__attribute__((min_vector_width(512))) void E_28i2A329(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = (val0+Lidx0);\n    float val4 = (*(((float*)(val3))+alu0));\n    float alu1 = (val4-val4);\n    *(((float*)(val2))+alu0) = (alu1*alu1*((float)(__builtin_inff())));\n  }\n}\n"
47 Ops.BINARY dtypes.u8 [] b"\n__attribute__((min_vector_width(512))) void E_28i2A329(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = (val0+Lidx0);\n    float val4 = (*(((float*)(val3))+alu0));\n    float alu1 = (val4-val4);\n    *(((float*)(val2))+alu0) = (alu1*alu1*((float)(__builtin_inff())));\n  }\n}\n"
48 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
49 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
50 Ops.PARAM dtypes.i32 [36] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 30), name="block_lo", addrspace=AddrSpace.ALU)
51 Ops.PARAM dtypes.i32 [36] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 30), name="block_hi", addrspace=AddrSpace.ALU)
52 Ops.PROGRAM dtypes.void [44, 45, 46, 47] ProgramInfo(global_size=(%39, 1, 1), params=(%48, %49, %50, %51), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
53 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
54 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
55 Ops.CALL dtypes.void [52, 53, 54] CallInfo()
56 Ops.LINEAR dtypes.void [55]

