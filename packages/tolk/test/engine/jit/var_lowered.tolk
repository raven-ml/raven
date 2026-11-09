# tolk D80 D141 D151 golden=9afb1e1b7d958862a35fa78923da1ece
0 Ops.CONST dtypes.weakint [] 30
1 Ops.CONST dtypes.weakint [] 24
2 Ops.PARAM dtypes.u8 [1] ParamArg(slot=0, dtype=dtypes.u8, name="args")
3 Ops.BITCAST dtypes.u64 [2] dtypes.u64
4 Ops.CONST dtypes.weakint [] 0
5 Ops.CAST dtypes.i32 [4] dtypes.i32
6 Ops.INDEX dtypes.u64 [3, 5]
7 Ops.LOAD dtypes.u64 [6]
8 Ops.REFPARAM dtypes.f32 [0, 7] ParamArg(slot=0, dtype=dtypes.f32)
9 Ops.BITCAST dtypes.i32 [2] dtypes.i32
10 Ops.CONST dtypes.weakint [] 2
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.INDEX dtypes.i32 [9, 11]
13 Ops.LOAD dtypes.i32 [12]
14 Ops.CONST dtypes.weakint [] 3
15 Ops.CAST dtypes.i32 [14] dtypes.i32
16 Ops.INDEX dtypes.i32 [9, 15]
17 Ops.LOAD dtypes.i32 [16]
18 Ops.SUB dtypes.i32 [17, 13]
19 Ops.RANGE dtypes.i32 [18] (AxisType.WEAK, 0)
20 Ops.ADD dtypes.i32 [13, 19]
21 Ops.INDEX dtypes.f32 [8, 20]
22 Ops.CONST dtypes.weakfloat [] nan
23 Ops.CAST dtypes.f32 [22] dtypes.f32
24 Ops.STORE dtypes.void [21, 23]
25 Ops.END dtypes.void [24, 19]
26 Ops.CONST dtypes.weakint [] 4
27 Ops.STACK dtypes.void []
28 Ops.PARAM dtypes.i32 [27] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(1, 10), name="i", addrspace=AddrSpace.ALU)
29 Ops.MUL dtypes.i32 [28, 14]
30 Ops.SUB dtypes.i32 [29, 4]
31 Ops.MUL dtypes.i32 [26, 30]
32 Ops.SINK dtypes.void [25] KernelInfo(name="E_(i*3)", estimates=Estimates(lds=%31, mem=%31), optimized=True, split=(%29, 1, 2))
33 Ops.LINEAR dtypes.void [1, 2, 10, 11, 9, 12, 13, 14, 15, 16, 17, 4, 5, 3, 6, 7, 0, 8, 22, 23, 18, 19, 20, 21, 24, 25, 32]
34 Ops.SOURCE dtypes.void [] "\n__attribute__((min_vector_width(512))) void E_28i2A329(unsigned char* restrict args_24) {\n  int val0 = (*(((int*)(args_24))+2));\n  int val1 = (*(((int*)(args_24))+3));\n  unsigned long val2 = (*(((unsigned long*)(args_24))+0));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    *(((float*)(val2))+(val0+Lidx0)) = __builtin_bit_cast(float, (unsigned int)(0x7fc00000u));\n  }\n}\n"
35 Ops.BINARY dtypes.u8 [] b"\n__attribute__((min_vector_width(512))) void E_28i2A329(unsigned char* restrict args_24) {\n  int val0 = (*(((int*)(args_24))+2));\n  int val1 = (*(((int*)(args_24))+3));\n  unsigned long val2 = (*(((unsigned long*)(args_24))+0));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    *(((float*)(val2))+(val0+Lidx0)) = __builtin_bit_cast(float, (unsigned int)(0x7fc00000u));\n  }\n}\n"
36 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
37 Ops.PARAM dtypes.i32 [27] ParamArg(slot=1, dtype=dtypes.i32, vmin_vmax=(0, 30), name="block_lo", addrspace=AddrSpace.ALU)
38 Ops.PARAM dtypes.i32 [27] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 30), name="block_hi", addrspace=AddrSpace.ALU)
39 Ops.PROGRAM dtypes.void [32, 33, 34, 35] ProgramInfo(global_size=(%29, 1, 1), params=(%36, %37, %38), globals=(0,), outs=(0,), target=Target(device="CPU", arch="x86_64,x86-64"))
40 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
41 Ops.CALL dtypes.void [39, 40] CallInfo()
42 Ops.LINEAR dtypes.void [41]

