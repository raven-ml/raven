# tolk D80 D141 D151 golden=b1d8c5bd353179826caca4166a8cce5d
0 Ops.CONST dtypes.weakint [] 100
1 Ops.CONST dtypes.weakint [] 32
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
21 Ops.CONST dtypes.weakint [] 10
22 Ops.CAST dtypes.i32 [21] dtypes.i32
23 Ops.MUL dtypes.i32 [20, 22]
24 Ops.CONST dtypes.weakint [] 4
25 Ops.CAST dtypes.i32 [24] dtypes.i32
26 Ops.INDEX dtypes.i32 [9, 25]
27 Ops.LOAD dtypes.i32 [26]
28 Ops.AFTER dtypes.i32 [27, 19]
29 Ops.RANGE dtypes.i32 [28] (AxisType.WEAK, 1)
30 Ops.ADD dtypes.i32 [23, 29]
31 Ops.INDEX dtypes.f32 [8, 30]
32 Ops.CONST dtypes.weakfloat [] nan
33 Ops.CAST dtypes.f32 [32] dtypes.f32
34 Ops.STORE dtypes.void [31, 33]
35 Ops.END dtypes.void [34, 29]
36 Ops.END dtypes.void [35, 19]
37 Ops.STACK dtypes.void []
38 Ops.PARAM dtypes.i32 [37] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(1, 10), name="i", addrspace=AddrSpace.ALU)
39 Ops.SUB dtypes.i32 [38, 4]
40 Ops.PARAM dtypes.i32 [37] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(1, 10), name="j", addrspace=AddrSpace.ALU)
41 Ops.MUL dtypes.i32 [39, 40]
42 Ops.MUL dtypes.i32 [24, 41]
43 Ops.SINK dtypes.void [36] KernelInfo(name="E_i_j", estimates=Estimates(lds=%42, mem=%42), optimized=True, split=(%38, 1, 2))
44 Ops.LINEAR dtypes.void [1, 2, 10, 11, 9, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 4, 5, 3, 6, 7, 0, 8, 32, 33, 21, 22, 18, 19, 28, 20, 23, 29, 30, 31, 34, 35, 36, 43]
45 Ops.SOURCE dtypes.void [] "\n__attribute__((min_vector_width(512))) void E_i_j(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+2));\n  int val1 = (*(((int*)(args_32))+3));\n  int val2 = (*(((int*)(args_32))+4));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+0));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    for (int Lidx1 = 0; Lidx1 < val2; Lidx1++) {\n      *(((float*)(val3))+(((val0+Lidx0)*10)+Lidx1)) = __builtin_bit_cast(float, (unsigned int)(0x7fc00000u));\n    }\n  }\n}\n"
46 Ops.BINARY dtypes.u8 [] b"\n__attribute__((min_vector_width(512))) void E_i_j(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+2));\n  int val1 = (*(((int*)(args_32))+3));\n  int val2 = (*(((int*)(args_32))+4));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+0));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    for (int Lidx1 = 0; Lidx1 < val2; Lidx1++) {\n      *(((float*)(val3))+(((val0+Lidx0)*10)+Lidx1)) = __builtin_bit_cast(float, (unsigned int)(0x7fc00000u));\n    }\n  }\n}\n"
47 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
48 Ops.PARAM dtypes.i32 [37] ParamArg(slot=1, dtype=dtypes.i32, vmin_vmax=(0, 10), name="block_lo", addrspace=AddrSpace.ALU)
49 Ops.PARAM dtypes.i32 [37] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 10), name="block_hi", addrspace=AddrSpace.ALU)
50 Ops.PROGRAM dtypes.void [43, 44, 45, 46] ProgramInfo(global_size=(%38, 1, 1), params=(%47, %48, %49, %40), globals=(0,), outs=(0,), target=Target(device="CPU", arch="x86_64,x86-64"))
51 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
52 Ops.CALL dtypes.void [50, 51] CallInfo()
53 Ops.LINEAR dtypes.void [52]

