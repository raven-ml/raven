# tolk D80 D151 golden=b7ad41f492aa8b5196e880fe2833d6d0
0 Ops.CONST dtypes.weakint [] 3
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.STACK dtypes.void []
3 Ops.PARAM dtypes.i32 [2] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 3), name="block_lo", addrspace=AddrSpace.ALU)
4 Ops.PARAM dtypes.i32 [2] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 3), name="block_hi", addrspace=AddrSpace.ALU)
5 Ops.SUB dtypes.i32 [4, 3]
6 Ops.RANGE dtypes.i32 [5] (AxisType.WEAK, 0)
7 Ops.ADD dtypes.i32 [3, 6]
8 Ops.INDEX dtypes.f32 [1, 7]
9 Ops.CONST dtypes.weakint [] 4
10 Ops.PARAM dtypes.f32 [9] ParamArg(slot=1, dtype=dtypes.f32)
11 Ops.INDEX dtypes.f32 [10, 7]
12 Ops.LOAD dtypes.f32 [11]
13 Ops.CONST dtypes.weakint [] 1
14 Ops.CAST dtypes.i32 [13] dtypes.i32
15 Ops.ADD dtypes.i32 [7, 14]
16 Ops.INDEX dtypes.f32 [10, 15]
17 Ops.LOAD dtypes.f32 [16]
18 Ops.ADD dtypes.f32 [12, 17]
19 Ops.STORE dtypes.void [8, 18]
20 Ops.END dtypes.void [19, 6]
21 Ops.SINK dtypes.void [20] KernelInfo(name="E_3", optimized=True, split=(3, 2, 3))
22 Ops.CONST dtypes.weakint [] 32
23 Ops.PARAM dtypes.u8 [22] ParamArg(slot=0, dtype=dtypes.u8, name="args")
24 Ops.BITCAST dtypes.u64 [23] dtypes.u64
25 Ops.CONST dtypes.weakint [] 0
26 Ops.CAST dtypes.i32 [25] dtypes.i32
27 Ops.INDEX dtypes.u64 [24, 26]
28 Ops.LOAD dtypes.u64 [27]
29 Ops.REFPARAM dtypes.f32 [0, 28] ParamArg(slot=0, dtype=dtypes.f32)
30 Ops.BITCAST dtypes.i32 [23] dtypes.i32
31 Ops.CAST dtypes.i32 [9] dtypes.i32
32 Ops.INDEX dtypes.i32 [30, 31]
33 Ops.LOAD dtypes.i32 [32]
34 Ops.CONST dtypes.weakint [] 5
35 Ops.CAST dtypes.i32 [34] dtypes.i32
36 Ops.INDEX dtypes.i32 [30, 35]
37 Ops.LOAD dtypes.i32 [36]
38 Ops.SUB dtypes.i32 [37, 33]
39 Ops.RANGE dtypes.i32 [38] (AxisType.WEAK, 0)
40 Ops.ADD dtypes.i32 [33, 39]
41 Ops.INDEX dtypes.f32 [29, 40]
42 Ops.INDEX dtypes.u64 [24, 14]
43 Ops.LOAD dtypes.u64 [42]
44 Ops.REFPARAM dtypes.f32 [9, 43] ParamArg(slot=1, dtype=dtypes.f32)
45 Ops.INDEX dtypes.f32 [44, 40]
46 Ops.LOAD dtypes.f32 [45]
47 Ops.ADD dtypes.i32 [40, 14]
48 Ops.INDEX dtypes.f32 [44, 47]
49 Ops.LOAD dtypes.f32 [48]
50 Ops.ADD dtypes.f32 [46, 49]
51 Ops.STORE dtypes.void [41, 50]
52 Ops.END dtypes.void [51, 39]
53 Ops.SINK dtypes.void [52] KernelInfo(name="E_3", estimates=Estimates(ops=3, lds=36, mem=28), optimized=True, split=(3, 2, 3))
54 Ops.LINEAR dtypes.void [22, 23, 9, 31, 30, 32, 33, 34, 35, 36, 37, 25, 26, 24, 27, 28, 13, 14, 42, 43, 0, 29, 44, 38, 39, 40, 45, 46, 47, 48, 49, 41, 50, 51, 52, 53]
55 Ops.SOURCE dtypes.void [] "\n__attribute__((min_vector_width(512))) void E_3(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = (val0+Lidx0);\n    float val4 = (*(((float*)(val3))+alu0));\n    float val5 = (*(((float*)(val3))+(alu0+1)));\n    *(((float*)(val2))+alu0) = (val4+val5);\n  }\n}\n"
56 Ops.PROGRAM dtypes.void [53, 54, 55] ProgramInfo(global_size=(3, 1, 1), params=(%1, %10, %3, %4), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
57 Ops.SINK dtypes.void [21, 56]

