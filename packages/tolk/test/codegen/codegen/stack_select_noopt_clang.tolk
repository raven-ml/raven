# tolk D80 D151 golden=80339f30e6bce41ac825c35e7192ad06
0 Ops.CONST dtypes.weakint [] 60
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.STACK dtypes.void []
3 Ops.PARAM dtypes.i32 [2] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 60), name="block_lo", addrspace=AddrSpace.ALU)
4 Ops.PARAM dtypes.i32 [2] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 60), name="block_hi", addrspace=AddrSpace.ALU)
5 Ops.SUB dtypes.i32 [4, 3]
6 Ops.RANGE dtypes.i32 [5] (AxisType.WEAK, 0)
7 Ops.ADD dtypes.i32 [3, 6]
8 Ops.INDEX dtypes.f32 [1, 7]
9 Ops.CONST dtypes.weakint [] 28
10 Ops.PARAM dtypes.f32 [9] ParamArg(slot=1, dtype=dtypes.f32)
11 Ops.CONST dtypes.weakint [] -12
12 Ops.CAST dtypes.i32 [11] dtypes.i32
13 Ops.ADD dtypes.i32 [7, 12]
14 Ops.INDEX dtypes.f32 [10, 13]
15 Ops.CONST dtypes.weakfloat [] 0.0
16 Ops.CAST dtypes.f32 [15] dtypes.f32
17 Ops.CONST dtypes.weakint [] 40
18 Ops.CAST dtypes.i32 [17] dtypes.i32
19 Ops.CMPLT dtypes.bool [7, 18]
20 Ops.CONST dtypes.weakint [] 11
21 Ops.CAST dtypes.i32 [20] dtypes.i32
22 Ops.CMPLT dtypes.bool [21, 7]
23 Ops.AND dtypes.bool [19, 22]
24 Ops.LOAD dtypes.f32 [14, 16, 23]
25 Ops.STORE dtypes.void [8, 24]
26 Ops.END dtypes.void [25, 6]
27 Ops.SINK dtypes.void [26] KernelInfo(name="E_60", optimized=True, split=(60, 2, 3))
28 Ops.CONST dtypes.weakint [] 32
29 Ops.PARAM dtypes.u8 [28] ParamArg(slot=0, dtype=dtypes.u8, name="args")
30 Ops.BITCAST dtypes.u64 [29] dtypes.u64
31 Ops.CONST dtypes.weakint [] 0
32 Ops.CAST dtypes.i32 [31] dtypes.i32
33 Ops.INDEX dtypes.u64 [30, 32]
34 Ops.LOAD dtypes.u64 [33]
35 Ops.REFPARAM dtypes.f32 [0, 34] ParamArg(slot=0, dtype=dtypes.f32)
36 Ops.BITCAST dtypes.i32 [29] dtypes.i32
37 Ops.CONST dtypes.weakint [] 4
38 Ops.CAST dtypes.i32 [37] dtypes.i32
39 Ops.INDEX dtypes.i32 [36, 38]
40 Ops.LOAD dtypes.i32 [39]
41 Ops.CONST dtypes.weakint [] 5
42 Ops.CAST dtypes.i32 [41] dtypes.i32
43 Ops.INDEX dtypes.i32 [36, 42]
44 Ops.LOAD dtypes.i32 [43]
45 Ops.SUB dtypes.i32 [44, 40]
46 Ops.RANGE dtypes.i32 [45] (AxisType.WEAK, 0)
47 Ops.ADD dtypes.i32 [40, 46]
48 Ops.INDEX dtypes.f32 [35, 47]
49 Ops.CONST dtypes.weakint [] 1
50 Ops.CAST dtypes.i32 [49] dtypes.i32
51 Ops.INDEX dtypes.u64 [30, 50]
52 Ops.LOAD dtypes.u64 [51]
53 Ops.REFPARAM dtypes.f32 [9, 52] ParamArg(slot=1, dtype=dtypes.f32)
54 Ops.ADD dtypes.i32 [47, 12]
55 Ops.INDEX dtypes.f32 [53, 54]
56 Ops.CMPLT dtypes.bool [47, 18]
57 Ops.CMPLT dtypes.bool [21, 47]
58 Ops.AND dtypes.bool [56, 57]
59 Ops.LOAD dtypes.f32 [55, 16, 58]
60 Ops.STORE dtypes.void [48, 59]
61 Ops.END dtypes.void [60, 46]
62 Ops.SINK dtypes.void [61] KernelInfo(name="E_60", estimates=Estimates(ops=180, lds=480, mem=352), optimized=True, split=(60, 2, 3))
63 Ops.LINEAR dtypes.void [28, 29, 37, 38, 36, 39, 40, 41, 42, 43, 44, 31, 32, 30, 33, 34, 49, 50, 51, 52, 0, 35, 9, 53, 15, 16, 11, 12, 20, 21, 17, 18, 45, 46, 47, 54, 55, 57, 56, 58, 59, 48, 60, 61, 62]
64 Ops.SOURCE dtypes.void [] "\n__attribute__((min_vector_width(512))) void E_60(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = (val0+Lidx0);\n    float val4 = (((alu0<40)&(11<alu0))?*(((float*)(val3))+(alu0+-12)):0.0f);\n    *(((float*)(val2))+alu0) = val4;\n  }\n}\n"
65 Ops.PROGRAM dtypes.void [62, 63, 64] ProgramInfo(global_size=(60, 1, 1), params=(%1, %10, %3, %4), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
66 Ops.SINK dtypes.void [27, 65]

