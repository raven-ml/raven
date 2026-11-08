# tolk D80 D151 golden=51a5ffd717d79b4487ebdd325043b7df
0 Ops.CONST dtypes.weakint [] 65536
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.STACK dtypes.void []
3 Ops.PARAM dtypes.i32 [2] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 256), name="block_lo", addrspace=AddrSpace.ALU)
4 Ops.PARAM dtypes.i32 [2] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 256), name="block_hi", addrspace=AddrSpace.ALU)
5 Ops.SUB dtypes.i32 [4, 3]
6 Ops.RANGE dtypes.i32 [5] (AxisType.WEAK, 0)
7 Ops.ADD dtypes.i32 [3, 6]
8 Ops.CONST dtypes.weakint [] 8
9 Ops.CAST dtypes.i32 [8] dtypes.i32
10 Ops.SHL dtypes.i32 [7, 9]
11 Ops.CONST dtypes.weakint [] 256
12 Ops.CAST dtypes.i32 [11] dtypes.i32
13 Ops.AFTER dtypes.i32 [12, 6]
14 Ops.RANGE dtypes.i32 [13] (AxisType.WEAK, 1)
15 Ops.ADD dtypes.i32 [10, 14]
16 Ops.INDEX dtypes.f32 [1, 15]
17 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
18 Ops.INDEX dtypes.f32 [17, 15]
19 Ops.CONST dtypes.weakfloat [] 0.0
20 Ops.CAST dtypes.f32 [19] dtypes.f32
21 Ops.CMPLT dtypes.bool [14, 7]
22 Ops.CONST dtypes.bool [] False
23 Ops.CAST dtypes.bool [22] dtypes.bool
24 Ops.CMPEQ dtypes.bool [21, 23]
25 Ops.LOAD dtypes.f32 [18, 20, 24]
26 Ops.STORE dtypes.void [16, 25]
27 Ops.END dtypes.void [26, 14]
28 Ops.END dtypes.void [27, 6]
29 Ops.SINK dtypes.void [28] KernelInfo(name="E_256_256", optimized=True, split=(256, 2, 3))
30 Ops.CONST dtypes.weakint [] 32
31 Ops.PARAM dtypes.u8 [30] ParamArg(slot=0, dtype=dtypes.u8, name="args")
32 Ops.BITCAST dtypes.u64 [31] dtypes.u64
33 Ops.CONST dtypes.weakint [] 0
34 Ops.CAST dtypes.i32 [33] dtypes.i32
35 Ops.INDEX dtypes.u64 [32, 34]
36 Ops.LOAD dtypes.u64 [35]
37 Ops.REFPARAM dtypes.f32 [0, 36] ParamArg(slot=0, dtype=dtypes.f32)
38 Ops.BITCAST dtypes.i32 [31] dtypes.i32
39 Ops.CONST dtypes.weakint [] 4
40 Ops.CAST dtypes.i32 [39] dtypes.i32
41 Ops.INDEX dtypes.i32 [38, 40]
42 Ops.LOAD dtypes.i32 [41]
43 Ops.CONST dtypes.weakint [] 5
44 Ops.CAST dtypes.i32 [43] dtypes.i32
45 Ops.INDEX dtypes.i32 [38, 44]
46 Ops.LOAD dtypes.i32 [45]
47 Ops.SUB dtypes.i32 [46, 42]
48 Ops.RANGE dtypes.i32 [47] (AxisType.WEAK, 0)
49 Ops.ADD dtypes.i32 [42, 48]
50 Ops.SHL dtypes.i32 [49, 9]
51 Ops.AFTER dtypes.i32 [12, 48]
52 Ops.RANGE dtypes.i32 [51] (AxisType.WEAK, 1)
53 Ops.ADD dtypes.i32 [50, 52]
54 Ops.INDEX dtypes.f32 [37, 53]
55 Ops.CONST dtypes.weakint [] 1
56 Ops.CAST dtypes.i32 [55] dtypes.i32
57 Ops.INDEX dtypes.u64 [32, 56]
58 Ops.LOAD dtypes.u64 [57]
59 Ops.REFPARAM dtypes.f32 [0, 58] ParamArg(slot=1, dtype=dtypes.f32)
60 Ops.INDEX dtypes.f32 [59, 53]
61 Ops.CMPLT dtypes.bool [52, 49]
62 Ops.CMPEQ dtypes.bool [61, 23]
63 Ops.LOAD dtypes.f32 [60, 20, 62]
64 Ops.STORE dtypes.void [54, 63]
65 Ops.END dtypes.void [64, 52]
66 Ops.END dtypes.void [65, 48]
67 Ops.SINK dtypes.void [66] KernelInfo(name="E_256_256", estimates=Estimates(ops=131072, lds=524288, mem=524288), optimized=True, split=(256, 2, 3))
68 Ops.LINEAR dtypes.void [30, 31, 39, 40, 38, 41, 42, 43, 44, 45, 46, 33, 34, 32, 35, 36, 55, 56, 57, 58, 0, 37, 59, 22, 23, 19, 20, 11, 12, 8, 9, 47, 48, 51, 49, 50, 52, 53, 60, 61, 62, 63, 54, 64, 65, 66, 67]
69 Ops.SOURCE dtypes.void [] "\n__attribute__((min_vector_width(512))) void E_256_256(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = (val0+Lidx0);\n    for (int Lidx1 = 0; Lidx1 < 256; Lidx1++) {\n      int alu1 = ((alu0<<8)+Lidx1);\n      float val4 = (((Lidx1<alu0)==0)?*(((float*)(val3))+alu1):0.0f);\n      *(((float*)(val2))+alu1) = val4;\n    }\n  }\n}\n"
70 Ops.PROGRAM dtypes.void [67, 68, 69] ProgramInfo(global_size=(256, 1, 1), params=(%1, %17, %3, %4), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
71 Ops.SINK dtypes.void [29, 70]

