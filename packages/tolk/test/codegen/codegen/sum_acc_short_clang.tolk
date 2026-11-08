# tolk D151 D342 golden=7602978f9688d7b342ceb806e71b78bb
0 Ops.CONST dtypes.weakint [] 1
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.INDEX dtypes.i32 [1, 3]
5 Ops.ALLOC dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32, addrspace=AddrSpace.REG)
6 Ops.INDEX dtypes.i32 [5, 3]
7 Ops.STORE dtypes.void [6, 3]
8 Ops.CONST dtypes.weakint [] 3
9 Ops.CAST dtypes.i32 [8] dtypes.i32
10 Ops.RANGE dtypes.i32 [9] (AxisType.WEAK, 0)
11 Ops.AFTER dtypes.i32 [5, 7, 10]
12 Ops.INDEX dtypes.i32 [11, 3]
13 Ops.LOAD dtypes.i32 [12]
14 Ops.CAST dtypes.u32 [13] dtypes.u32
15 Ops.PARAM dtypes.i16 [8] ParamArg(slot=1, dtype=dtypes.i16)
16 Ops.INDEX dtypes.i16 [15, 10]
17 Ops.LOAD dtypes.i16 [16]
18 Ops.CAST dtypes.i32 [17] dtypes.i32
19 Ops.CAST dtypes.u32 [18] dtypes.u32
20 Ops.ADD dtypes.u32 [14, 19]
21 Ops.CAST dtypes.i32 [20] dtypes.i32
22 Ops.STORE dtypes.void [12, 21]
23 Ops.END dtypes.void [22, 10]
24 Ops.AFTER dtypes.i32 [5, 23]
25 Ops.INDEX dtypes.i32 [24, 3]
26 Ops.LOAD dtypes.i32 [25]
27 Ops.STORE dtypes.void [4, 26]
28 Ops.SINK dtypes.void [27] KernelInfo(name="r_3", optimized=True)
29 Ops.CONST dtypes.weakint [] 16
30 Ops.PARAM dtypes.u8 [29] ParamArg(slot=0, dtype=dtypes.u8, name="args")
31 Ops.BITCAST dtypes.u64 [30] dtypes.u64
32 Ops.INDEX dtypes.u64 [31, 3]
33 Ops.LOAD dtypes.u64 [32]
34 Ops.REFPARAM dtypes.i32 [0, 33] ParamArg(slot=0, dtype=dtypes.i32)
35 Ops.INDEX dtypes.i32 [34, 3]
36 Ops.BUFFER dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32, addrspace=AddrSpace.REG)
37 Ops.INDEX dtypes.i32 [36, 3]
38 Ops.STORE dtypes.void [37, 3]
39 Ops.AFTER dtypes.i32 [36, 38, 10]
40 Ops.INDEX dtypes.i32 [39, 3]
41 Ops.LOAD dtypes.i32 [40]
42 Ops.CAST dtypes.u32 [41] dtypes.u32
43 Ops.CAST dtypes.i32 [0] dtypes.i32
44 Ops.INDEX dtypes.u64 [31, 43]
45 Ops.LOAD dtypes.u64 [44]
46 Ops.REFPARAM dtypes.i16 [8, 45] ParamArg(slot=1, dtype=dtypes.i16)
47 Ops.INDEX dtypes.i16 [46, 10]
48 Ops.LOAD dtypes.i16 [47]
49 Ops.CAST dtypes.i32 [48] dtypes.i32
50 Ops.CAST dtypes.u32 [49] dtypes.u32
51 Ops.ADD dtypes.u32 [42, 50]
52 Ops.CAST dtypes.i32 [51] dtypes.i32
53 Ops.STORE dtypes.void [40, 52]
54 Ops.END dtypes.void [53, 10]
55 Ops.AFTER dtypes.i32 [36, 54]
56 Ops.INDEX dtypes.i32 [55, 3]
57 Ops.LOAD dtypes.i32 [56]
58 Ops.STORE dtypes.void [35, 57]
59 Ops.SINK dtypes.void [58] KernelInfo(name="r_3", estimates=Estimates(ops=3, lds=10, mem=10), optimized=True)
60 Ops.LINEAR dtypes.void [29, 30, 0, 36, 43, 31, 44, 45, 8, 46, 2, 3, 37, 9, 38, 10, 47, 48, 39, 40, 41, 49, 42, 50, 51, 52, 53, 54, 55, 56, 57, 32, 33, 34, 35, 58, 59]
61 Ops.SOURCE dtypes.void [] "\n__attribute__((min_vector_width(512))) void r_3(unsigned char* restrict args_16) {\n  int buf0[1];\n  unsigned long val0 = (*(((unsigned long*)(args_16))+1));\n  *(buf0+0) = 0;\n  for (int Lidx0 = 0; Lidx0 < 3; Lidx0++) {\n    short val1 = (*(((short*)(val0))+Lidx0));\n    *(buf0+0) = ((int)((((unsigned int)((*(buf0+0))))+((unsigned int)(((int)(val1)))))));\n  }\n  unsigned long val2 = (*(((unsigned long*)(args_16))+0));\n  *(((int*)(val2))+0) = (*(buf0+0));\n}\n"
62 Ops.PROGRAM dtypes.void [59, 60, 61] ProgramInfo(params=(%1, %15), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
63 Ops.SINK dtypes.void [28, 62]

