# tolk D80 D141 D151 golden=54151c51d2fcb37f579bcd79696bdbf7
0 Ops.CONST dtypes.weakint [] 3
1 Ops.CONST dtypes.weakint [] 40
2 Ops.PARAM dtypes.u8 [1] ParamArg(slot=0, dtype=dtypes.u8, name="args")
3 Ops.BITCAST dtypes.u64 [2] dtypes.u64
4 Ops.CONST dtypes.weakint [] 0
5 Ops.CAST dtypes.i32 [4] dtypes.i32
6 Ops.INDEX dtypes.u64 [3, 5]
7 Ops.LOAD dtypes.u64 [6]
8 Ops.REFPARAM dtypes.i32 [0, 7] ParamArg(slot=0, dtype=dtypes.i32)
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
21 Ops.INDEX dtypes.i32 [8, 20]
22 Ops.CONST dtypes.weakint [] 1
23 Ops.CAST dtypes.i32 [22] dtypes.i32
24 Ops.INDEX dtypes.u64 [3, 23]
25 Ops.LOAD dtypes.u64 [24]
26 Ops.REFPARAM dtypes.i32 [0, 25] ParamArg(slot=1, dtype=dtypes.i32)
27 Ops.INDEX dtypes.i32 [26, 20]
28 Ops.LOAD dtypes.i32 [27]
29 Ops.CAST dtypes.u32 [28] dtypes.u32
30 Ops.CONST dtypes.weakint [] 6
31 Ops.CAST dtypes.i32 [30] dtypes.i32
32 Ops.INDEX dtypes.i32 [9, 31]
33 Ops.LOAD dtypes.i32 [32]
34 Ops.CAST dtypes.u32 [33] dtypes.u32
35 Ops.MUL dtypes.u32 [29, 34]
36 Ops.CAST dtypes.i32 [35] dtypes.i32
37 Ops.STORE dtypes.void [21, 36]
38 Ops.END dtypes.void [37, 19]
39 Ops.SINK dtypes.void [38] KernelInfo(name="E_3", estimates=Estimates(ops=3, lds=24, mem=24), optimized=True, split=(3, 2, 3))
40 Ops.LINEAR dtypes.void [1, 2, 10, 11, 9, 12, 13, 14, 15, 16, 17, 30, 31, 32, 33, 4, 5, 3, 6, 7, 22, 23, 24, 25, 0, 8, 26, 34, 18, 19, 20, 27, 28, 21, 29, 35, 36, 37, 38, 39]
41 Ops.SOURCE dtypes.void [] "\n__attribute__((min_vector_width(512))) void E_3(unsigned char* restrict args_40) {\n  int val0 = (*(((int*)(args_40))+4));\n  int val1 = (*(((int*)(args_40))+5));\n  int val2 = (*(((int*)(args_40))+6));\n  unsigned long val3 = (*(((unsigned long*)(args_40))+0));\n  unsigned long val4 = (*(((unsigned long*)(args_40))+1));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = (val0+Lidx0);\n    int val5 = (*(((int*)(val4))+alu0));\n    *(((int*)(val3))+alu0) = ((int)((((unsigned int)(val5))*((unsigned int)(val2)))));\n  }\n}\n"
42 Ops.BINARY dtypes.u8 [] b"\n__attribute__((min_vector_width(512))) void E_3(unsigned char* restrict args_40) {\n  int val0 = (*(((int*)(args_40))+4));\n  int val1 = (*(((int*)(args_40))+5));\n  int val2 = (*(((int*)(args_40))+6));\n  unsigned long val3 = (*(((unsigned long*)(args_40))+0));\n  unsigned long val4 = (*(((unsigned long*)(args_40))+1));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = (val0+Lidx0);\n    int val5 = (*(((int*)(val4))+alu0));\n    *(((int*)(val3))+alu0) = ((int)((((unsigned int)(val5))*((unsigned int)(val2)))));\n  }\n}\n"
43 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
44 Ops.PARAM dtypes.i32 [0] ParamArg(slot=1, dtype=dtypes.i32)
45 Ops.STACK dtypes.void []
46 Ops.PARAM dtypes.i32 [45] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 3), name="block_lo", addrspace=AddrSpace.ALU)
47 Ops.PARAM dtypes.i32 [45] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 3), name="block_hi", addrspace=AddrSpace.ALU)
48 Ops.PARAM dtypes.i32 [45] ParamArg(slot=4, dtype=dtypes.i32, vmin_vmax=(1, 8), name="v", addrspace=AddrSpace.ALU)
49 Ops.PROGRAM dtypes.void [39, 40, 41, 42] ProgramInfo(global_size=(3, 1, 1), params=(%43, %44, %46, %47, %48), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
50 Ops.BUFFER dtypes.i32 [0] ParamArg(slot=2, dtype=dtypes.i32, device="CPU")
51 Ops.BUFFER dtypes.i32 [0] ParamArg(slot=3, dtype=dtypes.i32, device="CPU")
52 Ops.CALL dtypes.void [49, 50, 51] CallInfo()
53 Ops.LINEAR dtypes.void [52]

