# tolk D80 D151 golden=20f5d2861d0f83a2cbb4d5c5509bf15d
0 Ops.CONST dtypes.weakint [] 4
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.STACK dtypes.void []
3 Ops.PARAM dtypes.i32 [2] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 4), name="block_lo", addrspace=AddrSpace.ALU)
4 Ops.PARAM dtypes.i32 [2] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 4), name="block_hi", addrspace=AddrSpace.ALU)
5 Ops.SUB dtypes.i32 [4, 3]
6 Ops.RANGE dtypes.i32 [5] (AxisType.WEAK, 0)
7 Ops.ADD dtypes.i32 [3, 6]
8 Ops.INDEX dtypes.f32 [1, 7]
9 Ops.CONST dtypes.weakint [] 1
10 Ops.ALLOC dtypes.f32 [9] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
11 Ops.AFTER dtypes.f32 [10, 7]
12 Ops.CONST dtypes.weakint [] 0
13 Ops.CAST dtypes.i32 [12] dtypes.i32
14 Ops.INDEX dtypes.f32 [11, 13]
15 Ops.CONST dtypes.weakfloat [] 0.0
16 Ops.CAST dtypes.f32 [15] dtypes.f32
17 Ops.STORE dtypes.void [14, 16]
18 Ops.CONST dtypes.weakint [] 8
19 Ops.CAST dtypes.i32 [18] dtypes.i32
20 Ops.AFTER dtypes.i32 [19, 6]
21 Ops.RANGE dtypes.i32 [20] (AxisType.WEAK, 1)
22 Ops.AFTER dtypes.f32 [10, 17, 21]
23 Ops.INDEX dtypes.f32 [22, 13]
24 Ops.LOAD dtypes.f32 [23]
25 Ops.CONST dtypes.weakint [] 32
26 Ops.PARAM dtypes.f32 [25] ParamArg(slot=1, dtype=dtypes.f32)
27 Ops.CONST dtypes.weakint [] 3
28 Ops.CAST dtypes.i32 [27] dtypes.i32
29 Ops.SHL dtypes.i32 [7, 28]
30 Ops.ADD dtypes.i32 [29, 21]
31 Ops.INDEX dtypes.f32 [26, 30]
32 Ops.LOAD dtypes.f32 [31]
33 Ops.ADD dtypes.f32 [24, 32]
34 Ops.STORE dtypes.void [23, 33]
35 Ops.END dtypes.void [34, 21]
36 Ops.AFTER dtypes.f32 [10, 35]
37 Ops.INDEX dtypes.f32 [36, 13]
38 Ops.LOAD dtypes.f32 [37]
39 Ops.BUFFER dtypes.f32 [9] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
40 Ops.AFTER dtypes.f32 [39, 7]
41 Ops.INDEX dtypes.f32 [39, 13]
42 Ops.INDEX dtypes.f32 [26, 29]
43 Ops.LOAD dtypes.f32 [42]
44 Ops.STORE dtypes.void [41, 43]
45 Ops.AFTER dtypes.f32 [40, 44]
46 Ops.INDEX dtypes.f32 [45, 13]
47 Ops.LOAD dtypes.f32 [46]
48 Ops.ADD dtypes.f32 [38, 47]
49 Ops.STORE dtypes.void [8, 48]
50 Ops.END dtypes.void [49, 6]
51 Ops.SINK dtypes.void [50] KernelInfo(name="r_4_8", optimized=True, split=(4, 2, 3))
52 Ops.PARAM dtypes.u8 [25] ParamArg(slot=0, dtype=dtypes.u8, name="args")
53 Ops.BITCAST dtypes.u64 [52] dtypes.u64
54 Ops.INDEX dtypes.u64 [53, 13]
55 Ops.LOAD dtypes.u64 [54]
56 Ops.REFPARAM dtypes.f32 [0, 55] ParamArg(slot=0, dtype=dtypes.f32)
57 Ops.BITCAST dtypes.i32 [52] dtypes.i32
58 Ops.CAST dtypes.i32 [0] dtypes.i32
59 Ops.INDEX dtypes.i32 [57, 58]
60 Ops.LOAD dtypes.i32 [59]
61 Ops.CONST dtypes.weakint [] 5
62 Ops.CAST dtypes.i32 [61] dtypes.i32
63 Ops.INDEX dtypes.i32 [57, 62]
64 Ops.LOAD dtypes.i32 [63]
65 Ops.SUB dtypes.i32 [64, 60]
66 Ops.RANGE dtypes.i32 [65] (AxisType.WEAK, 0)
67 Ops.ADD dtypes.i32 [60, 66]
68 Ops.INDEX dtypes.f32 [56, 67]
69 Ops.BUFFER dtypes.f32 [9] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
70 Ops.AFTER dtypes.f32 [69, 67]
71 Ops.INDEX dtypes.f32 [70, 13]
72 Ops.STORE dtypes.void [71, 16]
73 Ops.AFTER dtypes.i32 [19, 66]
74 Ops.RANGE dtypes.i32 [73] (AxisType.WEAK, 1)
75 Ops.AFTER dtypes.f32 [69, 72, 74]
76 Ops.INDEX dtypes.f32 [75, 13]
77 Ops.LOAD dtypes.f32 [76]
78 Ops.CAST dtypes.i32 [9] dtypes.i32
79 Ops.INDEX dtypes.u64 [53, 78]
80 Ops.LOAD dtypes.u64 [79]
81 Ops.REFPARAM dtypes.f32 [25, 80] ParamArg(slot=1, dtype=dtypes.f32)
82 Ops.SHL dtypes.i32 [67, 28]
83 Ops.ADD dtypes.i32 [82, 74]
84 Ops.INDEX dtypes.f32 [81, 83]
85 Ops.LOAD dtypes.f32 [84]
86 Ops.ADD dtypes.f32 [77, 85]
87 Ops.STORE dtypes.void [76, 86]
88 Ops.END dtypes.void [87, 74]
89 Ops.AFTER dtypes.f32 [69, 88]
90 Ops.INDEX dtypes.f32 [89, 13]
91 Ops.LOAD dtypes.f32 [90]
92 Ops.AFTER dtypes.f32 [39, 67]
93 Ops.INDEX dtypes.f32 [81, 82]
94 Ops.LOAD dtypes.f32 [93]
95 Ops.STORE dtypes.void [41, 94]
96 Ops.AFTER dtypes.f32 [92, 95]
97 Ops.INDEX dtypes.f32 [96, 13]
98 Ops.LOAD dtypes.f32 [97]
99 Ops.ADD dtypes.f32 [91, 98]
100 Ops.STORE dtypes.void [68, 99]
101 Ops.END dtypes.void [100, 66]
102 Ops.SINK dtypes.void [101] KernelInfo(name="r_4_8", estimates=Estimates(ops=36, lds=160, mem=144), optimized=True, split=(4, 2, 3))
103 Ops.LINEAR dtypes.void [25, 52, 9, 39, 69, 0, 58, 57, 59, 60, 61, 62, 63, 64, 12, 13, 53, 54, 55, 78, 79, 80, 56, 81, 41, 15, 16, 27, 28, 18, 19, 65, 66, 73, 67, 70, 71, 82, 72, 74, 83, 84, 85, 75, 76, 77, 86, 87, 88, 93, 94, 89, 90, 91, 92, 95, 96, 97, 98, 68, 99, 100, 101, 102]
104 Ops.SOURCE dtypes.void [] "\n__attribute__((min_vector_width(512))) void r_4_8(unsigned char* restrict args_32) {\n  float buf0[1];\n  float buf1[1];\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx0 = 0; Lidx0 < (val1-val0); Lidx0++) {\n    int alu0 = (val0+Lidx0);\n    int alu1 = (alu0<<3);\n    *(buf1+0) = 0.0f;\n    for (int Lidx1 = 0; Lidx1 < 8; Lidx1++) {\n      float val4 = (*(((float*)(val3))+(alu1+Lidx1)));\n      *(buf1+0) = ((*(buf1+0))+val4);\n    }\n    float val5 = (*(((float*)(val3))+alu1));\n    *(buf0+0) = val5;\n    *(((float*)(val2))+alu0) = ((*(buf1+0))+(*(buf0+0)));\n  }\n}\n"
105 Ops.PROGRAM dtypes.void [102, 103, 104] ProgramInfo(global_size=(4, 1, 1), params=(%1, %26, %3, %4), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
106 Ops.SINK dtypes.void [51, 105]

