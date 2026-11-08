# tolk D25 D80 D141 D151 golden=f3af4a014caf740623880ecf43e83bb1
0 Ops.CONST dtypes.weakint [] 10
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
19 Ops.RANGE dtypes.i32 [18] (AxisType.WEAK, 1)
20 Ops.ADD dtypes.i32 [13, 19]
21 Ops.INDEX dtypes.f32 [8, 20]
22 Ops.CONST dtypes.weakint [] 30
23 Ops.CONST dtypes.weakint [] 1
24 Ops.CAST dtypes.i32 [23] dtypes.i32
25 Ops.INDEX dtypes.u64 [3, 24]
26 Ops.LOAD dtypes.u64 [25]
27 Ops.REFPARAM dtypes.f32 [22, 26] ParamArg(slot=1, dtype=dtypes.f32)
28 Ops.CONST dtypes.weakint [] 3
29 Ops.CAST dtypes.i32 [28] dtypes.i32
30 Ops.MUL dtypes.i32 [20, 29]
31 Ops.INDEX dtypes.f32 [27, 30]
32 Ops.LOAD dtypes.f32 [31]
33 Ops.ADD dtypes.i32 [30, 24]
34 Ops.INDEX dtypes.f32 [27, 33]
35 Ops.LOAD dtypes.f32 [34]
36 Ops.ADD dtypes.f32 [32, 35]
37 Ops.CONST dtypes.weakint [] 2
38 Ops.CAST dtypes.i32 [37] dtypes.i32
39 Ops.ADD dtypes.i32 [30, 38]
40 Ops.INDEX dtypes.f32 [27, 39]
41 Ops.LOAD dtypes.f32 [40]
42 Ops.ADD dtypes.f32 [36, 41]
43 Ops.CONST dtypes.weakfloat [] 0.3333333432674408
44 Ops.CAST dtypes.f32 [43] dtypes.f32
45 Ops.MUL dtypes.f32 [42, 44]
46 Ops.STORE dtypes.void [21, 45]
47 Ops.END dtypes.void [46, 19]
48 Ops.STACK dtypes.void []
49 Ops.PARAM dtypes.i32 [48] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(1, 10), name="i", addrspace=AddrSpace.ALU)
50 Ops.SUB dtypes.i32 [49, 4]
51 Ops.MUL dtypes.i32 [50, 28]
52 Ops.CONST dtypes.weakint [] 16
53 Ops.MUL dtypes.i32 [50, 52]
54 Ops.SINK dtypes.void [47] KernelInfo(name="r_i_3", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST)),), estimates=Estimates(ops=%51, lds=%53, mem=%53), optimized=True, split=(%49, 2, 3))
55 Ops.LINEAR dtypes.void [1, 2, 10, 11, 9, 12, 13, 14, 15, 16, 17, 4, 5, 3, 6, 7, 23, 24, 25, 26, 0, 8, 22, 27, 43, 44, 37, 38, 28, 29, 18, 19, 20, 30, 33, 34, 35, 39, 40, 41, 31, 32, 21, 36, 42, 45, 46, 47, 54]
56 Ops.SOURCE dtypes.void [] "\n__attribute__((min_vector_width(512))) void r_i_3(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    int alu1 = (alu0*3);\n    float val4 = (*(((float*)(val3))+(alu1+1)));\n    float val5 = (*(((float*)(val3))+(alu1+2)));\n    float val6 = (*(((float*)(val3))+alu1));\n    *(((float*)(val2))+alu0) = ((val6+val4+val5)*0.3333333432674408f);\n  }\n}\n"
57 Ops.BINARY dtypes.u8 [] b"\n__attribute__((min_vector_width(512))) void r_i_3(unsigned char* restrict args_32) {\n  int val0 = (*(((int*)(args_32))+4));\n  int val1 = (*(((int*)(args_32))+5));\n  unsigned long val2 = (*(((unsigned long*)(args_32))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_32))+1));\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    int alu1 = (alu0*3);\n    float val4 = (*(((float*)(val3))+(alu1+1)));\n    float val5 = (*(((float*)(val3))+(alu1+2)));\n    float val6 = (*(((float*)(val3))+alu1));\n    *(((float*)(val2))+alu0) = ((val6+val4+val5)*0.3333333432674408f);\n  }\n}\n"
58 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
59 Ops.PARAM dtypes.f32 [22] ParamArg(slot=1, dtype=dtypes.f32)
60 Ops.PARAM dtypes.i32 [48] ParamArg(slot=2, dtype=dtypes.i32, vmin_vmax=(0, 10), name="block_lo", addrspace=AddrSpace.ALU)
61 Ops.PARAM dtypes.i32 [48] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 10), name="block_hi", addrspace=AddrSpace.ALU)
62 Ops.PROGRAM dtypes.void [54, 55, 56, 57] ProgramInfo(global_size=(%49, 1, 1), params=(%58, %59, %60, %61), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", arch="x86_64,x86-64"))
63 Ops.CONST dtypes.weakint [] 256
64 Ops.ALLOC dtypes.i8 [63] ParamArg(slot=5, dtype=dtypes.i8, device="CPU")
65 Ops.CONST dtypes.weakint [] 40
66 Ops.SHRINK dtypes.i8 [64, 4, 65]
67 Ops.BITCAST dtypes.f32 [66] dtypes.f32
68 Ops.PARAM dtypes.f32 [22] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
69 Ops.CALL dtypes.void [62, 67, 68] CallInfo()
70 Ops.PARAM dtypes.u8 [65] ParamArg(slot=0, dtype=dtypes.u8, name="args")
71 Ops.BITCAST dtypes.u64 [70] dtypes.u64
72 Ops.INDEX dtypes.u64 [71, 5]
73 Ops.LOAD dtypes.u64 [72]
74 Ops.REFPARAM dtypes.f32 [0, 73] ParamArg(slot=0, dtype=dtypes.f32)
75 Ops.BITCAST dtypes.i32 [70] dtypes.i32
76 Ops.CONST dtypes.weakint [] 6
77 Ops.CAST dtypes.i32 [76] dtypes.i32
78 Ops.INDEX dtypes.i32 [75, 77]
79 Ops.LOAD dtypes.i32 [78]
80 Ops.CONST dtypes.weakint [] 7
81 Ops.CAST dtypes.i32 [80] dtypes.i32
82 Ops.INDEX dtypes.i32 [75, 81]
83 Ops.LOAD dtypes.i32 [82]
84 Ops.SUB dtypes.i32 [83, 79]
85 Ops.RANGE dtypes.i32 [84] (AxisType.WEAK, 1)
86 Ops.ADD dtypes.i32 [79, 85]
87 Ops.INDEX dtypes.f32 [74, 86]
88 Ops.INDEX dtypes.u64 [71, 24]
89 Ops.LOAD dtypes.u64 [88]
90 Ops.REFPARAM dtypes.f32 [22, 89] ParamArg(slot=1, dtype=dtypes.f32)
91 Ops.MUL dtypes.i32 [86, 29]
92 Ops.INDEX dtypes.f32 [90, 91]
93 Ops.LOAD dtypes.f32 [92]
94 Ops.ADD dtypes.i32 [91, 24]
95 Ops.INDEX dtypes.f32 [90, 94]
96 Ops.LOAD dtypes.f32 [95]
97 Ops.ADD dtypes.i32 [91, 38]
98 Ops.INDEX dtypes.f32 [90, 97]
99 Ops.LOAD dtypes.f32 [98]
100 Ops.STACK dtypes.f32 [93, 96, 99]
101 Ops.INDEX dtypes.u64 [71, 38]
102 Ops.LOAD dtypes.u64 [101]
103 Ops.REFPARAM dtypes.f32 [0, 102] ParamArg(slot=2, dtype=dtypes.f32)
104 Ops.INDEX dtypes.f32 [103, 86]
105 Ops.LOAD dtypes.f32 [104]
106 Ops.SUB dtypes.f32 [100, 105]
107 Ops.INDEX dtypes.f32 [106, 38]
108 Ops.INDEX dtypes.f32 [106, 24]
109 Ops.INDEX dtypes.f32 [106, 5]
110 Ops.MUL dtypes.f32 [109, 109]
111 Ops.MULACC dtypes.f32 [108, 108, 110]
112 Ops.MULACC dtypes.f32 [107, 107, 111]
113 Ops.CONST dtypes.weakfloat [] 0.5
114 Ops.CAST dtypes.f32 [113] dtypes.f32
115 Ops.MUL dtypes.f32 [112, 114]
116 Ops.STORE dtypes.void [87, 115]
117 Ops.END dtypes.void [116, 85]
118 Ops.CONST dtypes.weakint [] 9
119 Ops.MUL dtypes.i32 [50, 118]
120 Ops.CONST dtypes.weakint [] 20
121 Ops.MUL dtypes.i32 [50, 120]
122 Ops.SINK dtypes.void [117] KernelInfo(name="r_i_3", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST)),), estimates=Estimates(ops=%119, lds=%121, mem=%121), optimized=True, split=(%49, 3, 4))
123 Ops.LINEAR dtypes.void [65, 70, 76, 77, 75, 78, 79, 80, 81, 82, 83, 4, 5, 71, 72, 73, 23, 24, 88, 89, 37, 38, 101, 102, 0, 74, 22, 90, 103, 113, 114, 28, 29, 84, 85, 86, 91, 94, 95, 96, 97, 98, 99, 92, 93, 104, 105, 100, 87, 106, 109, 108, 107, 110, 111, 112, 115, 116, 117, 122]
124 Ops.SOURCE dtypes.void [] "typedef float float3 __attribute__((aligned(4),ext_vector_type(3)));\n__attribute__((min_vector_width(512))) void r_i_3(unsigned char* restrict args_40) {\n  int val0 = (*(((int*)(args_40))+6));\n  int val1 = (*(((int*)(args_40))+7));\n  unsigned long val2 = (*(((unsigned long*)(args_40))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_40))+1));\n  unsigned long val4 = (*(((unsigned long*)(args_40))+2));\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    int alu1 = (alu0*3);\n    float val5 = (*(((float*)(val3))+(alu1+1)));\n    float val6 = (*(((float*)(val3))+(alu1+2)));\n    float val7 = (*(((float*)(val3))+alu1));\n    float val8 = (*(((float*)(val4))+alu0));\n    float3 alu2 = ((float3){val7,val5,val6}-val8);\n    *(((float*)(val2))+alu0) = (__builtin_fmaf(alu2[2],alu2[2],__builtin_fmaf(alu2[1],alu2[1],(alu2[0]*alu2[0])))*0.5f);\n  }\n}\n"
125 Ops.BINARY dtypes.u8 [] b"typedef float float3 __attribute__((aligned(4),ext_vector_type(3)));\n__attribute__((min_vector_width(512))) void r_i_3(unsigned char* restrict args_40) {\n  int val0 = (*(((int*)(args_40))+6));\n  int val1 = (*(((int*)(args_40))+7));\n  unsigned long val2 = (*(((unsigned long*)(args_40))+0));\n  unsigned long val3 = (*(((unsigned long*)(args_40))+1));\n  unsigned long val4 = (*(((unsigned long*)(args_40))+2));\n  for (int Lidx1 = 0; Lidx1 < (val1-val0); Lidx1++) {\n    int alu0 = (val0+Lidx1);\n    int alu1 = (alu0*3);\n    float val5 = (*(((float*)(val3))+(alu1+1)));\n    float val6 = (*(((float*)(val3))+(alu1+2)));\n    float val7 = (*(((float*)(val3))+alu1));\n    float val8 = (*(((float*)(val4))+alu0));\n    float3 alu2 = ((float3){val7,val5,val6}-val8);\n    *(((float*)(val2))+alu0) = (__builtin_fmaf(alu2[2],alu2[2],__builtin_fmaf(alu2[1],alu2[1],(alu2[0]*alu2[0])))*0.5f);\n  }\n}\n"
126 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
127 Ops.PARAM dtypes.i32 [48] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 10), name="block_lo", addrspace=AddrSpace.ALU)
128 Ops.PARAM dtypes.i32 [48] ParamArg(slot=4, dtype=dtypes.i32, vmin_vmax=(0, 10), name="block_hi", addrspace=AddrSpace.ALU)
129 Ops.PROGRAM dtypes.void [122, 123, 124, 125] ProgramInfo(global_size=(%49, 1, 1), params=(%58, %59, %126, %127, %128), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CPU", arch="x86_64,x86-64"))
130 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
131 Ops.CALL dtypes.void [129, 130, 68, 67] CallInfo()
132 Ops.LINEAR dtypes.void [69, 131]

