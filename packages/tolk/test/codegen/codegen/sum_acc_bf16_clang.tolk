# tolk D151 D430 golden=e47b11410bebc293f8e621c6295ce484
0 Ops.CONST dtypes.weakint [] 1
1 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=0, dtype=dtypes.bf16)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.INDEX dtypes.bf16 [1, 3]
5 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
6 Ops.INDEX dtypes.f32 [5, 3]
7 Ops.CONST dtypes.weakfloat [] 0.0
8 Ops.CAST dtypes.f32 [7] dtypes.f32
9 Ops.STORE dtypes.void [6, 8]
10 Ops.CONST dtypes.weakint [] 3
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.RANGE dtypes.i32 [11] (AxisType.WEAK, 0)
13 Ops.AFTER dtypes.f32 [5, 9, 12]
14 Ops.INDEX dtypes.f32 [13, 3]
15 Ops.LOAD dtypes.f32 [14]
16 Ops.PARAM dtypes.bf16 [10] ParamArg(slot=1, dtype=dtypes.bf16)
17 Ops.INDEX dtypes.bf16 [16, 12]
18 Ops.LOAD dtypes.bf16 [17]
19 Ops.BITCAST dtypes.u16 [18] dtypes.u16
20 Ops.CAST dtypes.u32 [19] dtypes.u32
21 Ops.CONST dtypes.weakint [] 16
22 Ops.CAST dtypes.u32 [21] dtypes.u32
23 Ops.SHL dtypes.u32 [20, 22]
24 Ops.BITCAST dtypes.f32 [23] dtypes.f32
25 Ops.ADD dtypes.f32 [15, 24]
26 Ops.STORE dtypes.void [14, 25]
27 Ops.END dtypes.void [26, 12]
28 Ops.AFTER dtypes.f32 [5, 27]
29 Ops.INDEX dtypes.f32 [28, 3]
30 Ops.LOAD dtypes.f32 [29]
31 Ops.CMPNE dtypes.bool [30, 30]
32 Ops.BITCAST dtypes.u32 [30] dtypes.u32
33 Ops.CONST dtypes.weakint [] 4194304
34 Ops.CAST dtypes.u32 [33] dtypes.u32
35 Ops.OR dtypes.u32 [32, 34]
36 Ops.SHR dtypes.u32 [32, 22]
37 Ops.CAST dtypes.u32 [0] dtypes.u32
38 Ops.AND dtypes.u32 [36, 37]
39 Ops.ADD dtypes.u32 [32, 38]
40 Ops.CONST dtypes.weakint [] 32767
41 Ops.CAST dtypes.u32 [40] dtypes.u32
42 Ops.ADD dtypes.u32 [39, 41]
43 Ops.WHERE dtypes.u32 [31, 35, 42]
44 Ops.SHR dtypes.u32 [43, 22]
45 Ops.CAST dtypes.u16 [44] dtypes.u16
46 Ops.BITCAST dtypes.bf16 [45] dtypes.bf16
47 Ops.STORE dtypes.void [4, 46]
48 Ops.SINK dtypes.void [47] KernelInfo(name="r_3", optimized=True)
49 Ops.PARAM dtypes.u8 [21] ParamArg(slot=0, dtype=dtypes.u8, name="args")
50 Ops.BITCAST dtypes.u64 [49] dtypes.u64
51 Ops.INDEX dtypes.u64 [50, 3]
52 Ops.LOAD dtypes.u64 [51]
53 Ops.REFPARAM dtypes.bf16 [0, 52] ParamArg(slot=0, dtype=dtypes.bf16)
54 Ops.INDEX dtypes.bf16 [53, 3]
55 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
56 Ops.INDEX dtypes.f32 [55, 3]
57 Ops.STORE dtypes.void [56, 8]
58 Ops.AFTER dtypes.f32 [55, 57, 12]
59 Ops.INDEX dtypes.f32 [58, 3]
60 Ops.LOAD dtypes.f32 [59]
61 Ops.CAST dtypes.i32 [0] dtypes.i32
62 Ops.INDEX dtypes.u64 [50, 61]
63 Ops.LOAD dtypes.u64 [62]
64 Ops.REFPARAM dtypes.bf16 [10, 63] ParamArg(slot=1, dtype=dtypes.bf16)
65 Ops.INDEX dtypes.bf16 [64, 12]
66 Ops.LOAD dtypes.bf16 [65]
67 Ops.BITCAST dtypes.u16 [66] dtypes.u16
68 Ops.CAST dtypes.u32 [67] dtypes.u32
69 Ops.SHL dtypes.u32 [68, 22]
70 Ops.BITCAST dtypes.f32 [69] dtypes.f32
71 Ops.ADD dtypes.f32 [60, 70]
72 Ops.STORE dtypes.void [59, 71]
73 Ops.END dtypes.void [72, 12]
74 Ops.AFTER dtypes.f32 [55, 73]
75 Ops.INDEX dtypes.f32 [74, 3]
76 Ops.LOAD dtypes.f32 [75]
77 Ops.CMPNE dtypes.bool [76, 76]
78 Ops.BITCAST dtypes.u32 [76] dtypes.u32
79 Ops.OR dtypes.u32 [78, 34]
80 Ops.SHR dtypes.u32 [78, 22]
81 Ops.AND dtypes.u32 [80, 37]
82 Ops.ADD dtypes.u32 [78, 81]
83 Ops.ADD dtypes.u32 [82, 41]
84 Ops.WHERE dtypes.u32 [77, 79, 83]
85 Ops.SHR dtypes.u32 [84, 22]
86 Ops.CAST dtypes.u16 [85] dtypes.u16
87 Ops.BITCAST dtypes.bf16 [86] dtypes.bf16
88 Ops.STORE dtypes.void [54, 87]
89 Ops.SINK dtypes.void [88] KernelInfo(name="r_3", estimates=Estimates(ops=14, lds=8, mem=8), optimized=True)
90 Ops.LINEAR dtypes.void [21, 49, 0, 55, 61, 50, 62, 63, 10, 64, 2, 3, 56, 7, 8, 11, 22, 57, 12, 65, 66, 58, 59, 60, 67, 68, 69, 70, 71, 72, 73, 51, 52, 74, 75, 76, 53, 54, 37, 40, 41, 33, 34, 78, 80, 81, 82, 83, 77, 79, 84, 85, 86, 87, 88, 89]
91 Ops.SOURCE dtypes.void [] "\n__attribute__((min_vector_width(512))) void r_3(unsigned char* restrict args_16) {\n  float buf0[1];\n  unsigned long val0 = (*(((unsigned long*)(args_16))+1));\n  *(buf0+0) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 3; Lidx0++) {\n    unsigned short val1 = (*(((unsigned short*)(val0))+Lidx0));\n    *(buf0+0) = ((*(buf0+0))+__builtin_bit_cast(float, (unsigned int)((((unsigned int)(__builtin_bit_cast(unsigned short, (unsigned short)(val1))))<<16u))));\n  }\n  unsigned long val2 = (*(((unsigned long*)(args_16))+0));\n  float val3 = (*(buf0+0));\n  unsigned int cast0 = __builtin_bit_cast(unsigned int, (float)(val3));\n  unsigned int alu3 = ((val3!=val3)?(cast0|4194304u):(cast0+((cast0>>16u)&1u)+32767u));\n  *(((unsigned short*)(val2))+0) = __builtin_bit_cast(unsigned short, (unsigned short)(((unsigned short)((alu3>>16u)))));\n}\n"
92 Ops.PROGRAM dtypes.void [89, 90, 91] ProgramInfo(params=(%1, %16), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
93 Ops.SINK dtypes.void [48, 92]

