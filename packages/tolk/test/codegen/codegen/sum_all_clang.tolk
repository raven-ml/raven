# tolk D151 D480 golden=9e4d6d723f0a0e50118c4a0571a29f42
0 Ops.CONST dtypes.weakint [] 1
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.INDEX dtypes.f32 [1, 3]
5 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
6 Ops.INDEX dtypes.f32 [5, 3]
7 Ops.CONST dtypes.weakfloat [] 0.0
8 Ops.CAST dtypes.f32 [7] dtypes.f32
9 Ops.STORE dtypes.void [6, 8]
10 Ops.CONST dtypes.weakint [] 16
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.RANGE dtypes.i32 [11] (AxisType.WEAK, 0)
13 Ops.AFTER dtypes.f32 [5, 9, 12]
14 Ops.INDEX dtypes.f32 [13, 3]
15 Ops.LOAD dtypes.f32 [14]
16 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
17 Ops.AFTER dtypes.f32 [16, 12]
18 Ops.INDEX dtypes.f32 [17, 3]
19 Ops.STORE dtypes.void [18, 8]
20 Ops.CONST dtypes.weakint [] 64
21 Ops.CAST dtypes.i32 [20] dtypes.i32
22 Ops.AFTER dtypes.i32 [21, 12]
23 Ops.RANGE dtypes.i32 [22] (AxisType.WEAK, 0, 256)
24 Ops.AFTER dtypes.f32 [16, 19, 23]
25 Ops.INDEX dtypes.f32 [24, 3]
26 Ops.LOAD dtypes.f32 [25]
27 Ops.CONST dtypes.weakint [] 4096
28 Ops.PARAM dtypes.f32 [27] ParamArg(slot=1, dtype=dtypes.f32)
29 Ops.CONST dtypes.weakint [] 8
30 Ops.CAST dtypes.i32 [29] dtypes.i32
31 Ops.SHL dtypes.i32 [12, 30]
32 Ops.CONST dtypes.weakint [] 2
33 Ops.CAST dtypes.i32 [32] dtypes.i32
34 Ops.SHL dtypes.i32 [23, 33]
35 Ops.ADD dtypes.i32 [31, 34]
36 Ops.CONST dtypes.weakint [] 4
37 Ops.CAST dtypes.i32 [36] dtypes.i32
38 Ops.SHRINK dtypes.f32 [28, 35, 37]
39 Ops.LOAD dtypes.f32 [38]
40 Ops.INDEX dtypes.f32 [39, 3]
41 Ops.CAST dtypes.i32 [0] dtypes.i32
42 Ops.INDEX dtypes.f32 [39, 41]
43 Ops.ADD dtypes.f32 [40, 42]
44 Ops.INDEX dtypes.f32 [39, 33]
45 Ops.ADD dtypes.f32 [43, 44]
46 Ops.CONST dtypes.weakint [] 3
47 Ops.CAST dtypes.i32 [46] dtypes.i32
48 Ops.INDEX dtypes.f32 [39, 47]
49 Ops.ADD dtypes.f32 [45, 48]
50 Ops.ADD dtypes.f32 [26, 49]
51 Ops.STORE dtypes.void [25, 50]
52 Ops.END dtypes.void [51, 23]
53 Ops.AFTER dtypes.f32 [16, 52]
54 Ops.INDEX dtypes.f32 [53, 3]
55 Ops.LOAD dtypes.f32 [54]
56 Ops.ADD dtypes.f32 [15, 55]
57 Ops.STORE dtypes.void [14, 56]
58 Ops.END dtypes.void [57, 12]
59 Ops.AFTER dtypes.f32 [5, 58]
60 Ops.INDEX dtypes.f32 [59, 3]
61 Ops.LOAD dtypes.f32 [60]
62 Ops.STORE dtypes.void [4, 61]
63 Ops.SINK dtypes.void [62] KernelInfo(name="r_1024_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
64 Ops.PARAM dtypes.u8 [10] ParamArg(slot=0, dtype=dtypes.u8, name="args")
65 Ops.BITCAST dtypes.u64 [64] dtypes.u64
66 Ops.INDEX dtypes.u64 [65, 3]
67 Ops.LOAD dtypes.u64 [66]
68 Ops.REFPARAM dtypes.f32 [0, 67] ParamArg(slot=0, dtype=dtypes.f32)
69 Ops.INDEX dtypes.f32 [68, 3]
70 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
71 Ops.INDEX dtypes.f32 [70, 3]
72 Ops.STORE dtypes.void [71, 8]
73 Ops.AFTER dtypes.f32 [70, 72, 12]
74 Ops.INDEX dtypes.f32 [73, 3]
75 Ops.LOAD dtypes.f32 [74]
76 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
77 Ops.AFTER dtypes.f32 [76, 12]
78 Ops.INDEX dtypes.f32 [77, 3]
79 Ops.STORE dtypes.void [78, 8]
80 Ops.AFTER dtypes.f32 [76, 79, 23]
81 Ops.INDEX dtypes.f32 [80, 3]
82 Ops.LOAD dtypes.f32 [81]
83 Ops.INDEX dtypes.u64 [65, 41]
84 Ops.LOAD dtypes.u64 [83]
85 Ops.REFPARAM dtypes.f32 [27, 84] ParamArg(slot=1, dtype=dtypes.f32)
86 Ops.SHRINK dtypes.f32 [85, 35, 37]
87 Ops.LOAD dtypes.f32 [86]
88 Ops.INDEX dtypes.f32 [87, 3]
89 Ops.INDEX dtypes.f32 [87, 41]
90 Ops.ADD dtypes.f32 [88, 89]
91 Ops.INDEX dtypes.f32 [87, 33]
92 Ops.ADD dtypes.f32 [90, 91]
93 Ops.INDEX dtypes.f32 [87, 47]
94 Ops.ADD dtypes.f32 [92, 93]
95 Ops.ADD dtypes.f32 [82, 94]
96 Ops.STORE dtypes.void [81, 95]
97 Ops.END dtypes.void [96, 23]
98 Ops.AFTER dtypes.f32 [76, 97]
99 Ops.INDEX dtypes.f32 [98, 3]
100 Ops.LOAD dtypes.f32 [99]
101 Ops.ADD dtypes.f32 [75, 100]
102 Ops.STORE dtypes.void [74, 101]
103 Ops.END dtypes.void [102, 12]
104 Ops.AFTER dtypes.f32 [70, 103]
105 Ops.INDEX dtypes.f32 [104, 3]
106 Ops.LOAD dtypes.f32 [105]
107 Ops.STORE dtypes.void [69, 106]
108 Ops.SINK dtypes.void [107] KernelInfo(name="r_1024_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), estimates=Estimates(ops=4112, lds=16388, mem=16388), optimized=True)
109 Ops.LINEAR dtypes.void [10, 64, 0, 76, 70, 41, 65, 83, 84, 27, 85, 2, 3, 71, 7, 8, 11, 32, 33, 46, 47, 36, 37, 20, 21, 29, 30, 72, 12, 22, 77, 78, 31, 79, 23, 80, 81, 82, 34, 35, 86, 87, 88, 89, 91, 93, 90, 92, 94, 95, 96, 97, 98, 99, 100, 73, 74, 75, 101, 102, 103, 66, 67, 104, 105, 106, 68, 69, 107, 108]
110 Ops.SOURCE dtypes.void [] "typedef float float4 __attribute__((aligned(4),ext_vector_type(4)));\n__attribute__((min_vector_width(512))) void r_1024_4(unsigned char* restrict args_16) {\n  float buf0[1];\n  float buf1[1];\n  unsigned long val0 = (*(((unsigned long*)(args_16))+1));\n  *(buf1+0) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 16; Lidx0++) {\n    *(buf0+0) = 0.0f;\n    for (int Lidx0_256 = 0; Lidx0_256 < 64; Lidx0_256++) {\n      float4 val1 = (*((float4*)((((float*)(val0))+((Lidx0<<8)+(Lidx0_256<<2))))));\n      *(buf0+0) = ((*(buf0+0))+(val1[0]+val1[1]+val1[2]+val1[3]));\n    }\n    *(buf1+0) = ((*(buf1+0))+(*(buf0+0)));\n  }\n  unsigned long val2 = (*(((unsigned long*)(args_16))+0));\n  *(((float*)(val2))+0) = (*(buf1+0));\n}\n"
111 Ops.PROGRAM dtypes.void [108, 109, 110] ProgramInfo(params=(%1, %28), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CPU", renderer="CLANG", arch="x86_64,x86-64"))
112 Ops.SINK dtypes.void [63, 111]

