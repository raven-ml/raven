# tolk D89 D487 golden=7cfe9862cb209b39f6ac10e061e1d3d8
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
10 Ops.CONST dtypes.weakint [] 32
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
13 Ops.SPECIAL dtypes.i32 [11] "lidx0"
14 Ops.AFTER dtypes.f32 [12, 13]
15 Ops.INDEX dtypes.f32 [14, 3]
16 Ops.STORE dtypes.void [15, 8]
17 Ops.CONST dtypes.weakint [] 128
18 Ops.CAST dtypes.i32 [17] dtypes.i32
19 Ops.RANGE dtypes.i32 [18] (AxisType.WEAK, 0)
20 Ops.AFTER dtypes.f32 [12, 16, 19]
21 Ops.INDEX dtypes.f32 [20, 3]
22 Ops.LOAD dtypes.f32 [21]
23 Ops.CONST dtypes.weakint [] 4096
24 Ops.PARAM dtypes.f32 [23] ParamArg(slot=1, dtype=dtypes.f32)
25 Ops.CONST dtypes.weakint [] 5
26 Ops.CAST dtypes.i32 [25] dtypes.i32
27 Ops.SHL dtypes.i32 [19, 26]
28 Ops.ADD dtypes.i32 [27, 13]
29 Ops.INDEX dtypes.f32 [24, 28]
30 Ops.LOAD dtypes.f32 [29]
31 Ops.ADD dtypes.f32 [22, 30]
32 Ops.STORE dtypes.void [21, 31]
33 Ops.END dtypes.void [32, 19]
34 Ops.AFTER dtypes.i32 [11, 33]
35 Ops.RANGE dtypes.i32 [34] (AxisType.WEAK, 101)
36 Ops.AFTER dtypes.f32 [5, 9, 35]
37 Ops.INDEX dtypes.f32 [36, 3]
38 Ops.LOAD dtypes.f32 [37]
39 Ops.ALLOC dtypes.f32 [10] ParamArg(slot=2, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
40 Ops.INDEX dtypes.f32 [39, 13]
41 Ops.AFTER dtypes.f32 [12, 33]
42 Ops.INDEX dtypes.f32 [41, 3]
43 Ops.LOAD dtypes.f32 [42]
44 Ops.STORE dtypes.void [40, 43]
45 Ops.BARRIER dtypes.void [44]
46 Ops.AFTER dtypes.f32 [39, 45]
47 Ops.INDEX dtypes.f32 [46, 35]
48 Ops.LOAD dtypes.f32 [47]
49 Ops.ADD dtypes.f32 [38, 48]
50 Ops.STORE dtypes.void [37, 49]
51 Ops.END dtypes.void [50, 35]
52 Ops.AFTER dtypes.f32 [5, 51]
53 Ops.INDEX dtypes.f32 [52, 3]
54 Ops.LOAD dtypes.f32 [53]
55 Ops.CMPEQ dtypes.bool [13, 3]
56 Ops.STORE dtypes.void [4, 54, 55]
57 Ops.SINK dtypes.void [56] KernelInfo(name="r_32_128", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(32, AxisType.LOCAL)),), optimized=True)
58 Ops.CONST dtypes.weakint [] 16
59 Ops.PARAM dtypes.u8 [58] ParamArg(slot=0, dtype=dtypes.u8, name="args")
60 Ops.BITCAST dtypes.u64 [59] dtypes.u64
61 Ops.INDEX dtypes.u64 [60, 3]
62 Ops.LOAD dtypes.u64 [61]
63 Ops.REFPARAM dtypes.f32 [0, 62] ParamArg(slot=0, dtype=dtypes.f32)
64 Ops.INDEX dtypes.f32 [63, 3]
65 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
66 Ops.INDEX dtypes.f32 [65, 3]
67 Ops.STORE dtypes.void [66, 8]
68 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
69 Ops.AFTER dtypes.f32 [68, 13]
70 Ops.INDEX dtypes.f32 [69, 3]
71 Ops.STORE dtypes.void [70, 8]
72 Ops.AFTER dtypes.f32 [68, 71, 19]
73 Ops.INDEX dtypes.f32 [72, 3]
74 Ops.LOAD dtypes.f32 [73]
75 Ops.CAST dtypes.i32 [0] dtypes.i32
76 Ops.INDEX dtypes.u64 [60, 75]
77 Ops.LOAD dtypes.u64 [76]
78 Ops.REFPARAM dtypes.f32 [23, 77] ParamArg(slot=1, dtype=dtypes.f32)
79 Ops.INDEX dtypes.f32 [78, 28]
80 Ops.LOAD dtypes.f32 [79]
81 Ops.ADD dtypes.f32 [74, 80]
82 Ops.STORE dtypes.void [73, 81]
83 Ops.END dtypes.void [82, 19]
84 Ops.AFTER dtypes.i32 [11, 83]
85 Ops.RANGE dtypes.i32 [84] (AxisType.WEAK, 101)
86 Ops.AFTER dtypes.f32 [65, 67, 85]
87 Ops.INDEX dtypes.f32 [86, 3]
88 Ops.LOAD dtypes.f32 [87]
89 Ops.BUFFER dtypes.f32 [10] ParamArg(slot=2, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
90 Ops.INDEX dtypes.f32 [89, 13]
91 Ops.AFTER dtypes.f32 [68, 83]
92 Ops.INDEX dtypes.f32 [91, 3]
93 Ops.LOAD dtypes.f32 [92]
94 Ops.STORE dtypes.void [90, 93]
95 Ops.BARRIER dtypes.void [94]
96 Ops.AFTER dtypes.f32 [89, 95]
97 Ops.INDEX dtypes.f32 [96, 85]
98 Ops.LOAD dtypes.f32 [97]
99 Ops.ADD dtypes.f32 [88, 98]
100 Ops.STORE dtypes.void [87, 99]
101 Ops.END dtypes.void [100, 85]
102 Ops.AFTER dtypes.f32 [65, 101]
103 Ops.INDEX dtypes.f32 [102, 3]
104 Ops.LOAD dtypes.f32 [103]
105 Ops.STORE dtypes.void [64, 104]
106 Ops.SINK dtypes.void [105] KernelInfo(name="r_32_128", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(32, AxisType.LOCAL)),), estimates=Estimates(ops=5152, lds=20736, mem=16388), optimized=True)
107 Ops.IF dtypes.void [55, 64]
108 Ops.ENDIF dtypes.void [107]
109 Ops.LINEAR dtypes.void [58, 59, 0, 68, 65, 10, 89, 75, 60, 76, 77, 11, 13, 23, 78, 69, 2, 3, 70, 7, 8, 17, 18, 25, 26, 71, 19, 27, 28, 79, 80, 72, 73, 74, 81, 82, 83, 91, 92, 93, 84, 90, 94, 95, 96, 66, 67, 85, 86, 87, 88, 97, 98, 99, 100, 101, 61, 62, 102, 103, 104, 63, 64, 55, 107, 105, 108, 106]
110 Ops.SOURCE dtypes.void [] "#include <metal_stdlib>\nusing namespace metal;\nkernel void r_32_128(device unsigned char* args_16, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n  float buf0[1];\n  float buf1[1];\n  threadgroup __attribute__((aligned(16))) float buf2[32];\n  unsigned long val0 = (*(((device unsigned long*)(args_16))+1));\n  int lidx0 = lid.x; /* 32 */\n  *(buf0+0) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 128; Lidx0++) {\n    float val1 = (*(((device float*)(val0))+((Lidx0<<5)+lidx0)));\n    *(buf0+0) = ((*(buf0+0))+val1);\n  }\n  *(buf2+lidx0) = (*(buf0+0));\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  *(buf1+0) = 0.0f;\n  for (int Lidx101 = 0; Lidx101 < 32; Lidx101++) {\n    float val2 = (*(buf2+Lidx101));\n    *(buf1+0) = ((*(buf1+0))+val2);\n  }\n  unsigned long val3 = (*(((device unsigned long*)(args_16))+0));\n  if ((lidx0==0)) {\n    *(((device float*)(val3))+0) = (*(buf1+0));\n  }\n}"
111 Ops.PROGRAM dtypes.void [106, 109, 110] ProgramInfo(local_size=(32, 1, 1), params=(%1, %24), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="METAL", renderer="METAL", arch="Apple9"))
112 Ops.SINK dtypes.void [57, 111]

