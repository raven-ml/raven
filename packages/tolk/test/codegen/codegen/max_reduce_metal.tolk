# tolk D89 golden=ad40eee1aa705f1240740ace27de5b77
0 Ops.CONST dtypes.weakint [] 1
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.INDEX dtypes.f32 [1, 3]
5 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
6 Ops.INDEX dtypes.f32 [5, 3]
7 Ops.CONST dtypes.weakfloat [] -inf
8 Ops.CAST dtypes.f32 [7] dtypes.f32
9 Ops.STORE dtypes.void [6, 8]
10 Ops.CONST dtypes.weakint [] 16
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
13 Ops.SPECIAL dtypes.i32 [11] "lidx0"
14 Ops.AFTER dtypes.f32 [12, 13]
15 Ops.INDEX dtypes.f32 [14, 3]
16 Ops.STORE dtypes.void [15, 8]
17 Ops.CONST dtypes.weakint [] 4
18 Ops.CAST dtypes.i32 [17] dtypes.i32
19 Ops.RANGE dtypes.i32 [18] (AxisType.WEAK, 0)
20 Ops.AFTER dtypes.f32 [12, 16, 19]
21 Ops.INDEX dtypes.f32 [20, 3]
22 Ops.LOAD dtypes.f32 [21]
23 Ops.CONST dtypes.weakint [] 64
24 Ops.PARAM dtypes.f32 [23] ParamArg(slot=0, dtype=dtypes.f32)
25 Ops.SHL dtypes.i32 [19, 18]
26 Ops.ADD dtypes.i32 [25, 13]
27 Ops.INDEX dtypes.f32 [24, 26]
28 Ops.LOAD dtypes.f32 [27]
29 Ops.CMPLT dtypes.bool [22, 28]
30 Ops.WHERE dtypes.f32 [29, 28, 22]
31 Ops.STORE dtypes.void [21, 30]
32 Ops.END dtypes.void [31, 19]
33 Ops.AFTER dtypes.i32 [11, 32]
34 Ops.RANGE dtypes.i32 [33] (AxisType.WEAK, 101)
35 Ops.AFTER dtypes.f32 [5, 9, 34]
36 Ops.INDEX dtypes.f32 [35, 3]
37 Ops.LOAD dtypes.f32 [36]
38 Ops.ALLOC dtypes.f32 [10] ParamArg(slot=2, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
39 Ops.INDEX dtypes.f32 [38, 13]
40 Ops.AFTER dtypes.f32 [12, 32]
41 Ops.INDEX dtypes.f32 [40, 3]
42 Ops.LOAD dtypes.f32 [41]
43 Ops.STORE dtypes.void [39, 42]
44 Ops.BARRIER dtypes.void [43]
45 Ops.AFTER dtypes.f32 [38, 44]
46 Ops.INDEX dtypes.f32 [45, 34]
47 Ops.LOAD dtypes.f32 [46]
48 Ops.CMPLT dtypes.bool [37, 47]
49 Ops.WHERE dtypes.f32 [48, 47, 37]
50 Ops.STORE dtypes.void [36, 49]
51 Ops.END dtypes.void [50, 34]
52 Ops.AFTER dtypes.f32 [5, 51]
53 Ops.INDEX dtypes.f32 [52, 3]
54 Ops.LOAD dtypes.f32 [53]
55 Ops.CMPEQ dtypes.bool [13, 3]
56 Ops.STORE dtypes.void [4, 54, 55]
57 Ops.SINK dtypes.void [56] KernelInfo(name="max_reduce", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.LOCAL)),), optimized=True)
58 Ops.PARAM dtypes.u8 [10] ParamArg(slot=0, dtype=dtypes.u8, name="args")
59 Ops.BITCAST dtypes.u64 [58] dtypes.u64
60 Ops.CAST dtypes.i32 [0] dtypes.i32
61 Ops.INDEX dtypes.u64 [59, 60]
62 Ops.LOAD dtypes.u64 [61]
63 Ops.REFPARAM dtypes.f32 [0, 62] ParamArg(slot=1, dtype=dtypes.f32)
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
75 Ops.INDEX dtypes.u64 [59, 3]
76 Ops.LOAD dtypes.u64 [75]
77 Ops.REFPARAM dtypes.f32 [23, 76] ParamArg(slot=0, dtype=dtypes.f32)
78 Ops.INDEX dtypes.f32 [77, 26]
79 Ops.LOAD dtypes.f32 [78]
80 Ops.CMPLT dtypes.bool [74, 79]
81 Ops.WHERE dtypes.f32 [80, 79, 74]
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
99 Ops.CMPLT dtypes.bool [88, 98]
100 Ops.WHERE dtypes.f32 [99, 98, 88]
101 Ops.STORE dtypes.void [87, 100]
102 Ops.END dtypes.void [101, 85]
103 Ops.AFTER dtypes.f32 [65, 102]
104 Ops.INDEX dtypes.f32 [103, 3]
105 Ops.LOAD dtypes.f32 [104]
106 Ops.STORE dtypes.void [64, 105]
107 Ops.SINK dtypes.void [106] KernelInfo(name="max_reduce", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.LOCAL)),), estimates=Estimates(ops=656, lds=1408, mem=260), optimized=True)
108 Ops.IF dtypes.void [55, 64]
109 Ops.ENDIF dtypes.void [108]
110 Ops.LINEAR dtypes.void [10, 58, 0, 68, 65, 89, 2, 3, 59, 75, 76, 11, 13, 23, 77, 69, 70, 7, 8, 17, 18, 71, 19, 25, 26, 78, 79, 72, 73, 74, 80, 81, 82, 83, 91, 92, 93, 84, 90, 94, 95, 96, 66, 67, 85, 86, 87, 88, 97, 98, 99, 100, 101, 102, 60, 61, 62, 103, 104, 105, 63, 64, 55, 108, 106, 109, 107]
111 Ops.SOURCE dtypes.void [] "#include <metal_stdlib>\nusing namespace metal;\nkernel void max_reduce(device unsigned char* args_16, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n  float buf0[1];\n  float buf1[1];\n  threadgroup __attribute__((aligned(16))) float buf2[16];\n  unsigned long val0 = (*(((device unsigned long*)(args_16))+0));\n  int lidx0 = lid.x; /* 16 */\n  *(buf0+0) = ((float)(-INFINITY));\n  for (int Lidx0 = 0; Lidx0 < 4; Lidx0++) {\n    float val1 = (*(((device float*)(val0))+((Lidx0<<4)+lidx0)));\n    float val2 = (*(buf0+0));\n    float alu1 = ((val2<val1)?val1:val2);\n    *(buf0+0) = alu1;\n  }\n  *(buf2+lidx0) = (*(buf0+0));\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  *(buf1+0) = ((float)(-INFINITY));\n  for (int Lidx101 = 0; Lidx101 < 16; Lidx101++) {\n    float val3 = (*(buf1+0));\n    float val4 = (*(buf2+Lidx101));\n    float alu7 = ((val3<val4)?val4:val3);\n    *(buf1+0) = alu7;\n  }\n  unsigned long val5 = (*(((device unsigned long*)(args_16))+1));\n  if ((lidx0==0)) {\n    *(((device float*)(val5))+0) = (*(buf1+0));\n  }\n}"
112 Ops.PROGRAM dtypes.void [107, 110, 111] ProgramInfo(local_size=(16, 1, 1), params=(%24, %1), globals=(0, 1), outs=(1,), ins=(0,), target=Target(device="METAL", renderer="METAL", arch="Apple9"))
113 Ops.SINK dtypes.void [57, 112]

