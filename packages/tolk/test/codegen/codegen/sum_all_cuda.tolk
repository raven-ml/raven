# tolk D89 golden=ac59ad27af49142c253c8c718e7acc79
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
12 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
13 Ops.SPECIAL dtypes.i32 [11] "lidx0"
14 Ops.AFTER dtypes.f32 [12, 13]
15 Ops.INDEX dtypes.f32 [14, 3]
16 Ops.STORE dtypes.void [15, 8]
17 Ops.CONST dtypes.weakint [] 256
18 Ops.CAST dtypes.i32 [17] dtypes.i32
19 Ops.RANGE dtypes.i32 [18] (AxisType.WEAK, 0)
20 Ops.AFTER dtypes.f32 [12, 16, 19]
21 Ops.INDEX dtypes.f32 [20, 3]
22 Ops.LOAD dtypes.f32 [21]
23 Ops.CONST dtypes.weakint [] 4096
24 Ops.PARAM dtypes.f32 [23] ParamArg(slot=1, dtype=dtypes.f32)
25 Ops.CONST dtypes.weakint [] 4
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
57 Ops.SINK dtypes.void [56] KernelInfo(name="r_16_256", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.LOCAL)),), optimized=True)
58 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, addrspace=AddrSpace.REG)
59 Ops.INDEX dtypes.f32 [58, 3]
60 Ops.STORE dtypes.void [59, 8]
61 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
62 Ops.AFTER dtypes.f32 [61, 13]
63 Ops.INDEX dtypes.f32 [62, 3]
64 Ops.STORE dtypes.void [63, 8]
65 Ops.AFTER dtypes.f32 [61, 64, 19]
66 Ops.INDEX dtypes.f32 [65, 3]
67 Ops.LOAD dtypes.f32 [66]
68 Ops.ADD dtypes.f32 [67, 30]
69 Ops.STORE dtypes.void [66, 68]
70 Ops.END dtypes.void [69, 19]
71 Ops.AFTER dtypes.i32 [11, 70]
72 Ops.RANGE dtypes.i32 [71] (AxisType.WEAK, 101)
73 Ops.AFTER dtypes.f32 [58, 60, 72]
74 Ops.INDEX dtypes.f32 [73, 3]
75 Ops.LOAD dtypes.f32 [74]
76 Ops.BUFFER dtypes.f32 [10] ParamArg(slot=2, dtype=dtypes.f32, addrspace=AddrSpace.LOCAL)
77 Ops.INDEX dtypes.f32 [76, 13]
78 Ops.AFTER dtypes.f32 [61, 70]
79 Ops.INDEX dtypes.f32 [78, 3]
80 Ops.LOAD dtypes.f32 [79]
81 Ops.STORE dtypes.void [77, 80]
82 Ops.BARRIER dtypes.void [81]
83 Ops.AFTER dtypes.f32 [76, 82]
84 Ops.INDEX dtypes.f32 [83, 72]
85 Ops.LOAD dtypes.f32 [84]
86 Ops.ADD dtypes.f32 [75, 85]
87 Ops.STORE dtypes.void [74, 86]
88 Ops.END dtypes.void [87, 72]
89 Ops.AFTER dtypes.f32 [58, 88]
90 Ops.INDEX dtypes.f32 [89, 3]
91 Ops.LOAD dtypes.f32 [90]
92 Ops.STORE dtypes.void [4, 91]
93 Ops.SINK dtypes.void [92] KernelInfo(name="r_16_256", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.LOCAL)),), estimates=Estimates(ops=4368, lds=17536, mem=16388), optimized=True)
94 Ops.IF dtypes.void [55, 4]
95 Ops.ENDIF dtypes.void [94]
96 Ops.LINEAR dtypes.void [0, 1, 23, 24, 61, 58, 10, 76, 11, 13, 62, 2, 3, 63, 7, 8, 17, 18, 25, 26, 64, 19, 27, 28, 29, 30, 65, 66, 67, 68, 69, 70, 78, 79, 80, 71, 77, 81, 82, 83, 59, 60, 72, 73, 74, 75, 84, 85, 86, 87, 88, 89, 90, 91, 4, 55, 94, 92, 95, 93]
97 Ops.SOURCE dtypes.void [] "typedef unsigned int uint;\n#define INFINITY (__int_as_float(0x7f800000))\n#define NAN (__int_as_float(0x7fffffff))\ntemplate <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }\nextern \"C\" __global__ void __launch_bounds__(16) r_16_256(float* data0_1, float* data1_4096) {\n  float buf0[1];\n  float buf1[1];\n  __shared__ __align__(16) float buf2[16];\n  int lidx0 = threadIdx.x; /* 16 */\n  *(buf0+0) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 256; Lidx0++) {\n    float val0 = (*(data1_4096+((Lidx0<<4)+lidx0)));\n    *(buf0+0) = ((*(buf0+0))+val0);\n  }\n  *(buf2+lidx0) = (*(buf0+0));\n  __syncthreads();\n  *(buf1+0) = 0.0f;\n  for (int Lidx101 = 0; Lidx101 < 16; Lidx101++) {\n    float val1 = (*(buf2+Lidx101));\n    *(buf1+0) = ((*(buf1+0))+val1);\n  }\n  if ((lidx0==0)) {\n    *(data0_1+0) = (*(buf1+0));\n  }\n}"
98 Ops.PROGRAM dtypes.void [93, 96, 97] ProgramInfo(local_size=(16, 1, 1), params=(%1, %24), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CUDA", renderer="CUDA", arch="sm_89"))
99 Ops.SINK dtypes.void [57, 98]

