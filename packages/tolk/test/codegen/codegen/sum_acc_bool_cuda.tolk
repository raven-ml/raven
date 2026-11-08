# tolk D89 D342 golden=e37f986f6971fc9da76779d739d1ebdb
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
15 Ops.PARAM dtypes.bool [8] ParamArg(slot=1, dtype=dtypes.bool)
16 Ops.INDEX dtypes.bool [15, 10]
17 Ops.LOAD dtypes.bool [16]
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
29 Ops.BUFFER dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32, addrspace=AddrSpace.REG)
30 Ops.INDEX dtypes.i32 [29, 3]
31 Ops.STORE dtypes.void [30, 3]
32 Ops.AFTER dtypes.i32 [29, 31, 10]
33 Ops.INDEX dtypes.i32 [32, 3]
34 Ops.LOAD dtypes.i32 [33]
35 Ops.CAST dtypes.u32 [34] dtypes.u32
36 Ops.ADD dtypes.u32 [35, 19]
37 Ops.CAST dtypes.i32 [36] dtypes.i32
38 Ops.STORE dtypes.void [33, 37]
39 Ops.END dtypes.void [38, 10]
40 Ops.AFTER dtypes.i32 [29, 39]
41 Ops.INDEX dtypes.i32 [40, 3]
42 Ops.LOAD dtypes.i32 [41]
43 Ops.STORE dtypes.void [4, 42]
44 Ops.SINK dtypes.void [43] KernelInfo(name="r_3", estimates=Estimates(ops=3, lds=7, mem=7), optimized=True)
45 Ops.LINEAR dtypes.void [0, 1, 8, 15, 29, 2, 3, 30, 9, 31, 10, 16, 17, 32, 33, 34, 18, 35, 19, 36, 37, 38, 39, 40, 41, 42, 4, 43, 44]
46 Ops.SOURCE dtypes.void [] "typedef unsigned int uint;\n#define INFINITY (__int_as_float(0x7f800000))\n#define NAN (__int_as_float(0x7fffffff))\ntemplate <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }\nextern \"C\" __global__ void __launch_bounds__(1) r_3(int* data0_1, bool* data1_3) {\n  int buf0[1];\n  *(buf0+0) = 0;\n  for (int Lidx0 = 0; Lidx0 < 3; Lidx0++) {\n    bool val0 = (*(data1_3+Lidx0));\n    *(buf0+0) = ((int)((((uint)((*(buf0+0))))+((uint)(((int)(val0)))))));\n  }\n  *(data0_1+0) = (*(buf0+0));\n}"
47 Ops.PROGRAM dtypes.void [44, 45, 46] ProgramInfo(params=(%1, %15), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CUDA", renderer="CUDA", arch="sm_89"))
48 Ops.SINK dtypes.void [28, 47]

