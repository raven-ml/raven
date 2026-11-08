# tolk D89 D342 golden=028e18b95cb8d6834708229a927799b1
0 Ops.CONST dtypes.weakint [] 256
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=2, dtype=dtypes.i32)
2 Ops.CAST dtypes.i32 [0] dtypes.i32
3 Ops.RANGE dtypes.i32 [2] (AxisType.LOOP, 0)
4 Ops.INDEX dtypes.i32 [1, 3]
5 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
6 Ops.INDEX dtypes.i32 [5, 3]
7 Ops.LOAD dtypes.i32 [6]
8 Ops.CAST dtypes.u32 [7] dtypes.u32
9 Ops.PARAM dtypes.i32 [0] ParamArg(slot=1, dtype=dtypes.i32)
10 Ops.INDEX dtypes.i32 [9, 3]
11 Ops.LOAD dtypes.i32 [10]
12 Ops.CAST dtypes.u32 [11] dtypes.u32
13 Ops.ADD dtypes.u32 [8, 12]
14 Ops.CAST dtypes.i32 [13] dtypes.i32
15 Ops.STORE dtypes.void [4, 14]
16 Ops.END dtypes.void [15, 3]
17 Ops.SINK dtypes.void [16] KernelInfo(name="elementwise_int32", optimized=True)
18 Ops.SINK dtypes.void [16] KernelInfo(name="elementwise_int32", estimates=Estimates(ops=256, lds=3072, mem=3072), optimized=True)
19 Ops.LINEAR dtypes.void [0, 5, 9, 1, 2, 3, 6, 7, 10, 11, 4, 8, 12, 13, 14, 15, 16, 18]
20 Ops.SOURCE dtypes.void [] "typedef unsigned int uint;\n#define INFINITY (__int_as_float(0x7f800000))\n#define NAN (__int_as_float(0x7fffffff))\ntemplate <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }\nextern \"C\" __global__ void __launch_bounds__(1) elementwise_int32(int* data0_256, int* data1_256, int* data2_256) {\n  for (int Lidx0 = 0; Lidx0 < 256; Lidx0++) {\n    int val0 = (*(data0_256+Lidx0));\n    int val1 = (*(data1_256+Lidx0));\n    *(data2_256+Lidx0) = ((int)((((uint)(val0))+((uint)(val1)))));\n  }\n}"
21 Ops.PROGRAM dtypes.void [18, 19, 20] ProgramInfo(params=(%5, %9, %1), globals=(0, 1, 2), outs=(2,), ins=(0, 1), target=Target(device="CUDA", renderer="CUDA", arch="sm_89"))
22 Ops.SINK dtypes.void [17, 21]

