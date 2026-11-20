# tolk D501 golden=25dd01c547d1d654311ee16e95c29c8f
0 Ops.CONST dtypes.weakint [] 299
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 23
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.SPECIAL dtypes.i32 [3] "gidx0"
5 Ops.CONST dtypes.weakint [] 13
6 Ops.CAST dtypes.i32 [5] dtypes.i32
7 Ops.MUL dtypes.i32 [4, 6]
8 Ops.SPECIAL dtypes.i32 [6] "lidx0"
9 Ops.ADD dtypes.i32 [7, 8]
10 Ops.INDEX dtypes.f32 [1, 9]
11 Ops.CONST dtypes.weakint [] 1
12 Ops.CAST dtypes.i32 [11] dtypes.i32
13 Ops.ADD dtypes.i32 [9, 12]
14 Ops.CAST dtypes.f32 [13] dtypes.f32
15 Ops.CONST dtypes.weakfloat [] 3.5
16 Ops.CAST dtypes.f32 [15] dtypes.f32
17 Ops.MUL dtypes.f32 [14, 16]
18 Ops.CONST dtypes.weakfloat [] 2.0
19 Ops.CAST dtypes.f32 [18] dtypes.f32
20 Ops.ADD dtypes.f32 [17, 19]
21 Ops.STORE dtypes.void [10, 20]
22 Ops.SINK dtypes.void [21] KernelInfo(name="E_23_13", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(13, AxisType.LOCAL)),), optimized=True)
23 Ops.SINK dtypes.void [21] KernelInfo(name="E_23_13", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(13, AxisType.LOCAL)),), estimates=Estimates(ops=897, lds=1196, mem=1196), optimized=True)
24 Ops.LINEAR dtypes.void [0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14, 18, 19, 15, 16, 17, 20, 21, 23]
25 Ops.SOURCE dtypes.void [] "typedef unsigned int uint;\n#define INFINITY (__int_as_float(0x7f800000))\n#define NAN (__int_as_float(0x7fffffff))\ntemplate <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }\nextern \"C\" __global__ void __launch_bounds__(13) E_23_13(float* data0_299) {\n  int gidx0 = blockIdx.x; /* 23 */\n  int lidx0 = threadIdx.x; /* 13 */\n  int alu0 = ((gidx0*13)+lidx0);\n  *(data0_299+alu0) = ((((float)((alu0+1)))*3.5f)+2.0f);\n}"
26 Ops.PROGRAM dtypes.void [23, 24, 25] ProgramInfo(global_size=(23, 1, 1), local_size=(13, 1, 1), params=(%1,), globals=(0,), outs=(0,), target=Target(device="CUDA", renderer="CUDA", arch="sm_89"))
27 Ops.SINK dtypes.void [22, 26]

