# tolk D486 D501 golden=ac112f34d3126c73fa5df0d1c8e0dc2e
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 1536
2 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.CONST dtypes.weakint [] 3
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.SPECIAL dtypes.i32 [4] "gidx0"
6 Ops.CONST dtypes.weakint [] 4
7 Ops.CAST dtypes.i32 [6] dtypes.i32
8 Ops.SHL dtypes.i32 [5, 7]
9 Ops.SPECIAL dtypes.i32 [7] "lidx0"
10 Ops.CONST dtypes.weakint [] 2
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.SHL dtypes.i32 [9, 11]
13 Ops.ADD dtypes.i32 [8, 12]
14 Ops.CONST dtypes.weakint [] 32
15 Ops.CAST dtypes.i32 [14] dtypes.i32
16 Ops.SPECIAL dtypes.i32 [15] "lidx1"
17 Ops.CONST dtypes.weakint [] 48
18 Ops.CAST dtypes.i32 [17] dtypes.i32
19 Ops.MUL dtypes.i32 [16, 18]
20 Ops.ADD dtypes.i32 [13, 19]
21 Ops.SHRINK dtypes.f32 [2, 20, 7]
22 Ops.PARAM dtypes.f32 [14] ParamArg(slot=1, dtype=dtypes.f32)
23 Ops.INDEX dtypes.f32 [22, 16]
24 Ops.LOAD dtypes.f32 [23]
25 Ops.PARAM dtypes.f32 [17] ParamArg(slot=2, dtype=dtypes.f32)
26 Ops.SHRINK dtypes.f32 [25, 13, 7]
27 Ops.LOAD dtypes.f32 [26]
28 Ops.CONST dtypes.weakint [] 0
29 Ops.CAST dtypes.i32 [28] dtypes.i32
30 Ops.INDEX dtypes.f32 [27, 29]
31 Ops.MUL dtypes.f32 [24, 30]
32 Ops.CONST dtypes.weakint [] 1
33 Ops.CAST dtypes.i32 [32] dtypes.i32
34 Ops.INDEX dtypes.f32 [27, 33]
35 Ops.MUL dtypes.f32 [24, 34]
36 Ops.INDEX dtypes.f32 [27, 11]
37 Ops.MUL dtypes.f32 [24, 36]
38 Ops.INDEX dtypes.f32 [27, 4]
39 Ops.MUL dtypes.f32 [24, 38]
40 Ops.STACK dtypes.f32 [31, 35, 37, 39]
41 Ops.STORE dtypes.void [21, 40]
42 Ops.AFTER dtypes.void [0, 41]
43 Ops.SINK dtypes.void [42] KernelInfo(name="E_3_4_32_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(32, AxisType.LOCAL))), optimized=True)
44 Ops.SINK dtypes.void [42] KernelInfo(name="E_3_4_32_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(32, AxisType.LOCAL))), estimates=Estimates(ops=1536, lds=12416, mem=6464), optimized=True)
45 Ops.LINEAR dtypes.void [1, 2, 14, 22, 17, 25, 15, 16, 23, 24, 3, 4, 5, 6, 7, 9, 10, 11, 8, 12, 13, 26, 27, 0, 28, 29, 30, 32, 33, 34, 36, 38, 31, 35, 37, 39, 40, 18, 19, 20, 21, 41, 42, 44]
46 Ops.SOURCE dtypes.void [] "typedef unsigned int uint;\n#define INFINITY (__int_as_float(0x7f800000))\n#define NAN (__int_as_float(0x7fffffff))\ntemplate <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }\nextern \"C\" __global__ void __launch_bounds__(128) E_3_4_32_4(float* data0_1536, float* data1_32, float* data2_48) {\n  int lidx1 = threadIdx.y; /* 32 */\n  float val0 = (*(data1_32+lidx1));\n  int gidx0 = blockIdx.x; /* 3 */\n  int lidx0 = threadIdx.x; /* 4 */\n  int alu0 = ((gidx0<<4)+(lidx0<<2));\n  float4 val1 = (*((float4*)((data2_48+alu0))));\n  *((float4*)((data0_1536+(alu0+(lidx1*48))))) = make_float4((val0*val1.x),(val0*val1.y),(val0*val1.z),(val0*val1.w));\n}"
47 Ops.PROGRAM dtypes.void [44, 45, 46] ProgramInfo(global_size=(3, 1, 1), local_size=(4, 32, 1), params=(%2, %22, %25), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="CUDA", renderer="CUDA", arch="sm_89"))
48 Ops.SINK dtypes.void [43, 47]

