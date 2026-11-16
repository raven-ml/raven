# tolk D486 golden=8b88e1cde27228a39dfe0ec692d249dc
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 128
2 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.CONST dtypes.weakint [] 4
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.SPECIAL dtypes.i32 [4] "lidx0"
6 Ops.CONST dtypes.weakint [] 2
7 Ops.CAST dtypes.i32 [6] dtypes.i32
8 Ops.SHL dtypes.i32 [5, 7]
9 Ops.CONST dtypes.weakint [] 8
10 Ops.CAST dtypes.i32 [9] dtypes.i32
11 Ops.SPECIAL dtypes.i32 [10] "lidx1"
12 Ops.SHL dtypes.i32 [11, 4]
13 Ops.ADD dtypes.i32 [8, 12]
14 Ops.SHRINK dtypes.f32 [2, 13, 4]
15 Ops.PARAM dtypes.f32 [1] ParamArg(slot=1, dtype=dtypes.f32)
16 Ops.CONST dtypes.weakint [] -4
17 Ops.CAST dtypes.i32 [16] dtypes.i32
18 Ops.MUL dtypes.i32 [5, 17]
19 Ops.ADD dtypes.i32 [18, 12]
20 Ops.CONST dtypes.weakint [] 12
21 Ops.CAST dtypes.i32 [20] dtypes.i32
22 Ops.ADD dtypes.i32 [19, 21]
23 Ops.SHRINK dtypes.f32 [15, 22, 4]
24 Ops.LOAD dtypes.f32 [23]
25 Ops.CONST dtypes.weakint [] 3
26 Ops.CAST dtypes.i32 [25] dtypes.i32
27 Ops.INDEX dtypes.f32 [24, 26]
28 Ops.INDEX dtypes.f32 [24, 7]
29 Ops.CONST dtypes.weakint [] 1
30 Ops.CAST dtypes.i32 [29] dtypes.i32
31 Ops.INDEX dtypes.f32 [24, 30]
32 Ops.CONST dtypes.weakint [] 0
33 Ops.CAST dtypes.i32 [32] dtypes.i32
34 Ops.INDEX dtypes.f32 [24, 33]
35 Ops.STACK dtypes.f32 [27, 28, 31, 34]
36 Ops.STORE dtypes.void [14, 35]
37 Ops.AFTER dtypes.void [0, 36]
38 Ops.SINK dtypes.void [37] KernelInfo(name="E_4_8_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.LOCAL))), optimized=True)
39 Ops.SINK dtypes.void [37] KernelInfo(name="E_4_8_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.LOCAL))), estimates=Estimates(lds=1024, mem=1024), optimized=True)
40 Ops.LINEAR dtypes.void [1, 2, 15, 3, 4, 5, 9, 10, 11, 16, 17, 20, 21, 18, 12, 19, 22, 23, 24, 0, 32, 33, 34, 29, 30, 31, 6, 7, 28, 25, 26, 27, 35, 8, 13, 14, 36, 37, 39]
41 Ops.SOURCE dtypes.void [] "typedef unsigned int uint;\n#define INFINITY (__int_as_float(0x7f800000))\n#define NAN (__int_as_float(0x7fffffff))\ntemplate <class T, class F> __device__ __forceinline__ T tg_bitcast(F v) { union U { F f; T t; }; U u; u.f = v; return u.t; }\nextern \"C\" __global__ void __launch_bounds__(32) E_4_8_4(float* data0_128, float* data1_128) {\n  int lidx0 = threadIdx.x; /* 4 */\n  int lidx1 = threadIdx.y; /* 8 */\n  int alu0 = (lidx1<<4);\n  float4 val0 = (*((float4*)((data1_128+((lidx0*-4)+alu0+12)))));\n  *((float4*)((data0_128+((lidx0<<2)+alu0)))) = make_float4(val0.w,val0.z,val0.y,val0.x);\n}"
42 Ops.PROGRAM dtypes.void [39, 40, 41] ProgramInfo(local_size=(4, 8, 1), params=(%2, %15), globals=(0, 1), outs=(0,), ins=(1,), target=Target(device="CUDA", renderer="CUDA", arch="sm_89"))
43 Ops.SINK dtypes.void [38, 42]

