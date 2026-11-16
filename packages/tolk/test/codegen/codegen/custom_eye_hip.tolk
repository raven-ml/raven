# tolk D486 golden=9919a84b06d29b42b7020f3433711bfa
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 64
2 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.CONST dtypes.weakint [] 2
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.SPECIAL dtypes.i32 [4] "lidx0"
6 Ops.SHL dtypes.i32 [5, 4]
7 Ops.CONST dtypes.weakint [] 8
8 Ops.CAST dtypes.i32 [7] dtypes.i32
9 Ops.SPECIAL dtypes.i32 [8] "lidx1"
10 Ops.CONST dtypes.weakint [] 3
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.SHL dtypes.i32 [9, 11]
13 Ops.ADD dtypes.i32 [6, 12]
14 Ops.CONST dtypes.weakint [] 4
15 Ops.CAST dtypes.i32 [14] dtypes.i32
16 Ops.SHRINK dtypes.f32 [2, 13, 15]
17 Ops.CMPEQ dtypes.bool [9, 6]
18 Ops.CAST dtypes.f32 [17] dtypes.f32
19 Ops.CONST dtypes.weakint [] 1
20 Ops.CAST dtypes.i32 [19] dtypes.i32
21 Ops.ADD dtypes.i32 [6, 20]
22 Ops.CMPEQ dtypes.bool [9, 21]
23 Ops.CAST dtypes.f32 [22] dtypes.f32
24 Ops.ADD dtypes.i32 [6, 4]
25 Ops.CMPEQ dtypes.bool [9, 24]
26 Ops.CAST dtypes.f32 [25] dtypes.f32
27 Ops.ADD dtypes.i32 [6, 11]
28 Ops.CMPEQ dtypes.bool [9, 27]
29 Ops.CAST dtypes.f32 [28] dtypes.f32
30 Ops.STACK dtypes.f32 [18, 23, 26, 29]
31 Ops.STORE dtypes.void [16, 30]
32 Ops.AFTER dtypes.void [0, 31]
33 Ops.SINK dtypes.void [32] KernelInfo(name="custom_eye_64", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.LOCAL))), optimized=True)
34 Ops.SINK dtypes.void [32] KernelInfo(name="custom_eye_64", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.LOCAL))), estimates=Estimates(ops=112, lds=256, mem=256), optimized=True)
35 Ops.LINEAR dtypes.void [1, 2, 3, 4, 5, 7, 8, 9, 0, 19, 20, 6, 21, 22, 23, 24, 25, 26, 10, 11, 27, 28, 29, 17, 18, 30, 14, 15, 12, 13, 16, 31, 32, 34]
36 Ops.SOURCE dtypes.void [] "typedef long unsigned int size_t;\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_local_size(unsigned int);\ntypedef float float4 __attribute__((ext_vector_type(4)));\nstatic inline __attribute__((device)) float4 make_float4(float x, float y, float z, float w) { return { x, y, z, w }; }\nextern \"C\" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 16))) custom_eye_64(float* data0_64) {\n  int lidx0 = __ockl_get_local_id(0); /* 2 */\n  int lidx1 = __ockl_get_local_id(1); /* 8 */\n  int alu0 = (lidx0<<2);\n  *((float4*)((data0_64+(alu0+(lidx1<<3))))) = make_float4(((float)((lidx1==alu0))),((float)((lidx1==(alu0+1)))),((float)((lidx1==(alu0+2)))),((float)((lidx1==(alu0+3)))));\n}"
37 Ops.PROGRAM dtypes.void [34, 35, 36] ProgramInfo(local_size=(2, 8, 1), params=(%2,), globals=(0,), outs=(0,), target=Target(device="AMD", renderer="HIP", arch="gfx1100"))
38 Ops.SINK dtypes.void [33, 37]

