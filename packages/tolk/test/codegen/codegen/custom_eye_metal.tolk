# tolk D486 golden=44340835769488a2a4d9b1ae7166a167
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
34 Ops.PARAM dtypes.u8 [7] ParamArg(slot=0, dtype=dtypes.u8, name="args")
35 Ops.BITCAST dtypes.u64 [34] dtypes.u64
36 Ops.CONST dtypes.weakint [] 0
37 Ops.CAST dtypes.i32 [36] dtypes.i32
38 Ops.INDEX dtypes.u64 [35, 37]
39 Ops.LOAD dtypes.u64 [38]
40 Ops.REFPARAM dtypes.f32 [1, 39] ParamArg(slot=0, dtype=dtypes.f32)
41 Ops.SHRINK dtypes.f32 [40, 13, 15]
42 Ops.STORE dtypes.void [41, 30]
43 Ops.AFTER dtypes.void [0, 42]
44 Ops.SINK dtypes.void [43] KernelInfo(name="custom_eye_64", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.LOCAL))), estimates=Estimates(ops=112, lds=256, mem=256), optimized=True)
45 Ops.LINEAR dtypes.void [7, 34, 36, 37, 35, 38, 39, 3, 4, 5, 8, 9, 1, 40, 0, 19, 20, 6, 21, 22, 23, 24, 25, 26, 10, 11, 27, 28, 29, 17, 18, 30, 14, 15, 12, 13, 41, 42, 43, 44]
46 Ops.SOURCE dtypes.void [] "#include <metal_stdlib>\nusing namespace metal;\nkernel void custom_eye_64(device unsigned char* args_8, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n  unsigned long val0 = (*(((device unsigned long*)(args_8))+0));\n  int lidx0 = lid.x; /* 2 */\n  int lidx1 = lid.y; /* 8 */\n  int alu0 = (lidx0<<2);\n  *((device float4*)((((device float*)(val0))+(alu0+(lidx1<<3))))) = float4(((float)((lidx1==alu0))),((float)((lidx1==(alu0+1)))),((float)((lidx1==(alu0+2)))),((float)((lidx1==(alu0+3)))));\n}"
47 Ops.PROGRAM dtypes.void [44, 45, 46] ProgramInfo(local_size=(2, 8, 1), params=(%2,), globals=(0,), outs=(0,), target=Target(device="METAL", renderer="METAL", arch="Apple9"))
48 Ops.SINK dtypes.void [33, 47]

