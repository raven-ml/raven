# tolk D501 golden=fa90ea6a09c05aac3cd0412c608ebfdb
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
23 Ops.CONST dtypes.weakint [] 8
24 Ops.PARAM dtypes.u8 [23] ParamArg(slot=0, dtype=dtypes.u8, name="args")
25 Ops.BITCAST dtypes.u64 [24] dtypes.u64
26 Ops.CONST dtypes.weakint [] 0
27 Ops.CAST dtypes.i32 [26] dtypes.i32
28 Ops.INDEX dtypes.u64 [25, 27]
29 Ops.LOAD dtypes.u64 [28]
30 Ops.REFPARAM dtypes.f32 [0, 29] ParamArg(slot=0, dtype=dtypes.f32)
31 Ops.INDEX dtypes.f32 [30, 9]
32 Ops.STORE dtypes.void [31, 20]
33 Ops.SINK dtypes.void [32] KernelInfo(name="E_23_13", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(13, AxisType.LOCAL)),), estimates=Estimates(ops=897, lds=1196, mem=1196), optimized=True)
34 Ops.LINEAR dtypes.void [23, 24, 26, 27, 25, 28, 29, 2, 3, 4, 5, 6, 8, 0, 30, 7, 9, 31, 11, 12, 13, 14, 18, 19, 15, 16, 17, 20, 32, 33]
35 Ops.SOURCE dtypes.void [] "#include <metal_stdlib>\nusing namespace metal;\nkernel void E_23_13(device unsigned char* args_8, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n  unsigned long val0 = (*(((device unsigned long*)(args_8))+0));\n  int gidx0 = gid.x; /* 23 */\n  int lidx0 = lid.x; /* 13 */\n  int alu0 = ((gidx0*13)+lidx0);\n  *(((device float*)(val0))+alu0) = ((((float)((alu0+1)))*3.5f)+2.0f);\n}"
36 Ops.PROGRAM dtypes.void [33, 34, 35] ProgramInfo(global_size=(23, 1, 1), local_size=(13, 1, 1), params=(%1,), globals=(0,), outs=(0,), target=Target(device="METAL", renderer="METAL", arch="Apple9"))
37 Ops.SINK dtypes.void [22, 36]

