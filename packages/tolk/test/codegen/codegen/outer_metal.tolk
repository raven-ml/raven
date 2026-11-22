# tolk D486 D501 golden=27a94fe729c32fb1e9ceb2d7fbd575f4
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
44 Ops.CONST dtypes.weakint [] 24
45 Ops.PARAM dtypes.u8 [44] ParamArg(slot=0, dtype=dtypes.u8, name="args")
46 Ops.BITCAST dtypes.u64 [45] dtypes.u64
47 Ops.INDEX dtypes.u64 [46, 29]
48 Ops.LOAD dtypes.u64 [47]
49 Ops.REFPARAM dtypes.f32 [1, 48] ParamArg(slot=0, dtype=dtypes.f32)
50 Ops.SHRINK dtypes.f32 [49, 20, 7]
51 Ops.INDEX dtypes.u64 [46, 33]
52 Ops.LOAD dtypes.u64 [51]
53 Ops.REFPARAM dtypes.f32 [14, 52] ParamArg(slot=1, dtype=dtypes.f32)
54 Ops.INDEX dtypes.f32 [53, 16]
55 Ops.LOAD dtypes.f32 [54]
56 Ops.INDEX dtypes.u64 [46, 11]
57 Ops.LOAD dtypes.u64 [56]
58 Ops.REFPARAM dtypes.f32 [17, 57] ParamArg(slot=2, dtype=dtypes.f32)
59 Ops.SHRINK dtypes.f32 [58, 13, 7]
60 Ops.LOAD dtypes.f32 [59]
61 Ops.INDEX dtypes.f32 [60, 29]
62 Ops.MUL dtypes.f32 [55, 61]
63 Ops.INDEX dtypes.f32 [60, 33]
64 Ops.MUL dtypes.f32 [55, 63]
65 Ops.INDEX dtypes.f32 [60, 11]
66 Ops.MUL dtypes.f32 [55, 65]
67 Ops.INDEX dtypes.f32 [60, 4]
68 Ops.MUL dtypes.f32 [55, 67]
69 Ops.STACK dtypes.f32 [62, 64, 66, 68]
70 Ops.STORE dtypes.void [50, 69]
71 Ops.AFTER dtypes.void [0, 70]
72 Ops.SINK dtypes.void [71] KernelInfo(name="E_3_4_32_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(32, AxisType.LOCAL))), estimates=Estimates(ops=1536, lds=12416, mem=6464), optimized=True)
73 Ops.LINEAR dtypes.void [44, 45, 28, 29, 46, 47, 48, 32, 33, 51, 52, 10, 11, 56, 57, 14, 15, 16, 53, 54, 55, 3, 4, 5, 6, 7, 9, 17, 58, 8, 12, 13, 59, 60, 1, 49, 0, 61, 63, 65, 67, 62, 64, 66, 68, 69, 18, 19, 20, 50, 70, 71, 72]
74 Ops.SOURCE dtypes.void [] "#include <metal_stdlib>\nusing namespace metal;\nkernel void E_3_4_32_4(device unsigned char* args_24, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n  unsigned long val0 = (*(((device unsigned long*)(args_24))+0));\n  unsigned long val1 = (*(((device unsigned long*)(args_24))+1));\n  unsigned long val2 = (*(((device unsigned long*)(args_24))+2));\n  int lidx1 = lid.y; /* 32 */\n  float val3 = (*(((device float*)(val1))+lidx1));\n  int gidx0 = gid.x; /* 3 */\n  int lidx0 = lid.x; /* 4 */\n  int alu0 = ((gidx0<<4)+(lidx0<<2));\n  float4 val4 = (*((device float4*)((((device float*)(val2))+alu0))));\n  *((device float4*)((((device float*)(val0))+(alu0+(lidx1*48))))) = float4((val3*val4.x),(val3*val4.y),(val3*val4.z),(val3*val4.w));\n}"
75 Ops.PROGRAM dtypes.void [72, 73, 74] ProgramInfo(global_size=(3, 1, 1), local_size=(4, 32, 1), params=(%2, %22, %25), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="METAL", renderer="METAL", arch="Apple9"))
76 Ops.SINK dtypes.void [43, 75]

