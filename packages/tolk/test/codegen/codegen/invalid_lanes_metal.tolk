# tolk D89 golden=3db65c00156a482e0f75917c785d2609
0 Ops.CONST dtypes.weakint [] 1
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.INDEX dtypes.f32 [1, 3]
5 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
6 Ops.INDEX dtypes.f32 [5, 3]
7 Ops.CONST dtypes.weakfloat [] 0.0
8 Ops.CAST dtypes.f32 [7] dtypes.f32
9 Ops.STORE dtypes.void [6, 8]
10 Ops.CONST dtypes.weakint [] 2
11 Ops.CAST dtypes.i32 [10] dtypes.i32
12 Ops.RANGE dtypes.i32 [11] (AxisType.WEAK, 0)
13 Ops.AFTER dtypes.f32 [5, 9, 12]
14 Ops.INDEX dtypes.f32 [13, 3]
15 Ops.LOAD dtypes.f32 [14]
16 Ops.CAST dtypes.i32 [0] dtypes.i32
17 Ops.CMPLT dtypes.bool [12, 16]
18 Ops.WHERE dtypes.f32 [17, 8, 8]
19 Ops.CONST dtypes.weakfloat [] 2.0
20 Ops.CAST dtypes.f32 [19] dtypes.f32
21 Ops.MUL dtypes.f32 [18, 20]
22 Ops.ADD dtypes.f32 [21, 18]
23 Ops.ADD dtypes.f32 [22, 18]
24 Ops.ADD dtypes.f32 [15, 23]
25 Ops.STORE dtypes.void [14, 24]
26 Ops.END dtypes.void [25, 12]
27 Ops.AFTER dtypes.f32 [5, 26]
28 Ops.INDEX dtypes.f32 [27, 3]
29 Ops.LOAD dtypes.f32 [28]
30 Ops.ADD dtypes.f32 [29, 8]
31 Ops.STORE dtypes.void [4, 30]
32 Ops.SINK dtypes.void [31] KernelInfo(name="r_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST)),), optimized=True)
33 Ops.CONST dtypes.weakint [] 8
34 Ops.PARAM dtypes.u8 [33] ParamArg(slot=0, dtype=dtypes.u8, name="args")
35 Ops.BITCAST dtypes.u64 [34] dtypes.u64
36 Ops.INDEX dtypes.u64 [35, 3]
37 Ops.LOAD dtypes.u64 [36]
38 Ops.REFPARAM dtypes.f32 [0, 37] ParamArg(slot=0, dtype=dtypes.f32)
39 Ops.INDEX dtypes.f32 [38, 3]
40 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, addrspace=AddrSpace.REG)
41 Ops.INDEX dtypes.f32 [40, 3]
42 Ops.STORE dtypes.void [41, 8]
43 Ops.AFTER dtypes.f32 [40, 42, 12]
44 Ops.INDEX dtypes.f32 [43, 3]
45 Ops.LOAD dtypes.f32 [44]
46 Ops.ADD dtypes.f32 [45, 23]
47 Ops.STORE dtypes.void [44, 46]
48 Ops.END dtypes.void [47, 12]
49 Ops.AFTER dtypes.f32 [40, 48]
50 Ops.INDEX dtypes.f32 [49, 3]
51 Ops.LOAD dtypes.f32 [50]
52 Ops.ADD dtypes.f32 [51, 8]
53 Ops.STORE dtypes.void [39, 52]
54 Ops.SINK dtypes.void [53] KernelInfo(name="r_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST)),), estimates=Estimates(ops=13, lds=4, mem=4), optimized=True)
55 Ops.LINEAR dtypes.void [33, 34, 0, 40, 2, 3, 41, 7, 8, 19, 20, 16, 10, 11, 42, 12, 43, 44, 45, 17, 18, 21, 22, 23, 46, 47, 48, 35, 36, 37, 49, 50, 51, 38, 39, 52, 53, 54]
56 Ops.SOURCE dtypes.void [] "#include <metal_stdlib>\nusing namespace metal;\nkernel void r_2_4(device unsigned char* args_8, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n  float buf0[1];\n  *(buf0+0) = 0.0f;\n  for (int Lidx0 = 0; Lidx0 < 2; Lidx0++) {\n    float alu1 = ((Lidx0<1)?0.0f:0.0f);\n    *(buf0+0) = ((*(buf0+0))+((alu1*2.0f)+alu1+alu1));\n  }\n  unsigned long val0 = (*(((device unsigned long*)(args_8))+0));\n  *(((device float*)(val0))+0) = ((*(buf0+0))+0.0f);\n}"
57 Ops.PROGRAM dtypes.void [54, 55, 56] ProgramInfo(params=(%1,), globals=(0,), outs=(0,), target=Target(device="METAL", renderer="METAL", arch="Apple9"))
58 Ops.SINK dtypes.void [32, 57]

