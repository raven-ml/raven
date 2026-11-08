# tolk D89 golden=d1a85e8d316d24d77bf9e443492c9bc2
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 64
2 Ops.PARAM dtypes.f32 [1] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.CONST dtypes.weakint [] 16
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.SPECIAL dtypes.i32 [4] "lidx0"
6 Ops.INDEX dtypes.f32 [2, 5]
7 Ops.PARAM dtypes.f32 [1] ParamArg(slot=1, dtype=dtypes.f32)
8 Ops.INDEX dtypes.f32 [7, 5]
9 Ops.LOAD dtypes.f32 [8]
10 Ops.CONST dtypes.weakfloat [] 0.0
11 Ops.CAST dtypes.f32 [10] dtypes.f32
12 Ops.ADD dtypes.f32 [9, 11]
13 Ops.STORE dtypes.void [6, 12]
14 Ops.ADD dtypes.i32 [5, 4]
15 Ops.INDEX dtypes.f32 [2, 14]
16 Ops.INDEX dtypes.f32 [7, 14]
17 Ops.LOAD dtypes.f32 [16]
18 Ops.CONST dtypes.weakint [] 4
19 Ops.PARAM dtypes.f32 [18] ParamArg(slot=2, dtype=dtypes.f32)
20 Ops.CONST dtypes.weakint [] 0
21 Ops.CAST dtypes.i32 [20] dtypes.i32
22 Ops.CONST dtypes.weakint [] 2
23 Ops.CAST dtypes.i32 [22] dtypes.i32
24 Ops.SHRINK dtypes.f32 [19, 21, 23]
25 Ops.LOAD dtypes.f32 [24]
26 Ops.INDEX dtypes.f32 [25, 21]
27 Ops.ADD dtypes.f32 [17, 26]
28 Ops.STORE dtypes.void [15, 27]
29 Ops.CONST dtypes.weakint [] 32
30 Ops.CAST dtypes.i32 [29] dtypes.i32
31 Ops.ADD dtypes.i32 [5, 30]
32 Ops.INDEX dtypes.f32 [2, 31]
33 Ops.INDEX dtypes.f32 [7, 31]
34 Ops.LOAD dtypes.f32 [33]
35 Ops.CONST dtypes.weakint [] 1
36 Ops.CAST dtypes.i32 [35] dtypes.i32
37 Ops.INDEX dtypes.f32 [25, 36]
38 Ops.ADD dtypes.f32 [34, 37]
39 Ops.STORE dtypes.void [32, 38]
40 Ops.CONST dtypes.weakint [] 48
41 Ops.CAST dtypes.i32 [40] dtypes.i32
42 Ops.ADD dtypes.i32 [5, 41]
43 Ops.INDEX dtypes.f32 [2, 42]
44 Ops.INDEX dtypes.f32 [7, 42]
45 Ops.LOAD dtypes.f32 [44]
46 Ops.INDEX dtypes.f32 [19, 23]
47 Ops.LOAD dtypes.f32 [46]
48 Ops.ADD dtypes.f32 [45, 47]
49 Ops.STORE dtypes.void [43, 48]
50 Ops.AFTER dtypes.void [0, 13, 28, 39, 49]
51 Ops.SINK dtypes.void [50] KernelInfo(name="E_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.LOCAL))), optimized=True)
52 Ops.SINK dtypes.void [50] KernelInfo(name="E_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.LOCAL))), estimates=Estimates(ops=64, lds=704, mem=528), optimized=True)
53 Ops.LINEAR dtypes.void [1, 2, 7, 18, 19, 3, 4, 5, 8, 9, 14, 16, 17, 29, 30, 31, 33, 34, 40, 41, 42, 44, 45, 22, 23, 46, 47, 20, 21, 24, 25, 0, 6, 15, 32, 43, 26, 35, 36, 37, 10, 11, 12, 27, 38, 48, 13, 28, 39, 49, 50, 52]
54 Ops.SOURCE dtypes.void [] "typedef long unsigned int size_t;\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_local_size(unsigned int);\ntypedef float float2 __attribute__((ext_vector_type(2)));\nstatic inline __attribute__((device)) float2 make_float2(float x, float y) { return { x, y }; }\nextern \"C\" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 16))) E_16_4(float* data0_64, float* data1_64, float* data2_4) {\n  int lidx0 = __ockl_get_local_id(0); /* 16 */\n  float val0 = (*(data1_64+lidx0));\n  int alu0 = (lidx0+16);\n  float val1 = (*(data1_64+alu0));\n  int alu1 = (lidx0+32);\n  float val2 = (*(data1_64+alu1));\n  int alu2 = (lidx0+48);\n  float val3 = (*(data1_64+alu2));\n  float val4 = (*(data2_4+2));\n  float2 val5 = (*((float2*)((data2_4+0))));\n  *(data0_64+lidx0) = (val0+0.0f);\n  *(data0_64+alu0) = (val1+val5.x);\n  *(data0_64+alu1) = (val2+val5.y);\n  *(data0_64+alu2) = (val3+val4);\n}"
55 Ops.PROGRAM dtypes.void [52, 53, 54] ProgramInfo(local_size=(16, 1, 1), params=(%2, %7, %19), globals=(0, 1, 2), outs=(0,), ins=(1, 2), target=Target(device="AMD", renderer="HIP", arch="gfx1100"))
56 Ops.SINK dtypes.void [51, 55]

