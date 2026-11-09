# tolk D175 golden=759716ea450e8c3fd7cff73adddfd245
0 Ops.CONST dtypes.weakint [] 65536
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [2] (AxisType.WARP, -1)
7 Ops.CONST dtypes.weakint [] 16
8 Ops.DIV dtypes.weakint [6, 7]
9 Ops.MOD dtypes.weakint [8, 4]
10 Ops.ADD dtypes.weakint [5, 9]
11 Ops.MUL dtypes.weakint [10, 4]
12 Ops.CONST dtypes.weakint [] 4
13 Ops.DIV dtypes.weakint [6, 12]
14 Ops.MOD dtypes.weakint [13, 4]
15 Ops.ADD dtypes.weakint [11, 14]
16 Ops.MUL dtypes.weakint [15, 4]
17 Ops.DIV dtypes.weakint [6, 4]
18 Ops.MOD dtypes.weakint [17, 4]
19 Ops.ADD dtypes.weakint [16, 18]
20 Ops.CONST dtypes.weakint [] 256
21 Ops.MUL dtypes.weakint [19, 20]
22 Ops.CONST dtypes.weakint [] 8
23 Ops.RANGE dtypes.weakint [22] (AxisType.GLOBAL, 2)
24 Ops.MUL dtypes.weakint [23, 12]
25 Ops.RANGE dtypes.weakint [12] (AxisType.LOCAL, 7)
26 Ops.ADD dtypes.weakint [24, 25]
27 Ops.MUL dtypes.weakint [26, 4]
28 Ops.DIV dtypes.weakint [6, 22]
29 Ops.MOD dtypes.weakint [28, 4]
30 Ops.ADD dtypes.weakint [27, 29]
31 Ops.MUL dtypes.weakint [30, 4]
32 Ops.CONST dtypes.weakint [] 1
33 Ops.DIV dtypes.weakint [6, 32]
34 Ops.MOD dtypes.weakint [33, 4]
35 Ops.ADD dtypes.weakint [31, 34]
36 Ops.MUL dtypes.weakint [35, 4]
37 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
38 Ops.ADD dtypes.weakint [36, 37]
39 Ops.ADD dtypes.weakint [21, 38]
40 Ops.INDEX dtypes.f32 [1, 39]
41 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
42 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
43 Ops.MUL dtypes.weakint [42, 4]
44 Ops.ADD dtypes.weakint [43, 29]
45 Ops.MUL dtypes.weakint [44, 4]
46 Ops.ADD dtypes.weakint [45, 34]
47 Ops.MUL dtypes.weakint [46, 4]
48 Ops.ADD dtypes.weakint [47, 37]
49 Ops.ADD dtypes.weakint [21, 48]
50 Ops.INDEX dtypes.f16 [41, 49]
51 Ops.PARAM dtypes.f16 [0] ParamArg(slot=2, dtype=dtypes.f16)
52 Ops.ADD dtypes.weakint [43, 9]
53 Ops.MUL dtypes.weakint [52, 4]
54 Ops.ADD dtypes.weakint [53, 14]
55 Ops.MUL dtypes.weakint [54, 4]
56 Ops.ADD dtypes.weakint [55, 18]
57 Ops.MUL dtypes.weakint [56, 20]
58 Ops.ADD dtypes.weakint [57, 38]
59 Ops.INDEX dtypes.f16 [51, 58]
60 Ops.CONST dtypes.weakfloat [] 0.0
61 Ops.CAST dtypes.f32 [60] dtypes.f32
62 Ops.STACK dtypes.f32 [61, 61]
63 Ops.WMMA dtypes.f32 [50, 59, 62] ((8, 8, 8), dtypes.f16, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
64 Ops.REDUCE dtypes.f32 [63, 42] (Ops.ADD, 0)
65 Ops.STORE dtypes.void [40, 64]
66 Ops.END dtypes.void [65, 3, 6, 23, 25, 37]
67 Ops.SINK dtypes.void [66] KernelInfo(name="r_32_8_4_32_2_32", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL))), optimized=True)

