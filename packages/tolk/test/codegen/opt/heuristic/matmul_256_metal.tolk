# tolk D175 golden=47486858d0a7d408acb93a476b931409
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
41 Ops.CONST dtypes.weakint [] 16384
42 Ops.PARAM dtypes.f32 [41] ParamArg(slot=1, dtype=dtypes.f32)
43 Ops.CONST dtypes.weakint [] 64
44 Ops.MUL dtypes.weakint [19, 43]
45 Ops.RANGE dtypes.weakint [22] (AxisType.WEAK, 0)
46 Ops.MUL dtypes.weakint [45, 4]
47 Ops.ADD dtypes.weakint [46, 29]
48 Ops.MUL dtypes.weakint [47, 4]
49 Ops.ADD dtypes.weakint [48, 34]
50 Ops.MUL dtypes.weakint [49, 4]
51 Ops.ADD dtypes.weakint [50, 37]
52 Ops.ADD dtypes.weakint [44, 51]
53 Ops.INDEX dtypes.f32 [42, 52]
54 Ops.PARAM dtypes.f32 [41] ParamArg(slot=2, dtype=dtypes.f32)
55 Ops.ADD dtypes.weakint [46, 9]
56 Ops.MUL dtypes.weakint [55, 4]
57 Ops.ADD dtypes.weakint [56, 14]
58 Ops.MUL dtypes.weakint [57, 4]
59 Ops.ADD dtypes.weakint [58, 18]
60 Ops.MUL dtypes.weakint [59, 20]
61 Ops.ADD dtypes.weakint [60, 38]
62 Ops.INDEX dtypes.f32 [54, 61]
63 Ops.CONST dtypes.weakfloat [] 0.0
64 Ops.CAST dtypes.f32 [63] dtypes.f32
65 Ops.STACK dtypes.f32 [64, 64]
66 Ops.WMMA dtypes.f32 [53, 62, 65] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
67 Ops.REDUCE dtypes.f32 [66, 45] (Ops.ADD, 0)
68 Ops.STORE dtypes.void [40, 67]
69 Ops.END dtypes.void [68, 3, 6, 23, 25, 37]
70 Ops.SINK dtypes.void [69] KernelInfo(name="r_32_8_4_32_2_8", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL))), optimized=True)

