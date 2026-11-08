# tolk D89 D280 golden=47486858d0a7d408acb93a476b931409
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
20 Ops.CONST dtypes.weakint [] 8
21 Ops.RANGE dtypes.weakint [20] (AxisType.GLOBAL, 2)
22 Ops.MUL dtypes.weakint [21, 4]
23 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 8)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.MUL dtypes.weakint [24, 4]
26 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 7)
27 Ops.ADD dtypes.weakint [25, 26]
28 Ops.MUL dtypes.weakint [27, 4]
29 Ops.DIV dtypes.weakint [6, 20]
30 Ops.MOD dtypes.weakint [29, 4]
31 Ops.ADD dtypes.weakint [28, 30]
32 Ops.MUL dtypes.weakint [31, 4]
33 Ops.CONST dtypes.weakint [] 1
34 Ops.DIV dtypes.weakint [6, 33]
35 Ops.MOD dtypes.weakint [34, 4]
36 Ops.ADD dtypes.weakint [32, 35]
37 Ops.MUL dtypes.weakint [36, 4]
38 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
39 Ops.ADD dtypes.weakint [37, 38]
40 Ops.AFTER dtypes.f32 [1, 19, 39]
41 Ops.CONST dtypes.weakint [] 256
42 Ops.MUL dtypes.weakint [19, 41]
43 Ops.ADD dtypes.weakint [42, 39]
44 Ops.INDEX dtypes.f32 [40, 43]
45 Ops.CONST dtypes.weakint [] 16384
46 Ops.PARAM dtypes.f32 [45] ParamArg(slot=1, dtype=dtypes.f32)
47 Ops.CONST dtypes.weakint [] 64
48 Ops.MUL dtypes.weakint [19, 47]
49 Ops.RANGE dtypes.weakint [20] (AxisType.WEAK, 0)
50 Ops.MUL dtypes.weakint [49, 4]
51 Ops.ADD dtypes.weakint [50, 30]
52 Ops.MUL dtypes.weakint [51, 4]
53 Ops.ADD dtypes.weakint [52, 35]
54 Ops.MUL dtypes.weakint [53, 4]
55 Ops.ADD dtypes.weakint [54, 38]
56 Ops.ADD dtypes.weakint [48, 55]
57 Ops.INDEX dtypes.f32 [46, 56]
58 Ops.PARAM dtypes.f32 [45] ParamArg(slot=2, dtype=dtypes.f32)
59 Ops.ADD dtypes.weakint [50, 9]
60 Ops.MUL dtypes.weakint [59, 4]
61 Ops.ADD dtypes.weakint [60, 14]
62 Ops.MUL dtypes.weakint [61, 4]
63 Ops.ADD dtypes.weakint [62, 18]
64 Ops.MUL dtypes.weakint [63, 41]
65 Ops.ADD dtypes.weakint [64, 39]
66 Ops.INDEX dtypes.f32 [58, 65]
67 Ops.CONST dtypes.weakfloat [] 0.0
68 Ops.CAST dtypes.f32 [67] dtypes.f32
69 Ops.STACK dtypes.f32 [68, 68]
70 Ops.WMMA dtypes.f32 [57, 66, 69] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
71 Ops.REDUCE dtypes.f32 [70, 49] (Ops.ADD, 0)
72 Ops.STORE dtypes.void [44, 71]
73 Ops.END dtypes.void [72, 3, 6, 21, 23, 26, 38]
74 Ops.SINK dtypes.void [73] KernelInfo(name="r_32_8_2_32_2_2_8", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.LOCAL))), optimized=True)

