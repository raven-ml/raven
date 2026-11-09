# tolk D89 golden=e722885103f3ec4d1cc5f6b71a4ffe57
0 Ops.CONST dtypes.weakint [] 262144
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 7)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 2
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 32
11 Ops.RANGE dtypes.weakint [10] (AxisType.WARP, -1)
12 Ops.DIV dtypes.weakint [11, 2]
13 Ops.MOD dtypes.weakint [12, 8]
14 Ops.ADD dtypes.weakint [9, 13]
15 Ops.MUL dtypes.weakint [14, 8]
16 Ops.DIV dtypes.weakint [11, 4]
17 Ops.MOD dtypes.weakint [16, 8]
18 Ops.ADD dtypes.weakint [15, 17]
19 Ops.MUL dtypes.weakint [18, 8]
20 Ops.DIV dtypes.weakint [11, 8]
21 Ops.MOD dtypes.weakint [20, 8]
22 Ops.ADD dtypes.weakint [19, 21]
23 Ops.CONST dtypes.weakint [] 512
24 Ops.MUL dtypes.weakint [22, 23]
25 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
26 Ops.MUL dtypes.weakint [25, 8]
27 Ops.RANGE dtypes.weakint [8] (AxisType.LOCAL, 9)
28 Ops.ADD dtypes.weakint [26, 27]
29 Ops.MUL dtypes.weakint [28, 8]
30 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 8)
31 Ops.ADD dtypes.weakint [29, 30]
32 Ops.MUL dtypes.weakint [31, 8]
33 Ops.CONST dtypes.weakint [] 8
34 Ops.DIV dtypes.weakint [11, 33]
35 Ops.MOD dtypes.weakint [34, 8]
36 Ops.ADD dtypes.weakint [32, 35]
37 Ops.MUL dtypes.weakint [36, 8]
38 Ops.CONST dtypes.weakint [] 1
39 Ops.DIV dtypes.weakint [11, 38]
40 Ops.MOD dtypes.weakint [39, 8]
41 Ops.ADD dtypes.weakint [37, 40]
42 Ops.MUL dtypes.weakint [41, 8]
43 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 3)
44 Ops.ADD dtypes.weakint [42, 43]
45 Ops.ADD dtypes.weakint [24, 44]
46 Ops.INDEX dtypes.f32 [1, 45]
47 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
48 Ops.CONST dtypes.weakint [] 64
49 Ops.RANGE dtypes.weakint [48] (AxisType.WEAK, 0)
50 Ops.MUL dtypes.weakint [49, 8]
51 Ops.ADD dtypes.weakint [50, 35]
52 Ops.MUL dtypes.weakint [51, 8]
53 Ops.ADD dtypes.weakint [52, 40]
54 Ops.MUL dtypes.weakint [53, 8]
55 Ops.ADD dtypes.weakint [54, 43]
56 Ops.ADD dtypes.weakint [24, 55]
57 Ops.INDEX dtypes.f16 [47, 56]
58 Ops.PARAM dtypes.f16 [0] ParamArg(slot=2, dtype=dtypes.f16)
59 Ops.ADD dtypes.weakint [50, 13]
60 Ops.MUL dtypes.weakint [59, 8]
61 Ops.ADD dtypes.weakint [60, 17]
62 Ops.MUL dtypes.weakint [61, 8]
63 Ops.ADD dtypes.weakint [62, 21]
64 Ops.MUL dtypes.weakint [63, 23]
65 Ops.ADD dtypes.weakint [64, 44]
66 Ops.INDEX dtypes.f16 [58, 65]
67 Ops.CONST dtypes.weakfloat [] 0.0
68 Ops.CAST dtypes.f32 [67] dtypes.f32
69 Ops.STACK dtypes.f32 [68, 68]
70 Ops.WMMA dtypes.f32 [57, 66, 69] ((8, 8, 8), dtypes.f16, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
71 Ops.REDUCE dtypes.f32 [70, 49] (Ops.ADD, 0)
72 Ops.STORE dtypes.void [46, 71]
73 Ops.END dtypes.void [72, 3, 6, 11, 25, 27, 30, 43]
74 Ops.SINK dtypes.void [73] KernelInfo(name="r_16_16_2_32_2_4_2_64", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.LOCAL))), optimized=True)

