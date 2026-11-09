# tolk D175 golden=4a1c808377b3f0e17826faebf1d03d8e
0 Ops.CONST dtypes.weakint [] 262144
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 4]
9 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.MUL dtypes.weakint [10, 4]
12 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.MUL dtypes.weakint [13, 4]
15 Ops.RANGE dtypes.weakint [2] (AxisType.WARP, -1)
16 Ops.CONST dtypes.weakint [] 16
17 Ops.DIV dtypes.weakint [15, 16]
18 Ops.MOD dtypes.weakint [17, 4]
19 Ops.ADD dtypes.weakint [14, 18]
20 Ops.CONST dtypes.weakint [] 512
21 Ops.MUL dtypes.weakint [19, 20]
22 Ops.CONST dtypes.weakint [] 8
23 Ops.RANGE dtypes.weakint [22] (AxisType.GLOBAL, 2)
24 Ops.CONST dtypes.weakint [] 4
25 Ops.MUL dtypes.weakint [23, 24]
26 Ops.RANGE dtypes.weakint [24] (AxisType.LOCAL, 10)
27 Ops.ADD dtypes.weakint [25, 26]
28 Ops.MUL dtypes.weakint [27, 4]
29 Ops.DIV dtypes.weakint [15, 22]
30 Ops.MOD dtypes.weakint [29, 4]
31 Ops.ADD dtypes.weakint [28, 30]
32 Ops.MUL dtypes.weakint [31, 4]
33 Ops.DIV dtypes.weakint [15, 24]
34 Ops.MOD dtypes.weakint [33, 4]
35 Ops.ADD dtypes.weakint [32, 34]
36 Ops.MUL dtypes.weakint [35, 4]
37 Ops.DIV dtypes.weakint [15, 4]
38 Ops.MOD dtypes.weakint [37, 4]
39 Ops.ADD dtypes.weakint [36, 38]
40 Ops.MUL dtypes.weakint [39, 4]
41 Ops.CONST dtypes.weakint [] 1
42 Ops.DIV dtypes.weakint [15, 41]
43 Ops.MOD dtypes.weakint [42, 4]
44 Ops.ADD dtypes.weakint [40, 43]
45 Ops.ADD dtypes.weakint [21, 44]
46 Ops.INDEX dtypes.f32 [1, 45]
47 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
48 Ops.ADD dtypes.weakint [5, 30]
49 Ops.MUL dtypes.weakint [48, 4]
50 Ops.ADD dtypes.weakint [49, 34]
51 Ops.MUL dtypes.weakint [50, 4]
52 Ops.ADD dtypes.weakint [51, 38]
53 Ops.MUL dtypes.weakint [52, 4]
54 Ops.ADD dtypes.weakint [53, 43]
55 Ops.MUL dtypes.weakint [54, 20]
56 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
57 Ops.MUL dtypes.weakint [56, 4]
58 Ops.ADD dtypes.weakint [57, 6]
59 Ops.MUL dtypes.weakint [58, 4]
60 Ops.ADD dtypes.weakint [59, 9]
61 Ops.MUL dtypes.weakint [60, 4]
62 Ops.ADD dtypes.weakint [61, 12]
63 Ops.MUL dtypes.weakint [62, 4]
64 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 9)
65 Ops.ADD dtypes.weakint [63, 64]
66 Ops.ADD dtypes.weakint [55, 65]
67 Ops.INDEX dtypes.f16 [47, 66]
68 Ops.PARAM dtypes.f16 [0] ParamArg(slot=2, dtype=dtypes.f16)
69 Ops.MUL dtypes.weakint [65, 20]
70 Ops.ADD dtypes.weakint [69, 44]
71 Ops.INDEX dtypes.f16 [68, 70]
72 Ops.CONST dtypes.weakfloat [] 0.0
73 Ops.CAST dtypes.f32 [72] dtypes.f32
74 Ops.STACK dtypes.f32 [73, 73, 73, 73, 73, 73, 73, 73]
75 Ops.WMMA dtypes.f32 [67, 71, 74] ((16, 16, 16), dtypes.f16, 32, ((((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 1))))
76 Ops.REDUCE dtypes.f32 [75, 56] (Ops.ADD, 0)
77 Ops.STORE dtypes.void [46, 76]
78 Ops.END dtypes.void [77, 3, 6, 9, 12, 15, 23, 26]
79 Ops.SINK dtypes.void [78] KernelInfo(name="r_32_8_4_32_2_2_2_2_32", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL))), optimized=True)

