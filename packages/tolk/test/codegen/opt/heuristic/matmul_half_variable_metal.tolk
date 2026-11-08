# tolk D89 D280 golden=095338dbb50053166884d7bcaacf7fa3
0 Ops.CONST dtypes.weakint [] 1048576
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.STACK dtypes.void []
3 Ops.PARAM dtypes.weakint [2] ParamArg(slot=-1, dtype=dtypes.weakint, vmin_vmax=(1, 64), name="n", addrspace=AddrSpace.ALU)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [5] (AxisType.GLOBAL, 1)
7 Ops.MUL dtypes.weakint [6, 4]
8 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 8)
9 Ops.ADD dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 2
11 Ops.MUL dtypes.weakint [9, 10]
12 Ops.CONST dtypes.weakint [] 32
13 Ops.RANGE dtypes.weakint [12] (AxisType.WARP, -1)
14 Ops.CONST dtypes.weakint [] 16
15 Ops.DIV dtypes.weakint [13, 14]
16 Ops.MOD dtypes.weakint [15, 10]
17 Ops.ADD dtypes.weakint [11, 16]
18 Ops.MUL dtypes.weakint [17, 10]
19 Ops.DIV dtypes.weakint [13, 4]
20 Ops.MOD dtypes.weakint [19, 10]
21 Ops.ADD dtypes.weakint [18, 20]
22 Ops.MUL dtypes.weakint [21, 10]
23 Ops.DIV dtypes.weakint [13, 10]
24 Ops.MOD dtypes.weakint [23, 10]
25 Ops.ADD dtypes.weakint [22, 24]
26 Ops.CONST dtypes.weakint [] 1
27 Ops.RANGE dtypes.weakint [26] (AxisType.GLOBAL, 3)
28 Ops.MUL dtypes.weakint [27, 10]
29 Ops.RANGE dtypes.weakint [10] (AxisType.LOCAL, 11)
30 Ops.ADD dtypes.weakint [28, 29]
31 Ops.MUL dtypes.weakint [30, 10]
32 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 10)
33 Ops.ADD dtypes.weakint [31, 32]
34 Ops.MUL dtypes.weakint [33, 4]
35 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 9)
36 Ops.ADD dtypes.weakint [34, 35]
37 Ops.MUL dtypes.weakint [36, 10]
38 Ops.CONST dtypes.weakint [] 8
39 Ops.DIV dtypes.weakint [13, 38]
40 Ops.MOD dtypes.weakint [39, 10]
41 Ops.ADD dtypes.weakint [37, 40]
42 Ops.MUL dtypes.weakint [41, 10]
43 Ops.DIV dtypes.weakint [13, 26]
44 Ops.MOD dtypes.weakint [43, 10]
45 Ops.ADD dtypes.weakint [42, 44]
46 Ops.MUL dtypes.weakint [45, 10]
47 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 4)
48 Ops.ADD dtypes.weakint [46, 47]
49 Ops.AFTER dtypes.f32 [1, 25, 48]
50 Ops.CONST dtypes.weakint [] 128
51 Ops.MUL dtypes.weakint [25, 50]
52 Ops.ADD dtypes.weakint [51, 48]
53 Ops.INDEX dtypes.f32 [49, 52]
54 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
55 Ops.RANGE dtypes.weakint [14] (AxisType.WEAK, 0)
56 Ops.MUL dtypes.weakint [55, 10]
57 Ops.ADD dtypes.weakint [56, 40]
58 Ops.MUL dtypes.weakint [57, 10]
59 Ops.ADD dtypes.weakint [58, 44]
60 Ops.MUL dtypes.weakint [59, 10]
61 Ops.ADD dtypes.weakint [60, 47]
62 Ops.ADD dtypes.weakint [51, 61]
63 Ops.INDEX dtypes.f16 [54, 62]
64 Ops.CONST dtypes.weakint [] 16384
65 Ops.PARAM dtypes.f16 [64] ParamArg(slot=2, dtype=dtypes.f16)
66 Ops.ADD dtypes.weakint [56, 16]
67 Ops.MUL dtypes.weakint [66, 10]
68 Ops.ADD dtypes.weakint [67, 20]
69 Ops.MUL dtypes.weakint [68, 10]
70 Ops.ADD dtypes.weakint [69, 24]
71 Ops.MUL dtypes.weakint [70, 50]
72 Ops.ADD dtypes.weakint [71, 48]
73 Ops.INDEX dtypes.f16 [65, 72]
74 Ops.CONST dtypes.weakfloat [] 0.0
75 Ops.CAST dtypes.f32 [74] dtypes.f32
76 Ops.STACK dtypes.f32 [75, 75]
77 Ops.WMMA dtypes.f32 [63, 73, 76] ((8, 8, 8), dtypes.f16, 32, ((((AxisType.UPCAST, 4), 2),), (((AxisType.UPCAST, 4), 2),), (((AxisType.UPCAST, 4), 2),)))
78 Ops.REDUCE dtypes.f32 [77, 55] (Ops.ADD, 0)
79 Ops.STORE dtypes.void [53, 78]
80 Ops.END dtypes.void [79, 6, 8, 13, 27, 29, 32, 35, 47]
81 Ops.SINK dtypes.void [80] KernelInfo(name="r_(n*4)_2_32_2_4_4_2_16", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.LOCAL))), optimized=True)

