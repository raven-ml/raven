# tolk D175 golden=36a802d3276fe1b19838849f344e61fd
0 Ops.CONST dtypes.weakint [] 112640
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
20 Ops.CONST dtypes.weakint [] 440
21 Ops.MUL dtypes.weakint [19, 20]
22 Ops.CONST dtypes.weakint [] 55
23 Ops.RANGE dtypes.weakint [22] (AxisType.GLOBAL, 2)
24 Ops.MUL dtypes.weakint [23, 4]
25 Ops.CONST dtypes.weakint [] 8
26 Ops.DIV dtypes.weakint [6, 25]
27 Ops.MOD dtypes.weakint [26, 4]
28 Ops.ADD dtypes.weakint [24, 27]
29 Ops.MUL dtypes.weakint [28, 4]
30 Ops.CONST dtypes.weakint [] 1
31 Ops.DIV dtypes.weakint [6, 30]
32 Ops.MOD dtypes.weakint [31, 4]
33 Ops.ADD dtypes.weakint [29, 32]
34 Ops.MUL dtypes.weakint [33, 4]
35 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
36 Ops.ADD dtypes.weakint [34, 35]
37 Ops.ADD dtypes.weakint [21, 36]
38 Ops.INDEX dtypes.f32 [1, 37]
39 Ops.CONST dtypes.weakint [] 16384
40 Ops.PARAM dtypes.f32 [39] ParamArg(slot=1, dtype=dtypes.f32)
41 Ops.CONST dtypes.weakint [] 64
42 Ops.MUL dtypes.weakint [19, 41]
43 Ops.RANGE dtypes.weakint [25] (AxisType.WEAK, 0)
44 Ops.MUL dtypes.weakint [43, 4]
45 Ops.ADD dtypes.weakint [44, 27]
46 Ops.MUL dtypes.weakint [45, 4]
47 Ops.ADD dtypes.weakint [46, 32]
48 Ops.MUL dtypes.weakint [47, 4]
49 Ops.ADD dtypes.weakint [48, 35]
50 Ops.ADD dtypes.weakint [42, 49]
51 Ops.INDEX dtypes.f32 [40, 50]
52 Ops.CONST dtypes.weakint [] 28160
53 Ops.PARAM dtypes.f32 [52] ParamArg(slot=2, dtype=dtypes.f32)
54 Ops.ADD dtypes.weakint [44, 9]
55 Ops.MUL dtypes.weakint [54, 4]
56 Ops.ADD dtypes.weakint [55, 14]
57 Ops.MUL dtypes.weakint [56, 4]
58 Ops.ADD dtypes.weakint [57, 18]
59 Ops.MUL dtypes.weakint [58, 20]
60 Ops.ADD dtypes.weakint [59, 36]
61 Ops.INDEX dtypes.f32 [53, 60]
62 Ops.CONST dtypes.weakfloat [] 0.0
63 Ops.CAST dtypes.f32 [62] dtypes.f32
64 Ops.STACK dtypes.f32 [63, 63]
65 Ops.WMMA dtypes.f32 [51, 61, 64] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
66 Ops.REDUCE dtypes.f32 [65, 43] (Ops.ADD, 0)
67 Ops.STORE dtypes.void [38, 66]
68 Ops.END dtypes.void [67, 3, 6, 23, 35]
69 Ops.SINK dtypes.void [68] KernelInfo(name="r_32_55_32_2_8", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), optimized=True)

