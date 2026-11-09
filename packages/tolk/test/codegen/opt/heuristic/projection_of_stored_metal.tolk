# tolk D89 golden=5a25330ebac45d75968bd7f1cad34169
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.CONST dtypes.weakint [] 32
7 Ops.RANGE dtypes.weakint [6] (AxisType.WARP, -1)
8 Ops.CONST dtypes.weakint [] 16
9 Ops.DIV dtypes.weakint [7, 8]
10 Ops.MOD dtypes.weakint [9, 4]
11 Ops.ADD dtypes.weakint [5, 10]
12 Ops.MUL dtypes.weakint [11, 4]
13 Ops.CONST dtypes.weakint [] 4
14 Ops.DIV dtypes.weakint [7, 13]
15 Ops.MOD dtypes.weakint [14, 4]
16 Ops.ADD dtypes.weakint [12, 15]
17 Ops.MUL dtypes.weakint [16, 4]
18 Ops.DIV dtypes.weakint [7, 4]
19 Ops.MOD dtypes.weakint [18, 4]
20 Ops.ADD dtypes.weakint [17, 19]
21 Ops.CONST dtypes.weakint [] 64
22 Ops.MUL dtypes.weakint [20, 21]
23 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
24 Ops.MUL dtypes.weakint [23, 4]
25 Ops.DIV dtypes.weakint [7, 2]
26 Ops.MOD dtypes.weakint [25, 4]
27 Ops.ADD dtypes.weakint [24, 26]
28 Ops.MUL dtypes.weakint [27, 4]
29 Ops.CONST dtypes.weakint [] 1
30 Ops.DIV dtypes.weakint [7, 29]
31 Ops.MOD dtypes.weakint [30, 4]
32 Ops.ADD dtypes.weakint [28, 31]
33 Ops.MUL dtypes.weakint [32, 4]
34 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
35 Ops.ADD dtypes.weakint [33, 34]
36 Ops.ADD dtypes.weakint [22, 35]
37 Ops.INDEX dtypes.f32 [1, 36]
38 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
39 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
40 Ops.MUL dtypes.weakint [39, 4]
41 Ops.ADD dtypes.weakint [40, 26]
42 Ops.MUL dtypes.weakint [41, 4]
43 Ops.ADD dtypes.weakint [42, 31]
44 Ops.MUL dtypes.weakint [43, 4]
45 Ops.ADD dtypes.weakint [44, 34]
46 Ops.ADD dtypes.weakint [22, 45]
47 Ops.INDEX dtypes.f32 [38, 46]
48 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=2, dtype=dtypes.bf16)
49 Ops.ADD dtypes.weakint [40, 10]
50 Ops.MUL dtypes.weakint [49, 4]
51 Ops.ADD dtypes.weakint [50, 15]
52 Ops.MUL dtypes.weakint [51, 4]
53 Ops.ADD dtypes.weakint [52, 19]
54 Ops.MUL dtypes.weakint [53, 21]
55 Ops.ADD dtypes.weakint [54, 35]
56 Ops.INDEX dtypes.bf16 [48, 55]
57 Ops.CAST dtypes.f32 [56] dtypes.f32
58 Ops.CONST dtypes.weakfloat [] 0.0
59 Ops.CAST dtypes.f32 [58] dtypes.f32
60 Ops.STACK dtypes.f32 [59, 59]
61 Ops.WMMA dtypes.f32 [47, 57, 60] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
62 Ops.REDUCE dtypes.f32 [61, 39] (Ops.ADD, 0)
63 Ops.STORE dtypes.void [37, 62]
64 Ops.END dtypes.void [63, 3, 7, 23, 34]
65 Ops.SINK dtypes.void [64] KernelInfo(name="r_8_8_32_2_8", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), optimized=True)

