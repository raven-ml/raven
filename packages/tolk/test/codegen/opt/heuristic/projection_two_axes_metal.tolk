# tolk D450 golden=e7fb2298d29607a9154079c0a7958f9c
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
38 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=1, dtype=dtypes.bf16)
39 Ops.MUL dtypes.weakint [20, 8]
40 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0, 1)
41 Ops.MUL dtypes.weakint [40, 4]
42 Ops.ADD dtypes.weakint [41, 26]
43 Ops.MUL dtypes.weakint [42, 4]
44 Ops.ADD dtypes.weakint [43, 31]
45 Ops.MUL dtypes.weakint [44, 4]
46 Ops.ADD dtypes.weakint [45, 34]
47 Ops.ADD dtypes.weakint [39, 46]
48 Ops.RANGE dtypes.weakint [13] (AxisType.WEAK, 0, 0)
49 Ops.CONST dtypes.weakint [] 1024
50 Ops.MUL dtypes.weakint [48, 49]
51 Ops.ADD dtypes.weakint [47, 50]
52 Ops.INDEX dtypes.bf16 [38, 51]
53 Ops.CAST dtypes.f32 [52] dtypes.f32
54 Ops.CONST dtypes.weakint [] 256
55 Ops.PARAM dtypes.f32 [54] ParamArg(slot=2, dtype=dtypes.f32)
56 Ops.MUL dtypes.weakint [20, 13]
57 Ops.ADD dtypes.weakint [56, 48]
58 Ops.INDEX dtypes.f32 [55, 57]
59 Ops.MUL dtypes.f32 [53, 58]
60 Ops.CAST dtypes.bf16 [59] dtypes.bf16
61 Ops.CAST dtypes.f32 [60] dtypes.f32
62 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=3, dtype=dtypes.bf16)
63 Ops.ADD dtypes.weakint [41, 10]
64 Ops.MUL dtypes.weakint [63, 4]
65 Ops.ADD dtypes.weakint [64, 15]
66 Ops.MUL dtypes.weakint [65, 4]
67 Ops.ADD dtypes.weakint [66, 19]
68 Ops.MUL dtypes.weakint [67, 21]
69 Ops.ADD dtypes.weakint [50, 68]
70 Ops.ADD dtypes.weakint [69, 35]
71 Ops.INDEX dtypes.bf16 [62, 70]
72 Ops.CAST dtypes.f32 [71] dtypes.f32
73 Ops.CONST dtypes.weakfloat [] 0.0
74 Ops.CAST dtypes.f32 [73] dtypes.f32
75 Ops.STACK dtypes.f32 [74, 74]
76 Ops.WMMA dtypes.f32 [61, 72, 75] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
77 Ops.REDUCE dtypes.f32 [76, 48, 40] (Ops.ADD, 0)
78 Ops.STORE dtypes.void [37, 77]
79 Ops.END dtypes.void [78, 3, 7, 23, 34]
80 Ops.SINK dtypes.void [79] KernelInfo(name="r_8_8_32_2_4_2", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), optimized=True)

