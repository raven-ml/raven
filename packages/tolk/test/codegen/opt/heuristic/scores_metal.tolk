# tolk D89 D175 golden=0d7c310cc97d1447d06fa2a66250eafa
0 Ops.CONST dtypes.weakint [] 131072
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.CONST dtypes.weakint [] 32
7 Ops.RANGE dtypes.weakint [6] (AxisType.WARP, -1)
8 Ops.DIV dtypes.weakint [7, 2]
9 Ops.MOD dtypes.weakint [8, 4]
10 Ops.ADD dtypes.weakint [5, 9]
11 Ops.MUL dtypes.weakint [10, 4]
12 Ops.CONST dtypes.weakint [] 4
13 Ops.DIV dtypes.weakint [7, 12]
14 Ops.MOD dtypes.weakint [13, 4]
15 Ops.ADD dtypes.weakint [11, 14]
16 Ops.MUL dtypes.weakint [15, 4]
17 Ops.DIV dtypes.weakint [7, 4]
18 Ops.MOD dtypes.weakint [17, 4]
19 Ops.ADD dtypes.weakint [16, 18]
20 Ops.CONST dtypes.weakint [] 128
21 Ops.MUL dtypes.weakint [19, 20]
22 Ops.RANGE dtypes.weakint [12] (AxisType.GLOBAL, 3)
23 Ops.MUL dtypes.weakint [22, 12]
24 Ops.RANGE dtypes.weakint [12] (AxisType.LOCAL, 8)
25 Ops.ADD dtypes.weakint [23, 24]
26 Ops.MUL dtypes.weakint [25, 4]
27 Ops.CONST dtypes.weakint [] 8
28 Ops.DIV dtypes.weakint [7, 27]
29 Ops.MOD dtypes.weakint [28, 4]
30 Ops.ADD dtypes.weakint [26, 29]
31 Ops.MUL dtypes.weakint [30, 4]
32 Ops.CONST dtypes.weakint [] 1
33 Ops.DIV dtypes.weakint [7, 32]
34 Ops.MOD dtypes.weakint [33, 4]
35 Ops.ADD dtypes.weakint [31, 34]
36 Ops.MUL dtypes.weakint [35, 4]
37 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
38 Ops.ADD dtypes.weakint [36, 37]
39 Ops.ADD dtypes.weakint [21, 38]
40 Ops.RANGE dtypes.weakint [27] (AxisType.GLOBAL, 1)
41 Ops.CONST dtypes.weakint [] 16384
42 Ops.MUL dtypes.weakint [40, 41]
43 Ops.ADD dtypes.weakint [39, 42]
44 Ops.INDEX dtypes.f32 [1, 43]
45 Ops.PARAM dtypes.f32 [41] ParamArg(slot=1, dtype=dtypes.f32)
46 Ops.MUL dtypes.weakint [19, 2]
47 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0)
48 Ops.MUL dtypes.weakint [47, 4]
49 Ops.ADD dtypes.weakint [48, 29]
50 Ops.MUL dtypes.weakint [49, 4]
51 Ops.ADD dtypes.weakint [50, 34]
52 Ops.MUL dtypes.weakint [51, 4]
53 Ops.ADD dtypes.weakint [52, 37]
54 Ops.ADD dtypes.weakint [46, 53]
55 Ops.CONST dtypes.weakint [] 2048
56 Ops.MUL dtypes.weakint [40, 55]
57 Ops.ADD dtypes.weakint [54, 56]
58 Ops.INDEX dtypes.f32 [45, 57]
59 Ops.PARAM dtypes.f32 [41] ParamArg(slot=2, dtype=dtypes.f32)
60 Ops.MUL dtypes.weakint [38, 2]
61 Ops.ADD dtypes.weakint [48, 9]
62 Ops.MUL dtypes.weakint [61, 4]
63 Ops.ADD dtypes.weakint [62, 14]
64 Ops.MUL dtypes.weakint [63, 4]
65 Ops.ADD dtypes.weakint [64, 18]
66 Ops.ADD dtypes.weakint [60, 65]
67 Ops.ADD dtypes.weakint [66, 56]
68 Ops.INDEX dtypes.f32 [59, 67]
69 Ops.CONST dtypes.weakfloat [] 0.0
70 Ops.CAST dtypes.f32 [69] dtypes.f32
71 Ops.STACK dtypes.f32 [70, 70]
72 Ops.WMMA dtypes.f32 [58, 68, 71] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 4), 2),), (((AxisType.UPCAST, 4), 2),), (((AxisType.UPCAST, 4), 2),)))
73 Ops.REDUCE dtypes.f32 [72, 47] (Ops.ADD, 0)
74 Ops.STORE dtypes.void [44, 73]
75 Ops.END dtypes.void [74, 40, 3, 7, 22, 24, 37]
76 Ops.SINK dtypes.void [75] KernelInfo(name="r_8_16_4_4_32_2_2", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.LOCAL))), optimized=True)

