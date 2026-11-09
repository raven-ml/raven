# tolk D450 golden=e7fb2298d29607a9154079c0a7958f9c
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 4]
9 Ops.CONST dtypes.weakint [] 32
10 Ops.RANGE dtypes.weakint [9] (AxisType.WARP, -1)
11 Ops.CONST dtypes.weakint [] 16
12 Ops.DIV dtypes.weakint [10, 11]
13 Ops.MOD dtypes.weakint [12, 4]
14 Ops.ADD dtypes.weakint [8, 13]
15 Ops.MUL dtypes.weakint [14, 4]
16 Ops.CONST dtypes.weakint [] 8
17 Ops.DIV dtypes.weakint [10, 16]
18 Ops.MOD dtypes.weakint [17, 4]
19 Ops.ADD dtypes.weakint [15, 18]
20 Ops.MUL dtypes.weakint [19, 4]
21 Ops.DIV dtypes.weakint [10, 2]
22 Ops.MOD dtypes.weakint [21, 4]
23 Ops.ADD dtypes.weakint [20, 22]
24 Ops.CONST dtypes.weakint [] 64
25 Ops.MUL dtypes.weakint [23, 24]
26 Ops.RANGE dtypes.weakint [16] (AxisType.GLOBAL, 2)
27 Ops.MUL dtypes.weakint [26, 4]
28 Ops.DIV dtypes.weakint [10, 4]
29 Ops.MOD dtypes.weakint [28, 4]
30 Ops.ADD dtypes.weakint [27, 29]
31 Ops.MUL dtypes.weakint [30, 4]
32 Ops.CONST dtypes.weakint [] 1
33 Ops.DIV dtypes.weakint [10, 32]
34 Ops.MOD dtypes.weakint [33, 4]
35 Ops.ADD dtypes.weakint [31, 34]
36 Ops.MUL dtypes.weakint [35, 4]
37 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
38 Ops.ADD dtypes.weakint [36, 37]
39 Ops.ADD dtypes.weakint [25, 38]
40 Ops.INDEX dtypes.f32 [1, 39]
41 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=1, dtype=dtypes.bf16)
42 Ops.ADD dtypes.weakint [5, 37]
43 Ops.MUL dtypes.weakint [42, 4]
44 Ops.ADD dtypes.weakint [43, 13]
45 Ops.MUL dtypes.weakint [44, 4]
46 Ops.ADD dtypes.weakint [45, 18]
47 Ops.MUL dtypes.weakint [46, 4]
48 Ops.ADD dtypes.weakint [47, 22]
49 Ops.MUL dtypes.weakint [48, 11]
50 Ops.RANGE dtypes.weakint [32] (AxisType.WEAK, 0, 1)
51 Ops.MUL dtypes.weakint [50, 4]
52 Ops.ADD dtypes.weakint [51, 6]
53 Ops.MUL dtypes.weakint [52, 4]
54 Ops.ADD dtypes.weakint [53, 29]
55 Ops.MUL dtypes.weakint [54, 4]
56 Ops.ADD dtypes.weakint [55, 34]
57 Ops.MUL dtypes.weakint [56, 4]
58 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 8)
59 Ops.ADD dtypes.weakint [57, 58]
60 Ops.ADD dtypes.weakint [49, 59]
61 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0, 0)
62 Ops.CONST dtypes.weakint [] 1024
63 Ops.MUL dtypes.weakint [61, 62]
64 Ops.ADD dtypes.weakint [60, 63]
65 Ops.INDEX dtypes.bf16 [41, 64]
66 Ops.CAST dtypes.f32 [65] dtypes.f32
67 Ops.CONST dtypes.weakint [] 256
68 Ops.PARAM dtypes.f32 [67] ParamArg(slot=2, dtype=dtypes.f32)
69 Ops.MUL dtypes.weakint [48, 2]
70 Ops.ADD dtypes.weakint [69, 61]
71 Ops.INDEX dtypes.f32 [68, 70]
72 Ops.MUL dtypes.f32 [66, 71]
73 Ops.CAST dtypes.bf16 [72] dtypes.bf16
74 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=3, dtype=dtypes.bf16)
75 Ops.ADD dtypes.weakint [57, 37]
76 Ops.MUL dtypes.weakint [75, 24]
77 Ops.ADD dtypes.weakint [63, 76]
78 Ops.ADD dtypes.weakint [27, 13]
79 Ops.MUL dtypes.weakint [78, 4]
80 Ops.ADD dtypes.weakint [79, 18]
81 Ops.MUL dtypes.weakint [80, 4]
82 Ops.ADD dtypes.weakint [81, 22]
83 Ops.ADD dtypes.weakint [77, 82]
84 Ops.INDEX dtypes.bf16 [74, 83]
85 Ops.CONST dtypes.weakfloat [] 0.0
86 Ops.CAST dtypes.f32 [85] dtypes.f32
87 Ops.STACK dtypes.f32 [86, 86, 86, 86]
88 Ops.WMMA dtypes.f32 [73, 84, 87] ((8, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 2)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1)), (((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 8), 1))))
89 Ops.REDUCE dtypes.f32 [88, 61, 50] (Ops.ADD, 0)
90 Ops.STORE dtypes.void [40, 89]
91 Ops.END dtypes.void [90, 3, 6, 10, 26, 37]
92 Ops.SINK dtypes.void [91] KernelInfo(name="r_4_8_32_2_2_2_4", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), optimized=True)

