# tolk D450 D451 golden=4fbc8a9e2e866edb7bb5d782510e360e
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
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
26 Ops.RANGE dtypes.weakint [16] (AxisType.GLOBAL, 3)
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
37 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
38 Ops.ADD dtypes.weakint [36, 37]
39 Ops.ADD dtypes.weakint [25, 38]
40 Ops.INDEX dtypes.f32 [1, 39]
41 Ops.CONST dtypes.weakint [] 16384
42 Ops.PARAM dtypes.bf16 [41] ParamArg(slot=1, dtype=dtypes.bf16)
43 Ops.RANGE dtypes.weakint [16] (AxisType.WEAK, 1)
44 Ops.MUL dtypes.weakint [43, 9]
45 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0)
46 Ops.MUL dtypes.weakint [45, 4]
47 Ops.ADD dtypes.weakint [46, 6]
48 Ops.MUL dtypes.weakint [47, 4]
49 Ops.ADD dtypes.weakint [48, 29]
50 Ops.MUL dtypes.weakint [49, 4]
51 Ops.ADD dtypes.weakint [50, 34]
52 Ops.MUL dtypes.weakint [51, 4]
53 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 9)
54 Ops.ADD dtypes.weakint [52, 53]
55 Ops.ADD dtypes.weakint [44, 54]
56 Ops.ADD dtypes.weakint [5, 37]
57 Ops.MUL dtypes.weakint [56, 4]
58 Ops.ADD dtypes.weakint [57, 13]
59 Ops.MUL dtypes.weakint [58, 4]
60 Ops.ADD dtypes.weakint [59, 18]
61 Ops.MUL dtypes.weakint [60, 4]
62 Ops.ADD dtypes.weakint [61, 22]
63 Ops.CONST dtypes.weakint [] 256
64 Ops.MUL dtypes.weakint [62, 63]
65 Ops.ADD dtypes.weakint [55, 64]
66 Ops.INDEX dtypes.bf16 [42, 65]
67 Ops.PARAM dtypes.i8 [41] ParamArg(slot=2, dtype=dtypes.i8)
68 Ops.ADD dtypes.weakint [52, 37]
69 Ops.ADD dtypes.weakint [44, 68]
70 Ops.ADD dtypes.weakint [27, 13]
71 Ops.MUL dtypes.weakint [70, 4]
72 Ops.ADD dtypes.weakint [71, 18]
73 Ops.MUL dtypes.weakint [72, 4]
74 Ops.ADD dtypes.weakint [73, 22]
75 Ops.MUL dtypes.weakint [74, 63]
76 Ops.ADD dtypes.weakint [69, 75]
77 Ops.INDEX dtypes.i8 [67, 76]
78 Ops.CAST dtypes.bf16 [77] dtypes.bf16
79 Ops.CONST dtypes.weakfloat [] 0.0
80 Ops.CAST dtypes.f32 [79] dtypes.f32
81 Ops.STACK dtypes.f32 [80, 80, 80, 80]
82 Ops.WMMA dtypes.f32 [66, 78, 81] ((8, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 9), 1)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 9), 1))))
83 Ops.REDUCE dtypes.f32 [82, 45] (Ops.ADD, 0)
84 Ops.CONST dtypes.weakint [] 512
85 Ops.PARAM dtypes.f32 [84] ParamArg(slot=3, dtype=dtypes.f32)
86 Ops.MUL dtypes.weakint [38, 16]
87 Ops.ADD dtypes.weakint [86, 43]
88 Ops.INDEX dtypes.f32 [85, 87]
89 Ops.MUL dtypes.f32 [83, 88]
90 Ops.REDUCE dtypes.f32 [89, 43] (Ops.ADD, 0)
91 Ops.STORE dtypes.void [40, 90]
92 Ops.END dtypes.void [91, 3, 6, 10, 26, 37]
93 Ops.SINK dtypes.void [92] KernelInfo(name="r_4_8_32_2_2_2_2_8", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), optimized=True)

