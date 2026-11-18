# tolk D450 D451 D493 golden=4fbc8a9e2e866edb7bb5d782510e360e
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 6)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.MUL dtypes.weakint [7, 4]
9 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.MUL dtypes.weakint [10, 4]
12 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.MUL dtypes.weakint [13, 4]
15 Ops.CONST dtypes.weakint [] 32
16 Ops.RANGE dtypes.weakint [15] (AxisType.WARP, -1)
17 Ops.CONST dtypes.weakint [] 16
18 Ops.DIV dtypes.weakint [16, 17]
19 Ops.MOD dtypes.weakint [18, 4]
20 Ops.ADD dtypes.weakint [14, 19]
21 Ops.CONST dtypes.weakint [] 64
22 Ops.MUL dtypes.weakint [20, 21]
23 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 3)
24 Ops.MUL dtypes.weakint [23, 4]
25 Ops.CONST dtypes.weakint [] 8
26 Ops.DIV dtypes.weakint [16, 25]
27 Ops.MOD dtypes.weakint [26, 4]
28 Ops.ADD dtypes.weakint [24, 27]
29 Ops.MUL dtypes.weakint [28, 4]
30 Ops.DIV dtypes.weakint [16, 2]
31 Ops.MOD dtypes.weakint [30, 4]
32 Ops.ADD dtypes.weakint [29, 31]
33 Ops.MUL dtypes.weakint [32, 4]
34 Ops.DIV dtypes.weakint [16, 4]
35 Ops.MOD dtypes.weakint [34, 4]
36 Ops.ADD dtypes.weakint [33, 35]
37 Ops.MUL dtypes.weakint [36, 4]
38 Ops.CONST dtypes.weakint [] 1
39 Ops.DIV dtypes.weakint [16, 38]
40 Ops.MOD dtypes.weakint [39, 4]
41 Ops.ADD dtypes.weakint [37, 40]
42 Ops.ADD dtypes.weakint [22, 41]
43 Ops.INDEX dtypes.f32 [1, 42]
44 Ops.CONST dtypes.weakint [] 16384
45 Ops.PARAM dtypes.bf16 [44] ParamArg(slot=1, dtype=dtypes.bf16)
46 Ops.RANGE dtypes.weakint [25] (AxisType.WEAK, 1)
47 Ops.MUL dtypes.weakint [46, 15]
48 Ops.RANGE dtypes.weakint [38] (AxisType.WEAK, 0)
49 Ops.MUL dtypes.weakint [48, 4]
50 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 11)
51 Ops.ADD dtypes.weakint [49, 50]
52 Ops.MUL dtypes.weakint [51, 4]
53 Ops.ADD dtypes.weakint [52, 6]
54 Ops.MUL dtypes.weakint [53, 4]
55 Ops.ADD dtypes.weakint [54, 9]
56 Ops.MUL dtypes.weakint [55, 4]
57 Ops.ADD dtypes.weakint [56, 12]
58 Ops.MUL dtypes.weakint [57, 4]
59 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 10)
60 Ops.ADD dtypes.weakint [58, 59]
61 Ops.ADD dtypes.weakint [47, 60]
62 Ops.ADD dtypes.weakint [5, 27]
63 Ops.MUL dtypes.weakint [62, 4]
64 Ops.ADD dtypes.weakint [63, 31]
65 Ops.MUL dtypes.weakint [64, 4]
66 Ops.ADD dtypes.weakint [65, 35]
67 Ops.MUL dtypes.weakint [66, 4]
68 Ops.ADD dtypes.weakint [67, 40]
69 Ops.CONST dtypes.weakint [] 256
70 Ops.MUL dtypes.weakint [68, 69]
71 Ops.ADD dtypes.weakint [61, 70]
72 Ops.INDEX dtypes.bf16 [45, 71]
73 Ops.PARAM dtypes.i8 [44] ParamArg(slot=2, dtype=dtypes.i8)
74 Ops.MUL dtypes.weakint [41, 69]
75 Ops.ADD dtypes.weakint [61, 74]
76 Ops.INDEX dtypes.i8 [73, 75]
77 Ops.CAST dtypes.bf16 [76] dtypes.bf16
78 Ops.CONST dtypes.weakfloat [] 0.0
79 Ops.CAST dtypes.f32 [78] dtypes.f32
80 Ops.STACK dtypes.f32 [79, 79, 79, 79, 79, 79, 79, 79]
81 Ops.WMMA dtypes.f32 [72, 77, 80] ((16, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 6), 2), ((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 10), 2)), (((AxisType.UPCAST, 6), 2), ((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 10), 2)), (((AxisType.UPCAST, 6), 2), ((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 10), 1))))
82 Ops.REDUCE dtypes.f32 [81, 48, 50] (Ops.ADD, 0)
83 Ops.CONST dtypes.weakint [] 512
84 Ops.PARAM dtypes.f32 [83] ParamArg(slot=3, dtype=dtypes.f32)
85 Ops.MUL dtypes.weakint [41, 25]
86 Ops.ADD dtypes.weakint [85, 46]
87 Ops.INDEX dtypes.f32 [84, 86]
88 Ops.MUL dtypes.f32 [82, 87]
89 Ops.REDUCE dtypes.f32 [88, 46] (Ops.ADD, 0)
90 Ops.STORE dtypes.void [43, 89]
91 Ops.END dtypes.void [90, 3, 6, 9, 12, 16, 23]
92 Ops.SINK dtypes.void [91] KernelInfo(name="r_4_4_32_2_2_2_2_8_2", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=7, arg=(0, AxisType.UPCAST))), optimized=True)

