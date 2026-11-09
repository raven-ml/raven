# tolk D450 golden=e7fb2298d29607a9154079c0a7958f9c
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
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
15 Ops.CONST dtypes.weakint [] 32
16 Ops.RANGE dtypes.weakint [15] (AxisType.WARP, -1)
17 Ops.CONST dtypes.weakint [] 16
18 Ops.DIV dtypes.weakint [16, 17]
19 Ops.MOD dtypes.weakint [18, 4]
20 Ops.ADD dtypes.weakint [14, 19]
21 Ops.CONST dtypes.weakint [] 64
22 Ops.MUL dtypes.weakint [20, 21]
23 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
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
44 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=1, dtype=dtypes.bf16)
45 Ops.ADD dtypes.weakint [5, 27]
46 Ops.MUL dtypes.weakint [45, 4]
47 Ops.ADD dtypes.weakint [46, 31]
48 Ops.MUL dtypes.weakint [47, 4]
49 Ops.ADD dtypes.weakint [48, 35]
50 Ops.MUL dtypes.weakint [49, 4]
51 Ops.ADD dtypes.weakint [50, 40]
52 Ops.MUL dtypes.weakint [51, 17]
53 Ops.RANGE dtypes.weakint [38] (AxisType.WEAK, 0, 1)
54 Ops.MUL dtypes.weakint [53, 4]
55 Ops.ADD dtypes.weakint [54, 6]
56 Ops.MUL dtypes.weakint [55, 4]
57 Ops.ADD dtypes.weakint [56, 9]
58 Ops.MUL dtypes.weakint [57, 4]
59 Ops.ADD dtypes.weakint [58, 12]
60 Ops.MUL dtypes.weakint [59, 4]
61 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 9)
62 Ops.ADD dtypes.weakint [60, 61]
63 Ops.ADD dtypes.weakint [52, 62]
64 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0, 0)
65 Ops.CONST dtypes.weakint [] 1024
66 Ops.MUL dtypes.weakint [64, 65]
67 Ops.ADD dtypes.weakint [63, 66]
68 Ops.INDEX dtypes.bf16 [44, 67]
69 Ops.CAST dtypes.f32 [68] dtypes.f32
70 Ops.CONST dtypes.weakint [] 256
71 Ops.PARAM dtypes.f32 [70] ParamArg(slot=2, dtype=dtypes.f32)
72 Ops.MUL dtypes.weakint [51, 2]
73 Ops.ADD dtypes.weakint [72, 64]
74 Ops.INDEX dtypes.f32 [71, 73]
75 Ops.MUL dtypes.f32 [69, 74]
76 Ops.CAST dtypes.bf16 [75] dtypes.bf16
77 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=3, dtype=dtypes.bf16)
78 Ops.MUL dtypes.weakint [62, 21]
79 Ops.ADD dtypes.weakint [66, 78]
80 Ops.ADD dtypes.weakint [79, 41]
81 Ops.INDEX dtypes.bf16 [77, 80]
82 Ops.CONST dtypes.weakfloat [] 0.0
83 Ops.CAST dtypes.f32 [82] dtypes.f32
84 Ops.STACK dtypes.f32 [83, 83, 83, 83, 83, 83, 83, 83]
85 Ops.WMMA dtypes.f32 [76, 81, 84] ((16, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 3), 2), ((AxisType.UPCAST, 9), 1))))
86 Ops.REDUCE dtypes.f32 [85, 64, 53] (Ops.ADD, 0)
87 Ops.STORE dtypes.void [43, 86]
88 Ops.END dtypes.void [87, 3, 6, 9, 12, 16, 23]
89 Ops.SINK dtypes.void [88] KernelInfo(name="r_4_4_32_2_2_2_2_4", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), optimized=True)

