# tolk D450 D451 golden=4fbc8a9e2e866edb7bb5d782510e360e
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
48 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0)
49 Ops.MUL dtypes.weakint [48, 4]
50 Ops.ADD dtypes.weakint [49, 6]
51 Ops.MUL dtypes.weakint [50, 4]
52 Ops.ADD dtypes.weakint [51, 9]
53 Ops.MUL dtypes.weakint [52, 4]
54 Ops.ADD dtypes.weakint [53, 12]
55 Ops.MUL dtypes.weakint [54, 4]
56 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 10)
57 Ops.ADD dtypes.weakint [55, 56]
58 Ops.ADD dtypes.weakint [47, 57]
59 Ops.ADD dtypes.weakint [5, 27]
60 Ops.MUL dtypes.weakint [59, 4]
61 Ops.ADD dtypes.weakint [60, 31]
62 Ops.MUL dtypes.weakint [61, 4]
63 Ops.ADD dtypes.weakint [62, 35]
64 Ops.MUL dtypes.weakint [63, 4]
65 Ops.ADD dtypes.weakint [64, 40]
66 Ops.CONST dtypes.weakint [] 256
67 Ops.MUL dtypes.weakint [65, 66]
68 Ops.ADD dtypes.weakint [58, 67]
69 Ops.INDEX dtypes.bf16 [45, 68]
70 Ops.PARAM dtypes.i8 [44] ParamArg(slot=2, dtype=dtypes.i8)
71 Ops.MUL dtypes.weakint [41, 66]
72 Ops.ADD dtypes.weakint [58, 71]
73 Ops.INDEX dtypes.i8 [70, 72]
74 Ops.CAST dtypes.bf16 [73] dtypes.bf16
75 Ops.CONST dtypes.weakfloat [] 0.0
76 Ops.CAST dtypes.f32 [75] dtypes.f32
77 Ops.STACK dtypes.f32 [76, 76, 76, 76, 76, 76, 76, 76]
78 Ops.WMMA dtypes.f32 [69, 74, 77] ((16, 16, 16), dtypes.bf16, 32, ((((AxisType.UPCAST, 6), 2), ((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 10), 2)), (((AxisType.UPCAST, 6), 2), ((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 10), 2)), (((AxisType.UPCAST, 6), 2), ((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 10), 1))))
79 Ops.REDUCE dtypes.f32 [78, 48] (Ops.ADD, 0)
80 Ops.CONST dtypes.weakint [] 512
81 Ops.PARAM dtypes.f32 [80] ParamArg(slot=3, dtype=dtypes.f32)
82 Ops.MUL dtypes.weakint [41, 25]
83 Ops.ADD dtypes.weakint [82, 46]
84 Ops.INDEX dtypes.f32 [81, 83]
85 Ops.MUL dtypes.f32 [79, 84]
86 Ops.REDUCE dtypes.f32 [85, 46] (Ops.ADD, 0)
87 Ops.STORE dtypes.void [43, 86]
88 Ops.END dtypes.void [87, 3, 6, 9, 12, 16, 23]
89 Ops.SINK dtypes.void [88] KernelInfo(name="r_4_4_32_2_2_2_2_2_8", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), optimized=True)

