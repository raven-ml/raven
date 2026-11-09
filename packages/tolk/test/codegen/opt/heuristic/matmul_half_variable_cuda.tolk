# tolk D89 golden=1972e278a57a17e7b3b2a77b3362d000
0 Ops.CONST dtypes.weakint [] 1048576
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.STACK dtypes.void []
3 Ops.PARAM dtypes.weakint [2] ParamArg(slot=-1, dtype=dtypes.weakint, vmin_vmax=(1, 64), name="n", addrspace=AddrSpace.ALU)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [5] (AxisType.GLOBAL, 1)
7 Ops.CONST dtypes.weakint [] 4
8 Ops.MUL dtypes.weakint [6, 7]
9 Ops.RANGE dtypes.weakint [7] (AxisType.UPCAST, 10)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.MUL dtypes.weakint [10, 4]
12 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.MUL dtypes.weakint [13, 4]
15 Ops.CONST dtypes.weakint [] 32
16 Ops.RANGE dtypes.weakint [15] (AxisType.WARP, -1)
17 Ops.CONST dtypes.weakint [] 16
18 Ops.DIV dtypes.weakint [16, 17]
19 Ops.MOD dtypes.weakint [18, 4]
20 Ops.ADD dtypes.weakint [14, 19]
21 Ops.MUL dtypes.weakint [20, 4]
22 Ops.CONST dtypes.weakint [] 8
23 Ops.DIV dtypes.weakint [16, 22]
24 Ops.MOD dtypes.weakint [23, 4]
25 Ops.ADD dtypes.weakint [21, 24]
26 Ops.MUL dtypes.weakint [25, 4]
27 Ops.DIV dtypes.weakint [16, 7]
28 Ops.MOD dtypes.weakint [27, 4]
29 Ops.ADD dtypes.weakint [26, 28]
30 Ops.CONST dtypes.weakint [] 128
31 Ops.MUL dtypes.weakint [29, 30]
32 Ops.RANGE dtypes.weakint [4] (AxisType.GLOBAL, 3)
33 Ops.MUL dtypes.weakint [32, 4]
34 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 12)
35 Ops.ADD dtypes.weakint [33, 34]
36 Ops.MUL dtypes.weakint [35, 7]
37 Ops.RANGE dtypes.weakint [7] (AxisType.UPCAST, 11)
38 Ops.ADD dtypes.weakint [36, 37]
39 Ops.MUL dtypes.weakint [38, 4]
40 Ops.DIV dtypes.weakint [16, 4]
41 Ops.MOD dtypes.weakint [40, 4]
42 Ops.ADD dtypes.weakint [39, 41]
43 Ops.MUL dtypes.weakint [42, 4]
44 Ops.CONST dtypes.weakint [] 1
45 Ops.DIV dtypes.weakint [16, 44]
46 Ops.MOD dtypes.weakint [45, 4]
47 Ops.ADD dtypes.weakint [43, 46]
48 Ops.MUL dtypes.weakint [47, 4]
49 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
50 Ops.ADD dtypes.weakint [48, 49]
51 Ops.ADD dtypes.weakint [31, 50]
52 Ops.INDEX dtypes.f32 [1, 51]
53 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
54 Ops.ADD dtypes.weakint [11, 49]
55 Ops.MUL dtypes.weakint [54, 4]
56 Ops.ADD dtypes.weakint [55, 19]
57 Ops.MUL dtypes.weakint [56, 4]
58 Ops.ADD dtypes.weakint [57, 24]
59 Ops.MUL dtypes.weakint [58, 4]
60 Ops.ADD dtypes.weakint [59, 28]
61 Ops.MUL dtypes.weakint [60, 30]
62 Ops.RANGE dtypes.weakint [22] (AxisType.WEAK, 0)
63 Ops.MUL dtypes.weakint [62, 4]
64 Ops.ADD dtypes.weakint [63, 12]
65 Ops.MUL dtypes.weakint [64, 4]
66 Ops.ADD dtypes.weakint [65, 41]
67 Ops.MUL dtypes.weakint [66, 4]
68 Ops.ADD dtypes.weakint [67, 46]
69 Ops.MUL dtypes.weakint [68, 4]
70 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 9)
71 Ops.ADD dtypes.weakint [69, 70]
72 Ops.ADD dtypes.weakint [61, 71]
73 Ops.INDEX dtypes.f16 [53, 72]
74 Ops.CONST dtypes.weakint [] 16384
75 Ops.PARAM dtypes.f16 [74] ParamArg(slot=2, dtype=dtypes.f16)
76 Ops.ADD dtypes.weakint [69, 49]
77 Ops.MUL dtypes.weakint [76, 30]
78 Ops.ADD dtypes.weakint [39, 19]
79 Ops.MUL dtypes.weakint [78, 4]
80 Ops.ADD dtypes.weakint [79, 24]
81 Ops.MUL dtypes.weakint [80, 4]
82 Ops.ADD dtypes.weakint [81, 28]
83 Ops.ADD dtypes.weakint [77, 82]
84 Ops.INDEX dtypes.f16 [75, 83]
85 Ops.CONST dtypes.weakfloat [] 0.0
86 Ops.CAST dtypes.f32 [85] dtypes.f32
87 Ops.STACK dtypes.f32 [86, 86, 86, 86]
88 Ops.WMMA dtypes.f32 [73, 84, 87] ((8, 16, 16), dtypes.f16, 32, ((((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 9), 2)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 9), 1)), (((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 9), 1))))
89 Ops.REDUCE dtypes.f32 [88, 62] (Ops.ADD, 0)
90 Ops.STORE dtypes.void [52, 89]
91 Ops.END dtypes.void [90, 6, 9, 12, 16, 32, 34, 37, 49]
92 Ops.SINK dtypes.void [91] KernelInfo(name="r_(n*2)_2_2_32_2_2_2_4_4_8", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.LOCAL))), optimized=True)

