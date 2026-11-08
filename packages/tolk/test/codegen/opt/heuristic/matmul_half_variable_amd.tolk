# tolk D89 golden=6bfb4bd45b9f559f17292b6f79b39065
0 Ops.CONST dtypes.weakint [] 1048576
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.STACK dtypes.void []
3 Ops.PARAM dtypes.weakint [2] ParamArg(slot=-1, dtype=dtypes.weakint, vmin_vmax=(1, 64), name="n", addrspace=AddrSpace.ALU)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [5] (AxisType.GLOBAL, 1)
7 Ops.CONST dtypes.weakint [] 4
8 Ops.MUL dtypes.weakint [6, 7]
9 Ops.RANGE dtypes.weakint [7] (AxisType.UPCAST, 11)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.MUL dtypes.weakint [10, 4]
12 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 6)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.MUL dtypes.weakint [13, 4]
15 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
16 Ops.ADD dtypes.weakint [14, 15]
17 Ops.MUL dtypes.weakint [16, 4]
18 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
19 Ops.ADD dtypes.weakint [17, 18]
20 Ops.MUL dtypes.weakint [19, 4]
21 Ops.CONST dtypes.weakint [] 32
22 Ops.RANGE dtypes.weakint [21] (AxisType.WARP, -1)
23 Ops.CONST dtypes.weakint [] 16
24 Ops.DIV dtypes.weakint [22, 23]
25 Ops.MOD dtypes.weakint [24, 4]
26 Ops.ADD dtypes.weakint [20, 25]
27 Ops.CONST dtypes.weakint [] 128
28 Ops.MUL dtypes.weakint [26, 27]
29 Ops.RANGE dtypes.weakint [4] (AxisType.GLOBAL, 3)
30 Ops.MUL dtypes.weakint [29, 7]
31 Ops.RANGE dtypes.weakint [7] (AxisType.LOCAL, 12)
32 Ops.ADD dtypes.weakint [30, 31]
33 Ops.MUL dtypes.weakint [32, 4]
34 Ops.CONST dtypes.weakint [] 8
35 Ops.DIV dtypes.weakint [22, 34]
36 Ops.MOD dtypes.weakint [35, 4]
37 Ops.ADD dtypes.weakint [33, 36]
38 Ops.MUL dtypes.weakint [37, 4]
39 Ops.DIV dtypes.weakint [22, 7]
40 Ops.MOD dtypes.weakint [39, 4]
41 Ops.ADD dtypes.weakint [38, 40]
42 Ops.MUL dtypes.weakint [41, 4]
43 Ops.DIV dtypes.weakint [22, 4]
44 Ops.MOD dtypes.weakint [43, 4]
45 Ops.ADD dtypes.weakint [42, 44]
46 Ops.MUL dtypes.weakint [45, 4]
47 Ops.CONST dtypes.weakint [] 1
48 Ops.DIV dtypes.weakint [22, 47]
49 Ops.MOD dtypes.weakint [48, 4]
50 Ops.ADD dtypes.weakint [46, 49]
51 Ops.ADD dtypes.weakint [28, 50]
52 Ops.INDEX dtypes.f32 [1, 51]
53 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
54 Ops.ADD dtypes.weakint [11, 36]
55 Ops.MUL dtypes.weakint [54, 4]
56 Ops.ADD dtypes.weakint [55, 40]
57 Ops.MUL dtypes.weakint [56, 4]
58 Ops.ADD dtypes.weakint [57, 44]
59 Ops.MUL dtypes.weakint [58, 4]
60 Ops.ADD dtypes.weakint [59, 49]
61 Ops.MUL dtypes.weakint [60, 27]
62 Ops.RANGE dtypes.weakint [34] (AxisType.WEAK, 0)
63 Ops.MUL dtypes.weakint [62, 4]
64 Ops.ADD dtypes.weakint [63, 12]
65 Ops.MUL dtypes.weakint [64, 4]
66 Ops.ADD dtypes.weakint [65, 15]
67 Ops.MUL dtypes.weakint [66, 4]
68 Ops.ADD dtypes.weakint [67, 18]
69 Ops.MUL dtypes.weakint [68, 4]
70 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 10)
71 Ops.ADD dtypes.weakint [69, 70]
72 Ops.ADD dtypes.weakint [61, 71]
73 Ops.INDEX dtypes.f16 [53, 72]
74 Ops.CONST dtypes.weakint [] 16384
75 Ops.PARAM dtypes.f16 [74] ParamArg(slot=2, dtype=dtypes.f16)
76 Ops.MUL dtypes.weakint [71, 27]
77 Ops.ADD dtypes.weakint [76, 50]
78 Ops.INDEX dtypes.f16 [75, 77]
79 Ops.CONST dtypes.weakfloat [] 0.0
80 Ops.CAST dtypes.f32 [79] dtypes.f32
81 Ops.STACK dtypes.f32 [80, 80, 80, 80, 80, 80, 80, 80]
82 Ops.WMMA dtypes.f32 [73, 78, 81] ((16, 16, 16), dtypes.f16, 32, ((((AxisType.UPCAST, 6), 2), ((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 10), 2)), (((AxisType.UPCAST, 6), 2), ((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 10), 2)), (((AxisType.UPCAST, 6), 2), ((AxisType.UPCAST, 5), 2), ((AxisType.UPCAST, 4), 2), ((AxisType.UPCAST, 10), 1))))
83 Ops.REDUCE dtypes.f32 [82, 62] (Ops.ADD, 0)
84 Ops.STORE dtypes.void [52, 83]
85 Ops.END dtypes.void [84, 6, 9, 12, 15, 18, 22, 29, 31]
86 Ops.SINK dtypes.void [85] KernelInfo(name="r_(n*2)_2_4_32_2_2_2_2_4_8", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.LOCAL))), optimized=True)

