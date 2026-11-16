# tolk D175 golden=2a583c876ca561c56fde224672f78516
0 Ops.CONST dtypes.weakint [] 262144
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 65536
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.CONST dtypes.weakint [] 8
7 Ops.RANGE dtypes.weakint [6] (AxisType.GLOBAL, 3)
8 Ops.MUL dtypes.weakint [7, 2]
9 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 9)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.CONST dtypes.weakint [] 2
12 Ops.MUL dtypes.weakint [10, 11]
13 Ops.CONST dtypes.weakint [] 32
14 Ops.RANGE dtypes.weakint [13] (AxisType.WARP, -1)
15 Ops.CONST dtypes.weakint [] 16
16 Ops.DIV dtypes.weakint [14, 15]
17 Ops.MOD dtypes.weakint [16, 11]
18 Ops.ADD dtypes.weakint [12, 17]
19 Ops.MUL dtypes.weakint [18, 11]
20 Ops.DIV dtypes.weakint [14, 2]
21 Ops.MOD dtypes.weakint [20, 11]
22 Ops.ADD dtypes.weakint [19, 21]
23 Ops.MUL dtypes.weakint [22, 11]
24 Ops.DIV dtypes.weakint [14, 11]
25 Ops.MOD dtypes.weakint [24, 11]
26 Ops.ADD dtypes.weakint [23, 25]
27 Ops.CONST dtypes.weakint [] 256
28 Ops.MUL dtypes.weakint [26, 27]
29 Ops.ADD dtypes.weakint [5, 28]
30 Ops.RANGE dtypes.weakint [6] (AxisType.GLOBAL, 4)
31 Ops.MUL dtypes.weakint [30, 2]
32 Ops.RANGE dtypes.weakint [2] (AxisType.LOCAL, 10)
33 Ops.ADD dtypes.weakint [31, 32]
34 Ops.MUL dtypes.weakint [33, 11]
35 Ops.DIV dtypes.weakint [14, 6]
36 Ops.MOD dtypes.weakint [35, 11]
37 Ops.ADD dtypes.weakint [34, 36]
38 Ops.MUL dtypes.weakint [37, 11]
39 Ops.CONST dtypes.weakint [] 1
40 Ops.DIV dtypes.weakint [14, 39]
41 Ops.MOD dtypes.weakint [40, 11]
42 Ops.ADD dtypes.weakint [38, 41]
43 Ops.MUL dtypes.weakint [42, 11]
44 Ops.RANGE dtypes.weakint [11] (AxisType.UPCAST, 5)
45 Ops.ADD dtypes.weakint [43, 44]
46 Ops.ADD dtypes.weakint [29, 45]
47 Ops.INDEX dtypes.f32 [1, 46]
48 Ops.PARAM dtypes.bool [0] ParamArg(slot=1, dtype=dtypes.bool)
49 Ops.INDEX dtypes.bool [48, 46]
50 Ops.CONST dtypes.weakint [] 16384
51 Ops.PARAM dtypes.f32 [50] ParamArg(slot=2, dtype=dtypes.f32)
52 Ops.CONST dtypes.weakint [] 4096
53 Ops.MUL dtypes.weakint [3, 52]
54 Ops.MUL dtypes.weakint [26, 15]
55 Ops.ADD dtypes.weakint [53, 54]
56 Ops.RANGE dtypes.weakint [11] (AxisType.WEAK, 0)
57 Ops.MUL dtypes.weakint [56, 11]
58 Ops.ADD dtypes.weakint [57, 36]
59 Ops.MUL dtypes.weakint [58, 11]
60 Ops.ADD dtypes.weakint [59, 41]
61 Ops.MUL dtypes.weakint [60, 11]
62 Ops.ADD dtypes.weakint [61, 44]
63 Ops.ADD dtypes.weakint [55, 62]
64 Ops.INDEX dtypes.f32 [51, 63]
65 Ops.PARAM dtypes.f32 [50] ParamArg(slot=3, dtype=dtypes.f32)
66 Ops.ADD dtypes.weakint [57, 17]
67 Ops.MUL dtypes.weakint [66, 11]
68 Ops.ADD dtypes.weakint [67, 21]
69 Ops.MUL dtypes.weakint [68, 11]
70 Ops.ADD dtypes.weakint [69, 25]
71 Ops.MUL dtypes.weakint [70, 27]
72 Ops.ADD dtypes.weakint [71, 53]
73 Ops.ADD dtypes.weakint [72, 45]
74 Ops.INDEX dtypes.f32 [65, 73]
75 Ops.CONST dtypes.weakfloat [] 0.0
76 Ops.CAST dtypes.f32 [75] dtypes.f32
77 Ops.STACK dtypes.f32 [76, 76]
78 Ops.WMMA dtypes.f32 [64, 74, 77] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 5), 2),), (((AxisType.UPCAST, 5), 2),), (((AxisType.UPCAST, 5), 2),)))
79 Ops.REDUCE dtypes.f32 [78, 56] (Ops.ADD, 0)
80 Ops.CONST dtypes.weakfloat [] -inf
81 Ops.WHERE dtypes.f32 [49, 79, 80]
82 Ops.STORE dtypes.void [47, 81]
83 Ops.END dtypes.void [82, 3, 7, 9, 14, 30, 32, 44]
84 Ops.SINK dtypes.void [83] KernelInfo(name="r_4_8_8_4_32_2_4_2", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.LOCAL))), optimized=True)

