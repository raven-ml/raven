# tolk D73 D188
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 3)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 4)
6 Ops.CONST dtypes.weakint [] 2
7 Ops.MUL dtypes.weakint [5, 6]
8 Ops.RANGE dtypes.weakint [6] (AxisType.UPCAST, 6)
9 Ops.ADD dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 4
11 Ops.MUL dtypes.weakint [9, 10]
12 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 5)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.AFTER dtypes.f32 [1, 3, 13]
15 Ops.CONST dtypes.weakint [] 64
16 Ops.MUL dtypes.weakint [3, 15]
17 Ops.ADD dtypes.weakint [16, 13]
18 Ops.INDEX dtypes.f32 [14, 17]
19 Ops.CONST dtypes.weakint [] 4096
20 Ops.PARAM dtypes.f32 [19] ParamArg(slot=1, dtype=dtypes.f32)
21 Ops.CONST dtypes.weakint [] 256
22 Ops.MUL dtypes.weakint [3, 21]
23 Ops.RANGE dtypes.weakint [15] (AxisType.WEAK, 2)
24 Ops.MUL dtypes.weakint [23, 10]
25 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 7)
26 Ops.ADD dtypes.weakint [24, 25]
27 Ops.ADD dtypes.weakint [22, 26]
28 Ops.INDEX dtypes.f32 [20, 27]
29 Ops.CONST dtypes.weakint [] 131072
30 Ops.PARAM dtypes.f32 [29] ParamArg(slot=2, dtype=dtypes.f32)
31 Ops.MUL dtypes.weakint [13, 21]
32 Ops.ADD dtypes.weakint [31, 26]
33 Ops.CAST dtypes.i32 [3] dtypes.i32
34 Ops.CONST dtypes.weakint [] 7
35 Ops.MUL dtypes.i32 [33, 34]
36 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 1)
37 Ops.CAST dtypes.i32 [36] dtypes.i32
38 Ops.CONST dtypes.weakint [] 3
39 Ops.MUL dtypes.i32 [37, 38]
40 Ops.ADD dtypes.i32 [35, 39]
41 Ops.CONST dtypes.weakint [] 11
42 Ops.MOD dtypes.i32 [40, 41]
43 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0)
44 Ops.CAST dtypes.i32 [43] dtypes.i32
45 Ops.MUL dtypes.i32 [44, 38]
46 Ops.ADD dtypes.i32 [35, 45]
47 Ops.MOD dtypes.i32 [46, 41]
48 Ops.REDUCE dtypes.i32 [47, 43] (Ops.MAX, 0)
49 Ops.CMPEQ dtypes.bool [42, 48]
50 Ops.CAST dtypes.i32 [49] dtypes.i32
51 Ops.CONST dtypes.weakint [] -1
52 Ops.MUL dtypes.i32 [37, 51]
53 Ops.ADD dtypes.i32 [52, 4]
54 Ops.MUL dtypes.i32 [50, 53]
55 Ops.REDUCE dtypes.i32 [54, 36] (Ops.MAX, 0)
56 Ops.MUL dtypes.i32 [55, 51]
57 Ops.ADD dtypes.i32 [4, 56]
58 Ops.CONST dtypes.weakint [] 0
59 Ops.MAX dtypes.i32 [57, 58]
60 Ops.XOR dtypes.i32 [59, 51]
61 Ops.CONST dtypes.weakint [] -8
62 Ops.MAX dtypes.i32 [60, 61]
63 Ops.XOR dtypes.i32 [62, 51]
64 Ops.CAST dtypes.weakint [63] dtypes.weakint
65 Ops.CONST dtypes.weakint [] 16384
66 Ops.MUL dtypes.weakint [64, 65]
67 Ops.ADD dtypes.weakint [32, 66]
68 Ops.INDEX dtypes.f32 [30, 67]
69 Ops.MUL dtypes.f32 [28, 68]
70 Ops.REDUCE dtypes.f32 [69, 23, 25] (Ops.ADD, 0)
71 Ops.STORE dtypes.void [18, 70]
72 Ops.END dtypes.void [71, 3, 5, 8, 12]
73 Ops.SINK dtypes.void [72] KernelInfo(name="r_16_8_4_2_8_8_64_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(4, AxisType.UPCAST))), optimized=True)

