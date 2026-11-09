# tolk D141 golden=2056fad018909c4c9a9290ad5fb37868
0 Ops.CONST dtypes.weakint [] 32768
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 4)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.CONST dtypes.weakint [] 2
6 Ops.RANGE dtypes.weakint [5] (AxisType.WEAK, 5)
7 Ops.CONST dtypes.weakint [] 16
8 Ops.MUL dtypes.weakint [6, 7]
9 Ops.RANGE dtypes.weakint [7] (AxisType.UPCAST, 6)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.ADD dtypes.weakint [4, 10]
12 Ops.CONST dtypes.weakint [] 8
13 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 3)
14 Ops.CONST dtypes.weakint [] 4
15 Ops.MUL dtypes.weakint [13, 14]
16 Ops.RANGE dtypes.weakint [14] (AxisType.UPCAST, 7)
17 Ops.ADD dtypes.weakint [15, 16]
18 Ops.CONST dtypes.weakint [] 1024
19 Ops.MUL dtypes.weakint [17, 18]
20 Ops.ADD dtypes.weakint [11, 19]
21 Ops.INDEX dtypes.f32 [1, 20]
22 Ops.CONST dtypes.weakint [] 3
23 Ops.RANGE dtypes.weakint [22] (AxisType.WEAK, 1)
24 Ops.ADD dtypes.weakint [23, 3]
25 Ops.CONST dtypes.weakint [] 1
26 Ops.CMPLT dtypes.bool [24, 25]
27 Ops.CONST dtypes.bool [] False
28 Ops.CMPEQ dtypes.bool [26, 27]
29 Ops.CONST dtypes.weakint [] 33
30 Ops.CMPLT dtypes.bool [24, 29]
31 Ops.AND dtypes.bool [28, 30]
32 Ops.RANGE dtypes.weakint [25] (AxisType.WEAK, 2)
33 Ops.MUL dtypes.weakint [32, 22]
34 Ops.RANGE dtypes.weakint [22] (AxisType.UPCAST, 8)
35 Ops.ADD dtypes.weakint [33, 34]
36 Ops.ADD dtypes.weakint [35, 10]
37 Ops.CMPLT dtypes.bool [36, 25]
38 Ops.CMPEQ dtypes.bool [37, 27]
39 Ops.CMPLT dtypes.bool [36, 29]
40 Ops.AND dtypes.bool [38, 39]
41 Ops.AND dtypes.bool [31, 40]
42 Ops.CONST dtypes.weakint [] 16384
43 Ops.PARAM dtypes.f32 [42] ParamArg(slot=1, dtype=dtypes.f32)
44 Ops.AND dtypes.bool [40, 31]
45 Ops.MUL dtypes.weakint [23, 2]
46 Ops.ADD dtypes.weakint [45, 4]
47 Ops.ADD dtypes.weakint [46, 36]
48 Ops.RANGE dtypes.weakint [7] (AxisType.WEAK, 0)
49 Ops.MUL dtypes.weakint [48, 18]
50 Ops.ADD dtypes.weakint [47, 49]
51 Ops.CONST dtypes.weakint [] -33
52 Ops.ADD dtypes.weakint [50, 51]
53 Ops.CONST dtypes.bool [] Invalid
54 Ops.WHERE dtypes.weakint [44, 52, 53]
55 Ops.INDEX dtypes.f32 [43, 54]
56 Ops.CONST dtypes.weakfloat [] 0.0
57 Ops.WHERE dtypes.f32 [41, 55, 56]
58 Ops.CONST dtypes.weakint [] 4608
59 Ops.PARAM dtypes.f32 [58] ParamArg(slot=2, dtype=dtypes.f32)
60 Ops.MUL dtypes.weakint [23, 22]
61 Ops.ADD dtypes.weakint [60, 35]
62 Ops.CONST dtypes.weakint [] 9
63 Ops.MUL dtypes.weakint [48, 62]
64 Ops.ADD dtypes.weakint [61, 63]
65 Ops.CONST dtypes.weakint [] 144
66 Ops.MUL dtypes.weakint [17, 65]
67 Ops.ADD dtypes.weakint [64, 66]
68 Ops.INDEX dtypes.f32 [59, 67]
69 Ops.MUL dtypes.f32 [57, 68]
70 Ops.REDUCE dtypes.f32 [69, 48, 23, 32, 34] (Ops.ADD, 0)
71 Ops.STORE dtypes.void [21, 70]
72 Ops.END dtypes.void [71, 13, 16, 3, 6, 9]
73 Ops.SINK dtypes.void [72] KernelInfo(name="r_8_32_2_16_4_16_3_3", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=7, arg=(0, AxisType.UPCAST))), optimized=True)

