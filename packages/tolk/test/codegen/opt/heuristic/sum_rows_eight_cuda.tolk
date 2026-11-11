# tolk D479 golden=4a0cb0d5902369ba11bf3f2e75309018
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.GLOBAL, 8)
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.CONST dtypes.weakint [] 4194304
5 Ops.PARAM dtypes.f32 [4] ParamArg(slot=1, dtype=dtypes.f32)
6 Ops.CONST dtypes.weakint [] 1024
7 Ops.MUL dtypes.weakint [2, 6]
8 Ops.CONST dtypes.weakint [] 64
9 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0)
10 Ops.CONST dtypes.weakint [] 16
11 Ops.MUL dtypes.weakint [9, 10]
12 Ops.RANGE dtypes.weakint [10] (AxisType.LOCAL, 9)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.ADD dtypes.weakint [7, 13]
15 Ops.INDEX dtypes.f32 [5, 14]
16 Ops.REDUCE dtypes.f32 [15, 9, 12] (Ops.ADD, 0)
17 Ops.PARAM dtypes.f32 [4] ParamArg(slot=2, dtype=dtypes.f32)
18 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 1)
19 Ops.MUL dtypes.weakint [18, 10]
20 Ops.ADD dtypes.weakint [19, 12]
21 Ops.ADD dtypes.weakint [7, 20]
22 Ops.INDEX dtypes.f32 [17, 21]
23 Ops.REDUCE dtypes.f32 [22, 18, 12] (Ops.ADD, 0)
24 Ops.ADD dtypes.f32 [16, 23]
25 Ops.PARAM dtypes.f32 [4] ParamArg(slot=3, dtype=dtypes.f32)
26 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 2)
27 Ops.MUL dtypes.weakint [26, 10]
28 Ops.ADD dtypes.weakint [27, 12]
29 Ops.ADD dtypes.weakint [7, 28]
30 Ops.INDEX dtypes.f32 [25, 29]
31 Ops.REDUCE dtypes.f32 [30, 26, 12] (Ops.ADD, 0)
32 Ops.ADD dtypes.f32 [24, 31]
33 Ops.PARAM dtypes.f32 [4] ParamArg(slot=4, dtype=dtypes.f32)
34 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 3)
35 Ops.MUL dtypes.weakint [34, 10]
36 Ops.ADD dtypes.weakint [35, 12]
37 Ops.ADD dtypes.weakint [7, 36]
38 Ops.INDEX dtypes.f32 [33, 37]
39 Ops.REDUCE dtypes.f32 [38, 34, 12] (Ops.ADD, 0)
40 Ops.ADD dtypes.f32 [32, 39]
41 Ops.PARAM dtypes.f32 [4] ParamArg(slot=5, dtype=dtypes.f32)
42 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 4)
43 Ops.MUL dtypes.weakint [42, 10]
44 Ops.ADD dtypes.weakint [43, 12]
45 Ops.ADD dtypes.weakint [7, 44]
46 Ops.INDEX dtypes.f32 [41, 45]
47 Ops.REDUCE dtypes.f32 [46, 42, 12] (Ops.ADD, 0)
48 Ops.ADD dtypes.f32 [40, 47]
49 Ops.PARAM dtypes.f32 [4] ParamArg(slot=6, dtype=dtypes.f32)
50 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 5)
51 Ops.MUL dtypes.weakint [50, 10]
52 Ops.ADD dtypes.weakint [51, 12]
53 Ops.ADD dtypes.weakint [7, 52]
54 Ops.INDEX dtypes.f32 [49, 53]
55 Ops.REDUCE dtypes.f32 [54, 50, 12] (Ops.ADD, 0)
56 Ops.ADD dtypes.f32 [48, 55]
57 Ops.PARAM dtypes.f32 [4] ParamArg(slot=7, dtype=dtypes.f32)
58 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 6)
59 Ops.MUL dtypes.weakint [58, 10]
60 Ops.ADD dtypes.weakint [59, 12]
61 Ops.ADD dtypes.weakint [7, 60]
62 Ops.INDEX dtypes.f32 [57, 61]
63 Ops.REDUCE dtypes.f32 [62, 58, 12] (Ops.ADD, 0)
64 Ops.ADD dtypes.f32 [56, 63]
65 Ops.PARAM dtypes.f32 [4] ParamArg(slot=8, dtype=dtypes.f32)
66 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 7)
67 Ops.MUL dtypes.weakint [66, 10]
68 Ops.ADD dtypes.weakint [67, 12]
69 Ops.ADD dtypes.weakint [7, 68]
70 Ops.INDEX dtypes.f32 [65, 69]
71 Ops.REDUCE dtypes.f32 [70, 66, 12] (Ops.ADD, 0)
72 Ops.ADD dtypes.f32 [64, 71]
73 Ops.STORE dtypes.void [3, 72]
74 Ops.END dtypes.void [73, 2]
75 Ops.SINK dtypes.void [74] KernelInfo(name="r_4096_16_64_64_64_64_64_64_64_64", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.LOCAL)),), optimized=True)

