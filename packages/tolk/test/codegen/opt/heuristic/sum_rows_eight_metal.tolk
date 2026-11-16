# tolk D479 D487 golden=4a0cb0d5902369ba11bf3f2e75309018
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.GLOBAL, 8)
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.CONST dtypes.weakint [] 4194304
5 Ops.PARAM dtypes.f32 [4] ParamArg(slot=1, dtype=dtypes.f32)
6 Ops.CONST dtypes.weakint [] 1024
7 Ops.MUL dtypes.weakint [2, 6]
8 Ops.CONST dtypes.weakint [] 32
9 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0)
10 Ops.MUL dtypes.weakint [9, 8]
11 Ops.RANGE dtypes.weakint [8] (AxisType.LOCAL, 9)
12 Ops.ADD dtypes.weakint [10, 11]
13 Ops.ADD dtypes.weakint [7, 12]
14 Ops.INDEX dtypes.f32 [5, 13]
15 Ops.REDUCE dtypes.f32 [14, 9, 11] (Ops.ADD, 0)
16 Ops.PARAM dtypes.f32 [4] ParamArg(slot=2, dtype=dtypes.f32)
17 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 1)
18 Ops.MUL dtypes.weakint [17, 8]
19 Ops.ADD dtypes.weakint [18, 11]
20 Ops.ADD dtypes.weakint [7, 19]
21 Ops.INDEX dtypes.f32 [16, 20]
22 Ops.REDUCE dtypes.f32 [21, 17, 11] (Ops.ADD, 0)
23 Ops.ADD dtypes.f32 [15, 22]
24 Ops.PARAM dtypes.f32 [4] ParamArg(slot=3, dtype=dtypes.f32)
25 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 2)
26 Ops.MUL dtypes.weakint [25, 8]
27 Ops.ADD dtypes.weakint [26, 11]
28 Ops.ADD dtypes.weakint [7, 27]
29 Ops.INDEX dtypes.f32 [24, 28]
30 Ops.REDUCE dtypes.f32 [29, 25, 11] (Ops.ADD, 0)
31 Ops.ADD dtypes.f32 [23, 30]
32 Ops.PARAM dtypes.f32 [4] ParamArg(slot=4, dtype=dtypes.f32)
33 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 3)
34 Ops.MUL dtypes.weakint [33, 8]
35 Ops.ADD dtypes.weakint [34, 11]
36 Ops.ADD dtypes.weakint [7, 35]
37 Ops.INDEX dtypes.f32 [32, 36]
38 Ops.REDUCE dtypes.f32 [37, 33, 11] (Ops.ADD, 0)
39 Ops.ADD dtypes.f32 [31, 38]
40 Ops.PARAM dtypes.f32 [4] ParamArg(slot=5, dtype=dtypes.f32)
41 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 4)
42 Ops.MUL dtypes.weakint [41, 8]
43 Ops.ADD dtypes.weakint [42, 11]
44 Ops.ADD dtypes.weakint [7, 43]
45 Ops.INDEX dtypes.f32 [40, 44]
46 Ops.REDUCE dtypes.f32 [45, 41, 11] (Ops.ADD, 0)
47 Ops.ADD dtypes.f32 [39, 46]
48 Ops.PARAM dtypes.f32 [4] ParamArg(slot=6, dtype=dtypes.f32)
49 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 5)
50 Ops.MUL dtypes.weakint [49, 8]
51 Ops.ADD dtypes.weakint [50, 11]
52 Ops.ADD dtypes.weakint [7, 51]
53 Ops.INDEX dtypes.f32 [48, 52]
54 Ops.REDUCE dtypes.f32 [53, 49, 11] (Ops.ADD, 0)
55 Ops.ADD dtypes.f32 [47, 54]
56 Ops.PARAM dtypes.f32 [4] ParamArg(slot=7, dtype=dtypes.f32)
57 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 6)
58 Ops.MUL dtypes.weakint [57, 8]
59 Ops.ADD dtypes.weakint [58, 11]
60 Ops.ADD dtypes.weakint [7, 59]
61 Ops.INDEX dtypes.f32 [56, 60]
62 Ops.REDUCE dtypes.f32 [61, 57, 11] (Ops.ADD, 0)
63 Ops.ADD dtypes.f32 [55, 62]
64 Ops.PARAM dtypes.f32 [4] ParamArg(slot=8, dtype=dtypes.f32)
65 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 7)
66 Ops.MUL dtypes.weakint [65, 8]
67 Ops.ADD dtypes.weakint [66, 11]
68 Ops.ADD dtypes.weakint [7, 67]
69 Ops.INDEX dtypes.f32 [64, 68]
70 Ops.REDUCE dtypes.f32 [69, 65, 11] (Ops.ADD, 0)
71 Ops.ADD dtypes.f32 [63, 70]
72 Ops.STORE dtypes.void [3, 71]
73 Ops.END dtypes.void [72, 2]
74 Ops.SINK dtypes.void [73] KernelInfo(name="r_4096_32_32_32_32_32_32_32_32_32", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(32, AxisType.LOCAL)),), optimized=True)

