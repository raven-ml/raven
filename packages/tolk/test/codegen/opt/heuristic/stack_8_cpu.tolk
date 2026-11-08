# tolk D141 D280 golden=ed5781c555b6926a6be1bdc1cd2f827a
0 Ops.CONST dtypes.weakint [] 8192
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0, 0, 0)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0, 0, 1)
9 Ops.MUL dtypes.weakint [8, 4]
10 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0, 1)
13 Ops.MUL dtypes.weakint [12, 4]
14 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 2)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.CONST dtypes.weakint [] 512
17 Ops.RANGE dtypes.weakint [16] (AxisType.WEAK, 1)
18 Ops.MUL dtypes.weakint [17, 4]
19 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
20 Ops.ADD dtypes.weakint [18, 19]
21 Ops.AFTER dtypes.f32 [1, 7, 11, 15, 20]
22 Ops.CONST dtypes.weakint [] 4096
23 Ops.MUL dtypes.weakint [7, 22]
24 Ops.CONST dtypes.weakint [] 2048
25 Ops.MUL dtypes.weakint [11, 24]
26 Ops.ADD dtypes.weakint [23, 25]
27 Ops.CONST dtypes.weakint [] 1024
28 Ops.MUL dtypes.weakint [15, 27]
29 Ops.ADD dtypes.weakint [26, 28]
30 Ops.ADD dtypes.weakint [29, 20]
31 Ops.INDEX dtypes.f32 [21, 30]
32 Ops.CMPLT dtypes.bool [7, 2]
33 Ops.CMPLT dtypes.bool [11, 2]
34 Ops.CMPLT dtypes.bool [15, 2]
35 Ops.PARAM dtypes.f32 [27] ParamArg(slot=1, dtype=dtypes.f32)
36 Ops.INDEX dtypes.f32 [35, 20]
37 Ops.PARAM dtypes.f32 [27] ParamArg(slot=2, dtype=dtypes.f32)
38 Ops.INDEX dtypes.f32 [37, 20]
39 Ops.WHERE dtypes.f32 [34, 36, 38]
40 Ops.PARAM dtypes.f32 [27] ParamArg(slot=3, dtype=dtypes.f32)
41 Ops.INDEX dtypes.f32 [40, 20]
42 Ops.PARAM dtypes.f32 [27] ParamArg(slot=4, dtype=dtypes.f32)
43 Ops.INDEX dtypes.f32 [42, 20]
44 Ops.WHERE dtypes.f32 [34, 41, 43]
45 Ops.WHERE dtypes.f32 [33, 39, 44]
46 Ops.PARAM dtypes.f32 [27] ParamArg(slot=5, dtype=dtypes.f32)
47 Ops.INDEX dtypes.f32 [46, 20]
48 Ops.PARAM dtypes.f32 [27] ParamArg(slot=6, dtype=dtypes.f32)
49 Ops.INDEX dtypes.f32 [48, 20]
50 Ops.WHERE dtypes.f32 [34, 47, 49]
51 Ops.PARAM dtypes.f32 [27] ParamArg(slot=7, dtype=dtypes.f32)
52 Ops.INDEX dtypes.f32 [51, 20]
53 Ops.PARAM dtypes.f32 [27] ParamArg(slot=8, dtype=dtypes.f32)
54 Ops.INDEX dtypes.f32 [53, 20]
55 Ops.WHERE dtypes.f32 [34, 52, 54]
56 Ops.WHERE dtypes.f32 [33, 50, 55]
57 Ops.WHERE dtypes.f32 [32, 45, 56]
58 Ops.STORE dtypes.void [31, 57]
59 Ops.END dtypes.void [58, 3, 6, 8, 10, 12, 14, 17, 19]
60 Ops.SINK dtypes.void [59] KernelInfo(name="E_512_2_2_2_2", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST))), optimized=True)

