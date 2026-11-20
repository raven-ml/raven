# tolk D212 D486 golden=90865c37a7193894996ef4923b8e59b1
0 Ops.CONST dtypes.weakint [] 1000000
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 9)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 125
9 Ops.CMPLT dtypes.bool [7, 8]
10 Ops.RANGE dtypes.weakint [4] (AxisType.GLOBAL, 2)
11 Ops.MUL dtypes.weakint [10, 2]
12 Ops.RANGE dtypes.weakint [2] (AxisType.LOCAL, 8)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.CMPLT dtypes.bool [13, 8]
15 Ops.CONST dtypes.bool [] True
16 Ops.AND dtypes.bool [14, 15]
17 Ops.AND dtypes.bool [9, 16]
18 Ops.CONST dtypes.weakint [] 2
19 Ops.MUL dtypes.weakint [7, 18]
20 Ops.RANGE dtypes.weakint [18] (AxisType.UPCAST, 6)
21 Ops.ADD dtypes.weakint [19, 20]
22 Ops.CONST dtypes.weakint [] 4
23 Ops.MUL dtypes.weakint [21, 22]
24 Ops.RANGE dtypes.weakint [22] (AxisType.UPCAST, 4)
25 Ops.ADD dtypes.weakint [23, 24]
26 Ops.CONST dtypes.weakint [] 1000
27 Ops.MUL dtypes.weakint [25, 26]
28 Ops.MUL dtypes.weakint [13, 18]
29 Ops.RANGE dtypes.weakint [18] (AxisType.UPCAST, 5)
30 Ops.ADD dtypes.weakint [28, 29]
31 Ops.MUL dtypes.weakint [30, 22]
32 Ops.RANGE dtypes.weakint [22] (AxisType.UPCAST, 3)
33 Ops.ADD dtypes.weakint [31, 32]
34 Ops.ADD dtypes.weakint [27, 33]
35 Ops.CONST dtypes.bool [] Invalid
36 Ops.WHERE dtypes.weakint [17, 34, 35]
37 Ops.INDEX dtypes.f32 [1, 36]
38 Ops.CONST dtypes.weakint [] 16000
39 Ops.PARAM dtypes.f32 [38] ParamArg(slot=1, dtype=dtypes.f32)
40 Ops.AND dtypes.bool [9, 15]
41 Ops.MUL dtypes.weakint [25, 2]
42 Ops.RANGE dtypes.weakint [22] (AxisType.WEAK, 0)
43 Ops.MUL dtypes.weakint [42, 22]
44 Ops.RANGE dtypes.weakint [22] (AxisType.UPCAST, 7)
45 Ops.ADD dtypes.weakint [43, 44]
46 Ops.ADD dtypes.weakint [41, 45]
47 Ops.WHERE dtypes.weakint [40, 46, 35]
48 Ops.INDEX dtypes.f32 [39, 47]
49 Ops.PARAM dtypes.f32 [38] ParamArg(slot=2, dtype=dtypes.f32)
50 Ops.MUL dtypes.weakint [45, 26]
51 Ops.ADD dtypes.weakint [50, 33]
52 Ops.WHERE dtypes.weakint [16, 51, 35]
53 Ops.INDEX dtypes.f32 [49, 52]
54 Ops.MUL dtypes.f32 [48, 53]
55 Ops.REDUCE dtypes.f32 [54, 42, 44] (Ops.ADD, 0)
56 Ops.STORE dtypes.void [37, 55]
57 Ops.END dtypes.void [56, 3, 6, 20, 24, 10, 12, 29, 32]
58 Ops.SINK dtypes.void [57] KernelInfo(name="r_16_8_16_8_4_4_2_2_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.PADTO, axis=1, arg=16), Opt(op=OptOps.PADTO, axis=0, arg=8), Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.LOCAL))), optimized=True)

