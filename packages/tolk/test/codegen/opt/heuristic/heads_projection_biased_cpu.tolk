# tolk D141 D188 D502 golden=891fe40cb84d09a8f39f58aea81ee2f1
0 Ops.CONST dtypes.weakint [] 1802240
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 7
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 4)
4 Ops.CONST dtypes.weakint [] 16
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 5)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 110
9 Ops.CMPLT dtypes.bool [7, 8]
10 Ops.CONST dtypes.bool [] True
11 Ops.AND dtypes.bool [9, 10]
12 Ops.CONST dtypes.weakint [] 128
13 Ops.RANGE dtypes.weakint [12] (AxisType.WEAK, 3)
14 Ops.CONST dtypes.weakint [] 2
15 Ops.MUL dtypes.weakint [13, 14]
16 Ops.RANGE dtypes.weakint [14] (AxisType.UPCAST, 6)
17 Ops.ADD dtypes.weakint [15, 16]
18 Ops.MUL dtypes.weakint [17, 8]
19 Ops.ADD dtypes.weakint [18, 7]
20 Ops.CONST dtypes.weakint [] 4
21 Ops.RANGE dtypes.weakint [20] (AxisType.WEAK, 2)
22 Ops.CONST dtypes.weakint [] 28160
23 Ops.MUL dtypes.weakint [21, 22]
24 Ops.ADD dtypes.weakint [19, 23]
25 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 1)
26 Ops.CONST dtypes.weakint [] 112640
27 Ops.MUL dtypes.weakint [25, 26]
28 Ops.ADD dtypes.weakint [24, 27]
29 Ops.CONST dtypes.bool [] Invalid
30 Ops.WHERE dtypes.weakint [11, 28, 29]
31 Ops.INDEX dtypes.f32 [1, 30]
32 Ops.CONST dtypes.weakint [] 262144
33 Ops.PARAM dtypes.f32 [32] ParamArg(slot=1, dtype=dtypes.f32)
34 Ops.CONST dtypes.weakint [] 64
35 Ops.MUL dtypes.weakint [17, 34]
36 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0)
37 Ops.MUL dtypes.weakint [36, 20]
38 Ops.RANGE dtypes.weakint [20] (AxisType.UPCAST, 7)
39 Ops.ADD dtypes.weakint [37, 38]
40 Ops.ADD dtypes.weakint [35, 39]
41 Ops.CONST dtypes.weakint [] 16384
42 Ops.MUL dtypes.weakint [25, 41]
43 Ops.ADD dtypes.weakint [40, 42]
44 Ops.INDEX dtypes.f32 [33, 43]
45 Ops.PARAM dtypes.f32 [22] ParamArg(slot=2, dtype=dtypes.f32)
46 Ops.MUL dtypes.weakint [21, 8]
47 Ops.CONST dtypes.weakint [] 109
48 Ops.WHERE dtypes.weakint [9, 7, 47]
49 Ops.ADD dtypes.weakint [46, 48]
50 Ops.CONST dtypes.weakint [] 440
51 Ops.MUL dtypes.weakint [39, 50]
52 Ops.ADD dtypes.weakint [49, 51]
53 Ops.INDEX dtypes.f32 [45, 52]
54 Ops.MUL dtypes.f32 [44, 53]
55 Ops.REDUCE dtypes.f32 [54, 36, 38] (Ops.ADD, 0)
56 Ops.PARAM dtypes.f32 [20] ParamArg(slot=3, dtype=dtypes.f32)
57 Ops.INDEX dtypes.f32 [56, 21]
58 Ops.ADD dtypes.f32 [55, 57]
59 Ops.STORE dtypes.void [31, 58]
60 Ops.END dtypes.void [59, 25, 21, 13, 16, 3, 6]
61 Ops.SINK dtypes.void [60] KernelInfo(name="r_16_4_128_7_16_2_16_4", applied_opts=(Opt(op=OptOps.PADTO, axis=3, arg=4), Opt(op=OptOps.SPLIT, axis=3, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(4, AxisType.UPCAST))), optimized=True)

