# tolk D73 D188 D502
0 Ops.CONST dtypes.weakint [] 768
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 2
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 16
9 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 2)
10 Ops.CONST dtypes.weakint [] 4
11 Ops.MUL dtypes.weakint [9, 10]
12 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 3)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.AFTER dtypes.f32 [1, 7, 13]
15 Ops.CONST dtypes.weakint [] 12
16 Ops.CMPLT dtypes.bool [7, 15]
17 Ops.CONST dtypes.bool [] True
18 Ops.AND dtypes.bool [16, 17]
19 Ops.CONST dtypes.weakint [] 64
20 Ops.MUL dtypes.weakint [7, 19]
21 Ops.ADD dtypes.weakint [20, 13]
22 Ops.CONST dtypes.bool [] Invalid
23 Ops.WHERE dtypes.weakint [18, 21, 22]
24 Ops.INDEX dtypes.f32 [14, 23]
25 Ops.CONST dtypes.weakint [] 3072
26 Ops.PARAM dtypes.f32 [25] ParamArg(slot=1, dtype=dtypes.f32)
27 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0, 0)
28 Ops.MUL dtypes.weakint [27, 4]
29 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 6)
30 Ops.ADD dtypes.weakint [28, 29]
31 Ops.MUL dtypes.weakint [30, 2]
32 Ops.CONST dtypes.weakint [] 1
33 Ops.RANGE dtypes.weakint [32] (AxisType.WEAK, 0, 1)
34 Ops.MUL dtypes.weakint [33, 2]
35 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 5)
36 Ops.ADD dtypes.weakint [34, 35]
37 Ops.ADD dtypes.weakint [31, 36]
38 Ops.CONST dtypes.weakint [] 11
39 Ops.WHERE dtypes.weakint [16, 7, 38]
40 Ops.CONST dtypes.weakint [] 256
41 Ops.MUL dtypes.weakint [39, 40]
42 Ops.ADD dtypes.weakint [37, 41]
43 Ops.INDEX dtypes.f32 [26, 42]
44 Ops.CONST dtypes.weakint [] 512
45 Ops.PARAM dtypes.f32 [44] ParamArg(slot=2, dtype=dtypes.f32)
46 Ops.CONST dtypes.weakint [] 8192
47 Ops.PARAM dtypes.u8 [46] ParamArg(slot=3, dtype=dtypes.u8)
48 Ops.CONST dtypes.weakint [] 128
49 Ops.MUL dtypes.weakint [13, 48]
50 Ops.ADD dtypes.weakint [49, 30]
51 Ops.INDEX dtypes.u8 [47, 50]
52 Ops.CAST dtypes.weakint [51] dtypes.weakint
53 Ops.MUL dtypes.weakint [52, 2]
54 Ops.ADD dtypes.weakint [53, 36]
55 Ops.INDEX dtypes.f32 [45, 54]
56 Ops.PARAM dtypes.f32 [44] ParamArg(slot=4, dtype=dtypes.f32)
57 Ops.MUL dtypes.weakint [13, 4]
58 Ops.DIV dtypes.weakint [30, 8]
59 Ops.ADD dtypes.weakint [57, 58]
60 Ops.INDEX dtypes.f32 [56, 59]
61 Ops.MUL dtypes.f32 [55, 60]
62 Ops.MUL dtypes.f32 [43, 61]
63 Ops.REDUCE dtypes.f32 [62, 27, 29, 33, 35] (Ops.ADD, 0)
64 Ops.STORE dtypes.void [24, 63]
65 Ops.END dtypes.void [64, 3, 6, 9, 12]
66 Ops.SINK dtypes.void [65] KernelInfo(name="r_2_16_4_8_16_2_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.PADTO, axis=0, arg=8), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(8, AxisType.UPCAST))), optimized=True)

