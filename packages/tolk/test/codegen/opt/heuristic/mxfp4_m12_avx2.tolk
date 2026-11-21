# tolk D73 D188 D502
0 Ops.CONST dtypes.weakint [] 768
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 2
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 2)
9 Ops.MUL dtypes.weakint [8, 4]
10 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.AFTER dtypes.f32 [1, 7, 11]
13 Ops.CONST dtypes.weakint [] 12
14 Ops.CMPLT dtypes.bool [7, 13]
15 Ops.CONST dtypes.bool [] True
16 Ops.AND dtypes.bool [14, 15]
17 Ops.CONST dtypes.weakint [] 64
18 Ops.MUL dtypes.weakint [7, 17]
19 Ops.ADD dtypes.weakint [18, 11]
20 Ops.CONST dtypes.bool [] Invalid
21 Ops.WHERE dtypes.weakint [16, 19, 20]
22 Ops.INDEX dtypes.f32 [12, 21]
23 Ops.CONST dtypes.weakint [] 3072
24 Ops.PARAM dtypes.f32 [23] ParamArg(slot=1, dtype=dtypes.f32)
25 Ops.CONST dtypes.weakint [] 32
26 Ops.RANGE dtypes.weakint [25] (AxisType.WEAK, 0, 0)
27 Ops.CONST dtypes.weakint [] 4
28 Ops.MUL dtypes.weakint [26, 27]
29 Ops.RANGE dtypes.weakint [27] (AxisType.UPCAST, 6)
30 Ops.ADD dtypes.weakint [28, 29]
31 Ops.MUL dtypes.weakint [30, 2]
32 Ops.CONST dtypes.weakint [] 1
33 Ops.RANGE dtypes.weakint [32] (AxisType.WEAK, 0, 1)
34 Ops.MUL dtypes.weakint [33, 2]
35 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 5)
36 Ops.ADD dtypes.weakint [34, 35]
37 Ops.ADD dtypes.weakint [31, 36]
38 Ops.CONST dtypes.weakint [] 11
39 Ops.WHERE dtypes.weakint [14, 7, 38]
40 Ops.CONST dtypes.weakint [] 256
41 Ops.MUL dtypes.weakint [39, 40]
42 Ops.ADD dtypes.weakint [37, 41]
43 Ops.INDEX dtypes.f32 [24, 42]
44 Ops.CONST dtypes.weakint [] 512
45 Ops.PARAM dtypes.f32 [44] ParamArg(slot=2, dtype=dtypes.f32)
46 Ops.CONST dtypes.weakint [] 8192
47 Ops.PARAM dtypes.u8 [46] ParamArg(slot=3, dtype=dtypes.u8)
48 Ops.CONST dtypes.weakint [] 128
49 Ops.MUL dtypes.weakint [11, 48]
50 Ops.ADD dtypes.weakint [49, 30]
51 Ops.INDEX dtypes.u8 [47, 50]
52 Ops.CAST dtypes.weakint [51] dtypes.weakint
53 Ops.MUL dtypes.weakint [52, 2]
54 Ops.ADD dtypes.weakint [53, 36]
55 Ops.INDEX dtypes.f32 [45, 54]
56 Ops.PARAM dtypes.f32 [44] ParamArg(slot=4, dtype=dtypes.f32)
57 Ops.MUL dtypes.weakint [11, 4]
58 Ops.CONST dtypes.weakint [] 16
59 Ops.DIV dtypes.weakint [30, 58]
60 Ops.ADD dtypes.weakint [57, 59]
61 Ops.INDEX dtypes.f32 [56, 60]
62 Ops.MUL dtypes.f32 [55, 61]
63 Ops.MUL dtypes.f32 [43, 62]
64 Ops.REDUCE dtypes.f32 [63, 26, 29, 33, 35] (Ops.ADD, 0)
65 Ops.STORE dtypes.void [22, 64]
66 Ops.END dtypes.void [65, 3, 6, 8, 10]
67 Ops.SINK dtypes.void [66] KernelInfo(name="r_2_8_8_8_32_2_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.PADTO, axis=0, arg=8), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

