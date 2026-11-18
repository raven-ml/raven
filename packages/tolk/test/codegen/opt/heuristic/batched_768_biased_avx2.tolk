# tolk D141 golden=0d11482f4a8f27e4a93d14009fd76faf
0 Ops.CONST dtypes.weakint [] 1572864
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 512
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
4 Ops.CONST dtypes.weakint [] 768
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.CONST dtypes.weakint [] 24
7 Ops.RANGE dtypes.weakint [6] (AxisType.WEAK, 3)
8 Ops.CONST dtypes.weakint [] 2
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 5)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.CONST dtypes.weakint [] 16
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 4)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.ADD dtypes.weakint [5, 15]
17 Ops.CONST dtypes.weakint [] 4
18 Ops.RANGE dtypes.weakint [17] (AxisType.WEAK, 1)
19 Ops.CONST dtypes.weakint [] 393216
20 Ops.MUL dtypes.weakint [18, 19]
21 Ops.ADD dtypes.weakint [16, 20]
22 Ops.INDEX dtypes.f32 [1, 21]
23 Ops.CONST dtypes.weakint [] 32768
24 Ops.PARAM dtypes.f32 [23] ParamArg(slot=1, dtype=dtypes.f32)
25 Ops.MUL dtypes.weakint [3, 12]
26 Ops.CONST dtypes.weakint [] 1
27 Ops.RANGE dtypes.weakint [26] (AxisType.WEAK, 0)
28 Ops.MUL dtypes.weakint [27, 12]
29 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 6)
30 Ops.ADD dtypes.weakint [28, 29]
31 Ops.ADD dtypes.weakint [25, 30]
32 Ops.CONST dtypes.weakint [] 8192
33 Ops.MUL dtypes.weakint [18, 32]
34 Ops.ADD dtypes.weakint [31, 33]
35 Ops.INDEX dtypes.f32 [24, 34]
36 Ops.CONST dtypes.weakint [] 49152
37 Ops.PARAM dtypes.f32 [36] ParamArg(slot=2, dtype=dtypes.f32)
38 Ops.MUL dtypes.weakint [30, 4]
39 Ops.ADD dtypes.weakint [38, 15]
40 Ops.CONST dtypes.weakint [] 12288
41 Ops.MUL dtypes.weakint [18, 40]
42 Ops.ADD dtypes.weakint [39, 41]
43 Ops.INDEX dtypes.f32 [37, 42]
44 Ops.MUL dtypes.f32 [35, 43]
45 Ops.REDUCE dtypes.f32 [44, 27, 29] (Ops.ADD, 0)
46 Ops.PARAM dtypes.f32 [19] ParamArg(slot=3, dtype=dtypes.f32)
47 Ops.INDEX dtypes.f32 [46, 16]
48 Ops.ADD dtypes.f32 [45, 47]
49 Ops.STORE dtypes.void [22, 48]
50 Ops.END dtypes.void [49, 18, 3, 7, 10, 14]
51 Ops.SINK dtypes.void [50] KernelInfo(name="r_4_512_24_16_2_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=5, arg=(0, AxisType.UPCAST))), optimized=True)

