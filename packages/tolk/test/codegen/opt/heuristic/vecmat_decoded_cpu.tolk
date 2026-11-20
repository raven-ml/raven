# tolk D73
0 Ops.CONST dtypes.weakint [] 1024
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 128
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 4
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 2)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.AFTER dtypes.f32 [1, 11]
13 Ops.INDEX dtypes.f32 [12, 11]
14 Ops.CONST dtypes.weakint [] 4096
15 Ops.PARAM dtypes.f32 [14] ParamArg(slot=1, dtype=dtypes.f32)
16 Ops.CONST dtypes.weakint [] 256
17 Ops.RANGE dtypes.weakint [16] (AxisType.WEAK, 0, 0)
18 Ops.CONST dtypes.weakint [] 8
19 Ops.MUL dtypes.weakint [17, 18]
20 Ops.RANGE dtypes.weakint [18] (AxisType.UPCAST, 5)
21 Ops.ADD dtypes.weakint [19, 20]
22 Ops.MUL dtypes.weakint [21, 4]
23 Ops.CONST dtypes.weakint [] 1
24 Ops.RANGE dtypes.weakint [23] (AxisType.WEAK, 0, 1)
25 Ops.MUL dtypes.weakint [24, 4]
26 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
27 Ops.ADD dtypes.weakint [25, 26]
28 Ops.ADD dtypes.weakint [22, 27]
29 Ops.INDEX dtypes.f32 [15, 28]
30 Ops.CONST dtypes.weakint [] 512
31 Ops.PARAM dtypes.f32 [30] ParamArg(slot=2, dtype=dtypes.f32)
32 Ops.CONST dtypes.weakint [] 2097152
33 Ops.PARAM dtypes.u8 [32] ParamArg(slot=3, dtype=dtypes.u8)
34 Ops.CONST dtypes.weakint [] 2048
35 Ops.MUL dtypes.weakint [11, 34]
36 Ops.ADD dtypes.weakint [35, 21]
37 Ops.INDEX dtypes.u8 [33, 36]
38 Ops.CAST dtypes.weakint [37] dtypes.weakint
39 Ops.MUL dtypes.weakint [38, 4]
40 Ops.ADD dtypes.weakint [39, 27]
41 Ops.INDEX dtypes.f32 [31, 40]
42 Ops.CONST dtypes.weakint [] 131072
43 Ops.PARAM dtypes.f32 [42] ParamArg(slot=4, dtype=dtypes.f32)
44 Ops.MUL dtypes.weakint [11, 2]
45 Ops.CONST dtypes.weakint [] 16
46 Ops.DIV dtypes.weakint [21, 45]
47 Ops.ADD dtypes.weakint [44, 46]
48 Ops.INDEX dtypes.f32 [43, 47]
49 Ops.MUL dtypes.f32 [41, 48]
50 Ops.MUL dtypes.f32 [29, 49]
51 Ops.REDUCE dtypes.f32 [50, 17, 20, 24, 26] (Ops.ADD, 0)
52 Ops.STORE dtypes.void [13, 51]
53 Ops.END dtypes.void [52, 3, 6, 10]
54 Ops.SINK dtypes.void [53] KernelInfo(name="r_128_4_2_256_2_8", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(8, AxisType.UPCAST))), optimized=True)

