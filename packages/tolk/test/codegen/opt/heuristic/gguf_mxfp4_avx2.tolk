# tolk D141 golden=6ca6d2a8f2d3b2d79b52fb77f2079749
0 Ops.CONST dtypes.weakint [] 64
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 2)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.INDEX dtypes.f32 [1, 7]
9 Ops.CONST dtypes.weakint [] 256
10 Ops.PARAM dtypes.f32 [9] ParamArg(slot=1, dtype=dtypes.f32)
11 Ops.CONST dtypes.weakint [] 8
12 Ops.RANGE dtypes.weakint [11] (AxisType.WEAK, 0, 0, 0)
13 Ops.CONST dtypes.weakint [] 32
14 Ops.MUL dtypes.weakint [12, 13]
15 Ops.CONST dtypes.weakint [] 2
16 Ops.RANGE dtypes.weakint [15] (AxisType.WEAK, 0, 0, 1)
17 Ops.MUL dtypes.weakint [16, 2]
18 Ops.ADD dtypes.weakint [14, 17]
19 Ops.CONST dtypes.weakint [] 1
20 Ops.RANGE dtypes.weakint [19] (AxisType.WEAK, 0, 1)
21 Ops.MUL dtypes.weakint [20, 2]
22 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 3)
23 Ops.ADD dtypes.weakint [21, 22]
24 Ops.ADD dtypes.weakint [18, 23]
25 Ops.INDEX dtypes.f32 [10, 24]
26 Ops.CMPLT dtypes.bool [16, 19]
27 Ops.CONST dtypes.weakint [] 8704
28 Ops.PARAM dtypes.u8 [27] ParamArg(slot=2, dtype=dtypes.u8)
29 Ops.CONST dtypes.weakint [] 17
30 Ops.MUL dtypes.weakint [12, 29]
31 Ops.ADD dtypes.weakint [30, 23]
32 Ops.CONST dtypes.weakint [] 136
33 Ops.MUL dtypes.weakint [7, 32]
34 Ops.ADD dtypes.weakint [31, 33]
35 Ops.ADD dtypes.weakint [34, 19]
36 Ops.INDEX dtypes.u8 [28, 35]
37 Ops.CONST dtypes.weakint [] 15
38 Ops.AND dtypes.u8 [36, 37]
39 Ops.SHR dtypes.u8 [36, 4]
40 Ops.WHERE dtypes.u8 [26, 38, 39]
41 Ops.CAST dtypes.f32 [40] dtypes.f32
42 Ops.ADD dtypes.weakint [30, 33]
43 Ops.INDEX dtypes.u8 [28, 42]
44 Ops.CAST dtypes.f32 [43] dtypes.f32
45 Ops.MUL dtypes.f32 [41, 44]
46 Ops.MUL dtypes.f32 [25, 45]
47 Ops.REDUCE dtypes.f32 [46, 12, 16, 20, 22] (Ops.ADD, 0)
48 Ops.STORE dtypes.void [8, 47]
49 Ops.END dtypes.void [48, 3, 6]
50 Ops.SINK dtypes.void [49] KernelInfo(name="r_16_4_8_2_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(0, AxisType.UPCAST))), optimized=True)

