# tolk D141 golden=94816b5fb97e3cfe8f35f394983be008
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
12 Ops.RANGE dtypes.weakint [11] (AxisType.WEAK, 0, 0)
13 Ops.CONST dtypes.weakint [] 32
14 Ops.MUL dtypes.weakint [12, 13]
15 Ops.RANGE dtypes.weakint [13] (AxisType.WEAK, 0, 1)
16 Ops.ADD dtypes.weakint [14, 15]
17 Ops.INDEX dtypes.f32 [10, 16]
18 Ops.CONST dtypes.weakint [] 17408
19 Ops.PARAM dtypes.u8 [18] ParamArg(slot=2, dtype=dtypes.u8)
20 Ops.CONST dtypes.weakint [] 34
21 Ops.MUL dtypes.weakint [12, 20]
22 Ops.ADD dtypes.weakint [21, 15]
23 Ops.CONST dtypes.weakint [] 272
24 Ops.MUL dtypes.weakint [7, 23]
25 Ops.ADD dtypes.weakint [22, 24]
26 Ops.CONST dtypes.weakint [] 2
27 Ops.ADD dtypes.weakint [25, 26]
28 Ops.INDEX dtypes.u8 [19, 27]
29 Ops.BITCAST dtypes.i8 [28] dtypes.i8
30 Ops.CAST dtypes.f32 [29] dtypes.f32
31 Ops.ADD dtypes.weakint [21, 24]
32 Ops.INDEX dtypes.u8 [19, 31]
33 Ops.CAST dtypes.u16 [32] dtypes.u16
34 Ops.CONST dtypes.weakint [] 1
35 Ops.ADD dtypes.weakint [31, 34]
36 Ops.INDEX dtypes.u8 [19, 35]
37 Ops.CAST dtypes.u16 [36] dtypes.u16
38 Ops.MUL dtypes.u16 [37, 9]
39 Ops.OR dtypes.u16 [33, 38]
40 Ops.BITCAST dtypes.f16 [39] dtypes.f16
41 Ops.CAST dtypes.f32 [40] dtypes.f32
42 Ops.MUL dtypes.f32 [30, 41]
43 Ops.MUL dtypes.f32 [17, 42]
44 Ops.REDUCE dtypes.f32 [43, 12, 15] (Ops.ADD, 0)
45 Ops.STORE dtypes.void [8, 44]
46 Ops.END dtypes.void [45, 3, 6]
47 Ops.SINK dtypes.void [46] KernelInfo(name="r_16_4_8_32", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)

