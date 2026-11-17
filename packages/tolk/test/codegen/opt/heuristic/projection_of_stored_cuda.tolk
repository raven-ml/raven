# tolk D89 golden=e9be97ec6bae0a765516572e4a1f1eff
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 9)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 2
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 6)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.CONST dtypes.weakint [] 4
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 4)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.CONST dtypes.weakint [] 64
17 Ops.MUL dtypes.weakint [15, 16]
18 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
19 Ops.MUL dtypes.weakint [18, 4]
20 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 8)
21 Ops.ADD dtypes.weakint [19, 20]
22 Ops.MUL dtypes.weakint [21, 8]
23 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 5)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.MUL dtypes.weakint [24, 12]
26 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 3)
27 Ops.ADD dtypes.weakint [25, 26]
28 Ops.ADD dtypes.weakint [17, 27]
29 Ops.INDEX dtypes.f32 [1, 28]
30 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
31 Ops.CONST dtypes.weakint [] 16
32 Ops.RANGE dtypes.weakint [31] (AxisType.WEAK, 0)
33 Ops.MUL dtypes.weakint [32, 12]
34 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 7)
35 Ops.ADD dtypes.weakint [33, 34]
36 Ops.ADD dtypes.weakint [17, 35]
37 Ops.INDEX dtypes.f32 [30, 36]
38 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=2, dtype=dtypes.bf16)
39 Ops.MUL dtypes.weakint [35, 16]
40 Ops.ADD dtypes.weakint [39, 27]
41 Ops.INDEX dtypes.bf16 [38, 40]
42 Ops.CAST dtypes.f32 [41] dtypes.f32
43 Ops.MUL dtypes.f32 [37, 42]
44 Ops.REDUCE dtypes.f32 [43, 32, 34] (Ops.ADD, 0)
45 Ops.STORE dtypes.void [29, 44]
46 Ops.END dtypes.void [45, 3, 6, 10, 14, 18, 20, 23, 26]
47 Ops.SINK dtypes.void [46] KernelInfo(name="r_8_8_4_4_2_2_16_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=6, arg=(4, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(8, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.LOCAL))), optimized=True)

