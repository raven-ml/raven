# tolk D141 golden=de2ecf1fc32634e3874f1e446ad88679
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 64
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.CONST dtypes.weakint [] 2
6 Ops.RANGE dtypes.weakint [5] (AxisType.WEAK, 2)
7 Ops.MUL dtypes.weakint [6, 5]
8 Ops.RANGE dtypes.weakint [5] (AxisType.UPCAST, 4)
9 Ops.ADD dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 16
11 Ops.MUL dtypes.weakint [9, 10]
12 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 3)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.ADD dtypes.weakint [4, 13]
15 Ops.INDEX dtypes.f32 [1, 14]
16 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=1, dtype=dtypes.bf16)
17 Ops.MUL dtypes.weakint [3, 10]
18 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 0, 1)
19 Ops.ADD dtypes.weakint [17, 18]
20 Ops.CONST dtypes.weakint [] 4
21 Ops.RANGE dtypes.weakint [20] (AxisType.WEAK, 0, 0)
22 Ops.CONST dtypes.weakint [] 1024
23 Ops.MUL dtypes.weakint [21, 22]
24 Ops.ADD dtypes.weakint [19, 23]
25 Ops.INDEX dtypes.bf16 [16, 24]
26 Ops.CAST dtypes.f32 [25] dtypes.f32
27 Ops.CONST dtypes.weakint [] 256
28 Ops.PARAM dtypes.f32 [27] ParamArg(slot=2, dtype=dtypes.f32)
29 Ops.MUL dtypes.weakint [3, 20]
30 Ops.ADD dtypes.weakint [29, 21]
31 Ops.INDEX dtypes.f32 [28, 30]
32 Ops.MUL dtypes.f32 [26, 31]
33 Ops.CAST dtypes.bf16 [32] dtypes.bf16
34 Ops.CAST dtypes.f32 [33] dtypes.f32
35 Ops.PARAM dtypes.bf16 [0] ParamArg(slot=3, dtype=dtypes.bf16)
36 Ops.MUL dtypes.weakint [18, 2]
37 Ops.ADD dtypes.weakint [23, 36]
38 Ops.ADD dtypes.weakint [37, 13]
39 Ops.INDEX dtypes.bf16 [35, 38]
40 Ops.CAST dtypes.f32 [39] dtypes.f32
41 Ops.MUL dtypes.f32 [34, 40]
42 Ops.REDUCE dtypes.f32 [41, 21, 18] (Ops.ADD, 0)
43 Ops.STORE dtypes.void [15, 42]
44 Ops.END dtypes.void [43, 3, 6, 8, 12]
45 Ops.SINK dtypes.void [44] KernelInfo(name="r_64_2_16_2_4_16", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST))), optimized=True)

