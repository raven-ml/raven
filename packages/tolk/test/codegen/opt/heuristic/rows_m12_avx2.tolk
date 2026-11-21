# tolk D141 D188 D502 golden=a2ac4bd48fd954200fcf635dbcbb09f0
0 Ops.CONST dtypes.weakint [] 384
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 2
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 12
9 Ops.CMPLT dtypes.bool [7, 8]
10 Ops.CONST dtypes.bool [] True
11 Ops.AND dtypes.bool [9, 10]
12 Ops.CONST dtypes.weakint [] 32
13 Ops.MUL dtypes.weakint [7, 12]
14 Ops.CONST dtypes.weakint [] 4
15 Ops.RANGE dtypes.weakint [14] (AxisType.WEAK, 2)
16 Ops.MUL dtypes.weakint [15, 4]
17 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
18 Ops.ADD dtypes.weakint [16, 17]
19 Ops.ADD dtypes.weakint [13, 18]
20 Ops.CONST dtypes.bool [] Invalid
21 Ops.WHERE dtypes.weakint [11, 19, 20]
22 Ops.INDEX dtypes.f32 [1, 21]
23 Ops.CONST dtypes.weakint [] 3072
24 Ops.PARAM dtypes.bf16 [23] ParamArg(slot=1, dtype=dtypes.bf16)
25 Ops.CONST dtypes.weakint [] 11
26 Ops.WHERE dtypes.weakint [9, 7, 25]
27 Ops.CONST dtypes.weakint [] 256
28 Ops.MUL dtypes.weakint [26, 27]
29 Ops.CONST dtypes.weakint [] 64
30 Ops.RANGE dtypes.weakint [29] (AxisType.WEAK, 0)
31 Ops.MUL dtypes.weakint [30, 14]
32 Ops.RANGE dtypes.weakint [14] (AxisType.UPCAST, 5)
33 Ops.ADD dtypes.weakint [31, 32]
34 Ops.ADD dtypes.weakint [28, 33]
35 Ops.INDEX dtypes.bf16 [24, 34]
36 Ops.CAST dtypes.f32 [35] dtypes.f32
37 Ops.CONST dtypes.weakint [] 8192
38 Ops.PARAM dtypes.bf16 [37] ParamArg(slot=2, dtype=dtypes.bf16)
39 Ops.MUL dtypes.weakint [18, 27]
40 Ops.ADD dtypes.weakint [39, 33]
41 Ops.INDEX dtypes.bf16 [38, 40]
42 Ops.CAST dtypes.f32 [41] dtypes.f32
43 Ops.MUL dtypes.f32 [36, 42]
44 Ops.REDUCE dtypes.f32 [43, 30, 32] (Ops.ADD, 0)
45 Ops.STORE dtypes.void [22, 44]
46 Ops.END dtypes.void [45, 3, 6, 15, 17]
47 Ops.SINK dtypes.void [46] KernelInfo(name="r_2_4_8_8_64_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.PADTO, axis=0, arg=8), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=4, arg=(4, AxisType.UPCAST))), optimized=True)

