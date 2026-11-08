# tolk D280 golden=d99fdb238a402edd0d8466dbc7ad5355
0 Ops.CONST dtypes.weakint [] 19
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.LOCAL, 4)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 12
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.UPCAST, 3)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.AFTER dtypes.f32 [1, 11]
13 Ops.CMPLT dtypes.bool [11, 0]
14 Ops.CONST dtypes.bool [] True
15 Ops.AND dtypes.bool [13, 14]
16 Ops.CONST dtypes.bool [] Invalid
17 Ops.WHERE dtypes.weakint [15, 11, 16]
18 Ops.INDEX dtypes.f32 [12, 17]
19 Ops.CONST dtypes.weakint [] 22
20 Ops.RANGE dtypes.weakint [19] (AxisType.LOCAL, 2)
21 Ops.MUL dtypes.weakint [20, 2]
22 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
23 Ops.ADD dtypes.weakint [21, 22]
24 Ops.CMPLT dtypes.bool [23, 4]
25 Ops.CONST dtypes.bool [] False
26 Ops.CMPEQ dtypes.bool [24, 25]
27 Ops.CMPLT dtypes.bool [23, 0]
28 Ops.AND dtypes.bool [26, 27]
29 Ops.CONST dtypes.weakint [] 323
30 Ops.PARAM dtypes.f32 [29] ParamArg(slot=1, dtype=dtypes.f32)
31 Ops.AND dtypes.bool [13, 28]
32 Ops.CONST dtypes.weakint [] -19
33 Ops.MUL dtypes.weakint [23, 32]
34 Ops.ADD dtypes.weakint [33, 11]
35 Ops.CONST dtypes.weakint [] 342
36 Ops.ADD dtypes.weakint [34, 35]
37 Ops.WHERE dtypes.weakint [31, 36, 16]
38 Ops.INDEX dtypes.f32 [30, 37]
39 Ops.CONST dtypes.weakfloat [] 0.0
40 Ops.WHERE dtypes.f32 [28, 38, 39]
41 Ops.REDUCE dtypes.f32 [40, 20, 22] (Ops.ADD, 0)
42 Ops.STORE dtypes.void [18, 41]
43 Ops.END dtypes.void [42, 3, 6, 10]
44 Ops.SINK dtypes.void [43] KernelInfo(name="r_22_2_12", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(0, AxisType.LOCAL, True)), Opt(op=OptOps.PADTO, axis=0, arg=8), Opt(op=OptOps.SPLIT, axis=0, arg=(12, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.LOCAL))), optimized=True)

