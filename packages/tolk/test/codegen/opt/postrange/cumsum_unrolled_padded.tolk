# tolk D502 golden=0b96dbe59fc611a5030848a0c7928c2e
0 Ops.CONST dtypes.weakint [] 195
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
4 Ops.CONST dtypes.weakint [] 15
5 Ops.CMPLT dtypes.bool [3, 4]
6 Ops.CONST dtypes.bool [] True
7 Ops.AND dtypes.bool [5, 6]
8 Ops.CONST dtypes.weakint [] 1
9 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 1)
10 Ops.CONST dtypes.weakint [] 13
11 Ops.MUL dtypes.weakint [9, 10]
12 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 4)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.MUL dtypes.weakint [13, 4]
15 Ops.ADD dtypes.weakint [14, 3]
16 Ops.CONST dtypes.bool [] Invalid
17 Ops.WHERE dtypes.weakint [7, 15, 16]
18 Ops.INDEX dtypes.f32 [1, 17]
19 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0)
20 Ops.MUL dtypes.weakint [19, 4]
21 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
22 Ops.ADD dtypes.weakint [20, 21]
23 Ops.ADD dtypes.weakint [22, 3]
24 Ops.CONST dtypes.weakint [] 14
25 Ops.CMPLT dtypes.bool [23, 24]
26 Ops.CONST dtypes.weakfloat [] 0.0
27 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
28 Ops.WHERE dtypes.weakint [5, 3, 24]
29 Ops.ADD dtypes.weakint [22, 28]
30 Ops.CMPLT dtypes.bool [29, 24]
31 Ops.CONST dtypes.bool [] False
32 Ops.CMPEQ dtypes.bool [30, 31]
33 Ops.ADD dtypes.weakint [29, 14]
34 Ops.CONST dtypes.weakint [] -14
35 Ops.ADD dtypes.weakint [33, 34]
36 Ops.WHERE dtypes.weakint [32, 35, 16]
37 Ops.INDEX dtypes.f32 [27, 36]
38 Ops.WHERE dtypes.f32 [25, 26, 37]
39 Ops.REDUCE dtypes.f32 [38, 19, 21] (Ops.ADD, 0)
40 Ops.STORE dtypes.void [18, 39]
41 Ops.END dtypes.void [40, 9, 12, 3]
42 Ops.SINK dtypes.void [41] KernelInfo(name="r_16_13_15", applied_opts=(Opt(op=OptOps.SPLIT, axis=2, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.PADTO, axis=0, arg=4)), optimized=True)

