# tolk D502 golden=e573e469e25c51ae3ed3c45c08444a18
0 Ops.CONST dtypes.weakint [] 289
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
4 Ops.CONST dtypes.weakint [] 17
5 Ops.CMPLT dtypes.bool [3, 4]
6 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
7 Ops.CMPLT dtypes.bool [6, 4]
8 Ops.CONST dtypes.bool [] True
9 Ops.AND dtypes.bool [7, 8]
10 Ops.AND dtypes.bool [5, 9]
11 Ops.MUL dtypes.weakint [6, 4]
12 Ops.ADD dtypes.weakint [11, 3]
13 Ops.CONST dtypes.bool [] Invalid
14 Ops.WHERE dtypes.weakint [10, 12, 13]
15 Ops.INDEX dtypes.f32 [1, 14]
16 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
17 Ops.CONST dtypes.weakint [] 16
18 Ops.WHERE dtypes.weakint [7, 6, 17]
19 Ops.MUL dtypes.weakint [18, 4]
20 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0)
21 Ops.ADD dtypes.weakint [19, 20]
22 Ops.INDEX dtypes.f32 [16, 21]
23 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
24 Ops.MUL dtypes.weakint [20, 4]
25 Ops.WHERE dtypes.weakint [5, 3, 17]
26 Ops.ADD dtypes.weakint [24, 25]
27 Ops.INDEX dtypes.f32 [23, 26]
28 Ops.MUL dtypes.f32 [22, 27]
29 Ops.REDUCE dtypes.f32 [28, 20] (Ops.ADD, 0)
30 Ops.STORE dtypes.void [15, 29]
31 Ops.END dtypes.void [30, 6, 3]
32 Ops.SINK dtypes.void [31] KernelInfo(name="r_32_32_17", applied_opts=(Opt(op=OptOps.PADTO, axis=0, arg=32), Opt(op=OptOps.PADTO, axis=1, arg=32)), optimized=True)

