# tolk D280 golden=e573e469e25c51ae3ed3c45c08444a18
0 Ops.CONST dtypes.weakint [] 289
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
5 Ops.AFTER dtypes.f32 [1, 3, 4]
6 Ops.CONST dtypes.weakint [] 17
7 Ops.CMPLT dtypes.bool [4, 6]
8 Ops.CMPLT dtypes.bool [3, 6]
9 Ops.CONST dtypes.bool [] True
10 Ops.AND dtypes.bool [8, 9]
11 Ops.AND dtypes.bool [7, 10]
12 Ops.MUL dtypes.weakint [3, 6]
13 Ops.ADD dtypes.weakint [12, 4]
14 Ops.CONST dtypes.bool [] Invalid
15 Ops.WHERE dtypes.weakint [11, 13, 14]
16 Ops.INDEX dtypes.f32 [5, 15]
17 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
18 Ops.RANGE dtypes.weakint [6] (AxisType.WEAK, 0)
19 Ops.ADD dtypes.weakint [12, 18]
20 Ops.WHERE dtypes.weakint [10, 19, 14]
21 Ops.INDEX dtypes.f32 [17, 20]
22 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
23 Ops.AND dtypes.bool [7, 9]
24 Ops.MUL dtypes.weakint [18, 6]
25 Ops.ADD dtypes.weakint [24, 4]
26 Ops.WHERE dtypes.weakint [23, 25, 14]
27 Ops.INDEX dtypes.f32 [22, 26]
28 Ops.MUL dtypes.f32 [21, 27]
29 Ops.REDUCE dtypes.f32 [28, 18] (Ops.ADD, 0)
30 Ops.STORE dtypes.void [16, 29]
31 Ops.END dtypes.void [30, 3, 4]
32 Ops.SINK dtypes.void [31] KernelInfo(name="r_32_32_17", applied_opts=(Opt(op=OptOps.PADTO, axis=0, arg=32), Opt(op=OptOps.PADTO, axis=1, arg=32)), optimized=True)

