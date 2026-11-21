# tolk D502 golden=d73b156a1a44106e4f47f32d60959564
0 Ops.CONST dtypes.weakint [] 16
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.CMPLT dtypes.bool [3, 4]
6 Ops.CONST dtypes.bool [] True
7 Ops.AND dtypes.bool [5, 6]
8 Ops.MUL dtypes.weakint [3, 4]
9 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 2)
10 Ops.ADD dtypes.weakint [8, 9]
11 Ops.CONST dtypes.bool [] Invalid
12 Ops.WHERE dtypes.weakint [7, 10, 11]
13 Ops.INDEX dtypes.f32 [1, 12]
14 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
15 Ops.CONST dtypes.weakint [] 3
16 Ops.WHERE dtypes.weakint [5, 3, 15]
17 Ops.MUL dtypes.weakint [16, 4]
18 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0)
19 Ops.ADD dtypes.weakint [17, 18]
20 Ops.INDEX dtypes.f32 [14, 19]
21 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
22 Ops.MUL dtypes.weakint [18, 4]
23 Ops.ADD dtypes.weakint [22, 9]
24 Ops.INDEX dtypes.f32 [21, 23]
25 Ops.MUL dtypes.f32 [20, 24]
26 Ops.REDUCE dtypes.f32 [25, 18] (Ops.ADD, 0)
27 Ops.STORE dtypes.void [13, 26]
28 Ops.END dtypes.void [27, 3, 9]
29 Ops.SINK dtypes.void [28] KernelInfo(name="r_8_4_4", applied_opts=(Opt(op=OptOps.PADTO, axis=0, arg=8),), optimized=True)

