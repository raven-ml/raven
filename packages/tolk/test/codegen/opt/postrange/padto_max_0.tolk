# tolk D502 golden=6d84bea9e419fe8ccd339a5b46effcc9
0 Ops.CONST dtypes.weakint [] 17
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CMPLT dtypes.bool [3, 0]
5 Ops.CONST dtypes.bool [] True
6 Ops.AND dtypes.bool [4, 5]
7 Ops.CONST dtypes.bool [] Invalid
8 Ops.WHERE dtypes.weakint [6, 3, 7]
9 Ops.INDEX dtypes.f32 [1, 8]
10 Ops.CONST dtypes.weakint [] 324
11 Ops.PARAM dtypes.f32 [10] ParamArg(slot=1, dtype=dtypes.f32)
12 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
13 Ops.CONST dtypes.weakint [] 18
14 Ops.MUL dtypes.weakint [12, 13]
15 Ops.CONST dtypes.weakint [] 16
16 Ops.WHERE dtypes.weakint [4, 3, 15]
17 Ops.ADD dtypes.weakint [14, 16]
18 Ops.INDEX dtypes.f32 [11, 17]
19 Ops.CONST dtypes.weakfloat [] -100.0
20 Ops.MUL dtypes.f32 [18, 19]
21 Ops.REDUCE dtypes.f32 [20, 12] (Ops.MAX, 0)
22 Ops.STORE dtypes.void [9, 21]
23 Ops.END dtypes.void [22, 3]
24 Ops.SINK dtypes.void [23] KernelInfo(name="r_32_17", applied_opts=(Opt(op=OptOps.PADTO, axis=0, arg=32),), optimized=True)

