# tolk D280 golden=636cf43a79536dbefaa56e452a61f2f0
0 Ops.CONST dtypes.weakint [] 17
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.AFTER dtypes.f32 [1, 3]
5 Ops.CMPLT dtypes.bool [3, 0]
6 Ops.CONST dtypes.bool [] True
7 Ops.AND dtypes.bool [5, 6]
8 Ops.CONST dtypes.bool [] Invalid
9 Ops.WHERE dtypes.weakint [7, 3, 8]
10 Ops.INDEX dtypes.f32 [4, 9]
11 Ops.CONST dtypes.weakint [] 324
12 Ops.PARAM dtypes.f32 [11] ParamArg(slot=1, dtype=dtypes.f32)
13 Ops.CONST dtypes.weakint [] 18
14 Ops.MUL dtypes.weakint [3, 13]
15 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
16 Ops.ADD dtypes.weakint [14, 15]
17 Ops.WHERE dtypes.weakint [7, 16, 8]
18 Ops.INDEX dtypes.f32 [12, 17]
19 Ops.CONST dtypes.weakfloat [] -1.0
20 Ops.MUL dtypes.f32 [18, 19]
21 Ops.CONST dtypes.weakfloat [] 100.0
22 Ops.MUL dtypes.f32 [20, 21]
23 Ops.REDUCE dtypes.f32 [22, 15] (Ops.MAX, 0)
24 Ops.STORE dtypes.void [10, 23]
25 Ops.END dtypes.void [24, 3]
26 Ops.SINK dtypes.void [25] KernelInfo(name="r_32_17", applied_opts=(Opt(op=OptOps.PADTO, axis=0, arg=32),), optimized=True)

