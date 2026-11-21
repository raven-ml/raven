# tolk D502 golden=6ec244f31aef6bf892b21f98ddaceb2e
0 Ops.CONST dtypes.weakint [] 17
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CMPLT dtypes.bool [3, 0]
5 Ops.CONST dtypes.bool [] True
6 Ops.AND dtypes.bool [4, 5]
7 Ops.CONST dtypes.bool [] Invalid
8 Ops.WHERE dtypes.weakint [6, 3, 7]
9 Ops.INDEX dtypes.i32 [1, 8]
10 Ops.CONST dtypes.weakfloat [] 1.0
11 Ops.CONST dtypes.weakint [] 289
12 Ops.PARAM dtypes.f32 [11] ParamArg(slot=1, dtype=dtypes.f32)
13 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
14 Ops.MUL dtypes.weakint [13, 0]
15 Ops.CONST dtypes.weakint [] 16
16 Ops.WHERE dtypes.weakint [4, 3, 15]
17 Ops.ADD dtypes.weakint [14, 16]
18 Ops.INDEX dtypes.f32 [12, 17]
19 Ops.REDUCE dtypes.f32 [18, 13] (Ops.MAX, 0)
20 Ops.CMPLT dtypes.bool [10, 19]
21 Ops.CONST dtypes.weakint [] 1
22 Ops.CONST dtypes.weakint [] 0
23 Ops.WHERE dtypes.weakint [20, 21, 22]
24 Ops.CAST dtypes.i32 [23] dtypes.i32
25 Ops.STORE dtypes.void [9, 24]
26 Ops.END dtypes.void [25, 3]
27 Ops.SINK dtypes.void [26] KernelInfo(name="r_32_17", applied_opts=(Opt(op=OptOps.PADTO, axis=0, arg=32),), optimized=True)

