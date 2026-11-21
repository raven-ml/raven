# tolk D502 golden=01bd16e735bf1bf9fa670c80b63ec4df
0 Ops.CONST dtypes.weakint [] 17
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
2 Ops.CONST dtypes.weakint [] 4
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.CONST dtypes.weakint [] 8
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 2)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CMPLT dtypes.bool [7, 0]
9 Ops.CONST dtypes.bool [] True
10 Ops.AND dtypes.bool [8, 9]
11 Ops.CONST dtypes.bool [] Invalid
12 Ops.WHERE dtypes.weakint [10, 7, 11]
13 Ops.INDEX dtypes.i32 [1, 12]
14 Ops.CONST dtypes.weakfloat [] 1.0
15 Ops.CONST dtypes.weakint [] 289
16 Ops.PARAM dtypes.f32 [15] ParamArg(slot=1, dtype=dtypes.f32)
17 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
18 Ops.MUL dtypes.weakint [17, 0]
19 Ops.CONST dtypes.weakint [] 16
20 Ops.WHERE dtypes.weakint [8, 7, 19]
21 Ops.ADD dtypes.weakint [18, 20]
22 Ops.INDEX dtypes.f32 [16, 21]
23 Ops.REDUCE dtypes.f32 [22, 17] (Ops.MAX, 0)
24 Ops.CMPLT dtypes.bool [14, 23]
25 Ops.CONST dtypes.weakint [] 1
26 Ops.CONST dtypes.weakint [] 0
27 Ops.WHERE dtypes.weakint [24, 25, 26]
28 Ops.CAST dtypes.i32 [27] dtypes.i32
29 Ops.STORE dtypes.void [13, 28]
30 Ops.END dtypes.void [29, 3, 6]
31 Ops.SINK dtypes.void [30] KernelInfo(name="r_4_8_17", applied_opts=(Opt(op=OptOps.PADTO, axis=0, arg=32), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.UPCAST))), optimized=True)

