# tolk D502 golden=cbda824c95fdaa4b548f207e74833690
0 Ops.CONST dtypes.weakint [] 16
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.CMPLT dtypes.bool [3, 4]
6 Ops.CONST dtypes.bool [] True
7 Ops.AND dtypes.bool [5, 6]
8 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 1)
9 Ops.MUL dtypes.weakint [8, 4]
10 Ops.ADD dtypes.weakint [9, 3]
11 Ops.CONST dtypes.bool [] Invalid
12 Ops.WHERE dtypes.weakint [7, 10, 11]
13 Ops.INDEX dtypes.f32 [1, 12]
14 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
15 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0)
16 Ops.ADD dtypes.weakint [9, 15]
17 Ops.INDEX dtypes.f32 [14, 16]
18 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
19 Ops.MUL dtypes.weakint [15, 4]
20 Ops.CONST dtypes.weakint [] 3
21 Ops.WHERE dtypes.weakint [5, 3, 20]
22 Ops.ADD dtypes.weakint [19, 21]
23 Ops.INDEX dtypes.f32 [18, 22]
24 Ops.MUL dtypes.f32 [17, 23]
25 Ops.REDUCE dtypes.f32 [24, 15] (Ops.ADD, 0)
26 Ops.STORE dtypes.void [13, 25]
27 Ops.END dtypes.void [26, 8, 3]
28 Ops.SINK dtypes.void [27] KernelInfo(name="r_4_8_4", applied_opts=(Opt(op=OptOps.PADTO, axis=1, arg=8),), optimized=True)

