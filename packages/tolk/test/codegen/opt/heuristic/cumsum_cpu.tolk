# tolk D141 golden=2035ccc12b4f0534ee8cdf5168399e9d
0 Ops.CONST dtypes.weakint [] 256
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 2)
6 Ops.ADD dtypes.weakint [4, 5]
7 Ops.INDEX dtypes.f32 [1, 6]
8 Ops.CONST dtypes.weakint [] 64
9 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 0)
10 Ops.CONST dtypes.weakint [] 4
11 Ops.MUL dtypes.weakint [9, 10]
12 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 3)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.ADD dtypes.weakint [13, 6]
15 Ops.CONST dtypes.weakint [] 255
16 Ops.CMPLT dtypes.bool [14, 15]
17 Ops.CONST dtypes.weakfloat [] 0.0
18 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
19 Ops.CONST dtypes.bool [] False
20 Ops.CMPEQ dtypes.bool [16, 19]
21 Ops.CONST dtypes.weakint [] -255
22 Ops.ADD dtypes.weakint [14, 21]
23 Ops.CONST dtypes.bool [] Invalid
24 Ops.WHERE dtypes.weakint [20, 22, 23]
25 Ops.INDEX dtypes.f32 [18, 24]
26 Ops.WHERE dtypes.f32 [16, 17, 25]
27 Ops.REDUCE dtypes.f32 [26, 9, 12] (Ops.ADD, 0)
28 Ops.STORE dtypes.void [7, 27]
29 Ops.END dtypes.void [28, 3, 5]
30 Ops.SINK dtypes.void [29] KernelInfo(name="r_16_16_64_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=2, arg=(4, AxisType.UPCAST))), optimized=True)

