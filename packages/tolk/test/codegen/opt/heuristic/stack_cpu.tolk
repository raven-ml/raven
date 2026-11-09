# tolk D141 golden=b8e704da83d5270f59ac4a4ea9dc0c6c
0 Ops.CONST dtypes.weakint [] 3072
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 1
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
4 Ops.CONST dtypes.weakint [] 3
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 2)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 1024
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.CONST dtypes.weakint [] 256
11 Ops.RANGE dtypes.weakint [10] (AxisType.WEAK, 1)
12 Ops.CONST dtypes.weakint [] 4
13 Ops.MUL dtypes.weakint [11, 12]
14 Ops.RANGE dtypes.weakint [12] (AxisType.UPCAST, 3)
15 Ops.ADD dtypes.weakint [13, 14]
16 Ops.ADD dtypes.weakint [9, 15]
17 Ops.INDEX dtypes.f32 [1, 16]
18 Ops.CONST dtypes.weakint [] 2
19 Ops.CMPLT dtypes.bool [7, 18]
20 Ops.MOD dtypes.weakint [7, 18]
21 Ops.CMPLT dtypes.bool [20, 2]
22 Ops.PARAM dtypes.f32 [8] ParamArg(slot=1, dtype=dtypes.f32)
23 Ops.INDEX dtypes.f32 [22, 15]
24 Ops.PARAM dtypes.f32 [8] ParamArg(slot=2, dtype=dtypes.f32)
25 Ops.INDEX dtypes.f32 [24, 15]
26 Ops.WHERE dtypes.f32 [21, 23, 25]
27 Ops.PARAM dtypes.f32 [8] ParamArg(slot=3, dtype=dtypes.f32)
28 Ops.INDEX dtypes.f32 [27, 15]
29 Ops.WHERE dtypes.f32 [19, 26, 28]
30 Ops.STORE dtypes.void [17, 29]
31 Ops.END dtypes.void [30, 3, 6, 11, 14]
32 Ops.SINK dtypes.void [31] KernelInfo(name="E_256_3_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(0, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST))), optimized=True)

