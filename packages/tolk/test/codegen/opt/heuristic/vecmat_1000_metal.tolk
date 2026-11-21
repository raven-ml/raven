# tolk D89 D487 golden=063d5ef19cbc8bd2433a22bef2638998
0 Ops.CONST dtypes.weakint [] 1000
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 16
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 32
9 Ops.MUL dtypes.weakint [7, 8]
10 Ops.RANGE dtypes.weakint [8] (AxisType.LOCAL, 2)
11 Ops.ADD dtypes.weakint [9, 10]
12 Ops.CMPLT dtypes.bool [11, 0]
13 Ops.CONST dtypes.bool [] True
14 Ops.AND dtypes.bool [12, 13]
15 Ops.CONST dtypes.bool [] Invalid
16 Ops.WHERE dtypes.weakint [14, 11, 15]
17 Ops.INDEX dtypes.f32 [1, 16]
18 Ops.CONST dtypes.weakint [] 4096
19 Ops.PARAM dtypes.f32 [18] ParamArg(slot=1, dtype=dtypes.f32)
20 Ops.CONST dtypes.weakint [] 128
21 Ops.RANGE dtypes.weakint [20] (AxisType.WEAK, 0)
22 Ops.MUL dtypes.weakint [21, 8]
23 Ops.RANGE dtypes.weakint [8] (AxisType.LOCAL, 4)
24 Ops.ADD dtypes.weakint [22, 23]
25 Ops.INDEX dtypes.f32 [19, 24]
26 Ops.CONST dtypes.weakint [] 4096000
27 Ops.PARAM dtypes.f32 [26] ParamArg(slot=2, dtype=dtypes.f32)
28 Ops.MUL dtypes.weakint [24, 0]
29 Ops.ADD dtypes.weakint [28, 11]
30 Ops.WHERE dtypes.weakint [14, 29, 15]
31 Ops.INDEX dtypes.f32 [27, 30]
32 Ops.MUL dtypes.f32 [25, 31]
33 Ops.REDUCE dtypes.f32 [32, 21, 23] (Ops.ADD, 0)
34 Ops.STORE dtypes.void [17, 33]
35 Ops.END dtypes.void [34, 3, 6, 10]
36 Ops.SINK dtypes.void [35] KernelInfo(name="r_16_32_32_2_128", applied_opts=(Opt(op=OptOps.PADTO, axis=0, arg=64), Opt(op=OptOps.SPLIT, axis=0, arg=(32, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(32, AxisType.LOCAL))), optimized=True)

