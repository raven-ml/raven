# tolk D89 D487 golden=063d5ef19cbc8bd2433a22bef2638998
0 Ops.CONST dtypes.weakint [] 1000
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 63
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.CONST dtypes.weakint [] 8
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
20 Ops.CONST dtypes.weakint [] 32
21 Ops.RANGE dtypes.weakint [20] (AxisType.WEAK, 0)
22 Ops.CONST dtypes.weakint [] 128
23 Ops.MUL dtypes.weakint [21, 22]
24 Ops.RANGE dtypes.weakint [22] (AxisType.LOCAL, 4)
25 Ops.ADD dtypes.weakint [23, 24]
26 Ops.INDEX dtypes.f32 [19, 25]
27 Ops.CONST dtypes.weakint [] 4096000
28 Ops.PARAM dtypes.f32 [27] ParamArg(slot=2, dtype=dtypes.f32)
29 Ops.MUL dtypes.weakint [25, 0]
30 Ops.ADD dtypes.weakint [29, 11]
31 Ops.WHERE dtypes.weakint [14, 30, 15]
32 Ops.INDEX dtypes.f32 [28, 31]
33 Ops.MUL dtypes.f32 [26, 32]
34 Ops.REDUCE dtypes.f32 [33, 21, 24] (Ops.ADD, 0)
35 Ops.STORE dtypes.void [17, 34]
36 Ops.END dtypes.void [35, 3, 6, 10]
37 Ops.SINK dtypes.void [36] KernelInfo(name="r_63_8_128_2_32", applied_opts=(Opt(op=OptOps.PADTO, axis=0, arg=16), Opt(op=OptOps.SPLIT, axis=0, arg=(8, AxisType.LOCAL)), Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=3, arg=(128, AxisType.LOCAL))), optimized=True)

