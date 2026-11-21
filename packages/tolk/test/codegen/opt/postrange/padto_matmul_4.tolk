# tolk D490 D502 golden=0787b38b0c7da3ac7a847cc9bd492dac
0 Ops.CONST dtypes.weakint [] 289
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 32
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 2)
4 Ops.CONST dtypes.weakint [] 17
5 Ops.CMPLT dtypes.bool [3, 4]
6 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 1)
7 Ops.CMPLT dtypes.bool [6, 4]
8 Ops.CONST dtypes.bool [] True
9 Ops.AND dtypes.bool [7, 8]
10 Ops.AND dtypes.bool [5, 9]
11 Ops.MUL dtypes.weakint [6, 4]
12 Ops.ADD dtypes.weakint [11, 3]
13 Ops.CONST dtypes.bool [] Invalid
14 Ops.WHERE dtypes.weakint [10, 12, 13]
15 Ops.INDEX dtypes.f32 [1, 14]
16 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
17 Ops.CMPLT dtypes.bool [16, 4]
18 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
19 Ops.AND dtypes.bool [17, 8]
20 Ops.CONST dtypes.weakint [] 16
21 Ops.WHERE dtypes.weakint [7, 6, 20]
22 Ops.MUL dtypes.weakint [21, 4]
23 Ops.ADD dtypes.weakint [22, 16]
24 Ops.WHERE dtypes.weakint [19, 23, 13]
25 Ops.INDEX dtypes.f32 [18, 24]
26 Ops.CONST dtypes.weakfloat [] 0.0
27 Ops.CAST dtypes.f32 [26] dtypes.f32
28 Ops.WHERE dtypes.f32 [17, 25, 27]
29 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32)
30 Ops.MUL dtypes.weakint [16, 4]
31 Ops.WHERE dtypes.weakint [5, 3, 20]
32 Ops.ADD dtypes.weakint [30, 31]
33 Ops.WHERE dtypes.weakint [19, 32, 13]
34 Ops.INDEX dtypes.f32 [29, 33]
35 Ops.WHERE dtypes.f32 [17, 34, 27]
36 Ops.MUL dtypes.f32 [28, 35]
37 Ops.REDUCE dtypes.f32 [36, 16] (Ops.ADD, 0)
38 Ops.STORE dtypes.void [15, 37]
39 Ops.END dtypes.void [38, 6, 3]
40 Ops.SINK dtypes.void [39] KernelInfo(name="r_32_32_32", applied_opts=(Opt(op=OptOps.PADTO, axis=0, arg=32), Opt(op=OptOps.PADTO, axis=1, arg=32), Opt(op=OptOps.PADTO, axis=2, arg=32)), optimized=True)

