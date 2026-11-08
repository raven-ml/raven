# tolk D27 golden=fc8a5ebfe36d5d8819d50ea25e6897eb
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 16
2 Ops.PARAM dtypes.fp8e5m2fnuz [1] ParamArg(slot=0, dtype=dtypes.fp8e5m2fnuz)
3 Ops.CONST dtypes.weakint [] 4
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.RANGE dtypes.i32 [4] (AxisType.WEAK, 0)
6 Ops.CONST dtypes.weakint [] 2
7 Ops.SHL dtypes.i32 [5, 6]
8 Ops.SHRINK dtypes.fp8e5m2fnuz [2, 7, 4]
9 Ops.PARAM dtypes.fp8e5m2fnuz [1] ParamArg(slot=1, dtype=dtypes.fp8e5m2fnuz)
10 Ops.CONST dtypes.weakint [] -4
11 Ops.MUL dtypes.i32 [5, 10]
12 Ops.CONST dtypes.weakint [] 12
13 Ops.ADD dtypes.i32 [11, 12]
14 Ops.SHRINK dtypes.fp8e5m2fnuz [9, 13, 4]
15 Ops.LOAD dtypes.fp8e5m2fnuz [14]
16 Ops.CONST dtypes.weakint [] 3
17 Ops.CAST dtypes.i32 [16] dtypes.i32
18 Ops.INDEX dtypes.fp8e5m2fnuz [15, 17]
19 Ops.CAST dtypes.i32 [6] dtypes.i32
20 Ops.INDEX dtypes.fp8e5m2fnuz [15, 19]
21 Ops.CONST dtypes.weakint [] 1
22 Ops.CAST dtypes.i32 [21] dtypes.i32
23 Ops.INDEX dtypes.fp8e5m2fnuz [15, 22]
24 Ops.CONST dtypes.weakint [] 0
25 Ops.CAST dtypes.i32 [24] dtypes.i32
26 Ops.INDEX dtypes.fp8e5m2fnuz [15, 25]
27 Ops.STACK dtypes.fp8e5m2fnuz [18, 20, 23, 26]
28 Ops.STORE dtypes.void [8, 27]
29 Ops.AFTER dtypes.void [0, 28]
30 Ops.END dtypes.void [29, 5]
31 Ops.SINK dtypes.void [30] KernelInfo(name="E_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
32 Ops.PARAM dtypes.u8 [1] ParamArg(slot=0, dtype=dtypes.u8)
33 Ops.SHRINK dtypes.u8 [32, 7, 4]
34 Ops.PARAM dtypes.u8 [1] ParamArg(slot=1, dtype=dtypes.u8)
35 Ops.SHRINK dtypes.u8 [34, 13, 4]
36 Ops.LOAD dtypes.u8 [35]
37 Ops.INDEX dtypes.u8 [36, 17]
38 Ops.INDEX dtypes.u8 [36, 19]
39 Ops.INDEX dtypes.u8 [36, 22]
40 Ops.INDEX dtypes.u8 [36, 25]
41 Ops.STACK dtypes.u8 [37, 38, 39, 40]
42 Ops.STORE dtypes.void [33, 41]
43 Ops.AFTER dtypes.void [0, 42]
44 Ops.END dtypes.void [43, 5]
45 Ops.SINK dtypes.void [44] KernelInfo(name="E_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
46 Ops.SINK dtypes.void [31, 45]

