# tolk D27 golden=069d85e70cbb0c29ca866e4fea0f8147
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 16
2 Ops.PARAM dtypes.f16 [1] ParamArg(slot=0, dtype=dtypes.f16)
3 Ops.CONST dtypes.weakint [] 4
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.RANGE dtypes.i32 [4] (AxisType.WEAK, 0)
6 Ops.CONST dtypes.weakint [] 2
7 Ops.SHL dtypes.i32 [5, 6]
8 Ops.SHRINK dtypes.f16 [2, 7, 4]
9 Ops.PARAM dtypes.u16 [1] ParamArg(slot=1, dtype=dtypes.u16)
10 Ops.INDEX dtypes.u16 [9, 7]
11 Ops.LOAD dtypes.u16 [10]
12 Ops.BITCAST dtypes.f16 [11] dtypes.f16
13 Ops.CONST dtypes.weakint [] 1
14 Ops.ADD dtypes.i32 [7, 13]
15 Ops.INDEX dtypes.u16 [9, 14]
16 Ops.LOAD dtypes.u16 [15]
17 Ops.BITCAST dtypes.f16 [16] dtypes.f16
18 Ops.ADD dtypes.i32 [7, 6]
19 Ops.INDEX dtypes.u16 [9, 18]
20 Ops.LOAD dtypes.u16 [19]
21 Ops.BITCAST dtypes.f16 [20] dtypes.f16
22 Ops.CONST dtypes.weakint [] 3
23 Ops.ADD dtypes.i32 [7, 22]
24 Ops.INDEX dtypes.u16 [9, 23]
25 Ops.LOAD dtypes.u16 [24]
26 Ops.BITCAST dtypes.f16 [25] dtypes.f16
27 Ops.STACK dtypes.f16 [12, 17, 21, 26]
28 Ops.STORE dtypes.void [8, 27]
29 Ops.AFTER dtypes.void [0, 28]
30 Ops.END dtypes.void [29, 5]
31 Ops.SINK dtypes.void [30] KernelInfo(name="E_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
32 Ops.PARAM dtypes.u16 [1] ParamArg(slot=0, dtype=dtypes.u16)
33 Ops.SHRINK dtypes.u16 [32, 7, 4]
34 Ops.STACK dtypes.u16 [11, 16, 20, 25]
35 Ops.STORE dtypes.void [33, 34]
36 Ops.AFTER dtypes.void [0, 35]
37 Ops.END dtypes.void [36, 5]
38 Ops.SINK dtypes.void [37] KernelInfo(name="E_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
39 Ops.SINK dtypes.void [31, 38]

