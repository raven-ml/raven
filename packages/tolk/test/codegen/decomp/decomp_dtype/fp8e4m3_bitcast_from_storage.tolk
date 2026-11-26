# tolk D27 golden=5c48978a24daf87c118ae07355f68cbf
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 16
2 Ops.PARAM dtypes.fp8e4m3 [1] ParamArg(slot=0, dtype=dtypes.fp8e4m3)
3 Ops.CONST dtypes.weakint [] 4
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.RANGE dtypes.i32 [4] (AxisType.WEAK, 0)
6 Ops.CONST dtypes.weakint [] 2
7 Ops.SHL dtypes.i32 [5, 6]
8 Ops.SHRINK dtypes.fp8e4m3 [2, 7, 4]
9 Ops.PARAM dtypes.u8 [1] ParamArg(slot=1, dtype=dtypes.u8)
10 Ops.INDEX dtypes.u8 [9, 7]
11 Ops.LOAD dtypes.u8 [10]
12 Ops.BITCAST dtypes.fp8e4m3 [11] dtypes.fp8e4m3
13 Ops.CONST dtypes.weakint [] 1
14 Ops.ADD dtypes.i32 [7, 13]
15 Ops.INDEX dtypes.u8 [9, 14]
16 Ops.LOAD dtypes.u8 [15]
17 Ops.BITCAST dtypes.fp8e4m3 [16] dtypes.fp8e4m3
18 Ops.ADD dtypes.i32 [7, 6]
19 Ops.INDEX dtypes.u8 [9, 18]
20 Ops.LOAD dtypes.u8 [19]
21 Ops.BITCAST dtypes.fp8e4m3 [20] dtypes.fp8e4m3
22 Ops.CONST dtypes.weakint [] 3
23 Ops.ADD dtypes.i32 [7, 22]
24 Ops.INDEX dtypes.u8 [9, 23]
25 Ops.LOAD dtypes.u8 [24]
26 Ops.BITCAST dtypes.fp8e4m3 [25] dtypes.fp8e4m3
27 Ops.STACK dtypes.fp8e4m3 [12, 17, 21, 26]
28 Ops.STORE dtypes.void [8, 27]
29 Ops.AFTER dtypes.void [0, 28]
30 Ops.END dtypes.void [29, 5]
31 Ops.SINK dtypes.void [30] KernelInfo(name="E_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
32 Ops.PARAM dtypes.u8 [1] ParamArg(slot=0, dtype=dtypes.u8)
33 Ops.SHRINK dtypes.u8 [32, 7, 4]
34 Ops.STACK dtypes.u8 [11, 16, 20, 25]
35 Ops.STORE dtypes.void [33, 34]
36 Ops.AFTER dtypes.void [0, 35]
37 Ops.END dtypes.void [36, 5]
38 Ops.SINK dtypes.void [37] KernelInfo(name="E_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
39 Ops.SINK dtypes.void [31, 38]

