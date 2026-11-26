# tolk D27 golden=639abc272df08c08081ccb850dbb93de
0 Ops.STACK dtypes.void []
1 Ops.CONST dtypes.weakint [] 16
2 Ops.PARAM dtypes.bf16 [1] ParamArg(slot=0, dtype=dtypes.bf16)
3 Ops.CONST dtypes.weakint [] 4
4 Ops.CAST dtypes.i32 [3] dtypes.i32
5 Ops.RANGE dtypes.i32 [4] (AxisType.WEAK, 0)
6 Ops.CONST dtypes.weakint [] 2
7 Ops.SHL dtypes.i32 [5, 6]
8 Ops.INDEX dtypes.bf16 [2, 7]
9 Ops.PARAM dtypes.u16 [1] ParamArg(slot=1, dtype=dtypes.u16)
10 Ops.INDEX dtypes.u16 [9, 7]
11 Ops.LOAD dtypes.u16 [10]
12 Ops.BITCAST dtypes.bf16 [11] dtypes.bf16
13 Ops.STORE dtypes.void [8, 12]
14 Ops.CONST dtypes.weakint [] 1
15 Ops.ADD dtypes.i32 [7, 14]
16 Ops.INDEX dtypes.bf16 [2, 15]
17 Ops.INDEX dtypes.u16 [9, 15]
18 Ops.LOAD dtypes.u16 [17]
19 Ops.BITCAST dtypes.bf16 [18] dtypes.bf16
20 Ops.STORE dtypes.void [16, 19]
21 Ops.ADD dtypes.i32 [7, 6]
22 Ops.INDEX dtypes.bf16 [2, 21]
23 Ops.INDEX dtypes.u16 [9, 21]
24 Ops.LOAD dtypes.u16 [23]
25 Ops.BITCAST dtypes.bf16 [24] dtypes.bf16
26 Ops.STORE dtypes.void [22, 25]
27 Ops.CONST dtypes.weakint [] 3
28 Ops.ADD dtypes.i32 [7, 27]
29 Ops.INDEX dtypes.bf16 [2, 28]
30 Ops.INDEX dtypes.u16 [9, 28]
31 Ops.LOAD dtypes.u16 [30]
32 Ops.BITCAST dtypes.bf16 [31] dtypes.bf16
33 Ops.STORE dtypes.void [29, 32]
34 Ops.AFTER dtypes.void [0, 13, 20, 26, 33]
35 Ops.END dtypes.void [34, 5]
36 Ops.SINK dtypes.void [35] KernelInfo(name="E_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
37 Ops.PARAM dtypes.u16 [1] ParamArg(slot=0, dtype=dtypes.u16)
38 Ops.INDEX dtypes.u16 [37, 7]
39 Ops.STORE dtypes.void [38, 11]
40 Ops.INDEX dtypes.u16 [37, 15]
41 Ops.STORE dtypes.void [40, 18]
42 Ops.INDEX dtypes.u16 [37, 21]
43 Ops.STORE dtypes.void [42, 24]
44 Ops.INDEX dtypes.u16 [37, 28]
45 Ops.STORE dtypes.void [44, 31]
46 Ops.AFTER dtypes.void [0, 39, 41, 43, 45]
47 Ops.END dtypes.void [46, 5]
48 Ops.SINK dtypes.void [47] KernelInfo(name="E_4_4", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(4, AxisType.UPCAST)),), optimized=True)
49 Ops.SINK dtypes.void [36, 48]

