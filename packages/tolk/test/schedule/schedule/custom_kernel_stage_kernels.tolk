# tolk D122 golden=1c714ed2834a4797d62542e6c8e5a3ff
0 Ops.CONST dtypes.weakint [] 1
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32, device="CPU")
2 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
3 Ops.CONST dtypes.weakint [] 0
4 Ops.INDEX dtypes.i32 [2, 3]
5 Ops.CONST dtypes.weakint [] 4
6 Ops.PARAM dtypes.i32 [5] ParamArg(slot=1, dtype=dtypes.i32, vmin_vmax=(0, 3))
7 Ops.CONST dtypes.weakint [] 2
8 Ops.INDEX dtypes.i32 [6, 7]
9 Ops.LOAD dtypes.i32 [8]
10 Ops.STORE dtypes.void [4, 9]
11 Ops.SINK dtypes.void [10] KernelInfo()
12 Ops.ALLOC dtypes.i32 [5] ParamArg(slot=3, dtype=dtypes.i32, vmin_vmax=(0, 3))
13 Ops.PARAM dtypes.i32 [5] ParamArg(slot=0, dtype=dtypes.i32, vmin_vmax=(0, 3))
14 Ops.CAST dtypes.i32 [5] dtypes.i32
15 Ops.RANGE dtypes.i32 [14] (AxisType.LOOP, 0)
16 Ops.INDEX dtypes.i32 [13, 15]
17 Ops.STORE dtypes.void [16, 15]
18 Ops.END dtypes.void [17, 15]
19 Ops.SINK dtypes.void [18] KernelInfo()
20 Ops.CALL dtypes.void [19, 12] CallInfo()
21 Ops.AFTER dtypes.i32 [12, 20]
22 Ops.CALL dtypes.void [11, 1, 21] CallInfo()
23 Ops.AFTER dtypes.i32 [1, 22]
24 Ops.PARAM dtypes.i32 [0] ParamArg(slot=1, dtype=dtypes.i32, device="CPU")
25 Ops.PARAM dtypes.i32 [0] ParamArg(slot=1, dtype=dtypes.i32)
26 Ops.INDEX dtypes.i32 [25, 3]
27 Ops.ADD dtypes.i32 [26, 0]
28 Ops.STORE dtypes.void [4, 27]
29 Ops.SINK dtypes.void [28] KernelInfo()
30 Ops.CALL dtypes.void [29, 24, 23] CallInfo()
31 Ops.AFTER dtypes.i32 [24, 30]
32 Ops.SINK dtypes.void [23, 31]

