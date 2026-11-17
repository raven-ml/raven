# tolk D350 golden=ccde520c47ad03a7ad0dee2506e34ab9
0 Ops.CONST dtypes.weakint [] 4
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
4 Ops.INDEX dtypes.f32 [2, 3]
5 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
6 Ops.INDEX dtypes.f32 [5, 3]
7 Ops.CONST dtypes.weakfloat [] 1.0
8 Ops.ADD dtypes.f32 [6, 7]
9 Ops.STORE dtypes.void [4, 8]
10 Ops.END dtypes.void [9, 3]
11 Ops.SINK dtypes.void [10] KernelInfo(name="add_one_4")
12 Ops.CONST dtypes.weakint [] 16
13 Ops.PARAM dtypes.f32 [12] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
14 Ops.SHRINK dtypes.f32 [13, 0, 0]
15 Ops.CALL dtypes.void [11, 1, 14] CallInfo()
16 Ops.AFTER dtypes.f32 [1, 15]
17 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
18 Ops.PARAM dtypes.f32 [12] ParamArg(slot=2, dtype=dtypes.f32)
19 Ops.INDEX dtypes.f32 [18, 3]
20 Ops.ADD dtypes.f32 [6, 19]
21 Ops.STORE dtypes.void [4, 20]
22 Ops.END dtypes.void [21, 3]
23 Ops.SINK dtypes.void [22] KernelInfo()
24 Ops.CALL dtypes.void [23, 17, 16, 13] CallInfo()
25 Ops.AFTER dtypes.f32 [17, 24]
26 Ops.SINK dtypes.void [16, 25]

