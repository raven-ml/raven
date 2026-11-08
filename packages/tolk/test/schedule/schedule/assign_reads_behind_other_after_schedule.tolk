# tolk D160 golden=b5e3e8319023f2914a3cb907c09b8539
0 Ops.CONST dtypes.weakint [] 2
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
5 Ops.INDEX dtypes.f32 [4, 2]
6 Ops.CONST dtypes.weakfloat [] 1.0
7 Ops.ADD dtypes.f32 [5, 6]
8 Ops.STORE dtypes.void [3, 7]
9 Ops.END dtypes.void [8, 2]
10 Ops.SINK dtypes.void [9] KernelInfo()
11 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
12 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
13 Ops.CALL dtypes.void [10, 11, 12] CallInfo()
14 Ops.CONST dtypes.weakint [] 1
15 Ops.PARAM dtypes.f32 [14] ParamArg(slot=0, dtype=dtypes.f32)
16 Ops.CONST dtypes.weakint [] 0
17 Ops.INDEX dtypes.f32 [15, 16]
18 Ops.REDUCE dtypes.f32 [5, 2] (Ops.ADD, 0)
19 Ops.STORE dtypes.void [17, 18]
20 Ops.SINK dtypes.void [19] KernelInfo()
21 Ops.ALLOC dtypes.f32 [14] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
22 Ops.CALL dtypes.void [20, 21, 11] CallInfo()
23 Ops.PARAM dtypes.f32 [14] ParamArg(slot=1, dtype=dtypes.f32)
24 Ops.INDEX dtypes.f32 [23, 16]
25 Ops.STORE dtypes.void [3, 24]
26 Ops.END dtypes.void [25, 2]
27 Ops.SINK dtypes.void [26] KernelInfo()
28 Ops.CALL dtypes.void [27, 12, 21] CallInfo()
29 Ops.LINEAR dtypes.void [13, 22, 28]

