# tolk D202 D473
0 Ops.CONST dtypes.weakint [] 4
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
5 Ops.INDEX dtypes.f32 [4, 2]
6 Ops.CONST dtypes.weakfloat [] 2.0
7 Ops.MUL dtypes.f32 [5, 6]
8 Ops.STORE dtypes.void [3, 7]
9 Ops.END dtypes.void [8, 2]
10 Ops.SINK dtypes.void [9] KernelInfo()
11 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
12 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
13 Ops.CALL dtypes.void [10, 11, 12] CallInfo()
14 Ops.CONST dtypes.weakfloat [] 1.0
15 Ops.ADD dtypes.f32 [3, 14]
16 Ops.STORE dtypes.void [3, 15]
17 Ops.END dtypes.void [16, 2]
18 Ops.SINK dtypes.void [17] KernelInfo()
19 Ops.CALL dtypes.void [18, 12] CallInfo()
20 Ops.ADD dtypes.f32 [3, 5]
21 Ops.STORE dtypes.void [3, 20]
22 Ops.END dtypes.void [21, 2]
23 Ops.SINK dtypes.void [22] KernelInfo()
24 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
25 Ops.CALL dtypes.void [23, 24, 12] CallInfo()
26 Ops.LINEAR dtypes.void [13, 19, 25]

