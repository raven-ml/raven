# tolk D24 golden=0f21e06a1e95076f651055d9844930a4
0 Ops.CONST dtypes.weakint [] 1
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.CONST dtypes.weakfloat [] 1.0
5 Ops.CAST dtypes.f32 [4] dtypes.f32
6 Ops.STORE dtypes.void [3, 5]
7 Ops.SINK dtypes.void [6] KernelInfo()
8 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
9 Ops.CALL dtypes.void [7, 8] CallInfo()
10 Ops.CONST dtypes.weakfloat [] 2.0
11 Ops.CAST dtypes.f32 [10] dtypes.f32
12 Ops.STORE dtypes.void [3, 11]
13 Ops.SINK dtypes.void [12] KernelInfo()
14 Ops.CALL dtypes.void [13, 8] CallInfo()
15 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
16 Ops.INDEX dtypes.f32 [15, 2]
17 Ops.CONST dtypes.weakfloat [] 0.0
18 Ops.ADD dtypes.f32 [16, 17]
19 Ops.STORE dtypes.void [3, 18]
20 Ops.SINK dtypes.void [19] KernelInfo()
21 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
22 Ops.CALL dtypes.void [20, 21, 8] CallInfo()
23 Ops.CONST dtypes.weakint [] 2
24 Ops.PARAM dtypes.f32 [23] ParamArg(slot=0, dtype=dtypes.f32)
25 Ops.RANGE dtypes.weakint [23] (AxisType.WEAK, 0)
26 Ops.INDEX dtypes.f32 [24, 25]
27 Ops.STORE dtypes.void [26, 18]
28 Ops.END dtypes.void [27, 25]
29 Ops.SINK dtypes.void [28] KernelInfo()
30 Ops.BUFFER dtypes.f32 [23] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
31 Ops.CALL dtypes.void [29, 30, 8] CallInfo()
32 Ops.LINEAR dtypes.void [9, 14, 22, 31]

