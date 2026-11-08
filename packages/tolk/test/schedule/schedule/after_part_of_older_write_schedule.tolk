# tolk D24 golden=225c694f2e97f265e56f9ab41450fb0d
0 Ops.CONST dtypes.weakint [] 2
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 0
3 Ops.INDEX dtypes.f32 [1, 2]
4 Ops.CONST dtypes.weakfloat [] 1.0
5 Ops.CAST dtypes.f32 [4] dtypes.f32
6 Ops.STORE dtypes.void [3, 5]
7 Ops.SINK dtypes.void [6] KernelInfo()
8 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
9 Ops.CALL dtypes.void [7, 8] CallInfo()
10 Ops.CONST dtypes.weakint [] 1
11 Ops.INDEX dtypes.f32 [1, 10]
12 Ops.CONST dtypes.weakfloat [] 2.0
13 Ops.CAST dtypes.f32 [12] dtypes.f32
14 Ops.STORE dtypes.void [11, 13]
15 Ops.SINK dtypes.void [14] KernelInfo()
16 Ops.CALL dtypes.void [15, 8] CallInfo()
17 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
18 Ops.INDEX dtypes.f32 [1, 17]
19 Ops.CONST dtypes.weakfloat [] 3.0
20 Ops.CAST dtypes.f32 [19] dtypes.f32
21 Ops.STORE dtypes.void [18, 20]
22 Ops.END dtypes.void [21, 17]
23 Ops.SINK dtypes.void [22] KernelInfo()
24 Ops.CALL dtypes.void [23, 8] CallInfo()
25 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
26 Ops.INDEX dtypes.f32 [25, 17]
27 Ops.CONST dtypes.weakfloat [] 0.0
28 Ops.ADD dtypes.f32 [26, 27]
29 Ops.STORE dtypes.void [18, 28]
30 Ops.END dtypes.void [29, 17]
31 Ops.SINK dtypes.void [30] KernelInfo()
32 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
33 Ops.CALL dtypes.void [31, 32, 8] CallInfo()
34 Ops.LINEAR dtypes.void [9, 16, 24, 33]

