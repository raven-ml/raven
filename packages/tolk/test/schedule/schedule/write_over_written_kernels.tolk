# tolk D330
0 Ops.CONST dtypes.weakint [] 1
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
3 Ops.CONST dtypes.weakint [] 0
4 Ops.INDEX dtypes.f32 [2, 3]
5 Ops.CONST dtypes.weakfloat [] 1.0
6 Ops.CAST dtypes.f32 [5] dtypes.f32
7 Ops.STORE dtypes.void [4, 6]
8 Ops.SINK dtypes.void [7] KernelInfo()
9 Ops.CALL dtypes.void [8, 1] CallInfo()
10 Ops.AFTER dtypes.f32 [1, 9]
11 Ops.CONST dtypes.weakfloat [] 2.0
12 Ops.CAST dtypes.f32 [11] dtypes.f32
13 Ops.STORE dtypes.void [4, 12]
14 Ops.SINK dtypes.void [13] KernelInfo()
15 Ops.CALL dtypes.void [14, 10] CallInfo()
16 Ops.AFTER dtypes.f32 [1, 15]
17 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
18 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
19 Ops.INDEX dtypes.f32 [18, 3]
20 Ops.CONST dtypes.weakfloat [] 0.0
21 Ops.ADD dtypes.f32 [19, 20]
22 Ops.STORE dtypes.void [4, 21]
23 Ops.SINK dtypes.void [22] KernelInfo()
24 Ops.CALL dtypes.void [23, 17, 16] CallInfo()
25 Ops.AFTER dtypes.f32 [17, 24]
26 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
27 Ops.CALL dtypes.void [23, 26, 10] CallInfo()
28 Ops.AFTER dtypes.f32 [26, 27]
29 Ops.SINK dtypes.void [10, 16, 25, 28]

