# tolk D473 golden=d156fd5c5e9abba55918ca008b65a695
0 Ops.CONST dtypes.weakint [] 4
1 Ops.PARAM dtypes.u32 [0] ParamArg(slot=0, dtype=dtypes.u32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
3 Ops.INDEX dtypes.u32 [1, 2]
4 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
5 Ops.INDEX dtypes.f32 [4, 2]
6 Ops.CONST dtypes.weakfloat [] 1.0
7 Ops.ADD dtypes.f32 [5, 6]
8 Ops.BITCAST dtypes.u32 [7] dtypes.u32
9 Ops.STORE dtypes.void [3, 8]
10 Ops.END dtypes.void [9, 2]
11 Ops.SINK dtypes.void [10] KernelInfo()
12 Ops.BUFFER dtypes.u32 [0] ParamArg(slot=2, dtype=dtypes.u32, device="CPU")
13 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
14 Ops.CALL dtypes.void [11, 12, 13] CallInfo()
15 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
16 Ops.INDEX dtypes.f32 [15, 2]
17 Ops.PARAM dtypes.u32 [0] ParamArg(slot=1, dtype=dtypes.u32)
18 Ops.INDEX dtypes.u32 [17, 2]
19 Ops.BITCAST dtypes.f32 [18] dtypes.f32
20 Ops.STORE dtypes.void [16, 19]
21 Ops.END dtypes.void [20, 2]
22 Ops.SINK dtypes.void [21] KernelInfo()
23 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
24 Ops.CALL dtypes.void [22, 23, 12] CallInfo()
25 Ops.LINEAR dtypes.void [14, 24]

