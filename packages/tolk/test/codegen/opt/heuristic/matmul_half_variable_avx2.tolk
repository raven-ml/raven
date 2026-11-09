# tolk D141 golden=3bc962351268730e026f0cf6fa8a4061
0 Ops.CONST dtypes.weakint [] 1048576
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.STACK dtypes.void []
3 Ops.PARAM dtypes.weakint [2] ParamArg(slot=-1, dtype=dtypes.weakint, vmin_vmax=(1, 64), name="n", addrspace=AddrSpace.ALU)
4 Ops.CONST dtypes.weakint [] 128
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [5] (AxisType.WEAK, 1)
7 Ops.MUL dtypes.weakint [6, 4]
8 Ops.CONST dtypes.weakint [] 4
9 Ops.RANGE dtypes.weakint [8] (AxisType.WEAK, 3)
10 Ops.CONST dtypes.weakint [] 2
11 Ops.MUL dtypes.weakint [9, 10]
12 Ops.RANGE dtypes.weakint [10] (AxisType.UPCAST, 5)
13 Ops.ADD dtypes.weakint [11, 12]
14 Ops.CONST dtypes.weakint [] 16
15 Ops.MUL dtypes.weakint [13, 14]
16 Ops.RANGE dtypes.weakint [14] (AxisType.UPCAST, 4)
17 Ops.ADD dtypes.weakint [15, 16]
18 Ops.ADD dtypes.weakint [7, 17]
19 Ops.INDEX dtypes.f32 [1, 18]
20 Ops.PARAM dtypes.f16 [0] ParamArg(slot=1, dtype=dtypes.f16)
21 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0)
22 Ops.ADD dtypes.weakint [7, 21]
23 Ops.INDEX dtypes.f16 [20, 22]
24 Ops.CONST dtypes.weakint [] 16384
25 Ops.PARAM dtypes.f16 [24] ParamArg(slot=2, dtype=dtypes.f16)
26 Ops.MUL dtypes.weakint [21, 4]
27 Ops.ADD dtypes.weakint [26, 17]
28 Ops.INDEX dtypes.f16 [25, 27]
29 Ops.MUL dtypes.f16 [23, 28]
30 Ops.CAST dtypes.f32 [29] dtypes.f32
31 Ops.REDUCE dtypes.f32 [30, 21] (Ops.ADD, 0)
32 Ops.STORE dtypes.void [19, 31]
33 Ops.END dtypes.void [32, 6, 9, 12, 16]
34 Ops.SINK dtypes.void [33] KernelInfo(name="r_(n*128)_4_16_2_128", applied_opts=(Opt(op=OptOps.SPLIT, axis=1, arg=(16, AxisType.UPCAST)), Opt(op=OptOps.SPLIT, axis=1, arg=(2, AxisType.UPCAST))), optimized=True)

