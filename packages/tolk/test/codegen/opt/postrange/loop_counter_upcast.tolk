# tolk D280 golden=cc535994af553f517e223ab935a515a7
0 Ops.CONST dtypes.weakint [] 4
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 2
3 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
4 Ops.MUL dtypes.weakint [3, 2]
5 Ops.RANGE dtypes.weakint [2] (AxisType.UPCAST, 2)
6 Ops.ADD dtypes.weakint [4, 5]
7 Ops.AFTER dtypes.f32 [1, 6]
8 Ops.INDEX dtypes.f32 [7, 6]
9 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
10 Ops.INDEX dtypes.f32 [9, 6]
11 Ops.CONST dtypes.weakint [] 1
12 Ops.BUFFER dtypes.i32 [11] ParamArg(slot=0, dtype=dtypes.i32, addrspace=AddrSpace.REG)
13 Ops.AFTER dtypes.i32 [12, 6]
14 Ops.CONST dtypes.weakint [] 0
15 Ops.INDEX dtypes.i32 [13, 14]
16 Ops.CAST dtypes.i32 [14] dtypes.i32
17 Ops.STORE dtypes.void [15, 16]
18 Ops.AFTER dtypes.i32 [13, 17]
19 Ops.NOOP dtypes.void []
20 Ops.RANGE dtypes.void [19] (AxisType.WEAK, 1)
21 Ops.AFTER dtypes.i32 [18, 20]
22 Ops.INDEX dtypes.i32 [21, 14]
23 Ops.ADD dtypes.i32 [22, 11]
24 Ops.STORE dtypes.void [22, 23]
25 Ops.CONST dtypes.weakint [] 3
26 Ops.CMPLT dtypes.bool [23, 25]
27 Ops.BACKEDGE dtypes.void [24, 20, 26]
28 Ops.AFTER dtypes.i32 [18, 27]
29 Ops.INDEX dtypes.i32 [28, 14]
30 Ops.CAST dtypes.f32 [29] dtypes.f32
31 Ops.ADD dtypes.f32 [10, 30]
32 Ops.STORE dtypes.void [8, 31]
33 Ops.END dtypes.void [32, 3, 5]
34 Ops.SINK dtypes.void [1, 33] KernelInfo(name="E_2_2", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(2, AxisType.UPCAST)),), optimized=True)

