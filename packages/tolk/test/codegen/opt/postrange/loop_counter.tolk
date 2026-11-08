# tolk D280 golden=046256f4a5f6daac4da381ddb54a46ec
0 Ops.CONST dtypes.weakint [] 4
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
3 Ops.AFTER dtypes.f32 [1, 2]
4 Ops.INDEX dtypes.f32 [3, 2]
5 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
6 Ops.INDEX dtypes.f32 [5, 2]
7 Ops.CONST dtypes.weakint [] 1
8 Ops.BUFFER dtypes.i32 [7] ParamArg(slot=0, dtype=dtypes.i32, addrspace=AddrSpace.REG)
9 Ops.AFTER dtypes.i32 [8, 2]
10 Ops.CONST dtypes.weakint [] 0
11 Ops.INDEX dtypes.i32 [9, 10]
12 Ops.CAST dtypes.i32 [10] dtypes.i32
13 Ops.STORE dtypes.void [11, 12]
14 Ops.AFTER dtypes.i32 [9, 13]
15 Ops.NOOP dtypes.void []
16 Ops.RANGE dtypes.void [15] (AxisType.WEAK, 1)
17 Ops.AFTER dtypes.i32 [14, 16]
18 Ops.INDEX dtypes.i32 [17, 10]
19 Ops.ADD dtypes.i32 [18, 7]
20 Ops.STORE dtypes.void [18, 19]
21 Ops.CONST dtypes.weakint [] 3
22 Ops.CMPLT dtypes.bool [19, 21]
23 Ops.BACKEDGE dtypes.void [20, 16, 22]
24 Ops.AFTER dtypes.i32 [14, 23]
25 Ops.INDEX dtypes.i32 [24, 10]
26 Ops.CAST dtypes.f32 [25] dtypes.f32
27 Ops.ADD dtypes.f32 [6, 26]
28 Ops.STORE dtypes.void [4, 27]
29 Ops.END dtypes.void [28, 2]
30 Ops.SINK dtypes.void [1, 29] KernelInfo(opts_to_apply=())

