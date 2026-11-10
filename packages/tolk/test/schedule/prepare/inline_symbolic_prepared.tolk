# tolk D473 golden=4650875dd26e02fb7438f5dcde2ea53b
0 Ops.CONST dtypes.weakint [] 10
1 Ops.ALLOC dtypes.f32 [0] ParamArg(slot=-1, dtype=dtypes.f32, device="CPU")
2 Ops.CONST dtypes.weakint [] 0
3 Ops.STACK dtypes.void []
4 Ops.PARAM dtypes.weakint [3] ParamArg(slot=1, dtype=dtypes.weakint, vmin_vmax=(1, 10), addrspace=AddrSpace.ALU)
5 Ops.SHRINK dtypes.f32 [1, 2, 4]
6 Ops.PARAM dtypes.f32 [0] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
7 Ops.SHRINK dtypes.f32 [6, 2, 4]
8 Ops.CONST dtypes.weakfloat [] 2.0
9 Ops.MUL dtypes.f32 [7, 8]
10 Ops.STORE dtypes.void [5, 9]
11 Ops.AFTER dtypes.f32 [1, 10]
12 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
13 Ops.SHRINK dtypes.f32 [12, 2, 4]
14 Ops.SHRINK dtypes.f32 [11, 2, 4]
15 Ops.CONST dtypes.weakfloat [] 1.0
16 Ops.ADD dtypes.f32 [14, 15]
17 Ops.STORE dtypes.void [13, 16]
18 Ops.AFTER dtypes.f32 [12, 17]
19 Ops.SINK dtypes.void [11, 18]

