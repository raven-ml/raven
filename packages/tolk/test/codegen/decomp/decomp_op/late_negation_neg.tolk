# tolk D413 golden=9cf38c0c6e4e2feebf7cc484e372ab44
0 Ops.STACK dtypes.void []
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(-5, 5), name="x", addrspace=AddrSpace.ALU)
2 Ops.CONST dtypes.weakint [] -1
3 Ops.MUL dtypes.i32 [1, 2]
4 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
5 Ops.CONST dtypes.weakfloat [] -1.0
6 Ops.MUL dtypes.f32 [4, 5]
7 Ops.PARAM dtypes.i32 [0] ParamArg(slot=-1, dtype=dtypes.i32, vmin_vmax=(-5, 5), name="y", addrspace=AddrSpace.ALU)
8 Ops.NEG dtypes.i32 [7]
9 Ops.ADD dtypes.i32 [1, 8]
10 Ops.ADD dtypes.i32 [8, 1]
11 Ops.PARAM dtypes.f32 [0] ParamArg(slot=1, dtype=dtypes.f32)
12 Ops.NEG dtypes.f32 [11]
13 Ops.ADD dtypes.f32 [4, 12]
14 Ops.PARAM dtypes.u32 [0] ParamArg(slot=-1, dtype=dtypes.u32, vmin_vmax=(0, 10), name="u", addrspace=AddrSpace.ALU)
15 Ops.CONST dtypes.weakint [] 4294967295
16 Ops.CAST dtypes.u32 [15] dtypes.u32
17 Ops.MUL dtypes.u32 [14, 16]
18 Ops.MUL dtypes.u32 [14, 15]
19 Ops.SINK dtypes.void [3, 6, 9, 10, 13, 17, 18]
20 Ops.NEG dtypes.i32 [1]
21 Ops.NEG dtypes.f32 [4]
22 Ops.NEG dtypes.u32 [14]
23 Ops.SINK dtypes.void [20, 21, 9, 10, 13, 17, 22]
24 Ops.SINK dtypes.void [19, 23]

