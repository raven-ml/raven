# tolk D41 golden=84cafbb8672eeb3dd039aa7b973f0c90
0 Ops.CONST dtypes.weakint [] 16
1 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=1, dtype=dtypes.i8, device="CPU")
3 Ops.SINK dtypes.void [1, 2]
4 Ops.CONST dtypes.weakint [] 512
5 Ops.BUFFER dtypes.i8 [4] ParamArg(slot=4, dtype=dtypes.i8, device="CPU")
6 Ops.CONST dtypes.weakint [] 0
7 Ops.CONST dtypes.weakint [] 64
8 Ops.SHRINK dtypes.i8 [5, 6, 7]
9 Ops.BITCAST dtypes.f32 [8] dtypes.f32
10 Ops.CONST dtypes.weakint [] 256
11 Ops.SHRINK dtypes.i8 [5, 10, 0]
12 Ops.CALL dtypes.void [3, 9, 11] CallInfo()
13 Ops.PARAM dtypes.i8 [0] ParamArg(slot=0, dtype=dtypes.i8, device="CPU")
14 Ops.PARAM dtypes.i8 [0] ParamArg(slot=1, dtype=dtypes.i8, device="CPU")
15 Ops.STORE dtypes.void [13, 14]
16 Ops.SHRINK dtypes.i8 [5, 6, 0]
17 Ops.CALL dtypes.void [15, 16, 11] CallInfo()
18 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=3, dtype=dtypes.f32, device="CPU")
19 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=2, dtype=dtypes.i8, device="CPU")
20 Ops.SINK dtypes.void [18, 19]
21 Ops.SHRINK dtypes.i8 [5, 10, 7]
22 Ops.BITCAST dtypes.f32 [21] dtypes.f32
23 Ops.CALL dtypes.void [20, 22, 16] CallInfo()
24 Ops.LINEAR dtypes.void [12, 17, 23]

