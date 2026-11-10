# tolk D477 golden=df8f011492d94b573caf14fdb3dcfcba
0 Ops.CONST dtypes.weakint [] 16
1 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.CONST dtypes.weakint [] 1099511627776
3 Ops.BUFFER dtypes.i8 [2] ParamArg(slot=1, dtype=dtypes.i8, device="CPU")
4 Ops.SINK dtypes.void [1, 3]
5 Ops.CONST dtypes.weakint [] 1125904201809920
6 Ops.BUFFER dtypes.i8 [5] ParamArg(slot=4, dtype=dtypes.i8, device="CPU")
7 Ops.CONST dtypes.weakint [] 0
8 Ops.SHRINK dtypes.i8 [6, 7, 2]
9 Ops.CALL dtypes.void [4, 1, 8] CallInfo()
10 Ops.CONST dtypes.weakint [] 1073741824
11 Ops.BUFFER dtypes.f32 [10] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
12 Ops.CONST dtypes.weakint [] 1125899906842624
13 Ops.BUFFER dtypes.i8 [12] ParamArg(slot=3, dtype=dtypes.i8, device="CPU")
14 Ops.SINK dtypes.void [11, 13]
15 Ops.CONST dtypes.weakint [] 4294967296
16 Ops.SHRINK dtypes.i8 [6, 12, 15]
17 Ops.BITCAST dtypes.f32 [16] dtypes.f32
18 Ops.SHRINK dtypes.i8 [6, 7, 12]
19 Ops.CALL dtypes.void [14, 17, 18] CallInfo()
20 Ops.LINEAR dtypes.void [9, 19]

