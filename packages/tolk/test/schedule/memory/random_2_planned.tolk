# tolk D477 golden=1d9f25eeab4f952bb76495329b76f704
0 Ops.CONST dtypes.weakint [] 1048576
1 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=0, dtype=dtypes.i8, device="CPU")
2 Ops.SINK dtypes.void [1]
3 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=7, dtype=dtypes.i8, device="CPU")
4 Ops.CALL dtypes.void [2, 3] CallInfo()
5 Ops.CONST dtypes.weakint [] 1000
6 Ops.BUFFER dtypes.i8 [5] ParamArg(slot=1, dtype=dtypes.i8, device="CPU")
7 Ops.CONST dtypes.weakint [] 1
8 Ops.BUFFER dtypes.i8 [7] ParamArg(slot=2, dtype=dtypes.i8, device="CPU")
9 Ops.CONST dtypes.weakint [] 65536
10 Ops.BUFFER dtypes.i8 [9] ParamArg(slot=3, dtype=dtypes.i8, device="CPU")
11 Ops.SINK dtypes.void [6, 8, 10]
12 Ops.SHRINK dtypes.i8 [3, 9, 5]
13 Ops.CONST dtypes.weakint [] 66560
14 Ops.SHRINK dtypes.i8 [3, 13, 7]
15 Ops.CONST dtypes.weakint [] 0
16 Ops.SHRINK dtypes.i8 [3, 15, 9]
17 Ops.CALL dtypes.void [11, 12, 14, 16] CallInfo()
18 Ops.CONST dtypes.weakint [] 4096
19 Ops.BUFFER dtypes.i8 [18] ParamArg(slot=4, dtype=dtypes.i8, device="CPU")
20 Ops.CONST dtypes.weakint [] 255
21 Ops.BUFFER dtypes.i8 [20] ParamArg(slot=5, dtype=dtypes.i8, device="CPU")
22 Ops.BUFFER dtypes.f32 [5] ParamArg(slot=6, dtype=dtypes.f32, device="CPU")
23 Ops.SINK dtypes.void [19, 21, 22]
24 Ops.SHRINK dtypes.i8 [3, 15, 18]
25 Ops.CONST dtypes.weakint [] 8192
26 Ops.SHRINK dtypes.i8 [3, 25, 20]
27 Ops.CONST dtypes.weakint [] 4000
28 Ops.SHRINK dtypes.i8 [3, 18, 27]
29 Ops.BITCAST dtypes.f32 [28] dtypes.f32
30 Ops.CALL dtypes.void [23, 24, 26, 29] CallInfo()
31 Ops.LINEAR dtypes.void [4, 17, 30]

