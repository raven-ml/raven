# tolk D477 golden=9efdfb56796cc0f5040bde883129838b
0 Ops.CONST dtypes.weakint [] 526713
1 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=0, dtype=dtypes.i8, device="CPU")
2 Ops.CONST dtypes.weakint [] 518937
3 Ops.BUFFER dtypes.f32 [2] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
4 Ops.CONST dtypes.weakint [] 1
5 Ops.BUFFER dtypes.i8 [4] ParamArg(slot=2, dtype=dtypes.i8, device="CPU")
6 Ops.CONST dtypes.weakint [] 1000
7 Ops.BUFFER dtypes.i8 [6] ParamArg(slot=3, dtype=dtypes.i8, device="CPU")
8 Ops.SINK dtypes.void [1, 3, 5, 7]
9 Ops.CONST dtypes.weakint [] 6771968
10 Ops.BUFFER dtypes.i8 [9] ParamArg(slot=9, dtype=dtypes.i8, device="CPU")
11 Ops.CONST dtypes.weakint [] 0
12 Ops.SHRINK dtypes.i8 [10, 11, 0]
13 Ops.CONST dtypes.weakint [] 4194304
14 Ops.CONST dtypes.weakint [] 2075748
15 Ops.SHRINK dtypes.i8 [10, 13, 14]
16 Ops.BITCAST dtypes.f32 [15] dtypes.f32
17 Ops.CONST dtypes.weakint [] 6771712
18 Ops.SHRINK dtypes.i8 [10, 17, 4]
19 Ops.CONST dtypes.weakint [] 988160
20 Ops.SHRINK dtypes.i8 [10, 19, 6]
21 Ops.CALL dtypes.void [8, 12, 16, 18, 20] CallInfo()
22 Ops.CONST dtypes.weakint [] 988113
23 Ops.BUFFER dtypes.i8 [22] ParamArg(slot=4, dtype=dtypes.i8, device="CPU")
24 Ops.SINK dtypes.void [23]
25 Ops.SHRINK dtypes.i8 [10, 11, 22]
26 Ops.CALL dtypes.void [24, 25] CallInfo()
27 Ops.SINK dtypes.void [7]
28 Ops.CALL dtypes.void [27, 20] CallInfo()
29 Ops.SINK dtypes.void [3, 5]
30 Ops.CALL dtypes.void [29, 16, 18] CallInfo()
31 Ops.CONST dtypes.weakint [] 1048576
32 Ops.BUFFER dtypes.f32 [31] ParamArg(slot=5, dtype=dtypes.f32, device="CPU")
33 Ops.CONST dtypes.weakint [] 257
34 Ops.BUFFER dtypes.f32 [33] ParamArg(slot=6, dtype=dtypes.f32, device="CPU")
35 Ops.CONST dtypes.weakint [] 499971
36 Ops.BUFFER dtypes.i8 [35] ParamArg(slot=7, dtype=dtypes.i8, device="CPU")
37 Ops.SINK dtypes.void [32, 5, 34, 36]
38 Ops.SHRINK dtypes.i8 [10, 11, 13]
39 Ops.BITCAST dtypes.f32 [38] dtypes.f32
40 Ops.CONST dtypes.weakint [] 6770432
41 Ops.CONST dtypes.weakint [] 1028
42 Ops.SHRINK dtypes.i8 [10, 40, 41]
43 Ops.BITCAST dtypes.f32 [42] dtypes.f32
44 Ops.CONST dtypes.weakint [] 6270208
45 Ops.SHRINK dtypes.i8 [10, 44, 35]
46 Ops.CALL dtypes.void [37, 39, 18, 43, 45] CallInfo()
47 Ops.BUFFER dtypes.f32 [31] ParamArg(slot=8, dtype=dtypes.f32, device="CPU")
48 Ops.SINK dtypes.void [3, 47, 5, 34]
49 Ops.CALL dtypes.void [48, 16, 39, 18, 43] CallInfo()
50 Ops.LINEAR dtypes.void [21, 26, 28, 30, 46, 49]

