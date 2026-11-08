# tolk D170 golden=afceb66932de14a00969607162acbcef
0 Ops.CONST dtypes.weakint [] 8
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
2 Ops.CONST dtypes.weakint [] 2
3 Ops.RANGE dtypes.weakint [2] (AxisType.LOOP, 1)
4 Ops.CONST dtypes.weakint [] 4
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 0)
7 Ops.ADD dtypes.weakint [5, 6]
8 Ops.INDEX dtypes.i32 [1, 7]
9 Ops.CONST dtypes.weakint [] 1
10 Ops.CMPLT dtypes.bool [6, 9]
11 Ops.CONST dtypes.bool [] False
12 Ops.CMPEQ dtypes.bool [10, 11]
13 Ops.CONST dtypes.weakint [] 3
14 Ops.CMPLT dtypes.bool [6, 13]
15 Ops.AND dtypes.bool [12, 14]
16 Ops.CONST dtypes.weakint [] 0
17 Ops.WHERE dtypes.weakint [15, 3, 16]
18 Ops.STORE dtypes.void [8, 17]
19 Ops.END dtypes.void [18, 6]
20 Ops.END dtypes.void [19, 3]
21 Ops.SINK dtypes.void [20] KernelInfo()
22 Ops.CONST dtypes.weakint [] 256
23 Ops.BUFFER dtypes.i8 [22] ParamArg(slot=5, dtype=dtypes.i8, device="CPU")
24 Ops.CONST dtypes.weakint [] 32
25 Ops.SHRINK dtypes.i8 [23, 16, 24]
26 Ops.BITCAST dtypes.i32 [25] dtypes.i32
27 Ops.CALL dtypes.void [21, 26] CallInfo()
28 Ops.RANGE dtypes.weakint [0] (AxisType.WEAK, 0)
29 Ops.INDEX dtypes.i32 [1, 28]
30 Ops.PARAM dtypes.i32 [0] ParamArg(slot=1, dtype=dtypes.i32)
31 Ops.INDEX dtypes.i32 [30, 28]
32 Ops.STORE dtypes.void [29, 31]
33 Ops.END dtypes.void [32, 28]
34 Ops.SINK dtypes.void [33] KernelInfo()
35 Ops.BUFFER dtypes.i32 [0] ParamArg(slot=1, dtype=dtypes.i32, device="CPU")
36 Ops.CALL dtypes.void [34, 35, 26] CallInfo()
37 Ops.RANGE dtypes.weakint [2] (AxisType.WEAK, 0)
38 Ops.MUL dtypes.weakint [37, 4]
39 Ops.RANGE dtypes.weakint [4] (AxisType.WEAK, 1)
40 Ops.ADD dtypes.weakint [38, 39]
41 Ops.INDEX dtypes.i32 [1, 40]
42 Ops.INDEX dtypes.i32 [30, 40]
43 Ops.ADD dtypes.i32 [42, 9]
44 Ops.STORE dtypes.void [41, 43]
45 Ops.END dtypes.void [44, 37, 39]
46 Ops.SINK dtypes.void [45] KernelInfo()
47 Ops.BUFFER dtypes.i32 [0] ParamArg(slot=2, dtype=dtypes.i32, device="CPU")
48 Ops.CALL dtypes.void [46, 47, 35] CallInfo()
49 Ops.LINEAR dtypes.void [27, 36, 48]

