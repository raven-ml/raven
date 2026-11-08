# tolk D413 golden=469b920caa19eeb6a44ebdd70c99e174
0 Ops.STACK dtypes.void []
1 Ops.PARAM dtypes.i64 [0] ParamArg(slot=0, dtype=dtypes.i64)
2 Ops.CONST dtypes.weakint [] -9223372036854775808
3 Ops.CMPLT dtypes.bool [1, 2]
4 Ops.CONST dtypes.bool [] False
5 Ops.CMPEQ dtypes.bool [3, 4]
6 Ops.CONST dtypes.weakint [] 9223372036854775807
7 Ops.CMPLT dtypes.bool [6, 1]
8 Ops.CMPEQ dtypes.bool [7, 4]
9 Ops.CONST dtypes.weakint [] -1
10 Ops.MUL dtypes.i64 [1, 9]
11 Ops.CMPLT dtypes.bool [10, 2]
12 Ops.PARAM dtypes.i64 [0] ParamArg(slot=1, dtype=dtypes.i64)
13 Ops.MUL dtypes.i64 [12, 2]
14 Ops.CMPLT dtypes.bool [10, 13]
15 Ops.CONST dtypes.weakint [] -9223372036854775807
16 Ops.CMPLT dtypes.bool [1, 15]
17 Ops.AND dtypes.bool [7, 16]
18 Ops.CMPLT dtypes.bool [2, 1]
19 Ops.CONST dtypes.weakint [] -9223372036854775806
20 Ops.CMPLT dtypes.bool [1, 19]
21 Ops.AND dtypes.bool [18, 20]
22 Ops.CONST dtypes.weakint [] 9223372036854775805
23 Ops.CMPLT dtypes.bool [22, 1]
24 Ops.CMPLT dtypes.bool [1, 6]
25 Ops.AND dtypes.bool [23, 24]
26 Ops.CMPLT dtypes.bool [9, 1]
27 Ops.CONST dtypes.weakint [] 1
28 Ops.CMPLT dtypes.bool [1, 27]
29 Ops.AND dtypes.bool [26, 28]
30 Ops.SINK dtypes.void [5, 8, 11, 14, 17, 21, 25, 29]
31 Ops.ADD dtypes.weakint [2, 27]
32 Ops.CAST dtypes.i64 [31] dtypes.i64
33 Ops.CMPEQ dtypes.bool [1, 32]
34 Ops.ADD dtypes.weakint [22, 27]
35 Ops.CAST dtypes.i64 [34] dtypes.i64
36 Ops.CMPEQ dtypes.bool [1, 35]
37 Ops.ADD dtypes.weakint [9, 27]
38 Ops.CAST dtypes.i64 [37] dtypes.i64
39 Ops.CMPEQ dtypes.bool [1, 38]
40 Ops.SINK dtypes.void [5, 8, 11, 14, 17, 33, 36, 39]
41 Ops.SINK dtypes.void [30, 40]

