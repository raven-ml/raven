# tolk D41 D477 golden=b6c47d3661f9d57dbece9345b0ff0b3d
0 Ops.CONST dtypes.weakint [] 1048576
1 Ops.BUFFER dtypes.i8 [0] ParamArg(slot=0, dtype=dtypes.i8, device="CPU")
2 Ops.CONST dtypes.weakint [] 1
3 Ops.BUFFER dtypes.i8 [2] ParamArg(slot=1, dtype=dtypes.i8, device="CPU")
4 Ops.SINK dtypes.void [1, 3]
5 Ops.CONST dtypes.weakint [] 6735104
6 Ops.BUFFER dtypes.i8 [5] ParamArg(slot=10, dtype=dtypes.i8, device="CPU")
7 Ops.CONST dtypes.weakint [] 5677312
8 Ops.SHRINK dtypes.i8 [6, 7, 0]
9 Ops.CONST dtypes.weakint [] 6734080
10 Ops.SHRINK dtypes.i8 [6, 9, 2]
11 Ops.CALL dtypes.void [4, 8, 10] CallInfo()
12 Ops.CONST dtypes.weakint [] 16
13 Ops.BUFFER dtypes.f32 [12] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
14 Ops.SINK dtypes.void [13]
15 Ops.CONST dtypes.weakint [] 6734336
16 Ops.CONST dtypes.weakint [] 64
17 Ops.SHRINK dtypes.i8 [6, 15, 16]
18 Ops.BITCAST dtypes.f32 [17] dtypes.f32
19 Ops.CALL dtypes.void [14, 18] CallInfo()
20 Ops.SINK dtypes.void [3]
21 Ops.CALL dtypes.void [20, 10] CallInfo()
22 Ops.BUFFER dtypes.i8 [2] ParamArg(slot=3, dtype=dtypes.i8, device="CPU")
23 Ops.CONST dtypes.weakint [] 370694
24 Ops.BUFFER dtypes.f32 [23] ParamArg(slot=4, dtype=dtypes.f32, device="CPU")
25 Ops.SINK dtypes.void [22, 24, 13]
26 Ops.CONST dtypes.weakint [] 6734592
27 Ops.SHRINK dtypes.i8 [6, 26, 2]
28 Ops.CONST dtypes.weakint [] 4194304
29 Ops.CONST dtypes.weakint [] 1482776
30 Ops.SHRINK dtypes.i8 [6, 28, 29]
31 Ops.BITCAST dtypes.f32 [30] dtypes.f32
32 Ops.CALL dtypes.void [25, 27, 31, 18] CallInfo()
33 Ops.SINK dtypes.void [22, 13, 3, 24]
34 Ops.CALL dtypes.void [33, 27, 18, 10, 31] CallInfo()
35 Ops.PARAM dtypes.f32 [23] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
36 Ops.PARAM dtypes.i8 [12] ParamArg(slot=1, dtype=dtypes.i8, device="CPU")
37 Ops.STORE dtypes.void [35, 36]
38 Ops.BUFFER dtypes.i8 [12] ParamArg(slot=5, dtype=dtypes.i8, device="CPU")
39 Ops.CALL dtypes.void [37, 31, 38] CallInfo()
40 Ops.BUFFER dtypes.f32 [2] ParamArg(slot=6, dtype=dtypes.f32, device="CPU")
41 Ops.CONST dtypes.weakint [] 4096
42 Ops.BUFFER dtypes.i8 [41] ParamArg(slot=7, dtype=dtypes.i8, device="CPU")
43 Ops.BUFFER dtypes.f32 [0] ParamArg(slot=8, dtype=dtypes.f32, device="CPU")
44 Ops.SINK dtypes.void [40, 42, 38, 43]
45 Ops.CONST dtypes.weakint [] 6734848
46 Ops.CONST dtypes.weakint [] 4
47 Ops.SHRINK dtypes.i8 [6, 45, 46]
48 Ops.BITCAST dtypes.f32 [47] dtypes.f32
49 Ops.CONST dtypes.weakint [] 6725888
50 Ops.SHRINK dtypes.i8 [6, 49, 41]
51 Ops.CONST dtypes.weakint [] 0
52 Ops.SHRINK dtypes.i8 [6, 51, 28]
53 Ops.BITCAST dtypes.f32 [52] dtypes.f32
54 Ops.CALL dtypes.void [44, 48, 50, 38, 53] CallInfo()
55 Ops.BUFFER dtypes.i8 [41] ParamArg(slot=9, dtype=dtypes.i8, device="CPU")
56 Ops.SINK dtypes.void [55]
57 Ops.CONST dtypes.weakint [] 6729984
58 Ops.SHRINK dtypes.i8 [6, 57, 41]
59 Ops.CALL dtypes.void [56, 58] CallInfo()
60 Ops.SINK dtypes.void [13, 43, 22, 24]
61 Ops.CALL dtypes.void [60, 18, 53, 27, 31] CallInfo()
62 Ops.SINK dtypes.void [38, 1, 42, 40]
63 Ops.CALL dtypes.void [62, 38, 8, 50, 48] CallInfo()
64 Ops.PARAM dtypes.i8 [2] ParamArg(slot=0, dtype=dtypes.i8, device="CPU")
65 Ops.PARAM dtypes.f32 [23] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
66 Ops.STORE dtypes.void [64, 65]
67 Ops.CALL dtypes.void [66, 10, 31] CallInfo()
68 Ops.SINK dtypes.void [43, 55, 13]
69 Ops.CALL dtypes.void [68, 53, 58, 18] CallInfo()
70 Ops.SINK dtypes.void [13, 1, 43, 55]
71 Ops.CALL dtypes.void [70, 18, 8, 53, 58] CallInfo()
72 Ops.SINK dtypes.void [1, 40]
73 Ops.CALL dtypes.void [72, 8, 48] CallInfo()
74 Ops.SINK dtypes.void [13, 38, 1]
75 Ops.CALL dtypes.void [74, 18, 38, 8] CallInfo()
76 Ops.SINK dtypes.void [24]
77 Ops.CALL dtypes.void [76, 31] CallInfo()
78 Ops.SINK dtypes.void [55, 42, 22]
79 Ops.CALL dtypes.void [78, 58, 50, 27] CallInfo()
80 Ops.SINK dtypes.void [3, 42, 1, 43]
81 Ops.CALL dtypes.void [80, 10, 50, 8, 53] CallInfo()
82 Ops.SINK dtypes.void [43]
83 Ops.CALL dtypes.void [82, 53] CallInfo()
84 Ops.LINEAR dtypes.void [11, 19, 21, 32, 34, 39, 54, 59, 59, 61, 63, 67, 69, 71, 73, 75, 77, 79, 81, 83]

