# tolk D175 golden=77854faa8660a4bf0b9eecb86ae92722
0 Ops.CONST dtypes.weakint [] 4096
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32)
2 Ops.CONST dtypes.weakint [] 8
3 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 1)
4 Ops.CONST dtypes.weakint [] 2
5 Ops.MUL dtypes.weakint [3, 4]
6 Ops.CONST dtypes.weakint [] 32
7 Ops.RANGE dtypes.weakint [6] (AxisType.WARP, -1)
8 Ops.CONST dtypes.weakint [] 16
9 Ops.DIV dtypes.weakint [7, 8]
10 Ops.MOD dtypes.weakint [9, 4]
11 Ops.ADD dtypes.weakint [5, 10]
12 Ops.MUL dtypes.weakint [11, 4]
13 Ops.CONST dtypes.weakint [] 4
14 Ops.DIV dtypes.weakint [7, 13]
15 Ops.MOD dtypes.weakint [14, 4]
16 Ops.ADD dtypes.weakint [12, 15]
17 Ops.MUL dtypes.weakint [16, 4]
18 Ops.DIV dtypes.weakint [7, 4]
19 Ops.MOD dtypes.weakint [18, 4]
20 Ops.ADD dtypes.weakint [17, 19]
21 Ops.CONST dtypes.weakint [] 64
22 Ops.MUL dtypes.weakint [20, 21]
23 Ops.RANGE dtypes.weakint [2] (AxisType.GLOBAL, 2)
24 Ops.MUL dtypes.weakint [23, 4]
25 Ops.DIV dtypes.weakint [7, 2]
26 Ops.MOD dtypes.weakint [25, 4]
27 Ops.ADD dtypes.weakint [24, 26]
28 Ops.MUL dtypes.weakint [27, 4]
29 Ops.CONST dtypes.weakint [] 1
30 Ops.DIV dtypes.weakint [7, 29]
31 Ops.MOD dtypes.weakint [30, 4]
32 Ops.ADD dtypes.weakint [28, 31]
33 Ops.MUL dtypes.weakint [32, 4]
34 Ops.RANGE dtypes.weakint [4] (AxisType.UPCAST, 3)
35 Ops.ADD dtypes.weakint [33, 34]
36 Ops.ADD dtypes.weakint [22, 35]
37 Ops.INDEX dtypes.f32 [1, 36]
38 Ops.CONST dtypes.weakint [] 65536
39 Ops.PARAM dtypes.f32 [38] ParamArg(slot=1, dtype=dtypes.f32)
40 Ops.CONST dtypes.weakint [] 1024
41 Ops.MUL dtypes.weakint [20, 40]
42 Ops.CONST dtypes.weakint [] 128
43 Ops.RANGE dtypes.weakint [42] (AxisType.WEAK, 0)
44 Ops.MUL dtypes.weakint [43, 4]
45 Ops.ADD dtypes.weakint [44, 26]
46 Ops.MUL dtypes.weakint [45, 4]
47 Ops.ADD dtypes.weakint [46, 31]
48 Ops.MUL dtypes.weakint [47, 4]
49 Ops.ADD dtypes.weakint [48, 34]
50 Ops.ADD dtypes.weakint [41, 49]
51 Ops.INDEX dtypes.f32 [39, 50]
52 Ops.PARAM dtypes.f32 [38] ParamArg(slot=2, dtype=dtypes.f32)
53 Ops.ADD dtypes.weakint [44, 10]
54 Ops.MUL dtypes.weakint [53, 4]
55 Ops.ADD dtypes.weakint [54, 15]
56 Ops.MUL dtypes.weakint [55, 4]
57 Ops.ADD dtypes.weakint [56, 19]
58 Ops.MUL dtypes.weakint [57, 21]
59 Ops.ADD dtypes.weakint [58, 35]
60 Ops.INDEX dtypes.f32 [52, 59]
61 Ops.CONST dtypes.weakfloat [] 0.0
62 Ops.CAST dtypes.f32 [61] dtypes.f32
63 Ops.STACK dtypes.f32 [62, 62]
64 Ops.WMMA dtypes.f32 [51, 60, 63] ((8, 8, 8), dtypes.f32, 32, ((((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),), (((AxisType.UPCAST, 3), 2),)))
65 Ops.REDUCE dtypes.f32 [64, 43] (Ops.ADD, 0)
66 Ops.STORE dtypes.void [37, 65]
67 Ops.END dtypes.void [66, 3, 7, 23, 34]
68 Ops.SINK dtypes.void [67] KernelInfo(name="r_8_8_32_2_128", applied_opts=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)),), optimized=True)

