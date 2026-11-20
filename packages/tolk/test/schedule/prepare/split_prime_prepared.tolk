# tolk D500 golden=b81626ee72d9817fbca9288ba67f6e08
0 Ops.CONST dtypes.weakint [] 1
1 Ops.PARAM dtypes.f32 [0] ParamArg(slot=0, dtype=dtypes.f32, device="CPU")
2 Ops.STACK dtypes.void []
3 Ops.RESHAPE dtypes.f32 [1, 2]
4 Ops.CONST dtypes.weakint [] 256
5 Ops.ALLOC dtypes.f32 [4] ParamArg(slot=2, dtype=dtypes.f32, device="CPU")
6 Ops.CONST dtypes.weakint [] 65537
7 Ops.PARAM dtypes.f32 [6] ParamArg(slot=1, dtype=dtypes.f32, device="CPU")
8 Ops.CONST dtypes.weakint [] 0
9 Ops.CONST dtypes.weakint [] 65792
10 Ops.PAD dtypes.f32 [7, 8, 9]
11 Ops.CONST dtypes.weakint [] 257
12 Ops.STACK dtypes.weakint [4, 11]
13 Ops.RESHAPE dtypes.f32 [10, 12]
14 Ops.STACK dtypes.weakint [0, 8]
15 Ops.PERMUTE dtypes.f32 [13, 14]
16 Ops.REDUCE dtypes.f32 [15] (Ops.ADD, 1)
17 Ops.STORE dtypes.void [5, 16]
18 Ops.AFTER dtypes.f32 [5, 17]
19 Ops.REDUCE dtypes.f32 [18] (Ops.ADD, 1)
20 Ops.STORE dtypes.void [3, 19]
21 Ops.AFTER dtypes.f32 [1, 20]
22 Ops.SINK dtypes.void [21]

