# tolk D501 golden=5fd00d2c230ada1dfdf773b1be5ca17a
0 Ops.CONST dtypes.weakint [] 255
1 Ops.PARAM dtypes.i32 [0] ParamArg(slot=0, dtype=dtypes.i32)
2 Ops.CONST dtypes.weakint [] 51
3 Ops.CAST dtypes.i32 [2] dtypes.i32
4 Ops.SPECIAL dtypes.i32 [3] "gidx0"
5 Ops.CONST dtypes.weakint [] 5
6 Ops.CAST dtypes.i32 [5] dtypes.i32
7 Ops.MUL dtypes.i32 [4, 6]
8 Ops.SPECIAL dtypes.i32 [6] "lidx0"
9 Ops.ADD dtypes.i32 [7, 8]
10 Ops.INDEX dtypes.i32 [1, 9]
11 Ops.STORE dtypes.void [10, 9]
12 Ops.SINK dtypes.void [11] KernelInfo(name="E_51_5", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(5, AxisType.LOCAL)),), optimized=True)
13 Ops.SINK dtypes.void [11] KernelInfo(name="E_51_5", applied_opts=(Opt(op=OptOps.SPLIT, axis=0, arg=(5, AxisType.LOCAL)),), estimates=Estimates(lds=1020, mem=1020), optimized=True)
14 Ops.LINEAR dtypes.void [0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 13]
15 Ops.SOURCE dtypes.void [] "typedef long unsigned int size_t;\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);\nextern \"C\" __attribute__((device, const)) size_t __ockl_get_local_size(unsigned int);\nextern \"C\" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 5))) E_51_5(int* data0_255) {\n  int gidx0 = __ockl_get_group_id(0); /* 51 */\n  int lidx0 = __ockl_get_local_id(0); /* 5 */\n  int alu0 = ((gidx0*5)+lidx0);\n  *(data0_255+alu0) = alu0;\n}"
16 Ops.PROGRAM dtypes.void [13, 14, 15] ProgramInfo(global_size=(51, 1, 1), local_size=(5, 1, 1), params=(%1,), globals=(0,), outs=(0,), target=Target(device="AMD", renderer="HIP", arch="gfx1100"))
17 Ops.SINK dtypes.void [12, 16]

