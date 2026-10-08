(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The thread trace programs each generation writes, as PM4 packets: Mesa's
   ac_sqtt.c sequence. A baseline: the .mli states what the program does
   (test_thread_trace.ml), these its words. Engine e's buffer is at (e + 1) MiB
   and its end at 0x1000 + 8e. *)

open Windtrap
open Rig_amd_abi
module S = Rig_amd_abi_support

let timeout = S.timeout

let program g =
  S.pm4 g
    (S.encode
       (Thread_trace.start g ~size:0x10_0000 (fun e -> (e + 1) lsl 20)
       @ Thread_trace.stop g (fun e -> 0x1000 + (8 * e))))

let programs =
  group ~timeout "programs"
    [
      test "GFX 9.4.3, two dies of two engines" (fun () ->
          expect (program (S.gpu ~xccs:2 ~shader_engines:2 (9, 4, 3)))
          @@ __POS_OF__
               {|
            ACQUIRE_MEM 0x28c40000 0xffffffff 0xffffffff 0x0 0x0 0xa
            SET_UCONFIG_REG regSPI_CONFIG_CNTL=0x362c688
            PRED_EXEC 0x1000024
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0x40000000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BASE2=0x0
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BASE=0x100
            SET_UCONFIG_REG regSQ_THREAD_TRACE_SIZE=0x100
            SET_UCONFIG_REG regSQ_THREAD_TRACE_CTRL=0x80000000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_MASK=0xcf80
            SET_UCONFIG_REG regSQ_THREAD_TRACE_TOKEN_MASK=0xffbfff
            SET_UCONFIG_REG regSQ_THREAD_TRACE_PERF_MASK=0xffffffff
            SET_UCONFIG_REG regSQ_THREAD_TRACE_TOKEN_MASK2=0xffffffff
            SET_UCONFIG_REG regSQ_THREAD_TRACE_HIWATER=0x4
            SET_UCONFIG_REG regSQ_THREAD_TRACE_STATUS=0x0
            SET_UCONFIG_REG regSQ_THREAD_TRACE_MODE=0x6249249
            PRED_EXEC 0x1000024
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0x40010000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BASE2=0x0
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BASE=0x200
            SET_UCONFIG_REG regSQ_THREAD_TRACE_SIZE=0x100
            SET_UCONFIG_REG regSQ_THREAD_TRACE_CTRL=0x80000000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_MASK=0xcf80
            SET_UCONFIG_REG regSQ_THREAD_TRACE_TOKEN_MASK=0xffbfff
            SET_UCONFIG_REG regSQ_THREAD_TRACE_PERF_MASK=0xffffffff
            SET_UCONFIG_REG regSQ_THREAD_TRACE_TOKEN_MASK2=0xffffffff
            SET_UCONFIG_REG regSQ_THREAD_TRACE_HIWATER=0x4
            SET_UCONFIG_REG regSQ_THREAD_TRACE_STATUS=0x0
            SET_UCONFIG_REG regSQ_THREAD_TRACE_MODE=0x6249249
            PRED_EXEC 0x2000024
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0x40000000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BASE2=0x0
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BASE=0x300
            SET_UCONFIG_REG regSQ_THREAD_TRACE_SIZE=0x100
            SET_UCONFIG_REG regSQ_THREAD_TRACE_CTRL=0x80000000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_MASK=0xcf80
            SET_UCONFIG_REG regSQ_THREAD_TRACE_TOKEN_MASK=0xff93ff
            SET_UCONFIG_REG regSQ_THREAD_TRACE_PERF_MASK=0xffffffff
            SET_UCONFIG_REG regSQ_THREAD_TRACE_TOKEN_MASK2=0xffffffff
            SET_UCONFIG_REG regSQ_THREAD_TRACE_HIWATER=0x4
            SET_UCONFIG_REG regSQ_THREAD_TRACE_STATUS=0x0
            SET_UCONFIG_REG regSQ_THREAD_TRACE_MODE=0x6249249
            PRED_EXEC 0x2000024
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0x40010000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BASE2=0x0
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BASE=0x400
            SET_UCONFIG_REG regSQ_THREAD_TRACE_SIZE=0x100
            SET_UCONFIG_REG regSQ_THREAD_TRACE_CTRL=0x80000000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_MASK=0xcf80
            SET_UCONFIG_REG regSQ_THREAD_TRACE_TOKEN_MASK=0xff93ff
            SET_UCONFIG_REG regSQ_THREAD_TRACE_PERF_MASK=0xffffffff
            SET_UCONFIG_REG regSQ_THREAD_TRACE_TOKEN_MASK2=0xffffffff
            SET_UCONFIG_REG regSQ_THREAD_TRACE_HIWATER=0x4
            SET_UCONFIG_REG regSQ_THREAD_TRACE_STATUS=0x0
            SET_UCONFIG_REG regSQ_THREAD_TRACE_MODE=0x6249249
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0xe0000000
            SET_SH_REG regCOMPUTE_THREAD_TRACE_ENABLE=0x1
            ACQUIRE_MEM 0x28c40000 0xffffffff 0xffffffff 0x0 0x0 0xa
            ACQUIRE_MEM 0x28c40000 0xffffffff 0xffffffff 0x0 0x0 0xa
            SET_SH_REG regCOMPUTE_THREAD_TRACE_ENABLE=0x0
            EVENT_WRITE 0x37
            PRED_EXEC 0x1000013
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0x40000000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_MODE=0x0
            WAIT_REG_MEM 0x3 0x33a 0x0 0x0 0x40000000 0x4
            COPY_DATA 0x100204 0xc339 0x0 0x1000 0x0
            PRED_EXEC 0x1000013
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0x40010000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_MODE=0x0
            WAIT_REG_MEM 0x3 0x33a 0x0 0x0 0x40000000 0x4
            COPY_DATA 0x100204 0xc339 0x0 0x1008 0x0
            PRED_EXEC 0x2000013
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0x40000000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_MODE=0x0
            WAIT_REG_MEM 0x3 0x33a 0x0 0x0 0x40000000 0x4
            COPY_DATA 0x100204 0xc339 0x0 0x1010 0x0
            PRED_EXEC 0x2000013
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0x40010000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_MODE=0x0
            WAIT_REG_MEM 0x3 0x33a 0x0 0x0 0x40000000 0x4
            COPY_DATA 0x100204 0xc339 0x0 0x1018 0x0
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0xe0000000
            SET_UCONFIG_REG regSPI_CONFIG_CNTL=0x62c688
            ACQUIRE_MEM 0x28c40000 0xffffffff 0xffffffff 0x0 0x0 0xa
            |});
      test "GFX 11.0.0, two engines" (fun () ->
          expect (program (S.gpu ~shader_engines:2 (11, 0, 0)))
          @@ __POS_OF__
               {|
            ACQUIRE_MEM 0x0 0xffffffff 0xffffffff 0x0 0x0 0x0 0xc3f1
            SET_UCONFIG_REG regSPI_CONFIG_CNTL=0xc362c688
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0x40000000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BUF0_SIZE=0x10000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BUF0_BASE=0x100
            SET_UCONFIG_REG regSQ_THREAD_TRACE_MASK=0x15400
            SET_UCONFIG_REG regSQ_THREAD_TRACE_TOKEN_MASK=0x3f1000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_CTRL=0x80023d41
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0x40010000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BUF0_SIZE=0x10000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BUF0_BASE=0x200
            SET_UCONFIG_REG regSQ_THREAD_TRACE_MASK=0x15400
            SET_UCONFIG_REG regSQ_THREAD_TRACE_TOKEN_MASK=0x3f1000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_CTRL=0x80023d41
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0xe0000000
            SET_SH_REG regCOMPUTE_THREAD_TRACE_ENABLE=0x1
            ACQUIRE_MEM 0x0 0xffffffff 0xffffffff 0x0 0x0 0x0 0xc3f1
            ACQUIRE_MEM 0x0 0xffffffff 0xffffffff 0x0 0x0 0x0 0xc3f1
            SET_SH_REG regCOMPUTE_THREAD_TRACE_ENABLE=0x0
            EVENT_WRITE 0x37
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0x40000000
            WAIT_REG_MEM 0x5 0xd9f4 0x0 0x1 0xfff000 0x4
            SET_UCONFIG_REG regSQ_THREAD_TRACE_CTRL=0x80023d40
            WAIT_REG_MEM 0x3 0xd9f4 0x0 0x0 0x2000000 0x4
            COPY_DATA 0x100204 0xd9ef 0x0 0x1000 0x0
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0x40010000
            WAIT_REG_MEM 0x5 0xd9f4 0x0 0x1 0xfff000 0x4
            SET_UCONFIG_REG regSQ_THREAD_TRACE_CTRL=0x80023d40
            WAIT_REG_MEM 0x3 0xd9f4 0x0 0x0 0x2000000 0x4
            COPY_DATA 0x100204 0xd9ef 0x0 0x1008 0x0
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0xe0000000
            SET_UCONFIG_REG regSPI_CONFIG_CNTL=0xc062c688
            ACQUIRE_MEM 0x0 0xffffffff 0xffffffff 0x0 0x0 0x0 0xc3f1
            |});
      test "GFX 12.0.1, two engines" (fun () ->
          expect (program (S.gpu ~shader_engines:2 (12, 0, 1)))
          @@ __POS_OF__
               {|
            ACQUIRE_MEM 0x0 0xffffffff 0xffffffff 0x0 0x0 0x0 0xc3f1
            SET_UCONFIG_REG regSPI_SQG_EVENT_CTL=0x3
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0x40000000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BUF0_SIZE=0x100
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BUF0_BASE_LO=0x100
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BUF0_BASE_HI=0x0
            SET_UCONFIG_REG regSQ_THREAD_TRACE_WPTR=0x0
            SET_UCONFIG_REG regSQ_THREAD_TRACE_MASK=0x15400
            SET_UCONFIG_REG regSQ_THREAD_TRACE_TOKEN_MASK=0x83f6000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_CTRL=0x80405d41
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0x40010000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BUF0_SIZE=0x100
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BUF0_BASE_LO=0x200
            SET_UCONFIG_REG regSQ_THREAD_TRACE_BUF0_BASE_HI=0x0
            SET_UCONFIG_REG regSQ_THREAD_TRACE_WPTR=0x0
            SET_UCONFIG_REG regSQ_THREAD_TRACE_MASK=0x15400
            SET_UCONFIG_REG regSQ_THREAD_TRACE_TOKEN_MASK=0x83f6000
            SET_UCONFIG_REG regSQ_THREAD_TRACE_CTRL=0x80405d41
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0xe0000000
            SET_SH_REG regCOMPUTE_THREAD_TRACE_ENABLE=0x1
            ACQUIRE_MEM 0x0 0xffffffff 0xffffffff 0x0 0x0 0x0 0xc3f1
            ACQUIRE_MEM 0x0 0xffffffff 0xffffffff 0x0 0x0 0x0 0xc3f1
            SET_SH_REG regCOMPUTE_THREAD_TRACE_ENABLE=0x0
            EVENT_WRITE 0x37
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0x40000000
            WAIT_REG_MEM 0x5 0xd9f4 0x0 0x1 0xfff000 0x4
            SET_UCONFIG_REG regSQ_THREAD_TRACE_CTRL=0x80405d40
            WAIT_REG_MEM 0x3 0xd9f4 0x0 0x0 0x2000000 0x4
            COPY_DATA 0x100204 0xd9ef 0x0 0x1000 0x0
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0x40010000
            WAIT_REG_MEM 0x5 0xd9f4 0x0 0x1 0xfff000 0x4
            SET_UCONFIG_REG regSQ_THREAD_TRACE_CTRL=0x80405d40
            WAIT_REG_MEM 0x3 0xd9f4 0x0 0x0 0x2000000 0x4
            COPY_DATA 0x100204 0xd9ef 0x0 0x1008 0x0
            SET_UCONFIG_REG regGRBM_GFX_INDEX=0xe0000000
            SET_UCONFIG_REG regSPI_SQG_EVENT_CTL=0x0
            ACQUIRE_MEM 0x0 0xffffffff 0xffffffff 0x0 0x0 0x0 0xc3f1
            |});
    ]

let () = exit (run "rig_amd_abi.trace_program" [ programs ])
