(* Parity case: C = A @ B, float16 inputs, float32 accumulate, M=N=K=32,
   scheduled by the explicit opts [TC:0:-1:0:1] and [UNROLL:0:0].

   Metal (Apple7) and CUDA (SM80) both carry tensor cores, and the shape is
   large enough that the accumulator holds several WMMAs — so this case
   observes the WMMA operand widths and the order the contracted axes reach
   the register array, neither of which a single-WMMA kernel can see. *)

open Tolk_uop
module U = Uop

let backends =
  [
    ("cuda", Tolk.Cstyle.cuda Tolk.Gpu_target.SM80);
    ("metal", Tolk.Cstyle.metal (Tolk.Gpu_target.Apple 7));
  ]

let kernel () =
  let m, n, k = (32, 32, 32) in
  let pa =
    U.param ~slot:0 ~dtype:Dtype.float16 ~shape:(U.const_int (m * k)) ()
  in
  let pb =
    U.param ~slot:1 ~dtype:Dtype.float16 ~shape:(U.const_int (k * n)) ()
  in
  let pc =
    U.param ~slot:2 ~dtype:Dtype.float32 ~shape:(U.const_int (m * n)) ()
  in
  let ri = U.range ~size:(U.const_int m) ~axis:0 ~kind:Axis_type.Global () in
  let rj = U.range ~size:(U.const_int n) ~axis:1 ~kind:Axis_type.Global () in
  let rk = U.range ~size:(U.const_int k) ~axis:2 ~kind:Axis_type.Reduce () in
  let open U.O in
  let ld_a = U.load ~src:(U.index ~ptr:pa ~idxs:[ (ri * int_ k) + rk ] ()) () in
  let ld_b = U.load ~src:(U.index ~ptr:pb ~idxs:[ (rk * int_ n) + rj ] ()) () in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:ld_a ~rhs:ld_b in
  let mulf = U.cast ~src:mul ~dtype:Dtype.float32 in
  let red = U.reduce ~op:Ops.Add ~src:mulf ~ranges:[ rk ] ~dtype:Dtype.float32 in
  let st =
    U.store ~dst:(U.index ~ptr:pc ~idxs:[ (ri * int_ n) + rj ] ()) ~value:red ()
  in
  let e = U.end_ ~value:st ~ranges:[ ri; rj ] in
  U.sink
    ~kernel_info:
      {
        U.name = "tc_matmul_32";
        axis_types = [ Axis_type.Global; Axis_type.Global; Axis_type.Reduce ];
        dont_use_locals = false;
        applied_opts = [];
        opts_to_apply =
          Some
            [
              U.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 1 };
              U.Opt.Unroll { axis = 0; amount = 0 };
            ];
        estimates = None;
        beam = 0;
      }
    [ e ]

let () =
  Helpers.dump ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (kernel ())
