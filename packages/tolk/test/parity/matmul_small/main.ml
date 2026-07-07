(* Parity case: C = A * B, M=N=K=4. GPU-only. *)

open Tolk_uop
module U = Uop

let kernel () =
  let m, n, k = (4, 4, 4) in
  let pA = U.param ~slot:0 ~dtype:Helpers.global_fptr () in
  let pB = U.param ~slot:1 ~dtype:Helpers.global_fptr () in
  let pC = U.param ~slot:2 ~dtype:Helpers.global_fptr () in
  let ri = U.range ~size:(Helpers.idx m) ~axis:0 ~kind:Axis_type.Global () in
  let rj = U.range ~size:(Helpers.idx n) ~axis:1 ~kind:Axis_type.Global () in
  let rk = U.range ~size:(Helpers.idx k) ~axis:2 ~kind:Axis_type.Reduce () in
  let open U.O in
  let a_idx = (ri * int_ k) + rk in
  let b_idx = (rk * int_ n) + rj in
  let c_idx = (ri * int_ n) + rj in
  let ld_a = U.load ~src:(U.index ~ptr:pA ~idxs:[a_idx] ~as_ptr:true ()) () in
  let ld_b = U.load ~src:(U.index ~ptr:pB ~idxs:[b_idx] ~as_ptr:true ()) () in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:ld_a ~rhs:ld_b in
  let red = U.reduce ~op:Ops.Add ~src:mul ~ranges:[ rk ] ~dtype:Dtype.Val.float32 in
  let st =
    U.store ~dst:(U.index ~ptr:pC ~idxs:[c_idx] ~as_ptr:true ()) ~value:red ()
  in
  let e = U.end_ ~value:st ~ranges:[ ri; rj ] in
  U.sink
    ~kernel_info:
      {
        U.name = "matmul_small";
        axis_types = [ Axis_type.Global; Axis_type.Global; Axis_type.Reduce ];
        dont_use_locals = false;
        applied_opts = [];
        opts_to_apply = Some [];
        estimates = None;
      beam = 0;
      }
    [ e ]

let () =
  Helpers.dump ~backends:Helpers.gpu_backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (kernel ())
