(* Parity case: b[0] = sum(a[i]); c[0] = sum(a[i]*a[i]), 1 Reduce range, 2 stores. *)

open Tolk_uop
module U = Uop

let kernel () =
  let p0 = U.param ~slot:0 ~dtype:Helpers.global_fptr () in
  let p1 = U.param ~slot:1 ~dtype:Helpers.global_fptr () in
  let p2 = U.param ~slot:2 ~dtype:Helpers.global_fptr () in
  let r0 = U.range ~size:(Helpers.idx 128) ~axis:0 ~kind:Axis_type.Reduce () in
  let ld = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let red1 = U.reduce ~op:Ops.Add ~src:ld ~ranges:[ r0 ] ~dtype:Dtype.float32 in
  let sq = U.alu_binary ~op:Ops.Mul ~lhs:ld ~rhs:ld in
  let red2 =
    U.reduce ~op:Ops.Add ~src:sq ~ranges:[ r0 ] ~dtype:Dtype.float32
  in
  let c0 = Helpers.idx 0 in
  let st1 =
    U.store ~dst:(U.index ~ptr:p1 ~idxs:[c0] ()) ~value:red1 ()
  in
  let st2 =
    U.store ~dst:(U.index ~ptr:p2 ~idxs:[c0] ()) ~value:red2 ()
  in
  U.sink
    ~kernel_info:
      {
        U.name = "parallel_reduce";
        axis_types = [ Axis_type.Reduce ];
        dont_use_locals = false;
        applied_opts = [];
        opts_to_apply = Some [];
        estimates = None;
      beam = 0;
      }
    [ st1; st2 ]

let () =
  Helpers.dump
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (kernel ())
