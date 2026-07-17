(* Parity case: c[i] = sqrt(a[i]), 1 Global range, unary SQRT. *)

open Tolk_uop
module U = Uop

let kernel () =
  let p0 = U.param ~slot:0 ~dtype:Helpers.global_fptr () in
  let p1 = U.param ~slot:1 ~dtype:Helpers.global_fptr () in
  let r0 = U.range ~size:(Helpers.idx 256) ~axis:0 ~kind:Axis_type.Global () in
  let ld = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let sq = U.alu_unary ~op:Ops.Sqrt ~src:ld in
  let st =
    U.store ~dst:(U.index ~ptr:p1 ~idxs:[r0] ()) ~value:sq ()
  in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  U.sink
    ~kernel_info:
      {
        U.name = "elementwise_sqrt";
        axis_types = [ Axis_type.Global ];
        dont_use_locals = false;
        applied_opts = [];
        opts_to_apply = Some [];
        estimates = None;
      beam = 0;
      }
    [ e ]

let () =
  Helpers.dump
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (kernel ())
