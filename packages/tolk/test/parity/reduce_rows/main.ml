(* Parity case: b[i] = sum_j(a[i*32+j]), 1 Global + 1 Reduce range. *)

open Tolk_uop
module U = Uop

let kernel () =
  let rows, cols = (8, 32) in
  let p0 = U.param ~slot:0 ~dtype:Helpers.global_fptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:Helpers.global_fptr ~shape:(U.const_int (-1)) () in
  let ri = U.range ~size:(Helpers.idx rows) ~axis:0 ~kind:Axis_type.Global () in
  let rj = U.range ~size:(Helpers.idx cols) ~axis:1 ~kind:Axis_type.Reduce () in
  let open U.O in
  let flat = (ri * int_ cols) + rj in
  let ld = U.load ~src:(U.index ~ptr:p0 ~idxs:[flat] ()) () in
  let red = U.reduce ~op:Ops.Add ~src:ld ~ranges:[ rj ] ~dtype:Dtype.float32 in
  let st =
    U.store ~dst:(U.index ~ptr:p1 ~idxs:[ri] ()) ~value:red ()
  in
  let e = U.end_ ~value:st ~ranges:[ ri ] in
  U.sink
    ~kernel_info:
      {
        U.name = "reduce_rows";
        axis_types = [ Axis_type.Global; Axis_type.Reduce ];
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
