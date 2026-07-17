(* Parity case: c[i*16+j] = a[i*16+j] + b[i*16+j], 2 Global ranges. GPU-only. *)

open Tolk_uop
module U = Uop

let kernel () =
  let rows, cols = (8, 16) in
  let p0 = U.param ~slot:0 ~dtype:Helpers.global_fptr () in
  let p1 = U.param ~slot:1 ~dtype:Helpers.global_fptr () in
  let p2 = U.param ~slot:2 ~dtype:Helpers.global_fptr () in
  let ri = U.range ~size:(Helpers.idx rows) ~axis:0 ~kind:Axis_type.Global () in
  let rj = U.range ~size:(Helpers.idx cols) ~axis:1 ~kind:Axis_type.Global () in
  let open U.O in
  let flat = (ri * int_ cols) + rj in
  let ld_a = U.load ~src:(U.index ~ptr:p0 ~idxs:[flat] ()) () in
  let ld_b = U.load ~src:(U.index ~ptr:p1 ~idxs:[flat] ()) () in
  let add = U.alu_binary ~op:Ops.Add ~lhs:ld_a ~rhs:ld_b in
  let st =
    U.store ~dst:(U.index ~ptr:p2 ~idxs:[flat] ()) ~value:add ()
  in
  let e = U.end_ ~value:st ~ranges:[ ri; rj ] in
  U.sink
    ~kernel_info:
      {
        U.name = "elementwise_2d";
        axis_types = [ Axis_type.Global; Axis_type.Global ];
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
