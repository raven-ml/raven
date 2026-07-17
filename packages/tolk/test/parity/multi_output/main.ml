(* Parity case: b[i] = a[i] + 1.0; c[i] = a[i] * 2.0, 1 Global range, 2 stores. *)

open Tolk_uop
module U = Uop

let kernel () =
  let p0 = U.param ~slot:0 ~dtype:Helpers.global_fptr () in
  let p1 = U.param ~slot:1 ~dtype:Helpers.global_fptr () in
  let p2 = U.param ~slot:2 ~dtype:Helpers.global_fptr () in
  let r0 = U.range ~size:(Helpers.idx 256) ~axis:0 ~kind:Axis_type.Global () in
  let ld_a = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let one = U.const (Const.float Dtype.float32 1.0) in
  let two = U.const (Const.float Dtype.float32 2.0) in
  let st1 =
    U.store
      ~dst:(U.index ~ptr:p1 ~idxs:[r0] ())
      ~value:(U.alu_binary ~op:Ops.Add ~lhs:ld_a ~rhs:one) ()
  in
  let e1 = U.end_ ~value:st1 ~ranges:[ r0 ] in
  let st2 =
    U.store
      ~dst:(U.index ~ptr:p2 ~idxs:[r0] ())
      ~value:(U.alu_binary ~op:Ops.Mul ~lhs:ld_a ~rhs:two) ()
  in
  let e2 = U.end_ ~value:st2 ~ranges:[ r0 ] in
  U.sink
    ~kernel_info:
      {
        U.name = "multi_output";
        axis_types = [ Axis_type.Global ];
        dont_use_locals = false;
        applied_opts = [];
        opts_to_apply = Some [];
        estimates = None;
      beam = 0;
      }
    [ e1; e2 ]

let () =
  Helpers.dump
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (kernel ())
