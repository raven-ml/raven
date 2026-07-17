(* Parity case: c[i] = a[i] + b[i] with store gated by i < 200, range size=256. *)

open Tolk_uop
module U = Uop

let kernel () =
  let p0 = U.param ~slot:0 ~dtype:Helpers.global_fptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:Helpers.global_fptr ~shape:(U.const_int (-1)) () in
  let p2 = U.param ~slot:2 ~dtype:Helpers.global_fptr ~shape:(U.const_int (-1)) () in
  let r0 = U.range ~size:(U.const_int 256) ~axis:0 ~kind:Axis_type.Global () in
  let ld_a = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let ld_b = U.load ~src:(U.index ~ptr:p1 ~idxs:[r0] ()) () in
  let add = U.alu_binary ~op:Ops.Add ~lhs:ld_a ~rhs:ld_b in
  let gate = U.alu_binary ~op:Ops.Cmplt ~lhs:r0 ~rhs:(U.const_int 200) in
  let value = U.O.where gate add (U.invalid ~dtype:Dtype.float32 ()) in
  let st =
    U.store
      ~dst:(U.index ~ptr:p2 ~idxs:[r0] ())
      ~value ()
  in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  U.sink
    ~kernel_info:
      {
        U.name = "gated_store";
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
