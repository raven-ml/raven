(* Parity case: c[i] = a[i] + b[i] (all int32), 1 Global range. *)

open Tolk_uop
module U = Uop

let i32_ptr = Dtype.int32

let kernel () =
  let p0 = U.param ~slot:0 ~dtype:i32_ptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:i32_ptr ~shape:(U.const_int (-1)) () in
  let p2 = U.param ~slot:2 ~dtype:i32_ptr ~shape:(U.const_int (-1)) () in
  let r0 = U.range ~size:(Helpers.idx 256) ~axis:0 ~kind:Axis_type.Global () in
  let ld_a = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let ld_b = U.load ~src:(U.index ~ptr:p1 ~idxs:[r0] ()) () in
  let add = U.alu_binary ~op:Ops.Add ~lhs:ld_a ~rhs:ld_b in
  let st =
    U.store ~dst:(U.index ~ptr:p2 ~idxs:[r0] ()) ~value:add ()
  in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  U.sink
    ~kernel_info:
      {
        U.name = "elementwise_int32";
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
