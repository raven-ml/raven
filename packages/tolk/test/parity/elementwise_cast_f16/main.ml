(* Parity case: c[i] = (float32)a_f16[i] + b[i], 1 Global range, mixed dtypes. *)

open Tolk_uop
module U = Uop

let f16_ptr = Dtype.float16

let kernel () =
  let p0 = U.param ~slot:0 ~dtype:f16_ptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:Helpers.global_fptr ~shape:(U.const_int (-1)) () in
  let p2 = U.param ~slot:2 ~dtype:Helpers.global_fptr ~shape:(U.const_int (-1)) () in
  let r0 = U.range ~size:(Helpers.idx 256) ~axis:0 ~kind:Axis_type.Global () in
  let ld_a = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let cast_a = U.cast ~src:ld_a ~dtype:Dtype.float32 in
  let ld_b = U.load ~src:(U.index ~ptr:p1 ~idxs:[r0] ()) () in
  let add = U.alu_binary ~op:Ops.Add ~lhs:cast_a ~rhs:ld_b in
  let st =
    U.store ~dst:(U.index ~ptr:p2 ~idxs:[r0] ()) ~value:add ()
  in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  U.sink
    ~kernel_info:
      {
        U.name = "elementwise_cast_f16";
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
