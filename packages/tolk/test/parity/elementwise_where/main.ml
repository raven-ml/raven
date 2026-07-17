(* Parity case: c[i] = (a[i] > 0) ? a[i] : 0.0 (ReLU), 1 Global range. *)

open Tolk_uop
module U = Uop

let kernel () =
  let p0 = U.param ~slot:0 ~dtype:Helpers.global_fptr () in
  let p1 = U.param ~slot:1 ~dtype:Helpers.global_fptr () in
  let r0 = U.range ~size:(Helpers.idx 256) ~axis:0 ~kind:Axis_type.Global () in
  let ld = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let zero = U.const (Const.float Dtype.float32 0.0) in
  let cond = U.alu_binary ~op:Ops.Cmplt ~lhs:zero ~rhs:ld in
  let w = U.alu_ternary ~op:Ops.Where ~a:cond ~b:ld ~c:zero in
  let st =
    U.store ~dst:(U.index ~ptr:p1 ~idxs:[r0] ()) ~value:w ()
  in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  U.sink
    ~kernel_info:
      {
        U.name = "elementwise_where";
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
