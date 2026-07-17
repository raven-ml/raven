(* Parity case: c[0] = sum_k(a[k] * b[k]) over a single Reduce range of 128. *)

open Tolk_uop
module U = Uop

let kernel () =
  let p0 = U.param ~slot:0 ~dtype:Helpers.global_fptr ~shape:(U.const_int (-1)) () in
  let p1 = U.param ~slot:1 ~dtype:Helpers.global_fptr ~shape:(U.const_int (-1)) () in
  let p2 = U.param ~slot:2 ~dtype:Helpers.global_fptr ~shape:(U.const_int (-1)) () in
  let r0 = U.range ~size:(Helpers.idx 128) ~axis:0 ~kind:Axis_type.Reduce () in
  let ld_a = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ()) () in
  let ld_b = U.load ~src:(U.index ~ptr:p1 ~idxs:[r0] ()) () in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:ld_a ~rhs:ld_b in
  let red = U.reduce ~op:Ops.Add ~src:mul ~ranges:[ r0 ] ~dtype:Dtype.float32 in
  let st =
    U.store
      ~dst:(U.index ~ptr:p2 ~idxs:[(Helpers.idx 0)] ())
      ~value:red ()
  in
  U.sink
    ~kernel_info:
      {
        U.name = "dot_product";
        axis_types = [ Axis_type.Reduce ];
        dont_use_locals = false;
        applied_opts = [];
        opts_to_apply = Some [];
        estimates = None;
      beam = 0;
      }
    [ st ]

let () =
  Helpers.dump
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (kernel ())
