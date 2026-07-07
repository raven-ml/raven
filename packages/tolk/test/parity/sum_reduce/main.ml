(* Parity case: b[0] = sum(a[i]) over a single Reduce range of 256 elements. *)

open Tolk_uop
module U = Uop

let kernel () =
  let p0 = U.param ~slot:0 ~dtype:Helpers.global_fptr () in
  let p1 = U.param ~slot:1 ~dtype:Helpers.global_fptr () in
  let r0 = U.range ~size:(Helpers.idx 256) ~axis:0 ~kind:Axis_type.Reduce () in
  let ld = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ~as_ptr:true ()) () in
  let red = U.reduce ~op:Ops.Add ~src:ld ~ranges:[ r0 ] ~dtype:Dtype.Val.float32 in
  let st =
    U.store
      ~dst:(U.index ~ptr:p1 ~idxs:[(Helpers.idx 0)] ~as_ptr:true ())
      ~value:red ()
  in
  U.sink
    ~kernel_info:
      {
        U.name = "sum_reduce";
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
