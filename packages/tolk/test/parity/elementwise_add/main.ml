(* Parity case: c[i] = a[i] + b[i] over a single Global range of 256 elements.
   Paired with main.py. Run `uv run main.py` to regenerate *.expected. *)

open Tolk_uop
module U = Uop

let kernel () =
  let p0 = U.param ~slot:0 ~dtype:Helpers.global_fptr () in
  let p1 = U.param ~slot:1 ~dtype:Helpers.global_fptr () in
  let p2 = U.param ~slot:2 ~dtype:Helpers.global_fptr () in
  let r0 = U.range ~size:(Helpers.idx 256) ~axis:0 ~kind:Axis_type.Global () in
  let ld_a = U.load ~src:(U.index ~ptr:p0 ~idxs:[r0] ~as_ptr:true ()) () in
  let ld_b = U.load ~src:(U.index ~ptr:p1 ~idxs:[r0] ~as_ptr:true ()) () in
  let add = U.alu_binary ~op:Ops.Add ~lhs:ld_a ~rhs:ld_b in
  let st =
    U.store ~dst:(U.index ~ptr:p2 ~idxs:[r0] ~as_ptr:true ()) ~value:add ()
  in
  let e = U.end_ ~value:st ~ranges:[ r0 ] in
  U.sink
    ~kernel_info:
      {
        U.name = "elementwise_add";
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
