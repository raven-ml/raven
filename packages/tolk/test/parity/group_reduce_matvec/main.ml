(* Parity case: matvec y[j] = sum_k w[j*64+k] * x[k], j=256, k=64,
   with GROUP(0,8) + LOCAL(0,4) + UPCAST(0,4) applied explicitly.

   The matvec grouping the heuristic picks for gpt2's decode step: a partial
   reduce accumulated through a shared-memory tile indexed by (local, group,
   upcast-lane). Pins the local-buffer materialization for a Stage carrying
   both an upstream local range and a vectorized value. Backends are
   limited to cuda and opencl: metal is unavailable where the reference
   corpus is generated. *)

open Tolk_uop
module U = Uop

let backends =
  List.filter
    (fun (name, _) -> name = "cuda" || name = "opencl")
    Helpers.all_backends

let kernel () =
  let n, k = (256, 64) in
  let pw = U.param ~slot:0 ~dtype:Helpers.global_fptr () in
  let px = U.param ~slot:1 ~dtype:Helpers.global_fptr () in
  let py = U.param ~slot:2 ~dtype:Helpers.global_fptr () in
  let rj = U.range ~size:(Helpers.idx n) ~axis:0 ~kind:Axis_type.Global () in
  let rk = U.range ~size:(Helpers.idx k) ~axis:1 ~kind:Axis_type.Reduce () in
  let open U.O in
  let ld_w =
    U.load ~src:(U.index ~ptr:pw ~idxs:[ (rj * int_ k) + rk ] ()) ()
  in
  let ld_x = U.load ~src:(U.index ~ptr:px ~idxs:[ rk ] ()) () in
  let red =
    U.reduce ~op:Ops.Add ~src:(ld_w * ld_x) ~ranges:[ rk ]
      ~dtype:Dtype.float32
  in
  let st =
    U.store ~dst:(U.index ~ptr:py ~idxs:[ rj ] ()) ~value:red ()
  in
  let e = U.end_ ~value:st ~ranges:[ rj ] in
  U.sink
    ~kernel_info:
      {
        U.name = "group_reduce_matvec";
        axis_types = [ Axis_type.Global; Axis_type.Reduce ];
        dont_use_locals = false;
        applied_opts = [];
        opts_to_apply =
          Some
            [
              U.Opt.Group { axis = 0; amount = 8 };
              U.Opt.Local { axis = 0; amount = 4 };
              U.Opt.Upcast { axis = 0; amount = 4 };
            ];
        estimates = None;
        beam = 0;
      }
    [ e ]

let () =
  Helpers.dump ~backends
    ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
    ~out_dir:Sys.argv.(1) (kernel ())
