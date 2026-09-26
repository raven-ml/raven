(* Large BF16/FNUZ contractions exercise accumulator ordering after expansion. *)

open Tolk_uop
module U = Uop

let backends =
  [
    ("metal_bf16", Tolk.Cstyle.metal (Tolk.Gpu_target.Apple 7), Dtype.bfloat16);
    ("cuda_bf16", Tolk.Cstyle.cuda Tolk.Gpu_target.SM80, Dtype.bfloat16);
    ("amd_gfx1100_bf16", Tolk.Cstyle.amd Tolk.Gpu_target.RDNA3, Dtype.bfloat16);
    ("amd_gfx1201_bf16", Tolk.Cstyle.amd Tolk.Gpu_target.RDNA4, Dtype.bfloat16);
    ("amd_gfx942_bf16", Tolk.Cstyle.amd Tolk.Gpu_target.CDNA3, Dtype.bfloat16);
    ("amd_gfx950_bf16", Tolk.Cstyle.amd Tolk.Gpu_target.CDNA4, Dtype.bfloat16);
    ("amd_gfx942_e4m3fnuz", Tolk.Cstyle.amd Tolk.Gpu_target.CDNA3, Dtype.fp8e4m3fnuz);
    ("amd_gfx942_e5m2fnuz", Tolk.Cstyle.amd Tolk.Gpu_target.CDNA3, Dtype.fp8e5m2fnuz);
  ]

let kernel renderer dtype =
  let m, n, k = (128, 128, 128) in
  let pa =
    U.param ~slot:0 ~dtype ~shape:(U.const_int (m * k)) ()
  in
  let pb =
    U.param ~slot:1 ~dtype ~shape:(U.const_int (k * n)) ()
  in
  let pc =
    U.param ~slot:2 ~dtype:Dtype.float32 ~shape:(U.const_int (m * n)) ()
  in
  let ri = U.range ~size:(U.const_int m) ~axis:0 ~kind:Axis_type.Global () in
  let rj = U.range ~size:(U.const_int n) ~axis:1 ~kind:Axis_type.Global () in
  let rk = U.range ~size:(U.const_int k) ~axis:2 ~kind:Axis_type.Reduce () in
  let open U.O in
  let ld_a = U.load ~src:(U.index ~ptr:pa ~idxs:[ (ri * int_ k) + rk ] ()) () in
  let ld_b = U.load ~src:(U.index ~ptr:pb ~idxs:[ (rk * int_ n) + rj ] ()) () in
  let mul = U.alu_binary ~op:Ops.Mul ~lhs:ld_a ~rhs:ld_b in
  let mulf = U.cast ~src:mul ~dtype:Dtype.float32 in
  let red = U.reduce ~op:Ops.Add ~src:mulf ~ranges:[ rk ] in
  let st =
    U.store ~dst:(U.index ~ptr:pc ~idxs:[ (ri * int_ n) + rj ] ()) ~value:red ()
  in
  let e = U.end_ ~value:st ~ranges:[ ri; rj ] in
  let ast = U.sink
    ~kernel_info:
      {
        U.name = "tc_matmul_wide_types";
        applied_opts = [];
        opts_to_apply =
          Some
            [
              U.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 1 };
            ];
        estimates = None;
        beam = 0;
      }
    [ e ] in
  let scheduler = Tolk.Postrange.create ast renderer in
  let tc = U.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 1 } in
  ignore (Tolk.Postrange.apply_opt scheduler tc);
  let axis = List.hd (Tolk.Postrange.unrollable_dims scheduler) in
  let info = Option.get (U.as_kernel_info ast) in
  U.replace ast ~arg:(U.Arg.Kernel_info { info with opts_to_apply = Some
      [ tc; U.Opt.Split { kind = Axis_type.Unroll; top = false; axis; amount = 0 } ] }) ()

let () =
  List.iter (fun (name, renderer, dtype) ->
      Helpers.dump ~backends:[ name, renderer ]
        ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
        ~out_dir:Sys.argv.(1) (kernel renderer dtype)) backends
