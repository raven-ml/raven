(* Symbolic contraction eligibility followed by actual WMMA lowering. *)

open Tolk_uop
module U = Uop

let backends =
  [ "metal", Tolk.Cstyle.metal (Tolk.Gpu_target.Apple 7);
    "cuda", Tolk.Cstyle.cuda Tolk.Gpu_target.SM80;
    "amd", Tolk.Cstyle.amd Tolk.Gpu_target.RDNA3 ]

let kernel aligned =
  let m, n, k = (128, 128, 128) in
  let pa =
    U.param ~slot:0 ~dtype:Dtype.float16 ~shape:(U.const_int (m * k)) ()
  in
  let pb =
    U.param ~slot:1 ~dtype:Dtype.float16 ~shape:(U.const_int (k * n)) ()
  in
  let pc =
    U.param ~slot:2 ~dtype:Dtype.float32 ~shape:(U.const_int (m * n)) ()
  in
  let ri = U.range ~size:(U.const_int m) ~axis:0 ~kind:Axis_type.Global () in
  let rj = U.range ~size:(U.const_int n) ~axis:1 ~kind:Axis_type.Global () in
  let extent = U.variable ~name:"tc_k" ~min_val:1 ~max_val:2 ~param:true () in
  let size = U.O.(extent * int_ (if aligned then 8 else 1)) in
  let rk = U.range ~size ~axis:2 ~kind:Axis_type.Reduce () in
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
        U.name = "tc_symbolic_extent";
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
  ast

let () =
  let lines = ref [] in
  List.iter (fun (label, aligned) ->
    List.iter (fun (name, renderer) ->
      let ast = kernel aligned in
      let scheduler = Tolk.Postrange.create ast renderer in
      let tc = U.Opt.Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = 1 } in
      match Tolk.Postrange.apply_opt scheduler tc with
      | exception Tolk.Postrange.Opt_error _ ->
          lines := Printf.sprintf "%s %s: rejected" label name :: !lines
      | _ ->
          let n, m, k = (Option.get (Tolk.Postrange.tensor_core scheduler)).dims in
          lines := Printf.sprintf "%s %s: %dx%dx%d" label name n m k :: !lines;
          Helpers.dump ~backends:[label ^ "_" ^ name, renderer]
            ~stages:[Helpers.Stage5; Helpers.Stage7] ~out_dir:Sys.argv.(1) ast)
      backends)
    ["aligned", true; "unaligned", false];
  let channel = open_out (Filename.concat Sys.argv.(1) "eligibility.actual") in
  Fun.protect ~finally:(fun () -> close_out channel) (fun () ->
    output_string channel (String.concat "\n" (List.rev !lines) ^ "\n"))
