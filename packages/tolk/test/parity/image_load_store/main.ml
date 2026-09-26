(* Image reads and writes through coalescing, coordinate selection and rendering. *)
open Tolk_uop
module U = Uop

let kernel dtype =
  let src = U.param ~slot:0 ~dtype ~shape:(U.const_int 256) () in
  let dst = U.param ~slot:1 ~dtype ~shape:(U.const_int 256) () in
  let y = U.range ~size:(U.const_int 8) ~axis:0 ~kind:Axis_type.Global () in
  let x = U.range ~size:(U.const_int 8) ~axis:1 ~kind:Axis_type.Global () in
  let c = U.range ~size:(U.const_int 4) ~axis:2 ~kind:Axis_type.Upcast () in
  let index = U.O.(y * int_ 32 + x * int_ 4 + c) in
  let value = U.alu_binary ~op:Ops.Add
      ~lhs:(U.load ~src:(U.index ~ptr:src ~idxs:[ index ] ()) ())
      ~rhs:(U.const (Const.float dtype 0.5)) in
  let store = U.store ~dst:(U.index ~ptr:dst ~idxs:[ index ] ()) ~value () in
  let kernel_info : U.kernel_info =
    { name = "image_copy"; applied_opts = []; opts_to_apply = Some [];
      estimates = None; beam = 0 } in
  U.sink ~kernel_info [ U.end_ ~value:store ~ranges:[ y; x; c ] ]

let () =
  Tolk.Helpers.Context_var.with_context [ Tolk.Helpers.Context_var.B (Tolk.Helpers.image, 2) ]
    (fun () ->
      List.iter (fun (name, dtype) ->
        Helpers.dump ~backends:[ name, Tolk.Cstyle.opencl "IMAGE_PITCH_ALIGNMENT=8" ]
          ~optimize:false ~stages:[ Helpers.Stage5; Helpers.Stage7 ]
          ~out_dir:Sys.argv.(1) (kernel dtype))
        [ "opencl_float", Dtype.float32; "opencl_half", Dtype.float16 ])
