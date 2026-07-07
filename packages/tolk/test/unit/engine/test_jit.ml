(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop

let renderer =
  Renderer.make ~name:"test" ~device:"TEST" ~has_local:false ~has_shared:false
    ~shared_max:0
    ~render:(fun ?name:_ _ -> "")
    ()

let allocator =
  let module Raw = struct
    type t = { data : bytes; offset : int; nbytes : int }
  end in
  Device.Allocator.Pack
    {
      Device.Allocator.alloc =
        (fun nbytes _spec ->
          Raw.{ data = Bytes.make nbytes '\000'; offset = 0; nbytes });
      free = (fun _ _ _ -> ());
      copyin = (fun raw src -> Bytes.blit src 0 raw.Raw.data raw.offset raw.nbytes);
      copyout =
        (fun dst raw -> Bytes.blit raw.Raw.data raw.offset dst 0 raw.nbytes);
      addr = (fun _ -> Nativeint.zero);
      offset =
        Some
          (fun raw nbytes byte_offset ->
            Raw.{ data = raw.data; offset = raw.offset + byte_offset; nbytes });
      transfer = None;
      supports_transfer = false;
      copy_from_disk = None;
      supports_copy_from_disk = false;
    }

let make_device ?(name = "TEST:0") ?(counter = ref 0) () =
  Device.make ~name ~allocator
    ~renderer_set:(Device.Renderer_set.make [ renderer, None ])
    ~runtime:(fun _ _ ~runtimevars:_ ->
      {
        Device.call =
          (fun _ ~global:_ ~local:_ ~vals:_ ~wait:_ ~timeout:_ ->
            incr counter;
            None);
        free = (fun () -> ());
      })
    ~synchronize:(fun () -> ())
    ()

let device = make_device ()

let to_program body =
  let info = U.program_info_from_sink body in
  U.program ~sink:body ~linear:(U.linear []) ~source:(U.source "")
    ~binary:(U.binary "") ~info ()

let buffer ?(size = 4) ?(dtype = Dtype.int32) () =
  Device.create_buffer ~size ~dtype device

let shape_const n = U.const (Const.int Dtype.Val.weakint n)

let buffer_node ~slot ?(size = 4) () =
  U.buffer ~slot ~dtype:Dtype.int32 ~shape:(shape_const size)
    ~device:(U.Single "TEST:0") ()

let kernel_info name : U.kernel_info =
  {
    name;
    axis_types = [];
    dont_use_locals = false;
    applied_opts = [];
    opts_to_apply = None;
    estimates = None;
    beam = 0;
  }

let call_info name : U.call_info =
  {
    grad_fxn = None;
    metadata = [];
    name;
    precompile = false;
    precompile_backward = false;
    aux = None;
  }

(* A JIT wrapping a single kernel with one external input and one output
   buffer. Returns the device buffer the runtime call counter, and a driver
   that runs the JIT with a fresh input. *)
let make_kernel_jit () =
  let counter = ref 0 in
  let dev = make_device ~counter () in
  let out_buf = Device.create_buffer ~size:4 ~dtype:Dtype.int32 dev in
  let out_node = buffer_node ~slot:(-1) () in
  let in_node = buffer_node ~slot:0 () in
  let body = U.sink ~kernel_info:(kernel_info "jit_k") [] in
  let linear =
    U.linear
      [ U.call ~body ~args:[ out_node; in_node ] ~info:(call_info (Some "jit_k")) ]
  in
  let buffers_ref = ref (fun _ -> None) in
  let fxn input_bufs _ =
    if Jit.is_capturing () then begin
      Jit.add_linear linear;
      let cap_in = input_bufs.(0) in
      buffers_ref :=
        (fun node ->
          if U.tag node = U.tag out_node then Some out_buf
          else if U.tag node = U.tag in_node then Some cap_in
          else None);
      "captured"
    end
    else "warmup"
  in
  let tjit = Jit.create ~device:dev ~to_program ~fxn () in
  let run input =
    Jit.call tjit [| input |] [] ~buffers:(fun n -> !buffers_ref n)
  in
  (counter, run)

let raises_jit_error fn =
  raises_match (function Jit.Jit_error _ -> true | _ -> false) fn

let () =
  run "Engine_jit"
    [
      group "TinyJit"
        [
          test "create requires a function or a captured schedule" (fun () ->
            raises_invalid_arg "need either a function or a CapturedJit"
              (fun () ->
                ignore (Jit.create ~device ~to_program ())));
          test "reset requires a function-backed jit" (fun () ->
            let t =
              Jit.create ~device ~to_program
                ~fxn:(fun _ _ -> ())
                ()
            in
            Jit.reset t);
          test "empty capture raises and clears capture state" (fun () ->
            let t =
              Jit.create ~device ~to_program
                ~fxn:(fun _ _ -> "ok")
                ()
            in
            equal string "ok" (Jit.call t [||] [] ~buffers:(fun _ -> None));
            raises_jit_error (fun () ->
                ignore (Jit.call t [||] [] ~buffers:(fun _ -> None)));
            equal bool false (Jit.is_capturing ()));
        ];
      group "Capture and replay"
        [
          test "warmup, capture, and replay run the kernel" (fun () ->
            let counter, run = make_kernel_jit () in
            equal string "warmup" (run (buffer ()));
            equal int 0 !counter;
            equal string "captured" (run (buffer ()));
            equal int 1 !counter;
            equal string "captured" (run (buffer ()));
            equal int 2 !counter);
          test "replay validates input shape dtype and device" (fun () ->
            let _counter, run = make_kernel_jit () in
            ignore (run (buffer ()));
            ignore (run (buffer ()));
            raises_jit_error (fun () ->
                ignore (run (buffer ~dtype:Dtype.float32 ()))));
        ];
    ]
