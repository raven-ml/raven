(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* JIT phase transitions over a fake device that records runtime launches.
   The jitted function schedules through [Schedule.create_linear_with_vars]
   and executes the returned linear, as the frontend does; during capture the
   schedule is recorded by the JIT and the returned linear is empty. *)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop

type runtime_state = {
  mutable calls : int;
  mutable vals : int64 array;
  mutable global : int array;
}

let runtime_state () = { calls = 0; vals = [||]; global = [||] }

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

let make_device ?(name = "TEST:0") ?(state = runtime_state ()) () =
  Device.make ~name ~allocator
    ~renderer_set:(Device.Renderer_set.make [ renderer, None ])
    ~runtime:(fun _ _ ~runtimevars:_ ->
      {
        Device.call =
          (fun _ ~global ~local:_ ~vals ~wait:_ ~timeout:_ ->
            state.calls <- state.calls + 1;
            state.global <- Array.copy global;
            state.vals <- Array.copy vals;
            None);
        free = (fun () -> ());
        handle = 0n;
      })
    ~synchronize:(fun () -> ())
    ()

let device = make_device ()

let to_program body =
  let info = U.program_info_from_sink body in
  U.program ~sink:body ~linear:(U.linear []) ~source:(U.source "")
    ~binary:(U.binary "") ~info ()

let shape_const n = U.const (Const.int Dtype.weakint n)

let buffer_node ?(size = 4) ?(dtype = Dtype.int32) () =
  U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype ~shape:(shape_const size)
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
    name;
    precompile = false;
    precompile_backward = false;
    aux = None;
  }

(* A JIT wrapping one kernel with one external input and one output buffer.
   [binds] injects BIND arguments into the scheduled graph from the current
   var_vals. Returns the runtime state and a driver that runs the JIT with a
   fresh input node. *)
let make_kernel_jit ?(body = U.sink ~kernel_info:(kernel_info "jit_k") [])
    ?(binds = fun _ -> []) () =
  let state = runtime_state () in
  let dev = make_device ~state () in
  let registry : (int, Device.Buffer.t) Hashtbl.t = Hashtbl.create 8 in
  let buffers node = Hashtbl.find_opt registry (U.tag node) in
  let out_node = buffer_node () in
  let cp_out = U.param ~slot:0 ~dtype:Dtype.int32 () in
  let cp_in = U.param ~slot:1 ~dtype:Dtype.int32 () in
  let fxn input_uops var_vals =
    let body_call =
      U.call ~body ~args:[ cp_out; cp_in ] ~info:(call_info (Some "jit_k"))
    in
    let big =
      U.call ~body:(U.linear [ body_call ])
        ~args:(binds var_vals @ [ out_node; input_uops.(0) ])
        ~info:(call_info (Some "jit"))
    in
    let linear, vv =
      Schedule.create_linear_with_vars ~get_kernel_graph:Fun.id big
    in
    let binding = Realize.Buffers.create ~device:dev in
    Realize.run_linear ~device:dev ~to_program binding ~var_vals:vv linear;
    "ran"
  in
  let tjit = Jit.create ~device:dev ~to_program ~fxn () in
  let run ?(var_vals = []) ?(dtype = Dtype.int32) () =
    let node = buffer_node ~dtype () in
    let buf = Device.create_buffer ~size:4 ~dtype dev in
    Hashtbl.replace registry (U.tag node) buf;
    Jit.call tjit [| node |] var_vals
      ~held_buffers:(fun () -> [ out_node ])
      ~buffers
  in
  (state, run)

let raises_jit_error fn =
  raises_match (function Jit.Jit_error _ -> true | _ -> false) fn

let no_buffers _ = None

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
          test "empty capture raises and clears the capture registry"
            (fun () ->
              let t =
                Jit.create ~device ~to_program
                  ~fxn:(fun _ _ -> "ok")
                  ()
              in
              equal string "ok" (Jit.call t [||] [] ~buffers:no_buffers);
              raises_jit_error (fun () ->
                  ignore (Jit.call t [||] [] ~buffers:no_buffers));
              is_true ~msg:"capture registry cleared"
                (match !Realize.capturing with [] -> true | _ -> false));
        ];
      group "Capture and replay"
        [
          test "warmup, capture, and replay run the kernel" (fun () ->
            let state, run = make_kernel_jit () in
            equal string "ran" (run ());
            equal int 1 state.calls;
            equal string "ran" (run ());
            equal int 2 state.calls;
            equal string "ran" (run ());
            equal int 3 state.calls);
          test "replay validates input size dtype and device" (fun () ->
            let _state, run = make_kernel_jit () in
            ignore (run ());
            ignore (run ());
            raises_jit_error (fun () ->
                ignore (run ~dtype:Dtype.float32 ())));
          test "replay passes per-call var_vals to the runtime" (fun () ->
            let n = U.variable ~name:"n" ~min_val:1 ~max_val:16 () in
            let body =
              U.sink
                ~kernel_info:(kernel_info "jit_sym")
                [ U.special ~name:"gidx0" ~size:n () ]
            in
            let binds var_vals =
              [ U.bind ~var:n
                  ~value:(U.const_int (List.assoc "n" var_vals)) ]
            in
            let state, run = make_kernel_jit ~body ~binds () in
            ignore (run ~var_vals:[ ("n", 3) ] ());
            ignore (run ~var_vals:[ ("n", 3) ] ());
            equal (array int64) [| 3L |] state.vals;
            equal (array int) [| 3; 1; 1 |] state.global;
            ignore (run ~var_vals:[ ("n", 5) ] ());
            equal (array int64) [| 5L |] state.vals;
            equal (array int) [| 5; 1; 1 |] state.global;
            ignore (run ~var_vals:[ ("n", 9) ] ());
            equal (array int64) [| 9L |] state.vals;
            equal (array int) [| 9; 1; 1 |] state.global);
        ];
    ]
