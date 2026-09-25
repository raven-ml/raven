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
  Device.Allocator.Pack
    (Storage.Host_allocator.make ~synchronize:(fun () -> ()))

let make_device ?(name = "TEST:0") ?(state = runtime_state ()) () =
  Device.make ~name ~allocator
    ~renderer_set:(Device.Renderer_set.make ~device:"TEST" [ "TEST", Fun.const renderer ])
    ~runtime:(fun _ ->
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
    ~synchronize:(fun timeout -> ignore timeout; ())
    ()

let device = make_device ()

let to_program device body =
  ignore device;
  let info = U.program_info_from_sink body in
  U.program ~sink:body ~linear:(U.linear (U.toposort body)) ~source:(U.source "")
    ~binary:(U.binary "") ~info ()

let shape_const n = U.const (Const.int Dtype.weakint n)

let buffer_node ?(size = 4) ?(dtype = Dtype.int32) () =
  U.buffer ~slot:(U.fresh_buffer_slot ()) ~dtype ~shape:(shape_const size)
    ~device:(U.Single "TEST:0") ()

let kernel_info name : U.kernel_info =
  {
    name;
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
    dtype = Dtype.void;
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
  let out_node = buffer_node () in
  let cp_out = U.param ~slot:0 ~dtype:Dtype.int32 () in
  let cp_in = U.param ~slot:1 ~dtype:Dtype.int32 () in
  let fxn input_uops var_vals =
    let body_call =
      U.call ~body ~args:[ cp_out; cp_in ] ~info:(call_info (Some (U.Label "jit_k")))
    in
    let big =
      U.call ~body:(U.linear [ body_call ])
        ~args:(binds var_vals @ [ out_node; input_uops.(0) ])
        ~info:(call_info (Some (U.Label "jit")))
    in
    let linear, vv =
      Schedule.create_linear_with_vars ~get_kernel_graph:Fun.id big
    in
    Realize.run_linear ~device:dev ~to_program ~var_vals:vv linear;
    "ran"
  in
  let tjit = Jit.create ~device:dev ~to_program ~fxn () in
  let run ?(var_vals = []) ?(dtype = Dtype.int32) () =
    let node = U.from_buffer (Device.create_buffer ~size:4 ~dtype dev) in
    Jit.call tjit [| node |] var_vals
      ~held_buffers:(fun () -> [ out_node ])
  in
  (state, run)

let raises_jit_error fn =
  raises_match (function Jit.Jit_error _ -> true | _ -> false) fn

let () =
  run "Engine_jit"
    [
      group "TinyJit"
        [
          test "create requires a function or a captured schedule" (fun () ->
            raises (Invalid_argument "need either a function or a CapturedJit")
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
              equal string "ok" (Jit.call t [||] []);
              raises_jit_error (fun () ->
                  ignore (Jit.call t [||] []));
              is_true ~msg:"capture scope exited"
                (Option.is_none (Realize.current_capture ())));
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
          test "capture and replay retain both signed int64 endpoints" (fun () ->
            let n = U.param ~slot:(-1) ~name:"wide" ~dtype:Dtype.int64
                ~addrspace:Dtype.Alu
                ~vmin_vmax:(Dtype.min Dtype.int64, Dtype.max Dtype.int64) () in
            let body = U.sink ~kernel_info:(kernel_info "jit_full_width") [n] in
            let variable = U.replace n ~op:Ops.Buffer () in
            let binds values = [U.bind ~var:variable
                ~value:(U.const (Const.int64 Dtype.int64 (List.assoc "wide" values)))] in
            let state, run = make_kernel_jit ~body ~binds () in
            List.iter (fun value ->
                ignore (run ~var_vals:["wide", value] ());
                equal (array int64) [|value|] state.vals)
              [Int64.min_int; Int64.max_int; Int64.min_int]);
          test "replay passes per-call var_vals to the runtime" (fun () ->
            let n = U.variable ~name:"n" ~min_val:1 ~max_val:16 () in
            let body =
              U.sink
                ~kernel_info:(kernel_info "jit_sym")
                [ U.special ~name:"gidx0" ~size:(U.replace n ~op:Ops.Param ()) () ]
            in
            let binds var_vals =
              [ U.bind ~var:n
                  ~value:(U.const (Const.int64 Dtype.weakint (List.assoc "n" var_vals))) ]
            in
            let state, run = make_kernel_jit ~body ~binds () in
            ignore (run ~var_vals:[ ("n", 3L) ] ());
            ignore (run ~var_vals:[ ("n", 3L) ] ());
            equal (array int64) [| 3L |] state.vals;
            equal (array int) [| 3; 1; 1 |] state.global;
            ignore (run ~var_vals:[ ("n", 5L) ] ());
            equal (array int64) [| 5L |] state.vals;
            equal (array int) [| 5; 1; 1 |] state.global;
            ignore (run ~var_vals:[ ("n", 9L) ] ());
            equal (array int64) [| 9L |] state.vals;
            equal (array int) [| 9; 1; 1 |] state.global);
        ];
    ]
