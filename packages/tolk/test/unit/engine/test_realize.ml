(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop

module U = Uop

type runtime_state = {
  mutable runtimevars : (string * int) list;
  mutable vals : int64 array;
  mutable global : int array;
  mutable nbufs : int;
}

let runtime_state () =
  { runtimevars = []; vals = [||]; global = [||]; nbufs = -1 }

let test_renderer =
  Renderer.make ~name:"test" ~device:"TEST" ~has_local:false
    ~has_shared:false ~shared_max:0 ~render:(fun ?name:_ _ -> "") ()

type allocator_stats = {
  mutable copyin_calls : int;
  mutable copyout_calls : int;
  mutable transfer_calls : int;
  mutable synchronize_calls : int;
}

let allocator_stats () =
  { copyin_calls = 0; copyout_calls = 0; transfer_calls = 0;
    synchronize_calls = 0 }

let test_allocator ?(transfer = false) stats =
  let alloc nbytes spec =
    ignore spec;
    Bytes.make nbytes '\000'
  in
  let free buf nbytes spec =
    ignore buf;
    ignore nbytes;
    ignore spec
  in
  let copyin buf src =
    stats.copyin_calls <- stats.copyin_calls + 1;
    Bytes.blit src 0 buf 0 (Bytes.length src)
  in
  let copyout dst buf =
    stats.copyout_calls <- stats.copyout_calls + 1;
    Bytes.blit buf 0 dst 0 (Bytes.length dst)
  in
  let addr buf =
    ignore buf;
    Nativeint.zero
  in
  let offset buf nbytes byte_offset = Bytes.sub buf byte_offset nbytes in
  let transfer_fn =
    if transfer then
      Some
        (fun ~dest ~src nbytes ->
           stats.transfer_calls <- stats.transfer_calls + 1;
           Bytes.blit src 0 dest 0 nbytes)
    else None
  in
  Device.Allocator.Pack
    Device.Allocator.
      {
        alloc;
        free;
        copyin;
        copyout;
        addr;
        offset = Some offset;
        transfer = transfer_fn;
        supports_transfer = transfer;
        copy_from_disk = None;
        supports_copy_from_disk = false;
      }

let test_device ?(name = "TEST:0") ?(stats = allocator_stats ())
    ?(transfer = false) state =
  let renderer_set = Device.Renderer_set.make [ test_renderer, None ] in
  let runtime _name _lib ~runtimevars =
    state.runtimevars <- runtimevars;
    let call bufs ~global ~local:_ ~vals ~wait:_ ~timeout:_ =
      state.nbufs <- Array.length bufs;
      state.global <- Array.copy global;
      state.vals <- Array.copy vals;
      None
    in
    Device.{ call; free = (fun () -> ()); handle = 0n }
  in
  let synchronize () =
    stats.synchronize_calls <- stats.synchronize_calls + 1
  in
  Device.make ~name
    ~allocator:(test_allocator ~transfer stats)
    ~renderer_set ~runtime ~synchronize ()

let variable name lo hi =
  U.variable ~name ~min_val:lo ~max_val:hi ~dtype:Dtype.int32 ()

let shape_const n = U.const (Const.int Dtype.weakint n)

let buffer_node ?(slot = 0) ?(size = 4) ?(dtype = Dtype.int32)
    ?(device = "TEST:0") () =
  U.buffer ~slot ~dtype ~shape:(shape_const size) ~device:(U.Single device) ()

let spec_of program =
  Program_spec.of_program ~name:"kern" ~src:"" ~device:"TEST"
    ~lib:Bytes.empty program

(* Build the empty PROGRAM the exec path dispatches on from a kernel sink. *)
let program_of body =
  let info = U.program_info_from_sink body in
  U.program ~sink:body ~linear:(U.linear []) ~source:(U.source "")
    ~binary:(U.binary "") ~info ()

let call_info name : U.call_info =
  {
    grad_fxn = None;
    name;
    precompile = false;
    precompile_backward = false;
    aux = None;
  }

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

let call_runner state program var_vals =
  let device = test_device state in
  let runner = Realize.Compiled_runner.create ~device (spec_of program) in
  Realize.Compiled_runner.call runner [] var_vals ~wait:true ~timeout:None

let payload n =
  Bytes.init n (fun i -> Char.chr ((i * 17 + 3) land 0xff))

let create_buffer ?(name = "TEST:0") ?(transfer = false) stats =
  let state = runtime_state () in
  let device = test_device ~name ~stats ~transfer state in
  let buffer = Device.create_buffer ~size:4 ~dtype:Dtype.int32 device in
  device, buffer

let run_copy ?(dest_name = "TEST:0") ?(src_name = "TEST:1")
    ?(dest_transfer = false) ?(src_transfer = false) () =
  let dest_stats = allocator_stats () in
  let src_stats = allocator_stats () in
  let device, dest =
    create_buffer ~name:dest_name ~transfer:dest_transfer dest_stats
  in
  let src_device, src =
    create_buffer ~name:src_name ~transfer:src_transfer src_stats
  in
  ignore src_device;
  let data = payload (Device.Buffer.nbytes src) in
  Device.Buffer.ensure_allocated src;
  Device.Buffer.copyin src data;
  let runner =
    Realize.buffer_copy ~device
      ~total_sz:(Device.Buffer.nbytes dest)
      ~dest_device:(Device.Buffer.device dest)
      ~src_device:(Device.Buffer.device src)
  in
  ignore (Realize.Runner.call runner [ dest; src ] []
            ~wait:true ~timeout:None);
  dest, data, dest_stats, src_stats

let () =
  run "Engine_realize"
    [
      group "Compiled_runner"
        [
          test "passes vals and runtimevars from program metadata" (fun () ->
            let state = runtime_state () in
            let n = variable "n" 0 16 in
            let core_id = variable "core_id" 0 3 in
            ignore (call_runner state [ n; core_id ] [ "n", 7 ]);
            equal (list (pair string int)) [ "core_id", 0 ] state.runtimevars;
            equal (array int64) [| 0L; 7L |] state.vals;
            equal (array int) [| 4; 1; 1 |] state.global;
            equal int 0 state.nbufs);
          test "requires non-runtime scalar variables" (fun () ->
            let state = runtime_state () in
            let n = variable "n" 0 16 in
            raises_invalid_arg "missing variable \"n\"" (fun () ->
                ignore (call_runner state [ n ] [])));
        ];
      group "Program cache"
        [
          test "uses semantic key for tagged-equivalent ASTs" (fun () ->
            let device = test_device (runtime_state ()) in
            let ast =
              U.sink ~kernel_info:(kernel_info "semantic_cache_test") []
            in
            let tagged_ast = U.with_tag "diagnostic" ast in
            let calls = ref 0 in
            let to_program body =
              incr calls;
              program_of body
            in
            let call_of body = U.call ~body ~args:[] ~info:(call_info None) in
            ignore
              (Realize.pm_compile ~device ~to_program (U.linear [ call_of ast ]));
            ignore
              (Realize.pm_compile ~device ~to_program
                 (U.linear [ call_of tagged_ast ]));
            equal int 1 !calls);
          test "keys cached programs by exact device name" (fun () ->
            let ast =
              U.sink ~kernel_info:(kernel_info "device_cache_test") []
            in
            let calls = ref 0 in
            let to_program body =
              incr calls;
              program_of body
            in
            let call_of body = U.call ~body ~args:[] ~info:(call_info None) in
            let dev0 = test_device ~name:"TEST:0" (runtime_state ()) in
            let dev1 = test_device ~name:"TEST:1" (runtime_state ()) in
            ignore
              (Realize.pm_compile ~device:dev0 ~to_program
                 (U.linear [ call_of ast ]));
            ignore
              (Realize.pm_compile ~device:dev1 ~to_program
                 (U.linear [ call_of ast ]));
            equal int 2 !calls);
          test "rewrites CALL(SINK) to CALL(PROGRAM) with source and binary"
            (fun () ->
              let device = test_device (runtime_state ()) in
              let body = U.sink ~kernel_info:(kernel_info "structural_test") [] in
              let to_program body =
                let info = U.program_info_from_sink body in
                U.program ~sink:body ~linear:(U.linear [])
                  ~source:(U.source "SRC") ~binary:(U.binary "BIN") ~info ()
              in
              let call = U.call ~body ~args:[] ~info:(call_info None) in
              let compiled =
                Realize.pm_compile ~device ~to_program (U.linear [ call ])
              in
              match U.children compiled with
              | [ c ] -> (
                  match U.as_call c with
                  | Some { body = prog; _ } -> (
                      equal bool true (Ops.equal (U.op prog) Ops.Program);
                      match U.children prog with
                      | [ _sink; _lin; source; binary ] ->
                          equal (option string) (Some "SRC")
                            (U.Arg.as_string (U.arg source));
                          equal (option string) (Some "BIN")
                            (U.Arg.as_string (U.arg binary))
                      | _ -> fail "expected PROGRAM(SINK, LINEAR, SOURCE, BINARY)")
                  | None -> fail "expected CALL(PROGRAM)")
              | _ -> fail "expected a single compiled call");
        ];
      group "Buffer copy"
        [
          test "uses allocator transfer for same backend devices" (fun () ->
            let dest, data, dest_stats, src_stats =
              run_copy ~dest_transfer:true ()
            in
            equal string (Bytes.to_string data)
              (Bytes.to_string (Device.Buffer.as_bytes dest));
            equal int 1 dest_stats.transfer_calls;
            equal int 0 dest_stats.copyin_calls;
            equal int 0 src_stats.copyout_calls;
            equal int 1 dest_stats.synchronize_calls);
          test "falls back to host bounce without allocator transfer" (fun () ->
            let dest, data, dest_stats, src_stats = run_copy () in
            equal string (Bytes.to_string data)
              (Bytes.to_string (Device.Buffer.as_bytes dest));
            equal int 0 dest_stats.transfer_calls;
            equal int 1 dest_stats.copyin_calls;
            equal int 1 src_stats.copyout_calls);
          test "does not transfer across backend prefixes" (fun () ->
            let dest, data, dest_stats, src_stats =
              run_copy ~dest_transfer:true ~src_name:"OTHER:0" ()
            in
            equal string (Bytes.to_string data)
              (Bytes.to_string (Device.Buffer.as_bytes dest));
            equal int 0 dest_stats.transfer_calls;
            equal int 1 dest_stats.copyin_calls;
            equal int 1 src_stats.copyout_calls);
          test "rejects size or dtype mismatches before copy" (fun () ->
            let state = runtime_state () in
            let stats = allocator_stats () in
            let device = test_device ~stats state in
            let dest =
              Device.create_buffer ~size:4 ~dtype:Dtype.int32 device
            in
            let src =
              Device.create_buffer ~size:8 ~dtype:Dtype.int32 device
            in
            let runner =
              Realize.buffer_copy ~device ~total_sz:16
                ~dest_device:(Device.Buffer.device dest)
                ~src_device:(Device.Buffer.device src)
            in
            raises_invalid_arg "buffer copy: size or dtype mismatch"
              (fun () ->
                 ignore (Realize.Runner.call runner [ dest; src ] []
                           ~wait:false ~timeout:None)));
        ];
      group "Buffer binding"
        [
          test "allocates and caches a buffer per BUFFER node" (fun () ->
            let device = test_device (runtime_state ()) in
            let binding = Realize.Buffers.create ~device in
            let node = buffer_node ~size:4 () in
            let b1 = Realize.Buffers.of_buffer_node binding node in
            let b2 = Realize.Buffers.of_buffer_node binding node in
            equal int 4 (Device.Buffer.size b1);
            equal bool true (Dtype.equal Dtype.int32 (Device.Buffer.dtype b1));
            equal int (Device.Buffer.id b1) (Device.Buffer.id b2));
          test "seed overrides lazy allocation" (fun () ->
            let device = test_device (runtime_state ()) in
            let binding = Realize.Buffers.create ~device in
            let node = buffer_node ~size:4 () in
            let seeded =
              Device.create_buffer ~size:4 ~dtype:Dtype.int32 device
            in
            Realize.Buffers.seed binding node seeded;
            let got = Realize.Buffers.of_buffer_node binding node in
            equal int (Device.Buffer.id seeded) (Device.Buffer.id got));
          test "resolves PARAM through input_uops" (fun () ->
            let device = test_device (runtime_state ()) in
            let binding = Realize.Buffers.create ~device in
            let input = buffer_node ~slot:0 ~size:4 () in
            let seeded =
              Device.create_buffer ~size:4 ~dtype:Dtype.int32 device
            in
            Realize.Buffers.seed binding input seeded;
            let param =
              U.param ~slot:0 ~dtype:Dtype.int32 ~shape:(shape_const 4)
                ~device:(U.Single "TEST:0") ()
            in
            let ctx = Realize.exec_context ~input_uops:[| input |] () in
            let got = Realize.resolve binding ctx param in
            equal int (Device.Buffer.id seeded) (Device.Buffer.id got));
          test "resolves SLICE as an offset view" (fun () ->
            let device = test_device (runtime_state ()) in
            let binding = Realize.Buffers.create ~device in
            let base_node = buffer_node ~size:4 () in
            let slice =
              U.slice ~src:base_node ~offset:(shape_const 1) ~size:2
                ~dtype:Dtype.int32
            in
            let ctx = Realize.exec_context () in
            let view = Realize.resolve binding ctx slice in
            equal int 2 (Device.Buffer.size view);
            equal int 4 (Device.Buffer.offset view));
          test "rejects an unbound PARAM" (fun () ->
            let device = test_device (runtime_state ()) in
            let binding = Realize.Buffers.create ~device in
            let param = U.param ~slot:5 ~dtype:Dtype.int32 () in
            let ctx = Realize.exec_context () in
            raises_match
              (function Invalid_argument _ -> true | _ -> false)
              (fun () -> ignore (Realize.resolve binding ctx param)));
        ];
      group "Linear execution"
        [
          test "runs a kernel call with resolved buffers" (fun () ->
            let state = runtime_state () in
            let device = test_device state in
            let body = U.sink ~kernel_info:(kernel_info "rl_kernel") [] in
            let info : U.call_info =
              {
                grad_fxn = None;
                name = Some "rl_kernel";
                precompile = false;
                precompile_backward = false;
                aux = None;
              }
            in
            let out = buffer_node ~slot:0 () in
            let inp = buffer_node ~slot:1 () in
            let call = U.call ~body ~args:[ out; inp ] ~info in
            let binding = Realize.Buffers.create ~device in
            Realize.run_linear ~device
              ~to_program:program_of
              binding
              (U.linear [ call ]);
            equal int 2 state.nbufs);
          test "resolves PARAM kernel args from input_uops" (fun () ->
            let state = runtime_state () in
            let device = test_device state in
            let body = U.sink ~kernel_info:(kernel_info "rl_param") [] in
            let info : U.call_info =
              {
                grad_fxn = None;
                name = Some "rl_param";
                precompile = false;
                precompile_backward = false;
                aux = None;
              }
            in
            let input = buffer_node ~slot:0 () in
            let param =
              U.param ~slot:0 ~dtype:Dtype.int32 ~shape:(shape_const 4)
                ~device:(U.Single "TEST:0") ()
            in
            let call = U.call ~body ~args:[ param ] ~info in
            let binding = Realize.Buffers.create ~device in
            Realize.run_linear ~device
              ~to_program:program_of
              binding ~input_uops:[| input |]
              (U.linear [ call ]);
            equal int 1 state.nbufs);
          test "binds an offset view for a SLICE call" (fun () ->
            let device = test_device (runtime_state ()) in
            let binding = Realize.Buffers.create ~device in
            let src_node = buffer_node ~slot:0 ~size:8 () in
            let out_node = buffer_node ~slot:1 ~size:2 () in
            let slice_body =
              U.slice ~src:src_node ~offset:(shape_const 4) ~size:2
                ~dtype:Dtype.int32
            in
            let info : U.call_info =
              {
                grad_fxn = None;
                name = None;
                precompile = false;
                precompile_backward = false;
                aux = None;
              }
            in
            let call = U.call ~body:slice_body ~args:[ out_node; src_node ] ~info in
            Realize.run_linear ~device
              ~to_program:program_of
              binding
              (U.linear [ call ]);
            let v =
              match Realize.Buffers.find_opt binding out_node with
              | Some v -> v
              | None -> failwith "expected a view binding"
            in
            equal int 2 (Device.Buffer.size v);
            equal int 16 (Device.Buffer.offset v));
        ];
    ]
