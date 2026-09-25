(* Copyright (c) 2026 The Raven authors. ISC License. *)
open Windtrap
open Tolk
open Tolk_uop
module U = Uop
module Queue = Tolk_cuda__Cuda_queue

let device_name = "CUDA:queue-compilation"
let parameter slot = U.param ~slot ~dtype:Dtype.int32 ~shape:(U.const_int 16)
    ~device:(U.Single device_name) ()

let program_call () =
  let output = U.param ~slot:0 ~dtype:Dtype.int32 ~shape:(U.const_int 16) () in
  let small = U.variable ~param:true ~name:"small" ~min_val:(-128) ~max_val:127 ~dtype:Dtype.int8 () in
  let count = U.variable ~param:true ~name:"count" ~min_val:1 ~max_val:16 ~dtype:Dtype.int64 () in
  let value = U.cast ~src:small ~dtype:Dtype.int32 in
  let store = U.store ~dst:(U.index ~ptr:output ~idxs:[U.const_int 0] ()) ~value () in
  let spec = Program_spec.of_program ~name:"queue_fixture" ~src:"" ~device:device_name
      ~lib:(Bytes.of_string "not executed") [output; small; count; value; store] in
  let info = { (Program_spec.program_info spec) with global_size = [U.Launch_sym count]; local_size = [U.Launch_int 1] } in
  let program = U.program ~sink:(U.sink [store]) ~linear:(U.linear (Program_spec.program spec))
      ~source:(U.source "") ~binary:(U.binary "not executed") ~info () in
  U.call ~body:program ~args:[parameter 0]
      ~info:{grad_fxn = None; name = None; precompile = false;
        precompile_backward = false; dtype = Dtype.void; aux = None}

let compile ?(profile = false) ?peer_group calls =
  let host = Tolk_cpu.create "CPU" in
  let allocator = Device.Allocator.Pack (Storage.Host_allocator.make ~synchronize:(fun () -> ())) in
  let register queue_name =
    let renderer_set = Device.Renderer_set.make ~device:queue_name
        ["CLANG", (fun target -> Renderer.with_target target (Device.renderer host))] in
    let queue = Device.{timestamp_divider = 1000.; profile_offset = (fun () -> 0.); completion = (fun () () -> ()); prepare = (fun () -> ()); host = "CPU"; max_kernel_bindings = None; config = (fun () -> ""); copy = (fun _ -> Some "COPY:0");
      encode = Queue.encode queue_name; lower = Queue.lower queue_name;
      compile = Codegen.to_program ~optimize:false host (Device.renderer host)} in
    let peer_group = if queue_name = device_name then None else peer_group in
    ignore (Device.make ?peer_group ~name:queue_name ~allocator ~renderer_set ~runtime:(Device.runtime host)
      ~synchronize:(fun () -> ()) ~queue ()) in
  List.iter register [device_name; "CUDA:queue-peer"];
  Hcq2.compile ~to_program:(fun device -> Codegen.to_program device (Device.renderer device))
    ~profile (U.linear calls)

let submission linear = match U.as_call (U.without_after (List.hd (U.children linear))) with
  | Some call -> call.body
  | None -> fail "expected compiled host call"

let symbols root = U.toposort ~enter_calls:true root |> List.filter_map (fun u ->
    match U.as_param u with
    | Some {param = {allocation = Some ("cfunc", data); _}; _} ->
        let libs, symbol = (Marshal.from_string data 0 : string list * string) in
        equal (list string) [] libs;
        Some symbol
    | _ -> None) |> List.sort_uniq String.compare

let peer slot = U.param ~slot ~dtype:Dtype.int32 ~shape:(U.const_int 16)
    ~device:(U.Single "CUDA:queue-peer") ()

let () = run "CUDA queue compilation" [
  test "compatible peers share a submission with cross-device dependencies" (fun () ->
      let compiled = compile [U.store_call ~dst:(parameter 1) ~src:(parameter 0);
          U.store_call ~dst:(peer 2) ~src:(parameter 1);
          U.store_call ~dst:(peer 3) ~src:(peer 2)] in
      equal int 1 (List.length (U.children compiled));
      match U.arg (U.without_after (List.hd (U.children compiled))) with
      | U.Arg.Call_info {aux = Some info; _} ->
          equal (list string) [device_name; "CUDA:queue-peer"] info.devices;
          equal int 3 (List.length info.accesses);
          equal int 0 (List.length info.host_deps)
      | _ -> fail "peers were not batched");
  test "independent groups regroup without crossing ordinary calls" (fun () ->
      let first = U.store_call ~dst:(parameter 1) ~src:(parameter 0)
      and second = U.store_call ~dst:(peer 3) ~src:(peer 2) in
      let compiled = compile ~peer_group:"separate" [first; second; first] in
      equal int 2 (List.length (U.children compiled));
      let groups = List.map (fun call ->
          match U.arg (U.without_after call) with
          | U.Arg.Call_info {aux = Some info; _} -> info.devices
          | _ -> fail "expected a queue submission") (U.children compiled) in
      equal (list (list string)) [[device_name]; ["CUDA:queue-peer"]] groups;
      let ordinary = U.store_call
          ~dst:(U.param ~slot:4 ~dtype:Dtype.int32 ~shape:(U.const_int 16) ~device:(U.Single "CPU") ())
          ~src:(U.param ~slot:5 ~dtype:Dtype.int32 ~shape:(U.const_int 16) ~device:(U.Single "CPU") ()) in
      equal int 3 (List.length (U.children (compile [first; ordinary; second]))));
  test "compiles mixed-width arguments and symbolic launch dimensions" (fun () ->
      let compiled = compile [program_call ()] in
      equal (list string) ["tolk_cuda_hcq_begin"; "tolk_cuda_hcq_launch";
        "tolk_cuda_hcq_poll"; "tolk_cuda_hcq_signal"; "tolk_cuda_hcq_wait"] (symbols compiled);
      let object_ = U.to_elf (submission compiled) in
      is_true ~msg:"host compiler produced an executable object" (Bytes.length object_.lib > 0);
      let scalars = List.filter_map (fun (a : Tiny_elf.argument) ->
          if a.addrspace = Dtype.Alu then Some a.dtype else None) object_.signature in
      equal (list string) ["i64"; "i8"] (List.map Dtype.to_string scalars |> List.sort String.compare));
  test "peer timelines use each device's own context" (fun () ->
      let peer = U.param ~slot:1 ~dtype:Dtype.int32 ~shape:(U.const_int 16)
          ~device:(U.Single "CUDA:queue-peer") () in
      let compiled = compile [U.store_call ~dst:peer ~src:(parameter 0)] in
      let contexts = U.toposort ~enter_calls:true compiled |> List.filter_map (fun u ->
          match U.as_param u with
          | Some {param = {allocation = Some ("cuda_context", _); device = Some (U.Single d); _}; _} -> Some d
          | _ -> None) |> List.sort_uniq String.compare in
      equal (list string) [device_name; "CUDA:queue-peer"] contexts);
  test "host copies retain an ordinary execution fallback" (fun () ->
      let host = U.param ~slot:1 ~dtype:Dtype.int32 ~shape:(U.const_int 16)
          ~device:(U.Single "CPU") () in
      let compiled = compile [U.store_call ~dst:(parameter 0) ~src:host] in
      match U.arg (U.without_after (List.hd (U.children compiled))) with
      | U.Arg.Call_info {aux = Some info; _} ->
          equal int 1 (List.length info.fallback);
          equal (list (pair string string)) [("CPU", device_name)] info.host_deps;
          is_true (List.for_all (fun (_, d) -> d = device_name) info.inputs)
      | _ -> fail "host copy was not enqueued");
  test "profiles compute and copy calls with native host callbacks" (fun () ->
      let compiled = compile ~profile:true [program_call ();
          U.store_call ~dst:(parameter 2) ~src:(parameter 0)] in
      is_true (List.mem "tolk_cuda_hcq_timestamp" (symbols compiled));
      match U.arg (U.without_after (List.hd (U.children compiled))) with
      | U.Arg.Call_info {aux = Some info; _} -> equal int 2 (List.length info.timings)
      | _ -> fail "queue lost timestamp metadata");
  test "compiles dependencies crossing compute and copy queues" (fun () ->
      let compiled = compile [U.store_call ~dst:(parameter 0) ~src:(parameter 1);
        program_call (); U.store_call ~dst:(parameter 2) ~src:(parameter 0)] in
      is_true (List.mem "tolk_cuda_hcq_copy" (symbols compiled));
      equal int 1 (List.length (U.children compiled));
      match U.arg (U.without_after (List.hd (U.children compiled))) with
      | U.Arg.Call_info {aux = Some info; _} -> equal int 3 (List.length info.accesses)
      | _ -> fail "queue lost original dispatch access metadata");
]
