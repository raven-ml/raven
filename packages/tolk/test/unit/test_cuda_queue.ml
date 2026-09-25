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

let compile calls =
  let host = Tolk_cpu.create "CPU" in
  let allocator = Device.Allocator.Pack (Storage.Host_allocator.make ~synchronize:(fun () -> ())) in
  let renderer_set = Device.Renderer_set.make ~device:device_name
      ["CLANG", (fun target -> Renderer.with_target target (Device.renderer host))] in
  let queue = Device.{host = "CPU"; copy = (fun _ -> true);
    encode = Queue.encode device_name; lower = Queue.lower device_name;
    compile = Codegen.to_program ~optimize:false host (Device.renderer host)} in
  ignore (Device.make ~name:device_name ~allocator ~renderer_set ~runtime:(Device.runtime host)
    ~synchronize:(fun () -> ()) ~queue ());
  Hcq2.compile (U.linear calls)

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

let () = run "CUDA queue compilation" [
  test "compiles mixed-width arguments and symbolic launch dimensions" (fun () ->
      let compiled = compile [program_call ()] in
      equal (list string) ["tolk_cuda_hcq_begin"; "tolk_cuda_hcq_launch";
        "tolk_cuda_hcq_poll"; "tolk_cuda_hcq_signal"; "tolk_cuda_hcq_wait"] (symbols compiled);
      let object_ = U.to_elf (submission compiled) in
      is_true ~msg:"host compiler produced an executable object" (Bytes.length object_.lib > 0);
      let scalars = List.filter_map (fun (a : Tiny_elf.argument) ->
          if a.addrspace = Dtype.Alu then Some a.dtype else None) object_.signature in
      equal (list string) ["i64"; "i8"] (List.map Dtype.to_string scalars |> List.sort String.compare));
  test "compiles dependencies crossing compute and copy queues" (fun () ->
      let compiled = compile [U.store_call ~dst:(parameter 0) ~src:(parameter 1);
        program_call (); U.store_call ~dst:(parameter 2) ~src:(parameter 0)] in
      is_true (List.mem "tolk_cuda_hcq_copy" (symbols compiled));
      equal int 1 (List.length (U.children compiled));
      match U.arg (U.without_after (List.hd (U.children compiled))) with
      | U.Arg.Call_info {aux = Some info; _} -> equal int 3 (List.length info.accesses)
      | _ -> fail "queue lost original dispatch access metadata");
]
