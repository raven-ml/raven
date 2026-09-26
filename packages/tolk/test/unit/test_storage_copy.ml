(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk
open Tolk_uop
module U = Uop
module B = Device.Buffer

type raw = { address : nativeint; visible : bool; staging : bool }

let fixture suffix =
  let name = "COPYTEST:" ^ suffix in
  let cpu = Tolk_cpu.create "CPU" in
  let pending = ref [] and fail_wait = ref false in
  let staging_sizes = ref [] and staging_live = ref 0 and staging_peak = ref 0 in
  let staging_frees = ref 0 and submissions = ref 0 in
  let drain_pending () =
    if !pending <> [] && !fail_wait then failwith "pending copy did not complete";
    let jobs = List.rev !pending in
    pending := [];
    List.iter (fun run -> run ()) jobs in
  let host = Storage.Host_allocator.make ~synchronize:(fun () -> ()) in
  let allocator = Device.Allocator.Pack Device.Allocator.{
    kind = Type.Id.make ();
    alloc = (fun size spec ->
      let staging = spec.Device.Buffer_spec.host && spec.cpu_access && spec.nolru in
      let address = host.alloc size spec in
      if staging then begin
        staging_sizes := size :: !staging_sizes;
        staging_live := !staging_live + size;
        staging_peak := max !staging_peak !staging_live
      end;
      {address; visible = spec.host || spec.cpu_access; staging});
    free = (fun raw size spec ->
      drain_pending ();
      host.free raw.address size spec;
      if raw.staging then begin
        staging_live := !staging_live - size;
        incr staging_frees
      end);
    host = (fun raw -> if raw.visible then Some raw.address else None);
    addr = Some (fun raw -> raw.address);
    offset = Some (fun raw size offset ->
      ignore size;
      {raw with address = Nativeint.add raw.address (Nativeint.of_int offset)});
    mapping = None;
    synchronize = drain_pending;
  } in
  let host_name = "CPU:storage-copy-" ^ suffix in
  let runtime obj =
    let program = Device.queue_runtime cpu obj in
    let call buffers ~global ~local ~vals ~wait ~timeout =
      incr submissions;
      pending := (fun () ->
          ignore (program.call buffers ~global ~local ~vals ~wait:false ~timeout)) :: !pending;
      if wait then drain_pending ();
      None in
    {program with call} in
  let renderer_set = Device.Renderer_set.make ~device:"CPU"
      ["CLANG", (fun target -> Renderer.with_target target (Device.renderer cpu))] in
  ignore (Device.make ~name:host_name
    ~allocator:(Device.Allocator.Pack host) ~renderer_set ~runtime
    ~synchronize:(fun timeout -> ignore timeout; drain_pending ()) ());
  let timeline = B.create ~device:name ~size:2 ~dtype:Dtype.uint64
      ~spec:{Device.Buffer_spec.default with host = true; cpu_access = true} allocator in
  B.ensure_allocated timeline;
  let encode node = match U.op node, U.arg node, U.children node with
    | Ops.Custom_function, U.Arg.String "submit_copytest_copy_0", [linear; dependency] ->
        let previous = ref [dependency] in
        let commands = List.map (fun op ->
            let command = match U.as_call op, U.arg op with
              | Some {args = [dst; src]; _}, _ ->
                  Hcq2.ccall ~after:!previous ~name:"memcpy" ~dtype:Dtype.uint64
                    [U.getaddr ~device:name ~src:dst ();
                     U.getaddr ~device:name ~src ();
                     U.const (Const.int Dtype.uint64
                       (U.max_numel src * Dtype.itemsize (U.dtype src)))]
              | _, U.Arg.Typed ("store", _) ->
                  U.store ~dst:(U.index
                    ~ptr:(U.after ~src:(U.src op).(0) ~deps:!previous)
                    ~idxs:[U.const_int 0] ()) ~value:(U.src op).(1) ()
              | _, U.Arg.Typed (("wait" | "barrier" | "timestamp"), _) ->
                  U.noop ()
              | _ -> failwith "unexpected copy queue instruction" in
            if U.op command <> Ops.Noop then previous := [command];
            command) (U.children linear) in
        Some (U.group commands)
    | _ -> None in
  let queue = Device.{
    timestamp_divider = 1.; profile_offset = (fun () -> 0.);
    completion = (fun () -> fun timeout -> ignore timeout; drain_pending ());
    prepare = (fun () -> ()); host = host_name;
    max_kernel_bindings = None; config = (fun () -> "");
    copy = (fun _ -> Some "COPY:0"); encode; lower = (fun _ -> None);
    compile = Codegen.to_program ~optimize:false (Device.renderer cpu);
  } in
  let device = Device.make ~name ~allocator ~renderer_set ~queue
      ~synchronize:(fun timeout -> ignore timeout; drain_pending ())
      ~bufferize:(fun node ->
        if U.node_tag node = Some "timeline" then Some timeline else None) () in
  device, fail_wait, pending, staging_sizes, staging_live, staging_peak,
  staging_frees, submissions

let roundtrip offset () =
  let device, _, pending, sizes, live, peak, freed, submissions =
    fixture ("roundtrip-" ^ string_of_int offset) in
  let chunk = 64 lsl 20 in
  let size = chunk + 17 in
  let root = Device.create_buffer ~size:(size + 2 * offset) ~dtype:Dtype.uint8 device in
  let buffer = if offset = 0 then root else
      B.view root ~size ~dtype:Dtype.uint8 ~offset in
  let edges = if offset = 0 then [] else
      [B.view root ~size:1 ~dtype:Dtype.uint8 ~offset:0, Bytes.make 1 '\165';
       B.view root ~size:1 ~dtype:Dtype.uint8 ~offset:(size + 1), Bytes.make 1 '\090'] in
  B.ensure_allocated buffer;
  Fun.protect ~finally:(fun () ->
      List.iter (fun (edge, _) -> B.deallocate edge) edges;
      if offset <> 0 then B.deallocate buffer;
      B.deallocate root) (fun () ->
    List.iter (fun (edge, bytes) -> B.ensure_allocated edge; B.copyin edge bytes) edges;
    let edge_count = List.length edges in
    is_true ~msg:"user storage has no direct host access" (B.host_addr buffer = None);
    let input = Bytes.init size (fun i -> Char.chr ((i + i / chunk * 71) land 0xff)) in
    B.copyin buffer input;
    equal ~msg:"upload retires staging before returning" int 0 !live;
    equal int (edge_count + 1) !freed;
    let output = Bytes.make size '\000' in
    B.copyout buffer output;
    is_true ~msg:"every byte survives upload and download across the chunk boundary"
      (Bytes.equal input output);
    equal ~msg:"one bounded staging allocation per operation" (list int)
      ([chunk; chunk] @ List.map (fun _ -> 1) edges) !sizes;
    equal int chunk !peak;
    equal int 0 !live;
    equal int (edge_count + 2) !freed;
    equal ~msg:"each upload/download crosses the chunk boundary"
      int (edge_count + 4) !submissions;
    is_true ~msg:"host reads and reuse wait for submitted copies" (!pending = []);
    List.iter (fun (edge, expected) -> equal bytes expected (B.as_bytes edge)) edges)

let failed_wait_retains_staging () =
  let device, fail_wait, pending, _, live, _, freed, _ = fixture "failed-wait" in
  let buffer = Device.create_buffer ~size:64 ~dtype:Dtype.uint8 device in
  B.ensure_allocated buffer;
  let input = Bytes.init 64 Char.chr in
  Fun.protect
    ~finally:(fun () -> fail_wait := false; Device.synchronize device; B.deallocate buffer)
    (fun () ->
      fail_wait := true;
      raises (Failure "pending copy did not complete") (fun () ->
          Storage.with_operation (fun () ->
              raises (Failure "pending copy did not complete")
                (fun () -> B.copyin buffer input);
              Gc.full_major (); Gc.full_major ();
              equal ~msg:"the operation defers staging finalization" int 0 !freed));
      equal ~msg:"pending GPU access keeps staging storage live" int 64 !live;
      equal int 0 !freed;
      is_true (!pending <> []);
      fail_wait := false;
      Device.synchronize device;
      Storage.with_operation (fun () -> Gc.full_major (); Gc.full_major ());
      equal ~msg:"completion cannot authorize retrying failed automatic teardown"
        int 64 !live;
      equal int 0 !freed;
      equal bytes input (B.as_bytes buffer);
      equal ~msg:"a subsequent successful copy retires its own staging" int 1 !freed;
      equal int 64 !live)

let () = run __FILE__
    [test "non-host byte copies use bounded ordered staging" (roundtrip 0);
     test "non-host offset views preserve bytes outside staging chunks" (roundtrip 1);
     test "failed staging teardown reports errors and retains backing" failed_wait_retains_staging]
