(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Contract tests for [Device.Buffer.copy_from], the canonical buffer-to-buffer
   move. The device layer keeps no executor of its own: the realize engine
   installs one at initialization through [install_copy_runner], and copy_from
   delegates to it. These tests exercise that installed delegation on CPU
   buffers. The complementary fail-loud half — copy_from raising before any
   runner is installed — lives in test_device_no_engine, which never links the
   engine; see the note at the bottom of this file for why the two halves
   cannot share one executable. *)

open Windtrap
open Tolk
open Tolk_uop
module D = Dtype

(* Referencing the realize engine forces its top-level installer to run in this
   executable, registering the canonical copy runner. Without a reference the
   linker would drop [Realize] and copy_from would stay unbacked. The installed
   runner resolves the destination device by name, so the CPU opener must be
   registered too. *)
let () = ignore (Sys.opaque_identity Realize.queue_submissions)
let () = Device.register "CPU" Tolk_cpu.create

let device = Device.get "CPU:device-test"

(* Helpers *)

let i32 = D.int32

let i32_to_bytes values =
  let b = Bytes.create (List.length values * 4) in
  List.iteri (fun i v -> Bytes.set_int32_le b (i * 4) (Int32.of_int v)) values;
  b

let read_i32 buf =
  let b = Device.Buffer.as_bytes buf in
  List.init (Bytes.length b / 4) (fun i ->
      Int32.to_int (Bytes.get_int32_le b (i * 4)))

let filled_i32 values =
  let buf =
    Device.create_buffer ~size:(List.length values) ~dtype:i32 device
  in
  Device.Buffer.ensure_allocated buf;
  Device.Buffer.copyin buf (i32_to_bytes values);
  buf

let empty_i32 n =
  let buf = Device.create_buffer ~size:n ~dtype:i32 device in
  Device.Buffer.ensure_allocated buf;
  buf

(* Tests *)

let copy_from_tests =
  group "Buffer.copy_from delegation"
    [
      test "moves bytes between same-size buffers" (fun () ->
          let src = filled_i32 [ 10; 20; 30; 40 ] in
          let dst = empty_i32 4 in
          Device.Buffer.copy_from ~dst ~src;
          equal (list int) [ 10; 20; 30; 40 ] (read_i32 dst));
      test "preserves dtype and size" (fun () ->
          let src = filled_i32 [ 7; 8 ] in
          let dst = empty_i32 2 in
          Device.Buffer.copy_from ~dst ~src;
          is_true ~msg:"dtype preserved"
            (Dtype.equal (Device.Buffer.dtype dst) (Device.Buffer.dtype src));
          equal int (Device.Buffer.size src) (Device.Buffer.size dst);
          equal (list int) [ 7; 8 ] (read_i32 dst));
      test "copies into an offset view" (fun () ->
          (* Write the source into the second half of a four-element buffer
             through a two-element view at byte offset 8, leaving the first
             half untouched — the view shares the base allocation. *)
          let src = filled_i32 [ 5; 6 ] in
          let dst = filled_i32 [ 1; 2; 3; 4 ] in
          let tail =
            Device.Buffer.view dst ~size:2 ~dtype:i32 ~offset:(2 * D.itemsize i32)
          in
          Device.Buffer.copy_from ~dst:tail ~src;
          equal (list int) [ 1; 2; 5; 6 ] (read_i32 dst));
    ]

(* The fail-loud half of the contract — copy_from raising [Invalid_argument]
   before any runner is installed — cannot be observed here. The delegation
   tests above require the realize engine, and linking it runs the installer at
   module-initialization time, before [main], so the uninstalled state is gone
   by the time any test runs. test_device_no_engine covers that half in a
   separate executable that never references the engine. *)

let failed_view_allocation_preserves_ownership () =
  let attempts = ref 0 and frees = ref 0 in
  let allocator = Device.Allocator.Pack {
      kind = Type.Id.make ();
      host = Fun.const None;
      mapping = None;
      synchronize = (fun () -> ());
      alloc = (fun _ _ -> ());
      free = (fun () _ _ -> incr frees);
      copyin = (fun () _ -> ()); copyout = (fun _ () -> ());
      addr = Some (fun () -> Nativeint.one);
      offset = Some (fun () _ _ ->
          incr attempts;
          if !attempts = 1 then failwith "offset failed");
      transfer = None; supports_transfer = false;
      copy_from_disk = None; supports_copy_from_disk = false;
    } in
  let base = Device.Buffer.create ~device:"VIEW_TEST" ~size:4 ~dtype:i32 allocator in
  let view = Device.Buffer.view base ~size:2 ~dtype:i32 ~offset:4 in
  raises (Failure "offset failed") (fun () -> Device.Buffer.allocate view);
  equal int 0 (Device.Buffer.allocated_views base);
  is_false (Device.Buffer.is_allocated view);
  Device.Buffer.ensure_allocated view;
  equal int 1 (Device.Buffer.allocated_views base);
  Device.Buffer.deallocate view;
  equal int 0 (Device.Buffer.allocated_views base);
  Device.Buffer.deallocate base;
  equal int 1 !frees


let empty_storage () =
  let unexpected op = fail ("empty storage called allocator " ^ op) in
  let allocator = Device.Allocator.Pack {
      kind = Type.Id.make ();
      host = Fun.const None;
      mapping = None;
      synchronize = (fun () -> ());
      alloc = (fun _ _ -> unexpected "alloc");
      free = (fun () _ _ -> unexpected "free");
      copyin = (fun () _ -> unexpected "copyin");
      copyout = (fun _ () -> unexpected "copyout");
      addr = Some (fun () -> unexpected "addr");
      offset = None; transfer = None; supports_transfer = false;
      copy_from_disk = None; supports_copy_from_disk = false;
    } in
  let create () = Device.Buffer.create ~device:"EMPTY" ~size:0 ~dtype:i32 allocator in
  let src = create () and dst = create () in
  is_false (Device.Buffer.is_allocated src);
  Device.Buffer.ensure_allocated src;
  Device.Buffer.ensure_allocated src;
  is_true (Device.Buffer.is_allocated src);
  equal nativeint 0n (Device.Buffer.addr src);
  Device.Buffer.copyin src Bytes.empty;
  equal string "" (Bytes.to_string (Device.Buffer.as_bytes src));
  is_true (Device.Buffer.transfer ~dst ~src);
  let view = Device.Buffer.view src ~size:0 ~dtype:i32 ~offset:0 in
  Device.Buffer.ensure_allocated view;
  equal int 0 (Device.Buffer.allocated_views src);
  Device.Buffer.deallocate src;
  equal nativeint 0n (Device.Buffer.addr view);
  let base = Device.Buffer.create ~device:"EMPTY" ~size:4 ~dtype:i32 allocator in
  let tail = Device.Buffer.view base ~size:0 ~dtype:i32 ~offset:(4 * D.itemsize i32) in
  Device.Buffer.ensure_allocated tail;
  is_false ~msg:"an empty view does not allocate its nonempty base"
    (Device.Buffer.is_allocated base);
  equal nativeint 0n (Device.Buffer.addr tail);
  List.iter Device.Buffer.deallocate [ tail; base; view; src; dst ]

let buffer_byte_ranges () =
  let allocator = Device.Buffer.allocator
      (Device.create_buffer ~size:0 ~dtype:i32 device) in
  let create size dtype = Device.Buffer.create ~device:"BOUNDS" ~size ~dtype allocator in
  let invalid_arg = function Invalid_argument _ -> true | _ -> false in
  raises_match invalid_arg (fun () -> create (-1) D.uint8);
  raises_match invalid_arg (fun () -> create ((max_int / 8) + 1) D.int64);
  let base = create max_int D.uint8 in
  raises_match invalid_arg (fun () -> Device.Buffer.view base ~size:(-1) ~dtype:D.uint8 ~offset:0);
  raises_match invalid_arg (fun () -> Device.Buffer.view base ~size:((max_int / 8) + 1) ~dtype:D.int64 ~offset:0);
  raises_match invalid_arg (fun () -> Device.Buffer.view base ~size:max_int ~dtype:D.uint8 ~offset:16);
  let tail = Device.Buffer.view base ~size:16 ~dtype:D.uint8 ~offset:(max_int - 16) in
  raises_match invalid_arg (fun () -> Device.Buffer.view tail ~size:16 ~dtype:D.uint8 ~offset:8);
  let nested = Device.Buffer.view tail ~size:8 ~dtype:D.uint8 ~offset:8 in
  equal int 8 (Device.Buffer.nbytes nested);
  let empty = Device.Buffer.view nested ~size:0 ~dtype:D.uint8 ~offset:8 in
  equal int 0 (Device.Buffer.nbytes empty);
  is_false (Device.Buffer.is_allocated base)

let interleaved_kernel_formals () =
  let module U = Uop in
  let allocator = Device.Buffer.allocator
      (Device.create_buffer ~size:0 ~dtype:i32 device) in
  let compiler = Compiler.make ~name:"SIGNATURE_TEST" ~compile:Bytes.of_string () in
  let renderer_set = Device.Renderer_set.make ~device:"SIGNATURE_TEST"
      [ "CUDA", (fun _ -> Renderer.with_compiler compiler (Cstyle.cuda Gpu_target.SM80)) ] in
  let dev = Device.make ~name:"SIGNATURE_TEST" ~allocator ~renderer_set
      ~runtime:(fun _ -> fail "source-only test loaded a binary")
      ~synchronize:(fun timeout -> ignore timeout; ()) () in
  let param slot = U.param ~slot ~dtype:i32 ~shape:(U.const_int 1) () in
  let output = param 7 and input = param 2 in
  let increment = U.variable ~param:true ~name:"increment" ~min_val:0 ~max_val:100 ~dtype:i32 () in
  let zero = U.const (Const.int i32 0) in
  let src = U.index ~ptr:input ~idxs:[ zero ] () in
  let dst = U.index ~ptr:output ~idxs:[ zero ] () in
  let loaded = U.load ~src () in
  let sum = U.alu_binary ~op:Ops.Add ~lhs:loaded ~rhs:increment in
  let linear = [ increment; output; zero; input; src; dst; loaded; sum;
                 U.store ~dst ~value:sum () ] in
  let spec = Device.compile_program dev ~name:"canonical_formals" linear in
  let prototype = String.split_on_char '\n' (Program_spec.src spec)
      |> List.find (String.starts_with ~prefix:"extern \"C\" __global__") in
  equal string
    "extern \"C\" __global__ void __launch_bounds__(1) canonical_formals(int* data7_1, int* data2_1, const int increment) {"
    prototype;
  equal (list int) [ 7; 2 ] (Program_spec.globals spec);
  let obj = Program_spec.to_elf spec in
  equal (list int) [ 0; 1; 2 ]
    (List.map (fun (arg : Tiny_elf.argument) -> arg.slot) obj.signature)

let node_owned_storage () =
  let node = Uop.buffer ~slot:(Uop.fresh_buffer_slot ()) ~dtype:i32
      ~shape:(Uop.const_int 4) ~device:(Uop.Single (Device.name device)) () in
  let a = Realize.resolve (Realize.exec_context ()) node in
  let b = Realize.resolve (Realize.exec_context ()) node in
  equal int (Device.Buffer.id a) (Device.Buffer.id b);
  is_false (Device.Buffer.is_allocated a);
  Device.Buffer.ensure_allocated a;
  Device.Buffer.copyin a (i32_to_bytes [1; 2; 3; 4]);
  Gc.full_major ();
  equal (list int) [1; 2; 3; 4] (read_i32 b);
  let imported = Uop.from_buffer a in
  is_true (imported == Uop.from_buffer a);
  equal int (Device.Buffer.id a)
    (Device.Buffer.id (Realize.resolve (Realize.exec_context ()) imported))

let storage_serialization () =
  let base = filled_i32 [10; 20; 30; 40] in
  let view = Device.Buffer.view base ~size:2 ~dtype:i32 ~offset:4 in
  Device.Buffer.ensure_allocated view;
  let graph = Uop.sink [Uop.from_buffer base; Uop.from_buffer view] in
  let restored = Uop.import (Uop.export graph) in
  equal string (Uop.semantic_key graph) (Uop.semantic_key restored);
  match Uop.children restored with
  | [base_node; view_node] ->
      let base' = Realize.resolve (Realize.exec_context ()) base_node in
      let view' = Realize.resolve (Realize.exec_context ()) view_node in
      equal (list int) [10; 20; 30; 40] (read_i32 base');
      equal (list int) [20; 30] (read_i32 view');
      equal int (Device.Buffer.id base') (Device.Buffer.base_id view');
      is_false (Device.Buffer.id base = Device.Buffer.id base');
      Device.Buffer.copyin view' (i32_to_bytes [7; 8]);
      equal (list int) [10; 7; 8; 40] (read_i32 base');
      equal (list int) [10; 20; 30; 40] (read_i32 base)
  | _ -> fail "serialized graph lost its two buffers"

let external_storage_serialization () =
  let owner = filled_i32 [6; 7] in
  let spec = { Device.Buffer_spec.default with
      external_ptr = Some (Device.Buffer.addr owner) } in
  let external_buffer = Device.create_buffer ~size:2 ~dtype:i32 ~spec device in
  Device.Buffer.ensure_allocated external_buffer;
  let node = Uop.from_buffer external_buffer in
  equal int (Device.Buffer.id external_buffer)
    (Device.Buffer.id (Realize.resolve (Realize.exec_context ()) node));
  let restored = Realize.resolve (Realize.exec_context ()) (Uop.import (Uop.export node)) in
  is_true (Option.is_none (Device.Buffer.spec restored).external_ptr);
  equal (list int) [6; 7] (read_i32 restored);
  Device.Buffer.copyin restored (i32_to_bytes [8; 9]);
  equal (list int) [6; 7] (read_i32 owner)

let lazy_storage_serialization () =
  let buf = Storage.on_device ~device:"UNOPENED" ~size:4 ~dtype:i32 () in
  let node = Uop.from_buffer buf in
  let restored = Uop.import (Uop.export node) in
  match Uop.Arg.as_param_arg (Uop.arg restored) with
  | Some { buffer = Some [buf']; _ } ->
      is_false (Storage.is_allocated buf');
      equal string "UNOPENED" (Storage.device buf');
      equal int 4 (Storage.size buf');
      is_false (Storage.id buf = Storage.id buf')
  | _ -> fail "serialized graph lost its lazy storage"

let typed_storage_identity () =
  let kind : bytes Type.Id.t = Type.Id.make () in
  let transfers = ref 0 in
  let allocator : bytes Device.Allocator.t = {
    kind;
    host = Fun.const None; mapping = None; synchronize = (fun () -> ());
    alloc = (fun size _ -> Bytes.make size '\000');
    free = (fun _ _ _ -> ());
    copyin = (fun dst src -> Bytes.blit src 0 dst 0 (Bytes.length src));
    copyout = (fun dst src -> Bytes.blit src 0 dst 0 (Bytes.length dst));
    addr = None;
    offset = None;
    transfer = Some (fun ~dest ~src ~dest_device ~src_device size ->
        equal string "OPAQUE" dest_device;
        equal string "OPAQUE" src_device;
        incr transfers; Bytes.blit src 0 dest 0 size; true);
    supports_transfer = true;
    copy_from_disk = None;
    supports_copy_from_disk = false;
  } in
  let create alloc = Device.Buffer.create ~device:"OPAQUE" ~size:1 ~dtype:i32
      (Device.Allocator.Pack alloc) in
  let src = create allocator and dst = create allocator in
  let incompatible = create { allocator with kind = Type.Id.make () } in
  raises_match (function Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Device.Buffer.get kind incompatible));
  is_false ~msg:"type rejection does not allocate" (Device.Buffer.is_allocated incompatible);
  is_false ~msg:"a matching device name does not prove representation equality"
    (Device.Buffer.transfer ~dst ~src:incompatible);
  equal int 0 !transfers;
  let raw = Option.get (Device.Buffer.get kind src) in
  Bytes.set_int32_le raw 0 42l;
  is_true (Device.Buffer.transfer ~dst ~src);
  equal int 1 !transfers;
  equal (list int) [42] (read_i32 dst);
  raises_match (function Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Device.Buffer.addr src));
  Device.Buffer.deallocate src

let mappings_follow_storage_ownership () =
  let source_kind : bytes Type.Id.t = Type.Id.make () in
  let target_kind : (bytes * int) Type.Id.t = Type.Id.make () in
  let events = ref [] in
  let record event = events := event :: !events in
  let source_allocator : bytes Device.Allocator.t = {
    kind = source_kind; host = Fun.const None; mapping = None;
    synchronize = (fun () -> record "source sync");
    alloc = (fun n _ -> Bytes.make n '\000');
    free = (fun _ _ _ -> record "source free");
    copyin = (fun dst src -> Bytes.blit src 0 dst 0 (Bytes.length src));
    copyout = (fun dst src -> Bytes.blit src 0 dst 0 (Bytes.length dst));
    addr = None; offset = Some (fun raw _ _ -> raw);
    transfer = Some (fun ~dest ~src ~dest_device ~src_device nbytes ->
        equal string "MAP_SOURCE" dest_device;
        equal string "MAP_SOURCE" src_device;
        record "native transfer";
        Bytes.blit src 0 dest 0 nbytes;
        true);
    supports_transfer = true;
    copy_from_disk = None; supports_copy_from_disk = false;
  } in
  let source = Device.Buffer.create ~device:"MAP_SOURCE" ~size:4 ~dtype:i32
      (Device.Allocator.Pack source_allocator) in
  let target name =
    let allocator : (bytes * int) Device.Allocator.t = {
      kind = target_kind; host = Fun.const None;
      mapping = Some {
        map = (fun source -> record (name ^ " map");
            match Device.Buffer.find_mapping target_kind source with
            | Some mapped -> record (name ^ " reuse"); mapped
            | None -> Option.get (Device.Buffer.get source_kind source), 0);
        unmap = (fun (_, offset) -> equal int 0 offset; record (name ^ " unmap"));
      };
      synchronize = (fun () -> record (name ^ " sync"));
      alloc = (fun _ _ -> fail "mapping must not allocate target storage");
      free = (fun _ _ _ -> fail "mapping must not free source through target");
      copyin = (fun (data, offset) src -> Bytes.blit src 0 data offset (Bytes.length src));
      copyout = (fun dst (data, offset) -> Bytes.blit data offset dst 0 (Bytes.length dst));
      addr = Some (fun (_, offset) -> Nativeint.of_int (0x1000 + offset));
      offset = Some (fun (raw, base) _ offset -> raw, base + offset);
      transfer = None; supports_transfer = false;
      copy_from_disk = None; supports_copy_from_disk = false;
    } in
    Device.make ~name ~allocator:(Device.Allocator.Pack allocator)
      ~renderer_set:(Device.Renderer_set.make ~device:name
          ["CLANG", (fun _ -> Device.renderer device)])
      ~runtime:(fun _ -> fail "mapping test does not execute code")
      ~synchronize:(fun timeout -> ignore timeout; allocator.synchronize ()) () in
  let first = target "MAP_TARGET:1" and second = target "MAP_TARGET:2" in
  is_true (Option.is_none (Device.Buffer.find_mapping target_kind source));
  let get device buf = Option.get (Device.Buffer.get ~device:(Device.name device) target_kind buf) in
  let data, offset = get first source in
  equal int 0 offset;
  Bytes.set_int32_le data 4 42l;
  let view = Device.Buffer.view source ~size:2 ~dtype:i32 ~offset:4 in
  let view_data, view_offset = get first view in
  is_true (data == view_data);
  equal int 4 view_offset;
  ignore (get first source);
  ignore (get second view);
  let cached_data, cached_offset = Option.get (Device.Buffer.find_mapping target_kind view) in
  is_true (cached_data == data);
  equal int 4 cached_offset;
  events := "address lookup" :: !events;
  equal nativeint 0x1004n (Device.Buffer.addr ~device:(Device.name second) view);
  equal string "address lookup" (List.hd !events);
  Device.Buffer.synchronize ~device:(Device.name second) view;
  equal string "source sync" (List.hd !events);
  let count event = List.length (List.filter (String.equal event) !events) in
  equal int 1 (count "MAP_TARGET:1 map");
  equal int 1 (count "MAP_TARGET:2 map");
  equal int 1 (count "MAP_TARGET:2 reuse");
  let syncs = count "MAP_TARGET:1 sync" in
  equal (list int) [0; 42; 0; 0] (read_i32 source);
  equal int (syncs + 1) (count "MAP_TARGET:1 sync");
  let destination = Device.Buffer.create ~device:"MAP_SOURCE" ~size:4 ~dtype:i32
      (Device.Allocator.Pack source_allocator) in
  let syncs = count "MAP_TARGET:1 sync" in
  is_true (Device.Buffer.transfer ~dst:destination ~src:source);
  equal int (syncs + 1) (count "MAP_TARGET:1 sync");
  equal string "native transfer" (List.hd !events);
  equal (list int) [0; 42; 0; 0] (read_i32 destination);
  Device.Buffer.deallocate destination;
  Device.Buffer.deallocate view;
  equal int 0 (count "MAP_TARGET:1 unmap");
  events := [];
  Device.Buffer.deallocate source;
  is_true (Option.is_none (Device.Buffer.find_mapping target_kind source));
  equal (list string)
    ["MAP_TARGET:2 sync"; "MAP_TARGET:2 unmap";
     "MAP_TARGET:1 sync"; "MAP_TARGET:1 unmap"; "source free"]
    (List.rev !events);
  ignore (get first source);
  equal int 1 (count "MAP_TARGET:1 map")

let foreign_completion_dependencies () =
  let owner = Tolk_cpu.create "CPU:pending-owner" in
  let submitted = ref 0 and captured = ref 0 and waited = ref [] in
  let timeouts = ref [] in
  let fail_wait = ref false and during_wait = ref (fun () -> ()) in
  let name = "PENDING:source" in
  let renderer_set = Device.Renderer_set.make ~device:name
      ["CLANG", (fun target -> Renderer.with_target target (Device.renderer owner))] in
  let completion () =
    incr captured;
    let value = !submitted in
    fun timeout ->
      timeouts := timeout :: !timeouts;
      waited := value :: !waited;
      !during_wait ();
      if !fail_wait then failwith "pending access failed" in
  let queue = Device.{timestamp_divider = 1.; profile_offset = (fun () -> 0.); completion; prepare = (fun () -> ());
    host = Device.name owner; max_kernel_bindings = None; config = (fun () -> ""); copy = (fun _ -> None); encode = (fun _ -> None);
    lower = (fun _ -> None); compile = (fun _ -> fail "not compiled")} in
  let allocator = Device.Allocator.Pack (Storage.Host_allocator.make ~synchronize:(fun () -> ())) in
  let source = Device.make ~name ~allocator ~renderer_set ~runtime:(Device.runtime owner)
      ~synchronize:(fun timeout -> ignore timeout; fail "must not drain later source work") ~queue () in
  equal string "PENDING" (Device.peer_group source);
  submitted := 3;
  Device.depend_on owner source;
  submitted := 5;
  Device.depend_on owner source;
  submitted := 9;
  equal int 2 !captured;
  equal (list int) [] !waited;
  Device.synchronize ~timeout:7 owner;
  equal (list (option int)) [Some 7] !timeouts;
  equal (list int) [5] !waited;
  Device.synchronize owner;
  equal (list int) [5] !waited;
  Device.depend_on owner owner;
  Device.depend_on owner source;
  fail_wait := true;
  raises (Failure "pending access failed") (fun () -> Device.synchronize owner);
  (* A newer access recorded during a failed wait must survive restoration. *)
  during_wait := (fun () -> submitted := 11; Device.depend_on owner source);
  raises (Failure "pending access failed") (fun () -> Device.synchronize owner);
  during_wait := (fun () -> ());
  fail_wait := false;
  Device.synchronize owner;
  equal (list int) [11; 9; 9; 5] !waited;
  Device.synchronize owner;
  equal (list int) [11; 9; 9; 5] !waited;
  submitted := 13;
  Device.depend_on owner source;
  Device.wait_dependencies owner ~ordered:[name];
  equal (list int) [11; 9; 9; 5] !waited;
  Device.wait_dependencies owner ~ordered:[];
  equal (list int) [13; 11; 9; 9; 5] !waited;
  Device.synchronize owner;
  equal (list int) [13; 11; 9; 9; 5] !waited;
  submitted := 17;
  Device.depend_on owner source;
  fail_wait := true;
  during_wait := (fun () -> submitted := 19; Device.depend_on owner source);
  raises (Failure "pending access failed")
    (fun () -> Device.wait_dependencies owner ~ordered:[]);
  during_wait := (fun () -> ());
  fail_wait := false;
  Device.wait_dependencies owner ~ordered:[name];
  equal (list int) [17; 13; 11; 9; 9; 5] !waited;
  Device.synchronize owner;
  equal (list int) [19; 17; 13; 11; 9; 9; 5] !waited

(* A backend callback reserves work before an allocation can run the GC.
   Releasing unrelated storage there must not synchronize an unsubmitted fence
   or re-enter its allocator. Exercise the same boundary for direct and host
   submission programs. *)
let finalizers_wait_for_device_operations () =
  let busy = ref false and releases = ref [] and probe = ref (fun () -> ()) in
  let host = Storage.Host_allocator.make ~synchronize:(fun () -> ()) in
  let allocator = Device.Allocator.Pack { host with
      alloc = (fun size spec -> !probe (); host.alloc size spec);
      free = (fun raw size spec ->
        releases := ("free", !busy) :: !releases;
        host.free raw size spec);
      copyin = (fun raw bytes -> !probe (); host.copyin raw bytes);
      copyout = (fun bytes raw -> !probe (); host.copyout bytes raw);
    } in
  let renderer_set = Device.Renderer_set.make ~device:"CPU"
      ["CLANG", (fun _ -> Device.renderer device)] in
  let mapping = Option.get host.mapping in
  let teardown_wait = ref (fun () -> ()) in
  let mapped = Device.Allocator.Pack {host with
      synchronize = (fun () ->
        releases := ("mapping wait", !busy) :: !releases;
        !teardown_wait ());
      mapping = Some {mapping with
        unmap = (fun raw -> releases := ("unmap", !busy) :: !releases;
          mapping.unmap raw)};
    } in
  ignore (Device.make ~name:"FINALIZER_MAP" ~allocator:mapped ~renderer_set
      ~runtime:(fun _ -> fail "mapping has no programs")
      ~synchronize:(fun timeout -> ignore timeout; ()) ());
  let abandon () =
    let buf = Device.Buffer.create ~device:"FINALIZER" ~size:1 ~dtype:D.uint8
        ~spec:{Device.Buffer_spec.default with nolru = true} allocator in
    Device.Buffer.ensure_allocated buf;
    let view = Device.Buffer.view buf ~size:1 ~dtype:D.uint8 ~offset:0 in
    Device.Buffer.ensure_allocated view;
    ignore (Device.Buffer.addr ~device:"FINALIZER_MAP" view) in
  let collect () =
    probe := (fun () -> ());
    abandon ();
    busy := true;
    Gc.full_major (); Gc.full_major ();
    busy := false in
  let queue = Device.{timestamp_divider = 1.;
      profile_offset = (fun () -> !probe (); 0.);
      completion = (fun () -> !probe (); fun timeout -> ignore timeout; !probe ());
      prepare = (fun () -> !probe ()); host = "CPU";
      copy = (fun _ -> None); encode = (fun _ -> None);
      lower = (fun _ -> None); compile = Fun.id} in
  let dev = Device.make ~name:"FINALIZER" ~allocator ~queue
      ~renderer_set
      ~runtime:(fun _ ->
        !probe ();
        {Device.call = (fun _ ~global:_ ~local:_ ~vals:_ ~wait:_ ~timeout:_ ->
            !probe (); None);
         free = (fun () -> !probe ()); handle = 0n})
      ~bufferize:(fun _ -> !probe (); None)
      ~invalidate_caches:(fun () -> !probe ())
      ~synchronize:(fun timeout -> ignore timeout; !probe ()) () in
  teardown_wait := (fun () -> Device.synchronize dev);
  let obj = Tiny_elf.{lib = Bytes.empty; name = "finalizer_probe";
      target = Renderer.target (Device.renderer device); signature = [];
      profile_key = None} in
  let direct = Device.runtime dev obj and submission = Device.queue_runtime dev obj in
  let run_program prg () =
    ignore (prg.Device.call [||] ~global:[|1;1;1|] ~local:None ~vals:[||]
      ~wait:false ~timeout:None) in
  let queue = Option.get (Device.queue dev) in
  let wait = queue.completion () in
  let wait_dependency () =
    (* The source callback runs while the owner's dependency lock is held. *)
    let source = Device.make ~name:"FINALIZER_SOURCE" ~allocator ~renderer_set
        ~runtime:(fun _ -> fail "source has no programs")
        ~synchronize:(fun timeout -> ignore timeout; ())
        ~queue:{queue with completion = (fun () -> fun timeout -> ignore timeout; !probe ())} () in
    Device.depend_on dev source;
    Device.wait_dependencies dev ~ordered:[] in
  let buffer = Device.Buffer.create ~device:"FINALIZER" ~size:1 ~dtype:D.uint8 allocator in
  List.iter (fun (name, run) ->
      releases := [];
      probe := collect;
      run ();
      equal ~msg:name (list (pair string bool))
        ["mapping wait", false; "unmap", false; "free", false]
        (List.rev !releases))
    ["allocation", (fun () -> Device.Buffer.ensure_allocated buffer);
     "upload", (fun () -> Device.Buffer.copyin buffer (Bytes.make 1 '\000'));
     "download", (fun () -> ignore (Device.Buffer.as_bytes buffer));
     "synchronize", (fun () -> Device.synchronize dev);
     "runtime creation", (fun () -> (Device.runtime dev obj).free ());
     "kernel", run_program direct;
     "host submission", run_program submission;
     "program release", direct.free;
     "queue preparation", queue.prepare;
     "queue clock", (fun () -> ignore (queue.profile_offset ()));
     "completion capture", (fun () -> (queue.completion ()) None);
     "completion wait", (fun () -> wait None);
     "foreign completion wait", wait_dependency;
     "command storage", (fun () -> ignore (Device.bufferize dev (Uop.const_int 0)));
     "cache invalidation", (fun () -> Device.invalidate_caches dev)];
  submission.free ();
  Device.Buffer.deallocate buffer;
  releases := [];
  probe := (fun () -> collect (); failwith "submission failed");
  raises (Failure "submission failed") (fun () -> Device.synchronize dev);
  equal ~msg:"failed operations retain pending storage" (list (pair string bool)) [] !releases;
  Device.synchronize dev;
  equal ~msg:"a successful wait releases pending storage" (list (pair string bool))
    ["mapping wait", false; "unmap", false; "free", false]
    (List.rev !releases)

(* A failed free may already have changed native state. It must surface once
   and retain the owner without retrying an uncertain teardown. *)
let failed_finalizer_is_not_retried () =
  let frees = ref 0 in
  let host = Storage.Host_allocator.make ~synchronize:(fun () -> ()) in
  let allocator = Device.Allocator.Pack {host with
      alloc = (fun _ _ -> 0n);
      free = (fun _ _ _ -> incr frees; failwith "teardown failed")} in
  let abandon () =
    let buf = Device.Buffer.create ~device:"FAILED_FINALIZER" ~size:1
        ~dtype:D.uint8 allocator in
    Device.Buffer.ensure_allocated buf in
  raises (Failure "teardown failed") (fun () ->
      Storage.with_operation (fun () ->
          abandon ();
          Gc.full_major (); Gc.full_major ();
          equal ~msg:"release waits for the operation" int 0 !frees));
  equal int 1 !frees;
  Storage.with_operation (fun () -> Gc.full_major (); Gc.full_major ());
  equal ~msg:"uncertain teardown is not retried" int 1 !frees

let () = run __FILE__ [ copy_from_tests;
  test "failed buffer finalizers are reported without retrying teardown" failed_finalizer_is_not_retried;
  test "buffer finalizers wait for device operations" finalizers_wait_for_device_operations;
  test "foreign access completion is captured, coalesced and retried" foreign_completion_dependencies;
  test "host storage owns zeroed pages suitable for GPU registration" (fun () ->
      List.iter (fun size ->
          let b = Device.create_buffer device ~size ~dtype:D.uint8
              ~spec:{Device.Buffer_spec.default with nolru = true} in
          let address = Option.get (Device.Buffer.host_addr b) in
          equal nativeint 0n (Nativeint.logand address 0xfffn);
          equal bytes (Bytes.make size '\000') (Device.Buffer.as_bytes b);
          Device.Buffer.deallocate b) [1; 4096; 4097]);
  test "per-device mappings share base ownership and release before storage" mappings_follow_storage_ownership;
  test "opaque storage dispatch and transfers require a type identity" typed_storage_identity;
  test "BUFFER owns storage across execution contexts" node_owned_storage;
  test "serialization preserves bytes and shared view ownership" storage_serialization;
  test "serialization copies external storage into an independent owner" external_storage_serialization;
  test "serialization keeps unopened storage lazy" lazy_storage_serialization; test "buffer byte ranges reject overflow" buffer_byte_ranges; test "compilation canonicalizes interleaved kernel arguments" interleaved_kernel_formals; test "empty storage never calls an allocator" empty_storage; test "failed view allocation preserves ownership" failed_view_allocation_preserves_ownership ]
