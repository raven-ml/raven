(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Contract tests for [Device.Buffer.copy_from], the canonical buffer-to-buffer
   move. The device layer keeps no executor of its own: the code generator
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

(* Referencing the code generator forces its top-level installer to run in this
   executable, registering the canonical copy runner. Without a reference the
   linker would drop [Codegen] and copy_from would stay unbacked. The installed
   runner resolves the destination device by name, so the CPU opener must be
   registered too. *)
let () = ignore (Sys.opaque_identity (Some Codegen.to_program))
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
   tests above require the code generator, and linking it runs the installer at
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
      addr = Some (fun () -> Nativeint.one);
      offset = Some (fun () _ _ ->
          incr attempts;
          if !attempts = 1 || !attempts = 3 then failwith "offset failed");
    } in
  let base = Device.Buffer.create ~device:"VIEW_TEST" ~size:4 ~dtype:i32 allocator in
  let view = Device.Buffer.view base ~size:2 ~dtype:i32 ~offset:4 in
  raises (Failure "offset failed") (fun () -> Device.Buffer.allocate view);
  equal int 0 (Device.Buffer.allocated_views base);
  is_false (Device.Buffer.is_allocated view);
  Device.Buffer.ensure_allocated view;
  equal int 1 (Device.Buffer.allocated_views base);
  Device.Buffer.deallocate base;
  is_false (Device.Buffer.is_allocated view);
  equal int 1 (Device.Buffer.allocated_views base);
  (* The allocator returns the same raw value, but the allocation is new. *)
  Device.Buffer.ensure_allocated base;
  is_false (Device.Buffer.is_allocated view);
  raises (Failure "offset failed") (fun () -> Device.Buffer.ensure_allocated view);
  is_false (Device.Buffer.is_allocated view);
  equal int 1 (Device.Buffer.allocated_views base);
  Device.Buffer.ensure_allocated view;
  is_true (Device.Buffer.is_allocated view);
  equal int 1 (Device.Buffer.allocated_views base);
  Device.Buffer.deallocate view;
  equal int 0 (Device.Buffer.allocated_views base);
  Device.Buffer.deallocate base;
  equal int 2 !frees

let stale_views_refresh_on_access () =
  let base = filled_i32 [1; 2; 3; 4] in
  let view = Device.Buffer.view base ~size:2 ~dtype:i32 ~offset:4 in
  raises (Invalid_argument "buffer is not allocated")
    (fun () -> Device.Buffer.copyin view (i32_to_bytes [7; 8]));
  raises (Invalid_argument "buffer is not allocated")
    (fun () -> Device.Buffer.copyout view (Bytes.create 8));
  raises (Invalid_argument "buffer is not allocated")
    (fun () -> Device.Buffer.as_buffer view);
  Device.Buffer.ensure_allocated view;
  let replace values =
    Device.Buffer.deallocate base;
    is_false (Device.Buffer.is_allocated view);
    Device.Buffer.ensure_allocated base;
    Device.Buffer.copyin base (i32_to_bytes values)
  in
  replace [10; 20; 30; 40];
  equal (list int) [20; 30] (read_i32 view);
  replace [11; 21; 31; 41];
  Device.Buffer.copyin view (i32_to_bytes [7; 8]);
  equal (list int) [11; 7; 8; 41] (read_i32 base);
  replace [12; 22; 32; 42];
  let bytes = Option.get (Device.Buffer.as_buffer view) in
  equal int 22 (Bigarray.Array1.get bytes 0);
  equal int 32 (Bigarray.Array1.get bytes 4);
  replace [13; 23; 33; 43];
  Device.Buffer.ensure_allocated view;
  equal int 1 (Device.Buffer.allocated_views base);
  let destination = empty_i32 2 in
  Device.Buffer.copy_from ~dst:destination ~src:view;
  equal (list int) [23; 33] (read_i32 destination);
  (* The shared executor resolves another temporary view. Its collection must
     not be required to release the base, and it must not remain retained. *)
  Device.Buffer.deallocate base;
  is_false (Device.Buffer.is_allocated base);
  is_false (Device.Buffer.is_allocated view);
  Device.Buffer.deallocate view;
  Gc.full_major ();
  Storage.with_operation (fun () -> ());
  equal int 0 (Device.Buffer.allocated_views base);
  Device.Buffer.deallocate destination

let external_views_refresh_without_freeing_owner () =
  let owner = filled_i32 [1; 2; 3; 4] in
  let address = Device.Buffer.addr owner in
  let spec = {Device.Buffer_spec.default with external_ptr = Some address} in
  let external_buffer = Device.create_buffer ~size:4 ~dtype:i32 ~spec device in
  let view = Device.Buffer.view external_buffer ~size:2 ~dtype:i32 ~offset:4 in
  Device.Buffer.ensure_allocated view;
  Device.Buffer.deallocate external_buffer;
  Device.Buffer.copyin owner (i32_to_bytes [10; 20; 30; 40]);
  equal nativeint (Nativeint.add address 4n) (Device.Buffer.addr view);
  equal (list int) [20; 30] (read_i32 view);
  equal int 1 (Device.Buffer.allocated_views external_buffer);
  Device.Buffer.deallocate external_buffer;
  Device.Buffer.deallocate view;
  equal (list int) [10; 20; 30; 40] (read_i32 owner);
  Device.Buffer.deallocate owner


let empty_storage () =
  let unexpected op = fail ("empty storage called allocator " ^ op) in
  let allocator = Device.Allocator.Pack {
      kind = Type.Id.make ();
      host = Fun.const None;
      mapping = None;
      synchronize = (fun () -> ());
      alloc = (fun _ _ -> unexpected "alloc");
      free = (fun () _ _ -> unexpected "free");
      addr = Some (fun () -> unexpected "addr");
      offset = None;
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
  Device.Buffer.ensure_allocated dst;
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

let device_initialization_registration () =
  let allocator = Device.Buffer.allocator
      (Device.create_buffer ~size:0 ~dtype:i32 device) in
  let create ?initialize name =
    Device.make ~name ~allocator
      ~renderer_set:(Device.Renderer_set.make ~device:name [])
      ~synchronize:(fun timeout -> ignore timeout) ?initialize () in
  let attempts = ref 0 in
  Device.register "BOOTSTRAP_TEST" (fun name ->
      incr attempts;
      create name ~initialize:(fun current ->
          is_true ~msg:"bootstrap can resolve its own device"
            (Device.get "bootstrap_test:0" == current);
          if !attempts = 1 then failwith "bootstrap failed"));
  raises (Failure "bootstrap failed") (fun () ->
      ignore (Device.get "BOOTSTRAP_TEST"));
  let ready = Device.get "BOOTSTRAP_TEST" in
  equal ~msg:"failed bootstrap must reopen the device" int 2 !attempts;
  is_true (Device.get "BOOTSTRAP_TEST" == ready);
  raises (Failure "replacement failed") (fun () ->
      ignore (create "bootstrap_test:0" ~initialize:(fun current ->
          is_true (Device.get "BOOTSTRAP_TEST" == current);
          failwith "replacement failed")));
  is_true ~msg:"a failed replacement preserves the previous device"
    (Device.get "BOOTSTRAP_TEST" == ready);
  let replacement = ref None in
  raises (Failure "superseded bootstrap failed") (fun () ->
      ignore (create "BOOTSTRAP_TEST" ~initialize:(fun current ->
          ignore current;
          replacement := Some (create "BOOTSTRAP_TEST");
          failwith "superseded bootstrap failed")));
  is_true ~msg:"rollback must not remove a subsequently registered device"
    (Device.get "BOOTSTRAP_TEST" == Option.get !replacement);
  Device.register "SUPERSEDED_OPENER" (fun name ->
      let earlier = create name in
      replacement := Some (create name);
      earlier);
  let latest = Device.get "SUPERSEDED_OPENER" in
  is_true ~msg:"an opener cannot republish a superseded device"
    (latest == Option.get !replacement)

let registry_gate () =
  let mutex = Mutex.create () and changed = Condition.create () in
  let released = ref false in
  let wait () = Mutex.protect mutex (fun () ->
      while not !released do Condition.wait changed mutex done) in
  let release () = Mutex.protect mutex (fun () ->
      released := true; Condition.broadcast changed) in
  wait, release

let registry_worker ~systhread run =
  let capture () = try Ok (run ()) with exn -> Error (exn, Printexc.get_raw_backtrace ()) in
  let joined =
    if systhread then begin
      let result = ref None in
      let worker = Thread.create (fun () -> result := Some (capture ())) () in
      fun () -> Thread.join worker; Option.get !result
    end else begin
      let worker = Domain.spawn capture in
      fun () -> Domain.join worker
    end in
  fun () -> match joined () with
    | Ok value -> value
    | Error (exn, bt) -> Printexc.raise_with_backtrace exn bt

let registry_factory () =
  let allocator = Device.Buffer.allocator
      (Device.create_buffer ~size:0 ~dtype:i32 device) in
  fun ?initialize name -> Device.make ~name ~allocator
    ~renderer_set:(Device.Renderer_set.make ~device:name [])
    ~synchronize:(fun timeout -> ignore timeout) ?initialize ()

let concurrent_device_opening () =
  let make = registry_factory () in
  let wait, release = registry_gate () in
  let started, announce = registry_gate () in
  let attempts = Atomic.make 0 in
  Device.register "OPEN_CONCURRENT" (fun name ->
      ignore (Atomic.fetch_and_add attempts 1);
      announce (); wait (); make name);
  let first = registry_worker ~systhread:false (fun () -> Device.get "OPEN_CONCURRENT") in
  started ();
  let followers = List.init 4 (fun i -> registry_worker ~systhread:(i mod 2 = 0)
      (fun () -> Device.get "open_concurrent:0")) in
  (* Keep the opener blocked while both systhreads and domains enter get. *)
  Thread.delay 0.02;
  release ();
  let devices = first () :: List.map (fun join -> join ()) followers in
  equal ~msg:"one opener owns the canonical name" int 1 (Atomic.get attempts);
  List.iter (fun current -> is_true (current == List.hd devices)) devices

let incomplete_device_is_private () =
  let make = registry_factory () in
  let wait, release = registry_gate () in
  let started, announce = registry_gate () in
  let initialized = Atomic.make false in
  let first = registry_worker ~systhread:false (fun () ->
      make "INIT_CONCURRENT" ~initialize:(fun current ->
          is_true (Device.get "init_concurrent:0" == current);
          announce (); wait (); Atomic.set initialized true)) in
  started ();
  let followers = List.init 4 (fun i -> registry_worker ~systhread:(i mod 2 = 0)
      (fun () -> let current = Device.get "INIT_CONCURRENT" in
        current, Atomic.get initialized)) in
  Thread.delay 0.02;
  (* An unrelated name can still initialize while this callback is blocked. *)
  let independent = make "INIT_INDEPENDENT" in
  is_true (Device.get "INIT_INDEPENDENT" == independent);
  release ();
  let ready = first () in
  List.iter (fun join ->
      let current, completed = join () in
      is_true ~msg:"get waits for initialization" completed;
      is_true (current == ready)) followers

let failed_initialization_wakes_waiters () =
  let make = registry_factory () in
  let wait, release = registry_gate () in
  let started, announce = registry_gate () in
  let attempts = Atomic.make 0 in
  Device.register "RETRY_CONCURRENT" (fun name ->
      let attempt = Atomic.fetch_and_add attempts 1 in
      make name ~initialize:(fun current ->
          is_true (Device.get name == current);
          if attempt = 0 then begin
            announce (); wait (); failwith "concurrent bootstrap failed"
          end));
  let first = registry_worker ~systhread:false (fun () ->
      try ignore (Device.get "RETRY_CONCURRENT"); false with
      | Failure message when message = "concurrent bootstrap failed" -> true) in
  started ();
  let follower = registry_worker ~systhread:true (fun () -> Device.get "RETRY_CONCURRENT") in
  Thread.delay 0.02;
  release ();
  let failed = first () in
  let ready = follower () in
  is_true ~msg:"the initiating caller receives its original failure" failed;
  equal ~msg:"a waiting caller retries after rollback" int 2 (Atomic.get attempts);
  is_true (Device.get "RETRY_CONCURRENT" == ready)

let failed_opener_does_not_publish () =
  let make = registry_factory () in
  List.iter (fun early_failure ->
      let name = if early_failure then "POST_MAKE_RETRY" else "POST_MAKE_FAILURE" in
      let attempts = ref 0 in
      Device.register name (fun name ->
          incr attempts;
          if early_failure && !attempts = 1 then
            (try ignore (make name ~initialize:(fun current ->
                 ignore current; failwith "early initialization failed")) with
             | Failure message when message = "early initialization failed" -> ());
          let current = make name in
          if !attempts = 1 then failwith "opener failed after make";
          current);
      raises (Failure "opener failed after make") (fun () -> ignore (Device.get name));
      let ready = Device.get name in
      equal ~msg:"failed opener must retry instead of publishing its partial device" int 2 !attempts;
      is_true (Device.get name == ready)) [false; true];
  let replacement = ref None in
  Device.register "POST_MAKE_SUPERSEDED" (fun name ->
      ignore (make name);
      replacement := Some (make name);
      failwith "superseded opener failed");
  raises (Failure "superseded opener failed") (fun () ->
      ignore (Device.get "POST_MAKE_SUPERSEDED"));
  is_true ~msg:"a failed opener preserves the newer nested registration"
    (Device.get "POST_MAKE_SUPERSEDED" == Option.get !replacement)

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
  let allocator : bytes Device.Allocator.t = {
    kind; host = Fun.const None; mapping = None; synchronize = (fun () -> ());
    alloc = (fun size _ -> Bytes.make size '\000');
    free = (fun _ _ _ -> ()); addr = None; offset = None;
  } in
  let create alloc = Device.Buffer.create ~device:"OPAQUE" ~size:1 ~dtype:i32
      (Device.Allocator.Pack alloc) in
  let src = create allocator in
  let incompatible = create {allocator with kind = Type.Id.make ()} in
  raises_match (function Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Device.Buffer.get kind incompatible));
  is_false ~msg:"type rejection does not allocate" (Device.Buffer.is_allocated incompatible);
  let raw = Option.get (Device.Buffer.get kind src) in
  Bytes.set_int32_le raw 0 42l;
  equal int32 42l (Bytes.get_int32_le (Option.get (Device.Buffer.get kind src)) 0);
  raises_match (function Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (Device.Buffer.addr src));
  Device.Buffer.deallocate src

let mappings_follow_storage_ownership () =
  let source_kind : nativeint Type.Id.t = Type.Id.make () in
  let target_kind : (nativeint * int) Type.Id.t = Type.Id.make () in
  let fail_source_free = ref false and fail_first_sync = ref false in
  let events = ref [] in
  let record event = events := event :: !events in
  let host = Storage.Host_allocator.make ~synchronize:(fun () -> ()) in
  let source_allocator = {host with
    kind = source_kind;
    synchronize = (fun () -> record "source sync");
    free = (fun raw size spec ->
        record "source free";
        if !fail_source_free then begin
          fail_source_free := false;
          failwith "source free failed"
        end;
        host.free raw size spec);
  } in
  let source = Device.Buffer.create ~device:"MAP_SOURCE" ~size:4 ~dtype:i32
      (Device.Allocator.Pack source_allocator) in
  let target name =
    let allocator : (nativeint * int) Device.Allocator.t = {
      kind = target_kind; host = Fun.const None;
      mapping = Some {
        map = (fun source -> record (name ^ " map");
            match Device.Buffer.find_mapping target_kind source with
            | Some mapped -> record (name ^ " reuse"); mapped
            | None -> Option.get (Device.Buffer.get source_kind source), 0);
        unmap = (fun (_, offset) -> equal int 0 offset; record (name ^ " unmap"));
      };
      synchronize = (fun () ->
          record (name ^ " sync");
          if name = "MAP_TARGET:1" && !fail_first_sync then begin
            fail_first_sync := false;
            failwith "import wait failed"
          end);
      alloc = (fun _ _ -> fail "mapping must not allocate target storage");
      free = (fun _ _ _ -> fail "mapping must not free source through target");
      addr = Some (fun (_, offset) -> Nativeint.of_int (0x1000 + offset));
      offset = Some (fun (raw, base) _ offset -> raw, base + offset);
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
  Device.Buffer.copyin source (i32_to_bytes [0; 42; 0; 0]);
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
  equal int 0 (count "MAP_TARGET:1 unmap");
  events := [];
  Device.Buffer.deallocate source;
  is_true (Option.is_none (Device.Buffer.find_mapping target_kind source));
  equal (list string)
    ["MAP_TARGET:2 sync"; "MAP_TARGET:2 unmap";
     "MAP_TARGET:1 sync"; "MAP_TARGET:1 unmap"; "source free"]
    (List.rev !events);
  is_false (Device.Buffer.is_allocated view);
  let new_data, new_offset = get first view in
  is_false (new_data == data);
  equal int 4 new_offset;
  equal int 1 (Device.Buffer.allocated_views source);
  equal int 1 (count "MAP_TARGET:1 map");
  ignore (get second view);
  fail_first_sync := true;
  raises (Failure "import wait failed") (fun () -> Device.Buffer.deallocate source);
  (* The second import was retired before waiting for the first failed. Its
     view must derive a fresh import even though the source allocation lives. *)
  let maps = count "MAP_TARGET:2 map" in
  let partial_data, partial_offset = get second view in
  is_true (partial_data == new_data);
  equal int 4 partial_offset;
  equal int (maps + 1) (count "MAP_TARGET:2 map");
  fail_source_free := true;
  raises (Failure "source free failed") (fun () -> Device.Buffer.deallocate source);
  is_true (Option.is_none (Device.Buffer.find_mapping target_kind source));
  let maps = count "MAP_TARGET:1 map" in
  let retained_data, retained_offset = get first view in
  is_true (retained_data == new_data);
  equal int 4 retained_offset;
  equal int (maps + 1) (count "MAP_TARGET:1 map");
  Device.Buffer.deallocate source;
  Device.Buffer.deallocate view;
  equal int 0 (Device.Buffer.allocated_views source)

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
      synchronize = (fun () -> !probe ());
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
     "cache invalidation", (fun () -> Option.get (Device.invalidate_caches dev) ())];
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
  let backing = Stdlib.Weak.create 1 in
  let host = Storage.Host_allocator.make ~synchronize:(fun () -> ()) in
  let allocator = Device.Allocator.Pack {host with
      alloc = (fun _ _ -> Nativeint.of_int (Sys.opaque_identity 1));
      free = (fun raw _ _ ->
        Stdlib.Weak.set backing 0 (Some raw);
        incr frees;
        failwith "teardown failed")} in
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
  equal ~msg:"uncertain teardown is not retried" int 1 !frees;
  is_true ~msg:"uncertain native backing survives collection"
    (Stdlib.Weak.check backing 0)

let program_storage_is_device_owned () =
  let host = Storage.Host_allocator.make ~synchronize:(fun () -> ()) in
  let allocations = Atomic.make 0 and fail = ref true in
  let make name =
    Device.make ~name ~allocator:(Device.Allocator.Pack host)
      ~renderer_set:(Device.Renderer_set.make ~device:name [])
      ~synchronize:(fun timeout -> ignore timeout)
      ~bufferize:(fun u ->
        if !fail then failwith "program setup failed";
        ignore (Atomic.fetch_and_add allocations 1);
        Unix.sleepf 0.001;
        Some (Device.Buffer.create ~device:name ~size:(Uop.max_numel u)
          ~dtype:(Uop.dtype u) (Device.Allocator.Pack host))) () in
  let owner = make "PROGRAM_OWNER" in
  let placeholder = Uop.placeholder ~slot:0 ~shape:[16] ~dtype:D.uint8
      ~device:(Uop.Single (Device.name owner)) () |> Uop.with_tag "program" in
  raises (Failure "program setup failed") (fun () ->
      ignore (Device.bufferize owner placeholder));
  fail := false;
  let worker = Domain.spawn (fun () -> Option.get (Device.bufferize owner placeholder)) in
  let buffer = Option.get (Device.bufferize owner placeholder) in
  let concurrent = Domain.join worker in
  equal ~msg:"concurrent links share one program allocation" int 1 (Atomic.get allocations);
  is_true (buffer == concurrent);
  let scratch = Uop.with_tag "kernargs" placeholder in
  let first = Option.get (Device.bufferize owner scratch)
  and second = Option.get (Device.bufferize owner scratch) in
  is_false ~msg:"writable command storage remains independent" (first == second);
  let other = make "PROGRAM_OTHER_OWNER" in
  let separate = Option.get (Device.bufferize other placeholder) in
  is_false ~msg:"program storage is owned by each device" (buffer == separate);
  equal int 4 (Atomic.get allocations)

let independent_views_share_one_root () =
  let root = filled_i32 [10; 20; 30; 40] in
  let domains = 4 and per_domain = 128 in
  let ready = Atomic.make 0 in
  let workers =
    Array.init domains (fun worker ->
        Domain.spawn (fun () ->
            ignore (Atomic.fetch_and_add ready 1);
            while Atomic.get ready <> domains do Domain.cpu_relax () done;
            Array.init per_domain (fun _ ->
                let view =
                  Device.Buffer.view root ~size:1 ~dtype:i32
                    ~offset:(worker * D.itemsize i32)
                in
                Device.Buffer.ensure_allocated view;
                equal (list int) [(worker + 1) * 10] (read_i32 view);
                view)))
  in
  let views = Array.map Domain.join workers in
  equal ~msg:"all independently initialized views are counted" int
    (domains * per_domain) (Device.Buffer.allocated_views root);
  Atomic.set ready 0;
  let workers =
    Array.map
      (fun owned ->
        Domain.spawn (fun () ->
            ignore (Atomic.fetch_and_add ready 1);
            while Atomic.get ready <> domains do Domain.cpu_relax () done;
            Array.iter Device.Buffer.deallocate owned))
      views
  in
  Array.iter Domain.join workers;
  equal ~msg:"independent view retirement balances the root count" int 0
    (Device.Buffer.allocated_views root);
  let[@inline never] drop_views () =
    let views =
      Array.init per_domain (fun _ ->
          let view = Device.Buffer.view root ~size:1 ~dtype:i32 ~offset:0 in
          Device.Buffer.ensure_allocated view;
          view)
    in
    equal int per_domain (Device.Buffer.allocated_views root);
    ignore (Sys.opaque_identity views)
  in
  drop_views ();
  Domain.join (Domain.spawn (fun () ->
      for _ = 1 to 3 do Gc.full_major () done));
  equal ~msg:"cross-domain collection balances the remaining view count" int 0
    (Device.Buffer.allocated_views root);
  equal (list int) [10; 20; 30; 40] (read_i32 root);
  Device.Buffer.deallocate root

let () = run __FILE__ [ copy_from_tests;
  test "independent views share root ownership across domains" independent_views_share_one_root;
  test "program storage belongs to the device across links" program_storage_is_device_owned;
  test "device bootstrap registration rolls back failed initialization" device_initialization_registration;
  test "concurrent device lookup runs one opener" concurrent_device_opening;
  test "incomplete devices are private to the initializing thread" incomplete_device_is_private;
  test "failed device initialization wakes waiting callers" failed_initialization_wakes_waiters;
  test "failed openers cannot publish provisional devices" failed_opener_does_not_publish;
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
  test "opaque storage access requires a type identity" typed_storage_identity;
  test "BUFFER owns storage across execution contexts" node_owned_storage;
  test "serialization preserves bytes and shared view ownership" storage_serialization;
  test "serialization copies external storage into an independent owner" external_storage_serialization;
  test "serialization keeps unopened storage lazy" lazy_storage_serialization; test "buffer byte ranges reject overflow" buffer_byte_ranges; test "compilation canonicalizes interleaved kernel arguments" interleaved_kernel_formals; test "empty storage never calls an allocator" empty_storage; test "failed view allocation preserves ownership" failed_view_allocation_preserves_ownership;
  test "stale views refresh on every storage access" stale_views_refresh_on_access;
  test "external views refresh without freeing their owner" external_views_refresh_without_freeing_owner ]
