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
let () = ignore (Sys.opaque_identity Realize.graph_launches)
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
      alloc = (fun _ _ -> ());
      free = (fun () _ _ -> incr frees);
      copyin = (fun () _ -> ()); copyout = (fun _ () -> ());
      addr = (fun () -> Nativeint.one);
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
  is_false (Device.Buffer.is_initialized view);
  Device.Buffer.ensure_allocated view;
  equal int 1 (Device.Buffer.allocated_views base);
  Device.Buffer.deallocate view;
  equal int 0 (Device.Buffer.allocated_views base);
  Device.Buffer.deallocate base;
  equal int 1 !frees


let empty_storage () =
  let unexpected op = fail ("empty storage called allocator " ^ op) in
  let allocator = Device.Allocator.Pack {
      alloc = (fun _ _ -> unexpected "alloc");
      free = (fun () _ _ -> unexpected "free");
      copyin = (fun () _ -> unexpected "copyin");
      copyout = (fun _ () -> unexpected "copyout");
      addr = (fun () -> unexpected "addr");
      offset = None; as_buffer = None; transfer = None; supports_transfer = false;
      copy_from_disk = None; supports_copy_from_disk = false;
    } in
  let create () = Device.Buffer.create ~device:"EMPTY" ~size:0 ~dtype:i32 allocator in
  let src = create () and dst = create () in
  is_false (Device.Buffer.is_initialized src);
  Device.Buffer.ensure_allocated src;
  Device.Buffer.ensure_allocated src;
  is_true (Device.Buffer.is_allocated src);
  is_true (Device.Buffer.is_initialized src);
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
    (Device.Buffer.is_initialized base);
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
  is_false (Device.Buffer.is_initialized base)

let interleaved_kernel_formals () =
  let module U = Uop in
  let allocator = Device.Buffer.allocator
      (Device.create_buffer ~size:0 ~dtype:i32 device) in
  let compiler = Compiler.make ~name:"SIGNATURE_TEST" ~compile:Bytes.of_string () in
  let renderer_set = Device.Renderer_set.make ~device:"SIGNATURE_TEST"
      [ "CUDA", (fun _ -> Renderer.with_compiler compiler (Cstyle.cuda Gpu_target.SM80)) ] in
  let dev = Device.make ~name:"SIGNATURE_TEST" ~allocator ~renderer_set
      ~runtime:(fun _ -> fail "source-only test loaded a binary")
      ~synchronize:(fun () -> ()) () in
  let param slot = U.param ~slot ~dtype:i32 ~shape:(U.const_int 1) () in
  let output = param 7 and input = param 2 in
  let increment = U.variable ~name:"increment" ~min_val:0 ~max_val:100 ~dtype:i32 () in
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

let () = run __FILE__ [ copy_from_tests; test "buffer byte ranges reject overflow" buffer_byte_ranges; test "compilation canonicalizes interleaved kernel arguments" interleaved_kernel_formals; test "empty storage never calls an allocator" empty_storage; test "failed view allocation preserves ownership" failed_view_allocation_preserves_ownership ]
