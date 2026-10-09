(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The requests the amdgpu path makes, with no GPU: each request's number and
   parameters are the bytes a C program compiled against Linux's uapi headers
   (kfd_ioctl.h, amdgpu_drm.h) packs for the same arguments, a pointer in them
   names the request's data, and each reader decodes the bytes such a program
   lays out for the kernel's answer. The expected bytes are that program's,
   printed on x86_64 Linux, with the pointers zeroed. *)

open Windtrap

let gpu = 0xdc43
let va = 0x7f12_3456_7000
let bytes = 0x20_0000
let handle = 0xdc43_0000_0005

(* Printing and reading bytes *)

external address : Request.params -> int = "caml_rig_amd_amdgpu_address"

let hex (p : Request.params) =
  String.concat ""
    (List.init (Bigarray.Array1.dim p) (fun i ->
         Printf.sprintf "%02x" (Char.code (Bigarray.Array1.get p i))))

let of_hex h =
  let n = String.length h / 2 in
  let p = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  for i = 0 to n - 1 do
    Bigarray.Array1.set p i
      (Char.chr (int_of_string ("0x" ^ String.sub h (2 * i) 2)))
  done;
  p

let blit (src : Request.params) (dst : Request.params) =
  Bigarray.Array1.blit src (Bigarray.Array1.sub dst 0 (Bigarray.Array1.dim src))

(* The pointer field of each request that has one. *)
let pointer name =
  match name with
  | "map" | "unmap" -> Some Defs.Map_memory_to_gpu.device_ids_array_ptr
  | "wait_3" | "wait_2" -> Some Defs.Wait_events.events_ptr
  | "device_info" -> Some Defs.Info.return_pointer
  | _ -> None

let get64 (p : Request.params) at =
  let b = Bytes.init 8 (fun i -> Bigarray.Array1.get p (at + i)) in
  Int64.to_int (Bytes.get_int64_le b 0)

(* [r]'s number and parameters, its pointer, which must name its data,
   zeroed. *)
let packed name (r : Request.t) =
  Option.iter
    (fun (at, _) ->
      equal ~msg:"the pointer names the data" int (address r.data)
        (get64 r.params at);
      Bigarray.Array1.fill (Bigarray.Array1.sub r.params at 8) '\000')
    (pointer name);
  Printf.sprintf "0x%x %s" r.number (hex r.params)

(* Requests *)

let queue k =
  Request.queue k ~gpu ~ring:0x7f00_0000_1000 ~ring_bytes:0x10000
    ~eop:0x7f00_0010_0000 ~eop_bytes:0x1000 ~save:0x7f00_0020_0000
    ~save_bytes:0x2a000 ~ctl_stack:0x3000 ~write:0x7f00_0030_0000
    ~read:0x7f00_0030_0008

let requests =
  [
    ("version", Request.version ());
    ("acquire_vm", Request.acquire_vm ~drm:7 ~gpu);
    ("runtime_enable", Request.runtime_enable ());
    ("alloc_gpu", Request.alloc ~gpu ~va ~bytes `Gpu);
    ("alloc_bar", Request.alloc ~gpu ~va ~bytes `Bar);
    ("alloc_system", Request.alloc ~gpu ~va ~bytes `System);
    ("alloc_userptr", Request.alloc ~gpu ~va ~bytes `Userptr);
    ("alloc_mmio", Request.alloc ~gpu ~va ~bytes `Mmio);
    ("free", Request.free handle);
    ("map", Request.map ~gpu handle);
    ("unmap", Request.unmap ~gpu handle);
    ("event_signal", Request.event `Signal ~page:0x1234);
    ("event_memory", Request.event `Memory ~page:0);
    ("event_hardware", Request.event `Hardware ~page:0);
    ("destroy_event", Request.destroy_event 0x42);
    ("reset_event", Request.reset_event 0x43);
    ("queue_pm4", queue `Pm4);
    ("queue_aql", queue `Aql);
    ("queue_sdma", queue `Sdma);
    ("destroy_queue", Request.destroy_queue 3);
    ("wait_3", Request.wait [| 0x40; 0x41; 0x42 |] ~ms:1000);
    ("wait_2", Request.wait [| 0x41; 0x42 |] ~ms:0);
    ("device_info", Request.device_info ());
    ("alloc_context", Request.alloc_context ());
    ("stable_pstate", Request.stable_pstate 5);
    ("free_context", Request.free_context 5);
  ]

let c_packed =
  [
    ("version", 0x80084b01, "0000000000000000");
    ("acquire_vm", 0x40084b15, "0700000043dc0000");
    ("runtime_enable", 0xc0104b25, "00000000000000000000000000000000");
    ( "alloc_gpu",
      0xc0284b16,
      "00705634127f0000000020000000000000000000000000000000000000000000"
      ^ "43dc0000010000d0" );
    ( "alloc_bar",
      0xc0284b16,
      "00705634127f0000000020000000000000000000000000000000000000000000"
      ^ "43dc0000010000f0" );
    ( "alloc_system",
      0xc0284b16,
      "00705634127f0000000020000000000000000000000000000000000000000000"
      ^ "43dc0000020000f6" );
    ( "alloc_userptr",
      0xc0284b16,
      "00705634127f00000000200000000000000000000000000000705634127f0000"
      ^ "43dc0000040000f6" );
    ( "alloc_mmio",
      0xc0284b16,
      "00705634127f0000000020000000000000000000000000000000000000000000"
      ^ "43dc000010000090" );
    ("free", 0x40084b17, "0500000043dc0000");
    ("map", 0xc0184b18, "0500000043dc000000000000000000000100000000000000");
    ("unmap", 0xc0184b19, "0500000043dc000000000000000000000100000000000000");
    ( "event_signal",
      0xc0204b08,
      "3412000000000000000000000000000001000000000000000000000000000000" );
    ( "event_memory",
      0xc0204b08,
      "0000000000000000000000000800000000000000000000000000000000000000" );
    ( "event_hardware",
      0xc0204b08,
      "0000000000000000000000000300000000000000000000000000000000000000" );
    ("destroy_event", 0x40084b09, "4200000000000000");
    ("reset_event", 0x40084b0b, "4300000000000000");
    ( "queue_pm4",
      0xc0604b02,
      "00100000007f000000003000007f000008003000007f00000000000000000000"
      ^ "0000010043dc00000000000064000000070000000000000000001000007f0000"
      ^ "001000000000000000002000007f000000a00200003000000000000000000000" );
    ( "queue_aql",
      0xc0604b02,
      "00100000007f000000003000007f000008003000007f00000000000000000000"
      ^ "0000010043dc00000200000064000000070000000000000000001000007f0000"
      ^ "001000000000000000002000007f000000a00200003000000000000000000000" );
    ( "queue_sdma",
      0xc0604b02,
      "00100000007f000000003000007f000008003000007f00000000000000000000"
      ^ "0000010043dc00000100000064000000070000000000000000001000007f0000"
      ^ "001000000000000000002000007f000000a00200003000000000000000000000" );
    ("destroy_queue", 0xc0084b03, "0300000000000000");
    ("wait_3", 0xc0184b0c, "00000000000000000300000000000000e803000000000000");
    ("wait_2", 0xc0184b0c, "000000000000000002000000000000000000000000000000");
    ( "device_info",
      0x40206445,
      "0000000000000000c00100001600000000000000000000000000000000000000" );
    ("alloc_context", 0xc0106442, "01000000000000000000000000000000");
    ("stable_pstate", 0xc0106442, "06000000010000000500000000000000");
    ("free_context", 0xc0106442, "02000000000000000500000000000000");
  ]

let packing =
  cases
    ~name:(fun (name, _, _) -> name)
    "a request packs as C does" c_packed
    (fun (name, number, h) ->
      equal string
        (Printf.sprintf "0x%x %s" number h)
        (packed name (List.assoc name requests)))

(* The data a request's pointer names, as C lays it out. *)
let data =
  cases ~name:fst "a request's data is laid out as C does"
    [
      ("map", "43dc0000");
      ("unmap", "43dc0000");
      ( "wait_3",
        "0000000000000000000000000000000000000000000000000000000000000000"
        ^ "0000000000000000400000000000000000000000000000000000000000000000"
        ^ "0000000000000000000000000000000000000000000000004100000000000000"
        ^ "0000000000000000000000000000000000000000000000000000000000000000"
        ^ "00000000000000004200000000000000" );
      ( "wait_2",
        "0000000000000000000000000000000000000000000000000000000000000000"
        ^ "0000000000000000410000000000000000000000000000000000000000000000"
        ^ "0000000000000000000000000000000000000000000000004200000000000000" );
    ]
    (fun (name, h) -> equal string h (hex (List.assoc name requests).data))

(* Answers *)

(* [r] with its parameters as C lays out the kernel's answer [h]. *)
let answered (r : Request.t) h =
  blit (of_hex h) r.params;
  r

let answers =
  group "answers"
    [
      test "the version" (fun () ->
          equal int 1018
            (Request.version_of
               (answered (Request.version ()) "0100000012000000")));
      test "an allocation's handle and offset" (fun () ->
          let r =
            answered
              (Request.alloc ~gpu ~va ~bytes `Gpu)
              ("00705634127f000000002000000000000500000043dc000000100000000000c0"
             ^ "43dc0000010000d0")
          in
          equal (pair int int64)
            (handle, 0xc000_0000_0000_1000L)
            (Request.allocation r));
      test "the GPUs a map reached" (fun () ->
          equal int 1
            (Request.mapped
               (answered (Request.map ~gpu handle)
                  "0500000043dc000000000000000000000100000001000000")));
      test "an event's id" (fun () ->
          equal int 0x42
            (Request.event_id
               (answered
                  (Request.event `Signal ~page:0x1234)
                  "3412000000000000000000000000000001000000000000004200000000000000")));
      test "a queue's id and doorbell" (fun () ->
          let r =
            answered (queue `Aql)
              ("00100000007f000000003000007f000008003000007f000000080000000000c0"
             ^ "0000010043dc00000200000064000000070000000300000000001000007f0000"
             ^ "001000000000000000002000007f000000a00200003000000000000000000000"
              )
          in
          equal (pair int int64)
            (3, 0xc000_0000_0000_0800L)
            (Request.queue_made r));
      test "a wait's exception events" (fun () ->
          let r = Request.wait [| 0x40; 0x41; 0x42 |] ~ms:1000 in
          blit
            (of_hex
               ("0000000000000000000000000000000000000000000000000000000000000000"
              ^ "0000000000000000400000000000000001000000000000000100000000000000"
              ^ "00b0adde007f000043dc00000200000000000000000000004100000000000000"
              ^ "010000000200000001000000efbe000000000000000000000000000000000000"
              ^ "00000000000000004200000000000000"))
            r.data;
          equal (pair int int) (0x41, gpu) (Request.exception_event r `Memory);
          equal (pair int int) (0x42, 0xbeef)
            (Request.exception_event r `Hardware);
          equal string
            "memory fault at 0x7f00deadb000 (not present 1, read-only 0, no \
             execute 1, imprecise 0, error type 2)"
            (Request.fault r `Memory);
          equal string
            "hardware exception (reset type 1, reset cause 2, memory lost 1)"
            (Request.fault r `Hardware));
      test "the GPU's clock and compute units" (fun () ->
          let r = Request.device_info () in
          equal int 448 (Bigarray.Array1.dim r.data);
          blit
            (of_hex
               ("00000000000000000000000000000000000000000000000000000000a0860100"
              ^ "0000000000000000000000000000000000000000000000000100000002000000"
              ^ "030000000400000005000000060000000700000008000000090000000a000000"
              ^ "0b0000000c0000000d0000000e0000000f000000100000000000000000000000"
               ))
            r.data;
          equal int 100000 (Request.clock_khz r);
          equal (array int)
            (Array.init 16 (fun i -> i + 1))
            (Request.compute_units r));
      test "a context's id" (fun () ->
          equal int 5
            (Request.context
               (answered (Request.alloc_context ())
                  "05000000000000000000000000000000")));
    ]

let () = exit (run "rig_amd_amdgpu.request" [ packing; data; answers ])
