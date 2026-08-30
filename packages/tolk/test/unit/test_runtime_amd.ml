(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module File_io = Tolk_amd.Hcq.File_io
module Mmio = Tolk_amd.Hcq.Mmio
module Buffer = Tolk_amd.Hcq.Buffer
module Q = Tolk_amd.Hcq.Q
module Signal = Tolk_amd.Hcq.Signal
module Kernargs = Tolk_amd.Hcq.Kernargs
module Compiler_amd = Tolk_amd.Compiler_amd

let is_invalid_arg = function Invalid_argument _ -> true | _ -> false

let contains hay needle =
  let nlen = String.length needle in
  let rec at i =
    i + nlen <= String.length hay
    && (String.equal (String.sub hay i nlen) needle || at (i + 1))
  in
  at 0

let is_comgr_compile_error = function
  | Tolk.Compiler.Compile_error msg -> contains msg "comgr fail"
  | _ -> false

let with_map size f =
  let addr =
    File_io.mmap ~addr:0n ~size
      ~prot:(File_io.prot_read lor File_io.prot_write)
      ~flags:(File_io.map_private lor File_io.map_anonymous)
      ~fd:(-1) ~offset:0L
  in
  Fun.protect
    ~finally:(fun () -> File_io.munmap addr ~size)
    (fun () -> f (Mmio.make ~addr ~size))

(* A 16-byte signal slot at the start of a mapped region; [va] lets the
   device address differ from the CPU mapping. *)
let slot_buf ?va m =
  let va = match va with Some v -> v | None -> Mmio.addr m in
  Buffer.make ~va ~size:16 ~view:(Mmio.view m ~off:0 ~size:16 ()) ~meta:() ()

let amd_dev ~target ~xccs ~gc_version ~nbio_version ~sdma_version ?sqtt_enabled
    () =
  Tolk_amd.device ~target ~xccs ~gc_version ~nbio_version ~sdma_version
    ?sqtt_enabled ~tmpring_size:0x00200008
    ~scratch:(Buffer.make ~va:0x200000n ~size:0x80000 ~meta:() ())
    ~is_am:false ~queue_event_mailbox_ptr:0x500000n
    ~queue_event:{ Tolk_amd.event_id = 0x2a } ()

let gfx1100 ?sqtt_enabled () =
  amd_dev ~target:(11, 0, 0) ~xccs:1 ~gc_version:(11, 0, 0)
    ~nbio_version:(4, 3, 0) ~sdma_version:(6, 0, 0) ?sqtt_enabled ()

let gfx942 () =
  amd_dev ~target:(9, 4, 2) ~xccs:8 ~gc_version:(9, 4, 3)
    ~nbio_version:(7, 9, 0) ~sdma_version:(4, 4, 2) ()

let amd_prog ?(private_segment = false) ?(dispatch_ptr = false) dev =
  {
    Tolk_amd.dev;
    prog_addr = 0x100000n;
    rsrc1 = 0;
    rsrc2 = 0;
    rsrc3 = 0;
    wave32 = true;
    enable_private_segment_sgpr = private_segment;
    enable_dispatch_ptr = dispatch_ptr;
  }

let reg ~addr =
  {
    Tolk_amd.Amd_tables.Reg.name = "regTEST";
    offset = 0;
    segment = 0;
    fields = [||];
    addr;
  }

let () =
  run "Amd_runtime"
    [
      group "File_io"
        [
          test "opens and closes a file" (fun () ->
              let fd = File_io.openfile "/dev/null" ~flags:File_io.o_rdonly in
              is_true (fd >= 0);
              File_io.close fd);
          test "open reports the system error" (fun () ->
              raises_match (Exn.failure ~substring:"No such file") (fun () ->
                  File_io.openfile "/nonexistent/tolk-amd-test"
                    ~flags:File_io.o_rdonly));
        ];
      group "Mmio"
        [
          test "make records address and size" (fun () ->
              with_map 8192 (fun m ->
                  is_true (Mmio.addr m <> 0n);
                  equal int 8192 (Mmio.size m)));
          test "32-bit roundtrip at byte offsets" (fun () ->
              with_map 8192 (fun m ->
                  Mmio.write32 m 0 0x11223344l;
                  Mmio.write32 m 4 0x55667788l;
                  Mmio.fence ();
                  equal int32 0x11223344l (Mmio.read32 m 0);
                  equal int32 0x55667788l (Mmio.read32 m 4)));
          test "64-bit roundtrip overlays 32-bit halves" (fun () ->
              with_map 8192 (fun m ->
                  Mmio.write64 m 8 0x0123456789ABCDEFL;
                  equal int64 0x0123456789ABCDEFL (Mmio.read64 m 8);
                  (* Mapped memory is little-endian on every supported
                     target. *)
                  equal int32 0x89ABCDEFl (Mmio.read32 m 8);
                  equal int32 0x01234567l (Mmio.read32 m 12)));
          test "views translate offsets to the parent region" (fun () ->
              with_map 8192 (fun m ->
                  let v = Mmio.view m ~off:16 () in
                  equal int (8192 - 16) (Mmio.size v);
                  Mmio.write32 v 0 0xCAFEBABEl;
                  equal int32 0xCAFEBABEl (Mmio.read32 m 16);
                  let nested = Mmio.view v ~off:8 ~size:8 () in
                  Mmio.write64 nested 0 0x1122334455667788L;
                  equal int64 0x1122334455667788L (Mmio.read64 m 24)));
          test "views are bounds-checked" (fun () ->
              with_map 8192 (fun m ->
                  raises_match is_invalid_arg (fun () ->
                      Mmio.view m ~off:8192 ~size:1 ());
                  raises_match is_invalid_arg (fun () ->
                      Mmio.view m ~off:(-1) ());
                  raises_match is_invalid_arg (fun () ->
                      Mmio.view m ~off:0 ~size:8193 ())));
          test "reads and writes are bounds-checked" (fun () ->
              with_map 8192 (fun m ->
                  raises_match is_invalid_arg (fun () ->
                      Mmio.read32 m 8190);
                  raises_match is_invalid_arg (fun () ->
                      Mmio.write64 m 8188 0L);
                  raises_match is_invalid_arg (fun () ->
                      Mmio.read32 m (-4))));
          test "bytes roundtrip through the region" (fun () ->
              with_map 8192 (fun m ->
                  let payload = Bytes.of_string "hello, mmio!" in
                  Mmio.blit_bytes m ~off:32 payload;
                  equal bytes payload
                    (Mmio.read_bytes m ~off:32 ~len:(Bytes.length payload));
                  raises_match is_invalid_arg (fun () ->
                      Mmio.blit_bytes m ~off:8181 payload)));
        ];
      group "Buffer"
        [
          test "offset narrows va, size, and view" (fun () ->
              with_map 4096 (fun m ->
                  let buf =
                    Buffer.make ~va:0x1000n ~size:4096 ~view:m ~meta:() ()
                  in
                  let sub = Buffer.offset buf ~off:256 ~size:64 () in
                  equal nativeint 0x1100n (Buffer.va sub);
                  equal int 64 (Buffer.size sub);
                  Mmio.write32 (Buffer.cpu_view sub) 0 0xCAFEBABEl;
                  equal int32 0xCAFEBABEl (Mmio.read32 m 256)));
          test "offset defaults size to the remainder" (fun () ->
              let buf = Buffer.make ~va:0n ~size:4096 ~meta:() () in
              let sub = Buffer.offset buf ~off:4000 () in
              equal int 96 (Buffer.size sub));
          test "offset is bounds-checked" (fun () ->
              let buf = Buffer.make ~va:0n ~size:64 ~meta:() () in
              raises_match is_invalid_arg (fun () ->
                  Buffer.offset buf ~off:(-1) ());
              raises_match is_invalid_arg (fun () ->
                  Buffer.offset buf ~off:32 ~size:33 ());
              raises_match is_invalid_arg (fun () ->
                  Buffer.offset buf ~off:0 ~size:65 ()));
          test "sub-buffers share meta and base" (fun () ->
              let meta = ref 0 in
              let buf = Buffer.make ~va:0n ~size:64 ~meta () in
              let nested =
                Buffer.offset (Buffer.offset buf ~off:16 ()) ~off:16 ()
              in
              is_true (Buffer.meta nested == meta);
              is_true (Buffer.base nested == buf);
              is_true (Buffer.base buf == buf);
              equal nativeint 32n (Buffer.va nested));
          test "cpu_view raises without a view" (fun () ->
              let buf = Buffer.make ~va:0n ~size:64 ~meta:() () in
              raises_match is_invalid_arg (fun () -> Buffer.cpu_view buf));
        ];
      group "Q"
        [
          test "push accumulates dwords in order" (fun () ->
              let q = Q.create () in
              Q.push q 0xC0065800;
              Q.push q 0;
              Q.push q 0xFFFFFFFF;
              equal int 3 (Q.length q);
              equal (array int) [| 0xC0065800; 0; 0xFFFFFFFF |] (Q.dwords q);
              equal int 0xFFFFFFFF (Q.get q 2);
              Q.clear q;
              equal int 0 (Q.length q);
              equal (array int) [||] (Q.dwords q));
          test "push rejects values wider than 32 bits" (fun () ->
              let q = Q.create () in
              raises_match is_invalid_arg (fun () -> Q.push q 0x100000000);
              raises_match is_invalid_arg (fun () -> Q.push q (-1)));
          test "set replaces a dword in place" (fun () ->
              let q = Q.create () in
              Q.push q 1;
              Q.push q 2;
              Q.set q 1 0xFFFFFFFF;
              equal (array int) [| 1; 0xFFFFFFFF |] (Q.dwords q);
              raises_match is_invalid_arg (fun () -> Q.set q 2 0);
              raises_match is_invalid_arg (fun () -> Q.set q (-1) 0);
              raises_match is_invalid_arg (fun () -> Q.set q 0 0x100000000);
              raises_match is_invalid_arg (fun () -> Q.set q 0 (-1)));
          test "push64 pushes the low dword first" (fun () ->
              let q = Q.create () in
              Q.push64 q 0x1122334455667788L;
              Q.push64 q (-1L);
              equal (array int)
                [| 0x55667788; 0x11223344; 0xFFFFFFFF; 0xFFFFFFFF |]
                (Q.dwords q));
          test "grows past the initial capacity" (fun () ->
              let q = Q.create () in
              for i = 0 to 299 do
                Q.push q i
              done;
              equal int 300 (Q.length q);
              equal int 0 (Q.get q 0);
              equal int 299 (Q.get q 299);
              raises_match is_invalid_arg (fun () -> Q.get q 300);
              raises_match is_invalid_arg (fun () -> Q.get q (-1)));
        ];
      group "Signal"
        [
          test "value roundtrips through slot memory" (fun () ->
              with_map 4096 (fun m ->
                  let s = Signal.make ~value:5 (slot_buf m) in
                  equal int 5 (Signal.value s);
                  equal int64 5L (Mmio.read64 m 0);
                  Signal.set_value s 42;
                  equal int64 42L (Mmio.read64 m 0);
                  Mmio.write64 m 0 77L;
                  equal int 77 (Signal.value s)));
          test "addresses come from the slot's device address" (fun () ->
              with_map 4096 (fun m ->
                  let s = Signal.make (slot_buf ~va:0x400000n m) in
                  equal nativeint 0x400000n (Signal.value_addr s);
                  equal nativeint 0x400008n (Signal.timestamp_addr s);
                  (* The device address is decoupled from the CPU mapping:
                     stores still land in the mapped slot. *)
                  Signal.set_value s 9;
                  equal int64 9L (Mmio.read64 m 0)));
          test "timestamp divides the raw counter" (fun () ->
              with_map 4096 (fun m ->
                  let s =
                    Signal.make ~timestamp_divider:100. (slot_buf m)
                  in
                  Mmio.write64 m 8 12345L;
                  equal (float 1e-9) 123.45 (Signal.timestamp s)));
          test "wait returns without sleeping when satisfied" (fun () ->
              with_map 4096 (fun m ->
                  let slept = ref 0 in
                  let s =
                    Signal.make ~value:7 ~sleep:(fun _ -> incr slept)
                      (slot_buf m)
                  in
                  Signal.wait s ~timeout_ms:1000 7;
                  equal int 0 !slept));
          test "wait times out with the last observed value" (fun () ->
              with_map 4096 (fun m ->
                  let slept = ref 0 in
                  let s =
                    Signal.make ~value:1 ~sleep:(fun _ -> incr slept)
                      (slot_buf m)
                  in
                  raises_match
                    (function
                      | Signal.Timeout { timeout_ms = 5; goal = 5; value = 1 }
                        ->
                          true
                      | _ -> false)
                    (fun () -> Signal.wait s ~timeout_ms:5 5);
                  is_true (!slept > 0)));
          test "progress through the sleep hook completes the wait" (fun () ->
              with_map 4096 (fun m ->
                  let sref = ref None in
                  let sleep _ =
                    match !sref with
                    | Some s -> Signal.set_value s (Signal.value s + 1)
                    | None -> ()
                  in
                  let s = Signal.make ~sleep (slot_buf m) in
                  sref := Some s;
                  Signal.wait s ~timeout_ms:1000 3;
                  equal int 3 (Signal.value s)));
          test "a raising sleep hook aborts the wait" (fun () ->
              with_map 4096 (fun m ->
                  let s =
                    Signal.make ~sleep:(fun _ -> failwith "fault") (slot_buf m)
                  in
                  raises_match (Exn.failure ~substring:"fault") (fun () ->
                      Signal.wait s ~timeout_ms:1000 1)));
          test "make validates the slot" (fun () ->
              with_map 4096 (fun m ->
                  raises_match is_invalid_arg (fun () ->
                      Signal.make (Buffer.make ~va:0n ~size:16 ~meta:() ()));
                  raises_match is_invalid_arg (fun () ->
                      Signal.make
                        (Buffer.make ~va:0n ~size:8
                           ~view:(Mmio.view m ~off:0 ~size:8 ())
                           ~meta:() ()))));
          test "pool carves pages into reusable slots" (fun () ->
              with_map 4096 (fun m ->
                  let root =
                    Buffer.make ~va:(Mmio.addr m) ~size:4096 ~view:m ~meta:()
                      ()
                  in
                  let pages_alloced = ref 0 in
                  let alloc_page () =
                    let off = !pages_alloced * 64 in
                    incr pages_alloced;
                    Buffer.offset root ~off ~size:64 ()
                  in
                  let pool = Signal.Pool.create ~alloc_page in
                  let slots = List.init 4 (fun _ -> Signal.Pool.get pool) in
                  equal int 1 !pages_alloced;
                  List.iter (fun s -> equal int 16 (Buffer.size s)) slots;
                  let base = Buffer.va root in
                  equal
                    (list nativeint)
                    [
                      base;
                      Nativeint.add base 16n;
                      Nativeint.add base 32n;
                      Nativeint.add base 48n;
                    ]
                    (List.sort compare (List.map Buffer.va slots));
                  let s5 = Signal.Pool.get pool in
                  equal int 2 !pages_alloced;
                  is_true (Buffer.va s5 >= Nativeint.add base 64n);
                  Signal.Pool.put pool s5;
                  is_true (Signal.Pool.get pool == s5);
                  equal int 2 (List.length (Signal.Pool.pages pool))));
        ];
      group "Kernargs"
        [
          test "alloc hands out 8-byte-aligned slots" (fun () ->
              with_map 4096 (fun m ->
                  let root =
                    Buffer.make ~va:0x300000n ~size:4096 ~view:m ~meta:() ()
                  in
                  let k = Kernargs.create root in
                  let a = Kernargs.alloc k 24 in
                  equal nativeint 0x300000n (Buffer.va a);
                  equal int 24 (Buffer.size a);
                  let b = Kernargs.alloc k 8 in
                  equal nativeint 0x300018n (Buffer.va b);
                  let c = Kernargs.alloc k 4 in
                  equal nativeint 0x300020n (Buffer.va c);
                  let d = Kernargs.alloc k 8 in
                  equal nativeint 0x300028n (Buffer.va d)));
          test "write_args lays out addresses then values" (fun () ->
              with_map 4096 (fun m ->
                  let root =
                    Buffer.make ~va:0x300000n ~size:4096 ~view:m ~meta:() ()
                  in
                  let slot = Kernargs.alloc (Kernargs.create root) 24 in
                  Kernargs.write_args slot ~bufs:[| 0x1000n; 0x2000n |]
                    ~vals:[| 7; -1 |];
                  equal bytes
                    (Bytes.of_string
                       "\x00\x10\x00\x00\x00\x00\x00\x00\
                        \x00\x20\x00\x00\x00\x00\x00\x00\
                        \x07\x00\x00\x00\xff\xff\xff\xff")
                    (Mmio.read_bytes m ~off:0 ~len:24)));
          test "write_args is bounds- and range-checked" (fun () ->
              with_map 4096 (fun m ->
                  let root =
                    Buffer.make ~va:0n ~size:4096 ~view:m ~meta:() ()
                  in
                  let slot = Buffer.offset root ~off:0 ~size:16 () in
                  raises_match is_invalid_arg (fun () ->
                      Kernargs.write_args slot
                        ~bufs:[| 0x1n; 0x2n; 0x3n |]
                        ~vals:[||]);
                  raises_match is_invalid_arg (fun () ->
                      Kernargs.write_args slot ~bufs:[||]
                        ~vals:[| 0x100000000 |])));
          test "the region wraps when exhausted" (fun () ->
              with_map 4096 (fun m ->
                  let root =
                    Buffer.make ~va:0x300000n ~size:64
                      ~view:(Mmio.view m ~off:0 ~size:64 ())
                      ~meta:() ()
                  in
                  let k = Kernargs.create root in
                  ignore (Kernargs.alloc k 48);
                  let wrapped = Kernargs.alloc k 32 in
                  equal nativeint 0x300000n (Buffer.va wrapped);
                  raises_match is_invalid_arg (fun () ->
                      Kernargs.alloc k 80)));
        ];
      group "Compute_queue"
        [
          test "wreg routes by register range" (fun () ->
              let module Cq = Tolk_amd.Compute_queue in
              let q = Cq.create (gfx1100 ()) in
              Cq.wreg q (reg ~addr:0x2c00) [| 0xAB |];
              Cq.wreg q (reg ~addr:0xc000) [| 0xCD |];
              equal (array int)
                [| 0xC0017600; 0x0; 0xAB; 0xC0017900; 0x0; 0xCD |]
                (Q.dwords (Cq.q q)));
          test "wreg rejects registers outside both ranges" (fun () ->
              let module Cq = Tolk_amd.Compute_queue in
              let q = Cq.create (gfx1100 ()) in
              raises_match is_invalid_arg (fun () ->
                  Cq.wreg q (reg ~addr:0x3000) [| 0 |]);
              raises_match is_invalid_arg (fun () ->
                  Cq.wreg q (reg ~addr:(0xc000 + 0xffff)) [| 0 |]);
              (* the last register of each range still routes *)
              Cq.wreg q (reg ~addr:0x2fff) [| 0 |];
              Cq.wreg q (reg ~addr:(0xc000 + 0xfffe)) [| 0 |];
              equal int 6 (Q.length (Cq.q q)));
          test "exec rejects unsupported programs" (fun () ->
              let module Cq = Tolk_amd.Compute_queue in
              let kernargs = Buffer.make ~va:0x300000n ~size:64 ~meta:() () in
              let exec dev prg =
                Cq.exec (Cq.create dev) prg ~kernargs ~global_size:(1, 1, 1)
                  ~local_size:(1, 1, 1)
              in
              let dev = gfx1100 () in
              raises_match is_invalid_arg (fun () ->
                  exec dev (amd_prog ~dispatch_ptr:true dev));
              let sqtt_dev = gfx1100 ~sqtt_enabled:true () in
              raises_match is_invalid_arg (fun () ->
                  exec sqtt_dev (amd_prog sqtt_dev));
              let multi_xcc = gfx942 () in
              raises_match is_invalid_arg (fun () ->
                  exec multi_xcc (amd_prog ~private_segment:true multi_xcc)));
          test "a command value wider than 32 bits is rejected" (fun () ->
              let module Cq = Tolk_amd.Compute_queue in
              with_map 4096 (fun m ->
                  let s = Signal.make (slot_buf ~va:0x400000n m) in
                  let q = Cq.create (gfx1100 ()) in
                  raises_match is_invalid_arg (fun () ->
                      Cq.wait q ~value:0x100000000 s)));
        ];
      group "Copy_queue"
        [
          test "copy chunks at the copy-size cap" (fun () ->
              let module Cp = Tolk_amd.Copy_queue in
              let dev = gfx1100 () in
              let src = Buffer.make ~va:0x10000000n ~size:0 ~meta:() () in
              let dst = Buffer.make ~va:0x20000000n ~size:0 ~meta:() () in
              let exact = Cp.create ~max_copy_size:0x1000 dev in
              Cp.copy exact ~dest:dst ~src 0x1000;
              equal (list int) [ 7 ] (Cp.cmd_sizes exact);
              equal int 0xfff (Q.get (Cp.q exact) 1);
              let split = Cp.create ~max_copy_size:0x1000 dev in
              Cp.copy split ~dest:dst ~src 0x1001;
              equal (list int) [ 7; 7 ] (Cp.cmd_sizes split);
              let q = Cp.q split in
              equal int 0xfff (Q.get q 1);
              (* the second chunk copies the single remaining byte at
                 +0x1000 *)
              equal int 0 (Q.get q 8);
              equal int 0x10001000 (Q.get q 10);
              equal int 0x20001000 (Q.get q 12));
          test "cmd_sizes records packet boundaries" (fun () ->
              let module Cp = Tolk_amd.Copy_queue in
              with_map 4096 (fun m ->
                  let dev = gfx942 () in
                  let s =
                    Signal.make ~is_timeline:true ~owner:dev
                      (slot_buf ~va:0x400000n m)
                  in
                  let q = Cp.create dev in
                  Cp.signal q ~value:1 s;
                  equal (list int) [ 4; 4; 2 ] (Cp.cmd_sizes q);
                  equal int 10 (Q.length (Cp.q q))));
        ];
      group "Compiler"
        [
          test "missing comgr degrades to Failure" (fun () ->
              match Compiler_amd.version () with
              | _ -> skip ~reason:"libamd_comgr is installed" ()
              | exception Failure _ ->
                  raises_match (Exn.failure ~substring:"comgr library")
                    (fun () -> Compiler_amd.version ());
                  let compiler = Compiler_amd.create ~arch:"gfx1100" in
                  raises_match (Exn.failure ~substring:"comgr library")
                    (fun () ->
                      Tolk.Compiler.compile compiler
                        "extern \"C\" __global__ void test() {}"));
          test "load failure is retried, not latched" (fun () ->
              match Compiler_amd.version () with
              | _ -> skip ~reason:"libamd_comgr is installed" ()
              | exception Failure _ ->
                  let msg f =
                    match f () with
                    | _ -> fail "expected Failure"
                    | exception Failure m -> m
                  in
                  equal string
                    (msg Compiler_amd.version)
                    (msg Compiler_amd.version));
          test "compiles a trivial HIP kernel" (fun () ->
              match Compiler_amd.version () with
              | exception Failure msg -> skip ~reason:msg ()
              | _ ->
                  let compiler = Compiler_amd.create ~arch:"gfx1100" in
                  let lib =
                    Tolk.Compiler.compile compiler
                      "extern \"C\" __global__ void test() {}"
                  in
                  is_true (Bytes.length lib > 4);
                  equal string "\x7fELF" (Bytes.sub_string lib 0 4));
          test "broken source raises Compile_error" (fun () ->
              match Compiler_amd.version () with
              | exception Failure msg -> skip ~reason:msg ()
              | _ ->
                  let compiler = Compiler_amd.create ~arch:"gfx1100" in
                  raises_match is_comgr_compile_error (fun () ->
                      Tolk.Compiler.compile compiler "this is not hip"));
        ];
    ]
