(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module File_io = Tolk_hcq.Hcq.File_io
module Mmio = Tolk_hcq.Hcq.Mmio
module Buffer = Tolk_hcq.Hcq.Buffer
module Q = Tolk_hcq.Hcq.Q
module Signal = Tolk_hcq.Hcq.Signal
module Timeline = Tolk_hcq.Hcq.Timeline
module Kernargs = Tolk_hcq.Hcq.Kernargs
module Compiler_amd = Tolk_amd.Compiler_amd
module Program = Tolk_amd.Program
module Pci_iface = Tolk_amd.Pci_iface
module Amdev = Tolk_amd.Amdev

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
    ?scratch () =
  let scratch =
    match scratch with
    | Some b -> b
    | None -> Buffer.make ~va:0x200000n ~size:0x80000 ~meta:() ()
  in
  Tolk_amd.device ~target ~xccs ~gc_version ~nbio_version ~sdma_version
    ?sqtt_enabled ~tmpring_size:0x00200008 ~scratch ~is_am:false
    ~queue_event_mailbox_ptr:0x500000n
    ~queue_event:{ Tolk_amd.event_id = 0x2a } ()

let gfx1100 ?sqtt_enabled ?scratch () =
  amd_dev ~target:(11, 0, 0) ~xccs:1 ~gc_version:(11, 0, 0)
    ~nbio_version:(4, 3, 0) ~sdma_version:(6, 0, 0) ?sqtt_enabled ?scratch ()

let gfx942 ?scratch () =
  amd_dev ~target:(9, 4, 2) ~xccs:8 ~gc_version:(9, 4, 3)
    ~nbio_version:(7, 9, 0) ~sdma_version:(4, 4, 2) ?scratch ()

let gfx1200 ?scratch () =
  amd_dev ~target:(12, 0, 0) ~xccs:1 ~gc_version:(12, 0, 0)
    ~nbio_version:(6, 3, 1) ~sdma_version:(7, 0, 0) ?scratch ()

(* An empty scratch buffer: the state of a freshly created device, before
   the first scratch sizing. *)
let no_scratch () = Buffer.make ~va:0n ~size:0 ~meta:() ()

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

(* A queue descriptor carved out of one mapping: the ring at offset 0,
   then the read pointer, write pointer, and a fake doorbell word. *)
let queue_desc ~ring_dwords m =
  let ring_bytes = ring_dwords * 4 in
  {
    Tolk_amd.Queue_desc.ring = Mmio.view m ~off:0 ~size:ring_bytes ();
    read_ptr = Mmio.view m ~off:ring_bytes ~size:8 ();
    write_ptr = Mmio.view m ~off:(ring_bytes + 8) ~size:8 ();
    doorbell = Mmio.view m ~off:(ring_bytes + 16) ~size:8 ();
    put_value = 0;
    flush_hdp = None;
    resetup = None;
  }

let ring_dword m i = Int32.to_int (Mmio.read32 m (i * 4)) land 0xFFFFFFFF
let ring_dwords m n = Array.init n (ring_dword m)
let set16 b off v = Bytes.set_uint16_le b off v
let set32 b off v = Bytes.set_int32_le b off (Int32.of_int v)
let set64 b off v = Bytes.set_int64_le b off (Int64.of_int v)

(* Hand-crafted 64-bit little-endian shared object shaped like a compiled
   kernel: [.text] at 0x100, [.rodata] at 0x40 carrying a 64-byte kernel
   descriptor with known field values, and one relocation patching
   [.text + 8] against a symbol at [.rodata + 4]. Field offsets within the
   descriptor are spelled as literals so the loader's parsing is checked
   against independently written numbers. *)
let hsaco_fixture ?(rodata_name = ".rodata") ?(reloc_type = 5)
    ?(undefined_sym = false) ?(group = 0x2000) ?(private_seg = 256)
    ?(kernarg = 24) ?(code_props = 0x400) () =
  let module Buf = Stdlib.Buffer in
  let text = Bytes.of_string "KERNCODE\000\000\000\000\000\000\000\000" in
  let desc = Bytes.make 64 '\000' in
  set32 desc 0 group (* group_segment_fixed_size *);
  set32 desc 4 private_seg (* private_segment_fixed_size *);
  set32 desc 8 kernarg (* kernarg_size *);
  set64 desc 16 0xC0 (* kernel_code_entry_byte_offset *);
  set32 desc 44 0x3333 (* compute_pgm_rsrc3 *);
  set32 desc 48 0x1111 (* compute_pgm_rsrc1 *);
  set32 desc 52 0x2222 (* compute_pgm_rsrc2 *);
  set16 desc 56 code_props (* kernel_code_properties *);
  let symtab = Bytes.make 48 '\000' in
  set32 symtab 24 1 (* st_name: "k" *);
  set16 symtab 30 (if undefined_sym then 0 else 2) (* st_shndx: .rodata *);
  set64 symtab 32 4 (* st_value *);
  let strtab = Bytes.of_string "\000k\000" in
  let rela = Bytes.make 24 '\000' in
  set64 rela 0 8 (* r_offset, within .text *);
  set64 rela 8 ((1 lsl 32) lor reloc_type) (* symbol 1 *);
  set64 rela 16 0x10 (* r_addend *);
  let buf = Buf.create 1024 in
  Buf.add_bytes buf (Bytes.make 64 '\000');
  let pad_to align =
    while Buf.length buf mod align <> 0 do
      Buf.add_char buf '\000'
    done
  in
  let add align content =
    pad_to align;
    let off = Buf.length buf in
    Buf.add_bytes buf content;
    off
  in
  let off_text = add 8 text in
  let off_desc = add 8 desc in
  let off_symtab = add 8 symtab in
  let off_strtab = add 1 strtab in
  let off_rela = add 8 rela in
  let shstr = Buf.create 64 in
  Buf.add_char shstr '\000';
  let name s =
    let off = Buf.length shstr in
    Buf.add_string shstr s;
    Buf.add_char shstr '\000';
    off
  in
  let n_text = name ".text" in
  let n_rodata = name rodata_name in
  let n_symtab = name ".symtab" in
  let n_strtab = name ".strtab" in
  let n_rela = name ".rela.text" in
  let n_shstrtab = name ".shstrtab" in
  let off_shstr = add 1 (Buf.to_bytes shstr) in
  pad_to 8;
  let e_shoff = Buf.length buf in
  let shdr ~nm ~ty ~flags ~addr ~off ~size ~link ~info ~salign ~entsize =
    let b = Bytes.make 64 '\000' in
    set32 b 0 nm;
    set32 b 4 ty;
    set64 b 8 flags;
    set64 b 16 addr;
    set64 b 24 off;
    set64 b 32 size;
    set32 b 40 link;
    set32 b 44 info;
    set64 b 48 salign;
    set64 b 56 entsize;
    Buf.add_bytes buf b
  in
  shdr ~nm:0 ~ty:0 ~flags:0 ~addr:0 ~off:0 ~size:0 ~link:0 ~info:0 ~salign:0
    ~entsize:0;
  shdr ~nm:n_text ~ty:1 ~flags:0x6 ~addr:0x100 ~off:off_text ~size:16 ~link:0
    ~info:0 ~salign:16 ~entsize:0;
  shdr ~nm:n_rodata ~ty:1 ~flags:0x2 ~addr:0x40 ~off:off_desc ~size:64 ~link:0
    ~info:0 ~salign:8 ~entsize:0;
  shdr ~nm:n_symtab ~ty:2 ~flags:0 ~addr:0 ~off:off_symtab ~size:48 ~link:4
    ~info:1 ~salign:8 ~entsize:24;
  shdr ~nm:n_strtab ~ty:3 ~flags:0 ~addr:0 ~off:off_strtab
    ~size:(Bytes.length strtab) ~link:0 ~info:0 ~salign:1 ~entsize:0;
  shdr ~nm:n_rela ~ty:4 ~flags:0 ~addr:0 ~off:off_rela ~size:24 ~link:3 ~info:1
    ~salign:8 ~entsize:24;
  shdr ~nm:n_shstrtab ~ty:3 ~flags:0 ~addr:0 ~off:off_shstr
    ~size:(Buf.length shstr) ~link:0 ~info:0 ~salign:1 ~entsize:0;
  let obj = Buf.to_bytes buf in
  Bytes.blit_string "\x7fELF\x02\x01\x01" 0 obj 0 7;
  set16 obj 16 3 (* e_type: ET_DYN *);
  set16 obj 18 0xE0 (* e_machine: AMDGPU *);
  set32 obj 20 1 (* e_version *);
  set64 obj 40 e_shoff;
  set16 obj 52 64 (* e_ehsize *);
  set16 obj 58 64 (* e_shentsize *);
  set16 obj 60 7 (* e_shnum *);
  set16 obj 62 6 (* e_shstrndx *);
  obj

(* Runs [f] with a [Program.load]-ready allocator over a fresh mapping:
   [alloc] records the sizes it served and hands out CPU-mapped buffers
   with device address 0xA00000. *)
let with_lib_alloc f =
  with_map 0x2000 (fun m ->
      let sizes = ref [] in
      let alloc size =
        sizes := size :: !sizes;
        Buffer.make ~va:0xA00000n ~size
          ~view:(Mmio.view m ~off:0 ~size ())
          ~meta:() ()
      in
      f alloc sizes m)

let lds64 = [ ("lds_size_in_kb", 64) ]

(* A sysfs tree holding just the PCI files the bus scan reads, so the
   device allowlist is checked through the real probe path. *)
let with_fake_sysfs devices f =
  let root = Filename.temp_file "tolk_sysfs" "" in
  Sys.remove root;
  let devdir =
    List.fold_left Filename.concat root [ "bus"; "pci"; "devices" ]
  in
  List.iter
    (fun d -> Sys.mkdir d 0o700)
    [
      root;
      Filename.concat root "bus";
      List.fold_left Filename.concat root [ "bus"; "pci" ];
      devdir;
    ];
  List.iter
    (fun (addr, vendor, device) ->
      let d = Filename.concat devdir addr in
      Sys.mkdir d 0o700;
      List.iter
        (fun (name, v) ->
          Out_channel.with_open_bin (Filename.concat d name) (fun oc ->
              Out_channel.output_string oc (Printf.sprintf "0x%04x\n" v)))
        [ ("vendor", vendor); ("device", device); ("class", 0x030000) ])
    devices;
  let rec rm_tree path =
    if Sys.is_directory path then begin
      Array.iter (fun e -> rm_tree (Filename.concat path e)) (Sys.readdir path);
      Sys.rmdir path
    end
    else Sys.remove path
  in
  Fun.protect ~finally:(fun () -> rm_tree root) (fun () -> f root)

(* Device-level fixtures: one lazily opened real device shared by the group;
   every test using it skips when this machine cannot provide one (no kernel
   driver, or an unsupported GPU). *)
let amd_device =
  let cached : Tolk.Device.t option ref = ref None in
  fun () ->
    match !cached with
    | Some device -> device
    | None -> (
        try
          let device = Tolk_amd.create "AMD" in
          cached := Some device;
          device
        with Failure msg -> skip ~reason:msg ())

module U = Tolk_uop.Uop
module D = Tolk_uop.Dtype

let i32_param ~slot =
  U.param ~slot ~dtype:D.int32
    ~shape:(U.stack [ U.const_int 16 ])
    ~addrspace:D.Global ()

(* dst[0] = src[0] + 1: the smallest kernel exercising a load, an ALU op,
   and a store through the whole compile-and-dispatch path. *)
let increment_program () =
  let p0 = i32_param ~slot:0 in
  let p1 = i32_param ~slot:1 in
  let c0 = U.const (Tolk_uop.Const.int D.int32 0) in
  let idx_src = U.index ~ptr:p1 ~idxs:[ c0 ] () in
  let idx_dst = U.index ~ptr:p0 ~idxs:[ c0 ] () in
  let l0 = U.load ~src:idx_src () in
  let c1 = U.const (Tolk_uop.Const.int D.int32 1) in
  let sum = U.alu_binary ~op:Tolk_uop.Ops.Add ~lhs:l0 ~rhs:c1 in
  let store = U.store ~dst:idx_dst ~value:sum () in
  [ p0; p1; c0; idx_src; idx_dst; l0; c1; sum; store ]

let i32_buf device values =
  let buf =
    Tolk.Device.create_buffer ~size:(List.length values) ~dtype:D.int32 device
  in
  Tolk.Device.Buffer.ensure_allocated buf;
  let bytes = Bytes.create (List.length values * 4) in
  List.iteri
    (fun i v -> Bytes.set_int32_le bytes (i * 4) (Int32.of_int v))
    values;
  Tolk.Device.Buffer.copyin buf bytes;
  buf

let read_i32 buf =
  let bytes = Tolk.Device.Buffer.as_bytes buf in
  List.init (Bytes.length bytes / 4) (fun i ->
      Int32.to_int (Bytes.get_int32_le bytes (i * 4)))

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
      group "Timeline"
        [
          test "a stalled wait folds the hang report into the timeout" (fun () ->
              with_map 4096 (fun m ->
                  let tl =
                    {
                      Timeline.timeline = Signal.make ~value:1 (slot_buf m);
                      shadow_timeline =
                        Signal.make (slot_buf ~va:0x10n (Mmio.view m ~off:16 ()));
                      timeline_value = 3;
                      error_state = None;
                      bounce = [||];
                      bounce_timeline = [||];
                      bounce_next = 0;
                      on_hang = (fun () -> failwith "MMU fault: 0xdead");
                    }
                  in
                  let expect = function
                    | Failure msg ->
                        contains msg "Wait timeout: 5 ms!"
                        && contains msg "(the signal is not set to 2, but 1)"
                        && contains msg "MMU fault: 0xdead"
                    | _ -> false
                  in
                  raises_match expect (fun () ->
                      Timeline.guarded_wait tl (fun () ->
                          Signal.wait tl.Timeline.timeline ~timeout_ms:5 2));
                  (* The same folded error is latched for later waits. *)
                  raises_match expect (fun () -> Timeline.synchronize tl)));
          test "an empty hang report leaves the timeout alone" (fun () ->
              with_map 4096 (fun m ->
                  let tl =
                    {
                      Timeline.timeline = Signal.make (slot_buf m);
                      shadow_timeline =
                        Signal.make (slot_buf ~va:0x10n (Mmio.view m ~off:16 ()));
                      timeline_value = 2;
                      error_state = None;
                      bounce = [||];
                      bounce_timeline = [||];
                      bounce_next = 0;
                      on_hang = (fun () -> failwith "");
                    }
                  in
                  raises_match
                    (function
                      | Failure msg ->
                          contains msg "Wait timeout: 5 ms!"
                          && not (contains msg "\n")
                      | _ -> false)
                    (fun () ->
                      Timeline.guarded_wait tl (fun () ->
                          Signal.wait tl.Timeline.timeline ~timeout_ms:5 1))));
          test "a duplicated fault report is not repeated" (fun () ->
              with_map 4096 (fun m ->
                  let tl =
                    {
                      Timeline.timeline = Signal.make (slot_buf m);
                      shadow_timeline =
                        Signal.make (slot_buf ~va:0x10n (Mmio.view m ~off:16 ()));
                      timeline_value = 2;
                      error_state = None;
                      bounce = [||];
                      bounce_timeline = [||];
                      bounce_next = 0;
                      on_hang = (fun () -> failwith "HW fault: reset_type=1");
                    }
                  in
                  raises_match
                    (function
                      | Failure msg ->
                          String.equal msg "HW fault: reset_type=1"
                      | _ -> false)
                    (fun () ->
                      Timeline.guarded_wait tl (fun () ->
                          failwith "HW fault: reset_type=1"))));
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
          test "write_args lays a prefix before addresses and values"
            (fun () ->
              with_map 4096 (fun m ->
                  let root =
                    Buffer.make ~va:0x300000n ~size:4096 ~view:m ~meta:() ()
                  in
                  let slot = Kernargs.alloc (Kernargs.create root) 32 in
                  Kernargs.write_args slot
                    ~prefix:[| 0xdeadbeef; 1 |]
                    ~bufs:[| 0x1000n |] ~vals:[| 7 |];
                  equal bytes
                    (Bytes.of_string
                       "\xef\xbe\xad\xde\x01\x00\x00\x00\
                        \x00\x10\x00\x00\x00\x00\x00\x00\
                        \x07\x00\x00\x00")
                    (Mmio.read_bytes m ~off:0 ~len:20);
                  raises_match is_invalid_arg (fun () ->
                      Kernargs.write_args slot ~prefix:[| -1 |] ~bufs:[||]
                        ~vals:[||])));
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
          test "submit copies the stream and rings the doorbell" (fun () ->
              let module Cq = Tolk_amd.Compute_queue in
              with_map 4096 (fun m ->
                  let qd = queue_desc ~ring_dwords:16 m in
                  let cq = Cq.create (gfx1100 ()) in
                  List.iter (Q.push (Cq.q cq)) [ 0x11; 0x22; 0x33 ];
                  Cq.submit cq qd;
                  equal (array int) [| 0x11; 0x22; 0x33 |] (ring_dwords m 3);
                  equal int 3 qd.Tolk_amd.Queue_desc.put_value;
                  equal int64 3L (Mmio.read64 m ((16 * 4) + 8));
                  equal int64 3L (Mmio.read64 m ((16 * 4) + 16));
                  (* the stream is kept: submitting again replays it *)
                  Cq.submit cq qd;
                  equal (array int)
                    [| 0x11; 0x22; 0x33; 0x11; 0x22; 0x33 |]
                    (ring_dwords m 6);
                  equal int64 6L (Mmio.read64 m ((16 * 4) + 16))));
          test "submit wraps dword by dword at the ring end" (fun () ->
              let module Cq = Tolk_amd.Compute_queue in
              with_map 4096 (fun m ->
                  let qd = queue_desc ~ring_dwords:8 m in
                  let cq = Cq.create (gfx1100 ()) in
                  List.iter (Q.push (Cq.q cq)) [ 0x11; 0x22; 0x33 ];
                  Cq.submit cq qd;
                  Cq.submit cq qd;
                  Cq.submit cq qd;
                  (* the third stream lands at indices 6, 7, 0 *)
                  equal (array int)
                    [| 0x33; 0x22; 0x33; 0x11; 0x22; 0x33; 0x11; 0x22 |]
                    (ring_dwords m 8);
                  equal int 9 qd.Tolk_amd.Queue_desc.put_value;
                  equal int64 9L (Mmio.read64 m ((8 * 4) + 16))));
          test "multi-die submit wraps the stream in an indirect buffer"
            (fun () ->
              let module Cq = Tolk_amd.Compute_queue in
              with_map 4096 (fun m ->
                  let dev = gfx942 () in
                  let module P = (val dev.Tolk_amd.pm4) in
                  let qd = queue_desc ~ring_dwords:32 m in
                  let cq = Cq.create dev in
                  List.iter (Q.push (Cq.q cq)) [ 0x11; 0x22; 0x33 ];
                  Cq.submit cq qd;
                  let ib_ptr =
                    Int64.add (Int64.of_nativeint (Mmio.addr m)) 20L
                  in
                  equal (array int)
                    [|
                      P.packet3 P.packet3_indirect_buffer 2;
                      Int64.to_int (Int64.logand ib_ptr 0xFFFFFFFFL);
                      Int64.to_int (Int64.shift_right_logical ib_ptr 32);
                      3 lor P.indirect_buffer_valid;
                      P.packet3 P.packet3_nop 2;
                      0x11;
                      0x22;
                      0x33;
                    |]
                    (ring_dwords m 8);
                  equal int 8 qd.Tolk_amd.Queue_desc.put_value));
          test "multi-die submit pads the indirect body past the wrap"
            (fun () ->
              let module Cq = Tolk_amd.Compute_queue in
              with_map 4096 (fun m ->
                  let dev = gfx942 () in
                  let module P = (val dev.Tolk_amd.pm4) in
                  let qd = queue_desc ~ring_dwords:32 m in
                  qd.Tolk_amd.Queue_desc.put_value <- 26;
                  let cq = Cq.create dev in
                  List.iter (Q.push (Cq.q cq)) [ 0x11; 0x22; 0x33 ];
                  Cq.submit cq qd;
                  (* header at 26; the one-dword pad fills index 31 so the
                     body starts back at index 0 *)
                  let ib_ptr = Int64.of_nativeint (Mmio.addr m) in
                  equal (array int)
                    [|
                      P.packet3 P.packet3_indirect_buffer 2;
                      Int64.to_int (Int64.logand ib_ptr 0xFFFFFFFFL);
                      Int64.to_int (Int64.shift_right_logical ib_ptr 32);
                      3 lor P.indirect_buffer_valid;
                      P.packet3 P.packet3_nop 3;
                      0;
                    |]
                    (Array.init 6 (fun i -> ring_dword m (26 + i)));
                  equal (array int) [| 0x11; 0x22; 0x33 |] (ring_dwords m 3);
                  equal int 35 qd.Tolk_amd.Queue_desc.put_value));
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
          test "submit copies packets and advances in bytes" (fun () ->
              let module Cp = Tolk_amd.Copy_queue in
              with_map 4096 (fun m ->
                  let qd = queue_desc ~ring_dwords:16 m in
                  let cp = Cp.create (gfx1100 ()) in
                  let buf = Buffer.make ~va:0x10000000n ~size:8 ~meta:() () in
                  Cp.write cp buf 0xABCDL;
                  Cp.submit cp qd;
                  equal (array int)
                    (Q.dwords (Cp.q cp))
                    (ring_dwords m 5);
                  equal int 20 qd.Tolk_amd.Queue_desc.put_value;
                  equal int64 20L (Mmio.read64 m ((16 * 4) + 8));
                  equal int64 20L (Mmio.read64 m ((16 * 4) + 16))));
          test "a packet that would straddle moves past a zero-filled tail"
            (fun () ->
              let module Cp = Tolk_amd.Copy_queue in
              with_map 4096 (fun m ->
                  let qd = queue_desc ~ring_dwords:16 m in
                  let dev = gfx1100 () in
                  let src = Buffer.make ~va:0x10000000n ~size:0 ~meta:() () in
                  let dst = Buffer.make ~va:0x20000000n ~size:0 ~meta:() () in
                  let first = Cp.create dev in
                  Cp.copy first ~dest:dst ~src 0x100;
                  Cp.copy first ~dest:dst ~src 0x100;
                  Cp.submit first qd;
                  equal int 56 qd.Tolk_amd.Queue_desc.put_value;
                  (* sentinels in the two dwords before the ring end prove
                     the zero-fill really writes them *)
                  Mmio.write32 m (14 * 4) 0xDEADBEEFl;
                  Mmio.write32 m (15 * 4) 0xDEADBEEFl;
                  (* the device consumed the first packet; without this the
                     overrun spin would never let the wrap through *)
                  Mmio.write64 m (16 * 4) 28L;
                  let second = Cp.create dev in
                  Cp.copy second ~dest:dst ~src 0x100;
                  Cp.submit second qd;
                  equal int 0 (ring_dword m 14);
                  equal int 0 (ring_dword m 15);
                  equal (array int)
                    (Q.dwords (Cp.q second))
                    (ring_dwords m 7);
                  equal int 92 qd.Tolk_amd.Queue_desc.put_value;
                  equal int64 92L (Mmio.read64 m ((16 * 4) + 16))));
          test "a stream that cannot fit the ring is rejected" (fun () ->
              let module Cp = Tolk_amd.Copy_queue in
              with_map 4096 (fun m ->
                  let qd = queue_desc ~ring_dwords:8 m in
                  let dev = gfx1100 () in
                  let src = Buffer.make ~va:0x10000000n ~size:0 ~meta:() () in
                  let dst = Buffer.make ~va:0x20000000n ~size:0 ~meta:() () in
                  let cp = Cp.create dev in
                  Cp.copy cp ~dest:dst ~src 0x100;
                  Cp.submit cp qd;
                  equal int 28 qd.Tolk_amd.Queue_desc.put_value;
                  (* even with the whole ring consumed, the wrapped stream
                     would need the full ring: rejected before blocking *)
                  Mmio.write64 m (8 * 4) 28L;
                  raises_match is_invalid_arg (fun () -> Cp.submit cp qd)));
        ];
      group "Program"
        [
          test "load derives launch parameters from the descriptor" (fun () ->
              with_lib_alloc (fun alloc sizes _m ->
                  let dev = gfx1100 () in
                  let prg =
                    Program.load dev ~alloc ~props:lds64 ~name:"k"
                      (hsaco_fixture ())
                  in
                  (* the 0x110-byte image is padded to a whole page *)
                  equal (list int) [ 0x1000 ] !sizes;
                  equal string "k" prg.Program.name;
                  equal nativeint 0xA00000n (Buffer.va prg.lib_gpu);
                  equal int 0x2000 prg.group_segment_size;
                  equal int 256 prg.private_segment_size;
                  equal int 24 prg.kernargs_segment_size;
                  equal int 24 prg.kernargs_alloc_size;
                  let p = prg.params in
                  (* entry point: .rodata (0x40) + entry offset (0xC0) *)
                  equal nativeint 0xA00100n p.Tolk_amd.prog_addr;
                  (* rsrc1 gains the generation-11 privileged bit *)
                  equal int (0x1111 lor (1 lsl 20)) p.rsrc1;
                  (* rsrc2 gains the 512-byte lds granule count at bit 15:
                     0x2000 bytes -> 16 granules *)
                  equal int (0x2222 lor (0x10 lsl 15)) p.rsrc2;
                  equal int 0x3333 p.rsrc3;
                  is_true p.wave32;
                  is_true (not p.enable_private_segment_sgpr);
                  is_true (not p.enable_dispatch_ptr);
                  is_true (p.dev == dev)));
          test "load uploads the relocated image" (fun () ->
              with_lib_alloc (fun alloc _sizes m ->
                  let dev = gfx942 () in
                  let prg =
                    Program.load dev ~alloc ~props:lds64 ~name:"k"
                      (hsaco_fixture ~code_props:0 ())
                  in
                  (* no privileged bit outside generation 11 *)
                  equal int 0x1111 prg.Program.params.rsrc1;
                  is_true (not prg.params.wave32);
                  equal string "KERNCODE"
                    (Bytes.to_string (Mmio.read_bytes m ~off:0x100 ~len:8));
                  (* patch site .text + 8: (.rodata + 4) - site + addend *)
                  equal int64
                    (Int64.of_int (0x44 - 0x108 + 0x10))
                    (Mmio.read64 m 0x108);
                  (* the descriptor is uploaded unmodified; rsrc adjustments
                     live only in the parsed parameters *)
                  equal int32 0x1111l (Mmio.read32 m (0x40 + 48))));
          test "load reads the code-property bits" (fun () ->
              with_lib_alloc (fun alloc _sizes _m ->
                  let dev = gfx942 () in
                  let scratch_prg =
                    Program.load dev ~alloc ~props:lds64 ~name:"k"
                      (hsaco_fixture ~code_props:0x401 ())
                  in
                  is_true scratch_prg.Program.params.enable_private_segment_sgpr;
                  is_true scratch_prg.params.wave32;
                  let dp =
                    Program.load dev ~alloc ~props:lds64 ~name:"k"
                      (hsaco_fixture ~code_props:0x2 ())
                  in
                  is_true dp.Program.params.enable_dispatch_ptr;
                  (* dispatch-pointer kernels stage a 64-byte packet after
                     the arguments *)
                  equal int 24 dp.kernargs_segment_size;
                  equal int (24 + 64) dp.kernargs_alloc_size));
          test "load fails loudly before touching device memory" (fun () ->
              with_lib_alloc (fun alloc sizes _m ->
                  let dev = gfx1100 () in
                  let load ?(props = lds64) lib =
                    Program.load dev ~alloc ~props ~name:"k" lib
                  in
                  raises_match
                    (Exn.failure ~substring:".rodata section not found")
                    (fun () -> load (hsaco_fixture ~rodata_name:".rodat" ()));
                  raises_match (Exn.failure ~substring:"unknown AMD reloc 4")
                    (fun () -> load (hsaco_fixture ~reloc_type:4 ()));
                  raises_match
                    (Exn.failure ~substring:"undefined symbol k")
                    (fun () -> load (hsaco_fixture ~undefined_sym:true ()));
                  equal (list int) [] !sizes;
                  (* 16 lds granules against a 4 KiB limit (8 granules) *)
                  raises_match
                    (Exn.failure ~substring:"Too many resources requested")
                    (fun () ->
                      load
                        ~props:[ ("lds_size_in_kb", 4) ]
                        (hsaco_fixture ()))));
          test "free releases the image memory" (fun () ->
              with_lib_alloc (fun alloc _sizes _m ->
                  let prg =
                    Program.load (gfx1100 ()) ~alloc ~props:lds64 ~name:"k"
                      (hsaco_fixture ())
                  in
                  let freed = ref [] in
                  Program.free
                    ~free:(fun b -> freed := Buffer.va b :: !freed)
                    prg;
                  equal (list nativeint) [ 0xA00000n ] !freed));
        ];
      group "Scratch"
        [
          test "sizing grows the buffer and encodes the ring" (fun () ->
              let dev = gfx1100 ~scratch:(no_scratch ()) () in
              let props =
                [
                  ("simd_count", 192);
                  ("simd_per_cu", 2);
                  ("array_count", 12);
                  ("simd_arrays_per_engine", 2);
                  ("max_slots_scratch_cu", 32);
                ]
              in
              let allocs = ref [] and frees = ref [] in
              let alloc size =
                allocs := size :: !allocs;
                Buffer.make ~va:0x900000n ~size ~meta:() ()
              in
              let free b = frees := Buffer.size b :: !frees in
              Tolk_amd.ensure_has_local_memory dev ~props ~alloc ~free 256;
              (* 256 B/thread * 64 lanes * 32 slots * 96 CUs *)
              equal (list int) [ 0x3000000 ] !allocs;
              equal (list int) [] !frees;
              equal int 0x3000000 (Buffer.size dev.Tolk_amd.scratch);
              (* 3072 waves over 6 engines, 64 alignment granules each *)
              equal int (512 lor (64 lsl 12)) dev.tmpring_size;
              equal int 256 dev.max_private_segment_size;
              (* already covered: nothing happens *)
              Tolk_amd.ensure_has_local_memory dev ~props ~alloc ~free 128;
              equal (list int) [ 0x3000000 ] !allocs;
              (* growing frees the old buffer first *)
              Tolk_amd.ensure_has_local_memory dev ~props ~alloc ~free 512;
              equal (list int) [ 0x6000000; 0x3000000 ] !allocs;
              equal (list int) [ 0x3000000 ] !frees;
              equal int (512 lor (128 lsl 12)) dev.tmpring_size;
              equal int 512 dev.max_private_segment_size);
          test "per-thread sizes round up to the wave granule" (fun () ->
              let dev = gfx1100 ~scratch:(no_scratch ()) () in
              let props =
                [
                  ("simd_count", 192);
                  ("simd_per_cu", 2);
                  ("array_count", 12);
                  ("simd_arrays_per_engine", 2);
                  ("max_slots_scratch_cu", 32);
                ]
              in
              let alloc size = Buffer.make ~va:0x900000n ~size ~meta:() () in
              Tolk_amd.ensure_has_local_memory dev ~props ~alloc
                ~free:(fun _ -> ())
                99;
              (* 99 rounds to 100 B/thread (4-byte granule); 25 alignment
                 granules per wave *)
              equal int 19660800 (Buffer.size dev.Tolk_amd.scratch);
              equal int (512 lor (25 lsl 12)) dev.tmpring_size);
          test "generation 9 uses its alignment and die count" (fun () ->
              let dev = gfx942 ~scratch:(no_scratch ()) () in
              let props =
                [
                  ("simd_count", 1216);
                  ("simd_per_cu", 4);
                  ("array_count", 32);
                  ("simd_arrays_per_engine", 1);
                  ("max_slots_scratch_cu", 32);
                ]
              in
              let alloc size = Buffer.make ~va:0x900000n ~size ~meta:() () in
              let free _ = () in
              Tolk_amd.ensure_has_local_memory dev ~props ~alloc ~free 4;
              (* 4 rounds to 16 B/thread (1024/64 granule); one granule per
                 wave; 8 dies of 16 B * 64 * 32 * 38 *)
              equal int 0x980000 (Buffer.size dev.Tolk_amd.scratch);
              equal int (1216 lor (1 lsl 12)) dev.tmpring_size;
              Tolk_amd.ensure_has_local_memory dev ~props ~alloc ~free 512;
              equal int 0x13000000 (Buffer.size dev.scratch);
              equal int (1216 lor (32 lsl 12)) dev.tmpring_size);
          test "generation 12 encodes the wider ring field" (fun () ->
              let dev = gfx1200 ~scratch:(no_scratch ()) () in
              let props =
                [
                  ("simd_count", 32);
                  ("simd_per_cu", 2);
                  ("array_count", 4);
                  ("simd_arrays_per_engine", 2);
                  ("max_slots_scratch_cu", 32);
                ]
              in
              Tolk_amd.ensure_has_local_memory dev ~props
                ~alloc:(fun size -> Buffer.make ~va:0x900000n ~size ~meta:() ())
                ~free:(fun _ -> ())
                0x20000;
              (* 0x8000 granules per wave overflows the 15-bit field of
                 generation 11; generation 12 carries 18 bits *)
              equal int 0x100000000 (Buffer.size dev.Tolk_amd.scratch);
              equal int (256 lor (0x8000 lsl 12)) dev.tmpring_size);
          test "a failed grow falls back to the old size" (fun () ->
              let dev = gfx1100 ~scratch:(no_scratch ()) () in
              let props =
                [
                  ("simd_count", 192);
                  ("simd_per_cu", 2);
                  ("array_count", 12);
                  ("simd_arrays_per_engine", 2);
                  ("max_slots_scratch_cu", 32);
                ]
              in
              let allocs = ref [] and frees = ref [] in
              let alloc size =
                if size > 0x3000000 then failwith "no memory";
                allocs := size :: !allocs;
                Buffer.make ~va:0x900000n ~size ~meta:() ()
              in
              let free b = frees := Buffer.size b :: !frees in
              Tolk_amd.ensure_has_local_memory dev ~props ~alloc ~free 256;
              Tolk_amd.ensure_has_local_memory dev ~props ~alloc ~free 1024;
              (* the old buffer is gone, so its size is re-allocated and the
                 sizing state stays at the 256-byte segment *)
              equal (list int) [ 0x3000000; 0x3000000 ] !allocs;
              equal (list int) [ 0x3000000 ] !frees;
              equal int 0x3000000 (Buffer.size dev.Tolk_amd.scratch);
              equal int (512 lor (64 lsl 12)) dev.tmpring_size;
              equal int 256 dev.max_private_segment_size);
          test "a missing property is loud" (fun () ->
              let dev = gfx1100 ~scratch:(no_scratch ()) () in
              raises_match
                (Exn.failure ~substring:"missing device property")
                (fun () ->
                  Tolk_amd.ensure_has_local_memory dev ~props:[]
                    ~alloc:(fun size ->
                      Buffer.make ~va:0x900000n ~size ~meta:() ())
                    ~free:(fun _ -> ())
                    64));
        ];
      group "Dispatch"
        [
          test "a timed call brackets the launch and reports elapsed time"
            (fun () ->
              let module Cq = Tolk_amd.Compute_queue in
              with_map 8192 (fun m ->
                  let dev = gfx1100 () in
                  let qd = queue_desc ~ring_dwords:512 m in
                  let aux = (512 * 4) + 24 in
                  let slot_at off va =
                    Buffer.make ~va ~size:16
                      ~view:(Mmio.view m ~off ~size:16 ())
                      ~meta:() ()
                  in
                  let tl =
                    Signal.make ~is_timeline:true ~owner:dev
                      (slot_at aux 0x400000n)
                  in
                  let st =
                    Signal.make ~timestamp_divider:100.
                      (slot_at (aux + 16) 0x410000n)
                  in
                  let en =
                    Signal.make ~timestamp_divider:100.
                      (slot_at (aux + 32) 0x410010n)
                  in
                  let kernargs =
                    Kernargs.create
                      (Buffer.make ~va:0x300000n ~size:256
                         ~view:(Mmio.view m ~off:(aux + 64) ~size:256 ())
                         ~meta:() ())
                  in
                  let prg =
                    {
                      Program.params = amd_prog dev;
                      name = "k";
                      lib_gpu =
                        Buffer.make ~va:0x100000n ~size:0x1000 ~meta:() ();
                      group_segment_size = 0;
                      private_segment_size = 0;
                      kernargs_segment_size = 24;
                      kernargs_alloc_size = 24;
                    }
                  in
                  (* completion and clock captures the device would write:
                     the timeline reaches the signaled value, and the raw
                     100 MHz counters span 250 us *)
                  Mmio.write64 m aux 0x43L;
                  Mmio.write64 m (aux + 16 + 8) 10000L;
                  Mmio.write64 m (aux + 32 + 8) 35000L;
                  let elapsed =
                    Program.call prg ~kernargs ~queue:qd ~timeline:tl
                      ~timeline_value:0x43 ~wait:(st, en)
                      ~bufs:[| 0x1000n; 0x2000n |] ~vals:[| 7 |]
                      ~global_size:(4, 3, 2) ~local_size:(8, 4, 1) ()
                  in
                  (match elapsed with
                  | Some dt -> equal (float 1e-12) 0.00025 dt
                  | None -> fail "expected an execution time");
                  equal int 0x43 (Signal.value tl);
                  (* the argument slot: two addresses then one value *)
                  equal bytes
                    (Bytes.of_string
                       "\x00\x10\x00\x00\x00\x00\x00\x00\
                        \x00\x20\x00\x00\x00\x00\x00\x00\
                        \x07\x00\x00\x00")
                    (Mmio.read_bytes m ~off:(aux + 64) ~len:20);
                  let expected =
                    let cq = Cq.create dev in
                    Cq.wait cq ~value:0x42 tl;
                    Cq.memory_barrier cq;
                    Cq.timestamp cq st;
                    Cq.exec cq (amd_prog dev)
                      ~kernargs:(Buffer.make ~va:0x300000n ~size:24 ~meta:() ())
                      ~global_size:(4, 3, 2) ~local_size:(8, 4, 1);
                    Cq.timestamp cq en;
                    Cq.signal cq ~value:0x43 tl;
                    Q.dwords (Cq.q cq)
                  in
                  equal (array int) expected
                    (ring_dwords m (Array.length expected));
                  equal int (Array.length expected)
                    qd.Tolk_amd.Queue_desc.put_value;
                  equal int64
                    (Int64.of_int (Array.length expected))
                    (Mmio.read64 m ((512 * 4) + 16))));
          test "call without wait neither times nor blocks" (fun () ->
              let module Cq = Tolk_amd.Compute_queue in
              with_map 8192 (fun m ->
                  let dev = gfx1100 () in
                  let qd = queue_desc ~ring_dwords:512 m in
                  let aux = (512 * 4) + 24 in
                  let tl =
                    Signal.make ~is_timeline:true ~owner:dev
                      (Buffer.make ~va:0x400000n ~size:16
                         ~view:(Mmio.view m ~off:aux ~size:16 ())
                         ~meta:() ())
                  in
                  let kernargs =
                    Kernargs.create
                      (Buffer.make ~va:0x300000n ~size:64
                         ~view:(Mmio.view m ~off:(aux + 16) ~size:64 ())
                         ~meta:() ())
                  in
                  let prg =
                    {
                      Program.params = amd_prog dev;
                      name = "k";
                      lib_gpu =
                        Buffer.make ~va:0x100000n ~size:0x1000 ~meta:() ();
                      group_segment_size = 0;
                      private_segment_size = 0;
                      kernargs_segment_size = 24;
                      kernargs_alloc_size = 24;
                    }
                  in
                  let r =
                    Program.call prg ~kernargs ~queue:qd ~timeline:tl
                      ~timeline_value:1 ~bufs:[||] ~vals:[||]
                      ~global_size:(1, 1, 1) ~local_size:(1, 1, 1) ()
                  in
                  is_true (r = None);
                  let expected =
                    let cq = Cq.create dev in
                    Cq.wait cq ~value:0 tl;
                    Cq.memory_barrier cq;
                    Cq.exec cq (amd_prog dev)
                      ~kernargs:(Buffer.make ~va:0x300000n ~size:24 ~meta:() ())
                      ~global_size:(1, 1, 1) ~local_size:(1, 1, 1);
                    Cq.signal cq ~value:1 tl;
                    Q.dwords (Cq.q cq)
                  in
                  equal (array int) expected
                    (ring_dwords m (Array.length expected))));
          test "call rejects dispatch-pointer programs before staging"
            (fun () ->
              with_map 4096 (fun m ->
                  let dev = gfx1100 () in
                  let qd = queue_desc ~ring_dwords:16 m in
                  let aux = (16 * 4) + 24 in
                  let tl =
                    Signal.make
                      (Buffer.make ~va:0x400000n ~size:16
                         ~view:(Mmio.view m ~off:aux ~size:16 ())
                         ~meta:() ())
                  in
                  let kernargs =
                    Kernargs.create
                      (Buffer.make ~va:0x300000n ~size:64
                         ~view:(Mmio.view m ~off:(aux + 16) ~size:64 ())
                         ~meta:() ())
                  in
                  let prog params kernargs_alloc_size =
                    {
                      Program.params;
                      name = "k";
                      lib_gpu =
                        Buffer.make ~va:0x100000n ~size:0x1000 ~meta:() ();
                      group_segment_size = 0;
                      private_segment_size = 0;
                      kernargs_segment_size = 24;
                      kernargs_alloc_size;
                    }
                  in
                  raises_match is_invalid_arg (fun () ->
                      Program.call
                        (prog (amd_prog ~dispatch_ptr:true dev) 88)
                        ~kernargs ~queue:qd ~timeline:tl ~timeline_value:1
                        ~bufs:[||] ~vals:[||] ~global_size:(1, 1, 1)
                        ~local_size:(1, 1, 1) ());
                  (* nothing was staged or submitted *)
                  equal nativeint 0x300000n
                    (Buffer.va (Kernargs.alloc kernargs 8));
                  equal int 0 qd.Tolk_amd.Queue_desc.put_value;
                  (* the timeline wait needs a value to wait on *)
                  raises_match is_invalid_arg (fun () ->
                      Program.call
                        (prog (amd_prog dev) 24)
                        ~kernargs ~queue:qd ~timeline:tl ~timeline_value:0
                        ~bufs:[||] ~vals:[||] ~global_size:(1, 1, 1)
                        ~local_size:(1, 1, 1) ())));
        ];
      group "Kfd_iface"
        [
          test "construction fails cleanly without the driver" (fun () ->
              if Sys.file_exists "/dev/kfd" then
                skip ~reason:"the AMD kernel driver is present" ()
              else begin
                raises_match
                  (function Failure _ -> true | _ -> false)
                  (fun () -> Tolk_amd.Kfd_iface.count ());
                raises_match
                  (function Failure _ -> true | _ -> false)
                  (fun () -> Tolk_amd.Kfd_iface.create ~device_id:0)
              end);
        ];
      group "Pci_iface"
        [
          test "the bus scan admits exactly the allowlisted ids" (fun () ->
              with_fake_sysfs
                [
                  (* Navi 31 and an RDNA4 part match; an iGPU, the GPU's
                     audio function, and a foreign vendor do not *)
                  ("0000:03:00.0", 0x1002, 0x744c);
                  ("0000:02:00.0", 0x1002, 0x7550);
                  ("0000:01:00.0", 0x1002, 0x164e);
                  ("0000:03:00.1", 0x1002, 0xab30);
                  ("0000:04:00.0", 0x10de, 0x744c);
                ]
                (fun sysfs ->
                  equal (list string)
                    [ "0000:02:00.0"; "0000:03:00.0" ]
                    (Tolk_hcq.System.pci_scan_bus ~sysfs
                       ~vendor:Pci_iface.vendor Pci_iface.pci_ids)));
          test "every allowlisted id is admitted" (fun () ->
              let ids = List.concat_map snd Pci_iface.pci_ids in
              with_fake_sysfs
                (List.mapi
                   (fun i id -> (Printf.sprintf "0000:%02x:00.0" i, 0x1002, id))
                   ids)
                (fun sysfs ->
                  equal int (List.length ids)
                    (List.length
                       (Tolk_hcq.System.pci_scan_bus ~sysfs
                          ~vendor:Pci_iface.vendor Pci_iface.pci_ids))));
          test "props synthesis from the discovered geometry" (fun () ->
              (* an RDNA3-style v2 geometry table *)
              let props =
                Pci_iface.compute_props
                  ~gc_info:
                    (Amdev.Gc_info_v2
                       {
                         num_se = 2;
                         num_cu_per_sh = 8;
                         num_sh_per_se = 1;
                         max_scratch_slots_per_cu = 32;
                         max_waves_per_simd = 16;
                         lds_size = 64;
                       })
                  ~gc_ver:(11, 0, 2) ~xccs:1
              in
              equal
                (list (pair string int))
                [
                  ("cu_per_simd_array", 8); ("simd_count", 32);
                  ("simd_per_cu", 2); ("array_count", 2);
                  ("max_slots_scratch_cu", 32); ("max_waves_per_simd", 16);
                  ("simd_arrays_per_engine", 1); ("lds_size_in_kb", 64);
                  ("num_xcc", 1); ("gfx_target_version", 110002);
                ]
                props;
              (* a v1 geometry table counts compute units in work-group
                 processor pairs *)
              let props =
                Pci_iface.compute_props
                  ~gc_info:
                    (Amdev.Gc_info_v1
                       {
                         num_se = 4;
                         num_wgp0_per_sa = 2;
                         num_wgp1_per_sa = 1;
                         num_sa_per_se = 2;
                         max_scratch_slots_per_cu = 32;
                         max_waves_per_simd = 16;
                         lds_size = 64;
                       })
                  ~gc_ver:(12, 0, 1) ~xccs:1
              in
              equal int 6 (List.assoc "cu_per_simd_array" props);
              equal int 8 (List.assoc "array_count" props);
              equal int 96 (List.assoc "simd_count" props);
              equal int 120001 (List.assoc "gfx_target_version" props);
              (* the one gfx-version quirk: 9.4.3 reports 9.4.2 *)
              let props =
                Pci_iface.compute_props
                  ~gc_info:
                    (Amdev.Gc_info_v2
                       {
                         num_se = 2;
                         num_cu_per_sh = 8;
                         num_sh_per_se = 1;
                         max_scratch_slots_per_cu = 32;
                         max_waves_per_simd = 16;
                         lds_size = 64;
                       })
                  ~gc_ver:(9, 4, 3) ~xccs:8
              in
              equal int 90402 (List.assoc "gfx_target_version" props);
              equal int 8 (List.assoc "num_xcc" props));
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
      group "Device"
        [
          test "create opens the device and synchronize completes" (fun () ->
              let device = amd_device () in
              equal string "AMD" (Tolk.Device.name device);
              Tolk.Device.synchronize device);
          test "compiles and runs one kernel" (fun () ->
              let device = amd_device () in
              (match Compiler_amd.version () with
              | exception Failure msg -> skip ~reason:msg ()
              | _ -> ());
              let spec =
                Tolk.Device.compile_program device ~name:"amd_add_one"
                  (increment_program ())
              in
              let dst = i32_buf device [ 0 ] in
              let src = i32_buf device [ 41 ] in
              let runner = Tolk.Realize.Compiled_runner.create ~device spec in
              (match
                 Tolk.Realize.Compiled_runner.call runner [ dst; src ] []
                   ~wait:true ~timeout:None
               with
              | Some tm -> is_true (tm >= 0.0)
              | None -> fail "expected a device execution time");
              Tolk.Device.synchronize device;
              equal (list int) [ 42 ] (read_i32 dst));
        ];
    ]
