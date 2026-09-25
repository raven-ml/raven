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
module Submission = Tolk_hcq.Hcq.Submission
module Timeline = Tolk_hcq.Hcq.Timeline
module Kernargs = Tolk_hcq.Hcq.Kernargs
module Compiler_amd = Tolk_amd.Compiler_amd
module Program = Tolk_amd.Program
module Pci_iface = Tolk_amd.Pci_iface
module Amdev = Tolk_amd.Amdev

let argument_layout nbufs dtypes =
  let open Tolk_uop in
  let arg slot addrspace dtype : Tiny_elf.argument =
    { name = None; slot; addrspace; dtype; shape = [] } in
  Tiny_elf.layout
    (List.init nbufs (fun slot -> arg slot Dtype.Global Dtype.uint8)
     @ List.mapi (fun i dtype -> arg (nbufs + i) Dtype.Alu dtype) dtypes)

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
    kernel_object = 0x100040n;
    group_segment_size = 0;
    private_segment_size = 0;
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
    Tolk_amd.Queue_desc.aql = None;
    ring = Mmio.view m ~off:0 ~size:ring_bytes ();
    read_ptr = Mmio.view m ~off:ring_bytes ~size:8 ();
    write_ptr = Mmio.view m ~off:(ring_bytes + 8) ~size:8 ();
    doorbell = Mmio.view m ~off:(ring_bytes + 16) ~size:8 ();
    hdp_flush = None;
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
  if Sys.win32 then
    skip ~reason:"sysfs device names contain ':', not a Windows file name" ();
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

let queue_fixture ?(timeout_ms = 30000) ?(dispatch_ptr = false) ?(scratch = 256) ?(aql = false) ?(multi = false) ~copies () =
  let open Tolk in
  let open Tolk_uop in
  let device_name = "AMD:queue-compilation" in
  let timeline = ref None in
  let submission = Submission.create () in
  let host = Tolk_cpu.create "CPU" in
  let parameter slot = U.param ~slot ~dtype:D.int32 ~shape:(U.const_int 16)
      ~device:(U.Single device_name) () in
  let output = U.param ~slot:0 ~dtype:D.int32 ~shape:(U.const_int 16) () in
  let small = U.variable ~param:true ~name:"small" ~min_val:(-128) ~max_val:127 ~dtype:D.int8 () in
  let count = U.variable ~param:true ~name:"count" ~min_val:1 ~max_val:16 ~dtype:D.int64 () in
  let value = U.cast ~src:small ~dtype:D.int32 in
  let store = U.store ~dst:(U.index ~ptr:output ~idxs:[U.const_int 0] ()) ~value () in
  let lib = hsaco_fixture ~private_seg:scratch ~code_props:(if dispatch_ptr then 0x402 else 0x400) () in
  let spec = Program_spec.of_program ~name:"queue_fixture" ~src:"" ~device:device_name ~lib
      [output; small; count; value; store] in
  let info = {(Program_spec.program_info spec) with global_size = [U.Launch_sym count];
    local_size = [U.Launch_int 1]} in
  let program = U.program ~sink:(U.sink [store]) ~linear:(U.linear (Program_spec.program spec))
      ~source:(U.source "") ~binary:(U.binary (Bytes.to_string lib)) ~info () in
  let call = U.call ~body:program ~args:[parameter 0]
      ~info:{grad_fxn = None; name = None; precompile = false; precompile_backward = false;
        dtype = D.void; aux = None} in
  let props = ["lds_size_in_kb", 64; "simd_count", 192; "simd_per_cu", 2;
    "array_count", 12; "simd_arrays_per_engine", 2; "max_slots_scratch_cu", 32] in
  let hw = if multi then gfx942 () else gfx1100 () in
  let hw = {hw with Tolk_amd.is_aql = aql} in
  let queue = Device.{timestamp_divider = 100.; completion = (fun () ->
      match !timeline with
      | None -> fun () -> ()
      | Some tl -> let value = Timeline.submitted tl in
          fun () -> Timeline.guarded_wait tl (fun () -> Signal.wait tl.Timeline.timeline value)); prepare = (fun () -> Option.iter Timeline.prepare !timeline; Submission.prepare ~timeout_ms submission);
    host = "CPU"; copy = (fun _ -> true);
    encode = Tolk_amd.Encoded_queue.encode hw ~props ~name:device_name
        ~compute_ring_size:4096 ~copy_ring_size:(Some 4096);
    lower = Tolk_amd.Encoded_queue.lower device_name;
    compile = Codegen.to_program ~optimize:false host (Device.renderer host)} in
  let allocator = Device.Allocator.Pack (Storage.Host_allocator.make ~synchronize:(fun () -> ())) in
  let renderer_set = Device.Renderer_set.make ~device:device_name
      ["CLANG", (fun target -> Renderer.with_target target (Device.renderer host))] in
  let buffers = Hashtbl.create 16 in
  let bufferize u = match U.as_param u with
    | Some {param = {allocation = Some ("hcq_submission", _); _}; _} ->
        Some (Submission.buffer submission)
    | _ ->
    let buffer = Device.Buffer.create ~device:device_name ~size:(U.max_numel u)
        ~dtype:(U.dtype u) allocator in
    Device.Buffer.ensure_allocated buffer;
    Option.iter (fun tag -> Hashtbl.replace buffers tag buffer) (U.node_tag u);
    if U.node_tag u = Some "timeline" then begin
      let address = Device.Buffer.addr buffer in
      let view = Mmio.make ~addr:address ~size:16 in
      let raw = Tolk_hcq.Hcq.Buffer.make ~va:address ~size:16 ~view ~meta:() () in
      timeline := Some {Timeline.timeline = Signal.make ~is_timeline:true raw;
        error_state = None; bounce = [||]; bounce_timeline = [||]; bounce_next = 0;
        on_hang = (fun () -> fail "unexpected fixture hang")}
    end;
    (match U.as_param u with
     | Some {param = {allocation = Some ("cfunc", data); _}; _} ->
         let libs, symbol = (Marshal.from_string data 0 : string list * string) in
         equal (list string) [] libs;
         let bytes = Bytes.create 8 in
         Bytes.set_int64_le bytes 0 (Int64.of_nativeint (Submission.symbol symbol));
         Device.Buffer.copyin buffer bytes
     | Some {param = {allocation = Some ("amd_image", data); _}; _} ->
         let requested, image = (Marshal.from_string data 0 : int * string) in
         equal int scratch requested;
         Device.Buffer.copyin buffer (Bytes.of_string image)
     | _ -> ());
    Some buffer in
  let device = Device.make ~name:device_name ~allocator ~renderer_set ~runtime:(Device.runtime host)
      ~synchronize:(fun () -> ()) ~queue ~bufferize () in
  let calls = if copies then [U.store_call ~dst:(parameter 0) ~src:(parameter 1);
      call; U.store_call ~dst:(parameter 2) ~src:(parameter 0)] else [call] in
  Hcq2.compile (U.linear calls), device, host, buffers, submission

let compile_queue ~copies =
  let compiled, _, _, _, _ = queue_fixture ~copies () in compiled

let execute_queue ~copies ~dispatch_ptr ~scratch =
  let open Tolk in
  let compiled, device, host, buffers, submission = queue_fixture ~copies ~dispatch_ptr ~scratch () in
  let binding = Realize.Buffers.create () in
  let linked = Realize.link_linear binding compiled in
  let get tag = Hashtbl.find buffers tag in
  let set_word tag value =
    let bytes = Bytes.create 8 in
    Bytes.set_int64_le bytes 0 (Int64.of_int value);
    Device.Buffer.copyin (get tag) bytes in
  let word tag = Int64.to_int (Bytes.get_int64_le (Device.Buffer.as_bytes (get tag)) 0) in
  let put = (1 lsl 32) + 1022 in
  set_word "write_ptr_compute" put;
  set_word "read_ptr_compute" 1022;
  if copies then begin
    set_word "write_ptr_copy" 4088; set_word "read_ptr_copy" 4088
  end;
  let to_program device = Codegen.to_program device (Device.renderer device) in
  List.iteri (fun replay (small, count) ->
      if replay = 1 then begin
        let bytes = Bytes.make 16 '\000' in
        Bytes.set_int64_le bytes 0 0x80000000L;
        Bytes.set_int64_le bytes 8 0x80000000L;
        Device.Buffer.copyin (get "timeline") bytes
      end;
      let inputs = Array.init (if copies then 3 else 1) (fun _ ->
          i32_buf device (List.init 16 Fun.id)) in
      Realize.run_linear ~device ~to_program binding ~jit:true
        ~var_vals:["small", small; "count", count] ~input_uops:(Array.map U.from_buffer inputs) linked;
      Submission.check submission;
      equal int (put + (replay + 1) * 4) (word "write_ptr_compute");
      equal int (word "write_ptr_compute") (word "doorbell_compute");
      let arena = Device.Buffer.as_bytes (get "kernargs") in
      equal int small (Bytes.get_int8 arena 8);
      equal int64 (Int64.of_int count) (Bytes.get_int64_le arena 16);
      equal int64 (Int64.of_nativeint (Device.Buffer.addr ~device:(Device.name device)
          inputs.(0))) (Bytes.get_int64_le arena 0);
      if dispatch_ptr then begin
        equal int 0x31502 (Int32.to_int (Bytes.get_int32_le arena 24));
        equal int count (Int32.to_int (Bytes.get_int32_le arena 36));
        equal int scratch (Int32.to_int (Bytes.get_int32_le arena 48))
      end;
      if copies then begin
        is_true (word "write_ptr_copy" > 4096);
        equal int (word "write_ptr_copy") (word "doorbell_copy");
        let ring = Device.Buffer.as_bytes (get "ring_copy") in
        equal int32 0l (Bytes.get_int32_le ring 4088);
        equal int32 0l (Bytes.get_int32_le ring 4092)
      end;
      (* Simulate completion; these tests execute submission code, not GPU packets. *)
      let timeline = Device.Buffer.as_bytes (get "timeline") in
      equal int64 (if replay = 0 then 1L else 0x100000001L)
        (Bytes.get_int64_le timeline 8);
      Bytes.set_int64_le timeline 0 (Bytes.get_int64_le timeline 8);
      Device.Buffer.copyin (get "timeline") timeline;
      set_word "read_ptr_compute" (word "write_ptr_compute");
      if copies then set_word "read_ptr_copy" (word "write_ptr_copy")) [(-17, 3); (29, 11)];
  is_true (Hashtbl.mem buffers "scratch")

let execute_aql_queue ~multi =
  let open Tolk in
  let compiled, device, host, buffers, submission = queue_fixture ~aql:true ~multi ~copies:false () in
  let binding = Realize.Buffers.create () in
  let linked = Realize.link_linear binding compiled in
  let get tag = Hashtbl.find buffers tag in
  let put tag value =
    let bytes = Bytes.create 8 in
    Bytes.set_int64_le bytes 0 (Int64.of_int value);
    Device.Buffer.copyin (get tag) bytes in
  put "write_ptr_compute" 62;
  put "read_ptr_compute" 62;
  let input = i32_buf device (List.init 16 Fun.id) in
  Realize.run_linear ~device ~to_program:(fun device -> Codegen.to_program device (Device.renderer device)) binding
    ~jit:true ~var_vals:["small", -17; "count", 3] ~input_uops:[|U.from_buffer input|] linked;
  Submission.check submission;
  let word tag = Int64.to_int (Bytes.get_int64_le (Device.Buffer.as_bytes (get tag)) 0) in
  equal int 65 (word "write_ptr_compute");
  equal int 64 (word "doorbell_compute");
  let ring = Device.Buffer.as_bytes (get "ring_compute") in
  equal int32 0x11500l (Bytes.get_int32_le ring (62 * 64));
  equal int32 0x31502l (Bytes.get_int32_le ring (63 * 64));
  equal int32 0x11500l (Bytes.get_int32_le ring 0);
  let packet = 63 * 64 in
  equal int32 0x10001l (Bytes.get_int32_le ring (packet + 4));
  equal int32 3l (Bytes.get_int32_le ring (packet + 12));
  equal int32 256l (Bytes.get_int32_le ring (packet + 24));
  equal int64 (Int64.add (Int64.of_nativeint (Device.Buffer.addr (get "program"))) 0x40L)
    (Bytes.get_int64_le ring (packet + 32));
  equal int64 (Int64.of_nativeint (Device.Buffer.addr (get "kernargs")))
    (Bytes.get_int64_le ring (packet + 40));
  let arena = Device.Buffer.as_bytes (get "kernargs") in
  equal int (-17) (Bytes.get_int8 arena 8);
  let stream = Device.Buffer.as_bytes (get "cmdbuf_compute") in
  if multi then equal int32 0x1000008l (Bytes.get_int32_le stream (Bytes.length stream - 36));
  let timeline = Device.Buffer.as_bytes (get "timeline") in
  Bytes.set_int64_le timeline 0 (Bytes.get_int64_le timeline 8);
  Device.Buffer.copyin (get "timeline") timeline;
  let rebound = i32_buf device (List.init 16 (fun i -> i + 1)) in
  Realize.run_linear ~device ~to_program:(fun device -> Codegen.to_program device (Device.renderer device)) binding
    ~jit:true ~var_vals:["small", 5; "count", 9] ~input_uops:[|U.from_buffer rebound|] linked;
  Submission.check submission;
  equal int 68 (word "write_ptr_compute");
  equal int 67 (word "doorbell_compute");
  equal int32 9l (Bytes.get_int32_le (Device.Buffer.as_bytes (get "ring_compute")) (2 * 64 + 12));
  let args = Device.Buffer.as_bytes (get "kernargs") in
  equal int 5 (Bytes.get_int8 args 8);
  equal int64 (Int64.of_nativeint (Device.Buffer.addr rebound)) (Bytes.get_int64_le args 0)

let direct_aql_queue () =
  with_map 0x3000 (fun m ->
      let dev = { (gfx1100 ()) with Tolk_amd.is_aql = true } in
      let base = queue_desc ~ring_dwords:128 m in
      let aql = Tolk_amd.Queue_desc.{descriptor = Mmio.view m ~off:0x300 ~size:0x100 ();
        commands = Mmio.view m ~off:0x1000 ~size:0x1000 (); address = 0xabcdef000n;
        allocator = Tolk.Bump.create ~size:0x1000 ~wrap:true ()} in
      let queue = {base with Tolk_amd.Queue_desc.aql = Some aql} in
      Mmio.write64 queue.write_ptr 0 7L;
      let cq = Tolk_amd.Compute_queue.create dev in
      let sg = Signal.make (Buffer.make ~va:0x987000n ~size:16
        ~view:(Mmio.view m ~off:0x500 ~size:16 ()) ~meta:() ()) in
      Tolk_amd.Compute_queue.wait cq ~value:7 sg;
      let args = Buffer.make ~va:0x777000n ~size:32 ~meta:() () in
      Tolk_amd.Compute_queue.exec cq (amd_prog dev) ~kernargs:args ~global_size:(3,2,1) ~local_size:(2,1,1);
      Tolk_amd.Compute_queue.signal cq ~value:8 sg;
      Tolk_amd.Compute_queue.submit cq queue;
      equal int64 10L (Mmio.read64 queue.write_ptr 0);
      equal int64 9L (Mmio.read64 queue.doorbell 0);
      equal int32 0x11500l (Mmio.read32 queue.ring (7 * 64));
      equal int64 0xabcdef000L (Mmio.read64 queue.ring (7 * 64 + 8));
      equal int32 0x31502l (Mmio.read32 queue.ring 0);
      equal int32 6l (Mmio.read32 queue.ring 12);
      equal int64 0x100040L (Mmio.read64 queue.ring 32);
      equal int64 0x777000L (Mmio.read64 queue.ring 40);
      equal int32 0x11500l (Mmio.read32 queue.ring 64))

let queue_timeout () =
  let open Tolk in
  let compiled, device, host, buffers, submission = queue_fixture ~timeout_ms:5 ~copies:false () in
  let binding = Realize.Buffers.create () in
  let linked = Realize.link_linear binding compiled in
  let input = i32_buf device (List.init 16 Fun.id) in
  let run () = Realize.run_linear ~device
      ~to_program:(fun device -> Codegen.to_program device (Device.renderer device)) binding ~jit:true
      ~var_vals:["small", 7; "count", 3] ~input_uops:[|U.from_buffer input|] linked in
  run ();
  let doorbell () = Device.Buffer.as_bytes (Hashtbl.find buffers "doorbell_compute") in
  let published = doorbell () in
  let protected = List.map (fun tag -> tag, Device.Buffer.as_bytes (Hashtbl.find buffers tag))
      ["timeline"; "slots"; "kernargs"; "cmdbuf_compute"] in
  run ();
  raises_match (Exn.failure ~substring:"HCQ submission timed out")
    (fun () -> Submission.check submission);
  equal bytes published (doorbell ());
  List.iter (fun (tag, before) -> equal ~msg:tag bytes before
      (Device.Buffer.as_bytes (Hashtbl.find buffers tag))) protected;
  raises_match (Exn.failure ~substring:"HCQ submission timed out") run

let queue_full () =
  let open Tolk in
  let compiled, device, host, buffers, submission = queue_fixture ~timeout_ms:5 ~copies:false () in
  let binding = Realize.Buffers.create () in
  let linked = Realize.link_linear binding compiled in
  let ring = Hashtbl.find buffers "ring_compute" in
  let before = Device.Buffer.as_bytes ring in
  let pointer = Hashtbl.find buffers "write_ptr_compute" in
  let initial = Bytes.create 8 in
  Bytes.set_int64_le initial 0 1021L;
  Device.Buffer.copyin pointer initial;
  let input = i32_buf device (List.init 16 Fun.id) in
  Realize.run_linear ~device
    ~to_program:(fun device -> Codegen.to_program device (Device.renderer device)) binding ~jit:true
    ~var_vals:["small", 7; "count", 3] ~input_uops:[|U.from_buffer input|] linked;
  raises_match (Exn.failure ~substring:"HCQ submission timed out")
    (fun () -> Submission.check submission);
  equal bytes before (Device.Buffer.as_bytes ring);
  equal int64 1021L (Bytes.get_int64_le (Device.Buffer.as_bytes pointer) 0)

let () =
  run "Amd_runtime"
    [ group "AQL"
        [test "compiled single-XCC packets wrap in dispatch units" (fun () -> execute_aql_queue ~multi:false);
         test "compiled multi-XCC completion is predicated after dispatch" (fun () -> execute_aql_queue ~multi:true);
         test "direct packets use GPU indirect addresses and the shared producer" direct_aql_queue];
      group "Compiled queues" [
        test "a replay timeout suppresses publication and latches failure" queue_timeout;
        test "a full ring times out without overwriting unread commands" queue_full;
        test "executes wrapped compute submissions and patches replay arguments" (fun () ->
            execute_queue ~copies:false ~dispatch_ptr:false ~scratch:256);
        test "executes SDMA wrapping with dispatch packets and minimum scratch" (fun () ->
            execute_queue ~copies:true ~dispatch_ptr:true ~scratch:0);
        test "compiles symbolic launch dimensions and mixed-width arguments" (fun () ->
            let compiled = compile_queue ~copies:false in
            let call = Option.get (U.as_call (U.without_after (List.hd (U.children compiled)))) in
            let object_ = U.to_elf call.body in
            is_true (Bytes.length object_.lib > 0);
            let scalars = List.filter_map (fun (a : Tolk_uop.Tiny_elf.argument) ->
                if a.addrspace = D.Alu then Some (D.to_string a.dtype) else None) object_.signature in
            equal (list string) ["i64"; "i8"] (List.sort String.compare scalars));
        test "compiles dependencies between SDMA and compute" (fun () ->
            let compiled = compile_queue ~copies:true in
            equal int 1 (List.length (U.children compiled));
            match U.arg (U.without_after (List.hd (U.children compiled))) with
            | U.Arg.Call_info {aux = Some info; _} -> equal int 3 (List.length info.accesses)
            | _ -> fail "queue lost argument access metadata");
      ];
      group "File_io"
        [
          test "opens and closes a file" (fun () ->
              let fd = File_io.openfile Filename.null ~flags:File_io.o_rdonly in
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
          test "view ranges reject integer overflow" (fun () ->
              let m = Mmio.make ~addr:0n ~size:64 in
              raises_match is_invalid_arg (fun () ->
                  Mmio.view m ~off:max_int ~size:4 ());
              raises_match is_invalid_arg (fun () ->
                  Mmio.view m ~off:32 ~size:max_int ());
              let large = Mmio.make ~addr:0n ~size:max_int in
              equal int 4 (Mmio.size (Mmio.view large ~off:(max_int - 4) ~size:4 ()));
              equal int 0 (Mmio.size (Mmio.view large ~off:max_int ~size:0 ())));
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
          test "offset ranges reject integer overflow" (fun () ->
              let buf = Buffer.make ~va:0n ~size:64 ~meta:() () in
              raises_match is_invalid_arg (fun () ->
                  Buffer.offset buf ~off:max_int ~size:4 ());
              raises_match is_invalid_arg (fun () ->
                  Buffer.offset buf ~off:32 ~size:max_int ());
              let large = Buffer.make ~va:0n ~size:max_int ~meta:() () in
              equal int 4 (Buffer.size (Buffer.offset large ~off:(max_int - 4) ~size:4 ()));
              equal int 0 (Buffer.size (Buffer.offset large ~off:max_int ~size:0 ())));
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
          test "direct submissions observe the counter written by compiled submission" (fun () ->
              with_map 4096 (fun m ->
                  let tl = {
                    Timeline.timeline = Signal.make ~is_timeline:true (slot_buf m);

                    error_state = None; bounce = [||]; bounce_timeline = [||];
                    bounce_next = 0; on_hang = (fun () -> fail "unexpected hang");
                  } in
                  equal int 0 (Timeline.submitted tl);
                  equal int 1 (Timeline.next_timeline tl);
                  equal int64 1L (Mmio.read64 m 8);
                  Mmio.write64 m 8 37L;
                  equal int 38 (Timeline.next_timeline tl);
                  Signal.set_value tl.Timeline.timeline 38;
                  Timeline.synchronize tl;
                  equal int64 38L (Mmio.read64 m 8)));
          test "rollover retains the signal address and host fence epochs" (fun () ->
              with_map 4096 (fun m ->
                  let signal = Signal.make ~is_timeline:true (slot_buf m) in
                  let tl = {Timeline.timeline = signal; error_state = None;
                    bounce = [||]; bounce_timeline = [|17|]; bounce_next = 0;
                    on_hang = (fun () -> fail "unexpected hang")} in
                  List.iter (fun epoch ->
                      let end_ = (epoch lsl 32) + (1 lsl 31) in
                      Mmio.write64 m 8 (Int64.of_int end_);
                      Signal.set_value signal end_;
                      Timeline.prepare tl;
                      is_true (tl.Timeline.timeline == signal);
                      let next = ((epoch + 1) lsl 32) + 1 in
                      equal int next (Timeline.next_timeline tl);
                      (* AMD/SDMA write only the low dword; the CPU retains the epoch. *)
                      Mmio.write32 m 0 1l;
                      equal int next (Signal.value signal);
                      Timeline.synchronize tl;
                      equal int 17 tl.Timeline.bounce_timeline.(0)) [0; 1]));
          test "a stalled wait folds the hang report into the timeout" (fun () ->
              with_map 4096 (fun m ->
                  let tl =
                    {
                      Timeline.timeline = Signal.make ~value:1 (slot_buf m);

                      error_state = None;
                      bounce = [||];
                      bounce_timeline = [||];
                      bounce_next = 0;
                      on_hang = (fun () -> failwith "MMU fault: 0xdead");
                    }
                  in
                  Mmio.write64 m 8 2L;
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
                  let a = Kernargs.alloc ~wait:(fun () -> ()) k 24 in
                  equal nativeint 0x300000n (Buffer.va a);
                  equal int 24 (Buffer.size a);
                  let b = Kernargs.alloc ~wait:(fun () -> ()) k 8 in
                  equal nativeint 0x300018n (Buffer.va b);
                  let c = Kernargs.alloc ~wait:(fun () -> ()) k 4 in
                  equal nativeint 0x300020n (Buffer.va c);
                  let d = Kernargs.alloc ~wait:(fun () -> ()) k 8 in
                  equal nativeint 0x300028n (Buffer.va d)));
          test "write_args lays out addresses then values" (fun () ->
              with_map 4096 (fun m ->
                  let root =
                    Buffer.make ~va:0x300000n ~size:4096 ~view:m ~meta:() ()
                  in
                  let slot = Kernargs.alloc ~wait:(fun () -> ()) (Kernargs.create root) 24 in
                  Kernargs.write_args (argument_layout 2 [ Tolk_uop.Dtype.int32; Tolk_uop.Dtype.int32 ]) slot ~bufs:[| 0x1000n; 0x2000n |]
                    ~vals:[| 7L; -1L |];
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
                  let slot = Kernargs.alloc ~wait:(fun () -> ()) (Kernargs.create root) 32 in
                  Kernargs.write_args (argument_layout 1 [ Tolk_uop.Dtype.int32 ]) slot
                    ~prefix:[| 0xdeadbeef; 1 |]
                    ~bufs:[| 0x1000n |] ~vals:[| 7L |];
                  equal bytes
                    (Bytes.of_string
                       "\xef\xbe\xad\xde\x01\x00\x00\x00\
                        \x00\x10\x00\x00\x00\x00\x00\x00\
                        \x07\x00\x00\x00")
                    (Mmio.read_bytes m ~off:0 ~len:20);
                  raises_match is_invalid_arg (fun () ->
                      Kernargs.write_args [] slot ~prefix:[| -1 |] ~bufs:[||]
                        ~vals:[||])));
          test "write_args checks slots and capacity before modifying memory" (fun () ->
              with_map 4096 (fun m ->
                  let root =
                    Buffer.make ~va:0n ~size:4096 ~view:m ~meta:() ()
                  in
                  let slot = Buffer.offset root ~off:0 ~size:16 () in
                  let before = Mmio.read_bytes m ~off:0 ~len:16 in
                  raises_match is_invalid_arg (fun () ->
                      Kernargs.write_args (argument_layout 3 []) slot
                        ~bufs:[| 0x1n; 0x2n; 0x3n |]
                        ~vals:[||]);
                  raises_match is_invalid_arg (fun () ->
                      Kernargs.write_args (argument_layout 1 []) slot ~bufs:[||]
                        ~vals:[| 0x100000000L |]);
                  let short = Buffer.make ~va:0n ~size:1 ~view:m ~meta:() () in
                  raises_match is_invalid_arg (fun () ->
                      Kernargs.write_args (argument_layout 0 [ Tolk_uop.Dtype.int64 ])
                        short ~bufs:[||] ~vals:[| 1L |]);
                  equal bytes before (Mmio.read_bytes m ~off:0 ~len:16)));
          test "write_args preserves mixed widths after a driver prefix" (fun () ->
              with_map 4096 (fun m ->
                  let slot = Buffer.make ~va:0n ~size:40 ~view:(Mmio.view m ~off:0 ~size:40 ()) ~meta:() () in
                  let open Tolk_uop in
                  let layout = argument_layout 1 [ Dtype.int8; Dtype.int16; Dtype.int32; Dtype.int64 ] in
                  Kernargs.write_args ~prefix:[| 0xdeadbeef; 1 |] layout slot
                    ~bufs:[| 0x100002000n |] ~vals:[| -7L; 300L; 12345L; Int64.min_int |];
                  equal bytes
                    (Bytes.of_string "\xef\xbe\xad\xde\x01\x00\x00\x00\x00\x20\x00\x00\x01\x00\x00\x00\xf9\x00\x2c\x01\x39\x30\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80")
                    (Mmio.read_bytes m ~off:0 ~len:32)));
          test "wrap waits once and leaves the cursor unchanged on failure" (fun () ->
              with_map 4096 (fun m ->
                  let root = Buffer.make ~va:0x300000n ~size:64
                      ~view:(Mmio.view m ~off:0 ~size:64 ()) ~meta:() () in
                  let k = Kernargs.create root in
                  let waits = ref 0 in
                  let wait () = incr waits in
                  ignore (Kernargs.alloc k 48 ~wait);
                  equal int 0 !waits;
                  raises_match (Exn.failure ~substring:"busy") (fun () ->
                      Kernargs.alloc k 32 ~wait:(fun () -> failwith "busy"));
                  equal nativeint 0x300030n (Buffer.va (Kernargs.alloc k 16 ~wait));
                  equal int 0 !waits;
                  equal nativeint 0x300000n (Buffer.va (Kernargs.alloc k 24 ~wait));
                  equal int 1 !waits;
                  raises_match is_invalid_arg (fun () -> Kernargs.alloc k 80 ~wait);
                  equal nativeint 0x300018n (Buffer.va (Kernargs.alloc k 8 ~wait));
                  equal int 1 !waits));
          test "the region wraps when exhausted" (fun () ->
              with_map 4096 (fun m ->
                  let root =
                    Buffer.make ~va:0x300000n ~size:64
                      ~view:(Mmio.view m ~off:0 ~size:64 ())
                      ~meta:() ()
                  in
                  let k = Kernargs.create root in
                  ignore (Kernargs.alloc ~wait:(fun () -> ()) k 48);
                  let wrapped = Kernargs.alloc ~wait:(fun () -> ()) k 32 in
                  equal nativeint 0x300000n (Buffer.va wrapped);
                  raises_match is_invalid_arg (fun () ->
                      Kernargs.alloc ~wait:(fun () -> ()) k 80)));
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
          test "exec rejects unsupported programs and missing dispatch packets" (fun () ->
              let module Cq = Tolk_amd.Compute_queue in
              let kernargs = Buffer.make ~va:0x300000n ~size:24 ~meta:() () in
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
          test "timeline epochs keep GPU waits and SDMA fences in one dword" (fun () ->
              with_map 4096 (fun m ->
                  let signal = Signal.make ~is_timeline:true (slot_buf ~va:0x400000n m) in
                  let compute = Tolk_amd.Compute_queue.create (gfx1100 ()) in
                  Tolk_amd.Compute_queue.wait compute ~value:0x100000005 signal;
                  equal int 5 (Q.dwords (Tolk_amd.Compute_queue.q compute)).(4);
                  let copy = Tolk_amd.Copy_queue.create (gfx1100 ()) in
                  Tolk_amd.Copy_queue.wait copy ~value:0x100000005 signal;
                  Tolk_amd.Copy_queue.signal copy ~value:0x100000006 signal;
                  let words = Q.dwords (Tolk_amd.Copy_queue.q copy) in
                  equal int 5 words.(3);
                  equal int 6 words.(9)));
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
                  equal int 3 (Int64.to_int (Mmio.read64 qd.Tolk_amd.Queue_desc.write_ptr 0));
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
                  equal int 9 (Int64.to_int (Mmio.read64 qd.Tolk_amd.Queue_desc.write_ptr 0));
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
                  equal int 8 (Int64.to_int (Mmio.read64 qd.Tolk_amd.Queue_desc.write_ptr 0))));
          test "multi-die submit pads the indirect body past the wrap"
            (fun () ->
              let module Cq = Tolk_amd.Compute_queue in
              with_map 4096 (fun m ->
                  let dev = gfx942 () in
                  let module P = (val dev.Tolk_amd.pm4) in
                  let qd = queue_desc ~ring_dwords:32 m in
                  Mmio.write64 qd.Tolk_amd.Queue_desc.write_ptr 0 26L;
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
                  equal int 35 (Int64.to_int (Mmio.read64 qd.Tolk_amd.Queue_desc.write_ptr 0))));
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
                  equal int 20 (Int64.to_int (Mmio.read64 qd.Tolk_amd.Queue_desc.write_ptr 0));
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
                  equal int 56 (Int64.to_int (Mmio.read64 qd.Tolk_amd.Queue_desc.write_ptr 0));
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
                  equal int 92 (Int64.to_int (Mmio.read64 qd.Tolk_amd.Queue_desc.write_ptr 0));
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
                  equal int 28 (Int64.to_int (Mmio.read64 qd.Tolk_amd.Queue_desc.write_ptr 0));
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
              (* growing retires the old buffer after replacement succeeds *)
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
                131;
              (* 131 rounds to 132 B/thread (4-byte granule); 33 alignment
                 granules per wave. *)
              equal int 25952256 (Buffer.size dev.Tolk_amd.scratch);
              equal int (512 lor (33 lsl 12)) dev.tmpring_size);
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
              (* Small requests retain the upstream 128 B/thread minimum:
                 8 dies of 128 B * 64 * 32 * 38. *)
              equal int 0x4c00000 (Buffer.size dev.Tolk_amd.scratch);
              equal int (1216 lor (8 lsl 12)) dev.tmpring_size;
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
          test "a failed grow preserves the old allocation and rejects the launch" (fun () ->
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
              let previous = dev.Tolk_amd.scratch in
              raises_match (Exn.failure ~substring:"no memory") (fun () ->
                  Tolk_amd.ensure_has_local_memory dev ~props ~alloc ~free 1024);
              is_true (dev.scratch == previous);
              (* The previous backing is still owned; failed growth neither
                 frees nor reallocates it. *)
              equal (list int) [ 0x3000000 ] !allocs;
              equal (list int) [] !frees;
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
          test "arena wrap waits before replacing live kernel arguments" (fun () ->
              with_map 8192 (fun m ->
                  let dev = gfx1100 () in
                  let qd = queue_desc ~ring_dwords:512 m in
                  let aux = 512 * 4 + 24 in
                  let tl = Signal.make ~is_timeline:true ~owner:dev
                      (Buffer.make ~va:0x400000n ~size:16
                         ~view:(Mmio.view m ~off:aux ~size:16 ()) ~meta:() ()) in
                  let arena = Mmio.view m ~off:(aux + 16) ~size:24 () in
                  let kernargs = Kernargs.create
                      (Buffer.make ~va:0x300000n ~size:24 ~view:arena ~meta:() ()) in
                  let prg = {
                    Program.params = amd_prog dev; name = "k";
                    lib_gpu = Buffer.make ~va:0x100000n ~size:0x1000 ~meta:() ();
                    group_segment_size = 0; private_segment_size = 0;
                    kernargs_segment_size = 24; kernargs_alloc_size = 24 } in
                  let launch timeline_value value = Program.call prg
                      ~layout:(argument_layout 2 [Tolk_uop.Dtype.int64]) ~kernargs
                      ~queue:qd ~timeline:tl ~timeline_value ~timeout_ms:0
                      ~bufs:[|0x1000n; 0x2000n|] ~vals:[|value|]
                      ~global_size:(1, 1, 1) ~local_size:(1, 1, 1) () in
                  ignore (launch 1 7L);
                  let before = Mmio.read_bytes arena ~off:0 ~len:24 in
                  let put = Mmio.read64 qd.Tolk_amd.Queue_desc.write_ptr 0 in
                  raises_match (function Signal.Timeout _ -> true | _ -> false)
                    (fun () -> launch 2 19L);
                  equal bytes before (Mmio.read_bytes arena ~off:0 ~len:24);
                  equal int64 put (Mmio.read64 qd.Tolk_amd.Queue_desc.write_ptr 0);
                  Signal.set_value tl 1;
                  ignore (launch 2 19L);
                  equal int64 19L (Mmio.read64 arena 16)));
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
                    Program.call prg ~layout:(argument_layout 2 [ Tolk_uop.Dtype.int64 ])
                      ~kernargs ~queue:qd ~timeline:tl
                      ~timeline_value:0x43 ~wait:(st, en)
                      ~bufs:[| 0x1000n; 0x2000n |] ~vals:[| 0x100000007L |]
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
                        \x07\x00\x00\x00\x01\x00\x00\x00")
                    (Mmio.read_bytes m ~off:(aux + 64) ~len:24);
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
                    (Int64.to_int (Mmio.read64 qd.Tolk_amd.Queue_desc.write_ptr 0));
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
                    Program.call prg ~layout:[] ~kernargs ~queue:qd ~timeline:tl
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
          test "call stages the dispatch pointer and HSA packet"
            (fun () ->
              with_map 4096 (fun m ->
                  let dev = gfx1100 () in
                  let qd = queue_desc ~ring_dwords:256 m in
                  let aux = (256 * 4) + 24 in
                  let tl =
                    Signal.make
                      (Buffer.make ~va:0x400000n ~size:16
                         ~view:(Mmio.view m ~off:aux ~size:16 ())
                         ~meta:() ())
                  in
                  let kernargs =
                    Kernargs.create
                      (Buffer.make ~va:0x300000n ~size:128
                         ~view:(Mmio.view m ~off:(aux + 16) ~size:128 ())
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
                  ignore (Program.call ~layout:[]
                    (prog (amd_prog ~dispatch_ptr:true dev) 88)
                    ~kernargs ~queue:qd ~timeline:tl ~timeline_value:1
                    ~bufs:[||] ~vals:[||] ~global_size:(2, 3, 4)
                    ~local_size:(8, 4, 2) ());
                  let packet = Mmio.view m ~off:(aux + 16 + 24) ~size:64 () in
                  equal int32 0x31502l (Mmio.read32 packet 0);
                  equal int32 0x40008l (Mmio.read32 packet 4);
                  equal int32 2l (Mmio.read32 packet 8);
                  equal int32 16l (Mmio.read32 packet 12);
                  equal int32 12l (Mmio.read32 packet 16);
                  equal int32 8l (Mmio.read32 packet 20);
                  let words = ring_dwords m (Int64.to_int
                    (Mmio.read64 qd.Tolk_amd.Queue_desc.write_ptr 0)) in
                  let pointer_pair = ref false in
                  for i = 0 to Array.length words - 4 do
                    if Array.sub words i 4 = [|0x300018; 0; 0x300000; 0|] then
                      pointer_pair := true
                  done;
                  is_true ~msg:"dispatch pointer precedes the kernarg pointer" !pointer_pair;
                  equal nativeint 0x300058n (Buffer.va (Kernargs.alloc ~wait:(fun () -> ()) kernargs 8));
                  (* the timeline wait needs a value to wait on *)
                  raises_match is_invalid_arg (fun () ->
                      Program.call ~layout:[]
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
          test "deallocation releases host and BAR CPU mappings" (fun () ->
              let device = amd_device () in
              let mapped address =
                In_channel.with_open_text "/proc/self/maps" (fun ic ->
                    let rec loop () =
                      match In_channel.input_line ic with
                      | None -> false
                      | Some line ->
                          let bounds = List.hd (String.split_on_char ' ' line) in
                          match String.split_on_char '-' bounds with
                          | [lo; hi] ->
                              let lo = Nativeint.of_string ("0x" ^ lo) in
                              let hi = Nativeint.of_string ("0x" ^ hi) in
                              (lo <= address && address < hi) || loop ()
                          | _ -> fail "invalid /proc/self/maps entry"
                    in loop ()) in
              List.iter (fun host ->
                  let spec = {Tolk.Device.Buffer_spec.default with
                    host; cpu_access = true; uncached = true; nolru = true} in
                  let buffer = Tolk.Device.create_buffer device ~size:4096
                      ~dtype:D.uint8 ~spec in
                  let address = match Tolk.Device.Buffer.host_addr buffer with
                    | Some address -> address
                    | None -> fail "CPU-visible allocation has no host mapping" in
                  is_true (mapped address);
                  Tolk.Device.Buffer.deallocate buffer;
                  is_false (mapped address)) [true; false]);
          test "create opens the device and synchronize completes" (fun () ->
              let device = amd_device () in
              equal string "AMD" (Tolk.Device.name device);
              Tolk.Device.synchronize device);
          test "dispatch maps a host view and preserves the external CPU allocation" (fun () ->
              let device = amd_device () in
              with_map 4096 (fun mapping ->
                  Mmio.write32 mapping 4 41l;
                  let allocator = Tolk_uop.Storage.Host_allocator.make
                      ~synchronize:(fun () -> ()) in
                  let spec = {Tolk.Device.Buffer_spec.default with
                    external_ptr = Some (Mmio.addr mapping)} in
                  let base = Tolk.Device.Buffer.create ~device:"CPU" ~size:1024
                      ~dtype:D.int32 ~spec (Tolk.Device.Allocator.Pack allocator) in
                  let view = Tolk.Device.Buffer.view base ~size:1 ~dtype:D.int32 ~offset:4 in
                  let dst = i32_buf device [0] in
                  let spec = Tolk.Device.compile_program device ~name:"amd_mapped_host"
                      (increment_program ()) in
                  let runner = Tolk.Realize.Compiled_runner.create ~device spec in
                  ignore (Tolk.Realize.Compiled_runner.call runner [dst; view] []
                    ~wait:true ~timeout:None);
                  equal (list int) [42] (read_i32 dst);
                  Tolk.Device.Buffer.deallocate view;
                  Tolk.Device.Buffer.deallocate base;
                  Mmio.write32 mapping 4 99l;
                  equal int32 99l (Mmio.read32 mapping 4)));
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
