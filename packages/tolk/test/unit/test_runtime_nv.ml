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
module Tables = Tolk_nv.Nv_tables
module Defs = Tolk_nv.Nv_tables.Defs
module Qmd = Tolk_nv.Qmd
module Nv_iface = Tolk_nv.Nv_iface
module Nvk_iface = Tolk_nv.Nvk_iface
module Pci_iface = Tolk_nv.Pci_iface
module Program = Tolk_nv.Program
module Submission = Tolk_hcq.Hcq.Submission

let is_invalid_arg = function Invalid_argument _ -> true | _ -> false

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

(* One 0x8000-byte anonymous mapping backs everything a test touches:
   the usermode register region at
   0x1000 and template descriptor storage at 0x6000. Device addresses are made up and distinct from
   the CPU mapping. *)
let with_fixture f = with_map 0x8000 f

let nv_dev ?(compute_class = Defs.ada_compute_a)
    ?(sass_version = 0x59) ?slm_per_thread m =
  Tolk_nv.device ~compute_class ~dma_class:Defs.ampere_dma_copy_b
    ~gpfifo_class:Defs.ampere_channel_gpfifo_a ~sass_version ?slm_per_thread
    ~shared_mem_window:0x729400000000n ~local_mem_window:0x729300000000n
    ~gpu_mmio:(Mmio.view m ~off:0x1000 ~size:0x1000 ())
    ()

let template_qmd ~compute_class m =
  Qmd.create
    ~view:
      (Mmio.view m ~off:0x6000 ~size:(Qmd.sizeof ~compute_class) ())
    ~compute_class

(* Wire-format tests compare structures as hex strings so a mismatch
   shows the whole layout. *)
let blob_hex (b : Tables.blob) =
  String.concat ""
    (List.init (Bigarray.Array1.dim b) (fun i ->
         Printf.sprintf "%02x" (Char.code (Bigarray.Array1.get b i))))

let bytes_hex b =
  String.concat ""
    (List.init (Bytes.length b) (fun i ->
         Printf.sprintf "%02x" (Char.code (Bytes.get b i))))

let version_blob s =
  let b =
    Tables.create_blob
      Defs.Nv0000_ctrl_system_get_build_version_v2_params.sizeof
  in
  String.iteri (fun i c -> Bigarray.Array1.set b i c) s;
  b

(* Opening the kernel driver needs Linux, the driver and a device; the
   first failure to do so skips every device test. *)
let nvk_iface =
  let cached = ref None in
  fun () ->
    match !cached with
    | Some i -> i
    | None -> (
        try
          let i = Nvk_iface.iface ~device_id:0 in
          cached := Some i;
          i
        with Failure msg -> skip ~reason:msg ())

let set16 b off v = Bytes.set_uint16_le b off v
let set32 b off v = Bytes.set_int32_le b off (Int32.of_int v)
let set64 b off v = Bytes.set_int64_le b off (Int64.of_int v)

(* Hand-crafted 64-bit little-endian shared object shaped like a cubin
   for a kernel named "k": [.text.k] at image address 0x2000, constant
   banks 0 and 3 at 0x12000 and 0x1a000, a shared-memory carrier of 0x80
   bytes, the global and per-kernel [.nv.info] descriptor sections, and
   three relocations patching [.text.k] against the constant banks.
   Field values are spelled as literals so the loader's parsing is
   checked against independently written numbers. *)
let cubin_fixture ?(reloc0 = 2) ?(undefined_sym = false) ?(bad_info = false)
    ?(regcount = 32) () =
  let module Buf = Stdlib.Buffer in
  let entry b typ param sz =
    Buf.add_char b (Char.chr typ);
    Buf.add_char b (Char.chr param);
    Buf.add_char b (Char.chr (sz land 0xff));
    Buf.add_char b (Char.chr (sz lsr 8))
  in
  let word b v =
    for i = 0 to 3 do
      Buf.add_char b (Char.chr ((v lsr (8 * i)) land 0xff))
    done
  in
  let info = Buf.create 64 in
  if bad_info then entry info 7 0x99 0;
  (* a short-form value entry, then MIN_STACK_SIZE 0x140, REGCOUNT, and
     an ignored payload entry (EXIT_INSTR_OFFSETS) *)
  entry info 3 0x25 0x10;
  entry info 4 0x12 8;
  word info 4;
  word info 0x140;
  entry info 4 0x2f 8;
  word info 4;
  word info regcount;
  entry info 4 0x1c 4;
  word info 0x120;
  let info_k = Buf.create 32 in
  (* an ignored payload entry (KPARAM_INFO), then PARAM_CBANK: a 32-bit
     bank ordinal followed by the 16-bit bank size *)
  entry info_k 4 0x17 12;
  word info_k 0;
  word info_k 0;
  word info_k 0;
  entry info_k 4 0xa 8;
  word info_k 0x18;
  word info_k 0x160;
  let info_b = Buf.to_bytes info and info_k_b = Buf.to_bytes info_k in
  let symtab = Bytes.make 72 '\000' in
  set32 symtab 24 1 (* st_name: "c0" *);
  set16 symtab 30 (if undefined_sym then 0 else 3) (* st_shndx *);
  set64 symtab 32 0x10 (* st_value *);
  set32 symtab 48 4 (* st_name: "c3" *);
  set16 symtab 54 4;
  set64 symtab 56 0x20;
  let strtab = Bytes.of_string "\000c0\000c3\000" in
  let rela = Bytes.make 72 '\000' in
  set64 rela 0 0x100 (* r_offset, within .text.k *);
  set64 rela 8 ((1 lsl 32) lor reloc0);
  set64 rela 24 0x200;
  set64 rela 32 ((1 lsl 32) lor 0x38);
  set64 rela 48 0x300;
  set64 rela 56 ((2 lsl 32) lor 0x39);
  let buf = Buf.create 16384 in
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
  let off_text = add 8 (Bytes.make 0x1800 '\xcc') in
  let off_const0 = add 8 (Bytes.make 0x160 '\xaa') in
  let off_const3 = add 8 (Bytes.make 0x200 '\xbb') in
  let off_info = add 4 info_b in
  let off_info_k = add 4 info_k_b in
  let off_symtab = add 8 symtab in
  let off_strtab = add 1 strtab in
  let off_rela = add 8 rela in
  let shstr = Buf.create 128 in
  Buf.add_char shstr '\000';
  let name s =
    let off = Buf.length shstr in
    Buf.add_string shstr s;
    Buf.add_char shstr '\000';
    off
  in
  let n_text = name ".text.k" in
  let n_shared = name ".nv.shared.k" in
  let n_const0 = name ".nv.constant0.k" in
  let n_const3 = name ".nv.constant3.k" in
  let n_info = name ".nv.info" in
  let n_info_k = name ".nv.info.k" in
  let n_symtab = name ".symtab" in
  let n_strtab = name ".strtab" in
  let n_rela = name ".rela.text.k" in
  let n_shstrtab = name ".shstrtab" in
  let shstr_b = Buf.to_bytes shstr in
  let off_shstr = add 1 shstr_b in
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
  shdr ~nm:n_text ~ty:1 ~flags:0x6 ~addr:0x2000 ~off:off_text ~size:0x1800
    ~link:0 ~info:0 ~salign:128 ~entsize:0;
  shdr ~nm:n_shared ~ty:8 ~flags:0 ~addr:0 ~off:0 ~size:0x80 ~link:0 ~info:0
    ~salign:16 ~entsize:0;
  shdr ~nm:n_const0 ~ty:1 ~flags:0x2 ~addr:0x12000 ~off:off_const0 ~size:0x160
    ~link:0 ~info:0 ~salign:4 ~entsize:0;
  shdr ~nm:n_const3 ~ty:1 ~flags:0x2 ~addr:0x1a000 ~off:off_const3 ~size:0x200
    ~link:0 ~info:0 ~salign:4 ~entsize:0;
  shdr ~nm:n_info ~ty:1 ~flags:0 ~addr:0 ~off:off_info
    ~size:(Bytes.length info_b) ~link:0 ~info:0 ~salign:4 ~entsize:0;
  shdr ~nm:n_info_k ~ty:1 ~flags:0 ~addr:0 ~off:off_info_k
    ~size:(Bytes.length info_k_b) ~link:0 ~info:0 ~salign:4 ~entsize:0;
  shdr ~nm:n_symtab ~ty:2 ~flags:0 ~addr:0 ~off:off_symtab ~size:72 ~link:8
    ~info:1 ~salign:8 ~entsize:24;
  shdr ~nm:n_strtab ~ty:3 ~flags:0 ~addr:0 ~off:off_strtab
    ~size:(Bytes.length strtab) ~link:0 ~info:0 ~salign:1 ~entsize:0;
  shdr ~nm:n_rela ~ty:4 ~flags:0 ~addr:0 ~off:off_rela ~size:72 ~link:7 ~info:1
    ~salign:8 ~entsize:24;
  shdr ~nm:n_shstrtab ~ty:3 ~flags:0 ~addr:0 ~off:off_shstr
    ~size:(Bytes.length shstr_b) ~link:0 ~info:0 ~salign:1 ~entsize:0;
  let obj = Buf.to_bytes buf in
  Bytes.blit_string "\x7fELF\x02\x01\x01" 0 obj 0 7;
  set16 obj 16 3 (* e_type: ET_DYN *);
  set16 obj 18 190 (* e_machine: CUDA *);
  set32 obj 20 1 (* e_version *);
  set64 obj 40 e_shoff;
  set16 obj 52 64 (* e_ehsize *);
  set16 obj 58 64 (* e_shentsize *);
  set16 obj 60 11 (* e_shnum *);
  set16 obj 62 10 (* e_shstrndx *);
  obj

let qmd_template_dwords qmd =
  let b = Qmd.to_bytes qmd in
  Array.init (Bytes.length b / 4) (fun i ->
      Int32.to_int (Bytes.get_int32_le b (4 * i)) land 0xffffffff)

let staged_dwords m ~off n =
  Array.init n (fun i -> Int32.to_int (Mmio.read32 m (off + (4 * i))) land 0xffffffff)

(* The nonzero dwords of the qmd_init goldens
   (test/golden/nvqueue/qmd_init_{ada,blackwell}.expected), which pin the
   reference template for the same program descriptor. *)
let qmd_expected words entries =
  let a = Array.make words 0 in
  List.iter (fun (i, v) -> a.(i) <- v) entries;
  a

let qmd_expected_ada =
  qmd_expected 64
    [
      (4, 0x0000007f); (5, 0x3c000000); (8, 0x00001000); (11, 0x44010000);
      (17, 0x34240480); (18, 0x00000030); (20, 0x00122009); (23, 0x08000000);
      (32, 0x00110000); (33, 0x0b040000); (38, 0x00118000); (39, 0x10000000);
      (48, 0x00100000); (50, 0x00000240); (51, 0x89003000);
    ]

let qmd_expected_blackwell =
  qmd_expected 96
    [
      (4, 0x013f0000); (14, 0x0f5003a4); (19, 0x00010000); (32, 0x00010000);
      (33, 0x03000000); (35, 0x00022000); (36, 0x04b44809); (37, 0x00240000);
      (42, 0x00004400); (43, 0x0b000000); (48, 0x00004600); (49, 0x10000000);
      (58, 0x00001009); (59, 0x00001000);
    ]

(* A ready timeline over two mapped slots; the counter starts at 1 as on
   a fresh device. *)
let timeline m =
  let sig_at off va =
    Signal.make ~is_timeline:true
      (Buffer.make ~va ~size:16 ~view:(Mmio.view m ~off ~size:16 ()) ~meta:() ())
  in
  {
    Timeline.timeline = sig_at 0x3000 0x200000010n;

    error_state = None;
    on_hang = (fun () -> ());
  }

(* A driver interface whose every unscripted call fails the test; the
   topology tests script [rm_control] and read through the seam. *)
type Nv_iface.nvdev += Fake_nvdev

let fake_iface ?nvdev
    ?(rm_control = fun ~obj:_ ~cmd:_ ?params:_ () -> fail "unscripted rm_control")
    () =
  {
    Nv_iface.root = 0xc1d00001;
    gpu_instance = 0;
    count = 1;
    defs = Tables.defs_for_driver ~major:570;
    set_device = (fun ~nvdevice:_ ~subdevice:_ ~virtmem:_ -> fail "unscripted set_device");
    rm_alloc = (fun ~parent:_ ~cls:_ ?params:_ () -> fail "unscripted rm_alloc");
    rm_control;
    alloc =
      (fun ?host:_ ?uncached:_ ?cpu_access:_ ?contiguous:_ ?force_devmem:_ ?map_flags:_
           ?cpu_addr:_ _ -> fail "unscripted alloc");
    free = (fun _ -> fail "unscripted free");
    kind = Type.Id.make ();
    hmemory = (fun _ -> fail "unscripted hmemory");
    map = (fun _ -> fail "unscripted map");
    unmap = (fun _ -> fail "unscripted unmap");
    setup_usermode = (fun () -> fail "unscripted setup_usermode");
    setup_vm = (fun ~vaspace:_ -> fail "unscripted setup_vm");
    setup_gpfifo_vm = (fun ~gpfifo:_ -> fail "unscripted setup_gpfifo_vm");
    sleep = (fun _ -> ());
    device_fini = (fun () -> ());
    nvdev;
  }

let topology_indices =
  [
    Defs.nv2080_ctrl_gr_info_index_litter_num_gpcs;
    Defs.nv2080_ctrl_gr_info_index_litter_num_tpc_per_gpc;
    Defs.nv2080_ctrl_gr_info_index_litter_num_sm_per_tpc;
    Defs.nv2080_ctrl_gr_info_index_max_warps_per_sm;
    Defs.nv2080_ctrl_gr_info_index_sm_version;
  ]

(* Device-level fixtures: one lazily opened real device shared by the
   group; every test using it skips when this machine cannot provide one
   (no kernel driver, or no device). *)
let nv_device =
  let cached : Tolk.Device.t option ref = ref None in
  fun () ->
    match !cached with
    | Some device -> device
    | None -> (
        try
          let device = Tolk_nv.create "NV" in
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

let time_spec device spec bufs =
  let open Tolk in
  let info = Program_spec.program_info spec in
  let kernel_info = U.{name = Program_spec.name spec; applied_opts = [];
    opts_to_apply = None; estimates = None; beam = 0} in
  let body = U.program ~sink:(U.sink ~kernel_info [U.linear (Program_spec.program spec)])
      ~linear:(U.linear (Program_spec.program spec)) ~source:(U.source (Program_spec.src spec))
      ~binary:(U.binary (Bytes.to_string (Option.get (Program_spec.lib spec)))) ~info () in
  let selected = List.combine info.globals bufs in
  let args = List.init (1 + List.fold_left max (-1) info.globals) (fun slot ->
      match List.assoc_opt slot selected with
      | Some buf -> U.from_buffer buf | None -> U.noop ()) in
  let call = U.call ~body ~args ~info:U.{grad_fxn = None; name = None;
    precompile = false; precompile_backward = false; dtype = D.void; aux = None} in
  let to_program device = Codegen.to_program ~optimize:false (Device.renderer device) in
  Realize.time_call ~device ~to_program call (fun sample -> sample ())

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

let failure_with prefix = function
  | Failure msg -> String.starts_with ~prefix msg
  | _ -> false

let contains ~needle haystack =
  let nl = String.length needle and hl = String.length haystack in
  if nl = 0 then true
  else
    let rec loop i =
      if i + nl > hl then false
      else if String.sub haystack i nl = needle then true
      else loop (i + 1)
    in
    loop 0

(* A sysfs tree holding just the PCI files the bus scan reads, so the device
   allowlist is checked through the real probe path. *)
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

let queue_fixture ?(timeout_ms = 30000) ?(chain = false) ?(extra_args = 0) ?(profile = false)
    ?lib ?global_size ?(local_size = [U.Launch_int 1]) ?image_address ?allocator
    ?(synchronize = fun () -> ())
    ~compute_class ~copies m =
  let open Tolk in
  let open Tolk_uop in
  let device_name = "NV:queue-compilation" in
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
  let lib = Option.value lib ~default:(cubin_fixture ()) in
  let spec = Program_spec.of_program ~name:"k" ~src:"" ~device:device_name ~lib
      [output; small; count; value; store] in
  let info = {(Program_spec.program_info spec) with
    global_size = Option.value global_size ~default:[U.Launch_sym count]; local_size} in
  let kernel_info = U.{name = "k"; applied_opts = []; opts_to_apply = None; estimates = None; beam = 0} in
  let program = U.program ~sink:(U.sink ~kernel_info [store]) ~linear:(U.linear (Program_spec.program spec))
      ~source:(U.source "") ~binary:(U.binary (Bytes.to_string lib)) ~info () in
  let call = U.call ~body:program ~args:[parameter 0]
      ~info:{grad_fxn = None; name = None; precompile = false; precompile_backward = false;
        dtype = D.void; aux = None} in
  let queue = Device.{timestamp_divider = 1000.; profile_offset = (fun () -> 0.); completion = (fun () ->
      match !timeline with
      | None -> Fun.const ()
      | Some tl -> let value = Timeline.submitted tl in
          fun timeout_ms -> Timeline.guarded_wait tl (fun () -> Signal.wait ?timeout_ms tl.Timeline.timeline value)); prepare = (fun () -> Option.iter Timeline.prepare !timeline; Submission.prepare ~timeout_ms submission);
    host = "CPU"; max_kernel_bindings = None; config = (fun () -> ""); copy = (fun _ -> Some "COPY:0");
    encode = Tolk_nv.Encoded_queue.encode (nv_dev ~compute_class m) ~name:device_name
        ~compute_entries:8 ~copy_entries:8 ~compute_token:0x123 ~copy_token:0x456;
    lower = Tolk_nv.Encoded_queue.lower device_name;
    compile = Codegen.to_program ~optimize:false (Device.renderer host)} in
  let image_addresses = Hashtbl.create 2 in
  let allocator = match allocator with
    | Some allocator -> allocator
    | None ->
        let host_allocator = Storage.Host_allocator.make ~synchronize:(fun () -> ()) in
        Device.Allocator.Pack {host_allocator with addr = Some (fun address ->
            Option.value (Hashtbl.find_opt image_addresses address) ~default:address)} in
  let renderer_set = Device.Renderer_set.make ~device:device_name
      ["CLANG", (fun target -> Renderer.with_target target (Device.renderer host))] in
  let buffers = Hashtbl.create 16 in
  let bufferize u = match U.as_param u with
    | Some {param = {allocation = Some ("hcq_submission", _); _}; _} ->
        Some (Submission.buffer submission)
    | _ ->
    let shared = match U.node_tag u with
      | Some ("timeline" | "ring_compute" | "ring_copy" | "gpput_compute" | "gpput_copy"
             | "progress_compute" | "progress_copy" | "doorbell" as tag) -> Hashtbl.find_opt buffers tag
      | _ -> None in
    match shared with Some buffer -> Some buffer | None ->
    let buffer = Device.Buffer.create ~device:device_name ~size:(U.max_numel u)
        ~dtype:(U.dtype u) allocator in
    Device.Buffer.ensure_allocated buffer;
    (match image_address, U.node_tag u with
     | Some address, Some "program" ->
         Hashtbl.replace image_addresses (Device.Buffer.addr buffer) address
     | _ -> ());
    if U.node_tag u = Some "program" then
      Device.Buffer.copyin buffer (Bytes.make (Device.Buffer.nbytes buffer) '\255');
    Option.iter (fun tag -> Hashtbl.add buffers tag buffer) (U.node_tag u);
    if U.node_tag u = Some "timeline" then begin
      let address = Device.Buffer.addr buffer in
      let view = Mmio.make ~addr:address ~size:16 in
      let raw = Tolk_hcq.Hcq.Buffer.make ~va:address ~size:16 ~view ~meta:() () in
      timeline := Some {Timeline.timeline = Signal.make ~is_timeline:true raw;
        error_state = None;
        on_hang = (fun () -> fail "unexpected fixture hang")}
    end;
    (match U.as_param u with
     | Some {param = {allocation = Some ("cfunc", data); _}; _} ->
         let libs, symbol = (Marshal.from_string data 0 : string list * string) in
         equal (list string) [] libs;
         let bytes = Bytes.create 8 in
         Bytes.set_int64_le bytes 0 (Int64.of_nativeint (Submission.symbol symbol));
         Device.Buffer.copyin buffer bytes
     | _ -> ());
    Some buffer in
  let device = Device.make ~name:device_name ~allocator ~renderer_set
      ~synchronize:(fun timeout -> ignore timeout; Submission.check submission; synchronize ()) ~queue ~bufferize () in
  let calls = if copies then [U.store_call ~dst:(parameter 0) ~src:(parameter 1);
      call; U.store_call ~dst:(parameter 2) ~src:(parameter 0)]
    else if chain then begin
      let extras = List.init extra_args (fun i -> U.variable ~param:true
          ~name:("extra_" ^ string_of_int i) ~min_val:(-1024) ~max_val:1024 ~dtype:D.int64 ()) in
      let extra_program = if extras = [] then program else
          U.program ~sink:(U.sink ~kernel_info [store])
            ~linear:(U.linear (Program_spec.program spec @ extras)) ~source:(U.source "")
            ~binary:(U.binary (Bytes.to_string lib)) ~info:{info with vars = info.vars @ extras} () in
      [call; U.replace call ~src:[|extra_program; parameter 1|] ()]
    end else [call] in
  Hcq2.compile ~profile ~to_program:(fun device -> Codegen.to_program ~beam_device:device (Device.renderer device)) (U.linear calls), device, host, buffers, submission

let slm_allocator ?(synchronize = fun () -> ()) () =
  let open Tolk in
  let allocs = ref [] and frees = ref [] and fail_next = ref false and attempts = ref [] in
  let base = Tolk_uop.Storage.Host_allocator.make ~synchronize in
  let alloc size (spec : Device.Buffer_spec.t) =
    if spec.nolru then begin
      allocs := size :: !allocs;
      if !fail_next then raise (Nv_iface.Out_of_memory "scripted")
    end;
    base.alloc size spec in
  let free raw size (spec : Device.Buffer_spec.t) =
    if spec.nolru then attempts := size :: !attempts;
    base.free raw size spec;
    if spec.nolru then frees := size :: !frees in
  Device.Allocator.Pack {base with alloc; free}, allocs, frees, fail_next, attempts

let execute_queue ~compute_class ~copies m =
  let open Tolk in
  let compiled, device, host, buffers, submission = queue_fixture ~compute_class ~copies m in
  let linked = Realize.link_linear compiled in
  let get tag = Hashtbl.find buffers tag in
  let set32 tag value =
    let bytes = Bytes.create 4 in
    Bytes.set_int32_le bytes 0 (Int32.of_int value);
    Device.Buffer.copyin (get tag) bytes in
  let word tag = Int32.to_int (Bytes.get_int32_le (Device.Buffer.as_bytes (get tag)) 0) in
  set32 "gpput_compute" 7;
  if copies then set32 "gpput_copy" 7;
  let to_program device = Codegen.to_program ~beam_device:device (Device.renderer device) in
  List.iteri (fun replay (small, count) ->
      let inputs = Array.init (if copies then 3 else 1) (fun _ ->
          Device.create_buffer ~size:16 ~dtype:D.int32 device) in
      Realize.run_linear ~device ~to_program ~jit:true
        ~var_vals:["small", Int64.of_int small; "count", Int64.of_int count] ~input_uops:(Array.map U.from_buffer inputs) linked;
      Submission.check submission;
      equal int replay (word "gpput_compute");
      let qmd_buffer = get "qmd" in
      let qmd_bytes = Device.Buffer.as_bytes qmd_buffer in
      let qmd = Qmd.create ~compute_class
          ~view:(Mmio.make ~addr:(Device.Buffer.addr qmd_buffer) ~size:(Bytes.length qmd_bytes)) in
      let qmd_size = if Qmd.version qmd < 4 then 256 else 512 in
      let at = qmd_size + (if Qmd.version qmd < 4 then 88 else 224) * 4 in
      equal int small (Bytes.get_int8 qmd_bytes (at + 8));
      equal int64 (Int64.of_int count) (Bytes.get_int64_le qmd_bytes (at + 16));
      equal int64 (Int64.of_nativeint (Device.Buffer.addr ~device:(Device.name device) inputs.(0)))
        (Bytes.get_int64_le qmd_bytes at);
      equal int count (Qmd.read qmd (if Qmd.version qmd < 4 then "cta_raster_width" else "grid_width"));
      equal int 1 (Qmd.read qmd "cta_thread_dimension0");
      equal int 1 (Qmd.read qmd "release0_enable");
      let image = Hashtbl.find buffers "program" in
      let image_addr = Int64.of_nativeint (Device.Buffer.addr image) in
      let shifted = Qmd.version qmd >= 4 in
      let address lower upper shift =
        Int64.shift_left (Int64.logor (Int64.of_int (Qmd.read qmd lower))
          (Int64.shift_left (Int64.of_int (Qmd.read qmd upper)) 32)) shift in
      let suffix = if shifted then "_shifted4" else "" in
      equal int64 (Int64.add image_addr 0x2000L)
        (address ("program_address_lower" ^ suffix) ("program_address_upper" ^ suffix)
          (if shifted then 4 else 0));
      let suffix = if shifted then "_shifted6_0" else "_0" in
      equal int64 (Int64.add (Int64.of_nativeint (Device.Buffer.addr qmd_buffer)) (Int64.of_int qmd_size))
        (address ("constant_buffer_addr_lower" ^ suffix) ("constant_buffer_addr_upper" ^ suffix)
          (if shifted then 6 else 0));
      equal int64 (Int64.add image_addr 0x12010L)
        (Bytes.get_int64_le (Device.Buffer.as_bytes image) 0x2100);
      let stream = get "cmdbuf_compute" in
      let ring = Device.Buffer.as_bytes (get "ring_compute") in
      let entry = Int64.logor (Int64.of_nativeint (Device.Buffer.addr stream))
          (Int64.of_int (((Device.Buffer.nbytes stream / 4) lsl 42) lor (1 lsl 41))) in
      let actual = Bytes.get_int64_le ring (if replay = 0 then 56 else 0) in
      equal int64 entry actual;
      equal int (Device.Buffer.nbytes stream / 4) (Int64.to_int (Int64.shift_right_logical actual 42));
      if copies then equal int replay (word "gpput_copy");
      equal int 0x123 (word "doorbell");
      (* Execute the generated host program, then model GPU completion. *)
      let timeline = Device.Buffer.as_bytes (get "timeline") in
      Bytes.set_int64_le timeline 0 (Bytes.get_int64_le timeline 8);
      Device.Buffer.copyin (get "timeline") timeline;
      List.iter (fun tag ->
          let progress = Device.Buffer.as_bytes (get tag) in
          Bytes.set_int32_le progress 8 (Int64.to_int32 (Bytes.get_int64_le progress 0));
          Device.Buffer.copyin (get tag) progress)
        ("progress_compute" :: if copies then ["progress_copy"] else [])) [(-17, 3); (29, 11)]

let raw_submissions m =
  let open Tolk in
  let compiled, device, _, buffers, submission =
    queue_fixture ~compute_class:Defs.ada_compute_a ~copies:false m in
  let get tag = Device.Buffer.as_bytes (Hashtbl.find buffers tag) in
  let check timeline compute copy =
    Submission.check submission;
    equal int64 timeline (Bytes.get_int64_le (get "timeline") 8);
    equal int64 compute (Bytes.get_int64_le (get "progress_compute") 0);
    if copy <> 0L then equal int64 copy (Bytes.get_int64_le (get "progress_copy") 0) in
  let setup = [|0x20012000; Defs.ada_compute_a|] in
  Tolk_nv.submit_commands ~device ~queue:"COMPUTE:0" setup;
  check 1L 1L 0L;
  equal int32 (Int32.of_int setup.(0)) (Bytes.get_int32_le (get "cmdbuf_compute") 24);
  Tolk_nv.submit_commands ~device ~queue:"COPY:0" [|0x20018000; Defs.ampere_dma_copy_b|];
  check 2L 1L 1L;
  let linked = Realize.link_linear compiled in
  let input = Device.create_buffer ~size:16 ~dtype:D.int32 device in
  Realize.run_linear ~device
    ~to_program:(fun device -> Codegen.to_program ~beam_device:device (Device.renderer device))
    ~jit:true ~var_vals:["small", 7L; "count", 3L] ~input_uops:[|U.from_buffer input|] linked;
  check 3L 2L 1L;
  Tolk_nv.submit_commands ~device ~queue:"COMPUTE:0" setup;
  check 4L 3L 1L

let raw_submission_timeout m =
  let open Tolk in
  let compiled, device, _, buffers, submission =
    queue_fixture ~timeout_ms:5 ~compute_class:Defs.ada_compute_a ~copies:false m in
  let linked = Realize.link_linear compiled in
  let progress = Hashtbl.find buffers "progress_compute" in
  let bytes = Device.Buffer.as_bytes progress in
  Bytes.set_int64_le bytes 0 7L;
  Device.Buffer.copyin progress bytes;
  let put = Hashtbl.find buffers "gpput_compute" in
  let bytes = Bytes.create 4 in
  Bytes.set_int32_le bytes 0 7l;
  Device.Buffer.copyin put bytes;
  let protected = List.map (fun tag ->
      let buffer = Hashtbl.find buffers tag in buffer, Device.Buffer.as_bytes buffer)
      ["timeline"; "ring_compute"; "gpput_compute"; "progress_compute"; "qmd"; "cmdbuf_compute"] in
  raises_match (Exn.failure ~substring:"HCQ submission timed out") (fun () ->
      Tolk_nv.submit_commands ~device ~queue:"COMPUTE:0" [|0x20012000; Defs.ada_compute_a|]);
  raises_match (Exn.failure ~substring:"HCQ submission timed out") (fun () -> Submission.check submission);
  List.iter (fun (buffer, before) -> equal Windtrap.bytes before (Device.Buffer.as_bytes buffer)) protected;
  ignore (Sys.opaque_identity linked)

let shared_calibration m =
  let open Tolk in
  let stamp = ref None and allocations = ref 0 and frees = ref 0 in
  let base = Tolk_uop.Storage.Host_allocator.make ~synchronize:(fun () -> ()) in
  let alloc size (spec : Device.Buffer_spec.t) =
    let raw = base.alloc size spec in
    if spec.host && spec.uncached && spec.nolru then begin
      equal int 16 size;
      incr allocations;
      stamp := Some raw
    end;
    raw in
  let free raw size spec =
    if Some raw = !stamp then incr frees;
    base.free raw size spec in
  let allocator = Device.Allocator.Pack {base with alloc; free} in
  let buffers = ref None and waits = ref 0 and fail_wait = ref false in
  let collecting = ref false in
  let synchronize () =
    if not !collecting then begin
    equal int 0 (Helpers.Context_var.get Helpers.debug);
    incr waits;
    let get tag = Device.Buffer.as_bytes (Hashtbl.find (Option.get !buffers) tag) in
    let timeline = get "timeline" in
    equal int64 (Int64.of_int !waits) (Bytes.get_int64_le timeline 8);
    let commands = get "cmdbuf_compute" in
    equal int64 (Int64.of_nativeint (Option.get !stamp)) (Bytes.get_int64_le commands 28);
    if !fail_wait then failwith "calibration wait failed";
    let stamp_view = Mmio.make ~addr:(Option.get !stamp) ~size:16 in
    Mmio.write64 stamp_view 8 1234000L;
    Bytes.set_int64_le timeline 0 (Int64.of_int !waits);
    Device.Buffer.copyin (Hashtbl.find (Option.get !buffers) "timeline") timeline;
    let progress = get "progress_compute" in
    Bytes.set_int32_le progress 8 (Int32.of_int !waits);
    Device.Buffer.copyin (Hashtbl.find (Option.get !buffers) "progress_compute") progress
    end in
  let calibration = Tolk_hcq.Hcq.profile_offset "NV:queue-compilation" in
  equal int 0 !allocations;
  let _, device, _, slots, _ = queue_fixture ~allocator ~synchronize
      ~compute_class:Defs.ada_compute_a ~copies:false m in
  buffers := Some slots;
  Helpers.Context_var.with_context [Helpers.Context_var.B (Helpers.debug, 2)] (fun () ->
      for round = 1 to 2 do
        let before = Unix.gettimeofday () *. 1e6 -. 1234. in
        let offset = calibration () in
        let after = Unix.gettimeofday () *. 1e6 -. 1234. in
        is_true (before <= offset && offset <= after);
        equal int (round * 5) !waits;
        equal int 1 !allocations;
        equal int 2 (Helpers.Context_var.get Helpers.debug)
      done;
      collecting := true;
      equal int 0 (List.length (Device.profile device));
      collecting := false;
      fail_wait := true;
      raises_match (Exn.failure ~substring:"calibration wait failed") calibration;
      equal int 11 !waits;
      equal int 2 (Helpers.Context_var.get Helpers.debug));
  Gc.full_major ();
  equal int 0 !frees;
  ignore (Sys.opaque_identity (device, calibration))

let queue_chain ?(extra_args = 0) ~compute_class m =
  let open Tolk in
  let compiled, device, _, buffers, submission =
    queue_fixture ~chain:true ~extra_args ~compute_class ~copies:false m in
  let first = Realize.link_linear ~allow_cache:false compiled in
  let arena = Hashtbl.find buffers "qmd" in
  equal int 1 (List.length (Hashtbl.find_all buffers "qmd"));
  equal int 1 (List.length (Hashtbl.find_all buffers "program"));
  let second = Realize.link_linear ~allow_cache:false compiled in
  let other = Hashtbl.find buffers "qmd" in
  is_false (Device.Buffer.id arena = Device.Buffer.id other);
  equal int 2 (List.length (Hashtbl.find_all buffers "qmd"));
  equal int 1 (List.length (Hashtbl.find_all buffers "program"));
  let qmd_size = if compute_class >= Defs.blackwell_compute_a then 512 else 256 in
  let prefix_size = if compute_class >= Defs.blackwell_compute_a then 896 else 0x160 in
  let stride = qmd_size + ((prefix_size + (3 + extra_args) * 8 + 255) / 256 * 256) in
  equal int (2 * stride) (Device.Buffer.nbytes arena);
  let inputs = Array.init 2 (fun _ -> Device.create_buffer ~size:16 ~dtype:D.int32 device) in
  let retire () =
    let timeline = Hashtbl.find buffers "timeline" in
    let bytes = Device.Buffer.as_bytes timeline in
    Bytes.set_int64_le bytes 0 (Bytes.get_int64_le bytes 8);
    Device.Buffer.copyin timeline bytes;
    let progress = Hashtbl.find buffers "progress_compute" in
    let bytes = Device.Buffer.as_bytes progress in
    Bytes.set_int32_le bytes 8 (Int64.to_int32 (Bytes.get_int64_le bytes 0));
    Device.Buffer.copyin progress bytes in
  let run linked arena small count inputs =
    Realize.run_linear ~device ~to_program:(fun device -> Codegen.to_program ~beam_device:device (Device.renderer device))
      ~jit:true ~var_vals:(["small", Int64.of_int small; "count", Int64.of_int count] @
        List.init extra_args (fun i -> "extra_" ^ string_of_int i, Int64.of_int (i + small)))
      ~input_uops:(Array.map U.from_buffer inputs) linked;
    Submission.check submission;
    let address = Device.Buffer.addr arena in
    equal int 0 (Nativeint.to_int address land 255);
    let descriptors = Array.init 2 (fun i ->
        Qmd.create ~compute_class ~view:(Mmio.make
          ~addr:(Nativeint.add address (Nativeint.of_int (i * stride))) ~size:qmd_size)) in
    equal int (((Nativeint.to_int address + stride) lsr 8) land 0xffffffff)
      (Qmd.read descriptors.(0) "dependent_qmd0_pointer");
    equal int 1 (Qmd.read descriptors.(0) "dependent_qmd0_action");
    equal int 1 (Qmd.read descriptors.(0) "dependent_qmd0_prefetch");
    equal int 0 (Qmd.read descriptors.(0) "release0_enable");
    equal int 1 (Qmd.read descriptors.(1) "release0_enable");
    let bytes = Device.Buffer.as_bytes arena in
    Array.iteri (fun i input ->
        let at = i * stride + qmd_size + prefix_size in
        equal int64 (Int64.of_nativeint (Device.Buffer.addr input)) (Bytes.get_int64_le bytes at);
        equal int small (Bytes.get_int8 bytes (at + 8));
        equal int64 (Int64.of_int count) (Bytes.get_int64_le bytes (at + 16));
        if i = 1 then for j = 0 to extra_args - 1 do
          equal int64 (Int64.of_int (j + small)) (Bytes.get_int64_le bytes (at + 24 + j * 8))
        done) inputs;
    retire () in
  run first arena 4 2 inputs;
  let first_bytes = Device.Buffer.as_bytes arena in
  run second other 7 5 (Array.of_list [inputs.(1); inputs.(0)]);
  equal bytes first_bytes (Device.Buffer.as_bytes arena);
  let other_bytes = Device.Buffer.as_bytes other in
  run first arena (-3) 8 (Array.of_list [inputs.(1); inputs.(0)]);
  equal bytes other_bytes (Device.Buffer.as_bytes other);
  equal int 1 (List.length (Hashtbl.find_all buffers "program"));
  let stream = Device.Buffer.as_bytes (Hashtbl.find buffers "cmdbuf_compute") in
  equal int 72 (Bytes.length stream)

let queue_timeout m =
  let open Tolk in
  let compiled, device, host, buffers, submission = queue_fixture ~timeout_ms:5 ~compute_class:Defs.ada_compute_a ~copies:false m in
  let linked = Realize.link_linear compiled in
  let input = Device.create_buffer ~size:16 ~dtype:D.int32 device in
  let run () = Realize.run_linear ~device
      ~to_program:(fun device -> Codegen.to_program ~beam_device:device (Device.renderer device)) ~jit:true
      ~var_vals:["small", 7L; "count", 3L] ~input_uops:[|U.from_buffer input|] linked in
  run ();
  let doorbell () = Device.Buffer.as_bytes (Hashtbl.find buffers "gpput_compute") in
  let published = doorbell () in
  let protected = List.map (fun tag -> tag, Device.Buffer.as_bytes (Hashtbl.find buffers tag))
      ["timeline"; "slots"; "qmd"; "cmdbuf_compute"] in
  run ();
  raises_match (Exn.failure ~substring:"HCQ submission timed out")
    (fun () -> Submission.check submission);
  equal bytes published (doorbell ());
  List.iter (fun (tag, before) -> equal ~msg:tag bytes before
      (Device.Buffer.as_bytes (Hashtbl.find buffers tag))) protected;
  raises_match (Exn.failure ~substring:"HCQ submission timed out") run

let queue_capacity ~copies ~resume m =
  let open Tolk in
  let compiled, device, host, buffers, submission =
    queue_fixture ~timeout_ms:100 ~compute_class:Defs.ada_compute_a ~copies m in
  let inputs = Array.init (if copies then 3 else 1) (fun _ ->
      U.from_buffer (Device.create_buffer ~size:16 ~dtype:D.int32 device)) in
  let run () =
    let linked = Realize.link_linear ~allow_cache:false compiled in
    Realize.run_linear ~device
      ~to_program:(fun device -> Codegen.to_program ~beam_device:device (Device.renderer device))
      ~jit:true ~var_vals:["small", 7L; "count", 3L]
      ~input_uops:inputs linked
  in
  for i = 1 to 7 do
    run ();
    Submission.check submission;
    equal int i (Int32.to_int (Bytes.get_int32_le
      (Device.Buffer.as_bytes (Hashtbl.find buffers "gpput_compute")) 0))
  done;
  let channels = "compute" :: if copies then ["copy"] else [] in
  let before = List.concat_map (fun suffix ->
      List.map (fun prefix -> let tag = prefix ^ suffix in
          tag, Device.Buffer.as_bytes (Hashtbl.find buffers tag))
        ["ring_"; "gpput_"; "progress_"]) channels in
  if resume then begin
    let progress = List.map (fun suffix ->
        let b = Hashtbl.find buffers ("progress_" ^ suffix) in
        Mmio.make ~addr:(Device.Buffer.addr b) ~size:16) channels in
    let gpu = Domain.spawn (fun () -> Unix.sleepf 0.01;
        List.iter (fun p -> Mmio.write32 p 8 3l) progress) in
    Fun.protect ~finally:(fun () -> Domain.join gpu) run;
    Submission.check submission;
    List.iter (fun suffix ->
        equal int32 0l (Bytes.get_int32_le
          (Device.Buffer.as_bytes (Hashtbl.find buffers ("gpput_" ^ suffix))) 0);
        equal int64 8L (Bytes.get_int64_le
          (Device.Buffer.as_bytes (Hashtbl.find buffers ("progress_" ^ suffix))) 0)) channels
  end else begin
    run ();
    raises_match (Exn.failure ~substring:"HCQ submission timed out")
      (fun () -> Submission.check submission);
    List.iter (fun (tag, bytes_before) -> equal ~msg:tag bytes bytes_before
        (Device.Buffer.as_bytes (Hashtbl.find buffers tag))) before
  end

let queue_counter_rollover m =
  let open Tolk in
  let compiled, device, host, buffers, submission =
    queue_fixture ~timeout_ms:100 ~compute_class:Defs.ada_compute_a ~copies:false m in
  let linked = Realize.link_linear compiled in
  let progress = Hashtbl.find buffers "progress_compute" in
  let data = Bytes.make 16 '\000' in
  Bytes.set_int64_le data 0 0xfffffffeL;
  Bytes.set_int32_le data 8 (-2l);
  Device.Buffer.copyin progress data;
  let input = Device.create_buffer ~size:16 ~dtype:D.int32 device in
  List.iter (fun expected ->
      Realize.run_linear ~device
        ~to_program:(fun device -> Codegen.to_program ~beam_device:device (Device.renderer device))
        ~jit:true ~var_vals:["small", 7L; "count", 3L]
        ~input_uops:[|U.from_buffer input|] linked;
      Submission.check submission;
      let data = Device.Buffer.as_bytes progress in
      equal int64 expected (Bytes.get_int64_le data 0);
      let commands = Device.Buffer.as_bytes (Hashtbl.find buffers "cmdbuf_compute") in
      equal int32 (Int64.to_int32 expected)
        (Bytes.get_int32_le commands (Bytes.length commands - 12));
      Bytes.set_int32_le data 8 (Int64.to_int32 expected);
      Device.Buffer.copyin progress data;
      let timeline = Hashtbl.find buffers "timeline" in
      let data = Device.Buffer.as_bytes timeline in
      Bytes.set_int64_le data 0 (Bytes.get_int64_le data 8);
      Device.Buffer.copyin timeline data) [0xffffffffL; 0x100000000L; 0x100000001L]

let queue_retirement_timeout m =
  let open Tolk in
  let compiled, device, host, buffers, submission =
    queue_fixture ~timeout_ms:5 ~compute_class:Defs.ada_compute_a ~copies:false m in
  let linked = Realize.link_linear compiled in
  let input = Device.create_buffer ~size:16 ~dtype:D.int32 device in
  let run small = Realize.run_linear ~device
      ~to_program:(fun device -> Codegen.to_program ~beam_device:device (Device.renderer device))
      ~jit:true ~var_vals:["small", Int64.of_int small; "count", 3L]
      ~input_uops:[|U.from_buffer input|] linked in
  run 7;
  (* The kernel timeline may finish before the channel consumes its tail. *)
  let timeline = Hashtbl.find buffers "timeline" in
  let data = Device.Buffer.as_bytes timeline in
  Bytes.set_int64_le data 0 (Bytes.get_int64_le data 8);
  Device.Buffer.copyin timeline data;
  let before = List.map (fun tag -> tag, Device.Buffer.as_bytes (Hashtbl.find buffers tag))
      ["qmd"; "cmdbuf_compute"; "ring_compute"; "gpput_compute"; "progress_compute"] in
  run 19;
  raises_match (Exn.failure ~substring:"HCQ submission timed out")
    (fun () -> Submission.check submission);
  List.iter (fun (tag, data) -> equal ~msg:tag bytes data
      (Device.Buffer.as_bytes (Hashtbl.find buffers tag))) before

let () =
  run "Nv_runtime"
    [
      group "compiled queues"
        [test "shared clock calibration owns its stamp and scopes debug waits" (fun () -> with_fixture shared_calibration);
         test "raw setup and kernels share timeline and FIFO progress" (fun () -> with_fixture raw_submissions);
         test "failed raw submission leaves live storage and counters unchanged" (fun () -> with_fixture raw_submission_timeout);
         test "Ada descriptors chain launches and release only the tail" (fun () ->
             with_fixture (queue_chain ~compute_class:Defs.ada_compute_a));
         test "Blackwell descriptors chain launches and release only the tail" (fun () ->
             with_fixture (queue_chain ~compute_class:Defs.blackwell_compute_b));
         test "mixed argument footprints share aligned per-submission arenas" (fun () ->
             List.iter (fun compute_class -> with_fixture (queue_chain ~extra_args:40 ~compute_class))
               [Defs.ada_compute_a; Defs.blackwell_compute_b]);
         test "independent retained batches cannot overfill the FIFO" (fun () ->
             with_fixture (queue_capacity ~copies:false ~resume:false));
         test "compute and copy FIFOs resume when the GPU retires work" (fun () ->
             with_fixture (queue_capacity ~copies:true ~resume:true));
         test "channel completion survives low-dword rollover" (fun () ->
             with_fixture queue_counter_rollover);
         test "a completed kernel does not retire its command tail" (fun () ->
             with_fixture queue_retirement_timeout);
         test "stalled replay times out and latches submission failure" (fun () -> with_fixture queue_timeout);
         test "Ada compute replay patches arguments and wraps the shared FIFO" (fun () ->
             with_fixture (execute_queue ~compute_class:Defs.ada_compute_a ~copies:false));
         test "Blackwell compute and DMA share ordered retained submissions" (fun () ->
             with_fixture (execute_queue ~compute_class:Defs.blackwell_compute_b ~copies:true))];
      group "qmd"
        [
          test "layout follows the compute class" (fun () ->
              equal int 0x100 (Qmd.sizeof ~compute_class:Defs.ada_compute_a);
              equal int 0x180
                (Qmd.sizeof ~compute_class:Defs.blackwell_compute_b);
              with_fixture (fun m ->
                  let v3 = template_qmd ~compute_class:Defs.ada_compute_a m in
                  let v5 =
                    template_qmd ~compute_class:Defs.blackwell_compute_b m
                  in
                  equal int 3 (Qmd.version v3);
                  equal int 5 (Qmd.version v5);
                  equal int 0x100 (Bytes.length (Qmd.to_bytes v3));
                  equal int 0x180 (Bytes.length (Qmd.to_bytes v5))));
          test "a view smaller than the descriptor is rejected" (fun () ->
              with_fixture (fun m ->
                  raises_match is_invalid_arg (fun () ->
                      Qmd.create
                        ~view:(Mmio.view m ~off:0 ~size:0xff ())
                        ~compute_class:Defs.ada_compute_a)));
          test "field writes round-trip and land at the table offsets"
            (fun () ->
              with_fixture (fun m ->
                  let q3 = template_qmd ~compute_class:Defs.ada_compute_a m in
                  (* CTA_RASTER_WIDTH is bits 415..384: bytes 48-51. *)
                  Qmd.write q3 [ ("cta_raster_width", 0xdeadbeef) ];
                  equal int 0xdeadbeef (Qmd.read q3 "cta_raster_width");
                  equal int 48 (Qmd.field_offset q3 "cta_raster_width");
                  let b = Qmd.to_bytes q3 in
                  equal int 0xef (Char.code (Bytes.get b 48));
                  equal int 0xbe (Char.code (Bytes.get b 49));
                  equal int 0xad (Char.code (Bytes.get b 50));
                  equal int 0xde (Char.code (Bytes.get b 51))));
          test "names are case-insensitive" (fun () ->
              with_fixture (fun m ->
                  let q3 = template_qmd ~compute_class:Defs.ada_compute_a m in
                  Qmd.write q3 [ ("CTA_Raster_Width", 7) ];
                  equal int 7 (Qmd.read q3 "cta_raster_width")));
          test "an unaligned field leaves its byte-sharing neighbours"
            (fun () ->
              with_fixture (fun m ->
                  let q3 = template_qmd ~compute_class:Defs.ada_compute_a m in
                  (* CONSTANT_BUFFER_ADDR_UPPER_0 (bits 1072..1056) and
                     CONSTANT_BUFFER_SIZE_SHIFTED4_0 (bits 1087..1075)
                     share byte 134. *)
                  Qmd.write q3 [ ("constant_buffer_addr_upper_0", 0x1ffff) ];
                  Qmd.write q3 [ ("constant_buffer_size_shifted4_0", 0x1abc) ];
                  equal int 0x1ffff (Qmd.read q3 "constant_buffer_addr_upper_0");
                  equal int 0x1abc
                    (Qmd.read q3 "constant_buffer_size_shifted4_0")));
          test "per-slot fields address distinct slots" (fun () ->
              with_fixture (fun m ->
                  let q3 = template_qmd ~compute_class:Defs.ada_compute_a m in
                  Qmd.write q3
                    [
                      ("constant_buffer_addr_lower_0", 0x11111111);
                      ("constant_buffer_addr_lower_3", 0x33333333);
                    ];
                  equal int 0x11111111
                    (Qmd.read q3 "constant_buffer_addr_lower_0");
                  equal int 0x33333333
                    (Qmd.read q3 "constant_buffer_addr_lower_3");
                  equal int 128 (Qmd.field_offset q3 "constant_buffer_addr_lower_0");
                  equal int 152 (Qmd.field_offset q3 "constant_buffer_addr_lower_3")));
          test "unknown names and oversized values are rejected" (fun () ->
              with_fixture (fun m ->
                  let q3 = template_qmd ~compute_class:Defs.ada_compute_a m in
                  raises_match is_invalid_arg (fun () ->
                      Qmd.read q3 "not_a_field");
                  raises_match is_invalid_arg (fun () ->
                      Qmd.write q3 [ ("not_a_field", 1) ]);
                  raises_match is_invalid_arg (fun () ->
                      Qmd.write q3 [ ("release0_enable", 2) ])));
          test "constant buffer addresses split per version" (fun () ->
              with_fixture (fun m ->
                  let q3 = template_qmd ~compute_class:Defs.ada_compute_a m in
                  Qmd.set_constant_buf_addr q3 0 0x234500000n;
                  equal int 0x34500000
                    (Qmd.read q3 "constant_buffer_addr_lower_0");
                  equal int 2 (Qmd.read q3 "constant_buffer_addr_upper_0");
                  let q5 =
                    template_qmd ~compute_class:Defs.blackwell_compute_b m
                  in
                  Qmd.set_constant_buf_addr q5 0 0x234500000n;
                  equal int 0x08d14000
                    (Qmd.read q5 "constant_buffer_addr_lower_shifted6_0");
                  equal int 0
                    (Qmd.read q5 "constant_buffer_addr_upper_shifted6_0")));
        ];
      group "iface wire formats"
        [
          test "the allocation envelope wires the nested parameter pointer"
            (fun () ->
              let inner = Tables.create_blob 0x38 in
              let b =
                Nvk_iface.nvos21_params ~root:0xc1d00001 ~parent:0xbeef
                  ~cls:0x80 ~params:inner ()
              in
              let expected = Bytes.make 0x20 '\000' in
              Bytes.set_int32_le expected 0x00 0xc1d00001l;
              Bytes.set_int32_le expected 0x04 0xbeefl;
              Bytes.set_int32_le expected 0x0c 0x80l;
              Bytes.set_int64_le expected 0x10
                (Int64.of_nativeint (Tables.blob_addr inner));
              equal string (bytes_hex expected) (blob_hex b);
              (* without a parameter structure the pointer stays null *)
              let bare = Nvk_iface.nvos21_params ~root:1 ~parent:2 ~cls:3 () in
              equal int 0
                (Tables.get_field bare Defs.Nvos21_parameters.pallocparms));
          test "memory allocation parameters compose the attribute words"
            (fun () ->
              (* cached, contiguous device memory in 2 MiB pages *)
              let cls, p =
                Nvk_iface.memory_allocation_params ~root:0xc1d00001
                  ~size:0x200000 ~page_size:0x200000 ~uncached:false
                  ~contiguous:true ~read_only:false
              in
              equal int Defs.nv1_memory_user cls;
              let expected = Bytes.make 0x80 '\000' in
              Bytes.set_int32_le expected 0x00 0xc1d00001l;
              (* map-not-required, handle-provided, forced alignment,
                 ignored bank placement, persistent *)
              Bytes.set_int32_le expected 0x08 0x1c101l;
              (* contiguous at 27, huge pages at 23 *)
              Bytes.set_int32_le expected 0x18 0x11800000l;
              (* cacheable at 2, huge 2 MiB at 20, no zbc *)
              Bytes.set_int32_le expected 0x1c 0x100005l;
              Bytes.set_int32_le expected 0x20 6l;
              Bytes.set_int64_le expected 0x40 0x200000L;
              Bytes.set_int64_le expected 0x48 0x200000L;
              Bytes.set_int64_le expected 0x58 0x1fffffL;
              equal string (bytes_hex expected) (blob_hex p);
              (* uncached, read-only system pages *)
              let cls, p =
                Nvk_iface.memory_allocation_params ~root:0xc1d00001 ~size:0x1000
                  ~page_size:0x1000 ~uncached:true ~contiguous:false
                  ~read_only:true
              in
              equal int Defs.nv1_memory_system cls;
              let expected = Bytes.make 0x80 '\000' in
              Bytes.set_int32_le expected 0x00 0xc1d00001l;
              (* notifier type *)
              Bytes.set_int32_le expected 0x04 0xdl;
              (* no persistent-vidmem flag for system pages *)
              Bytes.set_int32_le expected 0x08 0xc101l;
              (* noncontiguous at 27, system location at 25 *)
              Bytes.set_int32_le expected 0x18 0x1a000000l;
              (* uncacheable at 2, no zbc, read-only protection at 22 *)
              Bytes.set_int32_le expected 0x1c 0x400009l;
              Bytes.set_int32_le expected 0x20 6l;
              Bytes.set_int64_le expected 0x40 0x1000L;
              Bytes.set_int64_le expected 0x48 0x1000L;
              Bytes.set_int64_le expected 0x58 0xfffL;
              equal string (bytes_hex expected) (blob_hex p));
          test "the mapping request carries one gpu attribute" (fun () ->
              let uuid = Bytes.init 16 (fun i -> Char.chr (0xa0 + i)) in
              let b =
                Nvk_iface.map_external_params ~rm_ctrl_fd:7 ~root:0xc1d00001
                  ~va:0x1234500000n ~size:0x10000 ~mem_handle:0x5abc1234
                  ~gpu_uuid:uuid
              in
              let expected = Bytes.make 0x2430 '\000' in
              Bytes.set_int64_le expected 0x00 0x1234500000L;
              Bytes.set_int64_le expected 0x08 0x10000L;
              Bytes.blit uuid 0 expected 0x18 16;
              (* mapping type of the single attribute entry *)
              Bytes.set_int32_le expected 0x28 1l;
              Bytes.set_int64_le expected 0x2418 1L;
              Bytes.set_int32_le expected 0x2420 7l;
              Bytes.set_int32_le expected 0x2424 0xc1d00001l;
              Bytes.set_int32_le expected 0x2428 0x5abc1234l;
              equal string (bytes_hex expected) (blob_hex b);
              raises_match is_invalid_arg (fun () ->
                  Nvk_iface.map_external_params ~rm_ctrl_fd:0 ~root:0 ~va:0n
                    ~size:0 ~mem_handle:0 ~gpu_uuid:(Bytes.create 8)));
          test "escape request codes embed the parameter sizes" (fun () ->
              equal int 0xc00446c9
                (Tables.escape_code ~nr:Defs.nv_esc_register_fd
                   ~size:Defs.Nv_ioctl_register_fd.sizeof);
              (* card enumeration passes an array of 64 entries *)
              equal int 0xd20046c8
                (Tables.escape_code ~nr:Defs.nv_esc_card_info
                   ~size:(64 * Defs.Nv_ioctl_card_info.sizeof));
              equal int 0xc020462a
                (Tables.escape_code ~nr:Defs.nv_esc_rm_control
                   ~size:Defs.Nvos54_parameters.sizeof);
              equal int 0xc0104629
                (Tables.escape_code ~nr:Defs.nv_esc_rm_free
                   ~size:Defs.Nvos00_parameters.sizeof);
              (* the dma-mapping parameters grew at 580, moving the code *)
              equal int 0xc0384657
                (Tables.escape_code ~nr:Defs.nv_esc_rm_map_memory_dma
                   ~size:0x38);
              equal int 0xc0404657
                (Tables.escape_code ~nr:Defs.nv_esc_rm_map_memory_dma
                   ~size:0x40));
        ];
      group "driver version"
        [
          test "the reported version selects the layout generation" (fun () ->
              equal int 570
                (Nvk_iface.driver_version_major (version_blob "570.144.03"));
              let sel s =
                Tables.defs_for_driver
                  ~major:(Nvk_iface.driver_version_major (version_blob s))
              in
              is_true ~msg:"570" (sel "570.144.03" == Tables.Versions.v570);
              is_true ~msg:"575" (sel "575.51.02" == Tables.Versions.v570);
              is_true ~msg:"580" (sel "580.65.06" == Tables.Versions.v580);
              is_true ~msg:"609" (sel "609.1" == Tables.Versions.v580);
              is_true ~msg:"615" (sel "615.29" == Tables.Versions.v610);
              raises_match
                (function Failure _ -> true | _ -> false)
                (fun () ->
                  Nvk_iface.driver_version_major (version_blob "unknown")));
        ];
      group "va allocator"
        [
          test "cpu-visible ranges come from the low window" (fun () ->
              let a =
                Nativeint.to_int
                  (Nvk_iface.alloc_gpu_vaddr ~force_low:true 0x4000)
              in
              let b =
                Nativeint.to_int
                  (Nvk_iface.alloc_gpu_vaddr ~force_low:true 0x4000)
              in
              is_true ~msg:"low base" (a >= 0x1000000000);
              is_true ~msg:"below the split" (b + 0x4000 <= 0x2000000000);
              is_true ~msg:"disjoint" (b >= a + 0x4000);
              equal int 0 (a land 0xfff));
          test "device-only ranges come from above the split" (fun () ->
              let a = Nativeint.to_int (Nvk_iface.alloc_gpu_vaddr 0x1000) in
              is_true ~msg:"high base" (a >= 0x2000000000);
              let b =
                Nativeint.to_int
                  (Nvk_iface.alloc_gpu_vaddr ~alignment:0x200000 0x1000)
              in
              equal int 0 (b land 0x1fffff);
              is_true ~msg:"disjoint" (b >= a + 0x1000));
        ];
      group "program image"
        [
          test "image parses a hand-built kernel object" (fun () ->
              with_fixture (fun m ->
                  let data = Program.image ~name:"k" (cubin_fixture ()) in
                  (* image 0x1a29c rounds to 0x1b000 plus the 4 KiB guard *)
                  equal int 0x1c000 (Bytes.length data.image);
                  equal int 32 data.regs_usage;
                  equal int 0x480 data.shmem_usage;
                  equal int 0x380 data.lcmem_usage;
                  equal int 0x160 (snd (List.assoc 0 data.constbufs));
                  equal int 0x160 data.cbuf0_size;
                  equal int 2 (List.length data.constbufs);
                  equal (pair int int) (0x12000, 0x160) (List.assoc 0 data.constbufs);
                  equal (pair int int) (0x1a000, 0x200) (List.assoc 3 data.constbufs);
                  let _, prefix = Program.template
                      (nv_dev ~sass_version:0x89 ~slm_per_thread:0x240 m) data in
                  equal int 88 (Array.length prefix);
                  equal (array int) [|0; 0x7294; 0; 0x7293; 0xfffdc0; 0|]
                    (Array.sub prefix 6 6);
                  equal int32 0xccccccccl (Bytes.get_int32_le data.image 0x2000);
                  equal int32 0xaaaaaaaal (Bytes.get_int32_le data.image 0x12000);
                  equal int32 0xbbbbbbbbl (Bytes.get_int32_le data.image 0x1a000)));
          test "linking patches high image addresses and initializes the prefetch guard" (fun () ->
              with_fixture (fun m ->
                  List.iter (fun image_address ->
                      let compiled, _, _, buffers, _ = queue_fixture ~image_address
                          ~compute_class:Defs.ada_compute_a ~copies:false m in
                      let linked = Tolk.Realize.link_linear compiled in
                      let image = Tolk.Device.Buffer.as_bytes (Hashtbl.find buffers "program") in
                      let target = Int64.add (Int64.of_nativeint image_address) 0x12010L in
                      equal int64 target (Bytes.get_int64_le image 0x2100);
                      equal int32 (Int64.to_int32 target) (Bytes.get_int32_le image 0x2204);
                      equal int32 (Int64.to_int32 (Int64.shift_right_logical target 32))
                        (Bytes.get_int32_le image 0x2304);
                      equal bytes (Bytes.make 4096 '\000') (Bytes.sub image 0x1b000 4096);
                      ignore (Sys.opaque_identity linked)) [0xfe000n; 0x8000fe000n]));
          test "invalid objects fail before linked allocations" (fun () ->
              with_fixture (fun m ->
                  raises_match (failure_with "unknown NV reloc 55") (fun () ->
                      queue_fixture ~lib:(cubin_fixture ~reloc0:0x37 ())
                        ~compute_class:Defs.ada_compute_a ~copies:false m)));
          test "address-independent templates retain the qmd_init fields" (fun () ->
              with_fixture (fun m ->
                  let data = Program.image ~name:"k" (cubin_fixture ()) in
                  List.iter (fun (compute_class, sass_version, expected) ->
                      let dev = nv_dev ~compute_class ~sass_version ~slm_per_thread:0x240 m in
                      let template, _ = Program.template dev data in
                      let bytes = Bytes.create (Array.length expected * 4) in
                      Array.iteri (fun i word -> Bytes.set_int32_le bytes (i * 4) (Int32.of_int word)) expected;
                      let view = Mmio.view m ~off:0x6000 ~size:(Bytes.length bytes) () in
                      Mmio.blit_bytes view ~off:0 bytes;
                      let reference = Qmd.create ~compute_class ~view in
                      Qmd.set_constant_buf_addr reference 0 0n;
                      Qmd.set_constant_buf_addr reference 3 0n;
                      let suffix = if Qmd.version reference < 4 then "" else "_shifted4" in
                      Qmd.write reference ["program_address_lower" ^ suffix, 0;
                        "program_address_upper" ^ suffix, 0;
                        "program_prefetch_addr_lower_shifted", 0;
                        "program_prefetch_addr_upper_shifted", 0];
                      equal (array int) (qmd_template_dwords reference) (qmd_template_dwords template))
                    [Defs.ada_compute_a, 0x89, qmd_expected_ada;
                     Defs.blackwell_compute_b, 0xa4, qmd_expected_blackwell]));
          test "unsupported objects fail loudly" (fun () ->
              raises_match (failure_with "unknown NV reloc 55") (fun () ->
                  Program.image ~name:"k" (cubin_fixture ~reloc0:0x37 ()));
              raises_match (failure_with "Attempting to relocate against an undefined symbol c0")
                (fun () -> Program.image ~name:"k" (cubin_fixture ~undefined_sym:true ()));
              raises_match (failure_with "unknown EIATTR format 7")
                (fun () -> Program.image ~name:"k" (cubin_fixture ~bad_info:true ())));
          test "Blackwell templates use the wide driver-parameter layout" (fun () ->
              with_fixture (fun m ->
                  let data = Program.image ~name:"k" (cubin_fixture ()) in
                  let _, prefix = Program.template
                      (nv_dev ~compute_class:Defs.blackwell_compute_b
                         ~sass_version:0xa4 ~slm_per_thread:0x240 m) data in
                  equal int 224 (Array.length prefix);
                  equal (array int) [|0; 0x7294; 0; 0x7293|] (Array.sub prefix 188 4);
                  equal int 0xfffdc0 prefix.(223)));
        ];
      group "local memory"
        [
          test "growing sizes the store and submits setup through the compiled queue" (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev m in
                  let allocator, allocs, frees, _, _ = slm_allocator () in
                  let _, device, _, buffers, _ = queue_fixture ~allocator
                      ~compute_class:Defs.ada_compute_a ~copies:false m in
                  let ensure size = Tolk_nv.ensure_has_local_memory dev
                      ~num_gpcs:2 ~num_tpc_per_gpc:3 ~num_sm_per_tpc:2
                      ~max_warps_per_sm:48 ~device size in
                  ensure 0x100;
                  equal int 0x100 dev.Tolk_nv.slm_per_thread;
                  equal (list int) [0x480000] !allocs;
                  equal (list int) [] !frees;
                  let backing = Option.get dev.Tolk_nv.shader_local_mem in
                  equal int 0x480000 (Tolk.Device.Buffer.nbytes backing);
                  let get tag = Tolk.Device.Buffer.as_bytes (Hashtbl.find buffers tag) in
                  equal int64 1L (Bytes.get_int64_le (get "timeline") 8);
                  equal int32 1l (Bytes.get_int32_le (get "gpput_compute") 0);
                  let stream = get "cmdbuf_compute" in
                  (* A six-dword timeline wait precedes the two setup packets. *)
                  let address = Int64.of_nativeint (Tolk.Device.Buffer.addr backing) in
                  equal int32 (Int64.to_int32 (Int64.shift_right_logical address 32))
                    (Bytes.get_int32_le stream (7 * 4));
                  equal int32 (Int64.to_int32 address) (Bytes.get_int32_le stream (8 * 4));
                  equal int32 0xc0000l (Bytes.get_int32_le stream (11 * 4));
                  ensure 0x80;
                  equal int 1 (List.length !allocs);
                  equal int64 1L (Bytes.get_int64_le (get "timeline") 8);
                  ensure 0x200;
                  equal (list int) [0x480000] !frees;
                  equal int 0x200 dev.Tolk_nv.slm_per_thread;
                  equal int64 2L (Bytes.get_int64_le (get "timeline") 8)));
          test "failed growth preserves the existing store and rejects the request" (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev m in
                  let allocator, allocs, frees, fail_next, _ = slm_allocator () in
                  let _, device, _, buffers, _ = queue_fixture ~allocator
                      ~compute_class:Defs.ada_compute_a ~copies:false m in
                  let ensure size = Tolk_nv.ensure_has_local_memory dev
                      ~num_gpcs:2 ~num_tpc_per_gpc:3 ~num_sm_per_tpc:2
                      ~max_warps_per_sm:48 ~device size in
                  ensure 0x100;
                  let old = Option.get dev.Tolk_nv.shader_local_mem in
                  fail_next := true;
                  raises_match (function Nv_iface.Out_of_memory _ -> true | _ -> false)
                    (fun () -> ensure 0x200);
                  equal (list int) [0x480000; 0x900000] (List.rev !allocs);
                  equal (list int) [] !frees;
                  equal int 0x100 dev.Tolk_nv.slm_per_thread;
                  is_true (Option.get dev.Tolk_nv.shader_local_mem == old);
                  equal int64 1L (Bytes.get_int64_le
                    (Tolk.Device.Buffer.as_bytes (Hashtbl.find buffers "timeline")) 8)));
          test "failed setup completion retains old capacity and both allocations" (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev m in
                  let fail_wait = ref false and faulted = ref false in
                  let error = Failure "scripted NV completion failure" in
                  let check () = if !faulted then raise error in
                  let allocator, allocs, frees, _, attempts = slm_allocator ~synchronize:check () in
                  let _, device, _, _, _ = queue_fixture ~allocator
                      ~synchronize:(fun () ->
                        if !fail_wait then faulted := true;
                        check ())
                      ~compute_class:Defs.ada_compute_a ~copies:false m in
                  let ensure size = Tolk_nv.ensure_has_local_memory dev
                      ~num_gpcs:2 ~num_tpc_per_gpc:3 ~num_sm_per_tpc:2
                      ~max_warps_per_sm:48 ~device size in
                  ensure 0x100;
                  let old = Option.get dev.Tolk_nv.shader_local_mem in
                  fail_wait := true;
                  raises error (fun () -> ensure 0x200);
                  equal int 0x100 dev.Tolk_nv.slm_per_thread;
                  is_true (Option.get dev.Tolk_nv.shader_local_mem == old);
                  equal (list int) [0x480000; 0x900000] (List.rev !allocs);
                  equal (list int) [] !frees;
                  raises error (fun () -> Tolk.Device.synchronize device);
                  raises error (fun () -> Tolk.Device.Buffer.deallocate old);
                  equal (list int) [] !frees;
                  (* Drain finalizers while the injected fault is latched. Each
                     failed free is retained by Storage, never returned to the allocator. *)
                  let rec collect () =
                    match Tolk_uop.Storage.with_operation (fun () -> Gc.full_major ()) with
                    | () -> ()
                    | exception exn when exn = error -> collect () in
                  collect ();
                  is_true (List.mem 0x900000 !attempts);
                  equal (list int) [] !frees;
                  (* Reset only the fake allocator after checking quarantine,
                     so unrelated later finalizers do not inherit this fault. *)
                  faulted := false));
          test "out of memory before initial setup does not submit" (fun () ->
              with_fixture (fun m ->
                  let dev = nv_dev m in
                  let allocator, _, frees, fail_next, _ = slm_allocator () in
                  fail_next := true;
                  let _, device, _, buffers, _ = queue_fixture ~allocator
                      ~compute_class:Defs.ada_compute_a ~copies:false m in
                  raises_match (function Nv_iface.Out_of_memory _ -> true | _ -> false)
                    (fun () -> Tolk_nv.ensure_has_local_memory dev
                      ~num_gpcs:2 ~num_tpc_per_gpc:3 ~num_sm_per_tpc:2
                      ~max_warps_per_sm:48 ~device 0x10);
                  equal int 0 dev.Tolk_nv.slm_per_thread;
                  is_true (Option.is_none dev.Tolk_nv.shader_local_mem);
                  equal int 0 (Hashtbl.length buffers);
                  equal (list int) [] !frees));
        ];
      group "compiled launch validation"
        [
          test "compiled launches patch 3D geometry and profiling releases" (fun () ->
              with_fixture (fun m ->
                  let open Tolk in
                  let compiled, device, _, buffers, submission = queue_fixture ~profile:true
                      ~global_size:[U.Launch_int 4; U.Launch_int 3; U.Launch_int 2]
                      ~local_size:[U.Launch_int 8; U.Launch_int 4; U.Launch_int 1]
                      ~compute_class:Defs.ada_compute_a ~copies:false m in
                  let linked = Realize.link_linear compiled in
                  let input = Device.create_buffer ~size:16 ~dtype:D.int32 device in
                  Realize.run_linear ~device
                    ~to_program:(fun device -> Codegen.to_program ~beam_device:device (Device.renderer device))
                    ~jit:true ~var_vals:["small", -17L; "count", 3L]
                    ~input_uops:[|U.from_buffer input|] linked;
                  Submission.check submission;
                  let buffer = Hashtbl.find buffers "qmd" in
                  let qmd = Qmd.create ~compute_class:Defs.ada_compute_a
                      ~view:(Mmio.make ~addr:(Device.Buffer.addr buffer) ~size:(Device.Buffer.nbytes buffer)) in
                  List.iter (fun (field, expected) -> equal ~msg:field int expected (Qmd.read qmd field))
                    ["cta_raster_width", 4; "cta_raster_height", 3; "cta_raster_depth", 2;
                     "cta_thread_dimension0", 8; "cta_thread_dimension1", 4; "cta_thread_dimension2", 1;
                     "release0_enable", 1; "release1_enable", 1;
                     "release0_structure_size", 0; "release1_structure_size", 2];
                  let bytes = Device.Buffer.as_bytes buffer in
                  equal int64 (Int64.of_nativeint (Device.Buffer.addr input))
                    (Bytes.get_int64_le bytes (0x100 + 0x160));
                  equal int (-17) (Bytes.get_int8 bytes (0x100 + 0x160 + 8));
                  equal int64 3L (Bytes.get_int64_le bytes (0x100 + 0x160 + 16))));
          test "launch limits fail before linking command storage" (fun () ->
              with_fixture (fun m ->
                  let compile ?lib global_size local_size =
                    queue_fixture ?lib ~global_size ~local_size
                      ~compute_class:Defs.ada_compute_a ~copies:false m in
                  let dims a b c = [U.Launch_int a; U.Launch_int b; U.Launch_int c] in
                  raises (Invalid_argument "NV queue: invalid launch dimensions")
                    (fun () -> ignore (compile (dims 1 1 1) (dims 1 1 65)));
                  raises (Invalid_argument "NV queue: invalid launch dimensions")
                    (fun () -> ignore (compile (dims 1 0x10000 1) (dims 1 1 1)));
                  raises (Invalid_argument "NV queue: too many threads for the kernel's register allocation")
                    (fun () -> ignore (compile (dims 1 1 1) (dims 16 16 8)));
                  raises (Invalid_argument "NV queue: too many threads for the kernel's register allocation")
                    (fun () -> ignore (compile ~lib:(cubin_fixture ~regcount:256 ())
                        (dims 1 1 1) (dims 32 32 1)))));
        ];
      group "cubin fixture"
        [
          test "the recorded nvrtc kernel parses to its recorded fields"
            (fun () ->
              let dir = "../fixtures/nv" in
              let cubin = Filename.concat dir "simple_add_sm89.cubin" in
              let fields_file = Filename.concat dir "simple_add_sm89.fields" in
              let fields =
                List.filter_map
                  (fun line ->
                    match
                      String.split_on_char ' ' (String.trim line)
                    with
                    | [ k; v ] -> Some (k, v)
                    | _ -> None)
                  (String.split_on_char '\n'
                     (In_channel.with_open_bin fields_file
                        In_channel.input_all))
              in
              let fint k = int_of_string (List.assoc k fields) in
              let lib =
                Bytes.of_string
                  (In_channel.with_open_bin cubin In_channel.input_all)
              in
              with_fixture (fun m ->
                  let dev =
                    nv_dev ~sass_version:0x89 ~slm_per_thread:0x2000 m
                  in
                  let data = Program.image ~name:(List.assoc "name" fields) lib in
                  let _, prefix = Program.template dev data in
                  equal int (fint "regs_usage") data.regs_usage;
                  equal int (fint "shmem_usage") data.shmem_usage;
                  equal int (fint "lcmem_usage") data.lcmem_usage;
                  equal int (fint "constbuf0_size") (snd (List.assoc 0 data.constbufs));
                  equal int (fint "cbuf0_size") (Array.length prefix * 4)));
        ];
      group "device info"
        [
          test "arch and sass derive from the sm version" (fun () ->
              equal string "sm_89" (Tolk_nv.arch_of_sm_version 0x809);
              equal string "sm_86" (Tolk_nv.arch_of_sm_version 0x806);
              equal string "sm_120" (Tolk_nv.arch_of_sm_version 0xa04);
              (* a revision byte above 0xf keeps only its high nibble *)
              equal string "sm_91" (Tolk_nv.arch_of_sm_version 0x91f);
              equal int 0x89 (Tolk_nv.sass_of_sm_version 0x809);
              equal int 0xa4 (Tolk_nv.sass_of_sm_version 0xa04);
              equal int 0x9f (Tolk_nv.sass_of_sm_version 0x91f));
          test "topology reads through the info list" (fun () ->
              let module I = Defs.Nv2080_ctrl_gr_info in
              let module P = Defs.Nv2080_ctrl_gr_get_info_params in
              let rm_control ~obj ~cmd ?params () =
                equal ~msg:"obj" int 0x5d obj;
                equal ~msg:"cmd" int Defs.nv2080_ctrl_cmd_gr_get_info cmd;
                let b = Option.get params in
                equal ~msg:"list size" int 5 (Tables.get_field b P.grinfolistsize);
                let infos =
                  Mmio.make
                    ~addr:(Nativeint.of_int (Tables.get_field b P.grinfolist))
                    ~size:(5 * I.sizeof)
                in
                List.iteri
                  (fun i idx ->
                    equal ~msg:"index" int idx
                      (Int32.to_int (Mmio.read32 infos (i * I.sizeof)));
                    Mmio.write32 infos
                      ((i * I.sizeof) + fst I.data)
                      (Int32.of_int (100 + i)))
                  topology_indices
              in
              equal (list int)
                [ 100; 101; 102; 103; 104 ]
                (Tolk_nv.query_gpu_info
                   (fake_iface ~rm_control ())
                   ~subdevice:0x5d topology_indices));
          test "the driver-less arm reads the static engine info" (fun () ->
              let module P = Defs.Nv2080_ctrl_internal_static_gr_get_info_params
              in
              let module I = Defs.Nv2080_ctrl_internal_static_gr_info in
              let rm_control ~obj ~cmd ?params () =
                equal ~msg:"obj" int 0x5d obj;
                equal ~msg:"cmd" int
                  Defs.nv2080_ctrl_cmd_internal_static_kgr_get_info cmd;
                let b = Option.get params in
                equal ~msg:"size" int P.sizeof (Bigarray.Array1.dim b);
                List.iteri
                  (fun i idx ->
                    Tables.set_field
                      ~base:
                        (P.engineinfo_offset + I.infolist_offset
                        + (idx * I.infolist_elem_size))
                      b Defs.Nv2080_ctrl_internal_gr_info.data (200 + i))
                  topology_indices
              in
              equal (list int)
                [ 200; 201; 202; 203; 204 ]
                (Tolk_nv.query_gpu_info
                   (fake_iface ~nvdev:Fake_nvdev ~rm_control ())
                   ~subdevice:0x5d topology_indices));
        ];
      group "device hang"
        [
          test "a queued MMU fault reports each fault decoded by name"
            (fun () ->
              let module P =
                Defs.Nv83de_ctrl_debug_read_all_sm_error_states_params
              in
              let module M = Defs.Nv83de_ctrl_debug_read_mmu_fault_info_params
              in
              let module E = Defs.Nv83de_ctrl_debug_read_mmu_fault_info_entry
              in
              let seen = ref [] in
              let rm_control ~obj ~cmd ?params () =
                seen := (obj, cmd) :: !seen;
                let b = Option.get params in
                if cmd = Defs.nv83de_ctrl_cmd_debug_read_all_sm_error_states
                then begin
                  equal ~msg:"channel" int 0xc
                    (Tables.get_field b P.htargetchannel);
                  equal ~msg:"sms" int 100 (Tables.get_field b P.numsmstoread);
                  Tables.set_field b P.mmufault_valid 1
                end
                else begin
                  Tables.set_field b M.count 2;
                  Tables.set_field b E.faultaddress 0xdead0000;
                  Tables.set_field b E.faulttype 2;
                  Tables.set_field b E.accesstype 1;
                  Tables.set_field ~base:E.sizeof b E.faultaddress 0xbeef;
                  (* an id outside the name tables falls back to the number *)
                  Tables.set_field ~base:E.sizeof b E.faulttype 99;
                  Tables.set_field ~base:E.sizeof b E.accesstype 0
                end
              in
              raises_match
                (function
                  | Failure m ->
                      String.equal m
                        "MMU fault: 0xDEAD0000 | NV_PFAULT_FAULT_TYPE_PTE | \
                         WRITE\n\
                         MMU fault: 0xBEEF | 99 | READ"
                  | _ -> false)
                (fun () ->
                  Tolk_nv.on_device_hang
                    (fake_iface ~rm_control ())
                    ~debugger:0xd ~debug_channel:0xc ());
              equal
                (list (pair int int))
                [
                  (0xd, Defs.nv83de_ctrl_cmd_debug_read_all_sm_error_states);
                  (0xd, Defs.nv83de_ctrl_cmd_debug_read_mmu_fault_info);
                ]
                (List.rev !seen));
          test "without an MMU fault the latched SM errors are reported"
            (fun () ->
              let module P =
                Defs.Nv83de_ctrl_debug_read_all_sm_error_states_params
              in
              let module R = Defs.Nv83de_sm_error_state_registers in
              let rm_control ~obj:_ ~cmd:_ ?params () =
                let b = Option.get params in
                let base = P.smerrorstatearray_offset + (3 * R.sizeof) in
                Tables.set_field ~base b R.hwwglobalesr 5;
                Tables.set_field ~base b R.hwwwarpesr 0xab;
                Tables.set_field ~base b R.hwwwarpesrpc64 0x1234
              in
              raises_match
                (function
                  | Failure m ->
                      String.equal m
                        "SM 3 fault: esr=5 warp_esr=0xab warp_pc=0x1234"
                  | _ -> false)
                (fun () ->
                  Tolk_nv.on_device_hang
                    (fake_iface ~rm_control ())
                    ~debugger:0xd ~debug_channel:0xc ()));
        ];
      group "Pci_iface"
        [
          test "the bus scan admits exactly the allowlisted ids" (fun () ->
              with_fake_sysfs
                [
                  (* two supported NVIDIA parts match; a non-GPU function, a
                     non-matching NVIDIA id, and a foreign vendor do not *)
                  ("0000:03:00.0", 0x10de, 0x2204);
                  ("0000:02:00.0", 0x10de, 0x2f00);
                  ("0000:01:00.0", 0x10de, 0x1234);
                  ("0000:03:00.1", 0x10de, 0x1c03);
                  ("0000:04:00.0", 0x1002, 0x2204);
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
                   (fun i id -> (Printf.sprintf "0000:%02x:00.0" i, 0x10de, id))
                   ids)
                (fun sysfs ->
                  equal int (List.length ids)
                    (List.length
                       (Tolk_hcq.System.pci_scan_bus ~sysfs
                          ~vendor:Pci_iface.vendor Pci_iface.pci_ids))));
          test "opening after the kernel driver is refused" (fun () ->
              if not (Sys.file_exists "/dev/nvidiactl") then
                skip ~reason:"no kernel driver to initialize first" ()
              else begin
                ignore (nvk_iface ());
                raises_match
                  (function
                    | Failure m -> contains ~needle:"after the kernel driver" m
                    | _ -> false)
                  (fun () -> ignore (Pci_iface.create ~device_id:0))
              end);
        ];
      group "device"
        [
          test "the interface opens and enumerates" (fun () ->
              let i = nvk_iface () in
              is_true ~msg:"count" (i.Nv_iface.count >= 1);
              is_true ~msg:"root" (i.Nv_iface.root <> 0);
              is_true ~msg:"instance" (i.Nv_iface.gpu_instance >= 0);
              is_true ~msg:"kernel driver" (not (Nv_iface.is_nvd i)));
          test "create opens the device and synchronize completes" (fun () ->
              let device = nv_device () in
              equal string "NV" (Tolk.Device.name device);
              Tolk.Device.synchronize device);
          test "dispatch maps a host view and preserves the external CPU allocation" (fun () ->
              let device = nv_device () in
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
                  let spec = Tolk.Device.compile_program device ~name:"nv_mapped_host"
                      (increment_program ()) in
                  ignore (time_spec device spec [dst; view]);
                  equal (list int) [42] (read_i32 dst);
                  Tolk.Device.Buffer.deallocate view;
                  Tolk.Device.Buffer.deallocate base;
                  Mmio.write32 mapping 4 99l;
                  equal int32 99l (Mmio.read32 mapping 4)));
          test "compiles and runs one kernel" (fun () ->
              let device = nv_device () in
              let spec =
                Tolk.Device.compile_program device ~name:"nv_add_one"
                  (increment_program ())
              in
              let dst = i32_buf device [ 0 ] in
              let src = i32_buf device [ 41 ] in
              let elapsed = time_spec device spec [dst; src] in
              is_true (Float.is_finite elapsed && elapsed > 0.0);
              Tolk.Device.synchronize device;
              equal (list int) [ 42 ] (read_i32 dst));
        ];
    ]
