(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The driver-less AMD device core, on pieces that run without hardware:
   the IP-discovery parser on synthesized tables, the page-table entry
   encoding against hand-computed golden words, page-table walks over an
   anonymous mapping standing in for VRAM, named-bitfield register
   access over a fake register file, firmware loading and image
   splitting over synthesized firmware blobs, and the security-processor
   and power-management bring-up protocols over a scripted device. *)

open Windtrap
module Amdev = Tolk_amd.Amdev
module Firmware = Tolk_amd.Amdev.Firmware
module Am_ip = Tolk_amd.Am_ip
module Psp = Tolk_amd.Am_ip.Psp
module Smu = Tolk_amd.Am_ip.Smu
module Soc = Tolk_amd.Am_ip.Soc
module Gmc = Tolk_amd.Am_ip.Gmc
module Gfx = Tolk_amd.Am_ip.Gfx
module Ih = Tolk_amd.Am_ip.Ih
module Sdma = Tolk_amd.Am_ip.Sdma
module Am = Tolk_amd.Amd_tables.Am_defs
module Fw_defs = Tolk_amd.Amd_tables.Fw_defs
module Reg = Tolk_amd.Amd_tables.Reg
module Mmio = Tolk_hcq.Hcq.Mmio
module File_io = Tolk_hcq.Hcq.File_io
module Memory = Tolk.Memory
module Tlsf = Tolk.Tlsf

(* IP discovery fixtures: tables synthesized from the same struct
   layouts the parser reads, with the ip-discovery header at 0x100, the
   die at 0x200 and the gc table at 0x400. *)

let s8 = Bytes.set_uint8
let s16 = Bytes.set_uint16_le
let s32 b p v = Bytes.set_int32_le b p (Int32.of_int v)

let add_ip ~base64 b pos ~hw_id ~inst ~ver:(ma, mi, rv) bases =
  s16 b pos hw_id;
  s8 b (pos + 2) inst;
  s8 b (pos + 3) (List.length bases);
  s8 b (pos + 4) ma;
  s8 b (pos + 5) mi;
  s8 b (pos + 6) rv;
  List.iteri
    (fun i v ->
      if base64 then Bytes.set_int64_le b (pos + 8 + (i * 8)) (Int64.of_int v)
      else s32 b (pos + 8 + (i * 4)) v)
    bases;
  pos + 8 + ((if base64 then 8 else 4) * List.length bases)

let discovery_blob ?(base64 = false) ips =
  let b = Bytes.make 0x800 '\x00' in
  s32 b 0 Am.binary_signature;
  s16 b
    (Am.Binary_header.table_list_offset
    + (Am.table_ip_discovery * Am.Table_info.sizeof))
    0x100;
  s16 b
    (Am.Binary_header.table_list_offset + (Am.table_gc * Am.Table_info.sizeof))
    0x400;
  s32 b 0x100 Am.discovery_table_signature;
  s16 b (0x100 + 0xc) 1;
  s16 b (0x100 + Am.Ip_discovery_header.die_info_offset + 2) 0x200;
  s8 b (0x100 + 0x4e) (if base64 then 1 else 0);
  s16 b 0x200 0;
  s16 b 0x202 (List.length ips);
  let pos = ref (0x200 + Am.Die_header.sizeof) in
  List.iter
    (fun (hw_id, inst, ver, bases) ->
      pos := add_ip ~base64 b !pos ~hw_id ~inst ~ver bases)
    ips;
  (* gc geometry table, major version 2 *)
  s16 b 0x404 2;
  s16 b 0x406 0;
  s32 b (0x400 + 0xc) 2;
  s32 b (0x400 + 0x10) 8;
  s32 b (0x400 + 0x14) 1;
  s32 b (0x400 + 0x3c) 16;
  s32 b (0x400 + 0x40) 32;
  s32 b (0x400 + 0x44) 64;
  b

let sample_ips =
  [
    (0xb, 0, (11, 0, 2), [ 0x1c000; 0x2400 ]);
    (0xb, 1, (11, 0, 2), [ 0x5000 ]);
    (0x6c, 0, (4, 3, 0), [ 0xd20 ]);
    (0xff, 0, (13, 0, 10), [ 0x1000 ]);
  ]

(* Page-table fixtures over an anonymous mapping standing in for the
   VRAM BAR. *)

let with_fake_vram size f =
  let addr =
    File_io.mmap ~addr:0n ~size
      ~prot:(File_io.prot_read lor File_io.prot_write)
      ~flags:(File_io.map_private lor File_io.map_anonymous)
      ~fd:(-1) ~offset:0L
  in
  Fun.protect
    ~finally:(fun () -> File_io.munmap addr ~size)
    (fun () -> f (Mmio.make ~addr ~size))

let make_mm ~gc_ver ?paddr_base vram =
  let ops = Amdev.Am_page_table.ops ~vram ~gc_ver ?paddr_base () in
  let booting = ref true in
  let mm =
    Memory.create ~pt_ops:ops ~vram_size:(Mmio.size vram) ~boot_size:0x100000
      ~va_bits:48
      ~va_shifts:[ 12; 21; 30; 39 ]
      ~va_base:0
      ~palloc_ranges:[ (0x200000, 0x200000); (0x1000, 0x1000) ]
      ~va_allocator:(Tlsf.create ~size:0x40000000 ~base:0 ())
      ~is_booting:(fun () -> !booting)
      ~zero_vram:(fun ~paddr ~size ->
        Mmio.blit_bytes vram ~off:paddr (Bytes.make size '\000'))
      ~first_lv:Am.amdgpu_vm_pdb2 ()
  in
  booting := false;
  (mm, ops)

let flags ~gc_ver ?(lv = Am.amdgpu_vm_ptb) ?(table = false) ?(frag = 0)
    ?(uncached = false) ?(system = false) ?(snooped = false) ?(valid = true) ()
    =
  Amdev.Am_page_table.pte_flags ~gc_ver ~lv ~table ~frag ~uncached ~system
    ~snooped ~valid

(* Register fixture: a register file backed by a hash table. *)

let fake_register () =
  let store = Hashtbl.create 4 in
  let r =
    {
      Reg.name = "regTEST";
      offset = 0x10;
      segment = 0;
      fields = [| ("lo", (0, 7)); ("mid", (8, 19)); ("hi", (20, 31)) |];
      addr = 0x123;
    }
  in
  let amr =
    Amdev.Am_register.make ~reg:r
      ~rreg:(fun a -> Option.value ~default:0 (Hashtbl.find_opt store a))
      ~wreg:(fun a v -> Hashtbl.replace store a v)
  in
  (amr, store)

(* Firmware fixtures: blobs synthesized at the struct offsets the
   parser reads, with distinctive payload tags so descriptor slices can
   be asserted whole. *)

let put b off s = Bytes.blit_string s 0 b off (String.length s)

let common_header b ~ver:(ma, mi) ~ucode_off ~ucode_size =
  s16 b 8 ma;
  s16 b 0xa mi;
  s32 b 0x14 ucode_size;
  s32 b 0x18 ucode_off

(* A v2_0 sOS container with two components. *)
let sos_blob () =
  let b = Bytes.make 0x140 '\x00' in
  common_header b ~ver:(2, 0) ~ucode_off:0x100 ~ucode_size:0;
  s32 b 0x20 2;
  (* component descriptors at the v2_0 bin offset 0x24 *)
  s32 b 0x24 2;
  s32 b 0x2c 0;
  s32 b 0x30 4;
  s32 b 0x34 3;
  s32 b 0x3c 4;
  s32 b 0x40 8;
  put b 0x100 "SYS!";
  put b 0x104 "KDB-DATA";
  b

let smu_blob_gfx11 () =
  let b = Bytes.make 0x60 '\x00' in
  common_header b ~ver:(1, 0) ~ucode_off:0x40 ~ucode_size:8;
  put b 0x40 "SMUCODE!";
  b

(* A v2_1 SMU header whose soft-pptable list holds one P2S table. *)
let smu_blob_gfx9 () =
  let b = Bytes.make 0x100 '\x00' in
  common_header b ~ver:(2, 1) ~ucode_off:0 ~ucode_size:0;
  s32 b 0x24 2;
  s32 b 0x28 0x40;
  s32 b 0x40 0x11223344;
  s32 b 0x44 0x98;
  s32 b 0x48 4;
  s32 b 0x4c 0x50325358;
  s32 b 0x50 0x90;
  s32 b 0x54 6;
  put b 0x90 "P2STAB";
  b

let sdma_blob_v1 () =
  let b = Bytes.make 0x60 '\x00' in
  common_header b ~ver:(1, 0) ~ucode_off:0x40 ~ucode_size:12;
  put b 0x40 "SDMA-CODE-12";
  b

let sdma_blob_v2 () =
  let b = Bytes.make 0x80 '\x00' in
  common_header b ~ver:(2, 0) ~ucode_off:0x40 ~ucode_size:0;
  s32 b 0x24 8;
  s32 b 0x30 0x60;
  s32 b 0x34 6;
  put b 0x40 "CTXCODE!";
  put b 0x60 "CTLCOD";
  b

let sdma_blob_v3 () =
  let b = Bytes.make 0x60 '\x00' in
  common_header b ~ver:(3, 0) ~ucode_off:0x40 ~ucode_size:0;
  s32 b 0x28 8;
  put b 0x40 "SDMA3TH0";
  b

(* A v1_0 GFX header: code with a trailing jump table. *)
let gfx_blob_v1 () =
  let b = Bytes.make 0x80 '\x00' in
  common_header b ~ver:(1, 0) ~ucode_off:0x40 ~ucode_size:0x20;
  s32 b 0x24 6;
  s32 b 0x28 2;
  put b 0x40 "MEC-V1-CODE-24-BYTES-OK!";
  put b 0x58 "MECJTAB!";
  b

(* A v2_0 GFX header: code, stack data, and a start address. *)
let gfx_blob_v2 ~code ~stack ~start_lo ~start_hi =
  let b = Bytes.make 0x80 '\x00' in
  common_header b ~ver:(2, 0) ~ucode_off:0x40 ~ucode_size:0;
  s32 b 0x24 8;
  s32 b 0x2c 8;
  s32 b 0x30 0x60;
  s32 b 0x34 start_lo;
  s32 b 0x38 start_hi;
  put b 0x40 code;
  put b 0x60 stack;
  b

let imu_blob () =
  let b = Bytes.make 0x60 '\x00' in
  common_header b ~ver:(1, 0) ~ucode_off:0x40 ~ucode_size:0;
  s32 b 0x20 8;
  s32 b 0x28 4;
  put b 0x40 "IMUIRAM!";
  put b 0x48 "IMUD";
  b

let rlc_blob_v2_1 () =
  let b = Bytes.make 0x140 '\x00' in
  common_header b ~ver:(2, 1) ~ucode_off:0x100 ~ucode_size:8;
  s32 b 0x74 4;
  s32 b 0x78 0x110;
  s32 b 0x84 8;
  s32 b 0x88 0x118;
  s32 b 0x94 4;
  s32 b 0x98 0x120;
  put b 0x100 "RLCGCODE";
  put b 0x110 "CNTL";
  put b 0x118 "GPMLIST!";
  put b 0x120 "SRM!";
  b

let rlc_blob_v2_3 () =
  let b = Bytes.make 0x140 '\x00' in
  common_header b ~ver:(2, 3) ~ucode_off:0x100 ~ucode_size:8;
  s32 b 0x9c 8;
  s32 b 0xa0 0x110;
  s32 b 0xa4 4;
  s32 b 0xa8 0x118;
  s32 b 0xb4 8;
  s32 b 0xb8 0x120;
  s32 b 0xc4 4;
  s32 b 0xc8 0x128;
  put b 0x100 "RLCGCODE";
  put b 0x110 "RLCIRAM!";
  put b 0x118 "RLCD";
  put b 0x120 "RLCPCODE";
  put b 0x128 "RLCV";
  b

(* A loader over an in-memory file set that records what was asked. *)
let fw_loader files =
  let requested = ref [] in
  let load name =
    requested := name :: !requested;
    match List.assoc_opt name files with
    | Some b -> b
    | None -> fail ("unexpected firmware request " ^ name)
  in
  (load, fun () -> List.rev !requested)

let desc_strings descs =
  List.map (fun (types, data) -> (types, Bytes.to_string data)) descs

(* A firmware directory in a temp dir, handed to the loader's [dir]. *)
let with_fw_dir f =
  let dir = Filename.temp_file "tolk_fw_test" "" in
  Sys.remove dir;
  Sys.mkdir dir 0o700;
  Fun.protect
    ~finally:(fun () ->
      Array.iter
        (fun e -> Sys.remove (Filename.concat dir e))
        (Sys.readdir dir);
      Sys.rmdir dir)
    (fun () -> f dir)

let write_file path content =
  Out_channel.with_open_bin path (fun oc ->
      Out_channel.output_string oc content)

let sha_abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

(* Scripted devices for the IP-block protocols: a discovery table
   covering every register family the device core resolves, a register
   file over a hashtable with per-address read hooks and a write log,
   an anonymous mapping standing in for VRAM, and a clock that advances
   one millisecond per reading so waits and settle delays consume no
   wall time. *)

let mmhub_base = 0x3000
let mp0_base = 0x10000
let mp1_base = 0x20000

(* regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL sits at segment 2 offset 0x12d
   of the bus-interface family; scripted to hold byte address 0x54, so
   flushes write dword address 0x15. *)
let remap_hdp_addr = 0x42000 + 0x12d
let flush_target = 0x15

let dev_ips ~gc ~mp0 ~mp1 ~mmhub ~sdma ~bif ~osssys =
  [
    (0xb, 0, gc, [ 0x8000; 0x9000 ]);
    (0x29, 0, (6, 0, 0), [ 0xa000 ]);
    (0x2a, 0, sdma, [ 0xb000 ]);
    (0x22, 0, mmhub, [ mmhub_base ]);
    (0x6c, 0, bif, [ 0x40000; 0x41000; 0x42000; 0x43000; 0x44000; 0x45000 ]);
    (0xff, 0, mp0, [ mp0_base; 0x11000 ]);
    (1, 0, mp1, [ mp1_base ]);
    (0x28, 0, osssys, [ 0xc000 ]);
  ]

type fake_dev = {
  dev : Amdev.t;
  fvram : Mmio.t;
  store : (int, int) Hashtbl.t;
  reads : (int, unit -> int) Hashtbl.t;
  wr_hooks : (int, int -> unit) Hashtbl.t;
  log : (int * int) list ref;
}

let with_fake_dev ?(gc = (11, 0, 2)) ?(mp0 = (13, 0, 10))
    ?(mp1 = (13, 0, 10)) ?(mmhub = (3, 0, 0)) ?(sdma = (6, 0, 2))
    ?(bif = (4, 3, 0)) ?(osssys = (6, 0, 0)) ?(pre = fun _ -> ()) f =
  with_fake_vram 0x2000000 (fun vram ->
      let store = Hashtbl.create 16 in
      let reads = Hashtbl.create 16 in
      let wr_hooks = Hashtbl.create 16 in
      let log = ref [] in
      let clock = ref 0 in
      let rreg addr =
        match Hashtbl.find_opt reads addr with
        | Some hook -> hook ()
        | None -> Option.value ~default:0 (Hashtbl.find_opt store addr)
      in
      let wreg addr v =
        log := (addr, v) :: !log;
        Hashtbl.replace store addr v;
        match Hashtbl.find_opt wr_hooks addr with
        | Some hook -> hook v
        | None -> ()
      in
      (* The framebuffer base field reads as 0x10, so the memory
         controller sees local addresses at 0x10000000 + paddr; both
         family generations' register offsets are populated. *)
      Hashtbl.replace store (mmhub_base + 0x8ec) 0x10;
      Hashtbl.replace store (mmhub_base + 0xc9c) 0x10;
      Hashtbl.replace store remap_hdp_addr 0x54;
      pre store;
      let booting = ref true in
      let on_range_mapped = ref (fun () -> ()) in
      let mm =
        Memory.create
          ~pt_ops:(Amdev.Am_page_table.ops ~vram ~gc_ver:gc ())
          ~vram_size:(Mmio.size vram) ~boot_size:0x1800000 ~va_bits:48
          ~va_shifts:[ 12; 21; 30; 39 ]
          ~va_base:0
          ~palloc_ranges:[ (0x200000, 0x200000); (0x1000, 0x1000) ]
          ~va_allocator:(Tlsf.create ~size:0x40000000 ~base:0 ())
          ~is_booting:(fun () -> !booting)
          ~zero_vram:(fun ~paddr ~size ->
            Mmio.blit_bytes vram ~off:paddr (Bytes.make size '\000'))
          ~first_lv:Am.amdgpu_vm_pdb2
          ~on_range_mapped:(fun () -> !on_range_mapped ())
          ()
      in
      let dev =
        Amdev.make ~rreg ~wreg ~vram
          ~doorbell64:(Mmio.view vram ~off:0 ~size:0x1000 ())
          ~mmio:(Mmio.view vram ~off:0 ~size:0x1000 ())
          ~vram_size:(Mmio.size vram) ~large_bar:true ~reserved_vram_size:0
          ~discovery:
            (Amdev.parse_discovery
               (discovery_blob (dev_ips ~gc ~mp0 ~mp1 ~mmhub ~sdma ~bif ~osssys)))
          ~mm ~devfmt:"test"
          ~now_ms:(fun () ->
            incr clock;
            !clock)
          ~is_booting:booting ~on_range_mapped ()
      in
      f { dev; fvram = vram; store; reads; wr_hooks; log })

let raddr dev name = (Amdev.Am_register.reg (Amdev.reg dev name)).Reg.addr

let rencode fd name fields =
  Reg.encode (Amdev.Am_register.reg (Amdev.reg fd.dev name)) fields

let rdecode fd name v =
  Reg.decode (Amdev.Am_register.reg (Amdev.reg fd.dev name)) v

(* The last value written to a register, [0] when untouched. *)
let rstore fd name =
  Option.value ~default:0 (Hashtbl.find_opt fd.store (raddr fd.dev name))

let lo32 v = v land 0xffffffff
let hi32 v = v lsr 32

let hexdump b =
  String.concat ""
    (List.init (Bytes.length b) (fun i ->
         Printf.sprintf "%02x" (Bytes.get_uint8 b i)))

let equal_bytes expected actual = equal string (hexdump expected) (hexdump actual)

let has_substring s sub =
  let n = String.length sub in
  let rec go i =
    i + n <= String.length s && (String.sub s i n = sub || go (i + 1))
  in
  go 0

(* Expected PSP structures, built at the raw struct offsets so the
   implementation's encoders are checked against an independent
   derivation. *)

let psp_cmd id fields =
  let b = Bytes.make 0x400 '\x00' in
  s32 b 8 id;
  List.iter (fun (off, v) -> s32 b off v) fields;
  b

let rb_frame ~cmd_mc ~fence_mc ~fence_value =
  let b = Bytes.make 0x40 '\x00' in
  s32 b 0 (lo32 cmd_mc);
  s32 b 4 (hi32 cmd_mc);
  s32 b 0xc (lo32 fence_mc);
  s32 b 0x10 (hi32 fence_mc);
  s32 b 0x14 fence_value;
  b

(* An 8-byte firmware image as the staging buffer holds it: a 4-byte
   zero tail padded to 16 bytes. *)
let staged img =
  let b = Bytes.make 16 '\x00' in
  Bytes.blit_string img 0 b 0 (String.length img);
  Bytes.to_string b

(* Scripts the security processor's side of the protocol: the
   bootloader and secure OS always report ready, the OS reports alive
   once the sOS bootloader component was submitted, and each ring
   doorbell snapshots the submitted command frame and staging buffer,
   then completes the fence and answers a TMR size of 0x120000. *)

type psp_script = {
  r : int -> int;
  frames : bytes list ref;
  msg1s : string list ref;
}

let psp_script fd ~pref psp =
  let r n = raddr fd.dev (Printf.sprintf "%s_%d" pref n) in
  let frames = ref [] in
  let msg1s = ref [] in
  let sos_loaded = ref false in
  Hashtbl.replace fd.reads (r 35) (fun () -> 0x80000000);
  Hashtbl.replace fd.reads (r 64) (fun () -> 0x80000000);
  Hashtbl.replace fd.reads (r 81) (fun () -> Bool.to_int !sos_loaded);
  Hashtbl.replace fd.wr_hooks (r 35) (fun v ->
      msg1s :=
        Bytes.to_string
          (Mmio.read_bytes fd.fvram ~off:(Psp.msg1_paddr psp) ~len:16)
        :: !msg1s;
      if v = Am.psp_bl__load_sosdrv then sos_loaded := true);
  Hashtbl.replace fd.wr_hooks (r 67) (fun v ->
      frames :=
        Mmio.read_bytes fd.fvram ~off:(Psp.cmd_paddr psp)
          ~len:Am.Psp_gfx_cmd_resp.sizeof
        :: !frames;
      Mmio.write32 fd.fvram (Psp.cmd_paddr psp + 0x370) 0x120000l;
      Mmio.write32 fd.fvram (Psp.fence_paddr psp)
        (Int32.of_int (v - 0x10 + 1)));
  { r; frames; msg1s }

let no_fw =
  { Firmware.sos_fw = []; ucode_start = []; smu_psp_desc = None; descs = [] }

(* Addresses of the power-management message triple, and a script that
   acknowledges every message by raising the response register. *)
let smu_addrs fd =
  ( raddr fd.dev "mmMP1_SMN_C2PMSG_90",
    raddr fd.dev "mmMP1_SMN_C2PMSG_82",
    raddr fd.dev "mmMP1_SMN_C2PMSG_66" )

let ack_messages fd =
  let resp, _, msg = smu_addrs fd in
  Hashtbl.replace fd.wr_hooks msg (fun _ -> Hashtbl.replace fd.store resp 1)

(* Boot-machine fixtures: a firmware set covering every image the gfx11
   bring-up feeds somewhere — secure-OS components for the bootloader,
   ring-loaded images, and the compute-processor start address — plus
   the scripting a full boot polls beyond the security processor: the
   power-management block acknowledges every message and reports two
   clock levels, and the compute block's safe-mode handshake completes
   by clearing the command bit. *)

module Am_boot = Tolk_amd.Am_boot

let boot_fw =
  {
    Firmware.sos_fw =
      [
        (Am.psp_fw_type_psp_kdb, Bytes.of_string "KDBIMAGE");
        (Am.psp_fw_type_psp_sos, Bytes.of_string "SOSIMAGE");
      ];
    ucode_start = [ ("MEC", 0x2000) ];
    smu_psp_desc = Some ([ Am.gfx_fw_type_smu ], Bytes.of_string "SMUIMAGE");
    descs = [ ([ Am.gfx_fw_type_rlc_g ], Bytes.of_string "RLCIMAGE") ];
  }

let script_smu fd =
  ack_messages fd;
  let _, arg, _ = smu_addrs fd in
  Hashtbl.replace fd.reads arg (fun () -> 2)

let script_boot fd t =
  let s = psp_script fd ~pref:"regMP0_SMN_C2PMSG" t.Am_boot.psp in
  script_smu fd;
  let safe_mode = raddr fd.dev "regRLC_SAFE_MODE" in
  Hashtbl.replace fd.wr_hooks safe_mode (fun v ->
      Hashtbl.replace fd.store safe_mode (v land lnot 1));
  s

(* The chronological write log, and positions within it. *)
let writes fd = List.rev !(fd.log)

let first_write_to fd name log =
  let addr = raddr fd.dev name in
  let rec go i = function
    | [] -> fail (Printf.sprintf "no write to %s" name)
    | (a, _) :: rest -> if a = addr then i else go (i + 1) rest
  in
  go 0 log

let wrote fd name log = List.mem_assoc (raddr fd.dev name) log

let last_write log =
  match List.rev log with
  | last :: _ -> last
  | [] -> fail "empty write log"

(* The boot-session stamps must close the log, byte-exact: they are the
   contract another driver of the protocol reads back. *)
let check_boot_stamps fd log =
  equal int 0xA0000008 Am_boot.version;
  let rec last2 = function
    | [ a; b ] -> (a, b)
    | _ :: rest -> last2 rest
    | [] -> fail "empty write log"
  in
  let (r7, v7), (r6, v6) = last2 log in
  equal int (raddr fd.dev "regSCRATCH_REG7") r7;
  equal int 0xA0000008 v7;
  equal int (raddr fd.dev "regSCRATCH_REG6") r6;
  equal int 1 v6

let () =
  run "Amdev"
    [
      group "ip discovery"
        [
          test "parses versions and per-instance bases" (fun () ->
              let d = Amdev.parse_discovery (discovery_blob sample_ips) in
              equal
                (triple int int int)
                (11, 0, 2)
                (List.assoc Am.gc_hwip d.Amdev.ip_ver);
              equal
                (triple int int int)
                (13, 0, 10)
                (List.assoc Am.mp0_hwip d.Amdev.ip_ver);
              equal
                (list (pair int (array int)))
                [ (0, [| 0x1c000; 0x2400 |]); (1, [| 0x5000 |]) ]
                (List.assoc Am.gc_hwip d.Amdev.regs_offset));
          test "one bus-interface entry feeds both nbio and nbif" (fun () ->
              let d = Amdev.parse_discovery (discovery_blob sample_ips) in
              equal
                (triple int int int)
                (4, 3, 0)
                (List.assoc Am.nbio_hwip d.Amdev.ip_ver);
              equal
                (triple int int int)
                (4, 3, 0)
                (List.assoc Am.nbif_hwip d.Amdev.ip_ver);
              equal
                (list (pair int (array int)))
                [ (0, [| 0xd20 |]) ]
                (List.assoc Am.nbif_hwip d.Amdev.regs_offset));
          test "reads the gc geometry table" (fun () ->
              let d = Amdev.parse_discovery (discovery_blob sample_ips) in
              match d.Amdev.gc_info with
              | Amdev.Gc_info_v2 g ->
                  equal int 2 g.num_se;
                  equal int 8 g.num_cu_per_sh;
                  equal int 1 g.num_sh_per_se;
                  equal int 16 g.max_waves_per_simd;
                  equal int 32 g.max_scratch_slots_per_cu;
                  equal int 64 g.lds_size
              | Amdev.Gc_info_v1 _ -> fail "expected a v2 gc table");
          test "decodes 64-bit base addresses" (fun () ->
              let d =
                Amdev.parse_discovery
                  (discovery_blob ~base64:true
                     [ (0xb, 0, (12, 0, 1), [ 0x100000040 ]) ])
              in
              equal
                (list (pair int (array int)))
                [ (0, [| 0x100000040 |]) ]
                (List.assoc Am.gc_hwip d.Amdev.regs_offset));
          test "rejects corrupted signatures" (fun () ->
              let b = discovery_blob sample_ips in
              s32 b 0 0xdead;
              raises_match
                (Exn.failure ~substring:"discovery signatures mismatch")
                (fun () -> Amdev.parse_discovery b);
              let b = discovery_blob sample_ips in
              s32 b 0x100 0xdead;
              raises_match
                (Exn.failure ~substring:"discovery signatures mismatch")
                (fun () -> Amdev.parse_discovery b));
        ];
      group "pte flags"
        [
          test "gfx11 words" (fun () ->
              equal int64 0x71L (flags ~gc_ver:(11, 0, 0) ());
              equal int64 0x0003000000000071L
                (flags ~gc_ver:(11, 0, 0) ~uncached:true ());
              (* A leaf above the leaf level is a huge page. *)
              equal int64 0x400000000004F1L
                (flags ~gc_ver:(11, 0, 0) ~lv:Am.amdgpu_vm_pdb0 ~frag:9 ());
              equal int64 0x1L (flags ~gc_ver:(11, 0, 0) ~table:true ()));
          test "gfx12 words keep bit 63" (fun () ->
              let leaf = flags ~gc_ver:(12, 0, 0) () in
              equal int64 0x8000000000000071L leaf;
              equal bool true (Int64.logand leaf Am.amdgpu_pte_is_pte <> 0L);
              equal int64 0x80C0000000000071L
                (flags ~gc_ver:(12, 0, 0) ~lv:Am.amdgpu_vm_pdb0 ~uncached:true
                   ());
              equal int64 0x1L (flags ~gc_ver:(12, 0, 0) ~table:true ()));
          test "gfx9 words" (fun () ->
              equal int64 0x4800000000000001L
                (flags ~gc_ver:(9, 4, 3) ~lv:Am.amdgpu_vm_pdb1 ~table:true ());
              equal int64 0x0100000000000001L
                (flags ~gc_ver:(9, 4, 3) ~lv:Am.amdgpu_vm_pdb0 ~table:true ());
              equal int64 0x0040000000000071L
                (flags ~gc_ver:(9, 4, 3) ~lv:Am.amdgpu_vm_pdb1 ());
              equal int64 0x77L
                (flags ~gc_ver:(9, 4, 3) ~system:true ~snooped:true ()));
          test "huge-page detection per generation" (fun () ->
              let huge = Amdev.Am_page_table.is_pte_huge_page in
              equal bool true
                (huge ~gc_ver:(12, 0, 0) ~lv:Am.amdgpu_vm_pdb0
                   0x8000000000000000L);
              equal bool false (huge ~gc_ver:(12, 0, 0) ~lv:Am.amdgpu_vm_pdb0 0x71L);
              equal bool true
                (huge ~gc_ver:(11, 0, 0) ~lv:Am.amdgpu_vm_pdb0
                   Am.amdgpu_pde_pte);
              (* On gfx9 a pdb0 entry is a table iff it carries the
                 translate-further flag. *)
              equal bool false
                (huge ~gc_ver:(9, 4, 3) ~lv:Am.amdgpu_vm_pdb0 Am.amdgpu_pte_tf);
              equal bool true (huge ~gc_ver:(9, 4, 3) ~lv:Am.amdgpu_vm_pdb0 0L);
              equal bool true
                (huge ~gc_ver:(9, 4, 3) ~lv:Am.amdgpu_vm_pdb1
                   Am.amdgpu_pde_pte));
        ];
      group "page tables"
        [
          test "walk writes small pages at every level" (fun () ->
              with_fake_vram 0x800000 (fun vram ->
                  let mm, ops = make_mm ~gc_ver:(11, 0, 0) vram in
                  let (_ : Memory.virt_mapping) =
                    Memory.map_range mm ~vaddr:0x3000 ~size:0x2000
                      [ (0x300000, 0x2000) ]
                      Memory.Phys ()
                  in
                  equal int64 0x100001L (Mmio.read64 vram 0x0);
                  equal int64 0x101001L (Mmio.read64 vram 0x100000);
                  equal int64 0x102001L (Mmio.read64 vram 0x101000);
                  equal int64 0x300071L (Mmio.read64 vram (0x102000 + (3 * 8)));
                  equal int64 0x301071L (Mmio.read64 vram (0x102000 + (4 * 8)));
                  let pts = Memory.page_tables mm ~vaddr:0x3000 ~size:0x1000 in
                  equal (list int) [ 0; 1; 2; 3 ]
                    (List.map ops.Memory.lv pts);
                  equal (list int)
                    [ 0x0; 0x100000; 0x101000; 0x102000 ]
                    (List.map ops.Memory.paddr pts);
                  let ptb = List.nth pts 3 in
                  equal bool true (ops.Memory.valid ptb 3);
                  equal bool false (ops.Memory.valid ptb 5);
                  equal bool true (ops.Memory.is_page ptb 3);
                  equal int 0x300000 (ops.Memory.address ptb 3);
                  equal int64 0x301071L (ops.Memory.entry ptb 4)));
          test "walk maps a gfx12 huge page and unmaps it" (fun () ->
              with_fake_vram 0x800000 (fun vram ->
                  let mm, ops = make_mm ~gc_ver:(12, 0, 0) vram in
                  let (_ : Memory.virt_mapping) =
                    Memory.map_range mm ~vaddr:0x200000 ~size:0x200000
                      [ (0x400000, 0x200000) ]
                      Memory.Phys ()
                  in
                  equal int64 0x100001L (Mmio.read64 vram 0x0);
                  equal int64 0x101001L (Mmio.read64 vram 0x100000);
                  let huge = Mmio.read64 vram (0x101000 + 8) in
                  equal int64 0x80000000004004F1L huge;
                  equal bool true
                    (Int64.logand huge Am.amdgpu_pde_pte_gfx12 <> 0L);
                  let pts =
                    Memory.page_tables mm ~vaddr:0x200000 ~size:0x200000
                  in
                  equal (list int) [ 0; 1; 2 ] (List.map ops.Memory.lv pts);
                  Memory.unmap_range mm ~vaddr:0x200000 ~size:0x200000;
                  (* A cleared entry keeps its flag bits with valid off. *)
                  equal int64 0x8000000000000070L (Mmio.read64 vram 0x0);
                  equal bool false (ops.Memory.valid (List.nth pts 0) 0)));
          test "rebases device-local addresses" (fun () ->
              with_fake_vram 0x4000 (fun vram ->
                  let ops =
                    Amdev.Am_page_table.ops ~vram ~gc_ver:(9, 4, 3)
                      ~paddr_base:(fun () -> 0x10000000)
                      ()
                  in
                  let pt = ops.Memory.make ~paddr:0x1000 ~lv:Am.amdgpu_vm_ptb in
                  ops.Memory.set_entry pt ~idx:0 ~paddr:0x2000 ~valid:true ();
                  equal int64 0x10002071L (Mmio.read64 vram 0x1000);
                  equal int 0x2000 (ops.Memory.address pt 0)));
          test "system entries skip the rebase and refuse address" (fun () ->
              with_fake_vram 0x4000 (fun vram ->
                  let ops =
                    Amdev.Am_page_table.ops ~vram ~gc_ver:(9, 4, 3)
                      ~paddr_base:(fun () -> 0x10000000)
                      ()
                  in
                  let pt = ops.Memory.make ~paddr:0x1000 ~lv:Am.amdgpu_vm_ptb in
                  ops.Memory.set_entry pt ~idx:1 ~paddr:0x3000
                    ~aspace:Memory.Sys ~uncached:true ~snooped:true ~valid:true
                    ();
                  equal int64 0x0600000000003077L
                    (Mmio.read64 vram (0x1000 + 8));
                  raises_match
                    (Exn.invalid_arg ~substring:"system address")
                    (fun () -> ops.Memory.address pt 1)));
          test "rejects addresses beyond the address-space mask" (fun () ->
              with_fake_vram 0x4000 (fun vram ->
                  let ops =
                    Amdev.Am_page_table.ops ~vram ~gc_ver:(11, 0, 0) ()
                  in
                  let pt = ops.Memory.make ~paddr:0x1000 ~lv:Am.amdgpu_vm_ptb in
                  raises_match
                    (Exn.invalid_arg ~substring:"Invalid physical address")
                    (fun () ->
                      ops.Memory.set_entry pt ~idx:0 ~paddr:(1 lsl 45)
                        ~valid:true ())));
        ];
      group "registers"
        [
          test "field writes encode and decode back" (fun () ->
              let amr, store = fake_register () in
              Amdev.Am_register.write amr [ ("lo", 0xab); ("hi", 0xcd) ];
              equal int 0xcd000ab (Hashtbl.find store 0x123);
              equal
                (list (pair string int))
                [ ("lo", 0xab); ("mid", 0); ("hi", 0xcd) ]
                (Amdev.Am_register.read_bitfields amr));
          test "update touches only the named fields" (fun () ->
              let amr, _ = fake_register () in
              Amdev.Am_register.write amr [ ("lo", 0xab); ("hi", 0xcd) ];
              Amdev.Am_register.update amr [ ("mid", 5) ];
              equal int 0xcd005ab (Amdev.Am_register.read amr));
          test "a raw value passes through unchanged" (fun () ->
              let amr, store = fake_register () in
              Amdev.Am_register.write amr ~value:0xdeadbeef [];
              equal int 0xdeadbeef (Hashtbl.find store 0x123));
          test "unknown fields raise" (fun () ->
              let amr, _ = fake_register () in
              raises_match
                (Exn.invalid_arg ~substring:"has no field")
                (fun () -> Amdev.Am_register.write amr [ ("nope", 1) ]);
              raises_match
                (Exn.invalid_arg ~substring:"has no field")
                (fun () -> Amdev.Am_register.update amr [ ("nope", 1) ]));
        ];
      group "firmware"
        [
          test "loads and splits a gfx11 firmware set" (fun () ->
              let files =
                [
                  ("psp_13_0_10_sos.bin", sos_blob ());
                  ("smu_13_0_10.bin", smu_blob_gfx11 ());
                  ("sdma_6_0_2.bin", sdma_blob_v2 ());
                  ( "gc_11_0_2_mec.bin",
                    gfx_blob_v2 ~code:"MECCODE1" ~stack:"MECSTAK1"
                      ~start_lo:0x1000 ~start_hi:2 );
                  ("gc_11_0_2_imu.bin", imu_blob ());
                  ("gc_11_0_2_rlc.bin", rlc_blob_v2_3 ());
                ]
              in
              let load, requested = fw_loader files in
              let fw =
                Firmware.create ~load
                  [
                    (Am.gc_hwip, (11, 0, 2));
                    (Am.sdma0_hwip, (6, 0, 2));
                    (Am.mp0_hwip, (13, 0, 10));
                    (Am.mp1_hwip, (13, 0, 10));
                  ]
              in
              equal (list string) (List.map fst files) (requested ());
              equal
                (list (pair int string))
                [ (2, "SYS!"); (3, "KDB-DATA") ]
                (List.map
                   (fun (t, d) -> (t, Bytes.to_string d))
                   fw.Firmware.sos_fw);
              (match fw.Firmware.smu_psp_desc with
              | Some (types, data) ->
                  equal (list int) [ Am.gfx_fw_type_smu ] types;
                  equal string "SMUCODE!" (Bytes.to_string data)
              | None -> fail "expected an smu image");
              equal
                (list (pair string int))
                [ ("MEC", 0x200001000) ]
                fw.Firmware.ucode_start;
              equal
                (list (pair (list int) string))
                [
                  ([ Am.gfx_fw_type_sdma_ucode_th1 ], "CTLCOD");
                  ([ Am.gfx_fw_type_sdma_ucode_th0 ], "CTXCODE!");
                  ([ Am.gfx_fw_type_rs64_mec ], "MECCODE1");
                  ([ Am.gfx_fw_type_rs64_mec_p0_stack ], "MECSTAK1");
                  ([ Am.gfx_fw_type_imu_i ], "IMUIRAM!");
                  ([ Am.gfx_fw_type_imu_d ], "IMUD");
                  ([ Am.gfx_fw_type_rlc_iram ], "RLCIRAM!");
                  ([ Am.gfx_fw_type_rlc_dram_boot ], "RLCD");
                  ([ Am.gfx_fw_type_rlc_p ], "RLCPCODE");
                  ([ Am.gfx_fw_type_rlc_v ], "RLCV");
                  ([ Am.gfx_fw_type_rlc_g ], "RLCGCODE");
                ]
                (desc_strings fw.Firmware.descs));
          test "gfx9: pptable scan, jump table, save-restore lists"
            (fun () ->
              let files =
                [
                  ("psp_13_0_6_sos.bin", sos_blob ());
                  ("smu_13_0_6.bin", smu_blob_gfx9 ());
                  ("sdma_4_4_2.bin", sdma_blob_v1 ());
                  ("gc_9_4_3_mec.bin", gfx_blob_v1 ());
                  ("gc_9_4_3_rlc.bin", rlc_blob_v2_1 ());
                ]
              in
              let load, requested = fw_loader files in
              let fw =
                Firmware.create ~load
                  [
                    (Am.gc_hwip, (9, 4, 3));
                    (Am.sdma0_hwip, (4, 4, 2));
                    (Am.mp0_hwip, (13, 0, 6));
                    (Am.mp1_hwip, (13, 0, 6));
                  ]
              in
              equal (list string) (List.map fst files) (requested ());
              equal bool true (fw.Firmware.smu_psp_desc = None);
              equal (list (pair string int)) [] fw.Firmware.ucode_start;
              equal
                (list (pair (list int) string))
                [
                  ([ Am.gfx_fw_type_p2s_table ], "P2STAB");
                  ( [
                      Am.gfx_fw_type_sdma0; Am.gfx_fw_type_sdma1;
                      Am.gfx_fw_type_sdma2; Am.gfx_fw_type_sdma3;
                    ],
                    "SDMA-CODE-12" );
                  ([ Am.gfx_fw_type_cp_mec ], "MEC-V1-CODE-24-BYTES-OK!");
                  ([ Am.gfx_fw_type_cp_mec_me1 ], "MECJTAB!");
                  ([ Am.gfx_fw_type_rlc_restore_list_srm_cntl ], "CNTL");
                  ([ Am.gfx_fw_type_rlc_restore_list_gpm_mem ], "GPMLIST!");
                  ([ Am.gfx_fw_type_rlc_restore_list_srm_mem ], "SRM!");
                  ([ Am.gfx_fw_type_rlc_g ], "RLCGCODE");
                ]
                (desc_strings fw.Firmware.descs));
          test "gfx12: pfp and me images, smu 13.0.12 skipped" (fun () ->
              let files =
                [
                  ("psp_14_0_3_sos.bin", sos_blob ());
                  ("sdma_7_0_0.bin", sdma_blob_v3 ());
                  ( "gc_12_0_1_pfp.bin",
                    gfx_blob_v2 ~code:"PFPCODE1" ~stack:"PFPSTAK1"
                      ~start_lo:0x100 ~start_hi:0 );
                  ( "gc_12_0_1_me.bin",
                    gfx_blob_v2 ~code:"ME-CODE1" ~stack:"ME-STAK1"
                      ~start_lo:0x200 ~start_hi:0 );
                  ( "gc_12_0_1_mec.bin",
                    gfx_blob_v2 ~code:"MECCODE1" ~stack:"MECSTAK1"
                      ~start_lo:0x1000 ~start_hi:2 );
                  ("gc_12_0_1_imu.bin", imu_blob ());
                  ("gc_12_0_1_rlc.bin", rlc_blob_v2_3 ());
                ]
              in
              let load, requested = fw_loader files in
              let fw =
                Firmware.create ~load
                  [
                    (Am.gc_hwip, (12, 0, 1));
                    (Am.sdma0_hwip, (7, 0, 0));
                    (Am.mp0_hwip, (14, 0, 3));
                    (Am.mp1_hwip, (13, 0, 12));
                  ]
              in
              equal (list string) (List.map fst files) (requested ());
              equal bool true (fw.Firmware.smu_psp_desc = None);
              equal
                (list (pair string int))
                [ ("PFP", 0x100); ("ME", 0x200); ("MEC", 0x200001000) ]
                fw.Firmware.ucode_start;
              equal
                (list (list int))
                [
                  [ Am.gfx_fw_type_sdma_ucode_th0 ];
                  [ Am.gfx_fw_type_rs64_pfp ];
                  [ Am.gfx_fw_type_rs64_pfp_p0_stack ];
                  [ Am.gfx_fw_type_rs64_me ];
                  [ Am.gfx_fw_type_rs64_me_p0_stack ];
                  [ Am.gfx_fw_type_rs64_mec ];
                  [ Am.gfx_fw_type_rs64_mec_p0_stack ];
                  [ Am.gfx_fw_type_imu_i ];
                  [ Am.gfx_fw_type_imu_d ];
                  [ Am.gfx_fw_type_rlc_iram ];
                  [ Am.gfx_fw_type_rlc_dram_boot ];
                  [ Am.gfx_fw_type_rlc_p ];
                  [ Am.gfx_fw_type_rlc_v ];
                  [ Am.gfx_fw_type_rlc_g ];
                ]
                (List.map fst fw.Firmware.descs));
          test "rejects unknown image header versions" (fun () ->
              let b = Bytes.make 0x40 '\x00' in
              common_header b ~ver:(9, 9) ~ucode_off:0 ~ucode_size:0;
              raises_match
                (Exn.failure ~substring:"unhandled psp firmware header v9_9")
                (fun () ->
                  Firmware.create
                    ~load:(fun _ -> b)
                    [
                      (Am.gc_hwip, (11, 0, 2));
                      (Am.sdma0_hwip, (6, 0, 2));
                      (Am.mp0_hwip, (13, 0, 10));
                      (Am.mp1_hwip, (13, 0, 10));
                    ]));
          test "fetches files whose digests match" (fun () ->
              with_fw_dir (fun dir ->
                  write_file (Filename.concat dir "fw.bin") "abc";
                  equal string "abc"
                    (Bytes.to_string
                       (Firmware.fetch_fw ~dir "fw.bin" ~sha256:sha_abc));
                  (* a message padding into a second sha256 block *)
                  let msg =
                    "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
                  in
                  write_file (Filename.concat dir "fw2.bin") msg;
                  equal string msg
                    (Bytes.to_string
                       (Firmware.fetch_fw ~dir "fw2.bin"
                          ~sha256:
                            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"))));
          test "corrupted files fail naming both digests" (fun () ->
              with_fw_dir (fun dir ->
                  write_file (Filename.concat dir "smu_13_0_0.bin") "garbage";
                  let pinned = List.assoc "smu_13_0_0.bin" Fw_defs.hashes in
                  raises_match
                    (Exn.failure
                       ~substring:("fetch sha mismatch, expected " ^ pinned))
                    (fun () -> Firmware.load_fw ~dir "smu_13_0_0.bin");
                  (* the sha256 of "garbage" *)
                  raises_match
                    (Exn.failure
                       ~substring:
                         "795b6904e54f82411df4b0e27a373a55eea3f9d66dac5a9bce1dd92f7b401da5")
                    (fun () -> Firmware.load_fw ~dir "smu_13_0_0.bin")));
          test "missing files name the searched paths" (fun () ->
              with_fw_dir (fun dir ->
                  raises_match
                    (Exn.failure
                       ~substring:(Filename.concat dir "psp_13_0_0_sos.bin"))
                    (fun () -> Firmware.load_fw ~dir "psp_13_0_0_sos.bin");
                  raises_match
                    (Exn.failure
                       ~substring:
                         (Filename.concat dir "psp_13_0_0_sos.bin.zst"))
                    (fun () -> Firmware.load_fw ~dir "psp_13_0_0_sos.bin");
                  raises_match
                    (Exn.failure
                       ~substring:"gitlab.com/kernel-firmware/linux-firmware")
                    (fun () -> Firmware.load_fw ~dir "psp_13_0_0_sos.bin");
                  raises_match
                    (Exn.failure ~substring:"has no pinned sha256")
                    (fun () -> Firmware.load_fw ~dir "unknown_fw.bin")));
          test "decompresses the zst variant" (fun () ->
              if Sys.command "command -v zstd >/dev/null 2>&1" <> 0 then
                skip ~reason:"zstd not on PATH" ()
              else
                with_fw_dir (fun dir ->
                    let raw = Filename.concat dir "fw.raw" in
                    write_file raw "abc";
                    equal int 0
                      (Sys.command
                         (Printf.sprintf "zstd -q %s -o %s"
                            (Filename.quote raw)
                            (Filename.quote
                               (Filename.concat dir "fw.bin.zst"))));
                    equal string "abc"
                      (Bytes.to_string
                         (Firmware.fetch_fw ~dir "fw.bin" ~sha256:sha_abc))));
        ];
      group "address topology"
        [
          test "reads the framebuffer and fabric position at creation"
            (fun () ->
              with_fake_dev (fun fd ->
                  equal bool false (Amdev.is_hive fd.dev);
                  equal int 0x10000123 (Amdev.paddr2mc fd.dev 0x123);
                  equal int 0x123 (Amdev.paddr2xgmi fd.dev 0x123));
              with_fake_dev ~mmhub:(1, 8, 0)
                ~pre:(fun store ->
                  (* pf_lfb_region = 2, pf_lfb_size = 0x40 *)
                  Hashtbl.replace store (mmhub_base + 0xc97) 2;
                  Hashtbl.replace store (mmhub_base + 0xc98) 0x40)
                (fun fd ->
                  equal bool true (Amdev.is_hive fd.dev);
                  equal int 0x80000123 (Amdev.paddr2xgmi fd.dev 0x123);
                  equal int 0x123 (Amdev.xgmi2paddr fd.dev 0x80000123);
                  equal int 0x90000123 (Amdev.paddr2mc fd.dev 0x123)));
        ];
      group "psp protocol"
        [
          test "gfx11 boot: load sequence, command frames, staging"
            (fun () ->
              with_fake_dev (fun fd ->
                  let fw =
                    {
                      Firmware.sos_fw =
                        [
                          (Am.psp_fw_type_psp_kdb, Bytes.of_string "KDBIMAGE");
                          (Am.psp_fw_type_psp_sos, Bytes.of_string "SOSIMAGE");
                          (Am.psp_fw_type_psp_toc, Bytes.of_string "TOCIMAGE");
                        ];
                      ucode_start = [];
                      smu_psp_desc =
                        Some ([ Am.gfx_fw_type_smu ], Bytes.of_string "SMUIMAGE");
                      descs =
                        [ ([ Am.gfx_fw_type_rlc_g ], Bytes.of_string "RLCIMAGE") ];
                    }
                  in
                  let psp = Psp.create fd.dev ~fw in
                  let s = psp_script fd ~pref:"regMP0_SMN_C2PMSG" psp in
                  Psp.init_hw psp;
                  let mc = Amdev.paddr2mc fd.dev in
                  let msg1_addr = mc (Psp.msg1_paddr psp) in
                  let ring_mc = mc (Psp.ring_paddr psp) in
                  let tmr = Psp.tmr_paddr psp in
                  equal
                    (list (pair int int))
                    [
                      (flush_target, 0);
                      (s.r 36, msg1_addr lsr 20);
                      (s.r 35, Am.psp_bl__load_key_database);
                      (flush_target, 0);
                      (s.r 36, msg1_addr lsr 20);
                      (s.r 35, Am.psp_bl__load_tos_spl_table);
                      (flush_target, 0);
                      (s.r 36, msg1_addr lsr 20);
                      (s.r 35, Am.psp_bl__load_sosdrv);
                      (s.r 69, lo32 ring_mc);
                      (s.r 70, hi32 ring_mc);
                      (s.r 71, 0x10000);
                      (s.r 64, Am.psp_ring_type__km lsl 16);
                      (flush_target, 0);
                      (s.r 67, 0x10);
                      (flush_target, 0);
                      (s.r 67, 0x20);
                      (s.r 67, 0x30);
                      (flush_target, 0);
                      (s.r 67, 0x40);
                      (s.r 67, 0x50);
                    ]
                    (List.rev !(fd.log));
                  equal int 5 (List.length !(s.frames));
                  List.iter2 equal_bytes
                    [
                      psp_cmd Am.gfx_cmd_id_load_toc
                        [
                          (0x1c, lo32 msg1_addr); (0x20, hi32 msg1_addr);
                          (0x24, 8);
                        ];
                      psp_cmd Am.gfx_cmd_id_load_ip_fw
                        [
                          (0x1c, lo32 msg1_addr); (0x20, hi32 msg1_addr);
                          (0x24, 8); (0x28, Am.gfx_fw_type_smu);
                        ];
                      psp_cmd Am.gfx_cmd_id_setup_tmr
                        [
                          (0x1c, lo32 (mc tmr)); (0x20, hi32 (mc tmr));
                          (0x24, 0x120000); (0x28, 2); (0x2c, lo32 tmr);
                          (0x30, hi32 tmr);
                        ];
                      psp_cmd Am.gfx_cmd_id_load_ip_fw
                        [
                          (0x1c, lo32 msg1_addr); (0x20, hi32 msg1_addr);
                          (0x24, 8); (0x28, Am.gfx_fw_type_rlc_g);
                        ];
                      psp_cmd Am.gfx_cmd_id_autoload_rlc [];
                    ]
                    (List.rev !(s.frames));
                  List.iteri
                    (fun i expected ->
                      equal_bytes expected
                        (Mmio.read_bytes fd.fvram
                           ~off:(Psp.ring_paddr psp + (i * 0x40))
                           ~len:0x40))
                    (List.init 5 (fun i ->
                         rb_frame
                           ~cmd_mc:(mc (Psp.cmd_paddr psp))
                           ~fence_mc:(mc (Psp.fence_paddr psp))
                           ~fence_value:((i * 0x10) + 1)));
                  equal (list string)
                    [ staged "KDBIMAGE"; staged "KDBIMAGE"; staged "SOSIMAGE" ]
                    (List.rev !(s.msg1s));
                  (* the staging buffer last held the descriptor image *)
                  equal string (staged "RLCIMAGE")
                    (Bytes.to_string
                       (Mmio.read_bytes fd.fvram ~off:(Psp.msg1_paddr psp)
                          ~len:16))));
          test "mp0 14.x: MPASP mailbox, SPL key, boot-time TMR" (fun () ->
              with_fake_dev ~mp0:(14, 0, 3) (fun fd ->
                  let fw =
                    {
                      no_fw with
                      Firmware.sos_fw =
                        [
                          (Am.psp_fw_type_psp_spl, Bytes.of_string "SPLIMAGE");
                          (Am.psp_fw_type_psp_sos, Bytes.of_string "SOSIMAGE");
                        ];
                    }
                  in
                  let psp = Psp.create fd.dev ~fw in
                  let s = psp_script fd ~pref:"regMPASP_SMN_C2PMSG" psp in
                  Psp.init_hw psp;
                  equal int 0 (Psp.tmr_paddr psp);
                  let msg1_addr = Amdev.paddr2mc fd.dev (Psp.msg1_paddr psp) in
                  let ring_mc = Amdev.paddr2mc fd.dev (Psp.ring_paddr psp) in
                  equal
                    (list (pair int int))
                    [
                      (flush_target, 0);
                      (s.r 36, msg1_addr lsr 20);
                      (s.r 35, Am.psp_bl__load_tos_spl_table);
                      (flush_target, 0);
                      (s.r 36, msg1_addr lsr 20);
                      (s.r 35, Am.psp_bl__load_sosdrv);
                      (s.r 69, lo32 ring_mc);
                      (s.r 70, hi32 ring_mc);
                      (s.r 71, 0x10000);
                      (s.r 64, Am.psp_ring_type__km lsl 16);
                      (s.r 67, 0x10);
                    ]
                    (List.rev !(fd.log));
                  equal int 1 (List.length !(s.frames));
                  equal_bytes
                    (psp_cmd Am.gfx_cmd_id_autoload_rlc [])
                    (List.hd !(s.frames))));
          test "a live sOS skips the bootloader and recreates the ring"
            (fun () ->
              with_fake_dev (fun fd ->
                  let psp = Psp.create fd.dev ~fw:no_fw in
                  let s = psp_script fd ~pref:"regMP0_SMN_C2PMSG" psp in
                  Hashtbl.replace fd.reads (s.r 81) (fun () -> 1);
                  Hashtbl.replace fd.store (s.r 71) 1;
                  Psp.init_hw psp;
                  let mc = Amdev.paddr2mc fd.dev in
                  let ring_mc = mc (Psp.ring_paddr psp) in
                  let tmr = Psp.tmr_paddr psp in
                  equal
                    (list (pair int int))
                    [
                      (s.r 64, Am.gfx_ctrl_cmd_id_destroy_rings);
                      (s.r 69, lo32 ring_mc);
                      (s.r 70, hi32 ring_mc);
                      (s.r 71, 0x10000);
                      (s.r 64, Am.psp_ring_type__km lsl 16);
                      (s.r 67, 0x10);
                      (s.r 67, 0x20);
                    ]
                    (List.rev !(fd.log));
                  equal int 2 (List.length !(s.frames));
                  (* no TOC was loaded, so the TMR sets up with size 0 *)
                  List.iter2 equal_bytes
                    [
                      psp_cmd Am.gfx_cmd_id_setup_tmr
                        [
                          (0x1c, lo32 (mc tmr)); (0x20, hi32 (mc tmr));
                          (0x28, 2); (0x2c, lo32 tmr); (0x30, hi32 tmr);
                        ];
                      psp_cmd Am.gfx_cmd_id_autoload_rlc [];
                    ]
                    (List.rev !(s.frames))));
          test "an unanswered bootloader mailbox raises" (fun () ->
              with_fake_dev (fun fd ->
                  let fw =
                    {
                      no_fw with
                      Firmware.sos_fw =
                        [ (Am.psp_fw_type_psp_kdb, Bytes.of_string "KDBIMAGE") ];
                    }
                  in
                  let psp = Psp.create fd.dev ~fw in
                  raises_match
                    (function
                      | Am_ip.Timeout_error m ->
                          has_substring m "BL not ready"
                          && has_substring m "10000 ms"
                      | _ -> false)
                    (fun () -> Psp.init_hw psp)));
          test "a rejected command names its id and status" (fun () ->
              with_fake_dev (fun fd ->
                  let psp = Psp.create fd.dev ~fw:no_fw in
                  let s = psp_script fd ~pref:"regMP0_SMN_C2PMSG" psp in
                  Hashtbl.replace fd.reads (s.r 81) (fun () -> 1);
                  Hashtbl.replace fd.wr_hooks (s.r 67) (fun v ->
                      Mmio.write32 fd.fvram (Psp.cmd_paddr psp + 0x360) 3l;
                      Mmio.write32 fd.fvram (Psp.fence_paddr psp)
                        (Int32.of_int (v - 0x10 + 1)));
                  raises_match
                    (Exn.failure ~substring:"PSP command failed 5 3")
                    (fun () -> Psp.init_hw psp)));
        ];
      group "smu protocol"
        [
          test "init_hw speaks the discovered mp1 version's interface"
            (fun () ->
              let run_init mp1 =
                with_fake_dev ~mp1 (fun fd ->
                    let smu = Smu.create fd.dev in
                    ack_messages fd;
                    Smu.init_hw smu;
                    ( smu_addrs fd,
                      Amdev.paddr2mc fd.dev (Smu.driver_table_paddr smu),
                      List.rev !(fd.log) ))
              in
              let (resp, arg, msg), dt, log = run_init (13, 0, 0) in
              equal
                (list (pair int int))
                [
                  (resp, 0); (arg, hi32 dt); (msg, 0xe);
                  (resp, 0); (arg, lo32 dt); (msg, 0xf);
                  (resp, 0); (arg, 0); (msg, 6);
                ]
                log;
              (* 13.0.10 resolves the 13.0.6 interface: other ids *)
              let (resp, arg, msg), dt, log = run_init (13, 0, 10) in
              equal
                (list (pair int int))
                [
                  (resp, 0); (arg, hi32 dt); (msg, 0xd);
                  (resp, 0); (arg, lo32 dt); (msg, 0xe);
                  (resp, 0); (arg, 0); (msg, 5);
                ]
                log);
          test "mode1_reset picks the generation's message" (fun () ->
              (* mp0 13.0.10 resets through the debug mailbox *)
              with_fake_dev (fun fd ->
                  let smu = Smu.create fd.dev in
                  let r54 = raddr fd.dev "mmMP1_SMN_C2PMSG_54" in
                  let r53 = raddr fd.dev "mmMP1_SMN_C2PMSG_53" in
                  let r75 = raddr fd.dev "mmMP1_SMN_C2PMSG_75" in
                  Hashtbl.replace fd.wr_hooks r75 (fun _ ->
                      Hashtbl.replace fd.store r54 1);
                  Smu.mode1_reset smu;
                  equal
                    (list (pair int int))
                    [ (r54, 0); (r53, 0); (r75, 2) ]
                    (List.rev !(fd.log)));
              (* mp0 13.0.6 resets through the driver-reset message *)
              with_fake_dev ~mp0:(13, 0, 6) ~mp1:(13, 0, 6) (fun fd ->
                  let smu = Smu.create fd.dev in
                  ack_messages fd;
                  Smu.mode1_reset smu;
                  let resp, arg, msg = smu_addrs fd in
                  equal
                    (list (pair int int))
                    [ (resp, 0); (arg, 1); (msg, 3) ]
                    (List.rev !(fd.log)));
              (* other generations use the plain mode-1 message *)
              with_fake_dev ~mp0:(13, 0, 2) ~mp1:(13, 0, 0) (fun fd ->
                  let smu = Smu.create fd.dev in
                  ack_messages fd;
                  Smu.mode1_reset smu;
                  let resp, arg, msg = smu_addrs fd in
                  equal
                    (list (pair int int))
                    [ (resp, 0); (arg, 0); (msg, 0x2f) ]
                    (List.rev !(fd.log))));
          test "is_smu_alive polls the response register" (fun () ->
              with_fake_dev ~mp1:(13, 0, 0) (fun fd ->
                  let smu = Smu.create fd.dev in
                  equal bool false (Smu.is_smu_alive smu);
                  ack_messages fd;
                  equal bool true (Smu.is_smu_alive smu)));
          test "an unanswered message raises" (fun () ->
              with_fake_dev ~mp1:(13, 0, 0) (fun fd ->
                  let smu = Smu.create fd.dev in
                  raises_match
                    (function
                      | Am_ip.Timeout_error m ->
                          has_substring m "SMU msg 0xe timeout"
                      | _ -> false)
                    (fun () -> Smu.init_hw smu)));
          test "set_clocks queries levels once and pins by index" (fun () ->
              with_fake_dev ~mp1:(13, 0, 0) (fun fd ->
                  let smu = Smu.create fd.dev in
                  ack_messages fd;
                  let resp, arg, msg = smu_addrs fd in
                  let queue =
                    ref [ 2; 300; 800; 2; 300; 800; 2; 300; 800; 2; 300; 800 ]
                  in
                  Hashtbl.replace fd.reads arg (fun () ->
                      match !queue with
                      | v :: rest ->
                          queue := rest;
                          v
                      | [] -> fail "unexpected argument readback");
                  Smu.set_clocks smu ~level:(Some (-1));
                  let send m p = [ (resp, 0); (arg, p); (msg, m) ] in
                  let queries clck =
                    send 0x1f ((clck lsl 16) lor 0xff)
                    @ send 0x1f (clck lsl 16)
                    @ send 0x1f ((clck lsl 16) lor 1)
                  in
                  let sets clck v =
                    send 0x19 ((clck lsl 16) lor v)
                    @ send 0x1a ((clck lsl 16) lor v)
                  in
                  (* uclk, fclk, socclk, gfxclk of the 13.0.0 interface *)
                  let clks = [ 2; 3; 1; 0 ] in
                  equal
                    (list (pair int int))
                    (List.concat_map queries clks
                    @ List.concat_map (fun c -> sets c 800) clks)
                    (List.rev !(fd.log));
                  fd.log := [];
                  (* the second call hits the cache: no further queries *)
                  Smu.set_clocks smu ~level:(Some 0);
                  equal
                    (list (pair int int))
                    (List.concat_map (fun c -> sets c 300) clks)
                    (List.rev !(fd.log))));
        ];
      group "soc"
        [
          test "init_hw opens the doorbell aperture" (fun () ->
              with_fake_dev (fun fd ->
                  let soc = Soc.create fd.dev in
                  Soc.init_hw soc ~vmhubs:1;
                  equal
                    (list (pair int int))
                    [
                      (raddr fd.dev "regRCC_DEV0_EPF2_STRAP2", 0);
                      (raddr fd.dev "regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN", 1);
                    ]
                    (List.rev !(fd.log));
                  Soc.set_clockgating_state soc;
                  equal int
                    (rencode fd "regHDP_MEM_POWER_CTRL"
                       [
                         ("atomic_mem_power_ctrl_en", 1);
                         ("atomic_mem_power_ds_en", 1);
                       ])
                    (rstore fd "regHDP_MEM_POWER_CTRL")));
          test "doorbell_enable encodes the numbered port fields" (fun () ->
              with_fake_dev (fun fd ->
                  let soc = Soc.create fd.dev in
                  Soc.doorbell_enable soc ~port:2 ~awid:0xe
                    ~awaddr_31_28_value:0x3 ~offset:0x200 ~size:4 ();
                  equal int
                    (rencode fd "regS2A_DOORBELL_ENTRY_2_CTRL"
                       [
                         ("s2a_doorbell_port2_enable", 1);
                         ("s2a_doorbell_port2_awid", 0xe);
                         ("s2a_doorbell_port2_range_size", 4);
                         ("s2a_doorbell_port2_awaddr_31_28_value", 0x3);
                         ("s2a_doorbell_port2_range_offset", 0x200);
                       ])
                    (rstore fd "regS2A_DOORBELL_ENTRY_2_CTRL")));
          test "names interrupt clients and sources per generation" (fun () ->
              with_fake_dev (fun fd ->
                  let soc = Soc.create fd.dev in
                  (match Soc.ih_client_name soc 0x14 with
                  | Some n -> equal string "SOC21_IH_CLIENTID_GRBM_CP" n
                  | None -> fail "expected a client name");
                  equal string "CP_EOP_INTERRUPT"
                    (Soc.ih_src_name soc ~client:0x14 ~src:0xb5);
                  equal string "SDMA_TRAP"
                    (Soc.ih_src_name soc ~client:0xa ~src:0x31);
                  equal string "" (Soc.ih_src_name soc ~client:0x3 ~src:0xb5));
              with_fake_dev ~gc:(9, 4, 3) ~mmhub:(1, 8, 0) ~sdma:(4, 4, 2)
                ~mp0:(13, 0, 6) ~mp1:(13, 0, 6) (fun fd ->
                  let soc = Soc.create fd.dev in
                  (match Soc.ih_client_name soc 8 with
                  | Some n -> equal string "SOC15_IH_CLIENTID_SDMA0" n
                  | None -> fail "expected a client name");
                  equal string "SDMA_TRAP"
                    (Soc.ih_src_name soc ~client:8 ~src:0xe0);
                  equal string "SQ_INTERRUPT_ID"
                    (Soc.ih_src_name soc ~client:0xa ~src:0xef)));
        ];
      group "gmc"
        [
          test "init_hw programs the memory hub" (fun () ->
              with_fake_dev (fun fd ->
                  Hashtbl.replace fd.store
                    (raddr fd.dev "regMMMC_VM_FB_LOCATION_TOP")
                    0x1f;
                  let soc = Soc.create fd.dev in
                  let gmc = Gmc.create fd.dev in
                  equal int 1 (Gmc.vmhubs gmc);
                  Gmc.init_hw gmc ~soc;
                  equal int 0xffffff (rstore fd "regMMMC_VM_AGP_BOT");
                  (* the system aperture covers the framebuffer window *)
                  equal int 0x400
                    (rstore fd "regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR");
                  equal int 0x7c0
                    (rstore fd "regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR");
                  let scratch =
                    rstore fd "regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB"
                    lsl 12
                  in
                  equal bool true
                    (scratch > 0 && scratch < Mmio.size fd.fvram);
                  (* context 0 translates through the root page table *)
                  let root =
                    Amdev.Am_page_table.paddr
                      (Memory.root_page_table (Amdev.mm fd.dev))
                  in
                  equal int
                    (lo32 (root lor 1))
                    (rstore fd "regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32");
                  equal int (hi32 root)
                    (rstore fd "regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32");
                  equal int 0
                    (rstore fd "regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32");
                  equal int 0xffffffff
                    (rstore fd "regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32");
                  equal int 0x7
                    (rstore fd "regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32");
                  let cntl = rstore fd "regMMVM_CONTEXT0_CNTL" in
                  equal int 0x1800000 (cntl land 0x1800000);
                  let bf = rdecode fd "regMMVM_CONTEXT0_CNTL" cntl in
                  equal int 1 (List.assoc "enable_context" bf);
                  equal int 3 (List.assoc "page_table_depth" bf);
                  equal int 1
                    (List.assoc "range_protection_fault_enable_default" bf);
                  let cntl3 =
                    rdecode fd "regMMVM_L2_CNTL3" (rstore fd "regMMVM_L2_CNTL3")
                  in
                  equal int 9 (List.assoc "bank_select" cntl3);
                  equal int 6 (List.assoc "l2_cache_bigk_fragment_size" cntl3);
                  equal int
                    (rencode fd "regMMVM_L2_CNTL5"
                       [ ("walker_priority_client_id", 0x1ff) ])
                    (rstore fd "regMMVM_L2_CNTL5");
                  equal int 0xffffffff
                    (rstore fd "regMMVM_INVALIDATE_ENG17_ADDR_RANGE_LO32");
                  equal int 0x1f
                    (rstore fd "regMMVM_INVALIDATE_ENG17_ADDR_RANGE_HI32")));
          test "pf_status_reg names the generation's register" (fun () ->
              with_fake_dev (fun fd ->
                  let gmc = Gmc.create fd.dev in
                  equal string "regGCVM_L2_PROTECTION_FAULT_STATUS"
                    (Gmc.pf_status_reg gmc Gmc.Gc);
                  equal string "regMMVM_L2_PROTECTION_FAULT_STATUS"
                    (Gmc.pf_status_reg gmc Gmc.Mm));
              with_fake_dev ~gc:(12, 0, 1) ~bif:(6, 3, 1) (fun fd ->
                  let gmc = Gmc.create fd.dev in
                  equal string "regGCVM_L2_PROTECTION_FAULT_STATUS_LO32"
                    (Gmc.pf_status_reg gmc Gmc.Gc)));
          test "flush_tlb invalidates the hub and clears the semaphore"
            (fun () ->
              with_fake_dev (fun fd ->
                  let gmc = Gmc.create fd.dev in
                  Hashtbl.replace fd.reads
                    (raddr fd.dev "regMMVM_INVALIDATE_ENG17_SEM") (fun () -> 1);
                  Hashtbl.replace fd.reads
                    (raddr fd.dev "regMMVM_INVALIDATE_ENG17_ACK") (fun () -> 1);
                  fd.log := [];
                  Gmc.flush_tlb gmc ~xccs:1 Gmc.Mm ~vmid:0;
                  equal
                    (list (pair int int))
                    [
                      (flush_target, 0);
                      ( raddr fd.dev "regMMVM_INVALIDATE_ENG17_REQ",
                        rencode fd "regMMVM_INVALIDATE_ENG17_REQ"
                          [
                            ("per_vmid_invalidate_req", 1);
                            ("invalidate_l2_ptes", 1);
                            ("invalidate_l2_pde0", 1);
                            ("invalidate_l2_pde1", 1);
                            ("invalidate_l2_pde2", 1);
                            ("invalidate_l1_ptes", 1);
                          ] );
                      (raddr fd.dev "regMMVM_INVALIDATE_ENG17_SEM", 0);
                      ( raddr fd.dev "regMMVM_L2_BANK_SELECT_RESERVED_CID2",
                        rencode fd "regMMVM_L2_BANK_SELECT_RESERVED_CID2"
                          [ ("reserved_cache_private_invalidation", 1) ] );
                    ]
                    (List.rev !(fd.log))));
          test "flush_tlb skips a hub that is not programmed yet" (fun () ->
              with_fake_dev (fun fd ->
                  let gmc = Gmc.create fd.dev in
                  fd.log := [];
                  Gmc.flush_tlb gmc ~xccs:1 Gmc.Gc ~vmid:0;
                  equal
                    (list (pair int int))
                    [ (flush_target, 0) ]
                    (List.rev !(fd.log))));
          test "an installed mapping hook runs after map_range" (fun () ->
              with_fake_dev (fun fd ->
                  let calls = ref 0 in
                  Amdev.set_on_range_mapped fd.dev (fun () -> incr calls);
                  let mm = Amdev.mm fd.dev in
                  let paddr = Memory.palloc mm 0x1000 ~zero:false ~boot:true () in
                  let vaddr = Memory.alloc_vaddr mm 0x1000 () in
                  let (_ : Memory.virt_mapping) =
                    Memory.map_range mm ~vaddr ~size:0x1000
                      [ (paddr, 0x1000) ]
                      Memory.Phys ~boot:true ()
                  in
                  equal int 1 !calls));
        ];
      group "gfx engines"
        [
          test "init_hw boots the compute engines" (fun () ->
              with_fake_dev (fun fd ->
                  let soc = Soc.create fd.dev in
                  let gmc = Gmc.create fd.dev in
                  let fw =
                    { no_fw with Firmware.ucode_start = [ ("MEC", 0x2000) ] }
                  in
                  let psp = Psp.create fd.dev ~fw in
                  let gfx = Gfx.create fd.dev in
                  equal int 1 (Gfx.xccs gfx);
                  fd.log := [];
                  Gfx.init_hw gfx ~soc ~gmc ~psp ~fw ~partial_boot:false;
                  (* the graphics hub was programmed *)
                  equal int
                    (rencode fd "regGCVM_L2_CNTL5"
                       [ ("walker_priority_client_id", 0x1ff) ])
                    (rstore fd "regGCVM_L2_CNTL5");
                  equal int 1
                    (List.assoc "enable_context"
                       (rdecode fd "regGCVM_CONTEXT0_CNTL"
                          (rstore fd "regGCVM_CONTEXT0_CNTL")));
                  (* processors pointed at their instruction start *)
                  equal int (0x2000 lsr 2)
                    (rstore fd "regCP_MEC_RS64_PRGRM_CNTR_START");
                  equal int 0 (rstore fd "regCP_MEC_RS64_PRGRM_CNTR_START_HI");
                  equal int 0x20000000 (rstore fd "regTCP_CNTL");
                  equal int 0x1 (rstore fd "regRLC_CNTL");
                  equal int 0xf (rstore fd "regRLC_SPM_MC_CNTL");
                  equal int
                    (rencode fd "regRLC_SRM_CNTL"
                       [ ("srm_enable", 1); ("auto_incr_addr", 1) ])
                    (rstore fd "regRLC_SRM_CNTL");
                  (* doorbell routes for the engine ports *)
                  equal bool true
                    (Hashtbl.mem fd.store
                       (raddr fd.dev "regS2A_DOORBELL_ENTRY_0_CTRL"));
                  equal bool true
                    (Hashtbl.mem fd.store
                       (raddr fd.dev "regS2A_DOORBELL_ENTRY_3_CTRL"));
                  (* shader memory configured for all 16 vm contexts *)
                  equal int 16
                    (List.length
                       (List.filter
                          (fun (a, _) -> a = raddr fd.dev "regSH_MEM_CONFIG")
                          !(fd.log)));
                  equal int
                    (rencode fd "regSH_MEM_CONFIG"
                       [
                         ("initial_inst_prefetch", 3); ("address_mode", 0);
                         ("alignment_mode", 3);
                       ])
                    (rstore fd "regSH_MEM_CONFIG");
                  equal int
                    (rencode fd "regSH_MEM_BASES"
                       [ ("shared_base", 1); ("private_base", 2) ])
                    (rstore fd "regSH_MEM_BASES");
                  equal int 0 (rstore fd "regCP_MEC_DOORBELL_RANGE_LOWER");
                  equal int 0xf8 (rstore fd "regCP_MEC_DOORBELL_RANGE_UPPER");
                  (* processors released *)
                  equal int 1
                    (List.assoc "mec_pipe0_active"
                       (rdecode fd "regCP_MEC_RS64_CNTL"
                          (rstore fd "regCP_MEC_RS64_CNTL")));
                  equal int 0 (rstore fd "regGRBM_GFX_CNTL")));
          test "a partial boot resets the processors instead" (fun () ->
              with_fake_dev (fun fd ->
                  let soc = Soc.create fd.dev in
                  let gmc = Gmc.create fd.dev in
                  let fw =
                    { no_fw with Firmware.ucode_start = [ ("MEC", 0x2000) ] }
                  in
                  let psp = Psp.create fd.dev ~fw in
                  let gfx = Gfx.create fd.dev in
                  fd.log := [];
                  Gfx.init_hw gfx ~soc ~gmc ~psp ~fw ~partial_boot:true;
                  equal
                    (list (pair int int))
                    [
                      ( raddr fd.dev "regGRBM_SOFT_RESET",
                        rencode fd "regGRBM_SOFT_RESET"
                          [ ("soft_reset_cp", 1); ("soft_reset_cpc", 1) ] );
                      (raddr fd.dev "regGRBM_SOFT_RESET", 0);
                    ]
                    (List.filter
                       (fun (a, _) -> a = raddr fd.dev "regGRBM_SOFT_RESET")
                       (List.rev !(fd.log)));
                  equal int (0x2000 lsr 2)
                    (rstore fd "regCP_MEC_RS64_PRGRM_CNTR_START");
                  (* the full bring-up did not run *)
                  equal bool false
                    (Hashtbl.mem fd.store (raddr fd.dev "regRLC_SPM_MC_CNTL"))));
          test "setup_ring builds the queue descriptor and mirrors it"
            (fun () ->
              with_fake_dev (fun fd ->
                  let gfx = Gfx.create fd.dev in
                  fd.log := [];
                  let doorbell =
                    Gfx.setup_ring gfx ~ring_addr:0x100000 ~ring_size:0x800
                      ~rptr_addr:0x11000 ~wptr_addr:0x12000 ~eop_addr:0x13000
                      ~eop_size:0x1000 ~idx:0 ~aql:false
                  in
                  equal int 3 doorbell;
                  (* the descriptor's device address reaches the bring-up
                     registers; recover it to check the content *)
                  let base = raddr fd.dev "regCP_MQD_BASE_ADDR" in
                  let mqd_mc =
                    match List.assoc_opt base (List.rev !(fd.log)) with
                    | Some v -> v
                    | None -> fail "expected a descriptor base write"
                  in
                  let expected = Bytes.make 0x800 '\x00' in
                  s32 expected 0x0 0xC0310800;
                  s32 expected 0x200 mqd_mc;
                  s32 expected 0x210
                    (rencode fd "regCP_HQD_PERSISTENT_STATE"
                       [ ("preload_size", 0x55); ("preload_req", 1) ]);
                  s32 expected 0x214 0x2;
                  s32 expected 0x218 0xf;
                  s32 expected 0x21c 0x111;
                  s32 expected 0x220 (0x100000 lsr 8);
                  s32 expected 0x22c 0x11000;
                  s32 expected 0x234 0x12000;
                  s32 expected 0x23c
                    (rencode fd "regCP_HQD_PQ_DOORBELL_CONTROL"
                       [ ("doorbell_offset", 6); ("doorbell_en", 1) ]);
                  s32 expected 0x244
                    (rencode fd "regCP_HQD_PQ_CONTROL"
                       [ ("rptr_block_size", 5); ("queue_size", 8) ]);
                  s32 expected 0x254
                    (rencode fd "regCP_HQD_IB_CONTROL"
                       [ ("min_ib_avail_size", 3) ]);
                  s32 expected 0x280 0x20004000;
                  s32 expected 0x288
                    (rencode fd "regCP_MQD_CONTROL" [ ("priv_state", 1) ]);
                  s32 expected 0x294 (0x13000 lsr 8);
                  s32 expected 0x29c
                    (rencode fd "regCP_HQD_EOP_CONTROL" [ ("eop_size", 9) ]);
                  List.iter
                    (fun off -> s32 expected off 0xffffffff)
                    [ 0x5c; 0x60; 0x68; 0x6c; 0xb0; 0xb4; 0xb8; 0xbc ];
                  equal_bytes expected
                    (Mmio.read_bytes fd.fvram ~off:(mqd_mc - 0x10000000)
                       ~len:0x800);
                  (* the register block mirrors the descriptor from its
                     0x80th dword on *)
                  let last = raddr fd.dev "regCP_HQD_PQ_WPTR_HI" in
                  let expected_mirror =
                    List.init (last - base + 1) (fun i ->
                        ( base + i,
                          Int32.to_int
                            (Bytes.get_int32_le expected ((0x80 + i) * 4))
                          land 0xffffffff ))
                  in
                  let active = raddr fd.dev "regCP_HQD_ACTIVE" in
                  equal
                    (list (pair int int))
                    (expected_mirror
                    @
                    if active >= base && active <= last then [ (active, 1) ]
                    else [])
                    (List.filter
                       (fun (a, _) -> a >= base && a <= last)
                       (List.rev !(fd.log)));
                  equal int 1 (rstore fd "regCP_HQD_ACTIVE")));
          test "fini_hw drains active queues" (fun () ->
              with_fake_dev (fun fd ->
                  let gfx = Gfx.create fd.dev in
                  let active = ref 1 in
                  Hashtbl.replace fd.reads (raddr fd.dev "regCP_HQD_ACTIVE")
                    (fun () ->
                      let v = !active in
                      active := 0;
                      v);
                  fd.log := [];
                  Gfx.fini_hw gfx;
                  equal int 0x2 (rstore fd "regCP_HQD_DEQUEUE_REQUEST");
                  equal int 0x1 (rstore fd "regSPI_COMPUTE_QUEUE_RESET")));
        ];
      group "interrupt rings"
        [
          test "init_hw programs both rings" (fun () ->
              with_fake_dev (fun fd ->
                  let ih = Ih.create fd.dev in
                  fd.log := [];
                  Ih.init_hw ih;
                  let r n = raddr fd.dev n in
                  equal (list int)
                    [
                      r "regIH_RB_BASE"; r "regIH_RB_BASE_HI";
                      r "regIH_RB_CNTL"; r "regIH_RB_WPTR_ADDR_LO";
                      r "regIH_RB_WPTR_ADDR_HI"; r "regIH_RB_WPTR";
                      r "regIH_RB_RPTR"; r "regIH_DOORBELL_RPTR";
                      r "regIH_RB_BASE_RING1"; r "regIH_RB_BASE_HI_RING1";
                      r "regIH_RB_CNTL_RING1"; r "regIH_RB_WPTR_RING1";
                      r "regIH_RB_RPTR_RING1"; r "regIH_DOORBELL_RPTR_RING1";
                      r "regIH_STORM_CLIENT_LIST_CNTL";
                      r "regIH_INT_FLOOD_CNTL"; r "regIH_MSI_STORM_CTRL";
                      r "regIH_RB_CNTL"; r "regIH_RB_CNTL_RING1";
                    ]
                    (List.map fst (List.rev !(fd.log)));
                  let cntl0 =
                    rencode fd "regIH_RB_CNTL"
                      [
                        ("mc_space", 4); ("wptr_overflow_clear", 1);
                        ("rb_size", 16); ("mc_snoop", 1);
                        ("wptr_overflow_enable", 1); ("rptr_rearm", 1);
                      ]
                  in
                  (* the final toggle ors in the enable bits *)
                  equal int
                    (cntl0
                    lor rencode fd "regIH_RB_CNTL"
                          [ ("rb_enable", 1); ("enable_intr", 1) ])
                    (rstore fd "regIH_RB_CNTL");
                  equal int
                    (rencode fd "regIH_RB_CNTL_RING1"
                       [
                         ("mc_space", 4); ("wptr_overflow_clear", 1);
                         ("rb_size", 16); ("mc_snoop", 1);
                         ("rb_full_drain_enable", 1); ("rb_enable", 1);
                       ])
                    (rstore fd "regIH_RB_CNTL_RING1");
                  equal int
                    (rencode fd "regIH_MSI_STORM_CTRL" [ ("delay", 3) ])
                    (rstore fd "regIH_MSI_STORM_CTRL");
                  (* the ring lives in device memory *)
                  let ring_paddr =
                    (rstore fd "regIH_RB_BASE" lsl 8) - 0x10000000
                  in
                  equal bool true
                    (ring_paddr >= 0 && ring_paddr < Mmio.size fd.fvram)));
          test "interrupt_handler decodes entries and flags faults" (fun () ->
              with_fake_dev (fun fd ->
                  let soc = Soc.create fd.dev in
                  let gmc = Gmc.create fd.dev in
                  let smu = Smu.create fd.dev in
                  let ih = Ih.create fd.dev in
                  Ih.init_hw ih;
                  let ring_paddr =
                    (rstore fd "regIH_RB_BASE" lsl 8) - 0x10000000
                  in
                  (* three entries: a trap to skip, a shader wave report
                     and a translation fault *)
                  let put_entry i dwords =
                    List.iteri
                      (fun j v ->
                        Mmio.write32 fd.fvram
                          (ring_paddr + (((i * 8) + j) * 4))
                          (Int32.of_int v))
                      dwords
                  in
                  put_entry 0 [ 0x310a; 0; 0; 0; 0; 0; 0; 0 ];
                  put_entry 1 [ 0xef14; 0; 0; 0; 0; 0x40; 0; 0 ];
                  put_entry 2 [ 0x0014; 0; 0; 0; 0; 0; 0; 0 ];
                  Hashtbl.replace fd.reads (raddr fd.dev "regIH_RB_WPTR")
                    (fun () -> 24 lsl 2);
                  fd.log := [];
                  Ih.interrupt_handler ih ~soc ~gmc ~smu;
                  equal bool true (Amdev.is_err_state fd.dev);
                  (* the fault status was cleared and the ring drained *)
                  equal int
                    (rencode fd "regGCVM_L2_PROTECTION_FAULT_CNTL"
                       [ ("clear_protection_fault_status_addr", 1) ])
                    (rstore fd "regGCVM_L2_PROTECTION_FAULT_CNTL");
                  equal int 24 (rstore fd "regIH_RB_RPTR")));
          test "trap sources do not mark the device errored" (fun () ->
              with_fake_dev (fun fd ->
                  let soc = Soc.create fd.dev in
                  let gmc = Gmc.create fd.dev in
                  let smu = Smu.create fd.dev in
                  let ih = Ih.create fd.dev in
                  Ih.init_hw ih;
                  let ring_paddr =
                    (rstore fd "regIH_RB_BASE" lsl 8) - 0x10000000
                  in
                  Mmio.write32 fd.fvram ring_paddr (Int32.of_int 0x310a);
                  Hashtbl.replace fd.reads (raddr fd.dev "regIH_RB_WPTR")
                    (fun () -> 8 lsl 2);
                  Ih.interrupt_handler ih ~soc ~gmc ~smu;
                  equal bool false (Amdev.is_err_state fd.dev);
                  equal int 8 (rstore fd "regIH_RB_RPTR")));
          test "drain clears an overflowed ring" (fun () ->
              with_fake_dev (fun fd ->
                  let ih = Ih.create fd.dev in
                  Ih.init_hw ih;
                  Hashtbl.replace fd.reads (raddr fd.dev "regIH_RB_WPTR")
                    (fun () -> (4 lsl 2) lor 1);
                  let cntl_before = rstore fd "regIH_RB_CNTL" in
                  fd.log := [];
                  Ih.drain ih;
                  let ovf =
                    rencode fd "regIH_RB_CNTL" [ ("wptr_overflow_clear", 1) ]
                  in
                  equal
                    (list (pair int int))
                    [
                      (raddr fd.dev "regIH_RB_RPTR", 4);
                      (raddr fd.dev "regIH_RB_WPTR", 4 lsl 2);
                      (raddr fd.dev "regIH_RB_CNTL", cntl_before lor ovf);
                      (raddr fd.dev "regIH_RB_CNTL", cntl_before land lnot ovf);
                    ]
                    (List.rev !(fd.log))));
          test "bus fault lines dump the error banks" (fun () ->
              with_fake_dev (fun fd ->
                  let soc = Soc.create fd.dev in
                  let gmc = Gmc.create fd.dev in
                  let smu = Smu.create fd.dev in
                  let ih = Ih.create fd.dev in
                  Ih.init_hw ih;
                  ack_messages fd;
                  Hashtbl.replace fd.store
                    (raddr fd.dev "regBIF_BX0_BIF_DOORBELL_INT_CNTL")
                    (rencode fd "regBIF_BX0_BIF_DOORBELL_INT_CNTL"
                       [ ("ras_cntlr_interrupt_status", 1) ]);
                  Ih.interrupt_handler ih ~soc ~gmc ~smu;
                  equal bool true (Amdev.is_err_state fd.dev);
                  equal int
                    (rencode fd "regBIF_BX0_BIF_DOORBELL_INT_CNTL"
                       [ ("ras_cntlr_interrupt_clear", 1) ])
                    (rstore fd "regBIF_BX0_BIF_DOORBELL_INT_CNTL")));
        ];
      group "sdma engines"
        [
          test "init_hw configures the engine and routes its doorbell"
            (fun () ->
              with_fake_dev (fun fd ->
                  let soc = Soc.create fd.dev in
                  let sdma = Sdma.create fd.dev in
                  fd.log := [];
                  Sdma.init_hw sdma ~soc;
                  equal
                    (list (pair int int))
                    [
                      ( raddr fd.dev "regSDMA0_WATCHDOG_CNTL",
                        rencode fd "regSDMA0_WATCHDOG_CNTL"
                          [ ("queue_hang_count", 100) ] );
                      ( raddr fd.dev "regSDMA0_UTCL1_CNTL",
                        rencode fd "regSDMA0_UTCL1_CNTL"
                          [ ("resp_mode", 3); ("redo_delay", 9) ] );
                      ( raddr fd.dev "regSDMA0_UTCL1_PAGE",
                        rencode fd "regSDMA0_UTCL1_PAGE"
                          [
                            ("rd_l2_policy", 2); ("wr_l2_policy", 3);
                            ("llc_noalloc", 1);
                          ] );
                      (raddr fd.dev "regSDMA0_F32_CNTL", 0);
                      ( raddr fd.dev "regSDMA0_CNTL",
                        rencode fd "regSDMA0_CNTL" [ ("trap_enable", 1) ] );
                      ( raddr fd.dev "regS2A_DOORBELL_ENTRY_2_CTRL",
                        rencode fd "regS2A_DOORBELL_ENTRY_2_CTRL"
                          [
                            ("s2a_doorbell_port2_enable", 1);
                            ("s2a_doorbell_port2_awid", 0xe);
                            ("s2a_doorbell_port2_range_size", 4);
                            ("s2a_doorbell_port2_awaddr_31_28_value", 0x3);
                            ("s2a_doorbell_port2_range_offset", 0x200);
                          ] );
                    ]
                    (List.rev !(fd.log))));
          test "setup_ring programs the queue and returns its doorbell"
            (fun () ->
              with_fake_dev (fun fd ->
                  let sdma = Sdma.create fd.dev in
                  fd.log := [];
                  let doorbell =
                    Sdma.setup_ring sdma ~ring_addr:0x40000 ~ring_size:0x800
                      ~rptr_addr:0x11000 ~wptr_addr:0x12000 ~idx:0
                  in
                  equal int 0x100 doorbell;
                  let r n = raddr fd.dev ("regSDMA0_QUEUE0" ^ n) in
                  equal
                    (list (pair int int))
                    [
                      (r "_MINOR_PTR_UPDATE", 1);
                      (r "_RB_RPTR", 0); (r "_RB_RPTR_HI", 0);
                      (r "_RB_WPTR", 0); (r "_RB_WPTR_HI", 0);
                      (r "_RB_BASE", 0x400); (r "_RB_BASE_HI", 0);
                      (r "_RB_RPTR_ADDR_LO", 0x11000);
                      (r "_RB_RPTR_ADDR_HI", 0);
                      (r "_RB_WPTR_POLL_ADDR_LO", 0x12000);
                      (r "_RB_WPTR_POLL_ADDR_HI", 0);
                      ( r "_DOORBELL_OFFSET",
                        rencode fd "regSDMA0_QUEUE0_DOORBELL_OFFSET"
                          [ ("offset", 0x200) ] );
                      ( r "_DOORBELL",
                        rencode fd "regSDMA0_QUEUE0_DOORBELL"
                          [ ("enable", 1) ] );
                      (r "_MINOR_PTR_UPDATE", 0);
                      ( r "_RB_CNTL",
                        rencode fd "regSDMA0_QUEUE0_RB_CNTL"
                          [
                            ("f32_wptr_poll_enable", 1);
                            ("rptr_writeback_enable", 1);
                            ("rptr_writeback_timer", 4); ("rb_enable", 1);
                            ("rb_priv", 1); ("rb_size", 9);
                          ] );
                      ( r "_IB_CNTL",
                        rencode fd "regSDMA0_QUEUE0_IB_CNTL"
                          [ ("ib_enable", 1) ] );
                    ]
                    (List.rev !(fd.log));
                  raises_match
                    (Exn.failure ~substring:"sdma queue 1 is not available")
                    (fun () ->
                      Sdma.setup_ring sdma ~ring_addr:0x40000 ~ring_size:0x800
                        ~rptr_addr:0x11000 ~wptr_addr:0x12000 ~idx:1)));
          test "fini_hw disables the queues and pulses the soft reset"
            (fun () ->
              with_fake_dev (fun fd ->
                  let sdma = Sdma.create fd.dev in
                  let (_ : int) =
                    Sdma.setup_ring sdma ~ring_addr:0x40000 ~ring_size:0x800
                      ~rptr_addr:0x11000 ~wptr_addr:0x12000 ~idx:0
                  in
                  let rb_cntl = rstore fd "regSDMA0_QUEUE0_RB_CNTL" in
                  fd.log := [];
                  Sdma.fini_hw sdma;
                  let en =
                    rencode fd "regSDMA0_QUEUE0_RB_CNTL" [ ("rb_enable", 1) ]
                  in
                  equal
                    (list (pair int int))
                    [
                      ( raddr fd.dev "regSDMA0_QUEUE0_RB_CNTL",
                        rb_cntl land lnot en );
                      (raddr fd.dev "regSDMA0_QUEUE0_IB_CNTL", 0);
                      (raddr fd.dev "regSDMA0_QUEUE0_DOORBELL", 0);
                      (raddr fd.dev "regSDMA0_QUEUE0_DOORBELL_OFFSET", 0);
                      ( raddr fd.dev "regGRBM_SOFT_RESET",
                        rencode fd "regGRBM_SOFT_RESET"
                          [ ("soft_reset_sdma0", 1) ] );
                      (raddr fd.dev "regGRBM_SOFT_RESET", 0);
                    ]
                    (List.rev !(fd.log))));
        ];
      group "boot machine"
        [
          test "cold boot: block order and the scratch-register stamps"
            (fun () ->
              with_fake_dev ~mp1:(13, 0, 0) (fun fd ->
                  let t = Am_boot.create ~fw:boot_fw fd.dev in
                  let (_ : psp_script) = script_boot fd t in
                  fd.log := [];
                  Am_boot.init t;
                  equal bool false t.Am_boot.partial_boot;
                  equal bool false (Amdev.is_booting fd.dev);
                  let log = writes fd in
                  (* one landmark register per block, in bring-up order:
                     soc, gmc, ih, psp, smu, then gfx and sdma *)
                  let order =
                    List.map
                      (fun name -> first_write_to fd name log)
                      [
                        "regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN";
                        "regMMMC_VM_AGP_BOT"; "regIH_RB_BASE";
                        "regMP0_SMN_C2PMSG_35"; "mmMP1_SMN_C2PMSG_66";
                        "regRLC_SPM_MC_CNTL"; "regSDMA0_WATCHDOG_CNTL";
                      ]
                  in
                  equal (list int) (List.sort compare order) order;
                  check_boot_stamps fd log));
          test "partial boot skips the boot-memory blocks" (fun () ->
              with_fake_dev ~mp1:(13, 0, 0) (fun fd ->
                  Hashtbl.replace fd.store
                    (raddr fd.dev "regSCRATCH_REG7")
                    0xA0000008;
                  let t = Am_boot.create ~fw:boot_fw fd.dev in
                  let (_ : psp_script) = script_boot fd t in
                  fd.log := [];
                  Am_boot.init t;
                  equal bool true t.Am_boot.partial_boot;
                  equal bool false (Amdev.is_booting fd.dev);
                  let log = writes fd in
                  (* the boot-memory blocks kept the previous session's
                     state *)
                  equal bool false
                    (wrote fd "regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN" log);
                  equal bool false (wrote fd "regMMMC_VM_AGP_BOT" log);
                  equal bool false (wrote fd "regIH_RB_BASE" log);
                  equal bool false (wrote fd "regMP0_SMN_C2PMSG_35" log);
                  (* the compute processors were reset, not brought up *)
                  equal bool true (wrote fd "regGRBM_SOFT_RESET" log);
                  equal bool false (wrote fd "regRLC_SPM_MC_CNTL" log);
                  equal bool true (wrote fd "regSDMA0_WATCHDOG_CNTL" log);
                  check_boot_stamps fd log));
          test "a suspect previous session forces the full path" (fun () ->
              (* an unclean shutdown: the session flag never cleared *)
              with_fake_dev ~mp1:(13, 0, 0) (fun fd ->
                  Hashtbl.replace fd.store
                    (raddr fd.dev "regSCRATCH_REG7")
                    0xA0000008;
                  Hashtbl.replace fd.store (raddr fd.dev "regSCRATCH_REG6") 1;
                  let t = Am_boot.create ~fw:boot_fw fd.dev in
                  let (_ : psp_script) = script_boot fd t in
                  fd.log := [];
                  Am_boot.init t;
                  equal bool false t.Am_boot.partial_boot;
                  equal bool true (wrote fd "regMP0_SMN_C2PMSG_35" (writes fd)));
              (* a latched translation fault *)
              with_fake_dev ~mp1:(13, 0, 0) (fun fd ->
                  Hashtbl.replace fd.store
                    (raddr fd.dev "regSCRATCH_REG7")
                    0xA0000008;
                  let t = Am_boot.create ~fw:boot_fw fd.dev in
                  Hashtbl.replace fd.store
                    (raddr fd.dev (Gmc.pf_status_reg t.Am_boot.gmc Gmc.Gc))
                    1;
                  let (_ : psp_script) = script_boot fd t in
                  fd.log := [];
                  Am_boot.init t;
                  equal bool false t.Am_boot.partial_boot;
                  equal bool true (wrote fd "regMP0_SMN_C2PMSG_35" (writes fd))));
          test "a live external driver gets a mode1 reset first" (fun () ->
              with_fake_dev ~mp1:(13, 0, 0) (fun fd ->
                  let t = Am_boot.create ~fw:boot_fw fd.dev in
                  let s = script_boot fd t in
                  (* the previous driver's firmware still answers *)
                  Hashtbl.replace fd.reads (s.r 81) (fun () -> 1);
                  let r54 = raddr fd.dev "mmMP1_SMN_C2PMSG_54" in
                  let r75 = raddr fd.dev "mmMP1_SMN_C2PMSG_75" in
                  Hashtbl.replace fd.wr_hooks r75 (fun _ ->
                      Hashtbl.replace fd.store r54 1);
                  fd.log := [];
                  Am_boot.init t;
                  let log = writes fd in
                  (* the debug-mailbox reset ran before any block came up *)
                  equal bool true
                    (first_write_to fd "mmMP1_SMN_C2PMSG_75" log
                    < first_write_to fd "regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN"
                        log);
                  equal bool false t.Am_boot.partial_boot;
                  check_boot_stamps fd log));
          test "fini drops the clocks and records the session flag" (fun () ->
              with_fake_dev ~mp1:(13, 0, 0) (fun fd ->
                  let t = Am_boot.create ~fw:boot_fw fd.dev in
                  script_smu fd;
                  fd.log := [];
                  Am_boot.fini t;
                  let log = writes fd in
                  (* the engines went down, then the clocks *)
                  equal bool true (wrote fd "regGRBM_SOFT_RESET" log);
                  equal bool true (wrote fd "mmMP1_SMN_C2PMSG_66" log);
                  let r6, v6 = last_write log in
                  equal int (raddr fd.dev "regSCRATCH_REG6") r6;
                  equal int 0 v6;
                  (* a faulted session records the error flag instead *)
                  Amdev.set_err_state fd.dev true;
                  fd.log := [];
                  Am_boot.fini t;
                  let r6, v6 = last_write (writes fd) in
                  equal int (raddr fd.dev "regSCRATCH_REG6") r6;
                  equal int 1 v6));
          test "recover resets the compute processors" (fun () ->
              with_fake_dev ~mp1:(13, 0, 0) (fun fd ->
                  let t = Am_boot.create ~fw:boot_fw fd.dev in
                  fd.log := [];
                  (* a healthy device declines *)
                  equal bool false (Am_boot.recover t);
                  equal (list (pair int int)) [] (writes fd);
                  (* a faulted device restarts the processors and clears
                     its error state *)
                  Amdev.set_err_state fd.dev true;
                  equal bool true (Am_boot.recover t);
                  equal bool false (Amdev.is_err_state fd.dev);
                  equal bool true (wrote fd "regGRBM_SOFT_RESET" (writes fd));
                  equal int (0x2000 lsr 2)
                    (rstore fd "regCP_MEC_RS64_PRGRM_CNTR_START");
                  (* force runs it on a healthy device too *)
                  fd.log := [];
                  equal bool true (Am_boot.recover ~force:true t)));
        ];
    ]
