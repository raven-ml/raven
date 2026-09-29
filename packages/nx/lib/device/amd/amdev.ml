(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An AMD GPU driven over PCI: its BARs, registers, discovery table, firmware
   and page-table format. *)

module D = Amd_defs
module Mmio = Nx_device_support.Mmio
module Pci = Nx_device_support.Pci
module Page_table = Nx_device_support.Page_table

type version = int * int * int

external now_ms : unit -> (int[@untagged])
  = "caml_nx_amd_now_ms_byte" "caml_nx_amd_now_ms"
[@@noalloc]

let sleep_ms ms = Unix.sleepf (float_of_int ms /. 1000.)

(* Polls [f] until it returns [v], for at most [timeout_ms]; a timeout is a
   device failure. *)
let wait_cond ?(timeout_ms = 10_000) ~msg f v =
  let start = now_ms () in
  let rec go () =
    let x = f () in
    if x = v then ()
    else if now_ms () - start >= timeout_ms then
      failwith
        (Printf.sprintf "%s: timed out after %d ms (%d, expected %d)" msg
           timeout_ms x v)
    else go ()
  in
  go ()

let lo32 v = v land 0xffff_ffff
let hi32 v = (v lsr 32) land 0xffff_ffff

(* Struct fields, as the definitions give them: (byte offset, width). *)
let get s base (off, width) =
  let at = base + off in
  match width with
  | 1 -> Char.code s.[at]
  | 2 -> String.get_uint16_le s at
  | 4 -> Int32.to_int (String.get_int32_le s at) land 0xffff_ffff
  | 8 -> Int64.to_int (String.get_int64_le s at)
  | w -> invalid_arg (Printf.sprintf "a %d-byte field" w)

let get_bits s base (bit, width) =
  let v =
    get s (base + (bit / 8)) (0, 1)
    lor (get s (base + (bit / 8) + 1) (0, 1) lsl 8)
  in
  (v lsr (bit mod 8)) land ((1 lsl width) - 1)

let set b (off, width) v =
  match width with
  | 1 -> Bytes.set_uint8 b off (v land 0xff)
  | 2 -> Bytes.set_uint16_le b off (v land 0xffff)
  | 4 -> Bytes.set_int32_le b off (Int32.of_int (v land 0xffff_ffff))
  | 8 -> Bytes.set_int64_le b off (Int64.of_int v)
  | w -> invalid_arg (Printf.sprintf "a %d-byte field" w)

(* GC information from the discovery table. *)
type gc_info = {
  num_se : int;
  cu_per_sa : int;
  sh_per_se : int;
  max_scratch_slots_per_cu : int;
  max_waves_per_simd : int;
  lds_size : int; (* KiB *)
}

type t = {
  pci : Pci.t;
  bus : string;
  vram : Mmio.t; (* BAR 0: the GPU's memory, all of it on a large BAR *)
  doorbells : Mmio.t; (* BAR 2 *)
  mmio : Mmio.t; (* BAR 5: registers *)
  vram_size : int;
  large_bar : bool;
  is_vf : bool;
  mutable vf_access : int; (* the lease a VF holds, 0 if none *)
  mutable rlcg : (int * int) list; (* registers a VF reaches through the RLC *)
  ip_ver : (int * version) list; (* by IP block *)
  bases : (int * (int * int array) list) list; (* by block, then instance *)
  harvested : (int * int list) list;
  gc_info : gc_info;
  reserved_vram : int;
  regs : (string, Am_reg.t) Hashtbl.t;
  aids : int list;
  xccs : int;
  (* memory controller *)
  xgmi_phys_id : int;
  xgmi_max_region : int;
  xgmi_seg_sz : int;
  paddr_base : int;
  fb_base : int;
  fb_end : int;
  mc_base : int;
  address_mask : int;
  mm_insts : int list;
  mutable gc_hub : bool; (* whether the GC hub is initialized *)
  mutable kiq : (Mmio.t * int) option; (* a VF's KIQ memory and address *)
  mutable errors : string list; (* faults the interrupt handler found *)
}

let ip_version ip_ver hwip =
  match List.assoc_opt hwip ip_ver with
  | Some v -> v
  | None -> failwith (Printf.sprintf "the GPU has no IP block %d" hwip)

let version d hwip = ip_version d.ip_ver hwip
let gc d = version d D.gc_hwip

(* Registers *)

let reg d name =
  match Hashtbl.find_opt d.regs name with
  | Some r -> r
  | None ->
      failwith
        (Printf.sprintf "%s: no register %s on this GPU (GC %s)" d.bus name
           (Am_reg.pp_version (gc d)))

let has_reg d name = Hashtbl.mem d.regs name
let words d = Mmio.length d.mmio / 4

let rec rreg d ?(inst = 0) ?(direct = false) a =
  if (not direct) && List.exists (fun (lo, hi) -> lo <= a && a <= hi) d.rlcg
  then rlcg d a 0 inst ~read:true
  else if a >= words d then begin
    write d "regBIF_BX_PF0_RSMU_INDEX" ~value:(a * 4) [];
    read d "regBIF_BX_PF0_RSMU_DATA"
  end
  else Mmio.get32 d.mmio (a * 4)

and wreg d ?(inst = 0) ?(direct = false) a v =
  if (not direct) && List.exists (fun (lo, hi) -> lo <= a && a <= hi) d.rlcg
  then ignore (rlcg d a v inst ~read:false)
  else if a >= words d then begin
    write d "regBIF_BX_PF0_RSMU_INDEX" ~value:(a * 4) [];
    write d "regBIF_BX_PF0_RSMU_DATA" ~value:v []
  end
  else Mmio.set32 d.mmio (a * 4) v

and read d ?(inst = 0) ?direct name =
  rreg d ~inst ?direct (Am_reg.addr ~inst (reg d name))

and write d ?(inst = 0) ?direct ?(value = 0) name kvs =
  let r = reg d name in
  wreg d ~inst ?direct (Am_reg.addr ~inst r) (value lor Am_reg.encode r kvs)

(* The RLC gateway: a VF writes the registers it guards through scratch
   registers, and the RLC performs the access. *)
and rlcg d a v inst ~read:rd =
  let gfx_cntl = Am_reg.addr ~inst (reg d "regGRBM_GFX_CNTL")
  and gfx_index = Am_reg.addr ~inst (reg d "regGRBM_GFX_INDEX") in
  if a = gfx_cntl || a = gfx_index then begin
    write d ~inst ~direct:true ~value:v
      (if a = gfx_cntl then "regSCRATCH_REG2" else "regSCRATCH_REG3")
      [];
    v
  end
  else begin
    let cmd = ((a lor if rd then 1 lsl 28 else 0) lsl 32) lor v in
    write d ~inst ~direct:true ~value:(lo32 cmd) "regSCRATCH_REG0" [];
    write d ~inst ~direct:true ~value:(hi32 cmd) "regSCRATCH_REG1" [];
    write d ~inst ~direct:true ~value:1 "regRLC_SPARE_INT" [];
    wait_cond
      ~msg:(Printf.sprintf "RLC gateway on 0x%x" a)
      (fun () -> read d ~inst ~direct:true "regSCRATCH_REG1" land 0xFFFFF)
      0;
    read d ~inst ~direct:true "regSCRATCH_REG0"
  end

let fields d ?inst name = Am_reg.decode (reg d name) (read d ?inst name)

let field d ?inst name f =
  match List.assoc_opt f (fields d ?inst name) with
  | Some v -> v
  | None -> failwith (Printf.sprintf "%s has no field %s" name f)

let update d ?(inst = 0) name kvs =
  let r = reg d name in
  let old = read d ~inst name in
  write d ~inst
    ~value:(old land lnot (Am_reg.mask r (List.map fst kvs)))
    name kvs

let write_pair d ?inst ?direct base ~lo ~hi v =
  write d ?inst ?direct ~value:(lo32 v) (base ^ lo) [];
  write d ?inst ?direct ~value:(hi32 v) (base ^ hi) []

(* Writes [v] to the PCIe register [a] of the AID [aid] through the indirect
   window. *)
let indirect_wreg_pcie d ?(aid = 0) a v =
  let a =
    a * 4 lor if aid > 0 then ((aid land 3) lsl 32) lor (1 lsl 34) else 0
  in
  write d ~value:(lo32 a) "regBIF_BX0_PCIE_INDEX2" [];
  if hi32 a > 0 then
    write d ~value:(hi32 a land 0xff) "regBIF_BX0_PCIE_INDEX2_HI" [];
  write d ~value:v "regBIF_BX0_PCIE_DATA2" [];
  if hi32 a > 0 then write d ~value:0 "regBIF_BX0_PCIE_INDEX2_HI" []

let paddr2mc d pa = d.mc_base + pa
let paddr2xgmi d pa = d.paddr_base + pa
let xgmi2paddr d pa = pa - d.paddr_base
let is_hive d = d.xgmi_seg_sz > 0 && d.xgmi_max_region > 0

(* The VF mailbox *)

(* Sends [req] to the host's PF through the mailbox in [mmio] and, if [ready],
   waits until it grants access. Returns the lease to give back. *)
let vf_request ?(ready = true) mmio req =
  let mb = Mmio.sub mmio D.nv_maibox_control_trn_offset_byte 2 in
  Mmio.set8 mb 0 0;
  wait_cond ~timeout_ms:1000 ~msg:"VF mailbox acknowledgement did not clear"
    (fun () -> Mmio.get8 mb 0 land 2)
    0;
  List.iteri
    (fun i w -> Mmio.set32 mmio ((D.mmmailbox_msgbuf_trn_dw0 + i) * 4) w)
    [ req; 0; 0; 0 ];
  Mmio.set8 mb 0 1;
  wait_cond ~timeout_ms:D.nv_mailbox_poll_ack_timedout
    ~msg:(Printf.sprintf "VF mailbox request 0x%x was not acked" req)
    (fun () -> Mmio.get8 mb 0 land 2)
    2;
  Mmio.set8 mb 0 0;
  if ready then begin
    wait_cond ~timeout_ms:D.nv_mailbox_poll_msg_timedout
      ~msg:"VF mailbox: the PF never granted access"
      (fun () -> Mmio.get32 mmio (D.mmmailbox_msgbuf_rcv_dw0 * 4))
      D.idh_ready_to_access_gpu;
    Mmio.set8 mb 1 2
  end;
  req + 1

let release_vf_access d =
  let lease = d.vf_access in
  d.vf_access <- 0;
  try ignore (vf_request ~ready:false d.mmio lease) with Failure _ -> ()

(* Discovery *)

(* Reads VRAM a word at a time through the MM index window, for a small BAR that
   does not reach the discovery table. *)
let read_vram mmio a n =
  let b = Bytes.create n in
  for i = 0 to (n / 4) - 1 do
    let w = a + (4 * i) in
    Mmio.set32 mmio (0x06 * 4) (w lsr 31);
    Mmio.set32 mmio (0x00 * 4) (w land 0x7FFFFFFF lor 0x80000000);
    Bytes.set_int32_le b (4 * i) (Int32.of_int (Mmio.get32 mmio (0x01 * 4)))
  done;
  Bytes.to_string b

type discovery = {
  ip_ver : (int * version) list;
  bases : (int * (int * int array) list) list;
  harvested : (int * int list) list;
  gc_info : gc_info;
}

(* Version 1 describes RDNA GPUs by work-group processors, version 2 CDNA GPUs
   by compute units. *)
let parse_gc_info tbl off =
  let f field = get tbl off field in
  match f D.Gc_info_v1_0.header__version_major with
  | 1 ->
      let open D.Gc_info_v1_0 in
      {
        num_se = f gc_num_se;
        cu_per_sa = 2 * (f gc_num_wgp0_per_sa + f gc_num_wgp1_per_sa);
        sh_per_se = f gc_num_sa_per_se;
        max_scratch_slots_per_cu = f gc_max_scratch_slots_per_cu;
        max_waves_per_simd = f gc_max_waves_per_simd;
        lds_size = f gc_lds_size;
      }
  | 2 ->
      let open D.Gc_info_v2_0 in
      {
        num_se = f gc_num_se;
        cu_per_sa = f gc_num_cu_per_sh;
        sh_per_se = f gc_num_sh_per_se;
        max_scratch_slots_per_cu = f gc_max_scratch_slots_per_cu;
        max_waves_per_simd = f gc_max_waves_per_simd;
        lds_size = f gc_lds_size;
      }
  | v -> failwith (Printf.sprintf "unknown GC information version %d" v)

let parse_discovery tbl =
  let table i =
    let off, stride = D.Binary_header.table_list in
    get tbl (off + (i * stride)) D.Table_info.offset
  in
  let fail fmt = Printf.ksprintf failwith fmt in
  if get tbl 0 D.Binary_header.binary_signature <> D.binary_signature then
    fail "the discovery table's signature is wrong";
  let ih = table D.ip_discovery in
  if get tbl ih D.Ip_discovery_header.signature <> D.discovery_table_signature
  then fail "the IP discovery header's signature is wrong";
  let wide = get_bits tbl ih D.Ip_discovery_header.base_addr_64_bit_bits = 1 in
  let dies = get tbl ih D.Ip_discovery_header.num_dies in
  let bases = Hashtbl.create 16 and ip_ver = Hashtbl.create 16 in
  for die = 0 to dies - 1 do
    let die_off =
      let off, stride = D.Ip_discovery_header.die_info in
      get tbl (ih + off + (die * stride)) D.Die_info.die_offset
    in
    let n = get tbl die_off D.Die_header.num_ips in
    let at = ref (die_off + D.Die_header.sizeof) in
    for _ = 1 to n do
      let ip = !at in
      let hw_id = get tbl ip D.Ip_v4.hw_id in
      let inst = get tbl ip D.Ip_v4.instance_number in
      let count = get tbl ip D.Ip_v4.num_base_address in
      let width = if wide then 8 else 4 in
      let segs =
        Array.init count (fun i -> get tbl (ip + 8 + (i * width)) (0, width))
      in
      List.iter
        (fun (hw_ip, id) ->
          if id = hw_id then begin
            let insts =
              Option.value ~default:[] (Hashtbl.find_opt bases hw_ip)
            in
            Hashtbl.replace bases hw_ip
              ((inst, segs) :: List.remove_assoc inst insts);
            Hashtbl.replace ip_ver hw_ip
              ( get tbl ip D.Ip_v4.major,
                get tbl ip D.Ip_v4.minor,
                get tbl ip D.Ip_v4.revision )
          end)
        D.hw_id_map;
      at := ip + 8 + (width * count)
    done
  done;
  (* Harvested instances are fused off and must not be touched. *)
  let harvested = Hashtbl.create 4 in
  let hv = table D.harvest_info in
  if hv <> 0 && get tbl hv (0, 4) = D.harvest_table_signature then begin
    let ip_of id =
      List.fold_left
        (fun r (ip, i) -> if i = id then Some ip else r)
        None D.hw_id_map
    in
    for k = 0 to 31 do
      let e = get tbl (hv + 8 + (4 * k)) (0, 4) in
      match ip_of (e land 0xffff) with
      | Some ip ->
          let l = Option.value ~default:[] (Hashtbl.find_opt harvested ip) in
          Hashtbl.replace harvested ip (((e lsr 16) land 0xff) :: l)
      | None -> ()
    done
  end;
  let to_list h = Hashtbl.fold (fun k v acc -> (k, v) :: acc) h [] in
  {
    ip_ver = to_list ip_ver;
    bases = to_list bases;
    harvested = to_list harvested;
    gc_info = parse_gc_info tbl (table D.gc);
  }

(* The registers each block is programmed through, the later ones taking
   precedence, as the MP1 view of the SMU's message registers does. *)
let build_regs ~ip_ver ~bases ~is_vf =
  let regs = Hashtbl.create 1024 and rlcg = ref [] in
  let ver = ip_version ip_ver in
  let bases_of hwip = Option.value ~default:[] (List.assoc_opt hwip bases) in
  let gc_ver = ver D.gc_hwip in
  let mods =
    [
      ("mp", D.mp0_hwip);
      ("hdp", D.hdp_hwip);
      ("gc", D.gc_hwip);
      ("mmhub", D.mmhub_hwip);
      ("osssys", D.osssys_hwip);
      ((if compare gc_ver (12, 0, 0) < 0 then "nbio" else "nbif"), D.nbio_hwip);
    ]
    @
    if List.mem (ver D.sdma0_hwip) [ (4, 4, 2); (4, 4, 4) ] then
      [ ("sdma", D.sdma0_hwip) ]
    else []
  in
  List.iter
    (fun (prefix, hwip) ->
      List.iter
        (fun (name, r) -> Hashtbl.replace regs name r)
        (Am_reg.registers prefix (ver hwip) ~bases:(bases_of hwip));
      if prefix = "gc" && is_vf then
        rlcg :=
          List.sort compare
            (List.concat_map
               (fun (_, segs) ->
                 List.filter_map
                   (fun (seg, top) ->
                     if seg < Array.length segs then
                       Some (segs.(seg), segs.(seg) + top)
                     else None)
                   (D.gc_rlcg_extent (Am_reg.family "gc" gc_ver)))
               (bases_of hwip)))
    mods;
  List.iter
    (fun (name, r) -> Hashtbl.replace regs name r)
    (Am_reg.registers "mp" (11, 0, 0) ~bases:(bases_of D.mp1_hwip));
  (regs, !rlcg)

(* Firmware *)

let firmware_url =
  "https://gitlab.com/kernel-firmware/linux-firmware/-/raw/" ^ D.firmware_commit
  ^ "/"

type firmware = {
  sos : (int * string) list; (* the PSP's own components, by type *)
  descs : (int list * string) list; (* images the PSP loads, in order *)
  smu_psp : (int list * string) option; (* the SMU's, loaded before the TMR *)
  ucode_start : (string * int) list;
}

let fmt_ver (a, b, c) = Printf.sprintf "%d_%d_%d" a b c

let load_firmware ?dir d =
  let load name =
    let path = "amdgpu/" ^ name in
    match List.assoc_opt name D.firmware_sha256 with
    | None ->
        failwith (Printf.sprintf "no pinned firmware %s for this GPU" name)
    | Some sha256 -> (
        match
          Nx_device_support.Firmware.get ?dir ~url:firmware_url path ~sha256
        with
        | Ok blob -> blob
        | Error why -> failwith why)
  in
  let desc blob off n types = (types, String.sub blob off n) in
  let common blob f = get blob 0 f in
  let hmajor blob = common blob D.Common_firmware_header.header_version_major in
  let hminor blob = common blob D.Common_firmware_header.header_version_minor in
  let ucode_off blob =
    common blob D.Common_firmware_header.ucode_array_offset_bytes
  in
  let ucode_size blob = common blob D.Common_firmware_header.ucode_size_bytes in
  (* SOS *)
  let blob =
    load (Printf.sprintf "psp_%s_sos.bin" (fmt_ver (version d D.mp0_hwip)))
  in
  let count, bins =
    match hminor blob with
    | 1 ->
        ( get blob 0 D.Psp_firmware_header_v2_1.psp_aux_fw_bin_index,
          fst D.Psp_firmware_header_v2_1.psp_fw_bin )
    | _ ->
        ( get blob 0 D.Psp_firmware_header_v2_0.psp_fw_bin_count,
          fst D.Psp_firmware_header_v2_0.psp_fw_bin )
  in
  let sos =
    List.init count (fun i ->
        let at = bins + (i * D.Psp_fw_bin_desc.sizeof) in
        let start =
          get blob at D.Psp_fw_bin_desc.offset_bytes + ucode_off blob
        in
        ( get blob at D.Psp_fw_bin_desc.fw_type,
          String.sub blob start (get blob at D.Psp_fw_bin_desc.size_bytes) ))
  in
  let descs = ref [] and smu_psp = ref None and ucode_start = ref [] in
  let add ds = descs := !descs @ ds in
  (* SMU *)
  if version d D.mp1_hwip <> (13, 0, 12) then begin
    let blob =
      load (Printf.sprintf "smu_%s.bin" (fmt_ver (version d D.mp1_hwip)))
    in
    if compare (gc d) (11, 0, 0) >= 0 then
      smu_psp :=
        Some
          (desc blob (ucode_off blob) (ucode_size blob) [ D.gfx_fw_type_smu ])
    else
      let n = get blob 0 D.Smc_firmware_header_v2_1.pptable_count in
      let first = get blob 0 D.Smc_firmware_header_v2_1.pptable_entry_offset in
      for i = 0 to n - 1 do
        let e = first + (i * D.Smc_soft_pptable_entry.sizeof) in
        if get blob e D.Smc_soft_pptable_entry.id = 0x50325358 then
          add
            [
              desc blob
                (get blob e D.Smc_soft_pptable_entry.ppt_offset_bytes)
                (get blob e D.Smc_soft_pptable_entry.ppt_size_bytes)
                [ D.gfx_fw_type_p2s_table ];
            ]
      done
  end;
  (* SDMA *)
  let blob =
    load (Printf.sprintf "sdma_%s.bin" (fmt_ver (version d D.sdma0_hwip)))
  in
  (match hmajor blob with
  | 1 ->
      add
        [
          desc blob (ucode_off blob) (ucode_size blob)
            D.
              [
                gfx_fw_type_sdma0;
                gfx_fw_type_sdma1;
                gfx_fw_type_sdma2;
                gfx_fw_type_sdma3;
              ];
        ]
  | 2 ->
      let open D.Sdma_firmware_header_v2_0 in
      add
        [
          desc blob
            (get blob 0 ctl_ucode_offset)
            (get blob 0 ctl_ucode_size_bytes)
            [ D.gfx_fw_type_sdma_ucode_th1 ];
          desc blob (ucode_off blob)
            (get blob 0 ctx_ucode_size_bytes)
            [ D.gfx_fw_type_sdma_ucode_th0 ];
        ]
  | _ ->
      add
        [
          desc blob (ucode_off blob)
            (get blob 0 D.Sdma_firmware_header_v3_0.ucode_size_bytes)
            [ D.gfx_fw_type_sdma_ucode_th0 ];
        ]);
  (* PFP, ME, MEC *)
  (* Version 1 images carry a jump table, which only the MEC's has. *)
  let engines =
    (if compare (gc d) (12, 0, 0) >= 0 then
       [
         ( "PFP",
           D.gfx_fw_type_cp_pfp,
           None,
           D.gfx_fw_type_rs64_pfp,
           D.gfx_fw_type_rs64_pfp_p0_stack );
         ( "ME",
           D.gfx_fw_type_cp_me,
           None,
           D.gfx_fw_type_rs64_me,
           D.gfx_fw_type_rs64_me_p0_stack );
       ]
     else [])
    @ [
        ( "MEC",
          D.gfx_fw_type_cp_mec,
          Some D.gfx_fw_type_cp_mec_me1,
          D.gfx_fw_type_rs64_mec,
          D.gfx_fw_type_rs64_mec_p0_stack );
      ]
  in
  List.iter
    (fun (name, code, jt, rs64, stack) ->
      let blob =
        load
          (Printf.sprintf "gc_%s_%s.bin"
             (fmt_ver (gc d))
             (String.lowercase_ascii name))
      in
      let off = ucode_off blob in
      if hmajor blob = 1 then begin
        let open D.Gfx_firmware_header_v1_0 in
        let jt_size = get blob 0 jt_size and jt_offset = get blob 0 jt_offset in
        let jt =
          match jt with
          | Some jt -> jt
          | None -> failwith (Printf.sprintf "a version 1 %s image" name)
        in
        add
          [
            desc blob off (ucode_size blob - (jt_size * 4)) [ code ];
            desc blob (off + (jt_offset * 4)) (jt_size * 4) [ jt ];
          ]
      end
      else begin
        let open D.Gfx_firmware_header_v2_0 in
        add
          [
            desc blob off (get blob 0 ucode_size_bytes) [ rs64 ];
            desc blob
              (get blob 0 data_offset_bytes)
              (get blob 0 data_size_bytes)
              [ stack ];
          ];
        ucode_start :=
          ( name,
            get blob 0 ucode_start_addr_lo
            lor (get blob 0 ucode_start_addr_hi lsl 32) )
          :: !ucode_start
      end)
    engines;
  (* IMU *)
  if compare (gc d) (11, 0, 0) >= 0 then begin
    let blob = load (Printf.sprintf "gc_%s_imu.bin" (fmt_ver (gc d))) in
    let open D.Imu_firmware_header_v1_0 in
    let off = ucode_off blob in
    let i = get blob 0 imu_iram_ucode_size_bytes in
    add
      [
        desc blob off i [ D.gfx_fw_type_imu_i ];
        desc blob (off + i)
          (get blob 0 imu_dram_ucode_size_bytes)
          [ D.gfx_fw_type_imu_d ];
      ]
  end;
  (* RLC *)
  let blob = load (Printf.sprintf "gc_%s_rlc.bin" (fmt_ver (gc d))) in
  let minor = hminor blob in
  if minor = 1 then begin
    let open D.Rlc_firmware_header_v2_1 in
    add
      [
        desc blob
          (get blob 0 save_restore_list_cntl_offset_bytes)
          (get blob 0 save_restore_list_cntl_size_bytes)
          [ D.gfx_fw_type_rlc_restore_list_srm_cntl ];
        desc blob
          (get blob 0 save_restore_list_gpm_offset_bytes)
          (get blob 0 save_restore_list_gpm_size_bytes)
          [ D.gfx_fw_type_rlc_restore_list_gpm_mem ];
        desc blob
          (get blob 0 save_restore_list_srm_offset_bytes)
          (get blob 0 save_restore_list_srm_size_bytes)
          [ D.gfx_fw_type_rlc_restore_list_srm_mem ];
      ]
  end;
  if minor >= 2 then begin
    let open D.Rlc_firmware_header_v2_2 in
    add
      [
        desc blob
          (get blob 0 rlc_iram_ucode_offset_bytes)
          (get blob 0 rlc_iram_ucode_size_bytes)
          [ D.gfx_fw_type_rlc_iram ];
        desc blob
          (get blob 0 rlc_dram_ucode_offset_bytes)
          (get blob 0 rlc_dram_ucode_size_bytes)
          [ D.gfx_fw_type_rlc_dram_boot ];
      ]
  end;
  if minor = 3 then begin
    let open D.Rlc_firmware_header_v2_3 in
    add
      [
        desc blob
          (get blob 0 rlcp_ucode_offset_bytes)
          (get blob 0 rlcp_ucode_size_bytes)
          [ D.gfx_fw_type_rlc_p ];
        desc blob
          (get blob 0 rlcv_ucode_offset_bytes)
          (get blob 0 rlcv_ucode_size_bytes)
          [ D.gfx_fw_type_rlc_v ];
      ]
  end;
  add [ desc blob (ucode_off blob) (ucode_size blob) [ D.gfx_fw_type_rlc_g ] ];
  { sos; descs = !descs; smu_psp = !smu_psp; ucode_start = !ucode_start }

(* Page tables *)

let mtype_uc gc =
  match gc with
  | 9, _, _ -> D.soc9_mtype_uc
  | 11, _, _ -> D.soc11_mtype_uc
  | 12, _, _ -> D.soc12_mtype_uc
  | v -> failwith ("no memory types for GC " ^ Am_reg.pp_version v)

let ( |: ) = Int64.logor
let bit b = Int64.shift_left 1L b
let flag cond f = if cond then f else 0L

let pte_flags ~gc ~level ~table ~fragment ~uncached ~system ~snooped ~valid =
  let mtype = Int64.of_int (if uncached then mtype_uc gc else 0) in
  let base =
    flag system D.amdgpu_pte_system
    |: flag snooped D.amdgpu_pte_snooped
    |: flag valid D.amdgpu_pte_valid
    |: Int64.shift_left (Int64.of_int (fragment land 0x1f)) 7
    |: flag (not table)
         (D.amdgpu_pte_writeable |: D.amdgpu_pte_readable
        |: D.amdgpu_pte_executable)
  in
  let leaf_of_pd = (not table) && level <> D.amdgpu_vm_ptb in
  if compare gc (12, 0, 0) >= 0 then
    base |: Int64.shift_left mtype 54
    |:
    if leaf_of_pd then D.amdgpu_pde_pte_gfx12
    else flag (not table) D.amdgpu_pte_is_pte
  else if compare gc (10, 0, 0) >= 0 then
    base |: Int64.shift_left mtype 48 |: flag leaf_of_pd D.amdgpu_pde_pte
  else
    base |: Int64.shift_left mtype 57
    |: flag (table && level = D.amdgpu_vm_pdb1) (Int64.shift_left 0x9L 59)
    |: flag (table && level = D.amdgpu_vm_pdb0) D.amdgpu_pte_tf
    |: flag
         ((not table) && level <> D.amdgpu_vm_ptb && level <> D.amdgpu_vm_pdb0)
         D.amdgpu_pde_pte

let huge ~gc ~level e =
  let has f = Int64.logand e f <> 0L in
  if compare gc (10, 0, 0) < 0 then
    if level <> D.amdgpu_vm_pdb0 then has D.amdgpu_pde_pte
    else not (has D.amdgpu_pte_tf)
  else
    has
      (if compare gc (12, 0, 0) >= 0 then D.amdgpu_pde_pte_gfx12
       else D.amdgpu_pde_pte)

let address_bits = 0x0000_FFFF_FFFF_F000L

let entry d ~flush =
  let open Page_table in
  {
    levels = [ 12; 21; 30; 39 ];
    bits = 48;
    first = D.amdgpu_vm_pdb2;
    get = (fun ~table i -> Mmio.get64 d.vram (table + (8 * i)));
    set = (fun ~table i e -> Mmio.set64 d.vram (table + (8 * i)) e);
    encode =
      (fun ~level ~table space ~uncached ~snooped ~fragment ~valid pa ->
        let pa = if space = Phys then paddr2xgmi d pa else pa in
        if pa land d.address_mask <> pa then
          invalid_arg (Printf.sprintf "an invalid physical address 0x%x" pa);
        pte_flags ~gc:(gc d) ~level ~table ~fragment ~uncached
          ~system:(space = Sys) ~snooped ~valid
        |: Int64.logand (Int64.of_int pa) address_bits);
    valid = (fun e -> Int64.logand e D.amdgpu_pte_valid <> 0L);
    leaf = (fun ~level e -> level = D.amdgpu_vm_ptb || huge ~gc:(gc d) ~level e);
    address =
      (fun e -> xgmi2paddr d (Int64.to_int (Int64.logand e address_bits)));
    large = (fun ~level -> level >= D.amdgpu_vm_pdb2);
    zero = (fun pa n -> Mmio.fill d.vram pa n '\000');
    flush;
  }
