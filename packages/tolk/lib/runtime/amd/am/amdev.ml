(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module Am = Amd_tables.Am_defs
module Fw = Amd_tables.Fw_defs
module Helpers = Tolk.Helpers
module Mmio = Tolk_hcq.Hcq.Mmio
module System = Tolk_hcq.System
module Memory = Tolk.Memory
module Tlsf = Tolk.Tlsf

(* Am_register: amdev.py AMRegister *)

module Am_register = struct
  type t = {
    reg : Amd_tables.Reg.t;
    rreg : int -> int;
    wreg : int -> int -> unit;
  }

  let make ~reg ~rreg ~wreg = { reg; rreg; wreg }
  let reg t = t.reg
  let read t = t.rreg t.reg.Amd_tables.Reg.addr
  let read_bitfields t = Amd_tables.Reg.decode t.reg (read t)

  let write t ?(value = 0) fields =
    t.wreg t.reg.Amd_tables.Reg.addr
      (value lor Amd_tables.Reg.encode t.reg fields)

  let update t fields =
    let mask = Amd_tables.Reg.fields_mask t.reg (List.map fst fields) in
    write t ~value:(read t land lnot mask) fields
end

(* Firmware: amdev.py AMFirmware, with helpers.py fetch_fw folded in
   because the driver-less tier is its only consumer. The reference
   downloads a missing or mismatched file from the pinned
   linux-firmware tree; that fallback is deferred, so failures name
   the pinned source instead. *)

module Firmware = struct
  type desc = int list * bytes

  type t = {
    sos_fw : (int * bytes) list;
    ucode_start : (string * int) list;
    smu_psp_desc : desc option;
    descs : desc list;
  }

  let hex digest =
    let n = Bytes.length digest in
    let out = Bytes.create (n * 2) in
    for i = 0 to n - 1 do
      let v = Bytes.get_uint8 digest i in
      Bytes.set out (2 * i) "0123456789abcdef".[v lsr 4];
      Bytes.set out ((2 * i) + 1) "0123456789abcdef".[v land 0xf]
    done;
    Bytes.unsafe_to_string out

  let zstd_dc path =
    let tmp = Filename.temp_file "tolk_fw" ".bin" in
    Fun.protect
      ~finally:(fun () -> try Sys.remove tmp with Sys_error _ -> ())
      (fun () ->
        let cmd =
          Printf.sprintf "zstd -q -d -c %s > %s" (Filename.quote path)
            (Filename.quote tmp)
        in
        let status = Sys.command cmd in
        if status <> 0 then
          failwith
            (Printf.sprintf "zstd -d -c %s failed with exit code %d" path
               status);
        Bytes.of_string (In_channel.with_open_bin tmp In_channel.input_all))

  (* helpers.py fetch_fw *)
  let fetch_fw ?dir name ~sha256 =
    let dir =
      match dir with
      | Some d -> d
      | None -> Helpers.getenv_str "AMD_FW_PATH" "/lib/firmware/amdgpu"
    in
    let plain = Filename.concat dir name in
    let zst = plain ^ ".zst" in
    let blob =
      if Sys.file_exists plain then
        Bytes.of_string (In_channel.with_open_bin plain In_channel.input_all)
      else if Sys.file_exists zst then zstd_dc zst
      else
        failwith
          (Printf.sprintf
             "firmware %s not found: searched %s and %s; fetch it from %s/%s"
             name plain zst Fw.upstream name)
    in
    let actual = hex (Helpers.sha256 blob) in
    if not (String.equal actual sha256) then
      failwith
        (Printf.sprintf
           "fetch sha mismatch, expected %s but got %s for %s (pinned source \
            %s/%s)"
           sha256 actual plain Fw.upstream name);
    blob

  (* amdev.py:110 load_fw *)
  let load_fw ?dir fname =
    match List.assoc_opt fname Fw.hashes with
    | None ->
        failwith (Printf.sprintf "firmware %s has no pinned sha256" fname)
    | Some sha256 ->
        let blob = fetch_fw ?dir fname ~sha256 in
        if Helpers.getenv "AM_DEBUG" 0 >= 1 then
          Printf.printf "am: loading firmware %s: %s\n%!" fname sha256;
        blob

  (* amdev.py:118 desc *)
  let desc blob ~off ~size types = (types, Bytes.sub blob off size)

  (* amdev.py:25 AMFirmware.__init__, over the ip versions parsed from
     discovery so construction needs no device. Where the reference
     resolves a versioned header struct or a composed GFX_FW_TYPE name,
     an image outside the known set fails loudly here too. *)
  let create ?load ip_ver =
    let load = match load with Some f -> f | None -> fun n -> load_fw n in
    let ver hwip =
      match List.assoc_opt hwip ip_ver with
      | Some v -> v
      | None -> invalid_arg (Printf.sprintf "no discovered ip 0x%x" hwip)
    in
    let fmt_ver (ma, mi, rv) = Printf.sprintf "%d_%d_%d" ma mi rv in
    let header_version blob =
      ( Am.Common_firmware_header.header_version_major blob 0,
        Am.Common_firmware_header.header_version_minor blob 0 )
    in
    let gc_ver = ver Am.gc_hwip in

    (* Load SOS firmware. amdev.py:29 *)
    let sos_blob =
      load (Printf.sprintf "psp_%s_sos.bin" (fmt_ver (ver Am.mp0_hwip)))
    in
    let bin_count, bin_off =
      match header_version sos_blob with
      | 2, 0 ->
          ( Am.Psp_firmware_header_v2_0.psp_fw_bin_count sos_blob 0,
            Am.Psp_firmware_header_v2_0.psp_fw_bin_offset )
      | 2, 1 ->
          ( Am.Psp_firmware_header_v2_1.psp_fw_bin_count sos_blob 0,
            Am.Psp_firmware_header_v2_1.psp_fw_bin_offset )
      | ma, mi ->
          failwith
            (Printf.sprintf "unhandled psp firmware header v%d_%d" ma mi)
    in
    let sos_ucode_off =
      Am.Common_firmware_header.ucode_array_offset_bytes sos_blob 0
    in
    let sos_fw =
      List.init bin_count (fun i ->
          let d = bin_off + (i * Am.Psp_fw_bin_desc.sizeof) in
          let off = Am.Psp_fw_bin_desc.offset_bytes sos_blob d + sos_ucode_off in
          ( Am.Psp_fw_bin_desc.fw_type sos_blob d,
            Bytes.sub sos_blob off (Am.Psp_fw_bin_desc.size_bytes sos_blob d) ))
    in

    (* SMU firmware. amdev.py:44 *)
    let smu_psp_desc, p2s_descs =
      if ver Am.mp1_hwip = (13, 0, 12) then (None, [])
      else
        let blob =
          load (Printf.sprintf "smu_%s.bin" (fmt_ver (ver Am.mp1_hwip)))
        in
        if gc_ver >= (11, 0, 0) then
          ( Some
              (desc blob
                 ~off:(Am.Common_firmware_header.ucode_array_offset_bytes blob 0)
                 ~size:(Am.Common_firmware_header.ucode_size_bytes blob 0)
                 [ Am.gfx_fw_type_smu ]),
            [] )
        else
          match header_version blob with
          | 2, 1 ->
              let entry_off =
                Am.Smc_firmware_header_v2_1.pptable_entry_offset blob 0
              in
              let p2s =
                List.filter_map
                  (fun i ->
                    let e =
                      entry_off + (i * Am.Smc_soft_pptable_entry.sizeof)
                    in
                    (* amdev.py:52 __P2S_TABLE_ID_X *)
                    if Am.Smc_soft_pptable_entry.id blob e = 0x50325358 then
                      Some
                        (desc blob
                           ~off:
                             (Am.Smc_soft_pptable_entry.ppt_offset_bytes blob e)
                           ~size:
                             (Am.Smc_soft_pptable_entry.ppt_size_bytes blob e)
                           [ Am.gfx_fw_type_p2s_table ])
                    else None)
                  (List.init
                     (Am.Smc_firmware_header_v2_1.pptable_count blob 0)
                     Fun.id)
              in
              (None, p2s)
          | ma, mi ->
              failwith
                (Printf.sprintf "unhandled smc firmware header v%d_%d" ma mi)
    in

    (* SDMA firmware. amdev.py:55 *)
    let sdma_descs =
      let blob =
        load (Printf.sprintf "sdma_%s.bin" (fmt_ver (ver Am.sdma0_hwip)))
      in
      let ucode_off =
        Am.Common_firmware_header.ucode_array_offset_bytes blob 0
      in
      match header_version blob with
      | 1, 0 ->
          [
            desc blob ~off:ucode_off
              ~size:(Am.Common_firmware_header.ucode_size_bytes blob 0)
              [
                Am.gfx_fw_type_sdma0; Am.gfx_fw_type_sdma1;
                Am.gfx_fw_type_sdma2; Am.gfx_fw_type_sdma3;
              ];
          ]
      | 2, 0 ->
          [
            desc blob
              ~off:(Am.Sdma_firmware_header_v2_0.ctl_ucode_offset blob 0)
              ~size:(Am.Sdma_firmware_header_v2_0.ctl_ucode_size_bytes blob 0)
              [ Am.gfx_fw_type_sdma_ucode_th1 ];
            desc blob ~off:ucode_off
              ~size:(Am.Sdma_firmware_header_v2_0.ctx_ucode_size_bytes blob 0)
              [ Am.gfx_fw_type_sdma_ucode_th0 ];
          ]
      | 3, 0 ->
          [
            desc blob ~off:ucode_off
              ~size:(Am.Sdma_firmware_header_v3_0.ucode_size_bytes blob 0)
              [ Am.gfx_fw_type_sdma_ucode_th0 ];
          ]
      | ma, mi ->
          failwith
            (Printf.sprintf "unhandled sdma firmware header v%d_%d" ma mi)
    in

    (* PFP, ME, MEC firmware. amdev.py:65 *)
    let cp_fw_type = function
      | "PFP" -> Am.gfx_fw_type_cp_pfp
      | "ME" -> Am.gfx_fw_type_cp_me
      | "MEC" -> Am.gfx_fw_type_cp_mec
      | n -> invalid_arg (Printf.sprintf "no cp firmware type for %s" n)
    in
    let cp_me1_fw_type = function
      | "MEC" -> Am.gfx_fw_type_cp_mec_me1
      | n -> invalid_arg (Printf.sprintf "no cp jump-table firmware type for %s" n)
    in
    let rs64_fw_type = function
      | "PFP" -> Am.gfx_fw_type_rs64_pfp
      | "ME" -> Am.gfx_fw_type_rs64_me
      | "MEC" -> Am.gfx_fw_type_rs64_mec
      | n -> invalid_arg (Printf.sprintf "no rs64 firmware type for %s" n)
    in
    let rs64_stack_fw_type name i =
      match (name, i) with
      | "PFP", 0 -> Am.gfx_fw_type_rs64_pfp_p0_stack
      | "ME", 0 -> Am.gfx_fw_type_rs64_me_p0_stack
      | "MEC", 0 -> Am.gfx_fw_type_rs64_mec_p0_stack
      | n, i ->
          invalid_arg
            (Printf.sprintf "no rs64 stack firmware type for %s p%d" n i)
    in
    let gfx_descs, ucode_start =
      List.fold_left
        (fun (descs, starts) (fw_name, fw_cnt) ->
          let blob =
            load
              (Printf.sprintf "gc_%s_%s.bin" (fmt_ver gc_ver)
                 (String.lowercase_ascii fw_name))
          in
          let ucode_off =
            Am.Common_firmware_header.ucode_array_offset_bytes blob 0
          in
          match header_version blob with
          | 1, 0 ->
              let jt_offset = Am.Gfx_firmware_header_v1_0.jt_offset blob 0 in
              let jt_size = Am.Gfx_firmware_header_v1_0.jt_size blob 0 in
              ( descs
                @ [
                    (* Code *)
                    desc blob ~off:ucode_off
                      ~size:
                        (Am.Common_firmware_header.ucode_size_bytes blob 0
                        - (jt_size * 4))
                      [ cp_fw_type fw_name ];
                    (* JT *)
                    desc blob
                      ~off:(ucode_off + (jt_offset * 4))
                      ~size:(jt_size * 4)
                      [ cp_me1_fw_type fw_name ];
                  ],
                starts )
          | 2, 0 ->
              ( descs
                @ [
                    (* Code *)
                    desc blob ~off:ucode_off
                      ~size:(Am.Gfx_firmware_header_v2_0.ucode_size_bytes blob 0)
                      [ rs64_fw_type fw_name ];
                    (* Stack *)
                    desc blob
                      ~off:(Am.Gfx_firmware_header_v2_0.data_offset_bytes blob 0)
                      ~size:(Am.Gfx_firmware_header_v2_0.data_size_bytes blob 0)
                      (List.init fw_cnt (rs64_stack_fw_type fw_name));
                  ],
                starts
                @ [
                    ( fw_name,
                      Am.Gfx_firmware_header_v2_0.ucode_start_addr_lo blob 0
                      lor Am.Gfx_firmware_header_v2_0.ucode_start_addr_hi blob 0
                          lsl 32 );
                  ] )
          | ma, mi ->
              failwith
                (Printf.sprintf "unhandled gfx firmware header v%d_%d" ma mi))
        ([], [])
        ((if gc_ver >= (12, 0, 0) then [ ("PFP", 1); ("ME", 1) ] else [])
        @ [ ("MEC", 1) ])
    in

    (* IMU firmware. amdev.py:83 *)
    let imu_descs =
      if gc_ver >= (11, 0, 0) then (
        let blob = load (Printf.sprintf "gc_%s_imu.bin" (fmt_ver gc_ver)) in
        let imu_i_off =
          Am.Common_firmware_header.ucode_array_offset_bytes blob 0
        in
        let imu_i_sz =
          Am.Imu_firmware_header_v1_0.imu_iram_ucode_size_bytes blob 0
        in
        let imu_d_sz =
          Am.Imu_firmware_header_v1_0.imu_dram_ucode_size_bytes blob 0
        in
        [
          desc blob ~off:imu_i_off ~size:imu_i_sz [ Am.gfx_fw_type_imu_i ];
          desc blob ~off:(imu_i_off + imu_i_sz) ~size:imu_d_sz
            [ Am.gfx_fw_type_imu_d ];
        ])
      else []
    in

    (* RLC firmware. amdev.py:89 *)
    let rlc_descs =
      let blob = load (Printf.sprintf "gc_%s_rlc.bin" (fmt_ver gc_ver)) in
      let minor = Am.Common_firmware_header.header_version_minor blob 0 in
      (if minor = 1 then
         [
           desc blob
             ~off:
               (Am.Rlc_firmware_header_v2_1.save_restore_list_cntl_offset_bytes
                  blob 0)
             ~size:
               (Am.Rlc_firmware_header_v2_1.save_restore_list_cntl_size_bytes
                  blob 0)
             [ Am.gfx_fw_type_rlc_restore_list_srm_cntl ];
           desc blob
             ~off:
               (Am.Rlc_firmware_header_v2_1.save_restore_list_gpm_offset_bytes
                  blob 0)
             ~size:
               (Am.Rlc_firmware_header_v2_1.save_restore_list_gpm_size_bytes
                  blob 0)
             [ Am.gfx_fw_type_rlc_restore_list_gpm_mem ];
           desc blob
             ~off:
               (Am.Rlc_firmware_header_v2_1.save_restore_list_srm_offset_bytes
                  blob 0)
             ~size:
               (Am.Rlc_firmware_header_v2_1.save_restore_list_srm_size_bytes
                  blob 0)
             [ Am.gfx_fw_type_rlc_restore_list_srm_mem ];
         ]
       else [])
      @ (if minor >= 2 then
           [
             desc blob
               ~off:
                 (Am.Rlc_firmware_header_v2_2.rlc_iram_ucode_offset_bytes blob 0)
               ~size:
                 (Am.Rlc_firmware_header_v2_2.rlc_iram_ucode_size_bytes blob 0)
               [ Am.gfx_fw_type_rlc_iram ];
             desc blob
               ~off:
                 (Am.Rlc_firmware_header_v2_2.rlc_dram_ucode_offset_bytes blob 0)
               ~size:
                 (Am.Rlc_firmware_header_v2_2.rlc_dram_ucode_size_bytes blob 0)
               [ Am.gfx_fw_type_rlc_dram_boot ];
           ]
         else [])
      @ (if minor = 3 then
           [
             desc blob
               ~off:(Am.Rlc_firmware_header_v2_3.rlcp_ucode_offset_bytes blob 0)
               ~size:(Am.Rlc_firmware_header_v2_3.rlcp_ucode_size_bytes blob 0)
               [ Am.gfx_fw_type_rlc_p ];
             desc blob
               ~off:(Am.Rlc_firmware_header_v2_3.rlcv_ucode_offset_bytes blob 0)
               ~size:(Am.Rlc_firmware_header_v2_3.rlcv_ucode_size_bytes blob 0)
               [ Am.gfx_fw_type_rlc_v ];
           ]
         else [])
      @ [
          desc blob
            ~off:(Am.Common_firmware_header.ucode_array_offset_bytes blob 0)
            ~size:(Am.Common_firmware_header.ucode_size_bytes blob 0)
            [ Am.gfx_fw_type_rlc_g ];
        ]
    in
    {
      sos_fw;
      ucode_start;
      smu_psp_desc;
      descs = p2s_descs @ sdma_descs @ gfx_descs @ imu_descs @ rlc_descs;
    }
end

(* Am_page_table: amdev.py AMPageTableEntry; the entry encoding is
   ip.py AM_GMC.get_pte_flags/is_pte_huge_page, folded in here because
   the page tables are its only consumer. *)

module Am_page_table = struct
  type t = { view : Mmio.t; paddr : int; lv : int }

  let mtype_uc gc_ver =
    match gc_ver with
    | 9, _, _ -> Amd_soc_defs.Soc_9.mtype_uc
    | 11, _, _ -> Amd_soc_defs.Soc_11.mtype_uc
    | 12, _, _ -> Amd_soc_defs.Soc_12.mtype_uc
    | maj, _, _ ->
        invalid_arg (Printf.sprintf "no memory-type table for gfx%d" maj)

  let ( ||| ) = Int64.logor

  let pte_flags ~gc_ver ~lv ~table ~frag ~uncached ~system ~snooped ~valid =
    let flags =
      Am.amdgpu_pte_frag frag
      ||| (if system then Am.amdgpu_pte_system else 0L)
      ||| (if snooped then Am.amdgpu_pte_snooped else 0L)
      ||| if valid then Am.amdgpu_pte_valid else 0L
    in
    let flags =
      if table then flags
      else
        flags ||| Am.amdgpu_pte_writeable ||| Am.amdgpu_pte_readable
        ||| Am.amdgpu_pte_executable
    in
    let mtype = if uncached then mtype_uc gc_ver else 0 in
    if gc_ver >= (12, 0, 0) then
      flags
      ||| Am.amdgpu_pte_mtype_gfx12 0L mtype
      |||
      if (not table) && lv <> Am.amdgpu_vm_ptb then Am.amdgpu_pde_pte_gfx12
      else if not table then Am.amdgpu_pte_is_pte
      else 0L
    else if gc_ver >= (10, 0, 0) then
      flags
      ||| Am.amdgpu_pte_mtype_nv10 0L mtype
      |||
      if (not table) && lv <> Am.amdgpu_vm_ptb then Am.amdgpu_pde_pte else 0L
    else
      let flags = flags ||| Am.amdgpu_pte_mtype_vg10 0L mtype in
      let flags =
        if table && lv = Am.amdgpu_vm_pdb1 then flags ||| Am.amdgpu_pde_bfs 0x9
        else flags
      in
      let flags =
        if table && lv = Am.amdgpu_vm_pdb0 then flags ||| Am.amdgpu_pte_tf
        else flags
      in
      if (not table) && lv <> Am.amdgpu_vm_ptb && lv <> Am.amdgpu_vm_pdb0 then
        flags ||| Am.amdgpu_pde_pte
      else flags

  let is_pte_huge_page ~gc_ver ~lv pte =
    if gc_ver < (10, 0, 0) then
      if lv <> Am.amdgpu_vm_pdb0 then
        Int64.logand pte Am.amdgpu_pde_pte <> 0L
      else Int64.logand pte Am.amdgpu_pte_tf = 0L
    else
      Int64.logand pte
        (if gc_ver >= (12, 0, 0) then Am.amdgpu_pde_pte_gfx12
         else Am.amdgpu_pde_pte)
      <> 0L

  (* mi3xx has 48-bit, others have 44-bit address space *)
  let address_space_mask gc_ver =
    (1 lsl (match gc_ver with 9, (4 | 5), _ -> 48 | _ -> 44)) - 1

  let ops ~vram ~gc_ver ?(paddr_base = fun () -> 0) () =
    let mask = address_space_mask gc_ver in
    let entry pt idx = Mmio.read64 pt.view (idx * 8) in
    {
      Memory.make =
        (fun ~paddr ~lv ->
          { view = Mmio.view vram ~off:paddr ~size:0x1000 (); paddr; lv });
      set_entry =
        (fun pt ~idx ~paddr ?(table = false) ?(uncached = false)
             ?(aspace = Memory.Phys) ?(snooped = false) ?(frag = 0) ~valid () ->
          let system = aspace = Memory.Sys in
          let paddr =
            match aspace with
            | Memory.Phys -> paddr_base () + paddr
            | Memory.Sys | Memory.Peer -> paddr
          in
          if paddr land mask <> paddr then
            invalid_arg
              (Printf.sprintf "Invalid physical address 0x%x" paddr);
          let flags =
            pte_flags ~gc_ver ~lv:pt.lv ~table ~frag ~uncached ~system ~snooped
              ~valid
          in
          Mmio.write64 pt.view (idx * 8)
            (Int64.logor flags
               (Int64.logand (Int64.of_int paddr) 0x0000FFFFFFFFF000L)));
      entry;
      valid =
        (fun pt idx -> Int64.logand (entry pt idx) Am.amdgpu_pte_valid <> 0L);
      address =
        (fun pt idx ->
          let e = entry pt idx in
          if Int64.logand e Am.amdgpu_pte_system <> 0L then
            invalid_arg "should not be system address";
          Int64.to_int (Int64.logand e 0x0000FFFFFFFFF000L) - paddr_base ());
      is_page =
        (fun pt idx ->
          pt.lv = Am.amdgpu_vm_ptb
          || is_pte_huge_page ~gc_ver ~lv:pt.lv (entry pt idx));
      supports_huge_page = (fun pt ~paddr:_ -> pt.lv >= Am.amdgpu_vm_pdb2);
      paddr = (fun pt -> pt.paddr);
      lv = (fun pt -> pt.lv);
    }
end

(* IP discovery: amdev.py _run_discovery *)

type gc_info =
  | Gc_info_v1 of {
      num_se : int;
      num_wgp0_per_sa : int;
      num_wgp1_per_sa : int;
      num_sa_per_se : int;
      max_scratch_slots_per_cu : int;
      max_waves_per_simd : int;
      lds_size : int;
    }
  | Gc_info_v2 of {
      num_se : int;
      num_cu_per_sh : int;
      num_sh_per_se : int;
      max_scratch_slots_per_cu : int;
      max_waves_per_simd : int;
      lds_size : int;
    }

type discovery = {
  ip_ver : (int * (int * int * int)) list;
  regs_offset : (int * (int * int array) list) list;
  gc_info : gc_info;
}

let parse_discovery blob =
  if Am.Binary_header.binary_signature blob 0 <> Am.binary_signature then
    failwith "discovery signatures mismatch";
  let table_offset idx =
    Am.Table_info.offset blob
      (Am.Binary_header.table_list_offset + (idx * Am.Table_info.sizeof))
  in
  let ihdr = table_offset Am.table_ip_discovery in
  if Am.Ip_discovery_header.signature blob ihdr <> Am.discovery_table_signature
  then failwith "discovery signatures mismatch";
  let base64 = Am.Ip_discovery_header.base_addr_64_bit blob ihdr <> 0 in
  let ip_ver_tbl = Hashtbl.create 16 in
  let bases_tbl = Hashtbl.create 16 in
  for die = 0 to Am.Ip_discovery_header.num_dies blob ihdr - 1 do
    let die_off =
      Am.Die_info.die_offset blob
        (ihdr + Am.Ip_discovery_header.die_info_offset
        + (die * Am.Die_info.sizeof))
    in
    let ip_off = ref (die_off + Am.Die_header.sizeof) in
    for _ = 1 to Am.Die_header.num_ips blob die_off do
      let hw_id = Am.Ip_v4.hw_id blob !ip_off in
      let instance = Am.Ip_v4.instance_number blob !ip_off in
      let n = Am.Ip_v4.num_base_address blob !ip_off in
      let version =
        ( Am.Ip_v4.major blob !ip_off,
          Am.Ip_v4.minor blob !ip_off,
          Am.Ip_v4.revision blob !ip_off )
      in
      (* Base addresses sit at +8 even though the entry header is 7
         bytes; the entry stride is 8 + the base-address array. *)
      let bases =
        Array.init n (fun i ->
            if base64 then
              Int64.to_int (Bytes.get_int64_le blob (!ip_off + 8 + (i * 8)))
            else Am.g32 blob (!ip_off + 8 + (i * 4)))
      in
      List.iter
        (fun (hw_ip, mapped_id) ->
          if mapped_id = hw_id then begin
            Hashtbl.replace ip_ver_tbl hw_ip version;
            Hashtbl.replace bases_tbl (hw_ip, instance) bases
          end)
        Am.hw_id_map;
      ip_off := !ip_off + 8 + ((if base64 then 8 else 4) * n)
    done
  done;
  let ip_ver =
    Hashtbl.fold (fun k v acc -> (k, v) :: acc) ip_ver_tbl []
    |> List.sort (fun (a, _) (b, _) -> compare a b)
  in
  let regs_offset =
    List.map
      (fun (hw_ip, _) ->
        let insts =
          Hashtbl.fold
            (fun (h, i) b acc -> if h = hw_ip then (i, b) :: acc else acc)
            bases_tbl []
          |> List.sort (fun (a, _) (b, _) -> compare a b)
        in
        (hw_ip, insts))
      ip_ver
  in
  let gc = table_offset Am.table_gc in
  let gc_info =
    (* The minor versions of each major share these field offsets, so
       the v1_0 and v2_0 accessors read every published revision. *)
    match Am.Gpu_info_header.version_major blob gc with
    | 1 ->
        Gc_info_v1
          {
            num_se = Am.Gc_info_v1_0.gc_num_se blob gc;
            num_wgp0_per_sa = Am.Gc_info_v1_0.gc_num_wgp0_per_sa blob gc;
            num_wgp1_per_sa = Am.Gc_info_v1_0.gc_num_wgp1_per_sa blob gc;
            num_sa_per_se = Am.Gc_info_v1_0.gc_num_sa_per_se blob gc;
            max_scratch_slots_per_cu =
              Am.Gc_info_v1_0.gc_max_scratch_slots_per_cu blob gc;
            max_waves_per_simd = Am.Gc_info_v1_0.gc_max_waves_per_simd blob gc;
            lds_size = Am.Gc_info_v1_0.gc_lds_size blob gc;
          }
    | 2 ->
        Gc_info_v2
          {
            num_se = Am.Gc_info_v2_0.gc_num_se blob gc;
            num_cu_per_sh = Am.Gc_info_v2_0.gc_num_cu_per_sh blob gc;
            num_sh_per_se = Am.Gc_info_v2_0.gc_num_sh_per_se blob gc;
            max_scratch_slots_per_cu =
              Am.Gc_info_v2_0.gc_max_scratch_slots_per_cu blob gc;
            max_waves_per_simd = Am.Gc_info_v2_0.gc_max_waves_per_simd blob gc;
            lds_size = Am.Gc_info_v2_0.gc_lds_size blob gc;
          }
    | v -> failwith (Printf.sprintf "unsupported gc info version %d" v)
  in
  { ip_ver; regs_offset; gc_info }

(* Devices: amdev.py AMDev (without the boot state machine) *)

external monotonic_ms : unit -> int = "caml_tolk_hcq_monotonic_ms" [@@noalloc]

type t = {
  pci_dev : System.Pci_device.t option;
  devfmt : string;
  vram : Mmio.t;
  doorbell64 : Mmio.t;
  mmio : Mmio.t;
  vram_size : int;
  large_bar : bool;
  reserved_vram_size : int;
  discovery : discovery;
  rreg : int -> int;
  wreg : int -> int -> unit;
  reg : string -> Am_register.t;
  xgmi_seg_sz : int;
  paddr_base : int;
  mc_base : int;
  now_ms : unit -> int;
  is_booting : bool ref;
  mm : Am_page_table.t Memory.t;
}

let pci_dev t = t.pci_dev
let devfmt t = t.devfmt
let vram t = t.vram
let doorbell64 t = t.doorbell64
let mmio t = t.mmio
let vram_size t = t.vram_size
let large_bar t = t.large_bar
let reserved_vram_size t = t.reserved_vram_size
let discovery t = t.discovery
let gc_info t = t.discovery.gc_info
let is_booting t = !(t.is_booting)
let mm t = t.mm
let now_ms t = t.now_ms ()

let ip_ver t hwip =
  match List.assoc_opt hwip t.discovery.ip_ver with
  | Some v -> v
  | None -> invalid_arg (Printf.sprintf "no discovered ip 0x%x" hwip)

(* amdev.py:241 is_hive, amdev.py:243-245 paddr conversions *)
let is_hive t = t.xgmi_seg_sz > 0
let paddr2mc t paddr = t.mc_base + paddr
let paddr2xgmi t paddr = t.paddr_base + paddr
let xgmi2paddr t xgmi_paddr = xgmi_paddr - t.paddr_base

let rreg t r = t.rreg r
let wreg t r v = t.wreg r v
let reg t name = t.reg name

let wreg_pair t base ~lo ~hi v =
  Am_register.write (reg t (base ^ lo)) ~value:(v land 0xffffffff) [];
  Am_register.write (reg t (base ^ hi)) ~value:(v lsr 32) []

let indirect_wreg_pcie t ?(aid = 0) r v =
  let reg_addr =
    (r * 4) + if aid > 0 then ((aid land 0b11) lsl 32) lor (1 lsl 34) else 0
  in
  Am_register.write (reg t "regBIF_BX0_PCIE_INDEX2")
    ~value:(reg_addr land 0xffffffff) [];
  if reg_addr lsr 32 > 0 then
    Am_register.write (reg t "regBIF_BX0_PCIE_INDEX2_HI")
      ~value:((reg_addr lsr 32) land 0xff) [];
  Am_register.write (reg t "regBIF_BX0_PCIE_DATA2") ~value:v [];
  if reg_addr lsr 32 > 0 then
    Am_register.write (reg t "regBIF_BX0_PCIE_INDEX2_HI") ~value:0 []

(* Register access before the device record exists (discovery time),
   over the raw register BAR. *)
let raw_rreg mmio r = Int32.to_int (Mmio.read32 mmio (r * 4)) land 0xffffffff
let raw_wreg mmio r v = Mmio.write32 mmio (r * 4) (Int32.of_int v)

(* Reads vram through the mmMM_INDEX/mmMM_DATA window, for ranges the
   VRAM BAR does not reach. amdev.py:279 *)
let read_vram mmio ~addr ~size =
  if addr mod 4 <> 0 || size mod 4 <> 0 then
    invalid_arg (Printf.sprintf "Invalid address 0x%x or size 0x%x" addr size);
  let out = Bytes.create size in
  for i = 0 to (size / 4) - 1 do
    let caddr = addr + (i * 4) in
    raw_wreg mmio 0x06 (caddr lsr 31);
    raw_wreg mmio 0x00 ((caddr land 0x7FFFFFFF) lor 0x80000000);
    Bytes.set_int32_le out (i * 4) (Int32.of_int (raw_rreg mmio 0x01))
  done;
  out

(* amdev.py _build_regs *)
let build_ips discovery =
  let ip_version hwip =
    match List.assoc_opt hwip discovery.ip_ver with
    | Some v -> v
    | None -> failwith (Printf.sprintf "ip 0x%x missing from discovery" hwip)
  in
  let inst0_bases hwip =
    match List.assoc_opt hwip discovery.regs_offset with
    | Some insts -> List.assoc_opt 0 insts
    | None -> None
  in
  let gc_ver = ip_version Am.gc_hwip in
  let mods =
    [
      ("mp", Am.mp0_hwip);
      ("hdp", Am.hdp_hwip);
      ("gc", Am.gc_hwip);
      ("mmhub", Am.mmhub_hwip);
      ("osssys", Am.osssys_hwip);
      ((if gc_ver < (12, 0, 0) then "nbio" else "nbif"), Am.nbio_hwip);
    ]
    @
    if List.mem (ip_version Am.sdma0_hwip) [ (4, 4, 2); (4, 4, 4) ] then
      [ ("sdma", Am.sdma0_hwip) ]
    else []
  in
  let create_ip name hwip version =
    match inst0_bases hwip with
    | Some bases -> [ Amd_tables.Ip.create ~name ~version ~bases ]
    | None -> []
  in
  let ips =
    List.concat_map
      (fun (name, hwip) -> create_ip name hwip (ip_version hwip))
      mods
    @ create_ip "mp" Am.mp1_hwip (11, 0, 0)
  in
  List.rev ips

(* Fixed register to query memory size without known ip bases to find
   the discovery table; the table is located at the end of VRAM - 64KB
   and is 10KB in size. amdev.py:288 *)
let mm_rcc_config_memsize = 0xde3

(* One virtual address space shared by every device. *)
let va_base = 0x200000000000
let va_allocator = lazy (Tlsf.create ~size:(1 lsl 44) ~base:va_base ())

(* The register-access closures stored in [t]: named lookup with its
   cache, and dword access either over the register BAR (with the
   indirect index/data window beyond it, amdev.py:249-258) or over
   injected functions. *)
let reg_access ~ips access =
  let regs = Hashtbl.create 64 in
  (* Exact names only: Ip.reg's reg->mm fallback must not fire, or a
     name absent from a later family could shadow the exact definition
     in an earlier one. [ips] holds the most recently resolved family
     first, so lookups prefer later families and a name defined twice
     resolves as if the tables had been merged in resolution order. *)
  let find name =
    let rec loop = function
      | [] -> invalid_arg (Printf.sprintf "device has no register %s" name)
      | ip :: rest -> (
          match Amd_tables.Ip.reg ip name with
          | r when String.equal r.Amd_tables.Reg.name name -> r
          | _ -> loop rest
          | exception Invalid_argument _ -> loop rest)
    in
    loop ips
  in
  let rec rreg r =
    match access with
    | `Bar mmio ->
        if r >= Mmio.size mmio / 4 then indirect_rreg r else raw_rreg mmio r
    | `Fns (rreg, _) -> rreg r
  and wreg r v =
    match access with
    | `Bar mmio ->
        if r >= Mmio.size mmio / 4 then indirect_wreg r v
        else raw_wreg mmio r v
    | `Fns (_, wreg) -> wreg r v
  and indirect_rreg r =
    Am_register.write (reg "regBIF_BX_PF0_RSMU_INDEX") ~value:(r * 4) [];
    Am_register.read (reg "regBIF_BX_PF0_RSMU_DATA")
  and indirect_wreg r v =
    Am_register.write (reg "regBIF_BX_PF0_RSMU_INDEX") ~value:(r * 4) [];
    Am_register.write (reg "regBIF_BX_PF0_RSMU_DATA") ~value:v []
  and reg name =
    match Hashtbl.find_opt regs name with
    | Some r -> r
    | None ->
        let r = Am_register.make ~reg:(find name) ~rreg ~wreg in
        Hashtbl.add regs name r;
        r
  in
  (rreg, wreg, reg)

(* ip.py:54-64 AM_GMC.init_sw: the XGMI topology and framebuffer base
   behind the paddr conversions. Register-derived constants of the die,
   read once at device creation; devices without the XGMI registers
   read as a single-device topology. *)
let gmc_state reg =
  let bitfield name field =
    match reg name with
    | r -> List.assoc field (Am_register.read_bitfields r)
    | exception Invalid_argument _ -> 0
  in
  let xgmi_phys_id = bitfield "regMMMC_VM_XGMI_LFB_CNTL" "pf_lfb_region" in
  let xgmi_seg_sz = bitfield "regMMMC_VM_XGMI_LFB_SIZE" "pf_lfb_size" lsl 24 in
  let paddr_base = xgmi_phys_id * xgmi_seg_sz in
  let fb_base =
    (Am_register.read (reg "regMMMC_VM_FB_LOCATION_BASE") land 0xFFFFFF)
    lsl 24
  in
  (xgmi_seg_sz, paddr_base, fb_base + paddr_base)

let make ?pci_dev ?(now_ms = monotonic_ms) ?(is_booting = ref true) ~rreg ~wreg
    ~vram ~doorbell64 ~mmio ~vram_size ~large_bar ~reserved_vram_size
    ~discovery ~mm ~devfmt () =
  let rreg, wreg, reg =
    reg_access ~ips:(build_ips discovery) (`Fns (rreg, wreg))
  in
  let xgmi_seg_sz, paddr_base, mc_base = gmc_state reg in
  {
    pci_dev;
    devfmt;
    vram;
    doorbell64;
    mmio;
    vram_size;
    large_bar;
    reserved_vram_size;
    discovery;
    rreg;
    wreg;
    reg;
    xgmi_seg_sz;
    paddr_base;
    mc_base;
    now_ms;
    is_booting;
    mm;
  }

let create pci_dev =
  let vram = System.Pci_device.map_bar pci_dev 0 in
  let doorbell64 = System.Pci_device.map_bar pci_dev 2 in
  let mmio = System.Pci_device.map_bar pci_dev 5 in
  let vram_size = raw_rreg mmio mm_rcc_config_memsize lsl 20 in
  let large_bar = Mmio.size vram >= vram_size in
  let tmr_offset = vram_size - (64 lsl 10) in
  let tmr_size = 10 lsl 10 in
  let disc_tbl =
    if large_bar then Mmio.read_bytes vram ~off:tmr_offset ~len:tmr_size
    else read_vram mmio ~addr:tmr_offset ~size:tmr_size
  in
  let discovery = parse_discovery disc_tbl in
  let gc_ver =
    match List.assoc_opt Am.gc_hwip discovery.ip_ver with
    | Some v -> v
    | None -> failwith "ip discovery lists no graphics core"
  in
  let reserved_vram_size =
    match gc_ver with 9, (4 | 5), _ -> 384 lsl 20 | _ -> 64 lsl 20
  in
  let rreg, wreg, reg = reg_access ~ips:(build_ips discovery) (`Bar mmio) in
  let xgmi_seg_sz, paddr_base, mc_base = gmc_state reg in
  let is_booting = ref true in
  let devfmt = System.Pci_device.pcibus pci_dev in
  let lv_span = 9 * (3 - Am.amdgpu_vm_pdb2) in
  let mm =
    Memory.create
      ~pt_ops:
        (Am_page_table.ops ~vram ~gc_ver
           ~paddr_base:(fun () -> paddr_base)
           ())
      ~vram_size:(vram_size - reserved_vram_size)
      ~boot_size:(32 lsl 20) ~va_bits:48
      ~va_shifts:[ 12; 21; 30; 39 ]
      ~va_base
      ~palloc_ranges:
        (List.init (lv_span + 1) (fun k ->
             let i = lv_span - k in
             (1 lsl (i + 12), if i >= 9 then 2 lsl 20 else 0x1000)))
      ~va_allocator:(Lazy.force va_allocator)
      ~is_booting:(fun () -> !is_booting)
      ~zero_vram:(fun ~paddr ~size ->
        Mmio.blit_bytes vram ~off:paddr (Bytes.make size '\000'))
      ~first_lv:Am.amdgpu_vm_pdb2 ~reserve_ptable:(not large_bar)
      ~dbg_name:devfmt ()
  in
  {
    pci_dev = Some pci_dev;
    devfmt;
    vram;
    doorbell64;
    mmio;
    vram_size;
    large_bar;
    reserved_vram_size;
    discovery;
    rreg;
    wreg;
    reg;
    xgmi_seg_sz;
    paddr_base;
    mc_base;
    now_ms = monotonic_ms;
    is_booting;
    mm;
  }
