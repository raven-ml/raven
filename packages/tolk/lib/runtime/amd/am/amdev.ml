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
    rreg : direct:bool -> int -> int;
    wreg : direct:bool -> int -> int -> unit;
  }

  let make ~reg ~rreg ~wreg = { reg; rreg; wreg }
  let reg t = t.reg
  let read ?(direct = false) t = t.rreg ~direct t.reg.Amd_tables.Reg.addr
  let read_bitfields ?direct t = Amd_tables.Reg.decode t.reg (read ?direct t)

  let write t ?(direct = false) ?(value = 0) fields =
    t.wreg ~direct t.reg.Amd_tables.Reg.addr
      (value lor Amd_tables.Reg.encode t.reg fields)

  let update t ?direct fields =
    let mask = Amd_tables.Reg.fields_mask t.reg (List.map fst fields) in
    write t ?direct ~value:(read ?direct t land lnot mask) fields
end

(* Firmware: amdev.py AMFirmware and helpers.py fetch_fw. *)

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

  let command_output program args =
    let input = Unix.open_process_args_in program args in
    match In_channel.input_all input with
    | exception exn ->
        let backtrace = Printexc.get_raw_backtrace () in
        (try ignore (Unix.close_process_in input) with _ -> ());
        Printexc.raise_with_backtrace exn backtrace
    | output ->
        (match Unix.close_process_in input with
         | Unix.WEXITED 0 -> Bytes.of_string output
         | Unix.WEXITED code ->
             failwith (Printf.sprintf "%s failed with exit code %d" program code)
         | Unix.WSIGNALED signal | Unix.WSTOPPED signal ->
             failwith (Printf.sprintf "%s failed with signal %d" program signal))

  let fetch_fw ?dir name ~sha256 =
    let dir = Option.value dir
        ~default:(Helpers.getenv_str "AMD_FW_PATH" "/lib/firmware/amdgpu") in
    let plain = Filename.concat dir name in
    let matches blob = String.equal (hex (Helpers.sha256 blob)) sha256 in
    let local path read =
      if Sys.file_exists path then
        Option.bind (read path) (fun blob -> if matches blob then Some blob else None)
      else None in
    let local_blob = match local plain (fun path ->
        Some (Bytes.of_string (In_channel.with_open_bin path In_channel.input_all))) with
      | Some blob -> Some blob
      | None -> local (plain ^ ".zst") (fun path ->
          try Some (command_output "zstd" [|"zstd"; "-q"; "-d"; "-c"; path|])
          with Unix.Unix_error (Unix.ENOENT, _, _) -> None) in
    match local_blob with
    | Some blob -> blob
    | None ->
        let url = Fw.upstream ^ "/" ^ name in
        let key = url ^ "\n" ^ sha256 in
        match Tolk.Diskcache.get ~table:"firmware" ~key with
        | Some blob when matches blob -> blob
        | _ ->
            let blob = command_output "curl"
                [|"curl"; "--fail"; "--location"; "--silent"; "--show-error";
                  "--connect-timeout"; "10"; "--max-time"; "60"; url|] in
            let actual = hex (Helpers.sha256 blob) in
            if not (String.equal actual sha256) then
              failwith (Printf.sprintf
                "fetch sha mismatch, expected %s but got %s for %s" sha256 actual url);
            Tolk.Diskcache.put ~table:"firmware" ~key blob;
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
          ( Am.Psp_firmware_header_v2_1.psp_aux_fw_bin_index sos_blob 0,
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

  let paddr pt = pt.paddr
  let lv pt = pt.lv

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
  harvested : (int * int list) list;
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
  let harvested =
    let offset = table_offset Am.table_harvest in
    if offset = 0 then []
    else if offset > Bytes.length blob - 4 then failwith "truncated harvest table"
    else if Am.g32 blob offset <> Am.harvest_table_signature then []
    else begin
      if offset > Bytes.length blob - (8 + 32 * 4) then failwith "truncated harvest table";
      let entries = List.init 32 (fun i ->
          let at = offset + 8 + i * 4 in
          Am.g16 blob at, Am.g8 blob (at + 2)) in
      List.filter_map (fun (hwip, hwid) ->
          let instances = List.filter_map (fun (id, inst) ->
              if id = hwid then Some inst else None) entries |> List.sort_uniq Int.compare in
          if instances = [] then None else Some (hwip, instances)) Am.hw_id_map
      |> List.sort compare
    end in
  { ip_ver; regs_offset; harvested; gc_info }

(* Devices: amdev.py AMDev (without the boot state machine) *)

external monotonic_ms : unit -> int = "caml_tolk_hcq_monotonic_ms" [@@noalloc]

let system_sleep_ms ms = Unix.sleepf (float_of_int ms /. 1000.)

exception Timeout_error of string

let wait_on ~now_ms ?(timeout_ms = 10000) ~value ~msg cb =
  let start = now_ms () in
  let rec go last =
    if now_ms () - start < timeout_ms then begin
      let value_now = cb () in
      if value_now <> value then go value_now
    end else raise (Timeout_error (Printf.sprintf
      "%s. Timed out after %d ms, condition not met: %d != %d"
      msg timeout_ms last value))
  in
  go 0

let vf_mailbox_request ~rreg ~wreg ~rreg8 ~wreg8 ~now_ms
    ?(wait_ready = true) request =
  let control = Am.nv_maibox_control_trn_offset_byte in
  wreg8 control 0;
  wait_on ~now_ms ~timeout_ms:1000 ~value:0
    ~msg:"VF mailbox acknowledgement did not clear" (fun () -> rreg8 control land 2);
  List.iteri (fun index value -> wreg (Am.mmmailbox_msgbuf_trn_dw0 + index) value)
    [request; 0; 0; 0];
  wreg8 control 1;
  wait_on ~now_ms ~timeout_ms:Am.nv_mailbox_poll_ack_timedout ~value:2
    ~msg:(Printf.sprintf "VF mailbox request 0x%x was not acked" request)
    (fun () -> rreg8 control land 2);
  wreg8 control 0;
  if wait_ready then begin
    wait_on ~now_ms ~timeout_ms:Am.nv_mailbox_poll_msg_timedout
      ~value:Am.idh_ready_to_access_gpu ~msg:"VF mailbox: the PF never granted access"
      (fun () -> rreg Am.mmmailbox_msgbuf_rcv_dw0);
    wreg8 (control + 1) 2
  end;
  request + 1

type t = {
  pci_dev : System.Pci_device.t option;
  read_config : offset:int -> size:int -> int;
  devfmt : string;
  vram : Mmio.t;
  doorbell64 : Mmio.t;
  mmio : Mmio.t;
  vram_size : int;
  large_bar : bool;
  reserved_vram_size : int;
  discovery : discovery;
  rreg : ?inst:int -> ?direct:bool -> int -> int;
  wreg : ?inst:int -> ?direct:bool -> int -> int -> unit;
  reg : int -> string -> Am_register.t;
  is_vf : bool;
  vf_access : int ref;
  vf_mailbox_request : ?wait_ready:bool -> int -> int;
  is_hive : bool;
  paddr_base : int;
  mc_base : int;
  now_ms : unit -> int;
  sleep_ms : int -> unit;
  is_booting : bool ref;
  is_err_state : bool ref;
  on_range_mapped : (unit -> unit) ref;
  mm : Am_page_table.t Memory.t;
}

let is_vf t = t.is_vf

let release_vf_access t =
  let lease = !(t.vf_access) in
  t.vf_access := 0;
  if lease <> 0 then
    try ignore (t.vf_mailbox_request ~wait_ready:false lease)
    with Timeout_error _ -> ()

let acquire_fini_access t =
  if t.is_vf && !(t.vf_access) = 0 then
    try t.vf_access := t.vf_mailbox_request Am.idh_req_gpu_fini_access
    with Timeout_error _ -> ()

let wait_cond t ?timeout_ms ~value ~msg cb =
  wait_on ~now_ms:t.now_ms ?timeout_ms ~value ~msg cb

let pci_dev t = t.pci_dev
let read_config t ~offset ~size = t.read_config ~offset ~size
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
let set_is_booting t v = t.is_booting := v
let is_err_state t = !(t.is_err_state)
let set_err_state t v = t.is_err_state := v
let set_on_range_mapped t f = t.on_range_mapped := f
let mm t = t.mm
let now_ms t = t.now_ms ()
let sleep_ms t ms = t.sleep_ms ms

let ip_ver t hwip =
  match List.assoc_opt hwip t.discovery.ip_ver with
  | Some v -> v
  | None -> invalid_arg (Printf.sprintf "no discovered ip 0x%x" hwip)

(* amdev.py:241 is_hive, amdev.py:243-245 paddr conversions *)
let is_hive t = t.is_hive
let paddr2mc t paddr = t.mc_base + paddr
let paddr2xgmi t paddr = t.paddr_base + paddr
let xgmi2paddr t xgmi_paddr = xgmi_paddr - t.paddr_base

let rreg t ?inst ?direct r = t.rreg ?inst ?direct r
let wreg t ?inst ?direct r v = t.wreg ?inst ?direct r v
let reg t ?(inst = 0) name = t.reg inst name

let live_instances t hwip =
  let harvested = Option.value (List.assoc_opt hwip t.discovery.harvested) ~default:[] in
  Option.value (List.assoc_opt hwip t.discovery.regs_offset) ~default:[]
  |> List.filter_map (fun (inst, _) -> if List.mem inst harvested then None else Some inst)

let aids t =
  let live = live_instances t Am.sdma0_hwip in
  let maximum = List.fold_left (fun m inst -> max m (inst lsr 2)) 0 live in
  0 :: (List.init maximum (fun i -> i + 1) |> List.filter (fun aid ->
      let mask = List.fold_left (fun mask inst ->
          if inst lsr 2 = aid then mask lor (1 lsl (inst land 3)) else mask) 0 live in
      List.mem mask [0xf; 0x3; 0xc]))

let wreg_pair t ?(inst = 0) ?direct base ~lo ~hi v =
  Am_register.write (reg t ~inst (base ^ lo)) ?direct ~value:(v land 0xffffffff) [];
  Am_register.write (reg t ~inst (base ^ hi)) ?direct ~value:(v lsr 32) []

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
    match List.assoc_opt hwip discovery.regs_offset with
    | Some ((_, bases) :: _ as instances) ->
        [ Amd_tables.Ip.create ~name ~version ~bases, instances ]
    | Some [] | None -> []
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
let va_size = 1 lsl 44
let va_allocator = lazy (Tlsf.create ~size:va_size ~base:va_base ())

(* The register-access closures stored in [t]: named lookup with its
   cache, and dword access either over the register BAR (with the
   indirect index/data window beyond it, amdev.py:249-258) or over
   injected functions. *)
let reg_access ~ips ~is_vf ~now_ms access =
  let gated = if not is_vf then [] else
      List.concat_map (fun (ip, instances) ->
          if Amd_tables.Ip.name ip <> "gc" then [] else
          List.concat_map (fun (_, bases) ->
              List.map (fun (segment, last) -> bases.(segment), bases.(segment) + last)
                (Amd_tables.Ip.segment_extents ip)) instances) ips in
  let is_gated address = List.exists (fun (lo, hi) -> lo <= address && address <= hi) gated in
  let regs = Hashtbl.create 64 in
  (* Exact names only: Ip.reg's reg->mm fallback must not fire, or a
     name absent from a later family could shadow the exact definition
     in an earlier one. [ips] holds the most recently resolved family
     first, so lookups prefer later families and a name defined twice
     resolves as if the tables had been merged in resolution order. *)
  let find inst name =
    let rec loop = function
      | [] -> invalid_arg (Printf.sprintf "device has no register %s" name)
      | (ip, instances) :: rest -> (
          match Amd_tables.Ip.reg ip name with
          | r when String.equal r.Amd_tables.Reg.name name ->
              let bases = match List.assoc_opt inst instances with
                | Some bases -> bases
                | None -> invalid_arg (Printf.sprintf "register %s has no instance %d" name inst) in
              {r with addr = bases.(r.segment) + r.offset}
          | _ -> loop rest
          | exception Invalid_argument _ -> loop rest)
    in
    loop ips
  in
  let rec rreg ?(inst = 0) ?(direct = false) r =
    if not direct && is_gated r then rlcg_rw ~inst ~read:true r 0 else
    match access with
    | `Bar mmio ->
        if r >= Mmio.size mmio / 4 then indirect_rreg r else raw_rreg mmio r
    | `Fns (rreg, _) -> rreg r
  and wreg ?(inst = 0) ?(direct = false) r v =
    if not direct && is_gated r then ignore (rlcg_rw ~inst ~read:false r v) else
    match access with
    | `Bar mmio ->
        if r >= Mmio.size mmio / 4 then indirect_wreg r v
        else raw_wreg mmio r v
    | `Fns (_, wreg) -> wreg r v
  and rlcg_rw ~inst ~read address value =
    let gfx_cntl = (Am_register.reg (reg inst "regGRBM_GFX_CNTL")).Amd_tables.Reg.addr
    and gfx_index = (Am_register.reg (reg inst "regGRBM_GFX_INDEX")).Amd_tables.Reg.addr in
    if address = gfx_cntl || address = gfx_index then begin
      Am_register.write (reg inst (if address = gfx_cntl then "regSCRATCH_REG2" else "regSCRATCH_REG3"))
        ~direct:true ~value [];
      value
    end else begin
      Am_register.write (reg inst "regSCRATCH_REG0") ~direct:true ~value [];
      Am_register.write (reg inst "regSCRATCH_REG1") ~direct:true
        ~value:(address lor if read then 1 lsl 28 else 0) [];
      Am_register.write (reg inst "regRLC_SPARE_INT") ~direct:true ~value:1 [];
      wait_on ~now_ms ~value:0 ~msg:(Printf.sprintf "RLC gateway timeout on 0x%x" address)
        (fun () -> Am_register.read ~direct:true (reg inst "regSCRATCH_REG1") land 0xfffff);
      let error = Am_register.read ~direct:true (reg inst "regSCRATCH_REG1") land 0xf000000 in
      if Helpers.getenv "AM_DEBUG" 0 >= 1 && error <> 0 then
        Printf.eprintf "RLC gateway refused 0x%x: 0x%x\n%!" address error;
      Am_register.read ~direct:true (reg inst "regSCRATCH_REG0")
    end
  and indirect_rreg r =
    Am_register.write (reg 0 "regBIF_BX_PF0_RSMU_INDEX") ~value:(r * 4) [];
    Am_register.read (reg 0 "regBIF_BX_PF0_RSMU_DATA")
  and indirect_wreg r v =
    Am_register.write (reg 0 "regBIF_BX_PF0_RSMU_INDEX") ~value:(r * 4) [];
    Am_register.write (reg 0 "regBIF_BX_PF0_RSMU_DATA") ~value:v []
  and reg inst name =
    match Hashtbl.find_opt regs (name, inst) with
    | Some r -> r
    | None ->
        let r = Am_register.make ~reg:(find inst name)
            ~rreg:(fun ~direct address -> rreg ~inst ~direct address)
            ~wreg:(fun ~direct address value -> wreg ~inst ~direct address value) in
        Hashtbl.add regs (name, inst) r;
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
  let xgmi_phys_id = bitfield "regGCMC_VM_XGMI_LFB_CNTL" "pf_lfb_region" in
  let xgmi_seg_sz = bitfield "regGCMC_VM_XGMI_LFB_SIZE" "pf_lfb_size" lsl 24 in
  let xgmi_max_region = bitfield "regGCMC_VM_XGMI_LFB_CNTL" "pf_max_region" in
  let is_hive = xgmi_seg_sz > 0 && xgmi_max_region > 0 in
  let paddr_base = xgmi_phys_id * xgmi_seg_sz in
  let fb_base =
    (Am_register.read (reg "regMMMC_VM_FB_LOCATION_BASE") land 0xFFFFFF)
    lsl 24
  in
  (is_hive, paddr_base, fb_base + paddr_base)

let make ?pci_dev ?(now_ms = monotonic_ms) ?(sleep_ms = system_sleep_ms)
    ?(is_booting = ref true)
    ?(on_range_mapped = ref (fun () -> ())) ~read_config ~rreg ~wreg ~rreg8 ~wreg8 ~vram ~doorbell64
    ~mmio ~vram_size ~large_bar ~reserved_vram_size ~discovery ~mm ~devfmt ()
    =
  let is_vf = rreg Am.mmrcc_iov_func_identifier land 1 <> 0 in
  let vf_mailbox_request = vf_mailbox_request ~rreg ~wreg ~rreg8 ~wreg8 ~now_ms in
  let vf_access = ref (if is_vf then vf_mailbox_request Am.idh_req_gpu_init_access else 0) in
  System.with_rollback (fun rollback ->
  rollback (fun () ->
      let lease = !vf_access in
      vf_access := 0;
      if lease <> 0 then
        try ignore (vf_mailbox_request ~wait_ready:false lease)
        with Timeout_error _ -> ());
  let rreg, wreg, reg =
    reg_access ~ips:(build_ips discovery) ~is_vf ~now_ms (`Fns (rreg, wreg))
  in
  let is_hive, paddr_base, mc_base = gmc_state (reg 0) in
  {
    pci_dev;
    read_config;
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
    is_vf;
    vf_access;
    vf_mailbox_request;
    is_hive;
    paddr_base;
    mc_base;
    now_ms;
    sleep_ms;
    is_booting;
    is_err_state = ref false;
    on_range_mapped;
    mm;
  })

let create pci_dev =
  (* These are independent CPU BAR mappings, not GPU backing allocations.
     Firmware initialization begins only after this constructor returns. *)
  System.with_rollback (fun rollback ->
    System.Pci_device.disable_aspm pci_dev;
    let map_bar bar =
      let view = System.Pci_device.map_bar pci_dev bar in
      rollback (fun () ->
          Tolk_hcq.Hcq.File_io.munmap (Mmio.addr view) ~size:(Mmio.size view));
      view
    in
    let vram = map_bar 0 in
    let doorbell64 = map_bar 2 in
    let mmio = map_bar 5 in
    let is_vf = raw_rreg mmio Am.mmrcc_iov_func_identifier land 1 <> 0 in
    let vf_mailbox_request = vf_mailbox_request ~rreg:(raw_rreg mmio)
        ~wreg:(raw_wreg mmio) ~rreg8:(Mmio.read8 mmio) ~wreg8:(Mmio.write8 mmio)
        ~now_ms:monotonic_ms in
    let vf_access = ref (if is_vf then vf_mailbox_request Am.idh_req_gpu_init_access else 0) in
    rollback (fun () ->
        let lease = !vf_access in
        vf_access := 0;
        if lease <> 0 then
          try ignore (vf_mailbox_request ~wait_ready:false lease)
          with Timeout_error _ -> ());
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
    let rreg, wreg, reg = reg_access ~ips:(build_ips discovery)
        ~is_vf ~now_ms:monotonic_ms (`Bar mmio) in
    let is_hive, paddr_base, mc_base = gmc_state (reg 0) in
    let is_booting = ref true in
    let on_range_mapped = ref (fun () -> ()) in
    let devfmt = System.Pci_device.pcibus pci_dev in
    let lv_span = 9 * (3 - Am.amdgpu_vm_pdb2) in
    let mm =
      Memory.create
        ~pt_ops:
          (Am_page_table.ops ~vram ~gc_ver
             ~paddr_base:(fun () -> paddr_base)
             ())
        ~vram_size:(vram_size - reserved_vram_size)
        ~boot_size:(3 lsl 20) ~va_bits:48
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
        ~first_lv:Am.amdgpu_vm_pdb2 ~reserve_ptable:(not large_bar) ~clear_root:false
        ~dbg_name:devfmt
        ~on_range_mapped:(fun () -> !on_range_mapped ())
        ()
    in
    {
      pci_dev = Some pci_dev;
      read_config = System.Pci_device.read_config pci_dev;
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
      is_vf;
      vf_access;
      vf_mailbox_request;
      is_hive;
      paddr_base;
      mc_base;
      now_ms = monotonic_ms;
      sleep_ms = system_sleep_ms;
      is_booting;
      is_err_state = ref false;
      on_range_mapped;
      mm;
    })
