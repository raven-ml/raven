(* Typed access to the generated AMD hardware tables. *)

module Am_defs = Amd_am_defs
module Smu_defs = Amd_smu_defs
module Fw_defs = Amd_fw_defs

module Reg = struct
  type t = {
    name : string;
    offset : int;
    segment : int;
    fields : (string * (int * int)) array;
    addr : int;
  }

  let field t name =
    let n = Array.length t.fields in
    let rec go i =
      if i = n then
        invalid_arg (Printf.sprintf "register %s has no field %s" t.name name)
      else
        let fname, range = t.fields.(i) in
        if String.equal fname name then range else go (i + 1)
    in
    go 0

  let encode t values =
    List.fold_left
      (fun acc (name, value) ->
        let lo, _ = field t name in
        acc lor (value lsl lo))
      0 values

  let decode t v =
    Array.to_list t.fields
    |> List.map (fun (name, (lo, hi)) ->
           (name, (v lsr lo) land ((1 lsl (hi - lo + 1)) - 1)))

  let fields_mask t names =
    List.fold_left
      (fun acc name ->
        let lo, hi = field t name in
        acc lor (((1 lsl (hi - lo + 1)) - 1) lsl lo))
      0 names
end

module Ip = struct
  type t = {
    name : string;
    version : int * int * int;
    regs : (string, Reg.t) Hashtbl.t;
  }

  let major (m, _, _) = m

  let create ~name ~version ~bases =
    let resolved =
      List.fold_left
        (fun best (fam_name, fam_version, fam_regs) ->
          if
            String.equal fam_name name
            && major fam_version = major version
            && fam_version <= version
            &&
            match best with
            | Some (best_version, _) -> fam_version > best_version
            | None -> true
          then Some (fam_version, fam_regs)
          else best)
        None Amd_regs_defs.families
    in
    match resolved with
    | None ->
        let v0, v1, v2 = version in
        invalid_arg
          (Printf.sprintf "no register family for %s %d.%d.%d" name v0 v1 v2)
    | Some (_, fam_regs) ->
        let regs = Hashtbl.create (List.length fam_regs) in
        List.iter
          (fun (reg_name, (offset, segment, fields)) ->
            Hashtbl.replace regs reg_name
              {
                Reg.name = reg_name;
                offset;
                segment;
                fields = Array.of_list fields;
                addr = bases.(segment) + offset;
              })
          fam_regs;
        { name; version; regs }

  let name t = t.name
  let version t = t.version

  let reg t reg_name =
    match Hashtbl.find_opt t.regs reg_name with
    | Some r -> r
    | None -> (
        let fallback =
          if String.starts_with ~prefix:"reg" reg_name then
            Hashtbl.find_opt t.regs
              ("mm" ^ String.sub reg_name 3 (String.length reg_name - 3))
          else None
        in
        match fallback with
        | Some r -> r
        | None ->
            invalid_arg
              (Printf.sprintf "%s has no register %s"
                 (String.uppercase_ascii t.name)
                 reg_name))
end

module type Pm4 = sig
  val packet3 : int -> int -> int
  val packet3_nop : int
  val packet3_set_sh_reg : int
  val packet3_set_sh_reg_start : int
  val packet3_set_sh_reg_end : int
  val packet3_set_uconfig_reg : int
  val packet3_set_uconfig_reg_start : int
  val packet3_acquire_mem : int
  val packet3_release_mem : int
  val packet3_wait_reg_mem : int
  val packet3_dispatch_direct : int
  val packet3_event_write : int
  val packet3_indirect_buffer : int
  val indirect_buffer_valid : int
  val packet3_pred_exec : int
  val cache_flush_and_inv_ts_event : int
  val event_index__mec_release_mem__end_of_pipe : int
  val data_sel__mec_release_mem__send_32_bit_low : int
  val data_sel__mec_release_mem__send_64_bit_data : int
  val data_sel__mec_release_mem__send_gpu_clock_counter : int
  val int_sel__mec_release_mem__none : int
  val int_sel__mec_release_mem__send_interrupt_after_write_confirm : int
  val wait_reg_mem_function : int -> int
  val wait_reg_mem_mem_space : int -> int
  val wait_reg_mem_operation : int -> int
  val wait_reg_mem_engine : int -> int
  val event_type : int -> int
  val event_index : int -> int
end

let pm4 ~gfx9 : (module Pm4) =
  if gfx9 then (module Amd_pm4_defs.Soc15) else (module Amd_pm4_defs.Nv)

module type Sdma = sig
  val sdma_op_copy : int
  val sdma_op_write : int
  val sdma_op_indirect : int
  val sdma_op_fence : int
  val sdma_op_trap : int
  val sdma_op_poll_regmem : int
  val sdma_op_timestamp : int
  val sdma_subop_copy_linear : int
  val sdma_subop_timestamp_get_global : int
  val sdma_pkt_copy_linear_header_sub_op : int -> int
  val sdma_pkt_copy_linear_count_count : int -> int
  val sdma_pkt_indirect_header_vmid : int -> int
  val sdma_pkt_poll_regmem_header_func : int -> int
  val sdma_pkt_poll_regmem_header_mem_poll : int -> int
  val sdma_pkt_poll_regmem_dw5_interval : int -> int
  val sdma_pkt_poll_regmem_dw5_retry_count : int -> int
  val sdma_pkt_timestamp_get_header_sub_op : int -> int
  val sdma_pkt_trap_int_context_int_context : int -> int
end

module type Soc = sig
  val cs_partial_flush : int
end

let soc ~target_major : (module Soc) =
  match target_major with
  | 9 -> (module Amd_soc_defs.Soc_9)
  | 11 -> (module Amd_soc_defs.Soc_11)
  | 12 -> (module Amd_soc_defs.Soc_12)
  | _ ->
      invalid_arg (Printf.sprintf "no soc event table for gfx%d" target_major)

let sdma ~version : (module Sdma) =
  match Ip.major (min version (6, 0, 0)) with
  | 4 -> (module Amd_sdma_defs.V4_0_0)
  | 6 -> (module Amd_sdma_defs.V6_0_0)
  | _ ->
      let v0, v1, v2 = version in
      invalid_arg
        (Printf.sprintf "no sdma packet format for version %d.%d.%d" v0 v1 v2)

module type Smu = sig
  val ppsmc_msg_setdriverdramaddrhigh : int
  val ppsmc_msg_setdriverdramaddrlow : int
  val ppsmc_msg_enableallsmufeatures : int
  val ppsmc_msg_getsmuversion : int
  val ppsmc_msg_mode1reset : int option
  val ppsmc_msg_gfxdriverreset : int option
  val ppsmc_msg_transfertablesmu2dram : int option
  val ppsmc_msg_getmetricstable : int option
  val ppsmc_msg_getdpmfreqbyindex : int
  val ppsmc_msg_setsoftminbyfreq : int
  val ppsmc_msg_setsoftmaxbyfreq : int
  val ppsmc_msg_setpptlimit : int
  val ppsmc_msg_queryvalidmcacount : int option
  val ppsmc_msg_queryvalidmcacecount : int option
  val ppsmc_msg_mcabankdumpdw : int option
  val ppsmc_msg_mcabankcedumpdw : int option
  val ppclk_uclk : int
  val ppclk_fclk : int
  val ppclk_socclk : int
  val ppclk_gfxclk : int option
end

let smu ~version : (module Smu) =
  (* 13.0.7 firmware speaks the 13.0.0 message interface, not 13.0.6's. *)
  let version = if version = (13, 0, 7) then (13, 0, 0) else version in
  let resolved =
    List.fold_left
      (fun best (v, m) ->
        if
          Ip.major v = Ip.major version
          && v <= version
          && match best with Some (bv, _) -> v > bv | None -> true
        then Some (v, m)
        else best)
      None
      [
        ((13, 0, 0), (module Amd_smu_defs.V13_0_0 : Smu));
        ((13, 0, 6), (module Amd_smu_defs.V13_0_6));
        ((13, 0, 12), (module Amd_smu_defs.V13_0_12));
        ((14, 0, 2), (module Amd_smu_defs.V14_0_2));
      ]
  in
  match resolved with
  | Some (_, m) -> m
  | None ->
      let v0, v1, v2 = version in
      invalid_arg
        (Printf.sprintf "no smu message table for version %d.%d.%d" v0 v1 v2)

let tmpring_field name (shift, width) value =
  if value < 0 || value lsr width <> 0 then
    invalid_arg
      (Printf.sprintf "tmpring_size: %s = %d does not fit in %d bits" name
         value width);
  value lsl shift

let tmpring_size ~target_major ~waves ~wavesize =
  let waves_spec, wavesize_spec =
    let open Amd_hsa_defs.Compute_tmpring_size in
    match target_major with
    | 9 -> (gfx9_waves, gfx9_wavesize)
    | 11 -> (gfx11_waves, gfx11_wavesize)
    | 12 -> (gfx12_waves, gfx12_wavesize)
    | _ ->
        invalid_arg
          (Printf.sprintf "tmpring_size: unsupported target gfx%d" target_major)
  in
  tmpring_field "waves" waves_spec waves
  lor tmpring_field "wavesize" wavesize_spec wavesize
