(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Packet
open Repr
module D = Defs

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type 'v t = {
  layout : D.qmd;
  banks : int list; (* the indices of the launch's banks *)
  bytes : string;
  holes : 'v hole list; (* at most one at each offset *)
}

type axis = X | Y | Z
type dim = Grid of axis | Block of axis

(* Fields *)

let bank (f : D.banked) i = { f.first with lo = f.first.lo + (i * f.stride) }

(* Writes [v] into the field [f] of [b], bit by bit. *)
let write fn b (f : D.field) v =
  if v < 0 || v lsr f.bits <> 0 then
    invalid_argf "%s: 0x%x does not fit a field of %d bits" fn v f.bits;
  for i = 0 to f.bits - 1 do
    let at = (f.lo + i) / 8 and m = 1 lsl ((f.lo + i) mod 8) in
    let byte = Char.code (Bytes.get b at) in
    let byte = if (v lsr i) land 1 = 1 then byte lor m else byte land lnot m in
    Bytes.set b at (Char.chr byte)
  done

let read q (f : D.field) =
  let v = ref 0 in
  for i = f.bits - 1 downto 0 do
    let bit = f.lo + i in
    v := (!v lsl 1) lor ((Char.code q.bytes.[bit / 8] lsr (bit mod 8)) land 1)
  done;
  !v

(* [q] with the fields [fs] set, in one copy of its bytes. *)
let writes fn q fs =
  let b = Bytes.of_string q.bytes in
  List.iter (fun (f, v) -> write fn b f v) fs;
  { q with bytes = Bytes.unsafe_to_string b }

(* [q] with the field [f], which starts a byte, filled by the term [t]. *)
let hole q (f : D.field) t =
  let at = f.lo / 8 in
  let others = List.filter (fun h -> h.at <> at) q.holes in
  { q with holes = { at; bits = f.bits; value = t } :: others }

(* An address in the fields [lower] and [upper]: [t]'s low 32 bits, then the
   bits above. *)
let address q lower upper t = hole (hole q lower t) upper (Shift (t, 32))

(* Terms keep the shapes tolk's compiled programs mirror node for node: an
   address carries its field's shift even when it is 0, a size only when its
   field has one. *)
let shifted v n = Shift (Value v, n)
let scaled v n = if n = 0 then Value v else Shift (Value v, n)

(* A descriptor *)

(* Values no NVIDIA header or NVK prescribes for a CUDA launch, which kimchi's
   runs rely on: the group id, and one barrier a block may use. *)
let group_id = 0x3f
let barriers = 1

(* The code prefetched before the launch, in 256-byte units. *)
let prefetch_unit = 8

let make (launch : Launch.t) =
  let l = (launch :> Repr.launch) in
  let q = l.layout and k = l.kernel in
  let own =
    List.filter_map Fun.id
      [
        Option.map (fun f -> (f, D.qmd_type_grid_cta)) q.qmd_type;
        Option.map (fun f -> (f, 1)) q.sm_global_caching_enable;
      ]
  in
  let banks = Launch.banks launch in
  let bank_fields (b : Cubin.bank) =
    [
      (bank q.constant_buffer_size_shifted4 b.index, (b.bytes + 15) lsr 4);
      (bank q.constant_buffer_valid b.index, 1);
    ]
  in
  let prefetch = q.program_prefetch_size in
  let fields =
    [
      (q.qmd_major_version, q.version);
      (q.register_count, k.registers);
      (q.shared_memory_size, l.shared_bytes lsr q.shared_memory_shift);
      (q.qmd_group_id, group_id);
      (q.invalidate_texture_header_cache, 1);
      (q.invalidate_texture_sampler_cache, 1);
      (q.invalidate_texture_data_cache, 1);
      (q.invalidate_shader_data_cache, 1);
      (q.api_visible_call_limit, D.api_visible_call_limit_no_check);
      (q.sampler_index, D.sampler_index_via_header_index);
      (q.barrier_count, barriers);
      (q.cwd_membar_type, D.cwd_membar_type_l1_sysmembar);
      (bank q.constant_buffer_invalidate 0, 1);
      (q.min_sm_config_shared_mem_size, l.shared_config);
      (q.target_sm_config_shared_mem_size, l.shared_config);
      (q.max_sm_config_shared_mem_size, l.max_shared_config);
      ( prefetch,
        Int.min (k.code_bytes lsr prefetch_unit) ((1 lsl prefetch.bits) - 1) );
      (q.sass_version, l.gpu.sass_version);
    ]
    @ own
    @ List.concat_map bank_fields banks
  in
  let empty =
    {
      layout = q;
      banks = List.map (fun (b : Cubin.bank) -> b.index) banks;
      bytes = String.make q.bytes '\000';
      holes = [];
    }
  in
  writes "Qmd.make" empty fields

(* Sizes *)

(* CUDA's limits for compute capabilities 8.0 to 12.0 (CUDA C++ Programming
   Guide, "Technical Specifications per Compute Capability"). *)
let max_size = function
  | Grid X -> (1 lsl 31) - 1
  | Grid (Y | Z) -> 65535
  | Block (X | Y) -> 1024
  | Block Z -> 64

let field q = function
  | Grid X -> q.layout.grid_width
  | Grid Y -> q.layout.grid_height
  | Grid Z -> q.layout.grid_depth
  | Block X -> q.layout.cta_thread_dimension0
  | Block Y -> q.layout.cta_thread_dimension1
  | Block Z -> q.layout.cta_thread_dimension2

let set_dim d n q =
  if n < 0 || n > max_size d then
    invalid_argf "Qmd.set_dim: size %d, expected 0 to %d" n (max_size d);
  writes "Qmd.set_dim" q [ (field q d, n) ]

let patch_dim d v q = hole q (field q d) (Value v)

(* Addresses *)

let set_program addr q =
  let p = q.layout in
  let q =
    address q p.program_address_lower p.program_address_upper
      (shifted addr p.program_address_shift)
  in
  address q p.program_prefetch_addr_lower_shifted
    p.program_prefetch_addr_upper_shifted
    (shifted addr prefetch_unit)

let set_bank i addr q =
  if not (List.mem i q.banks) then
    invalid_argf "Qmd.set_bank: bank %d, expected one of %s" i
      (String.concat ", " (List.map string_of_int q.banks));
  let p = q.layout in
  address q
    (bank p.constant_buffer_addr_lower i)
    (bank p.constant_buffer_addr_upper i)
    (shifted addr p.constant_buffer_addr_shift)

let set_local_memory bytes q =
  let p = q.layout in
  hole q p.shader_local_memory_high_size
    (scaled bytes p.shader_local_memory_shift)

(* Completion *)

(* Every launch ends with a membar to the system (CWD_MEMBAR_TYPE, set by
   [make]), so a release needs none of its own at either scope. *)
let membar = function Agent | System -> D.release_membar_type_fe_none

let add_release fn s addr v q ~size =
  let r0, r1 = q.layout.releases in
  match
    List.find_opt (fun (r : D.release) -> read q r.enable = 0) [ r0; r1 ]
  with
  | None -> None
  | Some r ->
      let q =
        writes fn q
          [
            (r.enable, 1);
            (r.structure_size, size);
            (r.payload64b, 1);
            (r.membar_type, membar s);
          ]
      in
      let q = address q r.address_lower r.address_upper (Value addr) in
      Some (address q r.payload_lower r.payload_upper (Value v))

let release s addr v q =
  add_release "Qmd.release" s addr v q
    ~size:D.release_structure_size_semaphore_two_words

let release_stamp s addr v q =
  add_release "Qmd.release_stamp" s addr v q
    ~size:D.release_structure_size_semaphore_four_words

(* The descriptor's address, in 256-byte units. *)
let chain_unit = 8

let chain addr q =
  let p = q.layout in
  let q =
    writes "Qmd.chain" q
      [
        (p.dependent_qmd0_action, D.dependent_qmd0_action_qmd_schedule);
        (p.dependent_qmd0_prefetch, 1);
        (p.dependent_qmd0_enable, 1);
      ]
  in
  hole q p.dependent_qmd0_pointer (shifted addr chain_unit)

(* Layout *)

let structure q =
  let holes = List.sort (fun a b -> Int.compare a.at b.at) q.holes in
  { Repr.bytes = q.bytes; holes }
