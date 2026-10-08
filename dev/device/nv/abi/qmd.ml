(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Packet
open Repr
module D = Defs

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* A descriptor is the bytes [make] writes and what each setter did since,
   newest first: setting a field, or leaving one as a hole for a term. Setters
   add an operation and copy nothing; [structure] applies them in one copy of
   the bytes. *)
type 'v op = Set of D.field * int | Hole of 'v hole

type 'v t = {
  layout : D.qmd;
  banks : int list; (* the indices of the launch's banks *)
  base : string;
  ops : 'v op list;
}

type axis = X | Y | Z
type dim = Grid of axis | Block of axis

(* Fields *)

let bank (f : D.banked) i = { f.first with lo = f.first.lo + (i * f.stride) }

(* Writes [v] into the [bits] bits of [b] from bit [lo], a byte at a time, from
   its bit [i]. *)
let rec write_from b ~lo ~bits v i =
  if i < bits then begin
    let bit = lo + i in
    let at = bit / 8 and shift = bit mod 8 in
    let n = Int.min (8 - shift) (bits - i) in
    let m = ((1 lsl n) - 1) lsl shift in
    let byte = Char.code (Bytes.get b at) in
    let byte = byte land lnot m lor (((v lsr i) lsl shift) land m) in
    Bytes.set b at (Char.chr byte);
    write_from b ~lo ~bits v (i + n)
  end

let write fn b ~lo ~bits v =
  if v < 0 || v lsr bits <> 0 then
    invalid_argf "%s: %d does not fit the %d-bit field at bit %d" fn v bits lo;
  write_from b ~lo ~bits v 0

let read_bits s (f : D.field) =
  let v = ref 0 in
  for i = f.bits - 1 downto 0 do
    let bit = f.lo + i in
    v := (!v lsl 1) lor ((Char.code s.[bit / 8] lsr (bit mod 8)) land 1)
  done;
  !v

(* The value of the field [f]: the latest [Set] of it, else the base's. *)
let rec latest s (f : D.field) = function
  | [] -> read_bits s f
  | Set (g, v) :: _ when g.lo = f.lo -> v
  | _ :: ops -> latest s f ops

let read q f = latest q.base f q.ops
let set q f v = { q with ops = Set (f, v) :: q.ops }

(* [q] with the field [f], which starts a byte, filled by the term [t]. A field
   is known, in the bytes, or a hole, zero in the bytes: the later of a [Set]
   and a [Hole] of a field wins. *)
let hole q (f : D.field) t =
  { q with ops = Hole { at = f.lo / 8; bits = f.bits; value = t } :: q.ops }

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
  let b = Bytes.make q.bytes '\000' in
  let set (f : D.field) v = write "Qmd.make" b ~lo:f.lo ~bits:f.bits v in
  let prefetch = q.program_prefetch_size in
  set q.qmd_major_version q.version;
  set q.register_count k.registers;
  set q.shared_memory_size (l.shared_bytes lsr q.shared_memory_shift);
  set q.qmd_group_id group_id;
  set q.invalidate_texture_header_cache 1;
  set q.invalidate_texture_sampler_cache 1;
  set q.invalidate_texture_data_cache 1;
  set q.invalidate_shader_data_cache 1;
  set q.api_visible_call_limit D.api_visible_call_limit_no_check;
  set q.sampler_index D.sampler_index_via_header_index;
  set q.barrier_count barriers;
  set q.cwd_membar_type D.cwd_membar_type_l1_sysmembar;
  set (bank q.constant_buffer_invalidate 0) 1;
  set q.min_sm_config_shared_mem_size l.shared_config;
  set q.target_sm_config_shared_mem_size l.shared_config;
  set q.max_sm_config_shared_mem_size l.max_shared_config;
  set prefetch
    (Int.min (k.code_bytes lsr prefetch_unit) ((1 lsl prefetch.bits) - 1));
  set q.sass_version l.gpu.sass_version;
  Option.iter (fun f -> set f D.qmd_type_grid_cta) q.qmd_type;
  Option.iter (fun f -> set f 1) q.sm_global_caching_enable;
  let banks = Launch.banks launch in
  let bank_fields (c : Cubin.bank) =
    set (bank q.constant_buffer_size_shifted4 c.index) ((c.bytes + 15) lsr 4);
    set (bank q.constant_buffer_valid c.index) 1
  in
  List.iter bank_fields banks;
  {
    layout = q;
    banks = List.map (fun (c : Cubin.bank) -> c.index) banks;
    base = Bytes.unsafe_to_string b;
    ops = [];
  }

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
  set q (field q d) n

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

let add_release s addr v q ~size =
  let r0, r1 = q.layout.releases in
  match
    List.find_opt (fun (r : D.release) -> read q r.enable = 0) [ r0; r1 ]
  with
  | None -> None
  | Some r ->
      let sets =
        Set (r.membar_type, membar s)
        :: Set (r.payload64b, 1)
        :: Set (r.structure_size, size)
        :: Set (r.enable, 1)
        :: q.ops
      in
      let q = { q with ops = sets } in
      let q = address q r.address_lower r.address_upper (Value addr) in
      Some (address q r.payload_lower r.payload_upper (Value v))

let release s addr v q =
  add_release s addr v q ~size:D.release_structure_size_semaphore_two_words

let release_stamp s addr v q =
  add_release s addr v q ~size:D.release_structure_size_semaphore_four_words

(* The descriptor's address, in 256-byte units. *)
let chain_unit = 8

let chain addr q =
  let p = q.layout in
  let sets =
    Set (p.dependent_qmd0_enable, 1)
    :: Set (p.dependent_qmd0_prefetch, 1)
    :: Set (p.dependent_qmd0_action, D.dependent_qmd0_action_qmd_schedule)
    :: q.ops
  in
  let q = { q with ops = sets } in
  hole q p.dependent_qmd0_pointer (shifted addr chain_unit)

(* Layout *)

let rec without at = function
  | [] -> []
  | h :: hs -> if h.at = at then hs else h :: without at hs

let rec has_hole at = function
  | [] -> false
  | h :: hs -> h.at = at || has_hole at hs

(* Zeroes the field of the hole [h], which starts a byte. *)
let clear b h =
  let whole = h.bits / 8 and rest = h.bits mod 8 in
  Bytes.fill b h.at whole '\000';
  if rest > 0 then
    let at = h.at + whole in
    Bytes.set b at
      (Char.chr (Char.code (Bytes.get b at) land lnot ((1 lsl rest) - 1)))

(* Applies [ops], newest first in the list, oldest first to [b], and is the
   holes they leave, at most one at each offset. *)
let rec apply b = function
  | [] -> []
  | Set (f, v) :: older ->
      let holes = apply b older in
      write "Qmd.structure" b ~lo:f.lo ~bits:f.bits v;
      let at = f.lo / 8 in
      if f.lo mod 8 = 0 && has_hole at holes then without at holes else holes
  | Hole h :: older ->
      let holes = apply b older in
      clear b h;
      h :: (if has_hole h.at holes then without h.at holes else holes)

let structure q =
  let b = Bytes.of_string q.base in
  let holes = apply b q.ops in
  let holes = List.sort (fun a b -> Int.compare a.at b.at) holes in
  { Repr.bytes = Bytes.unsafe_to_string b; holes }
