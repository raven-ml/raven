(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Template
module D = Defs

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

type axis = X | Y | Z
type dim = Grid of axis | Block of axis
type release = Release of operand * operand | Stamped of operand * operand

type t = {
  program : operand;
  local : operand;
  banks : (int * operand) list;
  grid : operand * operand * operand;
  block : operand * operand * operand;
  shared : operand option;
  chain : operand option;
  releases : release list;
}

(* CUDA's limits for compute capabilities 8.0 to 12.0 (CUDA C++ Programming
   Guide, "Technical Specifications per Compute Capability"). *)
let max_size = function
  | Grid X -> (1 lsl 31) - 1
  | Grid (Y | Z) -> 65535
  | Block (X | Y) -> 1024
  | Block Z -> 64

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

let write b (f : D.field) v =
  if v < 0 || v lsr f.bits <> 0 then
    invalid_argf "Qmd: %d does not fit the %d-bit field at bit %d" v f.bits f.lo;
  write_from b ~lo:f.lo ~bits:f.bits v 0

(* A descriptor's bytes and patches while it is written. *)
type writing = { b : Bytes.t; mutable patches : patch list }

(* Sets the field [f], which starts a byte if [v] is a slot, to [(v + add) lsr
   shift]: a known value in the bytes, a slot as a patch, its field zero in the
   bytes. *)
let unsigned v =
  if v < 0 then invalid_argf "Qmd: %d, expected an unsigned integer" v

let field w (f : D.field) ?(add = 0) ?(shift = 0) = function
  | Int v ->
      unsigned v;
      write w.b f ((v + add) lsr shift)
  | Slot slot ->
      assert (f.lo mod 8 = 0);
      let add = Int64.of_int add in
      let p = { at = f.lo / 8; bits = f.bits; slot; add; shift } in
      w.patches <- p :: w.patches

(* [ps] by increasing offset. A sort allocates even an empty list's copy. *)
let sorted = function
  | [] -> []
  | ps -> List.sort (fun (a : patch) b -> Int.compare a.at b.at) ps

(* An address shifted right by [s], in the 32-bit field [lower] and the field
   [upper] above it. *)
let address w (lower : D.field) (upper : D.field) s = function
  | Int v ->
      unsigned v;
      write w.b lower ((v lsr s) land 0xffff_ffff);
      write w.b upper (v lsr (s + 32))
  | Slot _ as v ->
      field w lower ~shift:s v;
      field w upper ~shift:(s + 32) v

(* A descriptor *)

(* Values no NVIDIA header or NVK prescribes for a CUDA launch, which kimchi's
   runs rely on: the group id, and one barrier a block may use. *)
let group_id = 0x3f
let barriers = 1

(* The code prefetched before the launch, in 256-byte units, and a chained
   descriptor's address. *)
let prefetch_unit = 8
let chain_unit = 8

let make (launch : Launch.t) (d : t) =
  let l = (launch :> Repr.launch) in
  let q = l.layout and k = l.kernel in
  let w = { b = Bytes.make q.bytes '\000'; patches = [] } in
  let set f v = write w.b f v in
  let prefetch = q.program_prefetch_size in
  let banks = Launch.banks launch in
  set q.qmd_major_version q.version;
  set q.register_count k.registers;
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
  set q.max_sm_config_shared_mem_size l.max_shared_config;
  set prefetch
    (Int.min (k.code_bytes lsr prefetch_unit) ((1 lsl prefetch.bits) - 1));
  set q.sass_version l.gpu.sass_version;
  Option.iter (fun f -> set f D.qmd_type_grid_cta) q.qmd_type;
  Option.iter (fun f -> set f 1) q.sm_global_caching_enable;
  List.iter
    (fun (c : Cubin.bank) ->
      set (bank q.constant_buffer_size_shifted4 c.index) ((c.bytes + 15) lsr 4);
      set (bank q.constant_buffer_valid c.index) 1)
    banks;
  (* Sizes. *)
  let gx, gy, gz = d.grid and bx, by, bz = d.block in
  field w q.grid_width gx;
  field w q.grid_height gy;
  field w q.grid_depth gz;
  field w q.cta_thread_dimension0 bx;
  field w q.cta_thread_dimension1 by;
  field w q.cta_thread_dimension2 bz;
  (* The block's own shared memory, the launch's, plus the dynamic: the
     configuration of the most shared memory holds any dynamic size
     {!Launch.dynamic_shared} allows. *)
  let shift = q.shared_memory_shift in
  (match d.shared with
  | None ->
      set q.shared_memory_size (l.shared_bytes lsr shift);
      set q.min_sm_config_shared_mem_size l.shared_config;
      set q.target_sm_config_shared_mem_size l.shared_config
  | Some v ->
      field w q.shared_memory_size ~add:l.shared_bytes ~shift v;
      set q.min_sm_config_shared_mem_size l.max_shared_config;
      set q.target_sm_config_shared_mem_size l.max_shared_config);
  (* Addresses. *)
  address w q.program_address_lower q.program_address_upper
    q.program_address_shift d.program;
  address w q.program_prefetch_addr_lower_shifted
    q.program_prefetch_addr_upper_shifted prefetch_unit d.program;
  field w q.shader_local_memory_high_size ~shift:q.shader_local_memory_shift
    d.local;
  let indices = List.map (fun (c : Cubin.bank) -> c.index) banks in
  List.iter
    (fun (i, v) ->
      (* [memq]: ints are equal when physically equal. *)
      if not (List.memq i indices) then
        invalid_argf "Qmd.make: bank %d, expected one of %s" i
          (String.concat ", " (List.map string_of_int indices));
      address w
        (bank q.constant_buffer_addr_lower i)
        (bank q.constant_buffer_addr_upper i)
        q.constant_buffer_addr_shift v)
    d.banks;
  Option.iter
    (fun v ->
      set q.dependent_qmd0_enable 1;
      set q.dependent_qmd0_prefetch 1;
      set q.dependent_qmd0_action D.dependent_qmd0_action_qmd_schedule;
      field w q.dependent_qmd0_pointer ~shift:chain_unit v)
    d.chain;
  (* Every launch ends with a membar to the system (CWD_MEMBAR_TYPE, set above),
     so a release needs none of its own. *)
  let r0, r1 = q.releases in
  if List.length d.releases > 2 then
    invalid_argf "Qmd.make: %d releases, expected at most 2"
      (List.length d.releases);
  List.iteri
    (fun i release ->
      let (r : D.release) = if i = 0 then r0 else r1 in
      let addr, v, size =
        match release with
        | Release (a, v) -> (a, v, D.release_structure_size_semaphore_two_words)
        | Stamped (a, v) -> (a, v, D.release_structure_size_semaphore_four_words)
      in
      set r.enable 1;
      set r.structure_size size;
      set r.payload64b 1;
      set r.membar_type D.release_membar_type_fe_none;
      address w r.address_lower r.address_upper 0 addr;
      address w r.payload_lower r.payload_upper 0 v)
    d.releases;
  { bytes = Bytes.unsafe_to_string w.b; patches = sorted w.patches }

(* The driver's parameters *)

(* The driver's parameters at the start of bank 0, as 32-bit words: their count,
   and the first words of a block's sizes and of a grid's (x, y, z), of the
   shared and local memory windows (64 bits each) and of the stack limit. The
   CUDA driver's layout, which no NVIDIA header states: the sizes are where NVCC
   13.4's code reads blockDim and gridDim, as nvdisasm shows it for sm_89 and
   sm_120, and kimchi's runs rely on the rest. *)
type parameters = {
  words : int;
  block : int;
  grid : int;
  shared : int;
  local : int;
  stack : int;
  address : int;
}

let before_blackwell =
  {
    words = 18;
    block = 0;
    grid = 3;
    shared = 6;
    local = 8;
    stack = 10;
    address = 16;
  }

let blackwell =
  {
    words = 224;
    block = 216;
    grid = 220;
    shared = 188;
    local = 190;
    stack = 223;
    address = 210;
  }

let stack_limit = 0xfffdc0

(* The 32-bit word [i]. *)
let word i = { D.lo = 32 * i; bits = 32 }

let parameters (launch : Launch.t) (d : t) =
  let l = (launch :> Repr.launch) in
  (* CUDA's grid-constant parameters have generic addresses: NVCC 13.4's sm_89
     code reads bank 0's base at 0x40, sm_120 the kernel's parameters at 0x348.
     Each launch points into its own bank, chained launches too. *)
  let p, offset =
    if l.gpu.compute_class = D.blackwell_compute_b then
      (blackwell, l.kernel.params_offset)
    else (before_blackwell, 0)
  in
  let n = Int.max l.kernel.params_offset (4 * p.words) in
  let w = { b = Bytes.make n '\000'; patches = [] } in
  Bytes.set_int64_le w.b (4 * p.shared) (Int64.of_int l.gpu.shared_window);
  Bytes.set_int64_le w.b (4 * p.local) (Int64.of_int l.gpu.local_window);
  Bytes.set_int32_le w.b (4 * p.stack) (Int32.of_int stack_limit);
  let sizes at (x, y, z) =
    field w (word at) x;
    field w (word (at + 1)) y;
    field w (word (at + 2)) z
  in
  sizes p.block d.block;
  sizes p.grid d.grid;
  let at = 4 * p.address in
  (match List.assoc_opt 0 d.banks with
  | None -> ()
  | Some (Int a) ->
      unsigned a;
      Bytes.set_int64_le w.b at (Int64.of_int (a + offset))
  | Some (Slot slot) ->
      let add = Int64.of_int offset in
      w.patches <- { at; bits = 64; slot; add; shift = 0 } :: w.patches);
  { bytes = Bytes.unsafe_to_string w.b; patches = sorted w.patches }
