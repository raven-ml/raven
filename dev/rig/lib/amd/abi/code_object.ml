(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module K = Defs.Kernel_descriptor

let strf = Printf.sprintf
let ( let* ) = Result.bind

type axis = X | Y | Z

type hidden =
  | Block_count of axis
  | Group_size of axis
  | Remainder of axis
  | Global_offset of axis
  | Grid_dims
  | Dynamic_lds_size
  | Other of string

type kernel = {
  descriptor : int;
  entry : int;
  group_segment : int;
  private_segment : int;
  kernarg_size : int;
  rsrc1 : int;
  rsrc2 : int;
  rsrc3 : int;
  wave32 : bool;
  dispatch_ptr : bool;
  private_segment_buffer : bool;
  max_threads : int;
  hidden : (hidden * int) list;
}

type t = {
  elf : Rig_elf.t;
  target : string;
  size : int;
  patches : (int * string) list;
  names : string array; (* every kernel's name, in increasing order *)
  records : kernel option array;
      (* each kernel's record, in the order of [names], all [Some]: [kernel]
         returns one without allocating *)
}

(* LLVM's AMDGPU relocations, ELFRelocs/AMDGPU.def. *)
let r_amdgpu_rel64 = 5

(* The bytes of an R_AMDGPU_REL64 field. *)
let rel64_bytes = 8

(* The longest image: what a GPU's 48-bit virtual addresses reach. *)
let max_size = 1 lsl 48

(* The most LDS a kernel takes: what COMPUTE_PGM_RSRC2.LDS_SIZE's 9 bits hold in
   their smallest unit, 512 bytes (AMDGPUUsage, GRANULATED_LDS_SIZE), so that a
   dispatch on any GPU sets it. *)
let max_group_segment = 511 * 512

(* The most scratch a lane takes on processor [p]: its share of a 64-lane wave's
   most, what COMPUTE_TMPRING_SIZE.WAVESIZE holds, 18 bits of 256 bytes on
   GFX12, 15 bits of 256 on GFX11 and 13 bits of 1024 before (LLVM's
   GCNSubtarget.h, getMaxWaveScratchSize). [p]'s generation is the major version
   LLVM's name starts with: "gfx1201", "gfx90a", "gfx11-generic". *)
let max_private_segment p =
  let n = String.length p in
  let digits =
    match String.index_opt p '-' with Some i -> i - 3 | None -> n - 5
  in
  let bits, unit =
    match int_of_string (String.sub p 3 digits) with
    | m when m >= 12 -> (18, 256)
    | 11 -> (15, 256)
    | _ -> (13, 1024)
  in
  ((1 lsl bits) - 1) * unit / 64

(* The processor of [o]'s flags. A generic processor's code object carries the
   version of the processor's code, which LLVM numbers from 1, in a code object
   of version 6 or later. *)
let processor (o : Rig_elf.t) =
  let mach = o.flags land Defs.ef_amdgpu_mach in
  let version =
    (o.flags land Defs.ef_amdgpu_generic_version)
    lsr Defs.ef_amdgpu_generic_version_offset
  in
  match List.assoc_opt mach Defs.processors with
  | None -> Error (strf "EF_AMDGPU_MACH 0x%x names no processor" mach)
  | Some name when not (List.mem_assoc name Defs.generic) -> Ok name
  | Some name when o.abi_version < Defs.elfabiversion_amdgpu_hsa_v6 ->
      Error
        (strf
           "%s code object has ABI version %d, expected %d (code object \
            version 6) or later"
           name o.abi_version Defs.elfabiversion_amdgpu_hsa_v6)
  | Some name when version = 0 ->
      Error
        (strf "%s code object has generic version 0, expected 1 or more" name)
  | Some name -> Ok name

(* The patch of [r]: its target's offset from the field, which REL64 writes. *)
let patch ~size i (r : Rig_elf.relocation) =
  let* () =
    if r.kind = r_amdgpu_rel64 then Ok ()
    else
      Error
        (strf "relocation %d is of kind %d, expected R_AMDGPU_REL64" i r.kind)
  in
  let* target =
    match r.symbol.place with
    | Image { offset; _ } -> Ok offset
    | Undefined | Absolute _ | Outside _ ->
        Error
          (strf "relocation %d uses %S, whose bytes the image lacks" i
             r.symbol.name)
  in
  let* addend =
    match r.addend with
    | Explicit a -> Ok a
    | Implicit _ ->
        Error (strf "relocation %d has no addend in its entry (SHT_REL)" i)
  in
  if r.offset + rel64_bytes > size then
    Error
      (strf "relocation %d patches bytes past the image's end at %d" i r.offset)
  else
    let b = Bytes.create rel64_bytes in
    Bytes.set_int64_le b 0 (Int64.of_int (target + addend - r.offset));
    Ok (r.offset, Bytes.unsafe_to_string b)

let rec patches ~size i acc = function
  | [] -> Ok (List.rev acc)
  | r :: rs ->
      let* p = patch ~size i r in
      patches ~size (i + 1) (p :: acc) rs

(* The [n] bytes of the image at [at], which lie in it: the sections' bytes
   there, then the patches over them. *)
let read (o : Rig_elf.t) ps at n =
  let b = Bytes.make n '\000' in
  let blit src src_at dst_at len =
    let lo = Int.max at dst_at and hi = Int.min (at + n) (dst_at + len) in
    if lo < hi then
      Bytes.blit_string src (src_at + lo - dst_at) b (lo - at) (hi - lo)
  in
  let section (s : Rig_elf.section) =
    match s.offset with Some off -> blit o.file s.at off s.length | None -> ()
  in
  Iarray.iter section o.sections;
  List.iter (fun (off, p) -> blit p 0 off (String.length p)) ps;
  Bytes.unsafe_to_string b

let field s (off, width) =
  match width with
  | 2 -> String.get_uint16_le s off
  | 4 -> Int32.to_int (String.get_int32_le s off) land 0xffff_ffff
  | _ -> Int64.to_int (String.get_int64_le s off)

(* Metadata *)

(* A reader of MessagePack values (msgpack.org's spec) in the bytes of [s]
   from [pos] to [stop]. [next] reads a value's header and answers its kind;
   [n] is then an integer's value, a string's length or the elements of an
   array or a map, and [at] where a string's bytes start. [key] reads a map's
   key, whose bytes it keeps at [key_at] and [key_len]. It allocates
   nothing. *)
type reader = {
  s : string;
  mutable pos : int;
  stop : int;
  mutable n : int;
  mutable at : int;
  mutable key_at : int;
  mutable key_len : int;
}

exception Malformed

(* The kinds of value [next] answers; any other is skipped whole. *)
let k_int = 0
let k_str = 1
let k_arr = 2
let k_map = 3
let k_skipped = 4

let take r k =
  if k < 0 || r.pos + k > r.stop then raise_notrace Malformed;
  let at = r.pos in
  r.pos <- at + k;
  at

let uint r k =
  let at = take r k in
  let v = ref 0 in
  for i = 0 to k - 1 do
    v := (!v lsl 8) lor String.get_uint8 r.s (at + i)
  done;
  !v

let sint r k =
  let v = uint r k and bits = 8 * k in
  if bits < 64 && v land (1 lsl (bits - 1)) <> 0 then v - (1 lsl bits) else v

let answer r kind n =
  r.n <- n;
  kind

let str r n =
  r.at <- take r n;
  answer r k_str n

let skip r k =
  ignore (take r k);
  k_skipped

let byte r =
  let at = r.pos in
  if at >= r.stop then raise_notrace Malformed;
  r.pos <- at + 1;
  Char.code (String.unsafe_get r.s at)

(* Raises [Malformed] past [stop] or at 0xc1, which starts no value. *)
let next r =
  match byte r with
  | b when b < 0x80 -> answer r k_int b
  | b when b < 0x90 -> answer r k_map (b land 0xf)
  | b when b < 0xa0 -> answer r k_arr (b land 0xf)
  | b when b < 0xc0 -> str r (b land 0x1f)
  | b when b >= 0xe0 -> answer r k_int (b - 0x100)
  | 0xc0 | 0xc2 | 0xc3 -> k_skipped
  | 0xc4 -> skip r (uint r 1)
  | 0xc5 -> skip r (uint r 2)
  | 0xc6 -> skip r (uint r 4)
  | 0xc7 -> skip r (uint r 1 + 1)
  | 0xc8 -> skip r (uint r 2 + 1)
  | 0xc9 -> skip r (uint r 4 + 1)
  | 0xca -> skip r 4
  | 0xcb -> skip r 8
  | 0xcc -> answer r k_int (uint r 1)
  | 0xcd -> answer r k_int (uint r 2)
  | 0xce -> answer r k_int (uint r 4)
  | 0xcf -> answer r k_int (uint r 8)
  | 0xd0 -> answer r k_int (sint r 1)
  | 0xd1 -> answer r k_int (sint r 2)
  | 0xd2 -> answer r k_int (sint r 4)
  | 0xd3 -> answer r k_int (sint r 8)
  | 0xd4 -> skip r 2
  | 0xd5 -> skip r 3
  | 0xd6 -> skip r 5
  | 0xd7 -> skip r 9
  | 0xd8 -> skip r 17
  | 0xd9 -> str r (uint r 1)
  | 0xda -> str r (uint r 2)
  | 0xdb -> str r (uint r 4)
  | 0xdc -> answer r k_arr (uint r 2)
  | 0xdd -> answer r k_arr (uint r 4)
  | 0xde -> answer r k_map (uint r 2)
  | 0xdf -> answer r k_map (uint r 4)
  | _ -> raise_notrace Malformed

(* Skips the rest of the value whose header [next] read as [kind]. *)
let rec rest r kind =
  if kind = k_arr || kind = k_map then
    let n = if kind = k_map then 2 * r.n else r.n in
    for _ = 1 to n do
      rest r (next r)
    done

(* Whether the [len] bytes at [at] of [r]'s string are [k]. *)
let is r at len k =
  len = String.length k
  &&
  let i = ref 0 in
  while !i < len && String.unsafe_get r.s (at + !i) = String.unsafe_get k !i do
    incr i
  done;
  !i = len

(* Reads the next key of a map, and answers [true] if it is a string, whose
   value is next; else skips the key and its value. *)
let key r =
  let k = next r in
  if k = k_str then (
    r.key_at <- r.at;
    r.key_len <- r.n;
    true)
  else (
    rest r k;
    rest r (next r);
    false)

let is_key r k = is r r.key_at r.key_len k

(* The note that holds a code object's metadata (AMDGPUUsage, "Code Object
   Metadata"): its owner, and the keys this module reads. *)
let note_owner = "AMDGPU\000"
let kernels_key = "amdhsa.kernels"
let name_key = ".name"
let max_threads_key = ".max_flat_workgroup_size"
let args_key = ".args"
let offset_key = ".offset"
let kind_key = ".value_kind"

(* The most work-items of a workgroup on any GPU, and so of a kernel the
   metadata does not bound (AMDGPUUsage, "amdgpu-flat-work-group-size"). *)
let most_threads = 1024

(* The value kinds of implicit arguments (AMDGPUUsage, "Code Object V5
   Metadata"): each starts "hidden_"; "hidden_none" is no argument. *)
let hidden_prefix = "hidden_"
let hidden_none = "hidden_none"

let hiddens =
  [
    ("hidden_block_count_x", Block_count X);
    ("hidden_block_count_y", Block_count Y);
    ("hidden_block_count_z", Block_count Z);
    ("hidden_group_size_x", Group_size X);
    ("hidden_group_size_y", Group_size Y);
    ("hidden_group_size_z", Group_size Z);
    ("hidden_remainder_x", Remainder X);
    ("hidden_remainder_y", Remainder Y);
    ("hidden_remainder_z", Remainder Z);
    ("hidden_global_offset_x", Global_offset X);
    ("hidden_global_offset_y", Global_offset Y);
    ("hidden_global_offset_z", Global_offset Z);
    ("hidden_grid_dims", Grid_dims);
    ("hidden_dynamic_lds_size", Dynamic_lds_size);
  ]

(* The implicit argument of the value kind of the [len] bytes at [at], [None]
   for no implicit argument. *)
let hidden_of r at len =
  let prefix = String.length hidden_prefix in
  if not (len >= prefix && is r at prefix hidden_prefix) then None
  else if is r at len hidden_none then None
  else
    match List.find_opt (fun (name, _) -> is r at len name) hiddens with
    | Some (_, h) -> Some h
    | None -> Some (Other (String.sub r.s at len))

(* The bytes of an implicit argument; [0] for another, whose size its
   dispatcher knows. *)
let hidden_bytes = function
  | Block_count _ | Dynamic_lds_size -> 4
  | Group_size _ | Remainder _ | Grid_dims -> 2
  | Global_offset _ -> 8
  | Other _ -> 0

(* The spans of [o.file] holding the descriptions of the notes of [o]'s
   sections whose owner is [owner] and type [kind], or [None] if a note runs
   past its section: each note is its owner's and its description's lengths,
   its type, then its owner and its description, each padded to 4 bytes. *)
let notes (o : Rig_elf.t) ~owner ~kind =
  let u32 at = Int32.to_int (String.get_int32_le o.file at) land 0xffff_ffff in
  let pad n = (n + 3) / 4 * 4 in
  let owned at n =
    n = String.length owner && String.equal (String.sub o.file at n) owner
  in
  let rec walk at stop acc =
    if at = stop then Some acc
    else if at + 12 > stop then None
    else
      let namesz = u32 at and descsz = u32 (at + 4) and t = u32 (at + 8) in
      let desc = at + 12 + pad namesz in
      if desc + descsz > stop then None
      else
        let acc =
          if t = kind && owned (at + 12) namesz then (desc, descsz) :: acc
          else acc
        in
        walk (Int.min stop (desc + pad descsz)) stop acc
  in
  Iarray.fold_left
    (fun acc (s : Rig_elf.section) ->
      match acc with
      | Some acc when s.kind = Defs.sht_note -> walk s.at (s.at + s.length) acc
      | acc -> acc)
    (Some []) o.sections

(* What the metadata note at [at] of [len] bytes of [file] says of each
   kernel it names: its most work-items, if it bounds them, and its implicit
   arguments, by increasing offset. *)
let read_metadata file (at, len) =
  let r =
    { s = file; pos = at; stop = at + len; n = 0; at = 0; key_at = 0; key_len = 0 }
  in
  (* An argument's implicit argument and its offset, if it is one. *)
  let arg () =
    let v = next r in
    if v <> k_map then (
      rest r v;
      None)
    else
      let hidden = ref None and offset = ref (-1) in
      for _ = 1 to r.n do
        if key r then
          let v = next r in
          if is_key r kind_key && v = k_str then hidden := hidden_of r r.at r.n
          else if is_key r offset_key && v = k_int then offset := r.n
          else rest r v
      done;
      match !hidden with
      | Some h when !offset >= 0 -> Some (h, !offset)
      | _ -> None
  in
  let kernel acc =
    let v = next r in
    if v <> k_map then (
      rest r v;
      acc)
    else
      let name = ref "" and threads = ref None and args = ref [] in
      for _ = 1 to r.n do
        if key r then
          let v = next r in
          if is_key r name_key && v = k_str then
            name := String.sub file r.at r.n
          else if is_key r max_threads_key && v = k_int then threads := Some r.n
          else if is_key r args_key && v = k_arr then
            for _ = 1 to r.n do
              match arg () with Some a -> args := a :: !args | None -> ()
            done
          else rest r v
      done;
      let by_offset (_, a) (_, b) = Int.compare a b in
      if !name = "" then acc
      else (!name, (!threads, List.stable_sort by_offset (List.rev !args))) :: acc
  in
  let kernels = ref [] in
  let v = next r in
  if v <> k_map then rest r v
  else
    for _ = 1 to r.n do
      if key r then
        let v = next r in
        if is_key r kernels_key && v = k_arr then
          for _ = 1 to r.n do
            kernels := kernel !kernels
          done
        else rest r v
    done;
  !kernels

let metadata (o : Rig_elf.t) =
  match notes o ~owner:note_owner ~kind:Defs.nt_amdgpu_metadata with
  | None -> Error "a note runs past its section"
  | Some spans -> (
      match List.concat_map (read_metadata o.file) spans with
      | exception Malformed -> Error "the metadata note is malformed"
      | ks -> Ok (List.sort (fun (a, _) (b, _) -> String.compare a b) ks))

let kd_suffix = ".kd"

let kernel_of o ~target ~most ~size ~meta:(threads, hidden) ps name kd =
  if kd + K.sizeof > size then
    Error
      (strf "kernel %S's descriptor at %d lies past the image's end" name kd)
  else
    let d = read o ps kd K.sizeof in
    let entry = kd + field d K.kernel_code_entry_byte_offset in
    let group_segment = field d K.group_segment_fixed_size in
    let private_segment = field d K.private_segment_fixed_size in
    let has flag = field d K.kernel_code_properties land flag <> 0 in
    if entry < 0 || entry >= size then
      Error (strf "kernel %S's code at %d lies outside the image" name entry)
    else if group_segment > max_group_segment then
      Error
        (strf "kernel %S takes %d bytes of LDS, expected at most %d" name
           group_segment max_group_segment)
    else if private_segment > most then
      Error
        (strf
           "kernel %S takes %d bytes of scratch per lane, expected at most %d \
            for %s"
           name private_segment most target)
    else if threads < 1 || threads > most_threads then
      Error
        (strf "kernel %S has workgroups of at most %d work-items, expected 1 \
               to %d"
           name threads most_threads)
    else
      let kernarg_size = field d K.kernarg_size in
      let outside (h, at) = at < 0 || at + hidden_bytes h > kernarg_size in
      match List.find_opt outside hidden with
      | Some (_, at) ->
          Error
            (strf
               "kernel %S's implicit argument at %d lies outside its %d bytes \
                of arguments"
               name at kernarg_size)
      | None ->
          Ok
            {
              descriptor = kd;
              entry;
              group_segment;
              private_segment;
              kernarg_size;
              rsrc1 = field d K.compute_pgm_rsrc1;
              rsrc2 = field d K.compute_pgm_rsrc2;
              rsrc3 = field d K.compute_pgm_rsrc3;
              wave32 =
                has Defs.amd_kernel_code_properties_enable_wavefront_size32;
              dispatch_ptr =
                has Defs.amd_kernel_code_properties_enable_sgpr_dispatch_ptr;
              private_segment_buffer =
                has
                  Defs
                  .amd_kernel_code_properties_enable_sgpr_private_segment_buffer;
              max_threads = threads;
              hidden;
            }

(* The kernels of [o], by name in increasing order, each at the first symbol
   [name ^ ".kd"] in its image. *)
let descriptors (o : Rig_elf.t) =
  let descriptor (s : Rig_elf.symbol) =
    match s.place with
    | Image { offset; _ } when String.ends_with ~suffix:kd_suffix s.name ->
        let n = String.length s.name - String.length kd_suffix in
        Some (String.sub s.name 0 n, offset)
    | Image _ | Undefined | Absolute _ | Outside _ -> None
  in
  let rec first = function
    | ((a, _) as k) :: (b, _) :: rest when String.equal a b -> first (k :: rest)
    | k :: rest -> k :: first rest
    | [] -> []
  in
  List.filter_map descriptor (Iarray.to_list o.symbols)
  |> List.stable_sort (fun (a, _) (b, _) -> String.compare a b)
  |> first

(* The kernels of [o] named by the descriptors [ds], with what [metas] says
   of them; both are sorted by name. *)
let rec kernels o ~target ~most ~size ps acc metas ds =
  match (ds, metas) with
  | [], _ -> Ok (List.rev acc)
  | (name, _) :: _, (m, _) :: metas when String.compare m name < 0 ->
      kernels o ~target ~most ~size ps acc metas ds
  | (name, kd) :: ds, metas -> (
      let threads, hidden =
        match metas with
        | (m, meta) :: _ when String.equal m name -> meta
        | _ -> (None, [])
      in
      let meta = (Option.value threads ~default:most_threads, hidden) in
      match kernel_of o ~target ~most ~size ~meta ps name kd with
      | Ok k -> kernels o ~target ~most ~size ps ((name, k) :: acc) metas ds
      | Error _ as e -> e)

let of_string obj =
  let* o = Rig_elf.of_string obj in
  let* () =
    if o.machine = Defs.em_amdgpu then Ok ()
    else
      Error
        (strf "e_machine %d, expected EM_AMDGPU (%d)" o.machine Defs.em_amdgpu)
  in
  let* target = processor o in
  let* () =
    if o.size <= max_size then Ok ()
    else Error (strf "the image is %d bytes, expected at most 2^48" o.size)
  in
  let size = (o.size + 3) / 4 * 4 in
  let* ps = patches ~size 0 [] o.relocations in
  let most = max_private_segment target in
  let* metas = metadata o in
  let* ks = kernels o ~target ~most ~size ps [] metas (descriptors o) in
  let n = List.length ks in
  let names = Array.make n "" and records = Array.make n None in
  List.iteri
    (fun i (name, k) ->
      names.(i) <- name;
      records.(i) <- Some k)
    ks;
  Ok { elf = o; target; size; patches = ps; names; records }

let target co = co.target
let size co = co.size
let elf co = co.elf
let patches co = co.patches

let image co =
  let b = Bytes.make co.size '\000' in
  let put (s : Rig_elf.section) =
    match s.offset with
    | Some off -> Bytes.blit_string co.elf.file s.at b off s.length
    | None -> ()
  in
  Iarray.iter put co.elf.sections;
  let patch (off, p) = Bytes.blit_string p 0 b off (String.length p) in
  List.iter patch co.patches;
  Bytes.unsafe_to_string b

let kernels co = Array.to_list co.names

(* The position of [name] in [names], by bisection, or [-1]. *)
let rec find names name lo hi =
  if lo >= hi then -1
  else
    let mid = (lo + hi) / 2 in
    let c = String.compare name (Array.unsafe_get names mid) in
    if c = 0 then mid
    else if c < 0 then find names name lo mid
    else find names name (mid + 1) hi

let kernel co name =
  let i = find co.names name 0 (Array.length co.names) in
  if i < 0 then None else Array.unsafe_get co.records i

let runs_on co g =
  let gpu = Gpu.processor g in
  String.equal co.target gpu
  || List.exists
       (fun (g, members) -> String.equal g co.target && List.mem gpu members)
       Defs.generic
