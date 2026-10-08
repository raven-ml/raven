(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let ( let* ) = Result.bind

(* A relocation as the image needs it: the offset of the bytes it patches, how
   many, and the symbol's offset and the addend, which a base completes. *)
type relocation = {
  at : int;
  width : int;
  high : bool;
  offset : int;
  addend : int64;
}

(* The relocations a cubin holds: R_CUDA_64 writes a symbol's 64-bit address,
   R_CUDA_ABS32_LO_32 and R_CUDA_ABS32_HI_32 its low and high 32 bits. *)
let r_cuda_64 = 0x2
let r_cuda_abs32_lo_32 = 0x38
let r_cuda_abs32_hi_32 = 0x39

(* NVIDIA's machine, e_machine. *)
let em_cuda = 190

(* The cubin's sections' alignment in its image. *)
let section_align = 128

(* The longest image: what the 49-bit virtual addresses of GPUs before Hopper
   reach. *)
let max_image = 1 lsl 49

(* The GPU's instruction prefetch may read past the code: the image ends with
   zeros up to the next 4 KiB, and 4 KiB more. *)
let page = 0x1000

(* The [width] bytes of [file] at [p], an unsigned little-endian number. *)
let field file p width =
  if width = 8 then String.get_int64_le file p
  else Int64.of_int (Int32.to_int (String.get_int32_le file p) land 0xffff_ffff)

let relocation (o : Device_elf.t) i (r : Device_elf.relocation) =
  let* at, width, high =
    if r.kind = r_cuda_64 then Ok (r.offset, 8, false)
    else if r.kind = r_cuda_abs32_lo_32 then Ok (r.offset + 4, 4, false)
    else if r.kind = r_cuda_abs32_hi_32 then Ok (r.offset + 4, 4, true)
    else
      Error
        (strf
           "relocation %d is of kind 0x%x, expected R_CUDA_64, \
            R_CUDA_ABS32_LO_32 or R_CUDA_ABS32_HI_32"
           i r.kind)
  in
  let* offset =
    match r.symbol.place with
    | Image { offset; _ } -> Ok offset
    | Undefined | Absolute _ | Outside _ ->
        Error
          (strf "relocation %d uses %S, whose bytes the image lacks" i
             r.symbol.name)
  in
  if at + width > o.size then
    Error (strf "relocation %d patches bytes past the image's end at %d" i at)
  else
    (* A 32-bit field lies past the relocation's offset. *)
    let skip = at - r.offset in
    let* addend =
      match r.addend with
      | Explicit a -> Ok (Int64.of_int a)
      | Implicit { at = p; length } when skip + width <= length ->
          Ok (field o.file (p + skip) width)
      | Implicit _ ->
          Error
            (strf "relocation %d patches bytes past the end of its section" i)
    in
    Ok { at; width; high; offset; addend }

(* Kernels *)

type bank = { index : int; offset : int; bytes : int }

type kernel = {
  code : int;
  code_bytes : int;
  registers : int;
  shared_bytes : int;
  stack_bytes : int;
  params_offset : int;
  banks : bank list;
}

let text = ".text."
let shared = ".nv.shared."
let info = ".nv.info"
let own_info = ".nv.info."
let constant = ".nv.constant"

(* .nv.info attributes, whose parameters say what they record. *)
let eiattr_param_cbank = 0xa
let eiattr_min_stack_size = 0x12
let eiattr_regcount = 0x2f

(* The format of an attribute whose data follows its header; the others hold a
   16-bit value in place of their size. *)
let eifmt_sval = 4

(* FNV-1a, in OCaml's 63-bit ints. *)
let fnv_offset = 0x811c9dc5
let fnv_prime = 0x100000001b3

let rec hash_from s i h =
  if i = String.length s then h
  else hash_from s (i + 1) (h lxor Char.code (String.unsafe_get s i) * fnv_prime)

let rec compare_from s off n t m i =
  if i = n || i = m then n - m
  else
    let c =
      Char.code (String.unsafe_get s (off + i))
      - Char.code (String.unsafe_get t i)
    in
    if c <> 0 then c else compare_from s off n t m (i + 1)

let rec find_slot names hashes h name off lo hi =
  if lo >= hi then -1
  else
    let mid = (lo + hi) / 2 in
    let c = Int.compare h (Array.unsafe_get hashes mid) in
    let c =
      if c <> 0 then c
      else
        let t = Array.unsafe_get names mid in
        compare_from name off (String.length name - off) t (String.length t) 0
    in
    if c = 0 then mid
    else if c < 0 then find_slot names hashes h name off lo mid
    else find_slot names hashes h name off (mid + 1) hi

(* Kernels are found by name, and a section names its kernel by the rest of its
   name after a prefix. Names are kept by their hash, then by their bytes: [slot
   names hashes name off] is the position in [names] of the name that [name]
   holds from [off], or [-1]. It reads the name once and compares it in place,
   so that nothing is copied. *)
let slot names hashes name off =
  let h = hash_from name off fnv_offset in
  find_slot names hashes h name off 0 (Array.length names)

(* [names] without repeats, by hash, then bytes, and their hashes. *)
let order names =
  let keyed = List.map (fun n -> (hash_from n 0 fnv_offset, n)) names in
  let keyed = Array.of_list (List.sort_uniq compare keyed) in
  (Array.map snd keyed, Array.map fst keyed)

(* The position after [prefix] in [name], or [-1]. A loop of its own:
   [String.starts_with] allocates a closure each call. *)
let rec prefixed prefix name i =
  i = String.length prefix
  || String.unsafe_get prefix i = String.unsafe_get name i
     && prefixed prefix name (i + 1)

let after ~prefix name =
  if String.length name >= String.length prefix && prefixed prefix name 0 then
    String.length prefix
  else -1

(* An .nv.info section whose last attribute runs past its end, which [of_string]
   returns as its error. *)
exception Truncated of string

let truncated (s : Device_elf.section) at =
  raise_notrace
    (Truncated
       (strf "the attribute at %d of section %S is truncated" (at - s.at) s.name))

(* Calls [f param at] for each attribute of the .nv.info section [s] whose data
   follows its header and holds at least a symbol index and a 32-bit value, [at]
   being where the data starts in the object. Raises [Truncated] if an attribute
   runs past the section's end. *)
let iter_attributes (o : Device_elf.t) (s : Device_elf.section) f =
  let file = o.file and stop = s.at + s.length in
  (* [go at] is where an attribute from [at] on runs past [stop], or [-1]. *)
  let rec go at =
    if at >= stop then -1
    else if at + 4 > stop then at
    else
      let fmt = Char.code (String.unsafe_get file at)
      and param = Char.code (String.unsafe_get file (at + 1))
      and size = String.get_uint16_le file (at + 2) in
      if fmt <> eifmt_sval then go (at + 4)
      else if at + 4 + size > stop then at
      else (
        if size >= 8 then f param (at + 4);
        go (at + 4 + size))
  in
  let past = go s.at in
  if past >= 0 then truncated s past

(* A 32-bit little-endian unsigned integer, read as two halves, which no int32
   is boxed for. *)
let u32 s off =
  String.get_uint16_le s off lor (String.get_uint16_le s (off + 2) lsl 16)

(* The bank the section [name] holds, [(i, owner)]: [owner] is [-1] for
   .nv.constantI, every kernel's, and for .nv.constantI.k the position of k in
   [name]. *)
let bank_of name =
  let start = after ~prefix:constant name in
  if start < 0 then None
  else
    let n = String.length name in
    let rec digits i =
      if i < n && name.[i] >= '0' && name.[i] <= '9' then digits (i + 1) else i
    in
    let stop = digits start in
    match int_of_string_opt (String.sub name start (stop - start)) with
    | None -> None
    | Some i when stop = n -> Some (i, -1)
    | Some i when name.[stop] = '.' -> Some (i, stop + 1)
    | Some _ -> None

(* [banks] with [b], in place of the bank of its index if there is one. *)
let set_bank banks b =
  if List.exists (fun x -> x.index = b.index) banks then
    List.map (fun x -> if x.index = b.index then b else x) banks
  else banks @ [ b ]

(* The names of the kernels of the code sections the image holds, in order. *)
let kernel_names (o : Device_elf.t) =
  let code (s : Device_elf.section) =
    match s.offset with
    | Some _ when after ~prefix:text s.name >= 0 ->
        let n = String.length text in
        Some (String.sub s.name n (String.length s.name - n))
    | _ -> None
  in
  List.filter_map code (Iarray.to_list o.sections)

(* A kernel's record while the sections are read. *)
type reading = {
  mutable code : int;
  mutable code_bytes : int;
  mutable registers : int;
  mutable shared_bytes : int;
  mutable stack_bytes : int;
  mutable params_offset : int;
  mutable banks : bank list;
}

(* Every kernel's record, in the order of [names], in one pass over the sections
   in order: each section updates the kernels it is about, so a later section or
   attribute overrides an earlier one. *)
let read_kernels (o : Device_elf.t) names hashes =
  let reading _ =
    {
      code = 0;
      code_bytes = 0;
      registers = 0;
      shared_bytes = 0;
      stack_bytes = 0;
      params_offset = 0;
      banks = [];
    }
  in
  let ks = Array.map reading names in
  let at name off f =
    let i = slot names hashes name off in
    if i >= 0 then f (Array.unsafe_get ks i)
  in
  (* The kernel the symbol [i] is: the kernel of the code section it is in, else
     the kernel of its name. *)
  let function_of i f =
    if i >= 0 && i < Iarray.length o.symbols then
      let (s : Device_elf.symbol) = Iarray.get o.symbols i in
      let section =
        match s.place with
        | Image { section; _ } -> (Iarray.get o.sections section).name
        | Undefined | Absolute _ | Outside _ -> ""
      in
      let off = after ~prefix:text section in
      if off >= 0 then at section off f else if s.name <> "" then at s.name 0 f
  in
  (* An attribute's data is the index of the symbol it is about, then its value.
     In .nv.info, a function's registers and stack; in .nv.info.k, k's
     parameters. *)
  let global param data =
    if param = eiattr_regcount then
      let v = u32 o.file (data + 4) in
      function_of (u32 o.file data) (fun r -> r.registers <- v)
    else if param = eiattr_min_stack_size then
      let v = u32 o.file (data + 4) in
      function_of (u32 o.file data) (fun r -> r.stack_bytes <- v)
  in
  let own_attribute r param data =
    if param = eiattr_param_cbank then
      r.params_offset <- String.get_uint16_le o.file (data + 4)
  in
  let section (s : Device_elf.section) =
    let name = s.name in
    let code_at = after ~prefix:text name
    and shared_at = after ~prefix:shared name in
    match (s.offset, bank_of name) with
    | Some offset, _ when code_at >= 0 ->
        at name code_at (fun r ->
            r.code <- offset;
            r.code_bytes <- s.size)
    | _ when code_at >= 0 -> ()
    | _ when shared_at >= 0 ->
        at name shared_at (fun r -> r.shared_bytes <- s.size)
    | Some offset, Some (index, owner) ->
        let b = { index; offset; bytes = s.size } in
        let add r = r.banks <- set_bank r.banks b in
        if owner < 0 then Array.iter add ks else at name owner add
    | _ when name = info -> iter_attributes o s global
    | _ ->
        let k = after ~prefix:own_info name in
        if k >= 0 then
          at name k (fun r -> iter_attributes o s (own_attribute r))
  in
  Iarray.iter section o.sections;
  let freeze r =
    let k : kernel =
      {
        code = r.code;
        code_bytes = r.code_bytes;
        registers = r.registers;
        shared_bytes = r.shared_bytes;
        stack_bytes = r.stack_bytes;
        params_offset = r.params_offset;
        banks = r.banks;
      }
    in
    Some k
  in
  Array.map freeze ks

type t = {
  elf : Device_elf.t;
  size : int;
  relocations : relocation list;
  kernels : string list;
  names : string array; (* every kernel's name, by hash, then bytes *)
  hashes : int array;
  records : kernel option array;
      (* each kernel's record, in the order of [names], all [Some]: [kernel]
         returns one without allocating *)
}

(* The sections the GPU's memory holds: ELF's, but a kernel's shared memory,
   which NVIDIA's compilers mark allocated though it is on chip. *)
let held (s : Device_elf.section) =
  Device_elf.allocated ~machine:em_cuda s && after ~prefix:shared s.name < 0

let of_string obj =
  let* o = Device_elf.of_string ~align:section_align ~held obj in
  (* The object is NVIDIA's, and its image fits: the ELF image's end rounds up
     to a page, and a page follows. *)
  let* () =
    if o.machine <> em_cuda then
      Error (strf "e_machine %d, expected EM_CUDA (%d)" o.machine em_cuda)
    else if o.size <= max_image - page then Ok ()
    else
      Error
        (strf "the sections take %d bytes, expected at most 2^49 - 4096" o.size)
  in
  let rec relocations i acc = function
    | [] -> Ok (List.rev acc)
    | r :: rs ->
        let* r = relocation o i r in
        relocations (i + 1) (r :: acc) rs
  in
  let* relocations = relocations 0 [] o.relocations in
  let size = ((o.size + page - 1) / page * page) + page in
  let kernels = kernel_names o in
  let names, hashes = order kernels in
  match read_kernels o names hashes with
  | records ->
      Ok { elf = o; size; relocations; kernels; names; hashes; records }
  | exception Truncated msg -> Error msg

let size c = c.size
let elf c = c.elf

let patches c ~base =
  let patch (r : relocation) =
    (* Modulo 2^64: an int would wrap at 2^62. *)
    let v = Int64.(add (add (of_int base) (of_int r.offset)) r.addend) in
    let b = Bytes.create r.width in
    if r.width = 8 then Bytes.set_int64_le b 0 v
    else
      Bytes.set_int32_le b 0
        (Int64.to_int32 (if r.high then Int64.shift_right_logical v 32 else v));
    (r.at, Bytes.unsafe_to_string b)
  in
  List.map patch c.relocations

let kernels c = c.kernels

let kernel c name =
  let i = slot c.names c.hashes name 0 in
  if i < 0 then None else Array.unsafe_get c.records i
