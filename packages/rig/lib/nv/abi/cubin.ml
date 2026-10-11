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

let relocation (o : Rig_elf.t) i (r : Rig_elf.relocation) =
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
  params_bytes : int;
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

(* The position after [prefix] in [name], or [-1]. *)
let after ~prefix name =
  if String.starts_with ~prefix name then String.length prefix else -1

(* An .nv.info section with an attribute that runs past its end, or with one the
   reader reads whose data is too short, which [of_string] returns as its
   error. *)
exception Malformed of string

(* The attributes the reader reads hold a symbol's index and a 32-bit value
   (EIATTR_PARAM_CBANK: two 16-bit values). *)
let read_bytes = 8

let reads param =
  param = eiattr_regcount
  || param = eiattr_min_stack_size
  || param = eiattr_param_cbank

(* Raises [Malformed] for the attribute at [at] of [s], where the walk
   stopped. *)
let malformed file (s : Rig_elf.section) at =
  let stop = s.at + s.length in
  let size = if at + 4 <= stop then String.get_uint16_le file (at + 2) else 0 in
  let msg =
    if at + 4 <= stop && at + 4 + size <= stop then
      strf
        "the attribute at %d of section %S holds %d bytes, expected at least %d"
        (at - s.at) s.name size read_bytes
    else
      strf "the attribute at %d of section %S is truncated" (at - s.at) s.name
  in
  raise_notrace (Malformed msg)

(* Calls [f param at] for each attribute of the .nv.info section [s] whose data
   follows its header and holds at least [read_bytes], [at] being where the data
   starts in the object. Raises [Malformed] if an attribute runs past the
   section's end, or one the reader reads holds less. *)
let iter_attributes (o : Rig_elf.t) (s : Rig_elf.section) f =
  let file = o.file and stop = s.at + s.length in
  (* [go at] is where an attribute from [at] on is malformed, or [-1]. *)
  let rec go at =
    if at >= stop then -1
    else if at + 4 > stop then at
    else
      let fmt = Char.code (String.unsafe_get file at)
      and param = Char.code (String.unsafe_get file (at + 1))
      and size = String.get_uint16_le file (at + 2) in
      if fmt <> eifmt_sval then go (at + 4)
      else if at + 4 + size > stop then at
      else if size >= read_bytes then (
        f param (at + 4);
        go (at + 4 + size))
      else if reads param then at
      else go (at + 4 + size)
  in
  let past = go s.at in
  if past >= 0 then malformed file s past

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
let kernel_names (o : Rig_elf.t) =
  let code (s : Rig_elf.section) =
    match s.offset with
    | Some _ when after ~prefix:text s.name >= 0 ->
        let n = String.length text in
        Some (String.sub s.name n (String.length s.name - n))
    | _ -> None
  in
  List.filter_map code (Iarray.to_list o.sections)

let empty =
  {
    code = 0;
    code_bytes = 0;
    registers = 0;
    shared_bytes = 0;
    stack_bytes = 0;
    params_offset = 0;
    params_bytes = 0;
    banks = [];
  }

(* Every kernel's record, by name, in one pass over the sections in order: each
   section updates the kernels it is about, so a later section or attribute
   overrides an earlier one. *)
let read_kernels (o : Rig_elf.t) names =
  let ks = Hashtbl.create 16 in
  List.iter (fun name -> Hashtbl.replace ks name empty) names;
  (* Updates with [f] the kernel [name] names from [off], if there is one. *)
  let at name off f =
    let k = String.sub name off (String.length name - off) in
    match Hashtbl.find_opt ks k with
    | Some r -> Hashtbl.replace ks k (f r)
    | None -> ()
  in
  let named name off =
    Hashtbl.mem ks (String.sub name off (String.length name - off))
  in
  (* The kernel the symbol [i] is: the kernel of the code section it is in, else
     the kernel of its name. *)
  let function_of i f =
    if i >= 0 && i < Iarray.length o.symbols then
      let (s : Rig_elf.symbol) = Iarray.get o.symbols i in
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
      function_of (u32 o.file data) (fun r -> { r with registers = v })
    else if param = eiattr_min_stack_size then
      let v = u32 o.file (data + 4) in
      function_of (u32 o.file data) (fun r -> { r with stack_bytes = v })
  in
  let own_attribute name k param data =
    if param = eiattr_param_cbank then
      at name k (fun r ->
          {
            r with
            params_offset = String.get_uint16_le o.file (data + 4);
            params_bytes = String.get_uint16_le o.file (data + 6);
          })
  in
  let section (s : Rig_elf.section) =
    let name = s.name in
    let code_at = after ~prefix:text name
    and shared_at = after ~prefix:shared name in
    match (s.offset, bank_of name) with
    | Some offset, _ when code_at >= 0 ->
        at name code_at (fun r -> { r with code = offset; code_bytes = s.size })
    | _ when code_at >= 0 -> ()
    | _ when shared_at >= 0 ->
        at name shared_at (fun r -> { r with shared_bytes = s.size })
    | Some offset, Some (index, owner) ->
        let b = { index; offset; bytes = s.size } in
        let add r = { r with banks = set_bank r.banks b } in
        if owner < 0 then
          Hashtbl.filter_map_inplace (fun _ r -> Some (add r)) ks
        else at name owner add
    | _ when name = info -> iter_attributes o s global
    | _ ->
        let k = after ~prefix:own_info name in
        if k >= 0 && named name k then
          iter_attributes o s (own_attribute name k)
  in
  Iarray.iter section o.sections;
  ks

type t = {
  elf : Rig_elf.t;
  size : int;
  relocations : relocation list;
  kernels : string list;
  records : (string, kernel) Hashtbl.t; (* read only once made *)
}

(* The sections the GPU's memory holds: ELF's, but a kernel's shared memory,
   which NVIDIA's compilers mark allocated though it is on chip. *)
let held (s : Rig_elf.section) =
  Rig_elf.allocated ~machine:em_cuda s && after ~prefix:shared s.name < 0

let of_string obj =
  let* o = Rig_elf.of_string ~align:section_align ~held obj in
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
  match read_kernels o kernels with
  | records -> Ok { elf = o; size; relocations; kernels; records }
  | exception Malformed msg -> Error msg

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
let kernel c name = Hashtbl.find_opt c.records name
