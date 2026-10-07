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
  addend : int;
}

type t = { elf : Device_elf.t; size : int; relocations : relocation list }

(* The relocations a cubin holds: R_CUDA_64 writes a symbol's 64-bit address,
   R_CUDA_ABS32_LO_32 and R_CUDA_ABS32_HI_32 its low and high 32 bits. *)
let r_cuda_64 = 0x2
let r_cuda_abs32_lo_32 = 0x38
let r_cuda_abs32_hi_32 = 0x39

(* The cubin's sections' alignment in its image. *)
let section_align = 128

(* The longest image: what the 49-bit virtual addresses of GPUs before Hopper
   reach. *)
let max_image = 1 lsl 49

(* The GPU's instruction prefetch may read past the code: the image ends with
   zeros up to the next 4 KiB, and 4 KiB more. *)
let page = 0x1000

let relocation ~image i (r : Device_elf.relocation) =
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
  if at + width > image then
    Error (strf "relocation %d patches bytes past the image's end at %d" i at)
  else Ok { at; width; high; offset; addend = r.addend }

let of_string obj =
  let* o = Device_elf.of_string ~align:section_align obj in
  let* () =
    (* The ELF image's end rounds up to a page, and a page follows. *)
    if o.size <= max_image - page then Ok ()
    else
      Error
        (strf "the image would be longer than 2^49 bytes: its sections take %d"
           o.size)
  in
  let rec relocations i acc = function
    | [] -> Ok (List.rev acc)
    | r :: rs ->
        let* r = relocation ~image:o.size i r in
        relocations (i + 1) (r :: acc) rs
  in
  let* relocations = relocations 0 [] o.relocations in
  let size = ((o.size + page - 1) / page * page) + page in
  Ok { elf = o; size; relocations }

let size c = c.size
let elf c = c.elf

let patches c ~base =
  let patch r =
    (* Modulo 2^64: an int would wrap at 2^62. *)
    let v =
      Int64.(add (add (of_int base) (of_int r.offset)) (of_int r.addend))
    in
    let b = Bytes.create r.width in
    if r.width = 8 then Bytes.set_int64_le b 0 v
    else
      Bytes.set_int32_le b 0
        (Int64.to_int32 (if r.high then Int64.shift_right_logical v 32 else v));
    (r.at, Bytes.unsafe_to_string b)
  in
  List.map patch c.relocations

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
let info = ".nv.info"
let constant = ".nv.constant"

(* .nv.info attributes, whose parameters say what they record. *)
let eiattr_param_cbank = 0xa
let eiattr_min_stack_size = 0x12
let eiattr_regcount = 0x2f

(* The format of an attribute whose data follows its header; the others hold a
   16-bit value in place of their size. *)
let eifmt_sval = 4

let kernel_of_section name =
  if String.starts_with ~prefix:text name then
    Some
      (String.sub name (String.length text)
         (String.length name - String.length text))
  else None

(* The kernels of the code sections the image holds, in order. *)
let kernels c =
  let code (s : Device_elf.section) =
    match s.offset with Some _ -> kernel_of_section s.name | None -> None
  in
  List.filter_map code (Iarray.to_list c.elf.sections)

(* The attributes of the .nv.info section [s], as (parameter, data), up to the
   first that the section truncates. *)
let attributes (o : Device_elf.t) (s : Device_elf.section) =
  let n = s.length and file = o.file in
  let rec go off acc =
    if off + 4 > n then List.rev acc
    else
      let fmt = Char.code file.[s.at + off]
      and param = Char.code file.[s.at + off + 1]
      and size = String.get_uint16_le file (s.at + off + 2) in
      if fmt <> eifmt_sval then go (off + 4) ((param, "") :: acc)
      else if off + 4 + size > n then List.rev acc
      else
        go
          (off + 4 + size)
          ((param, String.sub file (s.at + off + 4) size) :: acc)
  in
  go 0 []

let u32 s off = Int32.to_int (String.get_int32_le s off) land 0xffff_ffff

(* The index of the bank the section [name] holds for the kernel [k]: of a
   section .nv.constantI, or .nv.constantI.k. *)
let bank_index k name =
  if not (String.starts_with ~prefix:constant name) then None
  else
    let n = String.length constant in
    let rec digits i =
      if i < String.length name && name.[i] >= '0' && name.[i] <= '9' then
        digits (i + 1)
      else i
    in
    let stop = digits n in
    let rest = String.sub name stop (String.length name - stop) in
    if stop = n || (rest <> "" && rest <> "." ^ k) then None
    else int_of_string_opt (String.sub name n (stop - n))

(* [banks] with [b], in place of the bank of its index if there is one. *)
let set_bank banks b =
  if List.exists (fun x -> x.index = b.index) banks then
    List.map (fun x -> if x.index = b.index then b else x) banks
  else banks @ [ b ]

(* The kernel the symbol [i] is: the kernel of the code section it is in, else
   its name. *)
let function_of (o : Device_elf.t) i =
  if i < 0 || i >= Iarray.length o.symbols then None
  else
    let (s : Device_elf.symbol) = Iarray.get o.symbols i in
    let in_code =
      match s.place with
      | Image { section; _ } ->
          kernel_of_section (Iarray.get o.sections section).name
      | Undefined | Absolute _ | Outside _ -> None
    in
    match in_code with
    | Some _ -> in_code
    | None -> if s.name = "" then None else Some s.name

let none =
  {
    code = 0;
    code_bytes = 0;
    registers = 0;
    shared_bytes = 0;
    stack_bytes = 0;
    params_offset = 0;
    banks = [];
  }

let kernel c name =
  let o = c.elf and own_info = info ^ "." ^ name in
  (* An attribute's data is the index of the symbol it is about, then its
     value. *)
  let attribute (s : Device_elf.section) k (param, data) =
    let long = String.length data >= 8 in
    let ours () = function_of o (u32 data 0) = Some name in
    if s.name = own_info && param = eiattr_param_cbank && long then
      { k with params_offset = String.get_uint16_le data 4 }
    else if s.name = info && param = eiattr_min_stack_size && long && ours ()
    then { k with stack_bytes = u32 data 4 }
    else if s.name = info && param = eiattr_regcount && long && ours () then
      { k with registers = u32 data 4 }
    else k
  in
  let section k (s : Device_elf.section) =
    match (s.offset, bank_index name s.name) with
    | Some offset, _ when s.name = text ^ name ->
        { k with code = offset; code_bytes = s.size }
    | _ when s.name = ".nv.shared." ^ name -> { k with shared_bytes = s.size }
    | Some offset, Some index ->
        { k with banks = set_bank k.banks { index; offset; bytes = s.size } }
    | _ when String.starts_with ~prefix:info s.name ->
        List.fold_left (attribute s) k (attributes o s)
    | _ -> k
  in
  if List.mem name (kernels c) then
    Some (Iarray.fold_left section none o.sections)
  else None
