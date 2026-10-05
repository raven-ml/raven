(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module Elf = Nx_device_elf

type t = {
  elf : Elf.t;
  image : string;
  relocations : (int * int * int) list;
      (* (image offset to patch, target offset plus addend, type) *)
}

let round_up n a = (n + a - 1) / a * a

(* Relocation types *)

let r_cuda_64 = 0x2
let r_cuda_abs32_lo = 0x38
let r_cuda_abs32_hi = 0x39

let relocation (r : Elf.relocation) =
  if not (List.mem r.kind [ r_cuda_64; r_cuda_abs32_lo; r_cuda_abs32_hi ]) then
    Error
      (Printf.sprintf "the cubin has a relocation of unknown type 0x%x" r.kind)
  else
    match r.target with
    | Offset target -> Ok (r.at, target + r.addend, r.kind)
    | External s ->
        Error (Printf.sprintf "the cubin refers to an undefined symbol %s" s)

let of_string obj =
  match Elf.load ~align:128 obj with
  | exception Failure why -> Error why
  | elf -> (
      let rec relocations acc = function
        | [] -> Ok (List.rev acc)
        | r :: rest -> (
            match relocation r with
            | Ok r -> relocations (r :: acc) rest
            | Error _ as e -> e)
      in
      match relocations [] elf.relocations with
      | Error _ as e -> e
      | Ok relocations ->
          let n = String.length elf.image in
          let pad = round_up n 0x1000 + 0x1000 - n in
          let image = elf.image ^ String.make pad '\000' in
          Ok { elf; image; relocations })

let image c = c.image

let relocate c ~base =
  let b = Bytes.of_string c.image in
  List.iter
    (fun (at, target, kind) ->
      let v = base + target in
      if kind = r_cuda_64 then Bytes.set_int64_le b at (Int64.of_int v)
      else if kind = r_cuda_abs32_lo then
        Bytes.set_int32_le b (at + 4) (Int32.of_int (v land 0xffff_ffff))
      else
        Bytes.set_int32_le b (at + 4)
          (Int32.of_int ((v lsr 32) land 0xffff_ffff)))
    c.relocations;
  Bytes.to_string b

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

(* .nv.info attributes, whose parameters say what they record. *)
let eiattr_param_cbank = 0xa
let eiattr_min_stack_size = 0x12
let eiattr_regcount = 0x2f

(* The format of an attribute whose data follows its header; the others hold a
   16-bit value in place of their size. *)
let eifmt_sval = 4

(* The attributes of an .nv.info section, as (parameter, data), up to the first
   that the section truncates. *)
let attributes (s : Elf.section) =
  let n = String.length s.contents in
  let rec go off acc =
    if off + 4 > n then List.rev acc
    else
      let fmt = Char.code s.contents.[off]
      and param = Char.code s.contents.[off + 1]
      and size = String.get_uint16_le s.contents (off + 2) in
      if fmt <> eifmt_sval then go (off + 4) ((param, "") :: acc)
      else if off + 4 + size > n then List.rev acc
      else
        go
          (off + 4 + size)
          ((param, String.sub s.contents (off + 4) size) :: acc)
  in
  go 0 []

let u32 s off = Int32.to_int (String.get_int32_le s off) land 0xffff_ffff

(* The index of the bank [name] holds for the kernel [k]: of a section
   .nv.constantI, or .nv.constantI.k. *)
let bank_index k name =
  let prefix = ".nv.constant" in
  if not (String.starts_with ~prefix name) then None
  else
    let n = String.length prefix in
    let rec digits i =
      if i < String.length name && name.[i] >= '0' && name.[i] <= '9' then
        digits (i + 1)
      else i
    in
    let stop = digits n in
    let rest = String.sub name stop (String.length name - stop) in
    if stop = n || (rest <> "" && rest <> "." ^ k) then None
    else int_of_string_opt (String.sub name n (stop - n))

(* [banks] with bank [b], in place if its index is there. *)
let set_bank banks b =
  if List.exists (fun x -> x.index = b.index) banks then
    List.map (fun x -> if x.index = b.index then b else x) banks
  else banks @ [ b ]

let text = ".text."

let kernel_of_section name =
  if String.starts_with ~prefix:text name then
    Some
      (String.sub name (String.length text)
         (String.length name - String.length text))
  else None

let kernels c =
  List.filter_map
    (fun (s : Elf.section) -> kernel_of_section s.name)
    c.elf.sections

(* The function the symbol [i] is: the kernel of the code section it is in, else
   its name. *)
let function_of c i =
  if i < 0 || i >= Array.length c.elf.symbols then None
  else
    let (s : Elf.symbol) = c.elf.symbols.(i) in
    let in_code =
      match s.place with
      | Defined { section; _ } ->
          Option.bind (List.nth_opt c.elf.sections section) (fun sec ->
              kernel_of_section sec.Elf.name)
      | Undefined -> None
    in
    match in_code with
    | Some _ -> in_code
    | None -> if s.name = "" then None else Some s.name

let kernel c name =
  let info = ".nv.info" and own_info = ".nv.info." ^ name in
  (* An attribute's data is the index of the symbol it is about, then its
     value. *)
  let attribute (s : Elf.section) k (param, data) =
    let long = String.length data >= 8 in
    let ours () = function_of c (u32 data 0) = Some name in
    if s.name = own_info && param = eiattr_param_cbank && long then
      { k with params_offset = String.get_uint16_le data 4 }
    else if s.name = info && param = eiattr_min_stack_size && long && ours ()
    then { k with stack_bytes = u32 data 4 }
    else if s.name = info && param = eiattr_regcount && long && ours () then
      { k with registers = u32 data 4 }
    else k
  in
  let section k (s : Elf.section) =
    if s.name = text ^ name then { k with code = s.offset; code_bytes = s.size }
    else if s.name = ".nv.shared." ^ name then { k with shared_bytes = s.size }
    else
      match bank_index name s.name with
      | Some index ->
          let b = { index; offset = s.offset; bytes = s.size } in
          { k with banks = set_bank k.banks b }
      | None when String.starts_with ~prefix:info s.name ->
          List.fold_left (attribute s) k (attributes s)
      | None -> k
  in
  let empty =
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
  if List.mem name (kernels c) then
    Some (List.fold_left section empty c.elf.sections)
  else None
