(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Cubins: the ELF objects of NVIDIA's compilers, laid out as the GPU runs them,
   with what a launch of one of their functions needs. *)

module Elf = Nx_device_support.Elf

type t = {
  image : string; (* laid out, with room after it for the GPU's prefetch *)
  relocations : Elf.relocation list;
  entry : int; (* offsets are in the image *)
  code_bytes : int;
  registers : int;
  shared_bytes : int;
  local_bytes : int;
  param_offset : int;
  banks : (int * int * int) list; (* (index, offset, bytes) *)
  max_threads : int;
}

let round_up n a = (n + a - 1) / a * a

(* The attributes of an [.nv.info] section: (parameter, value) of each, where a
   value is its payload or, for attributes without one, its size field. *)
let attributes s =
  let rec go off acc =
    if off + 4 > String.length s then List.rev acc
    else
      let fmt = Char.code s.[off] and param = Char.code s.[off + 1] in
      let size = String.get_uint16_le s (off + 2) in
      if fmt = 0x4 then
        go (off + 4 + size) ((param, `Data (String.sub s (off + 4) size)) :: acc)
      else go (off + 4) ((param, `Size size) :: acc)
  in
  go 0 []

(* nv.info attributes *)
let eiattr_param_cbank = 0xa
let eiattr_min_stack_size = 0x12
let eiattr_regcount = 0x2f

(* The index of a bank of constants, [.nv.constantN...]. *)
let bank s =
  let p = ".nv.constant" in
  let n = String.length p in
  if String.length s <= n || String.sub s 0 n <> p then None
  else
    let rest = String.sub s n (String.length s - n) in
    let digits =
      match String.index_opt rest '.' with
      | Some i -> String.sub rest 0 i
      | None -> rest
    in
    int_of_string_opt digits

let u32 s off = Int32.to_int (String.get_int32_le s off) land 0xffff_ffff

let load binary ~name =
  let o = Elf.load ~align:128 binary in
  let text =
    match
      List.find_opt
        (fun (s : Elf.section) -> s.name = ".text." ^ name)
        o.sections
    with
    | Some s -> s
    | None -> failwith (Printf.sprintf "the cubin has no function %s" name)
  in
  let shared =
    List.fold_left
      (fun n (s : Elf.section) ->
        if s.name = ".nv.shared." ^ name then round_up (0x400 + s.size) 128
        else n)
      0x400 o.sections
  in
  let banks =
    List.fold_left
      (fun banks (s : Elf.section) ->
        match bank s.name with
        | Some i ->
            (i, s.offset, s.size) :: List.filter (fun (j, _, _) -> j <> i) banks
        | None -> banks)
      [ (0, 0, 0x160) ]
      o.sections
  in
  let registers = ref 0 and stack = ref 0 and param = ref 0 in
  List.iter
    (fun (s : Elf.section) ->
      List.iter
        (function
          | p, `Data d
            when s.name = ".nv.info." ^ name && p = eiattr_param_cbank ->
              param := String.get_uint16_le d 4
          | p, `Data d when s.name = ".nv.info" && p = eiattr_min_stack_size ->
              stack := u32 d 4
          | p, `Data d when s.name = ".nv.info" && p = eiattr_regcount ->
              registers := u32 d 4
          | _ -> ())
        (if String.starts_with ~prefix:".nv.info" s.name then
           attributes s.contents
         else []))
    o.sections;
  List.iter
    (fun (r : Elf.relocation) ->
      if not (List.mem r.kind [ 2; 0x38; 0x39 ]) then
        failwith
          (Printf.sprintf "the cubin has a relocation of unknown type 0x%x"
             r.kind))
    o.relocations;
  let image =
    o.image
    ^ String.make
        (round_up (String.length o.image) 0x1000
        + 0x1000 - String.length o.image)
        '\000'
  in
  {
    image;
    relocations = o.relocations;
    entry = text.offset;
    code_bytes = text.size;
    registers = !registers;
    shared_bytes = shared;
    local_bytes = !stack + 0x240;
    param_offset = !param;
    banks;
    max_threads = 65536 / round_up (Int.max 1 !registers * 32) 256 / 4 * 4 * 32;
  }

(* [c]'s image with its relocations applied for its upload at [base]: the 64-bit
   address of a symbol, or its low or high 32 bits in the word after. *)
let relocate c ~base =
  let b = Bytes.of_string c.image in
  List.iter
    (fun (r : Elf.relocation) ->
      let v = base + r.target + r.addend in
      match r.kind with
      | 2 -> Bytes.set_int64_le b r.at (Int64.of_int v)
      | 0x38 ->
          Bytes.set_int32_le b (r.at + 4) (Int32.of_int (v land 0xffff_ffff))
      | _ ->
          Bytes.set_int32_le b (r.at + 4)
            (Int32.of_int ((v lsr 32) land 0xffff_ffff)))
    c.relocations;
  Bytes.to_string b
