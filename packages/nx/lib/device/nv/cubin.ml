(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Cubins: the ELF objects of NVIDIA's compilers, laid out as the GPU runs them,
   with the address of each function's code. *)

module Elf = Nx_device_elf

type t = {
  image : string; (* laid out, with room after it for the GPU's prefetch *)
  relocations : (int * int * int) list;
      (* (image offset to patch, target offset plus addend, type) *)
  entry : int; (* offsets are in the image *)
}

let round_up n a = (n + a - 1) / a * a

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
  let relocations =
    List.map
      (fun (r : Elf.relocation) ->
        if not (List.mem r.kind [ 2; 0x38; 0x39 ]) then
          failwith
            (Printf.sprintf "the cubin has a relocation of unknown type 0x%x"
               r.kind);
        match r.target with
        | Offset target -> (r.at, target + r.addend, r.kind)
        | Undefined s ->
            failwith
              (Printf.sprintf "the cubin refers to an undefined symbol %s" s))
      o.relocations
  in
  let image =
    o.image
    ^ String.make
        (round_up (String.length o.image) 0x1000
        + 0x1000 - String.length o.image)
        '\000'
  in
  { image; relocations; entry = text.offset }

(* [c]'s image with its relocations applied for its upload at [base]: the 64-bit
   address of a symbol, or its low or high 32 bits in the word after. *)
let relocate c ~base =
  let b = Bytes.of_string c.image in
  List.iter
    (fun (at, target, kind) ->
      let v = base + target in
      match kind with
      | 2 -> Bytes.set_int64_le b at (Int64.of_int v)
      | 0x38 ->
          Bytes.set_int32_le b (at + 4) (Int32.of_int (v land 0xffff_ffff))
      | _ ->
          Bytes.set_int32_le b (at + 4)
            (Int32.of_int ((v lsr 32) land 0xffff_ffff)))
    c.relocations;
  Bytes.to_string b
