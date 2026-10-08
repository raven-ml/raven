(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ELF objects.

   Host programs, GPU code and firmware come as ELF objects. Reading one lays
   out its image: the sections a loader's memory holds, each at an offset of the
   image, with symbols and relocations given in image offsets. A loader writes
   the image at some address and patches each relocation by its machine's
   formula.

   The object here is [poly.c] compiled for x86_64; the example reads it the
   same way on any machine. *)

let strf = Printf.sprintf

let place (p : Rig_elf.place) =
  match p with
  | Undefined -> "undefined"
  | Absolute v -> strf "absolute %d" v
  | Image { section; offset } -> strf "image 0x%x (section %d)" offset section
  | Outside s -> strf "outside the image (section %d)" s

(* The image as bytes: each section the image holds at its offset, zeros
   elsewhere. *)
let image (o : Rig_elf.t) =
  let b = Bytes.make o.size '\000' in
  let put (s : Rig_elf.section) =
    match s.offset with
    | Some off -> Bytes.blit_string o.file s.at b off s.length
    | None -> ()
  in
  Iarray.iter put o.sections;
  b

let () =
  let obj = In_channel.with_open_bin "poly_x86_64.o" In_channel.input_all in
  let o = Result.get_ok (Rig_elf.of_string obj) in
  Printf.printf
    "%d-bit, type %d, machine %d: an image of %d bytes, aligned to %d\n\n"
    o.bits o.kind o.machine o.size o.align;

  print_endline "sections in the image:";
  Iarray.iter
    (fun (s : Rig_elf.section) ->
      match s.offset with
      | Some off ->
          Printf.printf "  %-8s at 0x%02x, %3d bytes\n" s.name off s.size
      | None -> ())
    o.sections;

  print_endline "named symbols:";
  Iarray.iter
    (fun (s : Rig_elf.symbol) ->
      if s.name <> "" then Printf.printf "  %-8s %s\n" s.name (place s.place))
    o.symbols;

  (* Both kinds here are relative to the place P they patch: S + A - P, with S
     the symbol's address and A the addend. Image offsets give it without
     knowing where the image will lie. *)
  print_endline "relocations (S + A - P):";
  List.iter
    (fun (r : Rig_elf.relocation) ->
      let kind =
        match r.kind with 2 -> "PC32" | 4 -> "PLT32" | k -> strf "%d" k
      in
      match (r.symbol.place, r.addend) with
      | Image { offset = s; _ }, Explicit a ->
          Printf.printf "  at 0x%02x, %-5s to %s: %d\n" r.offset kind
            (place r.symbol.place)
            (s + a - r.offset)
      | _ ->
          Printf.printf "  at 0x%02x, %-5s to %s\n" r.offset kind
            (place r.symbol.place))
    o.relocations;

  (* A loader looks its entry up by name, and writes the image. *)
  let entry = Option.get (Rig_elf.symbol o "poly") in
  let bytes = image o in
  Printf.printf "\nentry poly at 0x%x; its first byte 0x%02x\n" entry
    (Char.code (Bytes.get bytes entry));

  (* Anything else is refused with a reason. *)
  match Rig_elf.of_string "not an object" with
  | Ok _ -> ()
  | Error why -> print_endline why
