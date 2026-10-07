(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Cubins: NVRTC's simple_add (fixtures/README.md), and cubins a small ELF
   writer builds, read against the ELF layout below them. Relocation types after
   NVIDIA's cuobjdump: R_CUDA_64 0x2, R_CUDA_ABS32_LO_32 0x38,
   R_CUDA_ABS32_HI_32 0x39. *)

open Windtrap
open Device_nv_abi
module S = Device_nv_abi_support

let strf = Printf.sprintf
let pp_error = Format.pp_print_string

let pp_cubin ppf c =
  Format.fprintf ppf "<cubin: %d bytes, kernels %s>" (Cubin.size c)
    (String.concat ", " (Cubin.kernels c))

let read obj = require_ok ~pp:pp_error (Cubin.of_string obj)
let kernel = Testable.make ~pp:S.pp_kernel ~equal:( = )

(* An ELF writer: a 64-bit little-endian object of EM_CUDA (190) *)

type sh = {
  name : string;
  kind : int;
  flags : int;
  addr : int;
  contents : string;
  size : int;
  link : int;
  info : int;
  align : int;
  entsize : int;
}

let progbits = 1
let symtab_kind = 2
let strtab_kind = 3
let rela_kind = 4
let nobits = 8
let nv_info_kind = 0x7000_0000
let alloc = 0x2
let exec = 0x4
let shn_undef = 0
let shn_abs = 0xfff1

let section ?(kind = progbits) ?(flags = alloc) ?(addr = 0) ?size ?(link = 0)
    ?(info = 0) ?(align = 1) ?(entsize = 0) name contents =
  let size = Option.value size ~default:(String.length contents) in
  { name; kind; flags; addr; contents; size; link; info; align; entsize }

let code ?addr ?(align = 128) name n =
  section ~flags:(alloc lor exec) ?addr ~align (".text." ^ name)
    (String.make n '\x5a')

let names_of names =
  let b = Buffer.create 256 in
  Buffer.add_char b '\000';
  let at n =
    if n = "" then 0
    else begin
      let at = Buffer.length b in
      Buffer.add_string b n;
      Buffer.add_char b '\000';
      at
    end
  in
  let offsets = List.map at names in
  (offsets, Buffer.contents b)

(* The object of [sections] at indexes 1, 2, ..., then its section names. *)
let write sections =
  let offsets, shstrtab =
    names_of (List.map (fun s -> s.name) sections @ [ ".shstrtab" ])
  in
  let sections =
    sections @ [ section ~kind:strtab_kind ~flags:0 ".shstrtab" shstrtab ]
  in
  let body = Buffer.create 4096 in
  let at =
    List.map
      (fun s ->
        let at = 64 + Buffer.length body in
        if s.kind <> nobits then Buffer.add_string body s.contents;
        at)
      sections
  in
  let count = List.length sections + 1 in
  let b = Buffer.create (64 + Buffer.length body + (64 * count)) in
  Buffer.add_string b "\x7fELF\002\001\001";
  Buffer.add_string b (String.make 9 '\000');
  Buffer.add_uint16_le b 2;
  Buffer.add_uint16_le b 190;
  Buffer.add_int32_le b 1l;
  Buffer.add_int64_le b 0L;
  Buffer.add_int64_le b 0L;
  Buffer.add_int64_le b (Int64.of_int (64 + Buffer.length body));
  Buffer.add_int32_le b 0l;
  List.iter (Buffer.add_uint16_le b) [ 64; 0; 0; 64; count; count - 1 ];
  Buffer.add_buffer b body;
  let header name s offset =
    Buffer.add_int32_le b (Int32.of_int name);
    Buffer.add_int32_le b (Int32.of_int s.kind);
    List.iter
      (fun v -> Buffer.add_int64_le b (Int64.of_int v))
      [ s.flags; s.addr; offset; s.size ];
    Buffer.add_int32_le b (Int32.of_int s.link);
    Buffer.add_int32_le b (Int32.of_int s.info);
    Buffer.add_int64_le b (Int64.of_int s.align);
    Buffer.add_int64_le b (Int64.of_int s.entsize)
  in
  header 0 (section ~kind:0 ~flags:0 "" "") 0;
  List.iter2
    (fun s (name, offset) -> header name s offset)
    sections (List.combine offsets at);
  Buffer.contents b

(* A symbol table's section and its names' section, linked to it at [index + 1]:
   each symbol is (name, section index, value), after the null symbol. *)
let symbols ~index syms =
  let offsets, strtab = names_of (List.map (fun (n, _, _) -> n) syms) in
  let b = Buffer.create 256 in
  let add name shndx value =
    Buffer.add_int32_le b (Int32.of_int name);
    Buffer.add_uint8 b 0x12;
    Buffer.add_uint8 b 0;
    Buffer.add_uint16_le b shndx;
    Buffer.add_int64_le b (Int64.of_int value);
    Buffer.add_int64_le b 0L
  in
  add 0 0 0;
  List.iter2 (fun (_, shndx, value) name -> add name shndx value) syms offsets;
  [
    section ~kind:symtab_kind ~flags:0 ~link:(index + 1) ~align:8 ~entsize:24
      ".symtab" (Buffer.contents b);
    section ~kind:strtab_kind ~flags:0 ".strtab" strtab;
  ]

(* A relocation section for the section [info], over the symbols at [link]: each
   relocation is (offset, symbol, type, addend). *)
let rela ~link ~info name relocs =
  let b = Buffer.create 64 in
  List.iter
    (fun (offset, sym, kind, addend) ->
      Buffer.add_int64_le b (Int64.of_int offset);
      Buffer.add_int64_le b
        Int64.(logor (shift_left (of_int sym) 32) (of_int kind));
      Buffer.add_int64_le b (Int64.of_int addend))
    relocs;
  section ~kind:rela_kind ~flags:0 ~link ~info ~align:8 ~entsize:24 name
    (Buffer.contents b)

(* .nv.info attributes: a format byte, a parameter byte and a 16-bit size, then
   for format EIFMT_SVAL (4) that many bytes; other formats hold a 16-bit value
   in place of the size. simple_add_sm89.cubin's are of this form
   (fixtures/README.md). *)
let eiattr_param_cbank = 0x0a
let eiattr_min_stack_size = 0x12
let eiattr_regcount = 0x2f

let attribute fmt param data =
  let h = Bytes.create 4 in
  Bytes.set_uint8 h 0 fmt;
  Bytes.set_uint8 h 1 param;
  Bytes.set_uint16_le h 2 (String.length data);
  Bytes.to_string h ^ data

let u32s vs =
  let b = Bytes.create (4 * List.length vs) in
  List.iteri (fun i v -> Bytes.set_int32_le b (4 * i) (Int32.of_int v)) vs;
  Bytes.to_string b

(* A symbol's attribute: its index, then a 32-bit value. *)
let of_symbol param sym v = attribute 4 param (u32s [ sym; v ])

(* EIATTR_PARAM_CBANK: a symbol's index, then the parameters' offset and size in
   16 bits each. *)
let param_cbank sym offset size =
  let d = Bytes.create 8 in
  Bytes.set_int32_le d 0 (Int32.of_int sym);
  Bytes.set_uint16_le d 4 offset;
  Bytes.set_uint16_le d 6 size;
  attribute 4 eiattr_param_cbank (Bytes.to_string d)

(* Attributes the reader passes over: a 16-bit value (EIFMT_HVAL 3) and data of
   a parameter it does not read (0x37, EIATTR_CUDA_API_VERSION). *)
let fillers = attribute 3 0x19 "" ^ attribute 4 0x37 (u32s [ 0x80 ])

let nv_info ?(name = ".nv.info") ?(info = 0) contents =
  section ~kind:nv_info_kind ~flags:0 ~info ~align:4 name contents

(* The ELF layout below a cubin *)

let layout obj = require_ok ~pp:pp_error (Device_elf.of_string ~align:128 obj)
let page = 4096
let round_up n a = (n + a - 1) / a * a

(* simple_add, as llvm-readelf reads it *)

let simple_add () =
  read
    (In_channel.with_open_bin "fixtures/simple_add_sm89.cubin"
       In_channel.input_all)

let nvrtc =
  group ~timeout:10. "simple_add_sm89"
    [
      test "its one kernel is simple_add" (fun () ->
          equal (list string) [ "simple_add" ] (Cubin.kernels (simple_add ())));
      test "simple_add's code, registers, stack, parameters and bank 0"
        (fun () ->
          equal (option kernel)
            (Some
               {
                 Cubin.code = 0x180;
                 code_bytes = 0x200;
                 registers = 12;
                 shared_bytes = 0;
                 stack_bytes = 0;
                 params_offset = 0x160;
                 banks = [ { index = 0; offset = 0; bytes = 0x17c } ];
               })
            (Cubin.kernel (simple_add ()) "simple_add"));
      test "its image of 0x380 bytes takes two pages" (fun () ->
          equal int (2 * page) (Cubin.size (simple_add ())));
      test "a relocation of its debugging information patches nothing"
        (fun () ->
          equal
            (list (pair int string))
            []
            (Cubin.patches (simple_add ()) ~base:0x7000_0000_0000));
      test "a kernel the cubin lacks is none" (fun () ->
          equal (option kernel) None (Cubin.kernel (simple_add ()) "simple"));
    ]

(* Relocations

   A cubin of a bank and a kernel's code, with symbols of every place and
   relocations of any type, at any offset of the code. *)

let r_cuda_64 = 0x2
let r_cuda_abs32_lo_32 = 0x38
let r_cuda_abs32_hi_32 = 0x39

type relocatable = {
  bank : int;  (** bytes of .nv.constant0.k *)
  text : int;  (** bytes of .text.k *)
  in_text : int;  (** the offset of a symbol in the code *)
  in_bank : int;  (** the offset of a symbol in the bank *)
  relocs : (int * int * int * int) list;
      (** offset in the code, symbol, type, addend *)
}

(* The symbols: k in the code, c in the bank, u undefined, a absolute, s in a
   section the image does not hold. *)
let relocatable r =
  let syms =
    [
      ("k", 2, r.in_text);
      ("c", 1, r.in_bank);
      ("u", shn_undef, 0);
      ("a", shn_abs, 0x1234);
      ("s", 3, 0);
    ]
  in
  write
    ([
       section ~align:4 ".nv.constant0.k" (String.make r.bank '\001');
       code "k" r.text;
       section ~kind:nobits ~flags:(alloc lor 1) ~size:16 ".nv.shared.k" "";
     ]
    @ symbols ~index:4 syms
    @ [ rela ~link:4 ~info:2 ".rela.text.k" r.relocs ])

let pp_relocatable ppf r =
  Format.fprintf ppf
    "{ bank = %d; text = %d; in_text = %d; in_bank = %d; relocs = [%s] }" r.bank
    r.text r.in_text r.in_bank
    (String.concat "; "
       (List.map
          (fun (o, s, k, a) -> strf "(%d, %d, 0x%x, %d)" o s k a)
          r.relocs))

(* Relocatable cubins; [valid] ones have relocations of the three types, to
   symbols in the image, whose bytes lie in it. *)
let relocatable_gen ~valid =
  let open Gen in
  let addend =
    frequency
      [
        (3, int_range (-0x1000) 0x1000);
        (1, of_list ~pp:Format.pp_print_int [ -(1 lsl 48); 1 lsl 48; -1 ]);
      ]
  in
  let kinds = [ r_cuda_64; r_cuda_abs32_lo_32; r_cuda_abs32_hi_32 ] in
  let gen =
    let* bank, text =
      pair (map (fun n -> 4 * n) (int_range 1 16)) (int_range 8 600)
    in
    let last = if valid then text - 8 else text - 1 in
    let+ in_text = int_range 0 text
    and+ in_bank = int_range 0 bank
    and+ relocs =
      list ~size:(int_range 0 6)
        (let+ offset =
           frequency
             [
               (3, int_range 0 last); (1, int_range (Int.max 0 (last - 8)) last);
             ]
         and+ sym = int_range 1 (if valid then 2 else 5)
         and+ kind =
           if valid then of_list ~pp:Format.pp_print_int kinds
           else
             frequency
               [
                 (6, of_list ~pp:Format.pp_print_int kinds);
                 (1, of_list ~pp:Format.pp_print_int [ 0x1; 0x3a; 0x5 ]);
               ]
         and+ addend = addend in
         (offset, sym, kind, addend))
    in
    { bank; text; in_text; in_bank; relocs }
  in
  with_pp pp_relocatable gen

let any_cubin = relocatable_gen ~valid:false
let valid_cubin = relocatable_gen ~valid:true

let base =
  Gen.frequency
    [
      (3, Gen.int_range 0 ((1 lsl 49) - 1));
      (1, Gen.of_list ~pp:Format.pp_print_int [ 0; (1 lsl 49) - 1 ]);
    ]

let le64 n =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 n;
  Bytes.to_string b

(* What a relocation writes, as cubin.mli states it, over the ELF layout: the
   symbol's address base + offset + addend, modulo 2^64. *)
let expected_patch ~base (r : Device_elf.relocation) =
  match r.symbol.place with
  | Image { offset; _ } ->
      let address =
        Int64.(add (add (of_int base) (of_int offset)) (of_int r.addend))
      in
      let w = le64 address in
      if r.kind = r_cuda_64 then Some (r.offset, w)
      else if r.kind = r_cuda_abs32_lo_32 then
        Some (r.offset + 4, String.sub w 0 4)
      else if r.kind = r_cuda_abs32_hi_32 then
        Some (r.offset + 4, String.sub w 4 4)
      else None
  | Undefined | Absolute _ | Outside _ -> None

(* Why a cubin is refused, if it is. *)
let refusal (o : Device_elf.t) =
  let bad (r : Device_elf.relocation) =
    match expected_patch ~base:0 r with
    | None ->
        if List.mem r.kind [ r_cuda_64; r_cuda_abs32_lo_32; r_cuda_abs32_hi_32 ]
        then Some "a symbol outside the image"
        else Some "a relocation type"
    | Some (at, w) ->
        if at + String.length w > o.size then Some "bytes past the image"
        else None
  in
  List.find_map bad o.relocations

let relocations =
  group ~timeout:10. "relocations"
    [
      prop ~count:300
        "a cubin reads unless a relocation is of another type, to a symbol \
         outside the image or past its end"
        any_cubin (fun r ->
          let obj = relocatable r in
          let o = layout obj in
          match (refusal o, Cubin.of_string obj) with
          | None, Ok _ -> cover "read" true
          | Some why, Error _ -> cover ("refused for " ^ why) true
          | None, Error e -> failf "refused: %s" e
          | Some why, Ok _ -> failf "read, with %s" why);
      prop ~count:300 "patches write each symbol's address, in relocation order"
        (Gen.pair valid_cubin base) (fun (r, base) ->
          let obj = relocatable r in
          let c = read obj and o = layout obj in
          cover "a patch" (o.relocations <> []);
          cover "an address that wraps below zero"
            (List.exists
               (fun (r : Device_elf.relocation) ->
                 match expected_patch ~base r with
                 | Some (_, w) when r.kind = r_cuda_64 ->
                     String.get_int64_le w 0 < 0L
                 | _ -> false)
               o.relocations);
          equal
            (list (pair int string))
            (List.filter_map (expected_patch ~base) o.relocations)
            (Cubin.patches c ~base));
      xfail
        ~reason:
          "patches computes base + offset + addend in 63-bit integers, so a \
           sum past 2^62 wraps modulo 2^63"
        (test "a symbol's address past 2^62 is taken modulo 2^64" (fun () ->
             let obj =
               relocatable
                 {
                   bank = 4;
                   text = 16;
                   in_text = 0;
                   in_bank = 0;
                   relocs = [ (0, 1, r_cuda_64, max_int) ];
                 }
             in
             let base = 0x7000_0000_0000 in
             equal
               (list (pair int string))
               (List.filter_map (expected_patch ~base) (layout obj).relocations)
               (Cubin.patches (read obj) ~base)));
      prop ~count:300 "every patch lies in the image of the ELF object"
        (Gen.pair valid_cubin base) (fun (r, base) ->
          let c = read (relocatable r) in
          List.iter
            (fun (at, p) ->
              at_least int ~than:0 at;
              at_most int ~than:(Cubin.elf c).size (at + String.length p))
            (Cubin.patches c ~base));
    ]

(* A cubin of 128 kernels, each scaling by a global the cubin relocates
   (many.cu), checked against the ELF layout. *)

let many () =
  In_channel.with_open_bin "fixtures/many_sm89.cubin" In_channel.input_all

let many_sm89 =
  group ~timeout:10. "many_sm89"
    [
      test "its kernels are its code sections, in order" (fun () ->
          let o = layout (many ()) in
          let code (s : Device_elf.section) =
            match s.offset with
            | Some _ when String.starts_with ~prefix:".text." s.name ->
                Some (String.sub s.name 6 (String.length s.name - 6))
            | _ -> None
          in
          let names = List.filter_map code (Iarray.to_list o.sections) in
          equal ~msg:"count" int 128 (List.length names);
          equal (list string) names (Cubin.kernels (read (many ()))));
      test "its 128 relocations write each global's address" (fun () ->
          let obj = many () in
          let base = 0x7fff_0000_0000 in
          let expected =
            List.filter_map (expected_patch ~base) (layout obj).relocations
          in
          equal ~msg:"count" int 128 (List.length expected);
          equal
            (list (pair int string))
            expected
            (Cubin.patches (read obj) ~base));
    ]

(* The image *)

(* A cubin of a 16-byte bank at address 0 and 16 bytes of code at [at]: an image
   of [at + 16] bytes. *)
let spanning at =
  write
    [
      section ~align:1 ".nv.constant0.k" (String.make 16 '\001');
      code ~addr:at ~align:1 "k" 16;
    ]

let image =
  group ~timeout:10. "image"
    [
      prop "the image is the ELF image, then zeros to a page and a page more"
        valid_cubin (fun r ->
          let c = read (relocatable r) in
          equal int (round_up (Cubin.elf c).size page + page) (Cubin.size c));
      test "an ELF image of a whole page takes two" (fun () ->
          let c =
            read
              (relocatable
                 {
                   bank = 4;
                   text = page - 128;
                   in_text = 0;
                   in_bank = 0;
                   relocs = [];
                 })
          in
          equal (pair int int)
            (page, 2 * page)
            ((Cubin.elf c).size, Cubin.size c));
      test "a cubin without allocated sections takes a page" (fun () ->
          let c = read (write [ nv_info "" ]) in
          equal (pair int int) (0, page) ((Cubin.elf c).size, Cubin.size c));
      test "an image of 2^49 bytes reads" (fun () ->
          let c = read (spanning ((1 lsl 49) - page - 16)) in
          equal int (1 lsl 49) (Cubin.size c));
      xfail
        ~reason:
          "of_string bounds the ELF image by 2^49, so a cubin of 2^49 + 4 KiB \
           bytes reads"
        (test "an image past 2^49 bytes is refused" (fun () ->
             let obj = spanning ((1 lsl 49) - page - 15) in
             equal ~msg:"the ELF image" int
               ((1 lsl 49) - page + 1)
               (layout obj).size;
             ignore (require_error ~pp:pp_cubin (Cubin.of_string obj))));
      test "an object that is not ELF is refused" (fun () ->
          ignore (require_error ~pp:pp_cubin (Cubin.of_string "\x7fELF")));
      test "reading copies nothing: elf's file is the object" (fun () ->
          let obj =
            relocatable
              { bank = 4; text = 16; in_text = 0; in_bank = 0; relocs = [] }
          in
          equal string obj (Cubin.elf (read obj)).file);
    ]

(* Kernels

   Cubins of up to three kernels with banks of their own and banks of the cubin,
   shared memory, and attributes naming their functions by symbol. *)

type target =
  | Section_symbol of int  (** the nameless symbol of kernel i's code *)
  | Named_in of int * string  (** a symbol of this name in kernel i's code *)
  | Undefined of string  (** a symbol of this name the cubin does not define *)

type kdesc = {
  kname : string;
  text : int;
  shared : int option;
  cbank : int option;
  own_regcount : int option;  (** in its own .nv.info.name, which is ignored *)
}

type desc = {
  kernels : kdesc list;
  banks : (int * int option * int) list;
      (** bank index, its kernel (None for the cubin's), bytes; in section order
      *)
  attrs : (int * target * int) list;  (** parameter, function, value *)
  dead : bool;  (** a .text.dead the image does not hold *)
  truncated : bool;  (** .nv.info ends inside an attribute *)
}

let pp_target ppf = function
  | Section_symbol i -> Format.fprintf ppf "Section_symbol %d" i
  | Named_in (i, n) -> Format.fprintf ppf "Named_in (%d, %S)" i n
  | Undefined n -> Format.fprintf ppf "Undefined %S" n

let pp_desc ppf d =
  let pp_opt ppf = function
    | None -> Format.pp_print_string ppf "None"
    | Some n -> Format.fprintf ppf "Some %d" n
  in
  Format.fprintf ppf "@[<v>";
  List.iter
    (fun k ->
      Format.fprintf ppf "%s: text %d, shared %a, cbank %a, own regcount %a@,"
        k.kname k.text pp_opt k.shared pp_opt k.cbank pp_opt k.own_regcount)
    d.kernels;
  List.iter
    (fun (i, k, n) ->
      Format.fprintf ppf "bank %d of %a, %d bytes@," i pp_opt k n)
    d.banks;
  List.iter
    (fun (p, t, v) -> Format.fprintf ppf "attr 0x%x %a %d@," p pp_target t v)
    d.attrs;
  Format.fprintf ppf "dead %b, truncated %b@]" d.dead d.truncated

(* Section indexes: the kernels' code from 1, then banks, shared memory,
   .nv.info sections, a dead code section, and the symbols. *)
let assemble d =
  let n = List.length d.kernels in
  let texts = List.map (fun k -> code k.kname k.text) d.kernels in
  let name_of = function
    | None -> ""
    | Some i -> "." ^ (List.nth d.kernels i).kname
  in
  let banks =
    List.map
      (fun (i, k, bytes) ->
        section ~align:4
          (strf ".nv.constant%d%s" i (name_of k))
          (String.make bytes '\002'))
      d.banks
  in
  let shared =
    List.filter_map
      (fun k ->
        Option.map
          (fun size ->
            section ~kind:nobits ~flags:(alloc lor 1) ~size
              (".nv.shared." ^ k.kname) "")
          k.shared)
      d.kernels
  in
  (* Symbols: each kernel's section symbol at 2i+1, its function at 2i+2, then
     the attributes' own. *)
  let extra =
    List.filter_map
      (function
        | _, Named_in (i, s), _ -> Some (s, 1 + i, 0)
        | _, Undefined s, _ -> Some (s, shn_undef, 0)
        | _, Section_symbol _, _ -> None)
      d.attrs
  in
  let syms =
    List.concat
      (List.mapi (fun i k -> [ ("", 1 + i, 0); (k.kname, 1 + i, 0) ]) d.kernels)
    @ extra
  in
  let index_of =
    let next = ref ((2 * n) + 1) in
    List.map
      (fun (_, t, _) ->
        match t with
        | Section_symbol i -> (2 * i) + 1
        | Named_in _ | Undefined _ ->
            let i = !next in
            incr next;
            i)
      d.attrs
  in
  let info =
    String.concat ""
      (fillers
      :: List.map2 (fun (p, _, v) sym -> of_symbol p sym v) d.attrs index_of)
    ^
    if d.truncated then String.sub (of_symbol eiattr_regcount 1 99) 0 8 else ""
  in
  let own =
    List.mapi
      (fun i k ->
        nv_info ~name:(".nv.info." ^ k.kname) ~info:(1 + i)
          (fillers
          ^ (match k.cbank with
            | Some off -> param_cbank ((2 * i) + 1) off 0x1c
            | None -> "")
          ^
          match k.own_regcount with
          | Some v -> of_symbol eiattr_regcount ((2 * i) + 1) v
          | None -> ""))
      d.kernels
  in
  let dead =
    if d.dead then [ section ~flags:0 ".text.dead" (String.make 16 '\000') ]
    else []
  in
  let before = texts @ banks @ shared @ (nv_info info :: own) @ dead in
  write (before @ symbols ~index:(List.length before + 1) syms)

let desc_gen =
  let open Gen in
  let gen =
    let* n = int_range 1 3 in
    let names = List.init n (strf "k%d") in
    let* kernels =
      List.fold_right
        (fun kname acc ->
          let+ text = int_range 1 300
          and+ shared = option (int_range 0 0xc000)
          and+ cbank = option (int_range 0 0x1000)
          and+ own_regcount = option (int_range 1 255)
          and+ rest = acc in
          { kname; text; shared; cbank; own_regcount } :: rest)
        names (constant [])
    in
    let target =
      one_of
        [
          map (fun i -> Section_symbol i) (int_range 0 (n - 1));
          (let+ i = int_range 0 (n - 1)
           and+ s =
             of_list ~pp:Format.pp_print_string [ "k0"; "k1"; "k2"; "zz" ]
           in
           Named_in (i, s));
          map
            (fun s -> Undefined s)
            (of_list ~pp:Format.pp_print_string [ "k0"; "k1"; "k2"; "zz" ]);
        ]
    in
    let+ banks =
      list ~size:(int_range 0 5)
        (let+ i = int_range 0 4
         and+ k = option (int_range 0 (n - 1))
         and+ bytes = int_range 0 0x200 in
         (i, k, bytes))
    and+ attrs =
      list ~size:(int_range 0 6)
        (let+ p =
           of_list ~pp:Format.pp_print_int
             [ eiattr_regcount; eiattr_min_stack_size ]
         and+ t = target
         and+ v = int_range 0 0xffff in
         (p, t, v))
    and+ dead = bool
    and+ truncated = bool in
    { kernels; banks; attrs; dead; truncated }
  in
  with_pp pp_desc gen

(* A kernel's record as cubin.mli states it, its offsets from the ELF layout. *)
let expected (o : Device_elf.t) d i =
  let k = List.nth d.kernels i in
  let offset name =
    let s =
      List.find
        (fun (s : Device_elf.section) -> s.name = name)
        (Iarray.to_list o.sections)
    in
    Option.get s.offset
  in
  let resolves = function
    | Section_symbol j -> j = i
    | Named_in (j, _) -> j = i
    | Undefined s -> s = k.kname
  in
  let last p =
    List.fold_left
      (fun acc (p', t, v) -> if p' = p && resolves t then v else acc)
      0 d.attrs
  in
  (* Banks: the cubin's and the kernel's, in section order; the last of an index
     gives its offset and size, the first gives its place. *)
  let sections =
    List.mapi (fun j b -> (j, b)) d.banks
    |> List.filter (fun (_, (_, owner, _)) -> owner = None || owner = Some i)
  in
  let image_offset j =
    Option.get (Iarray.get o.sections (1 + List.length d.kernels + j)).offset
  in
  let banks =
    List.fold_left
      (fun acc (j, (idx, _, bytes)) ->
        let bank = { Cubin.index = idx; offset = image_offset j; bytes } in
        if List.exists (fun (x : Cubin.bank) -> x.index = idx) acc then
          List.map
            (fun (x : Cubin.bank) -> if x.index = idx then bank else x)
            acc
        else acc @ [ bank ])
      [] sections
  in
  {
    Cubin.code = offset (".text." ^ k.kname);
    code_bytes = k.text;
    registers = last eiattr_regcount;
    shared_bytes = Option.value k.shared ~default:0;
    stack_bytes = last eiattr_min_stack_size;
    params_offset = Option.value k.cbank ~default:0;
    banks;
  }

let kernels =
  group ~timeout:10. "kernels"
    [
      prop ~count:300
        "a kernel's record is what its sections and attributes say" desc_gen
        (fun d ->
          let obj = assemble d in
          let o = layout obj and c = read obj in
          cover "an attribute by a symbol's name"
            (List.exists
               (function _, Undefined _, _ -> true | _ -> false)
               d.attrs);
          cover "a function named after another kernel"
            (List.exists
               (function
                 | _, Named_in (i, s), _ -> s <> (List.nth d.kernels i).kname
                 | _ -> false)
               d.attrs);
          cover "a bank of the cubin and of a kernel"
            (List.exists (fun (_, k, _) -> k = None) d.banks
            && List.exists (fun (_, k, _) -> k <> None) d.banks);
          List.iteri
            (fun i k ->
              equal ~msg:k.kname (option kernel)
                (Some (expected o d i))
                (Cubin.kernel c k.kname))
            d.kernels);
      prop "kernels are the code sections the image holds, in order" desc_gen
        (fun d ->
          let c = read (assemble d) in
          equal (list string)
            (List.map (fun k -> k.kname) d.kernels)
            (Cubin.kernels c);
          equal ~msg:"a code section outside the image" (option kernel) None
            (Cubin.kernel c "dead"));
      test "a cubin without code has no kernel" (fun () ->
          let c = read (write [ nv_info "" ]) in
          equal
            (pair (list string) (option kernel))
            ([], None)
            (Cubin.kernels c, Cubin.kernel c ""));
    ]

let () =
  exit
    (run "device_nv_abi.cubin"
       [ nvrtc; many_sm89; relocations; image; kernels ])
