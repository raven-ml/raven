(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk

let uname flag =
  try
    let ic = Unix.open_process_in ("uname " ^ flag) in
    let value = input_line ic in
    let _ = Unix.close_process_in ic in
    String.trim value
  with _ -> ""

let host_arch () = uname "-m"
let cc () = match Sys.getenv_opt "CC" with Some cc -> cc | None -> "clang"

let read_file path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr ic)
    (fun () ->
      let len = in_channel_length ic in
      really_input_string ic len)

let compile_c src =
  let arch = host_arch () in
  let arch_flag =
    match arch with
    | "x86_64" | "AMD64" -> "-march=native"
    | "riscv64" -> "-march=rv64g"
    | _ -> "-mcpu=native"
  in
  let src_path = Filename.temp_file "tolk_elf" ".c" in
  let obj_path = Filename.temp_file "tolk_elf" ".o" in
  let err_path = Filename.temp_file "tolk_elf" ".err" in
  Fun.protect
    ~finally:(fun () ->
      List.iter
        (fun path -> try Sys.remove path with Sys_error _ -> ())
        [ src_path; obj_path; err_path ])
    (fun () ->
      let oc = open_out_bin src_path in
      output_string oc src;
      close_out oc;
      let command =
        String.concat " "
          [
            Filename.quote (cc ());
            "-c"; "-x"; "c"; arch_flag;
            Filename.quote (Printf.sprintf "--target=%s-none-unknown-elf" arch);
            "-O2"; "-fPIC"; "-ffreestanding"; "-fno-math-errno";
            "-nostdlib"; "-fno-ident";
            Filename.quote src_path; "-o"; Filename.quote obj_path;
            "2>"; Filename.quote err_path;
          ]
      in
      match Sys.command command with
      | 0 -> Bytes.of_string (read_file obj_path)
      | _ ->
          let err = read_file err_path in
          failwith
            (if String.equal err "" then "clang failed"
             else "clang failed:\n" ^ err))

let load_c src = Elf.load (compile_c src)

let require_section elf name =
  match Elf.find_section elf name with
  | Some s -> s
  | None -> failwith ("expected " ^ name ^ " section")

let set16 b off v = Bytes.set_uint16_le b off v
let set32 b off v = Bytes.set_int32_le b off (Int32.of_int v)
let set64 b off v = Bytes.set_int64_le b off (Int64.of_int v)

(* Hand-crafted 64-bit little-endian shared object (ET_DYN), shaped like
   linker output: program data at fixed addresses ([.text] at 0x100,
   [.rodata] at 0x40), a non-allocatable [.comment] without one, an
   allocatable [.note] that must stay out of the image, and a relocation
   against the symbol table. *)
let et_dyn_fixture () =
  let text = Bytes.of_string "TEXTSECT" in
  let rodata = Bytes.of_string "RODA" in
  let note = Bytes.of_string "NOTE" in
  let comment = Bytes.of_string "cmnt!" in
  let symtab = Bytes.make 48 '\000' in
  set32 symtab 24 1 (* st_name: "kern" *);
  set16 symtab 30 1 (* st_shndx: .text *);
  set64 symtab 32 4 (* st_value *);
  let strtab = Bytes.of_string "\000kern\000" in
  let rela = Bytes.make 24 '\000' in
  set64 rela 0 4 (* r_offset *);
  set64 rela 8 ((1 lsl 32) lor 0x11) (* symbol 1, type 0x11 *);
  set64 rela 16 8 (* r_addend *);
  let buf = Buffer.create 1024 in
  Buffer.add_bytes buf (Bytes.make 64 '\000');
  let pad_to align =
    while Buffer.length buf mod align <> 0 do
      Buffer.add_char buf '\000'
    done
  in
  let add align content =
    pad_to align;
    let off = Buffer.length buf in
    Buffer.add_bytes buf content;
    off
  in
  let off_text = add 8 text in
  let off_rodata = add 1 rodata in
  let off_note = add 4 note in
  let off_comment = add 1 comment in
  let off_symtab = add 8 symtab in
  let off_strtab = add 1 strtab in
  let off_rela = add 8 rela in
  let shstr = Buffer.create 64 in
  Buffer.add_char shstr '\000';
  let name s =
    let off = Buffer.length shstr in
    Buffer.add_string shstr s;
    Buffer.add_char shstr '\000';
    off
  in
  let n_text = name ".text" in
  let n_rodata = name ".rodata" in
  let n_note = name ".note" in
  let n_comment = name ".comment" in
  let n_symtab = name ".symtab" in
  let n_strtab = name ".strtab" in
  let n_rela = name ".rela.text" in
  let n_shstrtab = name ".shstrtab" in
  let off_shstr = add 1 (Buffer.to_bytes shstr) in
  pad_to 8;
  let e_shoff = Buffer.length buf in
  let shdr ~nm ~ty ~flags ~addr ~off ~size ~link ~info ~salign ~entsize =
    let b = Bytes.make 64 '\000' in
    set32 b 0 nm;
    set32 b 4 ty;
    set64 b 8 flags;
    set64 b 16 addr;
    set64 b 24 off;
    set64 b 32 size;
    set32 b 40 link;
    set32 b 44 info;
    set64 b 48 salign;
    set64 b 56 entsize;
    Buffer.add_bytes buf b
  in
  shdr ~nm:0 ~ty:0 ~flags:0 ~addr:0 ~off:0 ~size:0 ~link:0 ~info:0 ~salign:0
    ~entsize:0;
  shdr ~nm:n_text ~ty:1 ~flags:0x6 ~addr:0x100 ~off:off_text ~size:8 ~link:0
    ~info:0 ~salign:16 ~entsize:0;
  shdr ~nm:n_rodata ~ty:1 ~flags:0x2 ~addr:0x40 ~off:off_rodata ~size:4 ~link:0
    ~info:0 ~salign:4 ~entsize:0;
  shdr ~nm:n_note ~ty:7 ~flags:0x2 ~addr:0x10 ~off:off_note ~size:4 ~link:0
    ~info:0 ~salign:4 ~entsize:0;
  shdr ~nm:n_comment ~ty:1 ~flags:0x30 ~addr:0 ~off:off_comment ~size:5 ~link:0
    ~info:0 ~salign:8 ~entsize:0;
  shdr ~nm:n_symtab ~ty:2 ~flags:0 ~addr:0 ~off:off_symtab ~size:48 ~link:6
    ~info:1 ~salign:8 ~entsize:24;
  shdr ~nm:n_strtab ~ty:3 ~flags:0 ~addr:0 ~off:off_strtab
    ~size:(Bytes.length strtab) ~link:0 ~info:0 ~salign:1 ~entsize:0;
  shdr ~nm:n_rela ~ty:4 ~flags:0 ~addr:0 ~off:off_rela ~size:24 ~link:5 ~info:1
    ~salign:8 ~entsize:24;
  shdr ~nm:n_shstrtab ~ty:3 ~flags:0 ~addr:0 ~off:off_shstr
    ~size:(Buffer.length shstr) ~link:0 ~info:0 ~salign:1 ~entsize:0;
  let obj = Buffer.to_bytes buf in
  Bytes.blit_string "\x7fELF\x02\x01\x01" 0 obj 0 7;
  set16 obj 16 3 (* e_type: ET_DYN *);
  set16 obj 18 0xB7 (* e_machine: AArch64 *);
  set32 obj 20 1 (* e_version *);
  set64 obj 40 e_shoff;
  set16 obj 52 64 (* e_ehsize *);
  set16 obj 58 64 (* e_shentsize *);
  set16 obj 60 9 (* e_shnum *);
  set16 obj 62 8 (* e_shstrndx *);
  obj

let () =
  run "Elf"
    [
      group "Parsing"
        [
          test "clang object exposes relocation sections" (fun () ->
            let elf =
              load_c
                {|
                  int something;
                  int test(int x) { return something + x; }
                |}
            in
            let names =
              Elf.sections elf |> Array.to_list
              |> List.map (fun (s : Elf.section) -> s.name)
            in
            is_true (List.mem ".text" names);
            is_true
              (List.mem ".rela.text" names || List.mem ".rel.text" names));
          test "bss is laid out in image" (fun () ->
            let elf =
              load_c
                {|
                  int counter;
                  int test(void) { return 1; }
                |}
            in
            let bss = require_section elf ".bss" in
            equal int 4 bss.size;
            is_true (Bytes.length (Elf.image elf) >= bss.addr + bss.size);
            equal string "\000\000\000\000" (Bytes.to_string bss.content));
          test "entry symbol offset is reported" (fun () ->
            let elf =
              load_c {|
                int test(int x) { return x + 1; }
              |}
            in
            let off = Elf.find_symbol_offset elf "test" in
            is_true (off >= 0);
            let text = require_section elf ".text" in
            is_true (off >= text.addr && off < text.addr + text.size));
          test "undefined external is preserved in relocations" (fun () ->
            let elf =
              load_c
                {|
                  float powf(float, float);
                  float test(float x, float y) { return powf(x, y); }
                |}
            in
            let names =
              Elf.relocs elf
              |> List.map (fun (r : Elf.reloc) -> r.symbol.name)
            in
            is_true (List.mem "powf" names));
        ];
      group "ET_DYN"
        [
          test "fixed section addresses are kept as image offsets" (fun () ->
              let elf = Elf.load (et_dyn_fixture ()) in
              let text = require_section elf ".text" in
              equal int 0x100 text.addr;
              let rodata = require_section elf ".rodata" in
              equal int 0x40 rodata.addr;
              let image = Elf.image elf in
              equal string "TEXTSECT" (Bytes.sub_string image 0x100 8);
              equal string "RODA" (Bytes.sub_string image 0x40 4));
          test "unplaced sections are appended after the fixed ones" (fun () ->
              let elf = Elf.load (et_dyn_fixture ()) in
              let comment = require_section elf ".comment" in
              equal int 0x108 comment.addr;
              let image = Elf.image elf in
              equal int (0x108 + 5) (Bytes.length image);
              equal string "cmnt!" (Bytes.sub_string image 0x108 5));
          test "non-program sections stay out of the image" (fun () ->
              let elf = Elf.load (et_dyn_fixture ()) in
              equal string "\000\000\000\000"
                (Bytes.sub_string (Elf.image elf) 0x10 4));
          test "relocations resolve against fixed addresses" (fun () ->
              let elf = Elf.load (et_dyn_fixture ()) in
              match Elf.relocs elf with
              | [ r ] ->
                  equal int 0x104 r.offset;
                  equal int 0x11 r.r_type;
                  equal int 8 r.addend;
                  equal string "kern" r.symbol.name
              | rs -> failf "expected one relocation, got %d" (List.length rs));
          test "symbols resolve through fixed addresses" (fun () ->
              let elf = Elf.load (et_dyn_fixture ()) in
              equal int 0x104 (Elf.find_symbol_offset elf "kern"));
          test "executables are still rejected" (fun () ->
              let obj = et_dyn_fixture () in
              set16 obj 16 2 (* e_type: ET_EXEC *);
              raises (Invalid_argument "unsupported ELF type") (fun () ->
                  Elf.load obj));
        ];
    ]
