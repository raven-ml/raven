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
            "-c";
            "-x";
            "c";
            arch_flag;
            Filename.quote (Printf.sprintf "--target=%s-none-unknown-elf" arch);
            "-O2";
            "-fPIC";
            "-ffreestanding";
            "-fno-math-errno";
            "-nostdlib";
            "-fno-ident";
            Filename.quote src_path;
            "-o";
            Filename.quote obj_path;
            "2>";
            Filename.quote err_path;
          ]
      in
      match Sys.command command with
      | 0 -> Bytes.of_string (read_file obj_path)
      | _ ->
          let err = read_file err_path in
          failwith
            (if String.equal err "" then "clang failed"
             else "clang failed:\n" ^ err))

let test_load_clang_object_exposes_relocation_sections () =
  let obj =
    compile_c
      {|
        int something;
        int test(int x) { return something + x; }
      |}
  in
  let elf = Elf.load obj in
  let section_names =
    Elf.sections elf |> Array.to_list
    |> List.map (fun (section : Elf.section) -> section.Elf.name)
  in
  is_true (List.mem ".text" section_names);
  is_true
    (List.mem ".rela.text" section_names || List.mem ".rel.text" section_names)

let test_bss_is_laid_out_in_image () =
  let obj =
    compile_c
      {|
        int counter;
        int test(void) { return 1; }
      |}
  in
  let elf = Elf.load obj in
  match Elf.find_section elf ".bss" with
  | None -> failwith "expected .bss section"
  | Some bss ->
      equal int 4 bss.size;
      is_true (Bytes.length (Elf.image elf) >= bss.addr + bss.size);
      equal string "\000\000\000\000" (Bytes.to_string bss.content)

let test_entry_symbol_offset_is_reported () =
  let obj = compile_c {|
        int test(int x) { return x + 1; }
      |} in
  let elf = Elf.load obj in
  let off = Elf.find_symbol_offset elf "test" in
  is_true (off >= 0);
  match Elf.find_section elf ".text" with
  | None -> failwith "expected .text section"
  | Some text -> is_true (off >= text.addr && off < text.addr + text.size)

let test_undefined_external_is_preserved_in_relocations () =
  let obj =
    compile_c
      {|
        float powf(float, float);
        float test(float x, float y) { return powf(x, y); }
      |}
  in
  let elf = Elf.load obj in
  let names =
    Elf.relocs elf |> List.map (fun (reloc : Elf.reloc) -> reloc.symbol.name)
  in
  is_true (List.mem "powf" names)

let () =
  run "Elf"
    [
      group "Parsing"
        [
          test "clang object exposes relocation sections"
            test_load_clang_object_exposes_relocation_sections;
          test "bss is laid out in image" test_bss_is_laid_out_in_image;
          test "entry symbol offset is reported"
            test_entry_symbol_offset_is_reported;
          test "undefined external is preserved in relocations"
            test_undefined_external_is_preserved_in_relocations;
        ];
    ]
