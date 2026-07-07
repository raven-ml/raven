(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk

let u8 bytes off = Char.code (Bytes.get bytes off)

let u16 bytes off =
  let b0 = u8 bytes off in
  let b1 = u8 bytes (off + 1) in
  b0 lor (b1 lsl 8)

let host_machine () =
  let machine =
    if String.equal Sys.os_type "Win32" then
      match Sys.getenv_opt "PROCESSOR_ARCHITECTURE" with
      | Some arch when String.trim arch <> "" -> arch
      | _ -> "amd64"
    else
      let ic = Unix.open_process_in "uname -m" in
      Fun.protect
        ~finally:(fun () -> ignore (Unix.close_process_in ic))
        (fun () -> input_line ic)
  in
  machine |> String.trim |> String.lowercase_ascii

let expected_machine () =
  match host_machine () with
  | "x86_64" | "amd64" -> Some 62
  | "arm64" | "aarch64" -> Some 183
  | "riscv64" -> Some 243
  | _ -> None

let require_supported_host () =
  match expected_machine () with
  | Some machine -> machine
  | None -> invalid_arg "unsupported host architecture for CPU compiler test"

let compile_test_obj () =
  Tolk_cpu__Compiler_cpu.compile_clang
    {|
      int test(int x) { return x + 1; }
    |}

let compiler_outputs_relocatable_elf () =
  let obj = compile_test_obj () in
  is_true (Bytes.length obj >= 64);
  equal char '\x7f' (Bytes.get obj 0);
  equal char 'E' (Bytes.get obj 1);
  equal char 'L' (Bytes.get obj 2);
  equal char 'F' (Bytes.get obj 3);
  equal int 2 (u8 obj 4);
  equal int 1 (u8 obj 5);
  equal int 1 (u16 obj 16);
  equal int (require_supported_host ()) (u16 obj 18)

let compiler_output_is_parseable_by_elf_support () =
  let elf = Elf.load (compile_test_obj ()) in
  let section_names =
    Elf.sections elf |> Array.to_list
    |> List.map (fun (s : Elf.section) -> s.name)
  in
  is_true (List.mem ".text" section_names);
  is_true (Elf.find_symbol_offset elf "test" >= 0)

let invalid_c_raises_compile_error () =
  raises_match
    (function
      | Compiler.Compile_error msg ->
          String.contains msg '\n'
          && String.starts_with ~prefix:"clang -x c failed:" msg
      | _ -> false)
    (fun () -> ignore (Tolk_cpu__Compiler_cpu.compile_clang "int test( {"))

let () =
  run "Runtime_cpu_compiler"
    [
      group "Clang"
        [
          test "outputs relocatable ELF for normalized host"
            compiler_outputs_relocatable_elf;
          test "output is parseable by ELF support"
            compiler_output_is_parseable_by_elf_support;
          test "invalid C raises Compile_error" invalid_c_raises_compile_error;
        ];
    ]
