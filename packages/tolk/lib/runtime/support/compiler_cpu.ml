(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Clang reads the source from a file and writes the object and its diagnostics
   to others, so that it never waits for this process to read. *)
let run prog args src =
  let temp_file contents =
    let path = Filename.temp_file "tolk" "" in
    Out_channel.with_open_bin path (fun oc -> output_string oc contents);
    path
  in
  let src_file = temp_file src and obj = temp_file "" and err = temp_file "" in
  Fun.protect ~finally:(fun () -> List.iter Sys.remove [ src_file; obj; err ])
  @@ fun () ->
  let read path = In_channel.with_open_bin path In_channel.input_all in
  match
    Sys.command
      (Filename.quote_command prog args ~stdin:src_file ~stdout:obj ~stderr:err)
  with
  | 0 -> read obj
  | _ -> raise (Renderer.Compiler.Compile_error (read err))

(* Clang *)

(* The compiler run, which picks the binary of every CPU program. *)
let cc = Helpers.variable_string "CC" "clang"

let clang arch =
  let machine, cpu, feats =
    match String.split_on_char ',' arch with
    | machine :: cpu :: feats -> (machine, cpu, feats)
    | _ ->
        invalid_arg
          (Printf.sprintf
             "invalid arch string: '%s', expected '<arch>,<cpu>,[<feats>]' \
              (eg. 'x86_64,znver2')"
             arch)
  in
  let off f = String.starts_with ~prefix:"-" f in
  let target_args =
    match machine with
    | "x86_64" ->
        ("-march=" ^ cpu)
        :: List.map (fun f -> if off f then "-mno" ^ f else "-m" ^ f) feats
    (* On arm, -march means "runs on this architecture and its supersets": x86's
       -march is arm's -mcpu. x18 is a reserved platform register: macOS
       clobbers it on context switches, and Windows keeps a pointer to the
       thread environment block in it. *)
    | "arm64" ->
        let feat f =
          if off f then "no" ^ String.sub f 1 (String.length f - 1) else f
        in
        [
          "-ffixed-x18";
          "-mcpu=" ^ String.concat "+" (cpu :: List.map feat feats);
        ]
    | "riscv64" ->
        let cpu = if cpu = "native" then "rv64g" else cpu in
        [ "-march=" ^ String.concat "_" (cpu :: feats) ]
    | _ -> invalid_arg (Printf.sprintf "unsupported arch: '%s'" machine)
  in
  (* -fno-math-errno is required for __builtin_sqrt to become an instruction
     instead of a function call. -ffp-contract=off keeps each product and sum
     its own rounding, as the graph states them. *)
  let args =
    [
      "-c";
      "-x";
      "c";
      "-O2";
      "-fPIC";
      "-ffreestanding";
      "-fno-math-errno";
      "-ffp-contract=off";
      "-nostdlib";
      "-fno-ident";
      "--target=" ^ machine ^ "-none-unknown-elf";
    ]
    @ target_args @ [ "-"; "-o"; "-" ]
  in
  Renderer.Compiler.v
    ~cachekey:
      (Printf.sprintf "compile_%s_obj_%s" cc
         (String.map (function ',' -> '_' | c -> c) arch))
    ~disassemble:Helpers.cpu_objdump
    (fun src -> run cc args src)
