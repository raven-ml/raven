(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Clang reads the source from a file and writes its output and its diagnostics
   to others, so that it never waits for this process to read. [run prog args
   src] is the exit code, the output and the diagnostics. *)
let run prog args src =
  let temp_file contents =
    let path = Filename.temp_file "tolk" "" in
    Out_channel.with_open_bin path (fun oc -> output_string oc contents);
    path
  in
  let src_file = temp_file src and out = temp_file "" and err = temp_file "" in
  Fun.protect ~finally:(fun () -> List.iter Sys.remove [ src_file; out; err ])
  @@ fun () ->
  let read path = In_channel.with_open_bin path In_channel.input_all in
  let code =
    Sys.command
      (Filename.quote_command prog args ~stdin:src_file ~stdout:out ~stderr:err)
  in
  (code, read out, read err)

let compile prog args src =
  match run prog args src with
  | 0, obj, _ -> obj
  | _, _, err -> raise (Renderer.Compiler.Compile_error err)

(* What Clang states it runs for [args] ([-###]), without running it: its
   version and installation, then the command of its compiler proper, which
   holds the processor and features [native] resolves to and every option. An
   object is a function of that and its source. A Clang that rejects [args], or
   does not run, states why, and compiles nothing. Each command is asked once
   per process. *)
let statement =
  let lock = Mutex.create () and stated = Hashtbl.create 4 in
  fun prog args ->
    Mutex.protect lock @@ fun () ->
    match Hashtbl.find_opt stated (prog, args) with
    | Some s -> s
    | None ->
        let code, out, err = run prog ("-###" :: args) "" in
        let s = Printf.sprintf "%d\n%s%s" code out err in
        Hashtbl.replace stated (prog, args) s;
        s

(* Clang *)

(* The compiler run, which picks the binary of every CPU program. *)
let cc = Helpers.Context_var.string ~reach:Output "CC" "clang"

let clang arch =
  let cc = Helpers.Context_var.value cc in
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
     its own rounding, as the graph states them. -ffile-compilation-dir=. keeps
     the working directory out of the object, and out of what Clang states. *)
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
      "-ffile-compilation-dir=.";
      "--target=" ^ machine ^ "-none-unknown-elf";
    ]
    @ target_args @ [ "-"; "-o"; "-" ]
  in
  let table () =
    Printf.sprintf "compile_%s_obj_%s_%s" cc
      (String.map (function ',' -> '_' | c -> c) arch)
      (Digest.to_hex (Digest.string (statement cc args)))
  in
  Renderer.Compiler.v ~cachekey:table ~disassemble:Helpers.cpu_objdump
    (fun src -> compile cc args src)
