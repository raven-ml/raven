(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Clang reads the source from a file and writes its output and its diagnostics
   to others, so that it never waits for this process to read. [run prog args
   src] is the exit status, the output and the diagnostics, or why [prog] does
   not start or its status is lost. *)
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
  match
    Helpers.run_program ~stdin:src_file ~stdout:out ~stderr:err prog args
  with
  | Ok (Some status) -> Ok (status, read out, read err)
  | Ok None -> Error (prog ^ ": its exit status is lost")
  | Error why -> Error why

let compile prog args src =
  match run prog args src with
  | Ok (WEXITED 0, obj, _) -> obj
  | Ok (_, _, err) | Error err -> raise (Renderer.Compiler.Compile_error err)

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
        let s =
          match run prog ("-###" :: args) "" with
          | Ok (WEXITED code, out, err) ->
              Printf.sprintf "%d\n%s%s" code out err
          | Ok ((WSIGNALED n | WSTOPPED n), out, err) ->
              Printf.sprintf "signal %d\n%s%s" n out err
          | Error why -> why
        in
        Hashtbl.replace stated (prog, args) s;
        s

(* The major version of LLVM that a statement names, as in "Ubuntu clang
   version 18.1.3". Apple's Clang numbers its own releases, so its statement
   names none. *)
let llvm_major stated =
  let key = "clang version " in
  let n = String.length key and len = String.length stated in
  let rec find i =
    if i + n > len then None
    else if String.sub stated i n = key then Some i
    else find (i + 1)
  in
  match find 0 with
  | None -> None
  | Some i when i >= 6 && String.sub stated (i - 6) 6 = "Apple " -> None
  | Some i ->
      let first = i + n in
      let rec last j =
        if j < len && stated.[j] >= '0' && stated.[j] <= '9' then last (j + 1)
        else j
      in
      int_of_string_opt (String.sub stated first (last first - first))

(* Clang *)

external mac_fp16 : unit -> bool = "caml_tolk_cpu_mac_fp16"

(* Clang names an arm64 Mac's processor from the family macOS reports, which a
   virtual Mac reports as 0: [native] is then a generic processor without half
   arithmetic, whatever features macOS lists. [native] takes the half
   arithmetic macOS lists, which on a Mac Clang knows changes nothing. *)
let mac_native = if mac_fp16 () then [ "fp16" ] else []

let clang arch =
  let cc = Setting.value Setting.cc in
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
        let feats =
          if cpu = "native" then mac_native @ feats else feats
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
  let base =
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
    @ target_args
  in
  let io = [ "-"; "-o"; "-" ] in
  (* LLVM 18's straight-line vectorizer narrows the integer lanes it packs and
     can then extend a lane to a narrower type, which is invalid IR: for four
     clamped gather indices on a processor with AVX-512 it wrote [sext <4 x
     i32> to <4 x i16>], on which instruction selection crashed. LLVM 19 packs
     them correctly. Under LLVM 18 only, the vectorizer is off, so that other
     Clangs compile as before. The flag follows -O2, which would turn the
     vectorizer back on. The statement is memoized: asking for the arguments
     again costs a lookup. *)
  let args () =
    match llvm_major (statement cc (base @ io)) with
    | Some 18 -> base @ ("-fno-slp-vectorize" :: io)
    | _ -> base @ io
  in
  let table () =
    Printf.sprintf "compile_%s_obj_%s_%s" cc
      (String.map (function ',' -> '_' | c -> c) arch)
      (Digest.to_hex (Digest.string (statement cc (args ()))))
  in
  Renderer.Compiler.v ~cachekey:table ~disassemble:Helpers.cpu_objdump
    (fun src -> compile cc (args ()) src)
