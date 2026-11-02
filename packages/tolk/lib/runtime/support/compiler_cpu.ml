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

(* What a Clang states about itself: its version and installation. *)
let version prog = statement prog [ "--version" ]

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

(* LLVM's loop vectorizer weighs each width it could pack a loop's iterations
   into against the whole body, and its time grows faster than the body: the
   scalar loop of float64 [betainccinv]'s gradient, 1.3 MB of C, took Clang
   19.7 s with it and 5.7 s without, loaded. A source of more than
   [long_source] bytes compiles without it. Of the kernels of tolk's bench and
   rune's compile bench and test suites, those past it are rune's gradients of
   special functions, of 432 KB to 1.3 MB, and a chain of incomplete betas
   without a loop, whose object stays the same; every other is under 140 KB. *)
let long_source = 400_000

(* Clang *)

external mac_fp16 : unit -> bool = "caml_tolk_cpu_mac_fp16"

(* Clang names an arm64 Mac's processor from the family macOS reports, which a
   virtual Mac reports as 0: [native] is then a generic processor without half
   arithmetic, whatever features macOS lists. [native] takes the half
   arithmetic macOS lists, which on a Mac Clang knows changes nothing. *)
let mac_native = if mac_fp16 () then [ "fp16" ] else []

let win64 arch =
  List.hd (String.split_on_char ',' arch) = "x86_64-windows"

(* LLVM 18 crashes, in machine loop-invariant code motion, compiling some
   loops over halves for a processor with AVX-512's half arithmetic
   ([avx512fp16]), such as one that stores a half's double bits in one row and
   the half converted to a short in the others: rune's casts from float16.
   AVX10.1 implies it, and [native] on a runner whose processor states AVX10.1
   enables both. LLVM 17 and 19 compile the loop. Under LLVM 18 these flags
   follow [arch]'s, so they win; avx10.1-512 is disabled first, since
   disabling avx10.1-256 leaves it on. A half is a storage format that
   computes in float ([Cstyle.clang]), so only the instructions converting it
   change: F16C's or AVX-512F's, as on a processor without AVX-512 FP16. *)
let llvm18_x86_off =
  [ "-mno-avx10.1-512"; "-mno-avx10.1-256"; "-mno-avx512fp16" ]

(* The machine of [arch] and the flags that make Clang compile for it: its
   triple, processor and features. *)
let selection arch =
  let machine, cpu, feats =
    match String.split_on_char ',' arch with
    | machine :: cpu :: feats ->
        (List.hd (String.split_on_char '-' machine), cpu, feats)
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
  (* An object for x86_64 Windows calls by Windows' convention, the convention
     of the process's functions, which it calls by name or which Clang calls
     itself, such as [fma] or [memcpy]; its entry is declared System V's, by
     which rig calls it. [w64-windows-gnu] is the system's triple that takes
     -fPIC. rig commits a thread's stack before it runs linked code (rig's
     host code), so the code probes none, and Windows may write below the
     stack pointer, so the code keeps nothing there. On arm64, Windows' convention for
     functions of fixed arguments is the ELF object's. *)
  let triple =
    if win64 arch then
      [
        "--target=x86_64-w64-windows-gnu-elf";
        "-mno-stack-arg-probe";
        "-mno-red-zone";
      ]
    else [ "--target=" ^ machine ^ "-none-unknown-elf" ]
  in
  (machine, triple @ target_args)

(* The flags of a [selection] for a Clang of LLVM [llvm]. *)
let fixed ~llvm (machine, flags) =
  if machine = "x86_64" && llvm = Some 18 then flags @ llvm18_x86_off
  else flags

let flags ~version arch = fixed ~llvm:(llvm_major version) (selection arch)

(* The vector registers of what Clang compiles [arch] for. Its statement names
   the processor alone, as [-target-cpu x86-64-v3], and the processor's features
   only for [native]; the macros the preprocessor defines name them for every
   processor. On x86_64: 16 registers, 32 with AVX-512, of 32 bytes with AVX and
   16 without. AVX-512's registers hold 64 bytes, but a processor with them is
   taken at 32 until one is measured: tolk renders its vectors, so Clang's
   preference for 256 bits on such processors does not choose for it. On arm64,
   NEON's 32 registers of 16 bytes. On riscv64, none known. A Clang that does
   not run gives SSE's 16 of 16 bytes, which every x86_64 processor has. Each
   Clang and [arch] is asked once per process. *)
let registers =
  let lock = Mutex.create () and known = Hashtbl.create 4 in
  fun arch ->
    let cc = Setting.value Setting.cc in
    let selected = selection arch in
    Mutex.protect lock @@ fun () ->
    match Hashtbl.find_opt known (cc, arch) with
    | Some r -> r
    | None ->
        let r : Renderer.registers option =
          match fst selected with
          | "x86_64" ->
              let flags = fixed ~llvm:(llvm_major (version cc)) selected in
              let macros =
                match
                  run cc ([ "-dM"; "-E"; "-x"; "c" ] @ flags @ [ "-" ]) ""
                with
                | Ok (WEXITED 0, out, _) -> String.split_on_char '\n' out
                | Ok _ | Error _ -> []
              in
              let defines m =
                List.exists
                  (String.starts_with ~prefix:("#define " ^ m ^ " "))
                  macros
              in
              Some
                {
                  count = (if defines "__AVX512F__" then 32 else 16);
                  bytes = (if defines "__AVX__" then 32 else 16);
                }
          | "arm64" -> Some { count = 32; bytes = 16 }
          | _ -> None
        in
        Hashtbl.replace known (cc, arch) r;
        r

(* -fno-math-errno is required for __builtin_sqrt to become an instruction
   instead of a function call. -ffp-contract=off keeps each product and sum its
   own rounding, as the graph states them. -ffile-compilation-dir=. keeps the
   working directory out of the object, and out of what Clang states. *)
let clang arch =
  let cc = Setting.value Setting.cc in
  let selected = selection arch in
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
    ]
  in
  let io = [ "-"; "-o"; "-" ] in
  (* LLVM 18's straight-line vectorizer narrows the integer lanes it packs and
     can then extend a lane to a narrower type, which is invalid IR: for four
     clamped gather indices on a processor with AVX-512 it wrote [sext <4 x
     i32> to <4 x i16>], on which instruction selection crashed. LLVM 19 packs
     them correctly. Under LLVM 18 only, that vectorizer is off, so that other
     Clangs compile as before. The flags follow -O2, which would turn the
     vectorizers back on. The statement is memoized: asking for the arguments
     again costs a lookup. *)
  let args ~long =
    let llvm = llvm_major (version cc) in
    let slp = if llvm = Some 18 then [ "-fno-slp-vectorize" ] else [] in
    base
    @ fixed ~llvm selected
    @ slp
    @ (if long then [ "-fno-vectorize" ] else [])
    @ io
  in
  (* An object is a function of its source and of the arguments its length
     selects, so the table names both statements and the length. *)
  let table () =
    let stated =
      String.concat "\n"
        [
          statement cc (args ~long:false);
          statement cc (args ~long:true);
          string_of_int long_source;
        ]
    in
    Printf.sprintf "compile_%s_obj_%s_%s" cc
      (String.map (function ',' -> '_' | c -> c) arch)
      (Digest.to_hex (Digest.string stated))
  in
  Renderer.Compiler.v ~cachekey:table ~disassemble:Helpers.cpu_objdump
    (fun src -> compile cc (args ~long:(String.length src > long_source)) src)
