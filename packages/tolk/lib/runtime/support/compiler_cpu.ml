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

(* The arguments of a command as Clang states it: each in double quotes, in
   which a backslash escapes the character after it. *)
let unquoted line =
  let n = String.length line and arg = Buffer.create 64 in
  let rec outside i args =
    if i >= n then List.rev args
    else if line.[i] = '"' then inside (i + 1) args
    else outside (i + 1) args
  and inside i args =
    if i >= n then List.rev args
    else
      match line.[i] with
      | '\\' when i + 1 < n ->
          Buffer.add_char arg line.[i + 1];
          inside (i + 2) args
      | '"' ->
          let a = Buffer.contents arg in
          Buffer.clear arg;
          outside (i + 1) (a :: args)
      | c ->
          Buffer.add_char arg c;
          inside (i + 1) args
  in
  outside 0 []

(* The command of Clang's compiler proper ([-cc1]) that a statement of a compile
   from standard input to standard output names, without its input and output
   and with LLVM's IR verifier on, or the statement if it names none: then Clang
   rejected the arguments or did not run.

   A release Clang turns the verifier off ([-disable-llvm-verifier]) to compile
   a few percent faster. An optimization that writes invalid IR then goes
   unseen, and the code compiles into wrong values or crashes the instruction
   selector: LLVM 19's straight-line vectorizer wrote [trunc <4 x i8> to <4 x
   i32>] for four lanes of a binary search, and the kernel read garbage row
   numbers. With the verifier on, the IR is checked before the optimizations and
   again before instruction selection, and invalid IR stops the compile with
   Clang's diagnostic, which names the function. *)
let cc1 stated =
  let command line =
    match unquoted line with
    | prog :: "-cc1" :: args -> Some (prog, "-cc1" :: args)
    | _ -> None
  in
  let rec kept = function
    | "-o" :: "-" :: args -> kept args
    | [ "-" ] -> []
    | "-disable-llvm-verifier" :: args -> kept args
    | a :: args -> a :: kept args
    | [] -> []
  in
  match List.find_map command (String.split_on_char '\n' stated) with
  | Some (prog, args) -> Ok (prog, kept args)
  | None -> Error stated

(* The command of the compiler proper for a source of more than [long_source]
   bytes: [command] without the loop vectorizer, which Clang's driver turns on
   with [-vectorize-loops] at [-O2] and leaves out under [-fno-vectorize], and
   nothing else. *)
let without_loop_vectorizer (prog, args) =
  (prog, List.filter (( <> ) "-vectorize-loops") args)

(* Several sources compiled by one run of Clang's compiler proper, each from a
   file of its own, as Clang's driver would compile it alone: the driver runs
   the compiler proper in its own process for one source, and starts one
   process for each of several, whose start costs about what a small kernel's
   compile does. Each source is the file [-] in a directory of its own, which
   the object names as it names standard input, and starts with a line marker
   that names it [<stdin>], as its diagnostics and [__FILE__] then do: the
   object is the one the driver makes of the source from standard input,
   wherever the directory is. The compiler writes it beside the source as
   [-.o]. The inputs are named in a response file, since Windows bounds a
   command line to 32767 characters. *)
let compile_files (prog, args) srcs =
  let out = Filename.temp_file "tolk" "" and err = Filename.temp_file "tolk" "" in
  let rsp = Filename.temp_file "tolk" ".rsp" in
  let dir = Filename.temp_dir "tolk" "" in
  let input i = Filename.concat (Filename.concat dir (string_of_int i)) "-" in
  (* A Clang killed mid-compile leaves its own temporary files beside an
     input, so each directory is emptied of whatever it holds. *)
  let remove () =
    let rm f = if Sys.file_exists f then Sys.remove f in
    let rm_dir d =
      if Sys.file_exists d then begin
        Array.iter (fun f -> rm (Filename.concat d f)) (Sys.readdir d);
        Sys.rmdir d
      end
    in
    List.iteri (fun i _ -> rm_dir (Filename.dirname (input i))) srcs;
    Sys.rmdir dir;
    List.iter rm [ out; err; rsp ]
  in
  Fun.protect ~finally:remove @@ fun () ->
  List.iteri
    (fun i src ->
      Sys.mkdir (Filename.dirname (input i)) 0o700;
      Out_channel.with_open_bin (input i) (fun oc ->
          output_string oc "# 1 \"<stdin>\"\n";
          output_string oc src))
    srcs;
  (* A response file is read by Windows' rules on Windows, where a path's
     backslashes may be slashes, and by the shell's elsewhere. *)
  let quoted path =
    let path =
      if Sys.win32 then String.map (function '\\' -> '/' | c -> c) path
      else
        String.concat ""
          (List.map
             (function
               | ('\\' | '"') as c -> "\\" ^ String.make 1 c
               | c -> String.make 1 c)
             (List.of_seq (String.to_seq path)))
    in
    "\"" ^ path ^ "\""
  in
  Out_channel.with_open_bin rsp (fun oc ->
      List.iteri (fun i _ -> output_string oc (quoted (input i) ^ "\n")) srcs);
  let read path = In_channel.with_open_bin path In_channel.input_all in
  let fail why = raise (Renderer.Compiler.Compile_error why) in
  match
    Helpers.run_program ~stdout:out ~stderr:err prog (args @ [ "@" ^ rsp ])
  with
  | Ok (Some (WEXITED 0)) -> List.mapi (fun i _ -> read (input i ^ ".o")) srcs
  | Ok (Some _) -> fail (read err)
  | Ok None -> fail (prog ^ ": its exit status is lost")
  | Error why -> fail why

(* A Clang's major release: LLVM's, or Apple's, which numbers its own. *)
type release = Llvm of int | Apple of int

(* The major release that a statement names, as in "Ubuntu clang version
   18.1.3" or "Apple clang version 17.0.0 (clang-1700.6.4.2)". *)
let stated_release stated =
  let key = "clang version " in
  let n = String.length key and len = String.length stated in
  let rec find i =
    if i + n > len then None
    else if String.sub stated i n = key then Some i
    else find (i + 1)
  in
  let major first =
    let rec last j =
      if j < len && stated.[j] >= '0' && stated.[j] <= '9' then last (j + 1)
      else j
    in
    int_of_string_opt (String.sub stated first (last first - first))
  in
  match find 0 with
  | None -> None
  | Some i when i >= 6 && String.sub stated (i - 6) 6 = "Apple " ->
      Option.map (fun r -> Apple r) (major (i + n))
  | Some i -> Option.map (fun r -> Llvm r) (major (i + n))

(* The line of a statement that names its Clang's version, as "Ubuntu clang
   version 14.0.0-1ubuntu1.1", or the statement if none does. *)
let version_line stated =
  let named l = String.length l > 0 && Option.is_some (stated_release l) in
  match List.find_opt named (String.split_on_char '\n' stated) with
  | Some l -> String.trim l
  | None -> stated

(* A release as the hint table keeps it: LLVM's as its number, Apple's after
   "apple". *)
let release_to_string = function
  | Llvm r -> string_of_int r
  | Apple r -> "apple" ^ string_of_int r

let release_of_string l =
  let apple = "apple" in
  let n = String.length apple in
  if String.starts_with ~prefix:apple l then
    Option.map
      (fun r -> Apple r)
      (int_of_string_opt (String.sub l n (String.length l - n)))
  else Option.map (fun r -> Llvm r) (int_of_string_opt l)

(* The release of each Clang, as the last process that asked it found it: a
   hint, kept on disk while {!Setting.ccache} holds, from which a process picks
   the flags it first runs Clang's preprocessor under ({!dumped}), so that a
   Clang of the hinted release is run once. *)
let hint_table = "clang_llvm"

let hinted cc =
  if not (Setting.value Setting.ccache) then None
  else
    match Helpers.Diskcache.get ~table:hint_table cc with
    | Some l -> release_of_string l
    | None | (exception (Failure _ | Sys_error _)) -> None

let hint cc release =
  let l = Option.fold ~none:"" ~some:release_to_string release in
  if Setting.value Setting.ccache then
    try Helpers.Diskcache.put ~table:hint_table cc l with Sys_error _ -> ()

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

(* Before LLVM 23, x86's instruction selection writes the wrong truth table for
   a ternary logic instruction under a write mask whose memory operand it moves
   to the last place (LLVM issue 157929, fixed in LLVM 23). AVX-512 alone has
   write masks. A selection of a bitwise select meets it: for rune's bfloat16
   sinh, [isnan x ? x : (r & 0x7fffffff) | (x & 0x80000000)] on x86-64-v4 became
   [vpternlogd $184] over a broadcast sign mask, which computes [(r &
   0x80000000) | (x & ~r)]: wrong for every [x] but a NaN or a zero. LLVM 18 and
   19 and Apple's Clang 17 write it for that kernel, LLVM 21 and 22 for LLVM's
   own tests of the instruction. Under those Clangs AVX-512 is off, and the code
   is AVX2's. These flags follow [arch]'s, so they win. AVX10.1 implies AVX-512,
   and [native] on a processor that states AVX10.1 enables it by name, which
   under LLVM 18 to 20 and Apple's Clang 17 is disabled first: avx10.1-512
   before avx10.1-256, since disabling avx10.1-256 leaves it on. LLVM 21's
   driver has no option that disables an AVX10.1 that [arch] or [native] names:
   its feature map applies AVX10.1 after every other feature, and AVX10.1
   enables AVX-512F again ([keeps_avx512]); [native] then compiles for its
   processor by name ([settled]). LLVM 22's disables it with AVX-512F. A Clang
   that states no release, such as a compiler built on Clang that names itself,
   is taken as one before LLVM 23.

   It also keeps LLVM 18 off AVX-512's half arithmetic, with which it crashes in
   machine loop-invariant code motion on rune's casts from float16. *)
let avx10_off = [ "-mno-avx10.1-512"; "-mno-avx10.1-256" ]

let avx512_off = function
  | Some (Llvm (18 | 19 | 20) | Apple 17) -> avx10_off @ [ "-mno-avx512f" ]
  | Some (Llvm r) when r >= 23 -> []
  | Some (Llvm _ | Apple _) | None -> [ "-mno-avx512f" ]

(* The machine of [arch] without its system, its processor, its features as
   written, and the flags that tune it. *)
let fields arch =
  match String.split_on_char ',' arch with
  | machine :: cpu :: fields ->
      let tune, feats =
        List.partition_map
          (fun f ->
            match String.split_on_char '=' f with
            | [ "tune"; t ] -> Left ("-mtune=" ^ t)
            | _ -> Right f)
          fields
      in
      (List.hd (String.split_on_char '-' machine), cpu, feats, tune)
  | _ ->
      invalid_arg
        (Printf.sprintf
           "invalid arch string: '%s', expected '<arch>,<cpu>,[<feats>]' \
            (eg. 'x86_64,znver2')"
           arch)

(* The features Clang is given for [machine]'s [cpu]: an arm64 Mac's
   [native] gains the half arithmetic macOS lists. *)
let features machine cpu feats =
  if machine = "arm64" && cpu = "native" then mac_native @ feats else feats

let has_feature arch f =
  let machine, cpu, feats, _ = fields arch in
  List.fold_left
    (fun on g -> if g = f then true else if g = "-" ^ f then false else on)
    false
    (features machine cpu feats)

(* The machine of [arch] and the flags that make Clang compile for it: its
   triple, processor and features, and the processor it tunes for. *)
let selection arch =
  let machine, cpu, feats, tune = fields arch in
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
        let feats = features machine cpu feats in
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
  (machine, triple @ target_args @ tune)

(* The flags of a [selection] for a Clang of release [release]. *)
let fixed ~release (machine, flags) =
  if machine = "x86_64" then flags @ avx512_off release else flags

let flags ~version arch =
  fixed ~release:(stated_release version) (selection arch)

(* -fno-math-errno is required for __builtin_sqrt to become an instruction
   instead of a function call. -ffp-contract=off keeps each product and sum its
   own rounding, as the graph states them. -ffile-compilation-dir=. keeps the
   working directory out of the object, and out of what Clang states.

   The straight-line vectorizers of LLVM 18 and 19 narrow the integer lanes
   they pack and can then cast a lane to a type of the wrong width, which is
   invalid IR. For four clamped gather indices on a processor with AVX-512,
   LLVM 18 wrote [sext <4 x i32> to <4 x i16>], on which instruction selection
   crashed. For four lanes of a binary search on a processor with AVX2, LLVM 19
   wrote [trunc <4 x i8> to <4 x i32>], which a Clang without its IR verifier
   compiles into wrong values: rune's routed rows read garbage row numbers, and
   with it ([cc1]) the compile fails. LLVM 19 and 20 also narrow lanes into
   valid IR of wrong values: a signed lane that is converted to float as
   unsigned can be narrowed to too few bits (LLVM issue 129244, fixed in
   LLVM 21). Two int64 halves converted this way became
   [uitofp <2 x i8>], so that a low half of 2^32 - 1 converted to 255. Apple's
   Clang 17 writes the binary search's invalid IR for x86_64 too, and neither
   kernel's defect for arm64, where without the vectorizer its objects of 466
   kernels of tolk's and rune's tests hold 9.5% more instructions.
   Under LLVM 18, 19 and 20, and under Apple's Clang 17 for x86_64, that
   vectorizer is off, so that other Clangs compile as before. The flags
   follow -O2, which would turn the vectorizers back on.

   On arm64, LLVM's interleaved load combine looks for loads that shuffles put
   together into an interleaved vector, to read them with one ldN, and its time
   grows much faster than the lanes it meets: a host kernel of rune's
   contraction of 8 groups of data-chosen rows that computes 1024 lanes
   gathered from 32 loaded rows took Clang 19 for apple-m1 293 s, 274 s of
   them in that pass, and 8 s without it. Every Clang compiles arm64 without
   it.

   LLVM's loop idiom recognition replaces a loop by the library call it
   computes, which a freestanding source has none of, by a bit-counting
   instruction for a loop that exits on its data, which no kernel tolk
   produces has, and from LLVM 22 a CRC's loop by a table. It changes no
   object of 10 213 kernels of rune's linear algebra and suites under Clang
   19, nor of 8694 of them under Clang 18 and LLVM 23. For a loop of a multiple
   of 8 trips, LLVM 22's and 23's CRC recogniser walks every path through the
   operands of the loop's running selection: the 32 maxima of rune's 32 x 32
   eigh, each reading the last twice, are 2^31 paths, which LLVM 23 did not
   walk in 300 s; without the pass the kernel compiles in 0.01 s. Every Clang
   compiles without it. *)
let compile_args selected ~release =
  let slp =
    match (release, fst selected) with
    | Some (Llvm (18 | 19 | 20)), _ | Some (Apple 17), "x86_64" ->
        [ "-fno-slp-vectorize" ]
    | _ -> []
  in
  let ldn =
    match selected with
    | "arm64", _ -> [ "-mllvm"; "-disable-interleaved-load-combine" ]
    | _ -> []
  in
  let idioms = [ "-mllvm"; "-disable-loop-idiom-all" ] in
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
  @ fixed ~release selected @ slp @ ldn @ idioms @ [ "-"; "-o"; "-" ]

(* What Clang's preprocessor gives under [flags]: the version its driver states
   ([-v]), which names its release as its statement does, Apple's among them,
   whose macros name none for an ELF target, and the lines of its macros, each
   [#define NAME VALUE]; or why it gave none. Clang is run once per process for
   each Clang and flags whose macros it gave: each run costs about what a small
   kernel's compile does. A run that fails is not kept, so that a failure to
   start it, which a later run may not meet, does not decide for the process. *)
type preprocessed = { version : string; macros : string list }

let preprocessed =
  let lock = Mutex.create () and known = Hashtbl.create 4 in
  fun cc flags ->
    Mutex.protect lock @@ fun () ->
    match Hashtbl.find_opt known (cc, flags) with
    | Some given -> Ok given
    | None -> (
        match
          run cc ([ "-v"; "-dM"; "-E"; "-x"; "c" ] @ flags @ [ "-" ]) ""
        with
        | Ok (WEXITED 0, out, err) ->
            let given =
              { version = err; macros = String.split_on_char '\n' out }
            in
            Hashtbl.replace known (cc, flags) given;
            Ok given
        | Ok (WEXITED code, _, err) ->
            Error (Printf.sprintf "%s -dM -E exited %d: %s" cc code err)
        | Ok ((WSIGNALED n | WSTOPPED n), _, _) ->
            Error (Printf.sprintf "%s -dM -E stopped by signal %d" cc n)
        | Error why -> Error why)

(* The value of the macro [m] among [macros], if it is defined. *)
let value macros m =
  let prefix = "#define " ^ m ^ " " in
  let n = String.length prefix in
  List.find_map
    (fun l ->
      if String.starts_with ~prefix l then
        Some (String.sub l n (String.length l - n))
      else None)
    macros

let defines macros m = Option.is_some (value macros m)

(* The release that [macros] name: [__clang_major__], which is Apple's own
   number where [__apple_build_version__] is defined. Apple's Clang defines it
   for Apple's targets alone, so for an ELF target its macros name LLVM's
   release of Apple's number. *)
let macro_release macros =
  match Option.bind (value macros "__clang_major__") int_of_string_opt with
  | None -> None
  | Some r when defines macros "__apple_build_version__" -> Some (Apple r)
  | Some r -> Some (Llvm r)

(* Whether the macros [selected] gave under [asked]'s flags are also those under
   [release]'s: the flags are the same, or [release]'s only disable AVX10.1
   first and the macros define none of AVX10's. Then no AVX10 feature is on, nor
   one that implies it, which would turn one on; a disable turns nothing on, so
   disabling AVX10.1 changes nothing: with LLVM 19, on a processor without
   AVX10.1, the macros of the release-less flags serve. *)
let same_macros selected ~asked ~release macros =
  fixed ~release selected = fixed ~release:asked selected
  || fst selected = "x86_64"
     && avx512_off release = avx10_off @ avx512_off asked
     && not (List.exists (String.starts_with ~prefix:"#define __AVX10") macros)

(* What the preprocessor gives for what [cc] compiles [selected] for, under the
   flags of the release it states, or why it gave nothing: the macros name, on
   x86_64, the processor's features for every processor, where Clang's statement
   names them only for [native]. The preprocessor runs under the flags of the
   hinted release, or of none if it rejects those, and again under those of the
   release it states unless its macros serve ({!same_macros}). What it gave is
   kept for the process. *)
let dumped =
  let lock = Mutex.create () and known = Hashtbl.create 4 in
  fun cc selected ->
    match
      Mutex.protect lock (fun () -> Hashtbl.find_opt known (cc, selected))
    with
    | Some given -> Ok given
    | None -> (
        let under release = preprocessed cc (fixed ~release selected) in
        let hinted = hinted cc in
        let asked, first =
          match under hinted with
          | Error _ when hinted <> None -> (None, under None)
          | first -> (hinted, first)
        in
        let given =
          match first with
          | Error _ as e -> e
          | Ok first ->
              let release = stated_release first.version in
              if release <> hinted then hint cc release;
              if same_macros selected ~asked ~release first.macros then Ok first
              else under release
        in
        match given with
        | Ok g ->
            Mutex.protect lock (fun () ->
                Hashtbl.replace known (cc, selected) g);
            given
        | Error _ -> given)

(* The release that [g]'s Clang states, or its macros name where it states none,
   as a compiler built on Clang may. *)
let given_release g =
  match stated_release g.version with
  | Some _ as release -> release
  | None -> macro_release g.macros

(* The macros of {!dumped}, none where the preprocessor gave none. *)
let macros cc selected =
  match dumped cc selected with Ok g -> g.macros | Error _ -> []

(* The release that [cc] states as it preprocesses for [selected], [None] where
   it states none or the preprocessor gives nothing. *)
let release cc selected =
  match dumped cc selected with
  | Ok g -> stated_release g.version
  | Error _ -> None

(* What [cc] states it runs to compile [selected], under the flags of its
   release: Clang states its version and installation before its commands. *)
let own_statement cc selected =
  statement cc (compile_args selected ~release:(release cc selected))

(* Whether [cc] keeps AVX-512 on for an x86_64 [selected] that [avx512_off]
   turns it off for, as LLVM 21 does for an AVX10.1 that [arch] or [native]
   names. *)
let keeps cc selected =
  fst selected = "x86_64"
  && avx512_off (release cc selected) <> []
  && defines (macros cc selected) "__AVX512F__"

(* The processor a statement's compiler proper compiles for. *)
let target_cpu stated =
  let rec after = function
    | "-target-cpu" :: cpu :: _ -> Some cpu
    | _ :: args -> after args
    | [] -> None
  in
  match cc1 stated with Ok (_, args) -> after args | Error _ -> None

(* What [cc] compiles [arch] for: [arch]'s {!selection}, or, where [cc] keeps
   AVX-512 on for [native], that of the processor [native] resolves to, by name.
   [native] names each feature of a processor that states AVX10.1, and LLVM 21
   applies a named AVX10.1 after every other feature, so that [-mno-avx512f]
   leaves AVX-512 on; for a processor named, it turns AVX-512 off, AVX10.1 with
   it. The code is that processor's, without AVX-512 and the features [native]
   names beyond the processor's. *)
let settled cc arch =
  let selected = selection arch in
  match String.split_on_char ',' arch with
  | machine :: "native" :: rest when keeps cc selected -> (
      match target_cpu (own_statement cc selected) with
      | Some cpu -> selection (String.concat "," (machine :: cpu :: rest))
      | None -> selected)
  | _ -> selected

(* [f cc selected] for the Clang [cc] and [arch], where [selected] is what that
   Clang compiles [arch] for, asked once per process. *)
let per_clang f =
  let lock = Mutex.create () and known = Hashtbl.create 4 in
  fun cc arch ->
    Mutex.protect lock @@ fun () ->
    match Hashtbl.find_opt known (cc, arch) with
    | Some r -> r
    | None ->
        let r = f cc (settled cc arch) in
        Hashtbl.replace known (cc, arch) r;
        r

(* The vector registers of what Clang compiles [arch] for. On x86_64: 16
   registers, 32 with AVX-512, of 32 bytes with AVX and 16 without. AVX-512's
   registers hold 64 bytes, but a processor with them is taken at 32 until one
   is measured: tolk renders its vectors, so Clang's preference for 256 bits on
   such processors does not choose for it. On arm64, NEON's 32 registers of 16
   bytes. On riscv64, none known. A Clang that does not run gives SSE's 16 of
   16 bytes, which every x86_64 processor has. *)
let registers_of =
  per_clang @@ fun cc selected : Renderer.registers option ->
  match fst selected with
  | "x86_64" ->
      let defines = defines (macros cc selected) in
      Some
        {
          count = (if defines "__AVX512F__" then 32 else 16);
          bytes = (if defines "__AVX__" then 32 else 16);
        }
  | "arm64" -> Some { count = 32; bytes = 16 }
  | _ -> None

let registers arch =
  let cc = Setting.value Setting.cc in
  (* A malformed [arch] raises now, as {!clang}'s does. *)
  let (_ : string * string list) = selection arch in
  fun () -> registers_of cc arch

(* The conversions between halves and floats that Clang calls where an x86_64
   processor has no F16C, which compiler-rt defines and no library of the
   process holds: rounded to nearest, ties to even, a subnormal half's value
   exact in a float, and a NaN made quiet with its payload's high bits, as
   F16C and arm64's conversions make it. Clang passes a half in a vector
   register, as a [_Float16]. *)
let half_conversions =
  {|
__attribute__((visibility("hidden"))) float __extendhfsf2(_Float16 x) {
  unsigned short h;
  __builtin_memcpy(&h, &x, 2);
  unsigned int sign = (unsigned int)(h & 0x8000) << 16;
  unsigned int e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
  if (e == 0x1f) bits = sign | 0x7f800000 | (m << 13) | (m ? 0x400000u : 0);
  else if (e != 0) bits = sign | ((e + 112) << 23) | (m << 13);
  else {
    float v = (float)m * 0x1p-24f;
    __builtin_memcpy(&bits, &v, 4);
    bits |= sign;
  }
  float f;
  __builtin_memcpy(&f, &bits, 4);
  return f;
}
__attribute__((visibility("hidden"))) _Float16 __truncsfhf2(float f) {
  unsigned int x;
  __builtin_memcpy(&x, &f, 4);
  unsigned int sign = (x >> 16) & 0x8000, a = x & 0x7fffffff, h;
  if (a > 0x7f800000) h = sign | 0x7e00 | ((a >> 13) & 0x3ff);
  else if (a >= 0x477ff000) h = sign | 0x7c00;
  else if (a >= 0x38800000)
    h = sign | ((a + 0xfff + ((a >> 13) & 1) - 0x38000000) >> 13);
  else if (a > 0x33000000) {
    unsigned int shift = 126 - (a >> 23), m = (a & 0x7fffff) | 0x800000;
    unsigned int q = m >> shift, rem = m & ((1u << shift) - 1);
    unsigned int tie = 1u << (shift - 1);
    if (rem > tie || (rem == tie && (q & 1))) q++;
    h = sign | q;
  } else h = sign;
  unsigned short s = (unsigned short)h;
  _Float16 r;
  __builtin_memcpy(&r, &s, 2);
  return r;
}
|}

(* Whether an object for [arch] defines {!half_conversions}: an x86_64
   processor without F16C, or a Clang that does not run. *)
let converts_halves =
  per_clang @@ fun cc selected ->
  fst selected = "x86_64" && not (defines (macros cc selected) "__F16C__")

(* Whether [macros], of what Clang compiles an x86_64 [selected] for, have no
   [_Float16] since they lack SSE2: a kernel converts its halves with it, and an
   object for a processor without F16C defines the conversions with it
   ({!half_conversions}). A Clang that does not run defines not even
   [__x86_64__]: its compile fails saying why. *)
let without_sse2 macros selected =
  fst selected = "x86_64"
  && defines macros "__x86_64__"
  && (not (defines macros "__FLT16_MAX__"))
  && not (defines macros "__SSE2__")

(* Whether Clang keeps AVX-512 on for what it compiles an x86_64 [arch] for,
   where [avx512_off] turns it off, as LLVM 21 does for an AVX10.1 that [arch]
   names: its masked ternary logic would compile into wrong values, so nothing
   compiles. *)
let keeps_avx512 = per_clang keeps

(* Whether a Clang of [release], as its macros name it, chooses the wrong zero.
   Before LLVM 16, InstCombine reads a selection between a value and a zero,
   under a strict comparison of that value with a zero of the other sign, as a
   minimum or a maximum, taking the two zeros for one, and writes it again with
   a comparison that holds at zero:

   [x > 0.0f ? -0.0f : x] becomes [x >= 0.0f ? -0.0f : x], [-0.0] at [x = +0.0].

   LLVM writes a comparison's [-0.0] as [+0.0], so a selection of [-0.0] can
   meet it whichever zero the source compares with, and LLVM moves a selection
   of integers that become floats onto the floats. So under Clang 14 and 15, a
   kernel's bfloat16 [where (c < x) c x], a selection of storage bits, computed
   [-0.0] at [c = -0.0] and [x = +0.0]. Which selections meet it depends on the
   order LLVM rewrites them in, so no source avoids it, and no option turns it
   off: constrained floating point, which hides the comparison from it, makes
   every kernel scalar, and Clang 15 crashes on it for halves. Clang 16.0.4,
   17.0.6 and 19.1.7 and Apple's 17.0.0 choose the right zero. LLVM fixed it
   (LLVM issue 57357) in August 2022, after LLVM 15 branched. Apple numbers its
   own releases: its Clang 15 (Xcode 15) is built on LLVM 16, and its 14.0.3 on
   LLVM 15. A Clang whose preprocessor gives no macros compiles nothing and says
   why. *)
let wrong_zeros = function
  | Some (Llvm r) -> r < 16
  | Some (Apple r) -> r < 15
  | None -> false

let waits = Atomic.make 0
let settle_waits () = Atomic.get waits

let clang arch =
  (* Every question of Clang goes to [cc], whatever {!Setting.cc} names
     later. *)
  let cc = Setting.value Setting.cc in
  (* A malformed [arch] raises now; what Clang compiles it for is asked when the
     table or a compile first needs it. *)
  let (_ : string * string list) = selection arch in
  (* Settled once, by the first of the domains that compile with it; the others
     wait for it, each wait counted ({!settle_waits}). *)
  let selected =
    let lock = Mutex.create () and settled_to = ref None in
    fun () ->
      if not (Mutex.try_lock lock) then (
        Atomic.incr waits;
        Mutex.lock lock);
      Fun.protect ~finally:(fun () -> Mutex.unlock lock) @@ fun () ->
      match !settled_to with
      | Some selected -> selected
      | None when Result.is_error (dumped cc (selection arch)) ->
          (* Without the preprocessor's macros [native] is not resolved, so
             nothing is kept: the next compile settles it again. *)
          selection arch
      | None ->
          let selected = settled cc arch in
          settled_to := Some selected;
          selected
  in
  (* What Clang states it runs for a source, asked once per process. *)
  let stated () = own_statement cc (selected ()) in
  let command ~long =
    let short = cc1 (stated ()) in
    match dumped cc (selected ()) with
    | Ok g when wrong_zeros (given_release g) ->
        Error
          (Printf.sprintf
             "%s: %s chooses the wrong zero for a selection between a value \
              and a zero (LLVM issue 57357): install Clang 16 or later, or \
              Apple's 15 or later, and name it in CC"
             cc
             (version_line (stated ())))
    | Ok g when without_sse2 g.macros (selected ()) ->
        Error
          (Printf.sprintf
             "%s has no _Float16 for %s, which turns SSE2 off: x86_64's \
              _Float16, with which tolk's kernels convert halves, needs SSE2"
             cc arch)
    | Error why when Result.is_ok short ->
        (* A Clang that runs, whose release is unknown, compiles nothing. *)
        Error
          (Printf.sprintf "%s: its release is unknown, as its macros are: %s" cc
             why)
    | _ when keeps_avx512 cc arch ->
        Error
          (Printf.sprintf
             "%s keeps AVX-512 on for %s after -mno-avx512f, and its masked \
              ternary logic computes wrong values before LLVM 23 (LLVM issue \
              157929): name a processor or level without the avx10.1 feature, \
              or use LLVM 22 or later"
             cc arch)
    | _ -> if long then Result.map without_loop_vectorizer short else short
  in
  (* An object is a function of its source, of the Clang, of the command its
     length selects and of the conversions it defines, so the table names what
     Clang states, its version and installation among it, both commands, or why
     Clang states none, the length and the conversions. *)
  let table () =
    let named = function
      | Ok (prog, args) -> String.concat "\n" (prog :: args)
      | Error stated -> stated
    in
    let identity =
      String.concat "\n"
        [
          stated ();
          named (command ~long:false);
          named (command ~long:true);
          string_of_int long_source;
          (if converts_halves cc arch then half_conversions else "");
        ]
    in
    Printf.sprintf "compile_%s_obj_%s_%s" cc
      (String.map (function ',' -> '_' | c -> c) arch)
      (Digest.to_hex (Digest.string identity))
  in
  (* Sources of each length class compile in one run, with that class's
     arguments; their objects come back in the sources' order. *)
  let compile srcs =
    let long src = String.length src > long_source in
    let whole src =
      if converts_halves cc arch then src ^ half_conversions else src
    in
    let run longer =
      match List.filter (fun src -> long src = longer) srcs with
      | [] -> []
      | some -> (
          match command ~long:longer with
          | Ok command -> compile_files command (List.map whole some)
          | Error why -> raise (Renderer.Compiler.Compile_error why))
    in
    let longs = ref (run true) and shorts = ref (run false) in
    List.map
      (fun src ->
        let objs = if long src then longs else shorts in
        let obj = List.hd !objs in
        objs := List.tl !objs;
        obj)
      srcs
  in
  Renderer.Compiler.batch ~cachekey:table ~disassemble:Helpers.cpu_objdump
    compile
