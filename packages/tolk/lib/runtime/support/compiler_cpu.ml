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

(* The LLVM release of each Clang, as the last process that asked it found it:
   a hint, kept on disk while {!Setting.ccache} holds, from which a process
   picks the arguments it asks Clang to state, so that a Clang of the hinted
   release is run once. *)
let hint_table = "clang_llvm"

let hinted cc =
  if not (Setting.value Setting.ccache) then None
  else
    match Helpers.Diskcache.get ~table:hint_table cc with
    | Some l -> int_of_string_opt l
    | None | (exception (Failure _ | Sys_error _)) -> None

let hint cc llvm =
  let l = Option.fold ~none:"" ~some:string_of_int llvm in
  if Setting.value Setting.ccache then
    try Helpers.Diskcache.put ~table:hint_table cc l with Sys_error _ -> ()

(* What [cc] states it runs for [args ~llvm], where [llvm] is the release of
   LLVM it names in that statement: Clang states its version and installation
   before its commands. [args] is first asked for the hinted release; a
   statement that names another, as after an upgrade, is asked again for the
   one it names, which is then hinted. Clang is asked at most three times: a
   Clang that starts once and not again names no release the second time. The
   hint picks arguments only: what Clang states for them is what runs, and
   what a table digests. *)
let own_statement =
  let lock = Mutex.create () and stated = Hashtbl.create 4 in
  fun cc args ->
    Mutex.protect lock @@ fun () ->
    let key = (cc, args ~llvm:None) in
    match Hashtbl.find_opt stated key with
    | Some s -> s
    | None ->
        let hinted = hinted cc in
        let rec settle llvm tries =
          let s = statement cc (args ~llvm) in
          let named = llvm_major s in
          if named = llvm then (
            if llvm <> hinted then hint cc llvm;
            s)
          else if tries = 0 then s
          else settle named (tries - 1)
        in
        let s = settle hinted 2 in
        Hashtbl.replace stated key s;
        s

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
   triple, processor and features, and the processor it tunes for. *)
let selection arch =
  let machine, cpu, feats, tune =
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
  (machine, triple @ target_args @ tune)

(* The flags of a [selection] for a Clang of LLVM [llvm]. *)
let fixed ~llvm (machine, flags) =
  if machine = "x86_64" && llvm = Some 18 then flags @ llvm18_x86_off
  else flags

let flags ~version arch = fixed ~llvm:(llvm_major version) (selection arch)

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
   valid IR of wrong values: for rune's attention under ALiBi on a processor
   with AVX2, they packed two int64 differences of rows and keys converted to
   float, narrowed the unsigned low halves to 8 bits and wrote [uitofp <2 x
   i8>], so that a low half of 2^32 - 1 converted to 255 and a score was off
   by 2^32 times its slope. LLVM 18, 21 and 22 compile the kernel. Under LLVM
   18, 19 and 20 only, that vectorizer is off, so that other Clangs compile
   as before. The flags follow -O2, which would turn the vectorizers back
   on.

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
let compile_args selected ~llvm =
  let slp =
    match llvm with Some (18 | 19 | 20) -> [ "-fno-slp-vectorize" ] | _ -> []
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
  @ fixed ~llvm selected @ slp @ ldn @ idioms @ [ "-"; "-o"; "-" ]

(* The macros Clang's preprocessor defines for an x86_64 [selected], which name
   the processor's features for every processor, where Clang's statement names
   them only for [native]; none where Clang does not run. *)
let x86_macros cc selected =
  let llvm = llvm_major (own_statement cc (compile_args selected)) in
  let flags = fixed ~llvm selected in
  let macros =
    match run cc ([ "-dM"; "-E"; "-x"; "c" ] @ flags @ [ "-" ]) "" with
    | Ok (WEXITED 0, out, _) -> String.split_on_char '\n' out
    | Ok _ | Error _ -> []
  in
  fun m ->
    List.exists (String.starts_with ~prefix:("#define " ^ m ^ " ")) macros

(* [f cc arch] for each Clang and [arch], asked once per process. *)
let per_clang f =
  let lock = Mutex.create () and known = Hashtbl.create 4 in
  fun arch ->
    let cc = Setting.value Setting.cc in
    let selected = selection arch in
    Mutex.protect lock @@ fun () ->
    match Hashtbl.find_opt known (cc, arch) with
    | Some r -> r
    | None ->
        let r = f cc selected in
        Hashtbl.replace known (cc, arch) r;
        r

(* The vector registers of what Clang compiles [arch] for. On x86_64: 16
   registers, 32 with AVX-512, of 32 bytes with AVX and 16 without. AVX-512's
   registers hold 64 bytes, but a processor with them is taken at 32 until one
   is measured: tolk renders its vectors, so Clang's preference for 256 bits on
   such processors does not choose for it. On arm64, NEON's 32 registers of 16
   bytes. On riscv64, none known. A Clang that does not run gives SSE's 16 of
   16 bytes, which every x86_64 processor has. *)
let registers =
  per_clang @@ fun cc selected : Renderer.registers option ->
  match fst selected with
  | "x86_64" ->
      let defines = x86_macros cc selected in
      Some
        {
          count = (if defines "__AVX512F__" then 32 else 16);
          bytes = (if defines "__AVX__" then 32 else 16);
        }
  | "arm64" -> Some { count = 32; bytes = 16 }
  | _ -> None

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
  fst selected = "x86_64" && not (x86_macros cc selected "__F16C__")

let clang arch =
  let cc = Setting.value Setting.cc in
  let selected = selection arch in
  (* What Clang states it runs for a source, asked once per process. *)
  let stated () = own_statement cc (compile_args selected) in
  let command ~long =
    let short = cc1 (stated ()) in
    if long then Result.map without_loop_vectorizer short else short
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
          (if converts_halves arch then half_conversions else "");
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
      if converts_halves arch then src ^ half_conversions else src
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
