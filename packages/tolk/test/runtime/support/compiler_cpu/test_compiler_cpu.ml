open Windtrap
open Tolk
module Compiler = Renderer.Compiler

(* The host's target, as the engine gives it. *)
let host_target = Tolk_engine.target Rig.host

(* Sources *)

let increment = "int increment(int x) { return x + 1; }"

let add =
  {|void add(float *restrict a, const float *restrict b, const float *restrict c) {
  for (int i = 0; i < 16; i++) a[i] = b[i] + c[i];
}|}

let square_root = "float root(float x) { return __builtin_sqrtf(x); }"

(* A loop over a count the function takes, which Clang's loop vectorizer packs
   into vectors of four floats. *)
let add_n =
  {|void add_n(float *restrict a, const float *restrict b, const float *restrict c, int n) {
  for (int i = 0; i < n; i++) a[i] = b[i] + c[i];
}
|}

(* The length past which a source compiles without the loop vectorizer, as
   compiler_cpu.mli states it. *)
let long_source = 400_000

(* [add_n] padded with a comment to [n] bytes. *)
let add_n_of n =
  let pad = n - String.length add_n - String.length "/**/" in
  add_n ^ "/*" ^ String.make pad ' ' ^ "*/"

(* A function that needs more registers than a processor has, so that the
   register allocator reaches for every register it may use. *)
let pressure =
  let n = 30 in
  let b = Buffer.create 4096 in
  Buffer.add_string b "void press(volatile long *a) {\n";
  for i = 0 to n - 1 do
    Printf.bprintf b "  long v%d = a[%d];\n" i i
  done;
  for i = 0 to n - 1 do
    Printf.bprintf b "  a[%d] = %s;\n" i
      (String.concat " ^ "
         (List.init n (fun j -> Printf.sprintf "(v%d * %d)" j (i + j + 1))))
  done;
  Buffer.add_string b "}\n";
  Buffer.contents b

(* Part of the kernel rune compiles for a uint4 sort along the last axis of a
   strided view: four gathers at clamped indices, two of 2 minus the clamp and
   two of 7 minus it, and the nibbles they read, selected in pairs. *)
let clamped_gathers =
  {|typedef unsigned char unsigned_char2 __attribute__((aligned(2),ext_vector_type(2)));
typedef long long2 __attribute__((aligned(16),ext_vector_type(2)));
typedef unsigned long unsigned_long2 __attribute__((aligned(16),ext_vector_type(2)));
typedef signed char signed_char2 __attribute__((aligned(2),ext_vector_type(2)));
__attribute__((min_vector_width(512))) void r_3_2(unsigned char* restrict data0_3, long* restrict data1_8, unsigned char* restrict data2_8) {
  long val2 = (*(data1_8+5));
  int alu2 = (((int)((val2&3l)))^-1);
  int alu3 = ((alu2<-3)?-3:alu2);
  unsigned char val3 = (*(data2_8+(2-(alu3^-1))));
  long val4 = (*(data1_8+6));
  int alu4 = (((int)((val4&3l)))^-1);
  int alu5 = ((alu4<-3)?-3:alu4);
  unsigned char val5 = (*(data2_8+(2-(alu5^-1))));
  long val6 = (*(data1_8+0));
  int alu6 = (((int)((val6&3l)))^-1);
  int alu7 = ((alu6<-3)?-3:alu6);
  unsigned char val7 = (*(data2_8+(7-(alu7^-1))));
  long val8 = (*(data1_8+1));
  int alu8 = (((int)((val8&3l)))^-1);
  int alu9 = ((alu8<-3)?-3:alu8);
  unsigned char val9 = (*(data2_8+(7-(alu9^-1))));
  unsigned_char2 cast0 = (unsigned_char2){((unsigned char)(0u)),((unsigned char)(4u))};
  unsigned_char2 alu12 = (__builtin_convertvector((__builtin_bit_cast(unsigned_long2, (long2)(((long2){val6,val8}&3l)))<3ul), signed_char2)?((unsigned_char2){val7,val9}&((unsigned char)(15u))):((unsigned char)(0u)));
  unsigned_char2 alu13 = ((alu12&((unsigned char)(15u)))<<cast0);
  unsigned_char2 alu16 = (__builtin_convertvector((__builtin_bit_cast(unsigned_long2, (long2)(((long2){val2,val4}&3l)))<3ul), signed_char2)?((unsigned_char2){val3,val5}&((unsigned char)(15u))):((unsigned char)(0u)));
  unsigned_char2 alu17 = ((alu16&((unsigned char)(15u)))<<cast0);
  *(data0_3+0) = (alu13[0]+alu13[1]);
  *(data0_3+2) = (alu17[0]+alu17[1]);
}|}

(* Part of the kernel rune compiles for casts from float16: in its first row
   the bits of each half as a double, in the others the half converted to a
   short, saturated, 0 at NaN. *)
let half_casts =
  {|void casts(unsigned long* restrict out, __fp16* restrict in, int n) {
  for (int row = 0; row < 21; row++) {
    for (int i = 0; i < n; i++) {
      __fp16 h = in[i];
      float f = ((float)(h));
      _Bool below = (f<-32768.0f);
      _Bool within = (f<32768.0f);
      float clamped = (((f!=f)|(below|(within==0)))?0.0f:f);
      short s = (below?((short)(-32768)):(within?((short)(clamped)):((short)(32767))));
      short s2 = ((f==f)?s:((short)(0)));
      unsigned long d = __builtin_bit_cast(unsigned long, (double)(h));
      out[(row*230)+i] = ((row==0)?d:((unsigned long)(__builtin_bit_cast(unsigned short, s2))));
    }
  }
}|}

(* The kernels of kernels.golden, each a linear program named after its kernel,
   whose sources are the kernel's nodes in order. *)
let kernels = lazy (Ops.src (Golden.sink "kernels.golden"))

let kernel name =
  let named l =
    match Ops.arg l with Ops.String s -> s = name | _ -> false
  in
  Ops.src (List.find named (Lazy.force kernels))

let target arch =
  {
    Helpers.Target.device = "CPU";
    renderer = "CLANG";
    arch;
    interface = "";
    indices = "";
  }

let rendered arch name = (Cstyle.clang (target arch)).render (kernel name)

(* Compiling *)

let on_macos_arm64 () =
  if Platform.system <> "macosx" || Platform.architecture <> "arm64" then
    skip ~reason:"not an Apple processor" ()

let host_machine =
  match Platform.architecture with
  | "amd64" -> "x86_64"
  | "riscv" -> "riscv64"
  | machine -> machine

let has_infix ~affix s =
  let n = String.length affix in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = affix || at (i + 1))
  in
  at 0

(* [compile arch src] is [src] compiled for [arch], or skips the test if the
   installed Clang has no backend for [arch]'s machine. *)
let compile arch src =
  let clang = Compiler_cpu.clang arch in
  match Compiler.compile clang src with
  | lib -> lib
  | exception Compiler.Compile_error msg
    when List.exists
           (fun affix -> has_infix ~affix msg)
           [ "No available targets"; "unable to create target" ] ->
      skip ~reason:("Clang has no backend for " ^ arch) ()

(* [disassembly arch src] is what disassembling [src]'s object prints, or skips
   the test if the installed objdump cannot read [arch]'s machine. *)
let disassembly arch src =
  let lib = compile arch src in
  ignore (output ());
  (try Compiler.disassemble (Compiler_cpu.clang arch) lib
   with Failure msg when has_infix ~affix:"can't disassemble" msg ->
     skip ~reason:("objdump cannot read " ^ arch) ());
  flush stdout;
  output ()

(* The words of a disassembly, split at everything but letters and digits, so
   that the register [x18] is a word and the offset [#0x18] is not. *)
let words text =
  let is_word = function
    | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' -> true
    | _ -> false
  in
  let b = Buffer.create 16 and ws = ref [] in
  let flush () =
    if Buffer.length b > 0 then (
      ws := Buffer.contents b :: !ws;
      Buffer.clear b)
  in
  String.iter
    (fun c -> if is_word c then Buffer.add_char b c else flush ())
    text;
  flush ();
  !ws

let mentions word text = List.mem word (words text)

let uses_vectors text =
  List.exists
    (fun w ->
      String.length w > 1 && w.[0] = 'v' && w.[1] >= '0' && w.[1] <= '9')
    (words text)

(* ELF headers *)

let u8 s i = Char.code s.[i]
let u16 s i = u8 s i lor (u8 s (i + 1) lsl 8)

let is_relocatable_elf64 ~machine lib =
  equal string "\x7fELF" (String.sub lib 0 4);
  equal ~msg:"class" int 2 (u8 lib 4);
  equal ~msg:"byte order" int 1 (u8 lib 5);
  equal ~msg:"object type" int 1 (u16 lib 16);
  equal ~msg:"machine" int machine (u16 lib 18)

let elf_machine = function
  | "x86_64" -> 62
  | "arm64" -> 183
  | "riscv64" -> 243
  | m -> invalid_arg ("no ELF machine for " ^ m)

(* Objects *)

let objects =
  group "objects"
    [
      cases "compiles to a relocatable ELF object for" ~id:Fun.id
        [ "x86_64,x86-64"; "arm64,generic"; "riscv64,rv64g" ] (fun arch ->
          let machine = List.hd (String.split_on_char ',' arch) in
          is_relocatable_elf64 ~machine:(elf_machine machine)
            (compile arch increment));
      test "native is the host's processor" (fun () ->
          is_relocatable_elf64 ~machine:(elf_machine host_machine)
            (compile (host_machine ^ ",native") increment));
      test "native on riscv64 is rv64g" (fun () ->
          equal string
            (compile "riscv64,rv64g" increment)
            (compile "riscv64,native" increment));
      test "one source compiles to the same bytes" (fun () ->
          let arch = host_machine ^ ",native" in
          equal string (compile arch add) (compile arch add));
      cases "a square root is one instruction on" ~id:fst
        [ ("x86_64,x86-64", "sqrtss"); ("arm64,generic", "fsqrt") ]
        (fun (arch, instruction) ->
          let text = disassembly arch square_root in
          is_true ~msg:text (mentions instruction text);
          not_contains ~sub:"sqrtf" text);
    ]

(* tinygrad's tests *)

let tinygrad =
  group "as tinygrad"
    [
      cases "the add kernel moves vectors with vmov iff AVX is on:" ~id:fst
        [ ("x86_64,x86-64,avx", true); ("x86_64,x86-64,-avx", false) ]
        (fun (arch, vmov) ->
          let text = disassembly arch (rendered arch "add") in
          equal ~msg:text bool vmov (has_infix ~affix:"vmov" text));
      test "a half addition on an Apple processor converts nothing" (fun () ->
          on_macos_arm64 ();
          let arch = host_target.arch in
          not_contains ~sub:"fcvt" (disassembly arch (rendered arch "half_add")));
    ]

(* Execution *)

let floats xs = Array.map (fun x -> `Float x) xs
let values = array Dtypes.value
let host = lazy (Cstyle.clang host_target)
let loaded name = Run.program (Lazy.force host) (kernel name)

let execution =
  group "execution on the host"
    [
      test "a compiled kernel adds two buffers" (fun () ->
          let a = Array.init 16 Float.of_int
          and b = Array.init 16 (fun i -> Float.of_int (100 - i)) in
          let out =
            Run.on_host (loaded "add") [ (1, floats a); (2, floats b) ]
          in
          equal values (floats (Array.make 16 100.)) (List.assoc 0 out));
      test "a compiled square root runs without a library" (fun () ->
          let squares = Array.init 16 (fun i -> Float.of_int (i * i)) in
          let out = Run.on_host (loaded "sqrt") [ (1, floats squares) ] in
          equal values (floats (Array.init 16 Float.of_int)) (List.assoc 0 out));
      test "a product and a sum round twice, never fused" (fun () ->
          (* (1 + 2^-12)^2 rounds to 1 + 2^-11, a tie to even, so the sum with
             -(1 + 2^-11) is 0; a fused multiply-add keeps the 2^-24. *)
          let x = 1. +. 0x1p-12 in
          let out =
            Run.on_host (loaded "muladd")
              [
                (1, floats (Array.make 16 x));
                (2, floats (Array.make 16 x));
                (3, floats (Array.make 16 (-.(1. +. 0x1p-11))));
              ]
          in
          equal values (floats (Array.make 16 0.)) (List.assoc 0 out));
    ]

(* Features *)

let features =
  group "features"
    [
      cases "a half addition on arm64 converts iff fp16 is off:" ~id:fst
        [ ("arm64,generic", true); ("arm64,generic,fp16", false) ]
        (fun (arch, fcvt) ->
          let text = disassembly arch (rendered arch "half_add") in
          equal ~msg:text bool fcvt (mentions "fcvt" text));
      test "arm64 vectorizes with its SIMD registers" (fun () ->
          is_true (uses_vectors (disassembly "arm64,generic" add)));
      test "-simd on arm64 disables them" (fun () ->
          let text = disassembly "arm64,generic,-simd" add in
          is_false ~msg:text (uses_vectors text));
      test "a feature on riscv64 is an extension the processor gains" (fun () ->
          not_equal string
            (compile "riscv64,rv64g" add)
            (compile "riscv64,rv64g,c" add));
      (* Clang 18 crashed splitting a vector of more than 16 halves that it
         negates, takes the absolute value of or copies a sign to, for these
         processors, which prefer 256-bit vectors. *)
      cases "a kernel over more than 16 halves compiles for AVX-512"
        ~id:(fun (arch, name) -> arch ^ " " ^ name)
        (List.concat_map
           (fun arch ->
             List.map
               (fun name -> (arch, name))
               [ "half_neg"; "half_abs"; "half_copysign"; "half_min" ])
           [
             "x86_64,skylake-avx512";
             "x86_64,icelake-server";
             "x86_64,sapphirerapids";
           ])
        (fun (arch, name) ->
          is_relocatable_elf64 ~machine:(elf_machine "x86_64")
            (compile arch (rendered arch name)));
      (* LLVM 18's straight-line vectorizer narrowed the four indices into
         invalid IR for these processors, and Clang 18 crashed on it. *)
      cases "four clamped gathers compile for AVX-512 on" ~id:Fun.id
        [ "x86_64,x86-64-v4"; "x86_64,skylake-avx512"; "x86_64,znver4" ]
        (fun arch ->
          is_relocatable_elf64 ~machine:(elf_machine "x86_64")
            (compile arch clamped_gathers));
      (* Clang 18 crashed on this loop for these processors, which have
         AVX-512's half arithmetic. *)
      cases "a loop of half casts compiles with AVX-512's half arithmetic on"
        ~id:Fun.id
        [ "x86_64,sapphirerapids"; "x86_64,x86-64-v4,avx512fp16" ]
        (fun arch ->
          is_relocatable_elf64 ~machine:(elf_machine "x86_64")
            (compile arch half_casts));
      (* SSE2 adds four floats with [addps] and one with [addss]: a packed
         loop adds its last floats one at a time. *)
      cases
        "a loop over a count it takes is packed into vectors iff its source \
         is at most 400000 bytes long:"
        ~id:(fun (n, _) -> string_of_int n)
        [
          (String.length add_n, [ "addps"; "addss" ]);
          (long_source, [ "addps"; "addss" ]);
          (long_source + 1, [ "addss" ]);
          (2 * long_source, [ "addss" ]);
        ]
        (fun (n, adds) ->
          let src = if n = String.length add_n then add_n else add_n_of n in
          equal int ~msg:"the source's length" n (String.length src);
          let text = disassembly "x86_64,x86-64" src in
          equal (list string) adds
            (List.filter (fun w -> mentions w text) [ "addps"; "addss" ]));
      test "arm64 leaves x18 alone" (fun () ->
          let text = disassembly "arm64,generic" pressure in
          is_false ~msg:"x18" (mentions "x18" text);
          is_false ~msg:"w18" (mentions "w18" text));
    ]

(* Flags *)

let llvm18 = "Ubuntu clang version 18.1.3 (1ubuntu1)"
let x86_64_elf = "--target=x86_64-none-unknown-elf"

let windows =
  [
    "--target=x86_64-w64-windows-gnu-elf";
    "-mno-stack-arg-probe";
    "-mno-red-zone";
  ]

let llvm18_off = [ "-mno-avx10.1-512"; "-mno-avx10.1-256"; "-mno-avx512fp16" ]

let flags =
  group "flags"
    [
      cases "under LLVM 18, x86_64 ends by disabling AVX10.1 and avx512fp16:"
        ~id:fst
        [
          ("x86_64,native", [ x86_64_elf; "-march=native" ] @ llvm18_off);
          ( "x86_64,graniterapids,avx10.1-256",
            [ x86_64_elf; "-march=graniterapids"; "-mavx10.1-256" ]
            @ llvm18_off );
          ( "x86_64,sapphirerapids,avx512fp16,-avx",
            [ x86_64_elf; "-march=sapphirerapids"; "-mavx512fp16"; "-mno-avx" ]
            @ llvm18_off );
          ("x86_64-windows,native", windows @ [ "-march=native" ] @ llvm18_off);
        ]
        (fun (arch, expected) ->
          equal (list string) expected
            (Compiler_cpu.flags ~version:llvm18 arch));
      test "LLVM's own release of 18 is LLVM 18" (fun () ->
          equal (list string)
            ([ x86_64_elf; "-march=native" ] @ llvm18_off)
            (Compiler_cpu.flags ~version:"clang version 18.1.8"
               "x86_64,native"));
      cases "x86_64,native takes its own arguments alone under" ~id:Fun.id
        [
          "Debian clang version 19.1.7 (3+b1)";
          "Homebrew clang version 20.1.8";
          "clang version 17.0.6";
          "Apple clang version 18.0.0 (clang-1800.0.31.1)";
          "";
        ]
        (fun version ->
          equal (list string)
            [ x86_64_elf; "-march=native" ]
            (Compiler_cpu.flags ~version "x86_64,native"));
      cases "another machine under LLVM 18 takes its own arguments alone:"
        ~id:fst
        [
          ( "arm64,generic,fp16",
            [
              "--target=arm64-none-unknown-elf";
              "-ffixed-x18";
              "-mcpu=generic+fp16";
            ] );
          ( "riscv64,rv64g,c",
            [ "--target=riscv64-none-unknown-elf"; "-march=rv64g_c" ] );
        ]
        (fun (arch, expected) ->
          equal (list string) expected
            (Compiler_cpu.flags ~version:llvm18 arch));
      test "a malformed architecture is refused" (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"'x86_64'")
            (fun () -> Compiler_cpu.flags ~version:llvm18 "x86_64"));
    ]

(* Architecture strings *)

let architectures =
  group "architectures"
    [
      cases "an architecture of fewer than two fields is refused, named:"
        ~id:(Printf.sprintf "%S") [ ""; "x86_64"; "arm64" ] (fun arch ->
          raises_match
            (Exn.invalid_arg ~sub:("'" ^ arch ^ "'"))
            (fun () -> Compiler_cpu.clang arch));
      cases "another machine is refused, named:" ~id:fst
        [
          ("mips,native", "mips");
          ("aarch64,native", "aarch64");
          ("X86_64,native", "X86_64");
        ]
        (fun (arch, machine) ->
          raises_match (Exn.invalid_arg ~sub:machine) (fun () ->
              Compiler_cpu.clang arch));
    ]

(* Errors *)

let errors =
  group "errors"
    [
      test "a rejected source raises Compile_error with Clang's diagnostics"
        (fun () ->
          match
            Compiler.compile
              (Compiler_cpu.clang (host_machine ^ ",native"))
              "int f(void) { return undeclared_name; }"
          with
          | _ -> fail "Clang accepted an undeclared name"
          | exception Compiler.Compile_error msg ->
              in_order ~subs:[ "error"; "undeclared_name" ] msg);
    ]

(* Without Clang: CC is read once, when the first compile runs, so these tests
   run in a process of their own, where CC names a program that does not
   exist. *)

let missing_clang = "/nonexistent/clang"

(* The macros of each processor name its features; Clang's statement names the
   processor alone. *)
let registers =
  let file arch =
    Option.map
      (fun (r : Renderer.registers) -> (r.count, r.bytes))
      (Compiler_cpu.registers arch)
  in
  cases "the vector registers, their number and bytes, of" ~id:fst
    [
      ("x86_64,x86-64", Some (16, 16));
      ("x86_64,x86-64-v3", Some (16, 32));
      ("x86_64,x86-64-v4", Some (32, 32));
      ("x86_64,znver4", Some (32, 32));
      ("x86_64-windows,x86-64-v3", Some (16, 32));
      ("arm64,apple-m1", Some (32, 16));
      ("riscv64,rv64g", None);
    ]
    (fun (arch, expected) -> equal (option (pair int int)) expected (file arch))

let without_clang =
  tag "no-clang"
  @@ group "without Clang"
       [
         test "a compile raises Compile_error naming the program CC names"
           (fun () ->
             raises_match
               (function
                 | Compiler.Compile_error msg ->
                     has_infix ~affix:missing_clang msg
                 | _ -> false)
               (fun () ->
                 Compiler.compile
                   (Compiler_cpu.clang (host_machine ^ ",native"))
                   increment));
         test "an x86_64 processor has SSE's 16 registers of 16 bytes"
           (fun () ->
             equal
               (option (pair int int))
               (Some (16, 16))
               (Option.map
                  (fun (r : Renderer.registers) -> (r.count, r.bytes))
                  (Compiler_cpu.registers "x86_64,x86-64-v3")));
         test "a cached object is served without a Clang that runs" (fun () ->
             let clang = Compiler_cpu.clang (host_machine ^ ",native") in
             let table = Option.get (Compiler.cachekey clang) in
             Helpers.Diskcache.put ~table increment "cached object";
             equal string "cached object"
               (Compiler.compile_cached clang increment));
         test "an object cached by the default compiler is not served under CC"
           (fun () ->
             let cachedb = Helpers.cachedb in
             let table =
               match
                 Disk_cache.child ~env:[ ("CC", "clang") ] ~cachedb "table"
               with
               | Ok table -> table
               | Error err ->
                   failf "the default compiler named no table: %s" err
             in
             let src = increment ^ "\n/* compiled by clang */\n" in
             Helpers.Diskcache.put ~table src "clang's object";
             raises_match
               (function Compiler.Compile_error _ -> true | _ -> false)
               (fun () ->
                 Compiler.compile_cached
                   (Compiler_cpu.clang (host_machine ^ ",native"))
                   src));
       ]

(* Disassembly *)

let disassembly_ =
  group "disassembly"
    [
      test "prints what objdump prints of the object" (fun () ->
          let text = disassembly "x86_64,x86-64" increment in
          in_order ~subs:[ "elf64-x86-64"; "<increment>:" ] text);
    ]

(* The cache *)

let ccache_off f = Setting.context [ B (Setting.ccache, false) ] f

(* The table of the host's compiler, as a process of its own names it. *)
let host_table () =
  print_string
    (Option.get
       (Compiler.cachekey (Compiler_cpu.clang (host_machine ^ ",native"))))

let parts = [ ("table", host_table) ]

(* [table_in ()] is the table a process names, from a directory of its own. *)
let table_in () =
  let cachedb = Filename.concat (Sys.getcwd ()) "unused_cache" in
  chdir (temp_dir ());
  Disk_cache.child ~cachedb "table"

let is_hex c = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')

(* [named ~prefix table] checks that [table] is [prefix] then a digest. *)
let named ~prefix table =
  let n = Int.min (String.length prefix) (String.length table) in
  equal string ~msg:"the name" prefix (String.sub table 0 n);
  let digest = String.sub table n (String.length table - n) in
  equal int ~msg:"the digest's length" 32 (String.length digest);
  equal (list char) ~msg:"the digest's other characters" []
    (List.filter (fun c -> not (is_hex c)) (List.of_seq (String.to_seq digest)))

let cache =
  group "cache"
    [
      cases "objects are cached in the table of the architecture and a digest"
        ~id:fst
        [
          ("x86_64,znver2", "compile_clang_obj_x86_64_znver2_");
          ( "x86_64,znver2,avx,-avx512f",
            "compile_clang_obj_x86_64_znver2_avx_-avx512f_" );
          ("arm64,native", "compile_clang_obj_arm64_native_");
        ]
        (fun (arch, prefix) ->
          named ~prefix
            (Option.get (Compiler.cachekey (Compiler_cpu.clang arch))));
      test "the table does not depend on the working directory" (fun () ->
          let a = table_in () in
          equal Disk_cache.outcome a (table_in ()));
      test "the table is named with ccache off" (fun () ->
          let table () =
            Compiler.cachekey (Compiler_cpu.clang "arm64,native")
          in
          equal (option string) (table ()) (ccache_off table));
    ]

let () =
  Disk_cache.play parts;
  exit
    (run "Tolk.Compiler_cpu"
       [
         objects;
         tinygrad;
         execution;
         features;
         flags;
         registers;
         architectures;
         errors;
         without_clang;
         disassembly_;
         cache;
       ])
