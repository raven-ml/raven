open Windtrap
open Tolk
module Compiler = Renderer.Compiler

(* The host's target, as the engine gives it. *)
let host_target = Tolk_engine.target Nx_device.host

(* Sources *)

let increment = "int increment(int x) { return x + 1; }"

let add =
  {|void add(float *restrict a, const float *restrict b, const float *restrict c) {
  for (int i = 0; i < 16; i++) a[i] = b[i] + c[i];
}|}

let square_root = "float root(float x) { return __builtin_sqrtf(x); }"

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

(* The kernels of kernels.golden, each a linear program named after its kernel,
   whose sources are the kernel's nodes in order. *)
let kernels = lazy (Ops.src (Golden.sink "kernels.golden"))

let kernel name =
  let named l =
    match Ops.arg l with Ops.Region r -> r.name = name | _ -> false
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
      cases ~name:Fun.id "compiles to a relocatable ELF object for"
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
      cases ~name:fst "a square root is one instruction on"
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
      cases ~name:fst "the add kernel moves vectors with vmov iff AVX is on:"
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
      cases ~name:fst "a half addition on arm64 converts iff fp16 is off:"
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
      test "arm64 leaves x18 alone" (fun () ->
          let text = disassembly "arm64,generic" pressure in
          is_false ~msg:"x18" (mentions "x18" text);
          is_false ~msg:"w18" (mentions "w18" text));
    ]

(* Architecture strings *)

let architectures =
  group "architectures"
    [
      cases ~name:(Printf.sprintf "%S")
        "an architecture of fewer than two fields is refused, named:"
        [ ""; "x86_64"; "arm64" ] (fun arch ->
          raises_match
            (Exn.invalid_arg ~substring:("'" ^ arch ^ "'"))
            (fun () -> Compiler_cpu.clang arch));
      cases ~name:fst "another machine is refused, named:"
        [
          ("mips,native", "mips");
          ("aarch64,native", "aarch64");
          ("X86_64,native", "X86_64");
        ]
        (fun (arch, machine) ->
          raises_match (Exn.invalid_arg ~substring:machine) (fun () ->
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

let without_clang =
  group ~tags:[ "no-clang" ] "without Clang"
    [
      test "a compile raises Compile_error naming the program CC names"
        (fun () ->
          raises_match
            (function
              | Compiler.Compile_error msg -> has_infix ~affix:missing_clang msg
              | _ -> false)
            (fun () ->
              Compiler.compile
                (Compiler_cpu.clang (host_machine ^ ",native"))
                increment));
      test "a cached object is served without running Clang" (fun () ->
          let clang = Compiler_cpu.clang (host_machine ^ ",native") in
          let table = Option.get (Compiler.cachekey clang) in
          Helpers.Diskcache.put ~table increment "cached object";
          equal string "cached object" (Compiler.compile_cached clang increment));
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

let ccache_off f = Helpers.context [ B (Helpers.ccache, false) ] f

let cache =
  group "cache"
    [
      cases ~name:fst "objects are cached in the table of the architecture"
        [
          ("x86_64,znver2", "compile_clang_obj_x86_64_znver2");
          ( "x86_64,znver2,avx,-avx512f",
            "compile_clang_obj_x86_64_znver2_avx_-avx512f" );
          ("arm64,native", "compile_clang_obj_arm64_native");
        ]
        (fun (arch, table) ->
          equal (option string) (Some table)
            (Compiler.cachekey (Compiler_cpu.clang arch)));
      test "objects are not cached without ccache" (fun () ->
          is_none
            (Compiler.cachekey
               (ccache_off (fun () -> Compiler_cpu.clang "arm64,native"))));
    ]

let () =
  exit
    (run "Tolk.Compiler_cpu"
       [
         objects;
         tinygrad;
         execution;
         features;
         architectures;
         errors;
         without_clang;
         disassembly_;
         cache;
       ])
