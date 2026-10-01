open Windtrap
open Tolk_next
module Compiler = Renderer.Compiler

let kernel name =
  Printf.sprintf
    {|#include <metal_stdlib>
using namespace metal;
kernel void %s(device int* data0, const device int* data1, const device int* data2,
               uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  for (int i = 0; i < 4; i++) data0[i] = data1[i] + data2[i];
}|}
    name

let has_infix ~affix s =
  let n = String.length affix in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = affix || at (i + 1))
  in
  at 0

let rejection f =
  match f () with _ -> None | exception Compiler.Compile_error msg -> Some msg

(* MTLCompiler's absence is a Compile_error that names the variable to set. *)
let absent msg = has_infix ~affix:"MTLCOMPILER_PATH" msg

(* [with_metal f] is [f ()], or skips the test if MTLCompiler is not on the
   machine. *)
let with_metal f =
  match f () with
  | v -> v
  | exception Compiler.Compile_error msg when absent msg -> skip ~reason:msg ()

let is_metal_library lib =
  starts_with ~affix:"MTLB" lib;
  ends_with ~affix:"ENDT" lib

(* The version of Metal the running macOS compiles kernels of, as
   [__METAL_VERSION__] writes it. *)
let metal_version () =
  let macos =
    try Helpers.system "sw_vers -productVersion"
    with Failure _ -> skip ~reason:"not macOS" ()
  in
  match int_of_string (List.hd (String.split_on_char '.' macos)) with
  | major when major >= 26 -> 400
  | major when major >= 14 -> 310
  | 13 -> 300
  | _ -> 200

(* The cache *)

let ccache_off f = Helpers.context [ B (Helpers.ccache, false) ] f

let cache =
  group "cache"
    [
      test "libraries are cached in the table compile_metal_direct" (fun () ->
          equal (option string) (Some "compile_metal_direct")
            (Compiler.cachekey (Compiler_metal.compiler ())));
      test "libraries are not cached without ccache" (fun () ->
          is_none (Compiler.cachekey (ccache_off Compiler_metal.compiler)));
    ]

(* Without MTLCompiler on the machine (D15) *)

let without_mtlcompiler =
  group "without MTLCompiler on the machine"
    [
      test
        "a compile raises Compile_error naming MTLCompiler and MTLCOMPILER_PATH"
        (fun () ->
          match
            rejection (fun () ->
                Compiler.compile (Compiler_metal.compiler ()) (kernel "add"))
          with
          | None -> skip ~reason:"MTLCompiler is on the machine" ()
          | Some msg -> in_order ~subs:[ "MTLCompiler"; "MTLCOMPILER_PATH" ] msg);
    ]

(* A library that does not load (D15): these tests run in a process of their
   own, where MTLCOMPILER_PATH names a file that is no library. *)

let load_failure () =
  rejection (fun () -> Compiler.compile (Compiler_metal.compiler ()) (kernel "add"))

let not_a_library =
  group ~tags:[ "no-library" ] "a library that does not load"
    [
      test "making a compiler loads nothing, and a compile raises Compile_error"
        (fun () ->
          let msg = require_some (load_failure ()) in
          contains ~sub:"MTLCompiler" msg;
          (* Elsewhere than on macOS, no file is loaded. *)
          if Platform.system = "macosx" then contains ~sub:"not_a_library" msg);
      test "every compile raises the same Compile_error" (fun () ->
          let first = load_failure () in
          equal (option string) first (load_failure ());
          is_some first);
      test "compiles from several domains at once raise it" (fun () ->
          let domains = List.init 4 (fun _ -> Domain.spawn load_failure) in
          let msgs = List.map Domain.join domains in
          is_some (List.hd msgs);
          List.iter (equal (option string) (List.hd msgs)) msgs);
      test "a cached library is served without loading MTLCompiler" (fun () ->
          let metal = Compiler_metal.compiler () in
          let table = Option.get (Compiler.cachekey metal) in
          Helpers.Diskcache.put ~table (kernel "add") "cached library";
          equal string "cached library"
            (Compiler.compile_cached metal (kernel "add")));
    ]

(* MTLCompiler *)

let metal =
  group "MTLCompiler"
    [
      test "compiles a kernel to a Metal library" (fun () ->
          is_metal_library
            (with_metal (fun () ->
                 Compiler.compile (Compiler_metal.compiler ()) (kernel "add"))));
      test "a source that draws warnings compiles to a Metal library" (fun () ->
          let src =
            "#include <metal_stdlib>\n\
             kernel void warned(device int* data0) { int unused = 1; data0[0] \
             = 0; }"
          in
          is_metal_library
            (with_metal (fun () -> Compiler.compile (Compiler_metal.compiler ()) src)));
      test "a rejected source raises Compile_error with the compiler's message"
        (fun () ->
          match
            Compiler.compile (Compiler_metal.compiler ()) "this is not valid metal"
          with
          | _ -> fail "the compiler accepted a source that is not Metal"
          | exception Compiler.Compile_error msg when absent msg ->
              skip ~reason:msg ()
          | exception Compiler.Compile_error msg -> contains ~sub:"error" msg);
      test "the language is the latest the running macOS compiles kernels of"
        (fun () ->
          let src =
            Printf.sprintf
              "#if __METAL_VERSION__ != %d\n#error wrong version\n#endif\n%s"
              (metal_version ()) (kernel "versioned")
          in
          is_metal_library
            (with_metal (fun () -> Compiler.compile (Compiler_metal.compiler ()) src)));
      test "a product and a sum compile under the no-contraction pragma (D25)"
        (fun () ->
          let src =
            "#include <metal_stdlib>\n\
             kernel void muladd(device float* d, const device float* a, const \
             device float* b, const device float* c) { d[0] = a[0] * b[0] + \
             c[0]; }"
          in
          is_metal_library
            (with_metal (fun () -> Compiler.compile (Compiler_metal.compiler ()) src)));
      test "fast math is off" (fun () ->
          let src =
            "#ifdef __FAST_MATH__\n#error fast math\n#endif\n" ^ kernel "exact"
          in
          is_metal_library
            (with_metal (fun () -> Compiler.compile (Compiler_metal.compiler ()) src)));
      test "compiles from several domains at once" (fun () ->
          let builds d =
            List.init 8 (fun i ->
                Compiler.compile (Compiler_metal.compiler ())
                  (kernel (Printf.sprintf "k%d_%d" d i)))
          in
          let domains = List.init 8 (fun d -> Domain.spawn (fun () -> builds d)) in
          List.iter
            (List.iter is_metal_library)
            (with_metal (fun () -> List.map Domain.join domains)));
      test "disassembly prints nothing" (fun () ->
          let metal = Compiler_metal.compiler () in
          let lib =
            with_metal (fun () -> Compiler.compile metal (kernel "add"))
          in
          ignore (output ());
          Compiler.disassemble metal lib;
          flush stdout;
          equal string "" (output ()));
    ]

let () =
  exit
    (run "Tolk_next.Compiler_metal"
       [ cache; without_mtlcompiler; not_a_library; metal ])
