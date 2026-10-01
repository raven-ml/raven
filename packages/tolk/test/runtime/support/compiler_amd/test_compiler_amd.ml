open Windtrap
open Tolk
module Compiler = Renderer.Compiler

let kernel =
  {|extern "C" __attribute__((global)) void add(float *a, const float *b) {
  a[0] += b[0];
}|}

(* Instructions, which HIP would reject. *)
let assembly = ".text\n.globl f\n.type f,@function\nf:\n  s_endpgm\n"

let has_infix ~affix s =
  let n = String.length affix in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = affix || at (i + 1))
  in
  at 0

let rejection f =
  match f () with _ -> None | exception Compiler.Compile_error msg -> Some msg

(* comgr's absence is a Compile_error that names the variable to set. *)
let absent msg = has_infix ~affix:"COMGR_PATH" msg

(* [with_comgr f] is [f ()], or skips the test if comgr is not on the
   machine. *)
let with_comgr f =
  match f () with
  | v -> v
  | exception Compiler.Compile_error msg when absent msg -> skip ~reason:msg ()

(* The cache *)

let ccache_off f = Helpers.context [ B (Helpers.ccache, false) ] f

let cache =
  group "cache"
    [
      test "code objects are cached in the table of hip and the architecture"
        (fun () ->
          equal (option string) (Some "compile_hip_gfx1100")
            (Compiler.cachekey (Compiler_amd.hip "gfx1100")));
      test "code objects are not cached without ccache" (fun () ->
          is_none
            (Compiler.cachekey
               (ccache_off (fun () -> Compiler_amd.hip "gfx1100"))));
    ]

(* Without comgr on the machine (D15) *)

let without_comgr =
  group "without comgr on the machine"
    [
      test "a compile raises Compile_error naming comgr and COMGR_PATH"
        (fun () ->
          match
            rejection (fun () ->
                Compiler.compile (Compiler_amd.hip "gfx1100") kernel)
          with
          | None -> skip ~reason:"comgr is on the machine" ()
          | Some msg -> in_order ~subs:[ "comgr"; "COMGR_PATH" ] msg);
    ]

(* A library that does not load (D15): these tests run in a process of their
   own, where COMGR_PATH names a file that is no library. *)

let load_failure () =
  rejection (fun () -> Compiler.compile (Compiler_amd.hip "gfx1100") kernel)

let not_a_library =
  group ~tags:[ "no-library" ] "a library that does not load"
    [
      test "making a compiler loads nothing, and a compile raises Compile_error"
        (fun () ->
          let msg = require_some (load_failure ()) in
          in_order ~subs:[ "comgr"; "not_a_library" ] msg);
      test "every compile raises the same Compile_error" (fun () ->
          let first = load_failure () in
          equal (option string) first (load_failure ());
          is_some first);
      test "compiles from several domains at once raise it" (fun () ->
          let domains = List.init 4 (fun _ -> Domain.spawn load_failure) in
          let msgs = List.map Domain.join domains in
          is_some (List.hd msgs);
          List.iter (equal (option string) (List.hd msgs)) msgs);
      test "a cached code object is served without loading comgr" (fun () ->
          let hip = Compiler_amd.hip "gfx1100" in
          let table = Option.get (Compiler.cachekey hip) in
          Helpers.Diskcache.put ~table kernel "cached code object";
          equal string "cached code object" (Compiler.compile_cached hip kernel));
    ]

(* comgr *)

let smoke =
  group "comgr"
    [
      slow "compiles a kernel to a code object" (fun () ->
          let lib =
            with_comgr (fun () ->
                Compiler.compile (Compiler_amd.hip "gfx1100") kernel)
          in
          starts_with ~affix:"\x7fELF" lib;
          contains ~sub:"add" lib);
      slow "assembles a source whose first line is .text" (fun () ->
          let lib =
            with_comgr (fun () ->
                Compiler.compile (Compiler_amd.hip "gfx1100") assembly)
          in
          starts_with ~affix:"\x7fELF" lib);
      slow "a rejected source raises Compile_error with comgr's log" (fun () ->
          match
            Compiler.compile
              (Compiler_amd.hip "gfx1100")
              "extern \"C\" __attribute__((global)) void f() { undeclared_name \
               = 1; }"
          with
          | _ -> fail "comgr accepted an undeclared name"
          | exception Compiler.Compile_error msg when absent msg ->
              skip ~reason:msg ()
          | exception Compiler.Compile_error msg ->
              contains ~sub:"undeclared_name" msg);
      slow "compiles from several domains at once" (fun () ->
          let archs = [ "gfx90a"; "gfx942"; "gfx1100"; "gfx1201" ] in
          let compile arch () =
            Compiler.compile (Compiler_amd.hip arch) kernel
          in
          ignore (with_comgr (compile "gfx1100"));
          let domains =
            List.map (fun arch -> Domain.spawn (compile arch)) archs
          in
          List.iter
            (fun lib -> starts_with ~affix:"\x7fELF" lib)
            (List.map Domain.join domains));
      slow "disassembles a code object" (fun () ->
          let hip = Compiler_amd.hip "gfx1100" in
          let lib = with_comgr (fun () -> Compiler.compile hip kernel) in
          ignore (output ());
          Compiler.disassemble hip lib;
          flush stdout;
          contains ~sub:"s_endpgm" (output ()));
    ]

let () =
  exit (run "Tolk.Compiler_amd" [ cache; without_comgr; not_a_library; smoke ])
