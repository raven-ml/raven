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

(* comgr's absence is a Failure, raised when a compiler is made, that names the
   variable to set. *)
let absent msg = has_infix ~affix:"COMGR_PATH" msg

(* [with_comgr f] is [f ()], or skips the test if comgr is not on the
   machine. *)
let with_comgr f =
  match f () with
  | v -> v
  | exception Failure msg when absent msg -> skip ~reason:msg ()

(* comgr's files *)

(* [remove path] removes the file or the directory tree at [path], following no
   symbolic link. *)
let rec remove path =
  match (Unix.lstat path).st_kind with
  | S_DIR ->
      Array.iter (fun n -> remove (Filename.concat path n)) (Sys.readdir path);
      Unix.rmdir path
  | _ -> Unix.unlink path

(* comgr writes its temporary files under TMPDIR, and its own cache under
   AMD_COMGR_CACHE_DIR, which the workers inherit: both name a directory of
   this process's own, removed when it exits, so that a compile that ends its
   worker, which leaves its temporary files behind, leaves them there. *)
let scratch =
  let dir = Filename.temp_dir "tolk-test-comgr" "" in
  at_exit (fun () -> remove dir);
  Unix.putenv "TMPDIR" dir;
  dir

let comgr_cache = Filename.concat scratch "comgr-cache"
let () = Unix.putenv "AMD_COMGR_CACHE_DIR" comgr_cache

(* The cache *)

let ccache_off f = Setting.context [ B (Setting.ccache, false) ] f

(* [named ~prefix table] checks that [table] is [prefix] then a digest. *)
let named ~prefix table =
  let n = Int.min (String.length prefix) (String.length table) in
  equal string ~msg:"the name" prefix (String.sub table 0 n);
  equal int ~msg:"the digest's length" 32 (String.length table - n)

(* These tests run where COMGR_PATH names a file, which the compiler finds. *)
let cache =
  tag "no-library"
  @@ group "cache"
    [
      test
        "code objects are cached in the table of hip, the architecture and a \
         digest" (fun () ->
          named ~prefix:"compile_hip_gfx1100_"
            (Option.get (Compiler.cachekey (Compiler_amd.hip "gfx1100"))));
      test "the table is named with ccache off" (fun () ->
          let table () = Compiler.cachekey (Compiler_amd.hip "gfx1100") in
          equal (option string) (table ()) (ccache_off table));
    ]

(* Without comgr on the machine *)

let without_comgr =
  group "without comgr on the machine"
    [
      test "making a compiler raises Failure naming comgr and COMGR_PATH"
        (fun () ->
          match Compiler_amd.hip "gfx1100" with
          | _ -> skip ~reason:"comgr is on the machine" ()
          | exception Failure msg ->
              in_order ~subs:[ "comgr"; "COMGR_PATH" ] msg);
    ]

(* Workers *)

(* [with_signal s behavior f] is [f ()] run with [behavior] for [s]. *)
let with_signal s behavior f =
  let previous = Sys.signal s behavior in
  Fun.protect ~finally:(fun () -> Sys.set_signal s previous) f

(* [f ()] with SIGPIPE's default action, which ends the process: a process may
   start with it ignored. *)
let sigpipe_kills f = with_signal Sys.sigpipe Signal_default f

(* The pids of this process's children, which pgrep lists, in order; the test
   skips where pgrep does not run. *)
let children () =
  let pid = string_of_int (Unix.getpid ()) in
  match Unix.open_process_args_in "pgrep" [| "pgrep"; "-P"; pid |] with
  | exception Unix.Unix_error _ -> skip ~reason:"pgrep does not run" ()
  | ic -> (
      let listed = In_channel.input_all ic in
      match Unix.close_process_in ic with
      | WEXITED (0 | 1) ->
          List.sort compare
            (List.filter_map int_of_string_opt
               (String.split_on_char '\n' listed))
      | _ -> skip ~reason:"pgrep failed" ())

(* Kills this process's children, idle workers, and waits for each to end. *)
let kill_children () =
  List.iter
    (fun pid ->
      Unix.kill pid Sys.sigkill;
      ignore (Unix.waitpid [] pid))
    (children ())

(* A library that does not load: these tests run in a process of their own,
   where COMGR_PATH names a file that is no library, which the compiler finds
   and each worker fails to load. *)

let load_failure () =
  rejection (fun () -> Compiler.compile (Compiler_amd.hip "gfx1100") kernel)

let not_a_library =
  tag "no-library"
  @@ group "a library that does not load"
       [
         test "a compile raises Compile_error naming comgr and the file"
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
         test "a compile in turn is served by an idle worker" (fun () ->
             ignore (load_failure ());
             let workers = children () in
             not_equal (list int) [] workers;
             ignore (load_failure ());
             equal (list int) workers (children ()));
         test
           "a compile after its idle workers are killed raises the same error"
           (fun () ->
             let first = load_failure () in
             kill_children ();
             equal (list int) [] (children ());
             equal (option string) first (sigpipe_kills load_failure));
       ]

(* comgr *)

let smoke =
  group "comgr"
    [
      slow
      @@ test "compiles a kernel to a code object" (fun () ->
          let lib =
            with_comgr (fun () ->
                Compiler.compile (Compiler_amd.hip "gfx1100") kernel)
          in
          starts_with ~prefix:"\x7fELF" lib;
          contains ~sub:"add" lib);
      slow
      @@ test "assembles a source whose first line is .text" (fun () ->
          let lib =
            with_comgr (fun () ->
                Compiler.compile (Compiler_amd.hip "gfx1100") assembly)
          in
          starts_with ~prefix:"\x7fELF" lib);
      slow
      @@ test "a rejected source raises Compile_error with comgr's log"
           (fun () ->
             match
               Compiler.compile
                 (Compiler_amd.hip "gfx1100")
                 "extern \"C\" __attribute__((global)) void f() { \
                  undeclared_name = 1; }"
             with
             | _ -> fail "comgr accepted an undeclared name"
             | exception Failure msg when absent msg -> skip ~reason:msg ()
             | exception Compiler.Compile_error msg ->
                 contains ~sub:"undeclared_name" msg);
      slow
      @@ test "compiles from several domains at once" (fun () ->
          let archs = [ "gfx90a"; "gfx942"; "gfx1100"; "gfx1201" ] in
          let compile arch () =
            Compiler.compile (Compiler_amd.hip arch) kernel
          in
          ignore (with_comgr (compile "gfx1100"));
          let domains =
            List.map (fun arch -> Domain.spawn (compile arch)) archs
          in
          List.iter
            (fun lib -> starts_with ~prefix:"\x7fELF" lib)
            (List.map Domain.join domains));
      slow
      @@ test "disassembles a code object" (fun () ->
          let hip = with_comgr (fun () -> Compiler_amd.hip "gfx1100") in
          let lib = Compiler.compile hip kernel in
          ignore (output ());
          Compiler.disassemble hip lib;
          flush stdout;
          contains ~sub:"s_endpgm" (output ()));
    ]

(* Compiles in processes of their own *)

(* An intrinsic of CDNA's matrix cores, which code generation for RDNA cannot
   select: LLVM ends the process that compiles it. *)
let unselectable =
  {|extern "C" __attribute__((device)) float mfma(float, float, float)
  __asm("llvm.amdgcn.mfma.f32.32x32x1f32");
extern "C" __attribute__((global)) void f(float *a) { a[0] = mfma(a[1], a[2], a[3]); }|}

let compile ?(arch = "gfx1100") src =
  Compiler.compile (Compiler_amd.hip arch) src

(* [kernel]'s code object from a worker that has compiled nothing else: idle
   workers are killed first, so that a compile in turn is compared with one
   that no earlier compile can have touched. *)
let fresh_reference () =
  with_comgr (fun () ->
      kill_children ();
      compile kernel)

let process =
  group "compiles in processes of their own"
    [
      slow
      @@ test
           "compiles of one kernel are equal, in turn and from several domains"
           (fun () ->
             let lib = with_comgr (fun () -> compile kernel) in
             equal string lib (compile kernel);
             let domains =
               List.init 4 (fun _ -> Domain.spawn (fun () -> compile kernel))
             in
             List.iter (equal string lib) (List.map Domain.join domains));
      slow
      @@ cases "a source that is no HIP raises Compile_error"
           ~id:(Printf.sprintf "%S")
           [
             "";
             "\xff\xfe\x7fELF\000\001";
             "}{";
             String.sub kernel 0 40;
             ".text\n  not_an_instruction v0\n";
           ]
           (fun src ->
             ignore (with_comgr (fun () -> compile kernel));
             raises_match
               (function Compiler.Compile_error _ -> true | _ -> false)
               (fun () -> compile src));
      slow
      @@ test
           "a source that ends its process raises Compile_error with LLVM's \
            error" (fun () ->
             ignore (with_comgr (fun () -> compile kernel));
             match compile unselectable with
             | _ -> fail "code generation selected a CDNA intrinsic for RDNA"
             | exception Compiler.Compile_error msg ->
                 in_order ~subs:[ "without a reply"; "LLVM ERROR" ] msg);
      slow
      @@ test
           "a source that code generation rejects raises Compile_error with \
            comgr's log" (fun () ->
             let src =
               {|extern "C" __attribute__((global)) void f(int *a) {
  asm volatile("v_not_an_instruction v0");
  a[0] = 1;
}|}
             in
             ignore (with_comgr (fun () -> compile kernel));
             match compile src with
             | _ -> fail "code generation assembled an unknown instruction"
             | exception Compiler.Compile_error msg ->
                 in_order
                   ~subs:[ "code generation failed"; "cannot compile inline asm" ]
                   msg);
      slow
      @@ test "a compile in turn is served by an idle worker" (fun () ->
          ignore (with_comgr (fun () -> compile kernel));
          let workers = children () in
          not_equal (list int) [] workers;
          ignore (compile kernel);
          equal (list int) workers (children ()));
      slow
      @@ test "compiles in turn for other architectures are each one's"
           (fun () ->
             let lib = fresh_reference () in
             let other = compile ~arch:"gfx1201" kernel in
             not_equal string lib other;
             equal string lib (compile kernel);
             equal string other (compile ~arch:"gfx1201" kernel));
      slow
      @@ test "a compile after a rejected source returns its code object"
           (fun () ->
             let lib = fresh_reference () in
             is_some (rejection (fun () -> compile "}{"));
             equal string lib (compile kernel));
      slow
      @@ test
           "a compile after a source that ends its process returns its code \
            object" (fun () ->
             let lib = fresh_reference () in
             is_some (rejection (fun () -> compile unselectable));
             equal string lib (compile kernel));
      slow
      @@ test
           "a compile after its idle workers are killed returns its code object"
           (fun () ->
             let lib = with_comgr (fun () -> compile kernel) in
             kill_children ();
             equal string lib (sigpipe_kills (fun () -> compile kernel)));
      slow
      @@ test "a compile writes nothing to comgr's own cache" (fun () ->
          ignore (with_comgr (fun () -> compile kernel));
          ignore (compile ~arch:"gfx1201" kernel);
          let entries =
            if Sys.file_exists comgr_cache then
              Array.to_list (Sys.readdir comgr_cache)
            else []
          in
          equal (list string) [] entries);
      slow
      @@ test "a worker started with SIGCHLD ignored returns its code object"
           (fun () ->
             let lib = with_comgr (fun () -> compile kernel) in
             kill_children ();
             equal string lib
               (with_signal Sys.sigchld Signal_ignore (fun () -> compile kernel)));
    ]

(* LLVM's verifier *)

(* A call of an intrinsic with an argument it does not take: IR that LLVM's
   verifier rejects and that code generation compiles without it, as it would IR
   that an optimization makes invalid. *)
let invalid_ir =
  {|extern "C" __attribute__((device)) int wid(int) __asm("llvm.amdgcn.workitem.id.x");
extern "C" __attribute__((global)) void f(int *a) { a[0] = wid(a[1]); }|}

let verifier =
  group "LLVM's verifier"
    [
      slow
      @@ test "invalid IR raises Compile_error with the verifier's finding"
           (fun () ->
             ignore (with_comgr (fun () -> compile kernel));
             (match Compiler_amd.comgr_version () with
             | None -> fail "comgr compiled but reports no version"
             | Some v when v < (2, 8) ->
                 skip ~reason:"comgr before 2.8 compiles without the verifier"
                   ()
             | Some _ -> ());
             match compile invalid_ir with
             | _ -> fail "comgr compiled IR that LLVM's verifier rejects"
             | exception Compiler.Compile_error msg ->
                 in_order
                   ~subs:
                     [
                       "Intrinsic has incorrect argument type";
                       "Broken module found";
                     ]
                   msg);
    ]

(* A process that cannot start: these tests run in a process of their own, where
   COMGR_PATH names a file, and forbid it to start any other for the rest of its
   life. *)

let no_spawn =
  tag "no-spawn"
  @@ group "a process that cannot start"
       [
         test "a compile raises Failure naming the worker" (fun () ->
             Spawn.forbid ();
             raises_match (Exn.failure ~sub:"comgr worker") (fun () ->
                 compile kernel));
       ]

let () =
  exit
    (run "Tolk.Compiler_amd"
       [
         cache; without_comgr; not_a_library; smoke; process; verifier; no_spawn;
       ])
