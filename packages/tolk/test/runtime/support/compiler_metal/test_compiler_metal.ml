open Windtrap
open Tolk
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

(* MTLCompiler's absence is a Failure, raised when a compiler is made, that
   names the variable to set. *)
let absent msg = has_infix ~affix:"MTLCOMPILER_PATH" msg

(* [with_metal f] is [f ()], or skips the test if MTLCompiler is not on the
   machine. *)
let with_metal f =
  match f () with
  | v -> v
  | exception Failure msg when absent msg -> skip ~reason:msg ()

let is_metal_library lib =
  starts_with ~prefix:"MTLB" lib;
  ends_with ~suffix:"ENDT" lib

let rejection f =
  match f () with _ -> None | exception Compiler.Compile_error msg -> Some msg

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

(* Workers *)

(* [remove path] removes the file or the directory tree at [path], following no
   symbolic link. *)
let rec remove path =
  match (Unix.lstat path).st_kind with
  | S_DIR ->
      Array.iter (fun n -> remove (Filename.concat path n)) (Sys.readdir path);
      Unix.rmdir path
  | _ -> Unix.unlink path

(* A directory of the run's own, made this process's temporary directory, where
   a program the cache cannot hold is written: a test that looks there acquires
   it before its first compile. *)
let scratch =
  fixture ~teardown:remove (fun () ->
      let dir = Filename.temp_dir "tolk-test-metal" "" in
      Filename.set_temp_dir_name dir;
      dir)

let named_in dir ~prefix =
  if not (Sys.file_exists dir) then []
  else
    List.filter (String.starts_with ~prefix) (Array.to_list (Sys.readdir dir))

(* [with_signal s behavior f] is [f ()] run with [behavior] for [s]. *)
let with_signal s behavior f =
  let previous = Sys.signal s behavior in
  Fun.protect ~finally:(fun () -> Sys.set_signal s previous) f

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

let digest lib = Digest.to_hex (Digest.string lib)
let compile src = Compiler.compile (Compiler_metal.compiler ()) src

(* A library that does not load: these tests run in a process of their own,
   where MTLCOMPILER_PATH names a file that is no library, which the compiler
   finds and each worker fails to load. Off macOS the worker loads none. *)

let load_failure () = rejection (fun () -> compile (kernel "add"))
let no_library = tag "no-library"

(* The table *)

let ccache_off f = Setting.context [ B (Setting.ccache, false) ] f

(* [named ~prefix table] checks that [table] is [prefix] then a digest. *)
let named ~prefix table =
  let n = Int.min (String.length prefix) (String.length table) in
  equal string ~msg:"the name" prefix (String.sub table 0 n);
  equal int ~msg:"the digest's length" 32 (String.length table - n)

let table () =
  Option.get (Compiler.cachekey (with_metal Compiler_metal.compiler))

let table_tests =
  group "the table"
    [
      no_library
      @@ test
           "libraries are cached in the table compile_metal_direct and a digest"
           (fun () -> named ~prefix:"compile_metal_direct_" (table ()));
      no_library
      @@ test "the table names the file MTLCOMPILER_PATH names" (fun () ->
          let default = table () in
          let file = Filename.concat (temp_dir ()) "MTLCompiler" in
          Out_channel.with_open_bin file (fun oc -> output_string oc "a");
          setenv "MTLCOMPILER_PATH" (Some file);
          let named_file = table () in
          not_equal string default named_file;
          Out_channel.with_open_bin file (fun oc -> output_string oc "changed");
          not_equal string ~msg:"once the file changed" named_file (table ()));
      no_library
      @@ test "the table is named with ccache off" (fun () ->
          equal string (table ()) (ccache_off table));
      test "a compile the table holds starts no process" (fun () ->
          let metal = with_metal Compiler_metal.compiler in
          let lib = Compiler.compile_cached metal (kernel "kept") in
          kill_children ();
          let again = Compiler.compile_cached metal (kernel "kept") in
          equal (list int) ~msg:"workers" [] (children ());
          equal string (digest lib) (digest again));
    ]

(* A library that loads but is no MTLCompiler: these tests run in processes of
   their own, where MTLCOMPILER_PATH names a library of the suite's. One has
   neither of MTLCompiler's functions; the other's makes no service, and aborts
   if a compile reaches it. A worker answers each compile with why, and
   lives. *)

(* [answers_why why] checks that a compile raises [Compile_error] saying [why]
   from a worker that answers the next compile alike. *)
let answers_why why =
  if Platform.system <> "macosx" then
    skip ~reason:"only macOS loads MTLCompiler" ();
  let msg = require_some (load_failure ()) in
  in_order ~subs:[ "MTLCompiler"; ".dylib"; why ] msg;
  let workers = children () in
  not_equal (list int) [] workers;
  equal (option string) (Some msg) (load_failure ());
  equal (list int) ~msg:"workers" workers (children ())

let compiler_tests =
  group "compiler"
    [
      test "compiles a kernel to a Metal library" (fun () ->
          is_metal_library (with_metal (fun () -> compile (kernel "add"))));
      test "a source that draws warnings compiles to a Metal library" (fun () ->
          let src =
            "#include <metal_stdlib>\n\
             kernel void warned(device int* data0) { int unused = 1; data0[0] \
             = 0; }"
          in
          is_metal_library (with_metal (fun () -> compile src)));
      test "the language is the latest the running macOS compiles kernels of"
        (fun () ->
          let src =
            Printf.sprintf
              "#if __METAL_VERSION__ != %d\n#error wrong version\n#endif\n%s"
              (metal_version ()) (kernel "versioned")
          in
          is_metal_library (with_metal (fun () -> compile src)));
      test "a product and a sum compile under the no-contraction pragma"
        (fun () ->
          let src =
            "#include <metal_stdlib>\n\
             kernel void muladd(device float* d, const device float* a, const \
             device float* b, const device float* c) { d[0] = a[0] * b[0] + \
             c[0]; }"
          in
          is_metal_library (with_metal (fun () -> compile src)));
      test "fast math is off" (fun () ->
          let src =
            "#ifdef __FAST_MATH__\n#error fast math\n#endif\n" ^ kernel "exact"
          in
          is_metal_library (with_metal (fun () -> compile src)));
      table_tests;
      test
        "without MTLCompiler, making a compiler raises Failure naming \
         MTLCOMPILER_PATH" (fun () ->
          match Compiler_metal.compiler () with
          | _ -> skip ~reason:"MTLCompiler is on the machine" ()
          | exception Failure msg ->
              in_order ~subs:[ "MTLCompiler"; "MTLCOMPILER_PATH" ] msg);
      test "a rejected source raises Compile_error with the compiler's message"
        (fun () ->
          let src =
            "kernel void f(device int* a) { a[0] = undeclared_name; }"
          in
          match with_metal (fun () -> compile src) with
          | _ -> fail "the compiler accepted an undeclared name"
          | exception Compiler.Compile_error msg ->
              contains ~sub:"undeclared_name" msg);
      no_library
      @@ test "making a compiler starts no process" (fun () ->
          kill_children ();
          ignore (Compiler_metal.compiler ());
          equal (list int) [] (children ()));
      no_library
      @@ test "a library that does not load raises Compile_error naming it"
           (fun () ->
             let msg = require_some (load_failure ()) in
             contains ~sub:"MTLCompiler" msg;
             (* Elsewhere than on macOS, no file is loaded. *)
             if Platform.system = "macosx" then
               contains ~sub:"not_a_library" msg);
      tag "no-symbol"
      @@ test
           "a library without MTLCompiler's functions raises Compile_error \
            saying so, from a worker that lives" (fun () ->
             answers_why "undefined symbol MTLCodeGenServiceCreate");
      tag "no-service"
      @@ test
           "a library that makes no service raises Compile_error saying so, \
            from a worker that lives" (fun () ->
             answers_why "MTLCodeGenServiceCreate made no service");
      (* This test runs in a process of its own, where MTLCOMPILER_PATH names a
         file, and forbids it to start any other for the rest of its life. *)
      tag "no-spawn"
      @@ test "a process that cannot start raises Failure naming the worker"
           (fun () ->
             Spawn.forbid ();
             raises_match (Exn.failure ~sub:"metal worker") (fun () ->
                 compile (kernel "add")));
      test "disassembly prints nothing" (fun () ->
          let metal = with_metal Compiler_metal.compiler in
          let lib = Compiler.compile metal (kernel "add") in
          ignore (output ());
          Compiler.disassemble metal lib;
          flush stdout;
          equal string "" (output ()));
    ]

(* Compiles in processes of their own *)

(* Kernel [i], scaled by [c]: distinct kernels for distinct [i]. *)
let scaled (i, c) =
  Printf.sprintf
    "#include <metal_stdlib>\n\
     kernel void scaled%d(device float* d, uint i [[thread_position_in_grid]]) \
     { d[i] = d[i] * %d.5f; }"
    i c

(* The digests of [srcs] compiled on [domains] domains at once, each taking
   every [domains]-th source, in [srcs]' order. *)
let at_once ~domains srcs =
  let srcs = Array.of_list srcs in
  let share d =
    List.filteri
      (fun i _ -> i mod domains = d)
      (List.init (Array.length srcs) Fun.id)
  in
  let spawned =
    List.init domains (fun d ->
        Domain.spawn (fun () ->
            List.map (fun i -> (i, digest (compile srcs.(i)))) (share d)))
  in
  List.concat_map Domain.join spawned |> List.sort compare |> List.map snd

(* 1 to 8 distinct kernels and 1 to 8 domains, a third of the cases on one
   domain and a third on more than four. *)
let concurrent_case =
  let open Gen in
  let* n = int_range 1 8 in
  let* consts = list ~size:(constant n) (int_range 0 999) in
  let+ domains =
    frequency [ (1, constant 1); (1, int_range 2 4); (1, int_range 5 8) ]
  in
  (List.mapi (fun i c -> scaled (i, c)) consts, domains)

(* The cases each cover needs, run whatever the seed. *)
let on_one_domain = ([ scaled (0, 1); scaled (1, 2) ], 1)
let on_eight_domains = (List.init 8 (fun i -> scaled (i, i)), 8)

let process =
  group "compiles in processes of their own"
    [
      prop ~count:30 ~examples:[ on_one_domain; on_eight_domains ]
        "kernels compiled on several domains at once are the bytes they are \
         compiled to in turn"
        concurrent_case (fun (srcs, domains) ->
          cover "one domain" (domains = 1);
          cover "more than four domains" (domains > 4);
          let in_turn =
            List.map
              (fun src -> digest (with_metal (fun () -> compile src)))
              srcs
          in
          equal (list string) in_turn (at_once ~domains srcs));
      no_library
      @@ test "a compile in turn is served by the idle worker, with its answer"
           (fun () ->
             let first = load_failure () in
             let workers = children () in
             not_equal (list int) [] workers;
             equal (option string) first (load_failure ());
             equal (list int) ~msg:"workers" workers (children ()));
      test "a worker started with SIGCHLD ignored returns its binary" (fun () ->
          let lib = digest (with_metal (fun () -> compile (kernel "add"))) in
          kill_children ();
          equal string lib
            (digest
               (with_signal Sys.sigchld Signal_ignore (fun () ->
                    compile (kernel "add")))));
    ]

(* A cache that cannot hold the program: these tests run in processes of their
   own. In one, XDG_CACHE_HOME names a file, under which no cache can be made;
   in the other, a directory of the suite's, whose program the test makes one
   that the system refuses to run. *)

(* The program's file in [dir], which a compile wrote: there is one. *)
let program_in dir =
  match
    List.filter
      (fun f -> not (Filename.check_suffix f ".tmp"))
      (named_in dir ~prefix:"metal-worker-")
  with
  | [ f ] -> Filename.concat dir f
  | files -> failf "%d programs in %s" (List.length files) dir

(* A program that leaves a file where it ran: what another user would plant at
   the program's name. *)
let planted () =
  let marker = Filename.concat (scratch ()) "planted-ran" in
  let script = Filename.concat (scratch ()) "planted.sh" in
  Out_channel.with_open_bin script (fun oc ->
      Printf.fprintf oc "#!/bin/sh\ntouch %s\n" (Filename.quote marker));
  Unix.chmod script 0o755;
  (script, marker)

(* [program] replaced by a link to [target], while no worker runs. *)
let link_over program target =
  kill_children ();
  Sys.remove program;
  Unix.symlink target program

(* [lib]'s compile, after [plant] changed the program in [dir]: the planted file
   never runs, and the compile returns the binary. *)
let not_run ~dir plant =
  let lib = digest (with_metal (fun () -> compile (kernel "add"))) in
  let script, marker = planted () in
  plant (program_in dir) script;
  kill_children ();
  equal string lib (digest (compile (kernel "add")));
  equal (list string) ~msg:"the planted program's marks" []
    (List.filter Sys.file_exists [ marker ])

let unwritable_cache =
  tag "unwritable-cache"
  @@ group "a cache that cannot be written"
       [
         test "a compile returns its binary" (fun () ->
             ignore (scratch ());
             is_metal_library (with_metal (fun () -> compile (kernel "add"))));
         test
           "a link at the program's name in the temporary directory is not run"
           (fun () -> not_run ~dir:(scratch ()) link_over);
       ]

(* A cache whose program changed under a running process: these tests run in a
   process of their own, where XDG_CACHE_HOME names a directory of the
   sandbox. *)

(* [f ()] with [change] run on the program's path each time its check is about
   to open it. *)
let while_checked change f =
  Compiler_metal.before_program_open := change;
  Fun.protect
    ~finally:(fun () -> Compiler_metal.before_program_open := ignore)
    f

(* What another process writing the program does: a whole copy written aside,
   then renamed to [program]. *)
let rewrite_as_peer program =
  let copy = program ^ ".peer" in
  let bytes = In_channel.with_open_bin program In_channel.input_all in
  Out_channel.with_open_gen [ Open_wronly; Open_creat; Open_trunc; Open_binary ]
    0o755 copy (fun oc -> output_string oc bytes);
  Sys.rename copy program

let program_changed =
  tag "program-changed"
  @@ group "a program that changed"
       [
         test "a program another process writes again while it is checked runs"
           (fun () ->
             let lib = digest (with_metal (fun () -> compile (kernel "add"))) in
             kill_children ();
             let compiled () = digest (compile (kernel "add")) in
             equal string lib (while_checked rewrite_as_peer compiled));
         test "a link swapped in while the program is checked is not run"
           (fun () ->
             (* The program and the script to link there, once planted. *)
             let armed = ref None and swapped = ref [] in
             let swap path =
               match !armed with
               | Some (program, script) when path = program ->
                   armed := None;
                   swapped := path :: !swapped;
                   Sys.remove path;
                   Unix.symlink script path
               | _ -> ()
             in
             while_checked swap (fun () ->
                 not_run ~dir:Helpers.cache_dir (fun program script ->
                     armed := Some (program, script)));
             equal int ~msg:"links swapped in" 1 (List.length !swapped));
         test "a link at the program's name is not run" (fun () ->
             not_run ~dir:Helpers.cache_dir link_over);
         test "a program rewritten after it ran is not run" (fun () ->
             not_run ~dir:Helpers.cache_dir (fun program script ->
                 Out_channel.with_open_bin program (fun oc ->
                     output_string oc
                       (In_channel.with_open_bin script In_channel.input_all))));
         test "a program other users can write is replaced before it runs"
           (fun () ->
             let lib = digest (with_metal (fun () -> compile (kernel "add"))) in
             let program = program_in Helpers.cache_dir in
             kill_children ();
             Unix.chmod program 0o777;
             equal string lib (digest (compile (kernel "add")));
             equal int ~msg:"the mode others can write" 0
               ((Unix.stat program).st_perm land 0o022));
         test "a program removed after it ran is written again" (fun () ->
             let lib = digest (with_metal (fun () -> compile (kernel "add"))) in
             let program = program_in Helpers.cache_dir in
             kill_children ();
             Sys.remove program;
             equal string lib (digest (compile (kernel "add"))));
         (* macOS holds the start of a quarantined program until the user
            approves it, unless a run of that file was approved before: the
            program is a new file, quarantined before its first run. *)
         test "a quarantined program is replaced before it runs" (fun () ->
             let lib = digest (with_metal (fun () -> compile (kernel "add"))) in
             let program = program_in Helpers.cache_dir in
             kill_children ();
             let bytes =
               In_channel.with_open_bin program In_channel.input_all
             in
             Sys.remove program;
             Out_channel.with_open_gen [ Open_wronly; Open_creat; Open_binary ]
               0o755 program (fun oc -> output_string oc bytes);
             let quarantine =
               [|
                 "xattr"; "-w"; "com.apple.quarantine"; "0081;0;tolk;"; program;
               |]
             in
             (match Unix.open_process_args_in "xattr" quarantine with
             | exception Unix.Unix_error _ ->
                 skip ~reason:"xattr does not run" ()
             | ic -> (
                 match Unix.close_process_in ic with
                 | WEXITED 0 -> ()
                 | _ -> skip ~reason:"xattr failed" ()));
             equal string lib (digest (compile (kernel "add"))));
       ]

let unrunnable_cache =
  tag "unrunnable-cache"
  @@ group "a cache whose program cannot run"
       [
         test "a compile returns its binary" (fun () ->
             let lib = digest (with_metal (fun () -> compile (kernel "add"))) in
             let programs =
               named_in Helpers.cache_dir ~prefix:"metal-worker-"
             in
             not_equal (list string) ~msg:"programs in the cache" [] programs;
             List.iter
               (fun p -> Unix.chmod (Filename.concat Helpers.cache_dir p) 0o644)
               programs;
             kill_children ();
             equal string lib (digest (compile (kernel "add"))));
       ]

let () =
  exit
    (run "Tolk.Compiler_metal"
       [
         compiler_tests;
         process;
         unwritable_cache;
         unrunnable_cache;
         program_changed;
       ])
