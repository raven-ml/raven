open Windtrap
open Tolk
module Compiler = Renderer.Compiler

let kernel =
  {|extern "C" __attribute__((global)) void add(float *a, const float *b) {
  a[0] += b[0];
}|}

(* A loop that exits on its data, written as the renderer writes a loop of no
   bound, its test at the end: it counts a nonzero word's set bits by clearing
   the lowest until none is left, which loop idiom recognition replaces by a
   population count. *)
let bit_count =
  {|extern "C" __attribute__((global)) void count(unsigned long *a, int *c) {
  unsigned long x = a[0];
  int n = 0;
  if (x != 0) {
    for (;;) {
      x &= x - 1;
      n++;
      if (!(x != 0)) { break; }
    }
  }
  c[0] = n;
}|}

(* The branches of [text], a disassembly, that jump backward: an
   [s_cbranch_*] whose 16-bit word offset, printed unsigned, is negative. *)
let backward_branches text =
  String.split_on_char '\n' text
  |> List.filter_map (fun line ->
      match String.split_on_char ' ' (String.trim line) with
      | op :: offset :: _ when String.starts_with ~prefix:"s_cbranch" op -> (
          match int_of_string_opt offset with
          | Some o when o >= 0x8000 -> Some op
          | _ -> None)
      | _ -> None)

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

(* This process's temporary directory, where workers' temporary directories are
   made, its TMPDIR, and AMD_COMGR_CACHE_DIR, where comgr keeps its own cache,
   are a directory of this process's own, removed when it exits: what a compile
   leaves behind, in a worker's directory or in TMPDIR, is seen there, and gone
   after. *)
let scratch =
  let dir = Filename.temp_dir "tolk-test-comgr" "" in
  at_exit (fun () -> remove dir);
  Filename.set_temp_dir_name dir;
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

(* The pids of process [pid]'s children, which pgrep lists, in order; the test
   skips where pgrep does not run. *)
let children_of pid =
  let pid = string_of_int pid in
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

let children () = children_of (Unix.getpid ())

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
      slow
      @@ test "a loop that exits on its data stays a loop: idioms are off"
           (fun () ->
             let hip = with_comgr (fun () -> Compiler_amd.hip "gfx1201") in
             let lib = Compiler.compile hip bit_count in
             ignore (output ());
             Compiler.disassemble hip lib;
             flush stdout;
             let text = output () in
             equal (list string) ~msg:"bit counts" []
               (List.filter
                  (fun w -> has_infix ~affix:w text)
                  [ "s_bcnt1"; "v_bcnt" ]);
             not_equal (list string) ~msg:"backward branches" []
               (backward_branches text));
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

(* [until ~seconds p] waits until [p ()] holds, asking every 10 ms: false if
   [seconds] pass first. *)
let until ~seconds p =
  let deadline = Unix.gettimeofday () +. seconds in
  let rec ask () =
    p ()
    || Unix.gettimeofday () < deadline
       && begin
         Unix.sleepf 0.01;
         ask ()
       end
  in
  ask ()

(* Whether process [pid] has ended: there is none, or it waits for its parent to
   collect it (Linux's /proc says so). *)
let ended pid =
  match Unix.kill pid 0 with
  | exception Unix.Unix_error (ESRCH, _, _) -> true
  | () -> (
      let stat = Printf.sprintf "/proc/%d/stat" pid in
      match In_channel.with_open_bin stat In_channel.input_all with
      | exception Sys_error _ -> true
      | s -> (
          match String.rindex_opt s ')' with
          | Some i -> i + 2 < String.length s && s.[i + 2] = 'Z'
          | None -> false))

(* The paths under [dir] of comgr's temporary directories, named comgr-XXXXXX,
   which a compile that aborts leaves behind. *)
let rec comgr_files dir =
  List.concat_map
    (fun name ->
      let path = Filename.concat dir name in
      if String.starts_with ~prefix:"comgr-" name then [ path ]
      else
        match (Unix.lstat path).st_kind with
        | S_DIR -> comgr_files path
        | _ -> [])
    (Array.to_list (Sys.readdir dir))

(* The temporary directory of worker [pid], its program's second argument, as
   Linux's /proc/PID/cmdline lists them. *)
let temp_of pid =
  let cmdline = Printf.sprintf "/proc/%d/cmdline" pid in
  match In_channel.with_open_bin cmdline In_channel.input_all with
  | s -> (
      match String.split_on_char '\000' s with
      | _ :: _ :: temp :: _ -> Some temp
      | _ -> None)
  | exception Sys_error _ -> None

(* In a forked process: forks a process that sleeps until killed, holding what
   this one holds, and writes its pid to [out]. *)
let fork_sleeper out =
  match Unix.fork () with
  | 0 ->
      Unix.sleepf 60.;
      Unix._exit 0
  | pid ->
      let s = string_of_int pid in
      ignore (Unix.write_substring out s 0 (String.length s));
      Unix.close out

(* The pid that [fork_sleeper] writes to [input], if any. *)
let read_sleeper input =
  let buf = Bytes.create 16 in
  let n = try Unix.read input buf 0 16 with Unix.Unix_error _ -> 0 in
  Unix.close input;
  int_of_string_opt (Bytes.sub_string buf 0 n)

(* [kernel] including the FIFO [fifo]: its compile waits in the preprocessor for
   a writer to open [fifo], then for the writer to close it. *)
let waiting fifo = Printf.sprintf "#include \"%s\"\n%s" fifo kernel

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
                   ~subs:
                     [ "code generation failed"; "cannot compile inline asm" ]
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
      @@ test "a compile that ends its process leaves no temporary file"
           (fun () ->
             ignore (with_comgr (fun () -> compile kernel));
             is_some (rejection (fun () -> compile unselectable));
             equal (list string) [] (comgr_files scratch));
      slow
      @@ test
           "a worker whose parent is killed during a compile ends, though a \
            process forked from the parent lives" (fun () ->
             ignore (with_comgr (fun () -> compile kernel));
             let fifo = Filename.concat scratch "include.fifo" in
             Unix.mkfifo fifo 0o600;
             Fun.protect ~finally:(fun () -> Sys.remove fifo) @@ fun () ->
             let input, out = Unix.pipe ~cloexec:true () in
             let parent =
               match Unix.fork () with
               | 0 ->
                   Unix.close input;
                   (try
                      ignore (compile kernel);
                      fork_sleeper out;
                      ignore (compile (waiting fifo))
                    with _ -> ());
                   Unix._exit 0
               | pid -> pid
             in
             Unix.close out;
             let sleeper = read_sleeper input in
             Fun.protect ~finally:(fun () ->
                 Option.iter (fun pid -> Unix.kill pid Sys.sigkill) sleeper)
             @@ fun () ->
             (* The FIFO opens for writing once the compile has opened it to
                read, which then waits until the writer closes it. *)
             let writer = ref None in
             let opened () =
               let flags = Unix.[ O_WRONLY; O_NONBLOCK; O_CLOEXEC ] in
               match Unix.openfile fifo flags 0 with
               | fd ->
                   writer := Some fd;
                   true
               | exception Unix.Unix_error (ENXIO, _, _) -> false
             in
             Fun.protect ~finally:(fun () -> Option.iter Unix.close !writer)
             @@ fun () ->
             let reached = until ~seconds:60. opened in
             let workers =
               List.filter (fun pid -> Some pid <> sleeper) (children_of parent)
             in
             let temps = List.filter_map temp_of workers in
             Unix.kill parent Sys.sigkill;
             ignore (Unix.waitpid [] parent);
             is_some ~msg:"the forked process" sleeper;
             equal bool ~msg:"the compile opened the FIFO" true reached;
             not_equal (list int) [] workers;
             let alive =
               List.filter
                 (fun pid -> not (until ~seconds:3. (fun () -> ended pid)))
                 workers
             in
             equal (list int) ~msg:"workers alive" [] alive;
             not_equal (list string) [] temps;
             equal (list string) ~msg:"their temporary directories" []
               (List.filter Sys.file_exists temps));
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
