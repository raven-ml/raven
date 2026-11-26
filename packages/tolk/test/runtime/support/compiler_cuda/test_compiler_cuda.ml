open Windtrap
open Tolk
module Compiler = Renderer.Compiler

let kernel =
  {|extern "C" __global__ void add(float *a, const float *b) {
  a[threadIdx.x] += b[threadIdx.x];
}|}

let has_infix ~affix s =
  let n = String.length affix in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = affix || at (i + 1))
  in
  at 0

let rejection f =
  match f () with _ -> None | exception Compiler.Compile_error msg -> Some msg

(* NVRTC's absence is a Failure, raised when a compiler is made, that names the
   variable to set. *)
let absent msg = has_infix ~affix:"NVRTC_PATH" msg

(* [nvrtc ?ptx arch] is [Compiler_cuda.nvrtc ?ptx arch], or skips the test if
   NVRTC is not on the machine. *)
let nvrtc ?ptx arch =
  match Compiler_cuda.nvrtc ?ptx arch with
  | c -> c
  | exception Failure msg when absent msg -> skip ~reason:msg ()

let ptx ?(arch = "sm_89") src = Compiler.compile (nvrtc arch) src
let cubin src = Compiler.compile (nvrtc ~ptx:false "sm_89") src

(* Workers *)

(* [remove path] removes the file or the directory tree at [path], following no
   symbolic link. *)
let rec remove path =
  match (Unix.lstat path).st_kind with
  | S_DIR ->
      Array.iter (fun n -> remove (Filename.concat path n)) (Sys.readdir path);
      Unix.rmdir path
  | _ -> Unix.unlink path

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

(* Kills the processes [pids], idle workers, and waits for each to end. Where
   SIGCHLD is ignored the system reaps a child as it ends, and [waitpid] then
   fails once every child has ended. *)
let kill pids =
  List.iter
    (fun pid ->
      Unix.kill pid Sys.sigkill;
      try ignore (Unix.waitpid [] pid)
      with Unix.Unix_error (ECHILD, _, _) -> ())
    pids

let kill_children () = kill (children ())

(* This process's temporary directory, where workers' temporary directories are
   made from the first call: a directory of its own, removed at the end of the
   run. The workers idle before it are killed, so that every later one starts
   under it. *)
let scratch =
  fixture ~teardown:remove (fun () ->
      let dir = Filename.temp_dir "tolk-test-nvrtc" "" in
      Filename.set_temp_dir_name dir;
      kill_children ();
      dir)

(* The temporary directories of workers started since [scratch ()]. *)
let worker_dirs () =
  List.filter
    (String.starts_with ~prefix:"tolk-nvrtc-")
    (Array.to_list (Sys.readdir (scratch ())))

(* A library that does not load: these tests run in a process of their own,
   where NVRTC_PATH names a file that is no library, which the compiler finds
   and each worker fails to load. A worker that fails to load it replies so to
   each request and serves the next. *)

let load_failure () =
  rejection (fun () -> Compiler.compile (Compiler_cuda.nvrtc "sm_89") kernel)

let no_library = tag "no-library"

(* The table *)

let table c = Option.get (Compiler.cachekey c)

(* [named ~prefix table] checks that [table] is [prefix] then a digest. *)
let named ~prefix table =
  let n = Int.min (String.length prefix) (String.length table) in
  equal string ~msg:"the name" prefix (String.sub table 0 n);
  equal int ~msg:"the digest's length" 32 (String.length table - n)

let ccache_off f = Setting.context [ B (Setting.ccache, false) ] f

(* A copy of NVRTC_PATH's file in a directory of the test's, which NVRTC_PATH
   then names. *)
let library_copy () =
  let file = Filename.concat (temp_dir ()) "libnvrtc.so" in
  let bytes = In_channel.with_open_bin "not_a_library" In_channel.input_all in
  Out_channel.with_open_bin file (fun oc -> output_string oc bytes);
  setenv "NVRTC_PATH" (Some file);
  file

(* A directory holding [include/tolk_test.h], which defines [TOLK_TEST]. *)
let toolkit_with_header () =
  let dir = temp_dir () in
  let headers = Filename.concat dir "include" in
  Unix.mkdir headers 0o755;
  Out_channel.with_open_bin (Filename.concat headers "tolk_test.h") (fun oc ->
      output_string oc "#define TOLK_TEST 1234567\n");
  dir

let table_tests =
  group "the table"
    [
      no_library
      @@ test
           "binaries are cached in the table of cuda, the architecture and a \
            digest" (fun () ->
             named ~prefix:"compile_cuda_sm_89_"
               (table (Compiler_cuda.nvrtc "sm_89")));
      no_library
      @@ test "cubins and PTX are cached in different tables" (fun () ->
          not_equal string
            (table (Compiler_cuda.nvrtc "sm_89"))
            (table (Compiler_cuda.nvrtc ~ptx:false "sm_89")));
      no_library
      @@ test "cache_key names the table" (fun () ->
          named ~prefix:"compile_nv_sm_120_"
            (table (Compiler_cuda.nvrtc ~ptx:false ~cache_key:"nv" "sm_120")));
      no_library
      @@ test "the table is named with ccache off" (fun () ->
          let table () = Compiler.cachekey (Compiler_cuda.nvrtc "sm_89") in
          equal (option string) (table ()) (ccache_off table));
      no_library
      @@ test "a library that changes changes the table" (fun () ->
          let file = library_copy () in
          let before = table (Compiler_cuda.nvrtc "sm_89") in
          Out_channel.with_open_bin file (fun oc -> output_string oc "changed");
          not_equal string before (table (Compiler_cuda.nvrtc "sm_89")));
      no_library
      @@ test "the directory of Setting.cuda_path changes the table" (fun () ->
          let made_in dir =
            Setting.context
              [ B (Setting.cuda_path, dir) ]
              (fun () -> table (Compiler_cuda.nvrtc "sm_89"))
          in
          not_equal string (made_in "") (made_in (temp_dir ())));
    ]

let nvrtc_tests =
  group "nvrtc"
    [
      test "compiles a kernel to PTX for its architecture" (fun () ->
          in_order ~subs:[ ".target sm_89"; ".entry add" ] (ptx kernel));
      test "compiles a kernel to a cubin with ~ptx:false" (fun () ->
          starts_with ~prefix:"\x7fELF" (cubin kernel));
      test
        "searches headers in include of the directory Setting.cuda_path names"
        (fun () ->
          let toolkit = toolkit_with_header () in
          let c =
            Setting.context
              [ B (Setting.cuda_path, toolkit) ]
              (fun () -> nvrtc "sm_89")
          in
          contains ~sub:"1234567"
            (Compiler.compile c
               "#include \"tolk_test.h\"\n\
                extern \"C\" __global__ void f(int *a) { a[0] = TOLK_TEST; }"));
      table_tests;
      test "without NVRTC, making a compiler raises Failure naming NVRTC_PATH"
        (fun () ->
          match Compiler_cuda.nvrtc "sm_89" with
          | _ -> skip ~reason:"NVRTC is on the machine" ()
          | exception Failure msg ->
              in_order ~subs:[ "nvrtc"; "NVRTC_PATH" ] msg);
      test "a rejected source raises Compile_error with NVRTC's error and log"
        (fun () ->
          match
            ptx "extern \"C\" __global__ void f() { undeclared_name = 1; }"
          with
          | _ -> fail "NVRTC accepted an undeclared name"
          | exception Compiler.Compile_error msg ->
              in_order
                ~subs:[ "NVRTC_ERROR_COMPILATION"; "undeclared_name" ]
                msg);
      no_library
      @@ test
           "a library that does not load raises Compile_error naming the file"
           (fun () ->
             let msg = require_some (load_failure ()) in
             in_order ~subs:[ "nvrtc"; "not_a_library" ] msg);
      no_library
      @@ test "a compile in turn is served by the idle worker, with its answer"
           (fun () ->
             let first = load_failure () in
             let workers = children () in
             not_equal (list int) [] workers;
             equal (option string) first (load_failure ());
             equal (list int) ~msg:"workers" workers (children ()));
      tag "no-spawn"
      @@ test "a process that cannot start raises Failure naming the worker"
           (fun () ->
             Spawn.forbid ();
             raises_match (Exn.failure ~sub:"nvrtc worker") (fun () ->
                 Compiler.compile (Compiler_cuda.nvrtc "sm_89") kernel));
      test "disassembling a PTX that ptxas cannot assemble prints why"
        (fun () ->
          Compiler.disassemble (nvrtc "sm_89") "not ptx";
          flush stdout;
          contains ~sub:"ptxas -arch=sm_89 " (output ()));
      test "disassembling a cubin that nvdisasm cannot read prints why"
        (fun () ->
          Compiler.disassemble (nvrtc ~ptx:false "sm_89") "not a cubin";
          flush stdout;
          let text = output () in
          contains ~sub:"nvdisasm " text;
          not_contains ~sub:"ptxas -arch" text);
      test "disassembly leaves no file in the temporary directory" (fun () ->
          let dir = scratch () in
          let files () = List.sort compare (Array.to_list (Sys.readdir dir)) in
          let before = files () in
          Compiler.disassemble (nvrtc "sm_89") "not ptx";
          Compiler.disassemble (nvrtc ~ptx:false "sm_89") "not a cubin";
          equal (list string) before (files ()));
    ]

(* Compiles in processes of their own *)

(* A valid kernel on which the assembler of NVRTC 13.3 and 13.4 crashes, for
   sm_80 to sm_89 with [--fmad=false]: deep chains of fused multiply-adds inside
   square roots, under the register limit of 1,024 threads. *)
let crashes_assembler =
  {|extern "C" __global__ void __launch_bounds__(1024) test(float* data0_65536, float* data1_65536, float* data4_128, float* data7_128, float* data9_1920, float* data10_65536) {
  int lidx0 = threadIdx.x;
  int lidx1 = threadIdx.y;
  int lidx2 = threadIdx.z;
  int alu0 = ((lidx1<<7));
  int alu1 = (lidx0<<2);
  int alu65 = (alu0+alu1);
  float4 val23 = (*((float4*)((data1_65536+alu65))));
  float4 val24 = (*((float4*)((data4_128))));
  float4 val25 = (*((float4*)((data7_128))));
  float4 val27 = (*((float4*)((data9_1920+(alu1)))));
  float4 val28 = (*((float4*)((data9_1920))));
  float4 val29 = (*((float4*)((1280))));
  float4 val30 = (*((float4*)((1408))));
  float4 val31 = (*((float4*)((1536))));
  float4 val32 = (*((float4*)((1664))));
  float4 val33 = (*((float4*)((1792))));
  float4 val34 = (*((float4*)((256))));
  float4 val35 = (*((float4*)((384))));
  float4 val36 = (*((float4*)((512))));
  float4 val37 = (*((float4*)((640))));
  float4 val38 = (*((float4*)((768))));
  float4 val39 = (*((float4*)((896))));
  float4 val40 = (*((float4*)((data9_1920))));
  float4 cast0 = make_float4((((val24.x+sqrt(((val24.x*val24.x)+4.0f)))*((__fmaf_rn(0.0f,val33.x,__fmaf_rn(0.0f,val32.x,__fmaf_rn(0.0f,val31.x,__fmaf_rn(0.0f,val30.x,__fmaf_rn(0.0f,val29.x,__fmaf_rn(0.0f,val27.x,__fmaf_rn(0.0f,val39.x,__fmaf_rn(0.0f,val38.x,__fmaf_rn(0.0f,val37.x,(val36.x)))))))))))))),((((((val24.y))))*(((val25.y)-val23.y)+(__fmaf_rn(0.0f,val33.y,__fmaf_rn(0.0f,val32.y,__fmaf_rn(0.0f,val31.y,__fmaf_rn(0.0f,val30.y,__fmaf_rn(0.0f,val29.y,__fmaf_rn(0.0f,val27.y,__fmaf_rn(0.0f,val39.y,__fmaf_rn(0.0f,val38.y,__fmaf_rn(0.0f,val37.y,__fmaf_rn(0.0f,val36.y,(val40.y))))))))))))))),((((((val24.z))))*(((val25.z)-val23.z)+(__fmaf_rn(0.0f,val33.z,__fmaf_rn(0.0f,val32.z,__fmaf_rn(0.0f,val31.z,__fmaf_rn(0.0f,val30.z,__fmaf_rn(0.0f,val29.z,__fmaf_rn(0.0f,val27.z,__fmaf_rn(0.0f,val39.z,__fmaf_rn(0.0f,val38.z,__fmaf_rn(0.0f,val37.z,__fmaf_rn(0.0f,val36.z,__fmaf_rn(0.0f,val35.z,__fmaf_rn(0.0f,val34.z,(val28.z))))))))))))))))),(0.0f+((sqrt(((val24.w))))*(((val25.w)-val23.w)+(__fmaf_rn(0.0f,val33.w,__fmaf_rn(0.0f,val32.w,__fmaf_rn(0.0f,val31.w,__fmaf_rn(0.0f,val30.w,__fmaf_rn(0.0f,val29.w,__fmaf_rn(0.0f,val27.w,__fmaf_rn(0.0f,val39.w,__fmaf_rn(0.0f,val38.w,__fmaf_rn(0.0f,val37.w,__fmaf_rn(0.0f,val36.w,__fmaf_rn(0.0f,val35.w,__fmaf_rn(0.0f,val34.w,(val28.w))))))))))))))))));
  bool alu70 = (lidx2==0);
  if (alu70) {
    *((float4*)((data0_65536+alu65))) = cast0;
    *((float4*)((data10_65536))) = cast0;
  }
}|}

(* [kernel] held in its compile, which includes a FIFO in [dir] that no process
   writes: NVRTC opens an included file, and the open of a FIFO waits for a
   writer. *)
let held_in_compile dir =
  let fifo = Filename.concat dir "held" in
  match Unix.mkfifo fifo 0o600 with
  | () -> Printf.sprintf "#include \"%s\"\n%s" fifo kernel
  | exception (Unix.Unix_error _ | Invalid_argument _) ->
      skip ~reason:"no FIFOs here" ()

(* The one child of this process, once it has started. *)
let rec started_worker () =
  match children () with
  | [ worker ] -> worker
  | [] ->
      Domain.cpu_relax ();
      started_worker ()
  | workers -> failf "%d workers" (List.length workers)

(* The lines of Linux's /proc/PID/status for process [pid] that give the signals
   it blocks and those it ignores. *)
let signal_sets pid =
  let status = Printf.sprintf "/proc/%d/status" pid in
  In_channel.with_open_bin status In_channel.input_all
  |> String.split_on_char '\n'
  |> List.filter (fun l ->
      String.starts_with ~prefix:"SigBlk:" l
      || String.starts_with ~prefix:"SigIgn:" l)

(* The files in NVRTC's cache of binaries, the directory the test's dune file
   names in CUDA_CACHE_PATH. *)
let nvrtc_cache () =
  let dir = Sys.getenv "CUDA_CACHE_PATH" in
  let rec files dir =
    Array.to_list (Sys.readdir dir)
    |> List.concat_map (fun f ->
        let path = Filename.concat dir f in
        if Sys.is_directory path then files path else [ path ])
  in
  if Sys.file_exists dir then files dir else []

(* A memory file that cannot run: these tests run in a process of their own,
   where NVRTC_PATH names a file that is no library and XDG_CACHE_HOME a
   directory of the sandbox. The test makes the program's memory file one that
   the system refuses to run, as a sandbox's policy may refuse them all. *)

(* The memory file this process runs the program from, through /proc/self/fd:
   there is one, or the test skips where the system refuses to make one, as
   under vm.memfd_noexec=2. *)
let memory_file () =
  let fds = "/proc/self/fd" in
  let ours fd =
    match Unix.readlink (Filename.concat fds fd) with
    | link -> String.starts_with ~prefix:"/memfd:tolk-nvrtc-worker" link
    | exception Unix.Unix_error _ -> false
  in
  match List.filter ours (Array.to_list (Sys.readdir fds)) with
  | [ fd ] -> Filename.concat fds fd
  | [] -> skip ~reason:"the system makes no memory file of the program" ()
  | found -> failf "%d memory files of the program" (List.length found)

(* The program's files in the cache. *)
let cached_programs () =
  List.filter
    (fun f ->
      String.starts_with ~prefix:"nvrtc-worker-" f
      && not (Filename.check_suffix f ".tmp"))
    (Array.to_list (Sys.readdir Helpers.cache_dir))

let archs = [ "sm_75"; "sm_80"; "sm_86"; "sm_89" ]

let process =
  group "compiles in processes of their own"
    [
      test
        "kernels compiled on several domains at once are the bytes they are \
         compiled to in turn" (fun () ->
          let in_turn = List.map (fun arch -> ptx ~arch kernel) archs in
          let domains =
            List.map
              (fun arch -> Domain.spawn (fun () -> ptx ~arch kernel))
              archs
          in
          equal (list string) in_turn (List.map Domain.join domains));
      no_library
      @@ test "compiles from several domains at once each get the answer"
           (fun () ->
             let first = load_failure () in
             let domains = List.init 4 (fun _ -> Domain.spawn load_failure) in
             List.iter
               (equal (option string) first)
               (List.map Domain.join domains));
      no_library
      @@ test "a compile after its idle workers are killed gets the answer"
           (fun () ->
             let first = load_failure () in
             kill_children ();
             equal (list int) [] (children ());
             equal (option string) first (sigpipe_kills load_failure));
      no_library
      @@ test "the temporary directories of workers that ended are removed"
           (fun () ->
             ignore (scratch ());
             ignore (load_failure ());
             kill_children ();
             ignore (load_failure ());
             let dirs = worker_dirs () in
             equal int ~msg:(String.concat " " dirs) 1 (List.length dirs));
      no_library
      @@ test
           "a worker starts with every signal at its default action and none \
            blocked" (fun () ->
             if not (Sys.file_exists "/proc/self/status") then
               skip ~reason:"no /proc/PID/status: Linux only" ();
             kill_children ();
             let mask = Unix.sigprocmask SIG_BLOCK [ Sys.sigusr2 ] in
             ( Fun.protect ~finally:(fun () ->
                   ignore (Unix.sigprocmask SIG_SETMASK mask))
             @@ fun () ->
               with_signal Sys.sigchld Signal_ignore (fun () ->
                   ignore (load_failure ())) );
             let worker =
               match children () with
               | [ worker ] -> worker
               | workers -> failf "%d workers" (List.length workers)
             in
             equal (list string)
               [ "SigBlk:\t0000000000000000"; "SigIgn:\t0000000000000000" ]
               (signal_sets worker));
      (* With SIGCHLD ignored, a worker's status is lost: the system reaps it as
         it ends, and the reply alone decides. *)
      no_library
      @@ test
           "with SIGCHLD ignored, a compile after its idle workers are killed \
            gets the answer" (fun () ->
             let first = load_failure () in
             let workers = children () in
             with_signal Sys.sigchld Signal_ignore (fun () ->
                 kill workers;
                 equal (option string) first (sigpipe_kills load_failure)));
      test "a worker started with SIGCHLD ignored returns its binary" (fun () ->
          let lib = cubin kernel in
          kill_children ();
          equal string lib
            (with_signal Sys.sigchld Signal_ignore (fun () -> cubin kernel)));
      test
        "a compile whose process is killed raises Compile_error naming the \
         signal, and the next compile returns its binary" (fun () ->
          let lib = cubin kernel in
          let src = held_in_compile (temp_dir ()) in
          kill_children ();
          (* The compile starts a worker, which it never gives another request:
             killed before or after it reads this one, the worker fails it. *)
          let compile =
            Domain.spawn (fun () -> rejection (fun () -> cubin src))
          in
          Unix.kill (started_worker ()) Sys.sigkill;
          in_order
            ~subs:[ "nvrtc worker was killed by SIGKILL"; "without a reply" ]
            (require_some (Domain.join compile));
          equal string lib (cubin kernel));
      test
        "with SIGCHLD ignored, a compile whose process is killed raises \
         Compile_error saying it ended" (fun () ->
          ignore (cubin kernel);
          let src = held_in_compile (temp_dir ()) in
          kill_children ();
          let compile =
            Domain.spawn (fun () -> rejection (fun () -> cubin src))
          in
          let worker = started_worker () in
          (* The system reaps the worker as it ends: its status is lost. *)
          with_signal Sys.sigchld Signal_ignore (fun () ->
              Unix.kill worker Sys.sigkill;
              in_order
                ~subs:[ "nvrtc worker ended without a reply" ]
                (require_some (Domain.join compile))));
      test "a source that crashes NVRTC's assembler fails that compile only"
        (fun () ->
          let lib = cubin kernel in
          (match cubin crashes_assembler with
          | crashed -> starts_with ~prefix:"\x7fELF" crashed
          | exception Compiler.Compile_error msg ->
              in_order
                ~subs:[ "nvrtc worker was killed by SIG"; "without a reply" ]
                msg);
          equal string lib (cubin kernel));
      test "compiles leave NVRTC's cache of binaries empty" (fun () ->
          ignore (cubin kernel);
          ignore (ptx kernel);
          equal (list string) [] (nvrtc_cache ()));
      tag "unrunnable-memory"
      @@ test
           "where the memory file cannot run, a compile is served by a worker \
            run from the cache" (fun () ->
             if not (Sys.file_exists "/proc/self/fd") then
               skip ~reason:"no memory files: Linux only" ();
             let first = require_some (load_failure ()) in
             let memory = memory_file () in
             kill_children ();
             Unix.chmod memory 0o644;
             equal (option string) (Some first) (load_failure ());
             equal int ~msg:"programs in the cache" 1
               (List.length (cached_programs ())));
    ]

let () = exit (run "Tolk.Compiler_cuda" [ nvrtc_tests; process ])
