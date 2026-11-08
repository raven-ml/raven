open Windtrap
open Tolk
module Compiler = Renderer.Compiler

(* Compiles across a fork. This program spawns no domain: a domain forbids a
   fork for the rest of a process's life. *)

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

(* A compile's outcome as text: the binary's digest, or the error. *)
let compile () =
  match Compiler.compile (Compiler_cuda.nvrtc ~ptx:false "sm_89") kernel with
  | lib -> "binary " ^ Digest.to_hex (Digest.string lib)
  | exception Compiler.Compile_error msg -> "error " ^ msg

(* [compile ()], or a skip where no NVRTC is found, which starts no worker. *)
let first_compile () =
  match compile () with
  | outcome -> outcome
  | exception Failure msg when has_infix ~affix:"NVRTC_PATH" msg ->
      skip ~reason:msg ()

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

(* The pids of process [pid]'s children. *)
let children_of pid =
  match
    Unix.open_process_args_in "pgrep" [| "pgrep"; "-P"; string_of_int pid |]
  with
  | exception Unix.Unix_error _ -> skip ~reason:"pgrep does not run" ()
  | ic -> (
      let listed = In_channel.input_all ic in
      match Unix.close_process_in ic with
      | WEXITED (0 | 1) ->
          List.filter_map int_of_string_opt (String.split_on_char '\n' listed)
      | _ -> skip ~reason:"pgrep failed" ())

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

(* The exit code of [f ()] run in a forked child, which exits with it at
   once; the test skips where processes do not fork. *)
let in_child f =
  flush_all ();
  match Unix.fork () with
  | exception Invalid_argument _ -> skip ~reason:"no fork here" ()
  | 0 -> Unix._exit (try f () with _ -> 99)
  | pid -> (
      match snd (Unix.waitpid [] pid) with
      | WEXITED n -> n
      | WSIGNALED s | WSTOPPED s -> 128 + s)

(* The child's exit codes. *)
let shares_workers = 2
and differs = 3

let fork =
  group "a forked process"
    [
      test "starts workers of its own, and both compile alike" (fun () ->
          let first = first_compile () in
          let workers = children () in
          not_equal (list int) [] workers;
          let code =
            in_child (fun () ->
                let outcome = compile () in
                if children () = [] then shares_workers
                else if outcome <> first then differs
                else 0)
          in
          equal int ~msg:"the child's exit code" 0 code;
          equal string first (compile ());
          equal (list int) workers (children ()));
      test "with standard input and output closed compiles alike" (fun () ->
          let first = first_compile () in
          let code =
            in_child (fun () ->
                Unix.close Unix.stdin;
                Unix.close Unix.stdout;
                let a = compile () in
                (* Reaches a descriptor that took standard output's place. *)
                (try
                   print_string "not a request";
                   flush stdout
                 with Sys_error _ -> ());
                let b = compile () in
                if a <> first || b <> first then differs else 0)
          in
          equal int ~msg:"the child's exit code" 0 code);
    ]

(* A worker whose parent dies during a compile *)

(* [kernel] including the FIFO [fifo]: its compile waits in the open of [fifo]
   until a writer opens it. NVRTC then reads as many bytes as the FIFO's size,
   none. *)
let waiting fifo = Printf.sprintf "#include \"%s\"\n%s" fifo kernel

(* Whether process [pid] waits in the open of a FIFO for a writer, as Linux's
   /proc/PID/wchan names it. *)
let waits_for_writer pid =
  let wchan = Printf.sprintf "/proc/%d/wchan" pid in
  match In_channel.with_open_bin wchan In_channel.input_all with
  | s -> s = "wait_for_partner"
  | exception Sys_error _ -> false

(* Lets a compile waiting in the open of [fifo] go on: a writer that opens and
   closes it. *)
let release fifo =
  match Unix.openfile fifo [ O_WRONLY; O_NONBLOCK; O_CLOEXEC ] 0 with
  | fd -> Unix.close fd
  | exception Unix.Unix_error (ENXIO, _, _) -> ()

let parent_death =
  group "a parent that dies"
    [
      test "during a compile ends its worker" (fun () ->
          if not (String.starts_with ~prefix:"binary" (first_compile ())) then
            skip ~reason:"NVRTC does not compile here" ();
          let dir = Filename.temp_dir "tolk-test-fork" "" in
          let fifo = Filename.concat dir "include.fifo" in
          Unix.mkfifo fifo 0o600;
          Fun.protect ~finally:(fun () ->
              release fifo;
              Sys.remove fifo;
              Unix.rmdir dir)
          @@ fun () ->
          flush_all ();
          let parent =
            match Unix.fork () with
            | exception Invalid_argument _ -> skip ~reason:"no fork here" ()
            | 0 ->
                (try
                   ignore
                     (Compiler.compile
                        (Compiler_cuda.nvrtc ~ptx:false "sm_89")
                        (waiting fifo))
                 with _ -> ());
                Unix._exit 0
            | pid -> pid
          in
          let waits () = List.exists waits_for_writer (children_of parent) in
          let reached = until ~seconds:20. waits in
          let workers = children_of parent in
          Unix.kill parent Sys.sigkill;
          ignore (Unix.waitpid [] parent);
          equal bool ~msg:"the compile waits on the FIFO" true reached;
          not_equal (list int) [] workers;
          let alive =
            List.filter
              (fun pid -> not (until ~seconds:10. (fun () -> ended pid)))
              workers
          in
          equal (list int) ~msg:"workers alive" [] alive);
    ]

let () = exit (run "Tolk.Compiler_cuda (fork)" [ fork; parent_death ])
