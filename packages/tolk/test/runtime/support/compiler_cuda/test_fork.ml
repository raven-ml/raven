open Windtrap
open Tolk
module Compiler = Renderer.Compiler

(* Compiles across a fork. This program spawns no domain: a domain forbids a
   fork for the rest of a process's life. It runs with NVRTC and where
   NVRTC_PATH names a file that is no library, whose workers answer each compile
   with the reason. *)

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

let cubin src = Compiler.compile (Compiler_cuda.nvrtc ~ptx:false "sm_89") src

(* A compile's outcome as text: the binary's digest, or the error. *)
let compile () =
  match cubin kernel with
  | lib -> "binary " ^ Digest.to_hex (Digest.string lib)
  | exception Compiler.Compile_error msg -> "error " ^ msg

(* [compile ()], or a skip where no NVRTC is found, which starts no worker. *)
let first_compile () =
  match compile () with
  | outcome -> outcome
  | exception Failure msg when has_infix ~affix:"NVRTC_PATH" msg ->
      skip ~reason:msg ()

(* The pids of process [pid]'s children, which pgrep lists, in order; the test
   skips where pgrep does not run. *)
let children_of pid =
  match
    Unix.open_process_args_in "pgrep" [| "pgrep"; "-P"; string_of_int pid |]
  with
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

(* [f ()] run in a forked child, which writes it to a pipe and exits: the text,
   or the exception it raised. The test skips where processes do not fork. *)
let in_child f =
  flush_all ();
  let input, output = Unix.pipe ~cloexec:true () in
  match Unix.fork () with
  | exception Invalid_argument _ -> skip ~reason:"no fork here" ()
  | 0 ->
      Unix.close input;
      let text = try f () with e -> "raised " ^ Printexc.to_string e in
      ignore (Unix.write_substring output text 0 (String.length text));
      Unix._exit 0
  | pid ->
      Unix.close output;
      let text = In_channel.input_all (Unix.in_channel_of_descr input) in
      Unix.close input;
      ignore (Unix.waitpid [] pid);
      text

let fork =
  group "a forked process"
    [
      test "starts workers of its own, and both compile alike" (fun () ->
          let first = first_compile () in
          let workers = children () in
          not_equal (list int) ~msg:first [] workers;
          let own = Printf.sprintf "%s, with workers of its own: %b" in
          equal string (own first true)
            (in_child (fun () ->
                 let outcome = compile () in
                 own outcome (children () <> [])));
          equal string first (compile ());
          equal (list int) ~msg:"the parent's workers" workers (children ()));
      test "with standard input, output and error closed compiles alike"
        (fun () ->
          let first = first_compile () in
          equal string
            (first ^ "\n" ^ first)
            (in_child (fun () ->
                 Unix.close Unix.stdin;
                 Unix.close Unix.stdout;
                 Unix.close Unix.stderr;
                 let a = compile () in
                 (* Reaches a descriptor that took standard output's place. *)
                 (try
                    print_string "not a request";
                    flush stdout
                  with Sys_error _ -> ());
                 a ^ "\n" ^ compile ())));
    ]

(* A worker whose parent dies *)

(* [kernel] held in its compile, which includes a FIFO in [dir] that no process
   writes: NVRTC opens an included file, and the open of a FIFO waits for a
   writer. *)
let held_in_compile dir =
  let fifo = Filename.concat dir "held" in
  match Unix.mkfifo fifo 0o600 with
  | () -> Printf.sprintf "#include \"%s\"\n%s" fifo kernel
  | exception (Unix.Unix_error _ | Invalid_argument _) ->
      skip ~reason:"no FIFOs here" ()

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

(* The number of the openat system call on the host, as Linux's
   /proc/PID/syscall writes it first, from each architecture's system call table
   (unistd.h), where [uname -m] names one. *)
let openat_number =
  lazy
    (match Unix.open_process_args_in "uname" [| "uname"; "-m" |] with
    | exception Unix.Unix_error _ -> None
    | ic -> (
        let machine = String.trim (In_channel.input_all ic) in
        ignore (Unix.close_process_in ic);
        match machine with
        | "x86_64" -> Some "257"
        | "aarch64" | "riscv64" -> Some "56"
        | "ppc64" | "ppc64le" -> Some "286"
        | "s390x" -> Some "288"
        | _ -> None))

(* Returns once worker [pid] waits inside a compile in the open of a FIFO: its
   /proc/PID/wchan names the wait for a writer, or its main thread is in openat
   as /proc/PID/syscall says. Skips where /proc tells neither; fails once
   [parent], whose compile it serves, has ended, as when the compile returned
   instead of waiting. Asks every millisecond: no clock decides, and
   WINDTRAP_TIMEOUT ends a wait that never comes. *)
let rec until_in_fifo_open ~parent pid =
  let read file =
    match
      In_channel.with_open_bin
        (Printf.sprintf "/proc/%d/%s" pid file)
        In_channel.input_all
    with
    | s -> Some (String.trim s)
    | exception Sys_error _ -> None
  in
  let in_openat syscall =
    match (Lazy.force openat_number, String.split_on_char ' ' syscall) with
    | Some n, first :: _ -> first = n
    | _ -> false
  in
  match (read "wchan", read "syscall") with
  | Some "wait_for_partner", _ -> ()
  | _, Some syscall when in_openat syscall -> ()
  | (None | Some "0"), None ->
      skip ~reason:"/proc names no wait of the worker" ()
  | _ -> (
      match Unix.waitpid [ WNOHANG ] parent with
      | 0, _ ->
          Unix.sleepf 0.001;
          until_in_fifo_open ~parent pid
      | _ -> fail "the compile returned instead of waiting on the FIFO")

(* Returns once process [pid] has ended, asking every millisecond: no clock
   decides, and WINDTRAP_TIMEOUT ends a wait for an end that never comes. *)
let rec until_ended pid =
  if not (ended pid) then begin
    Unix.sleepf 0.001;
    until_ended pid
  end

(* In the forked parent: forks a process that sleeps until killed, holding what
   this one holds, and writes its pid to [out]. *)
let fork_sleeper out =
  match Unix.fork () with
  | 0 ->
      Unix.close out;
      Unix.sleepf 60.;
      Unix._exit 0
  | pid ->
      let s = string_of_int pid in
      ignore (Unix.write_substring out s 0 (String.length s));
      Unix.close out

(* The pid that [fork_sleeper] writes to [input], once it has. *)
let read_sleeper input =
  let pid = In_channel.input_all (Unix.in_channel_of_descr input) in
  Unix.close input;
  require_some ~msg:"the forked process" (int_of_string_opt pid)

(* The worker serves its parent's compile, which never returns, and is in it
   when the parent is killed. The forked process holds the parent's end of the
   worker's input open, so only the parent's death ends it. *)
let parent_death =
  group "a worker"
    [
      test
        "in a compile whose parent is killed ends and removes its directory, \
         though a process forked from the parent lives" (fun () ->
          if not (String.starts_with ~prefix:"binary" (first_compile ())) then
            skip ~reason:"NVRTC does not compile here" ();
          let held = held_in_compile (temp_dir ()) in
          flush_all ();
          let input, out = Unix.pipe ~cloexec:true () in
          let parent =
            match Unix.fork () with
            | exception Invalid_argument _ -> skip ~reason:"no fork here" ()
            | 0 ->
                Unix.close input;
                (try
                   ignore (compile ());
                   fork_sleeper out;
                   ignore (cubin held)
                 with _ -> ());
                Unix._exit 0
            | pid -> pid
          in
          Unix.close out;
          let sleeper = read_sleeper input in
          Fun.protect ~finally:(fun () -> Unix.kill sleeper Sys.sigkill)
          @@ fun () ->
          let workers =
            List.filter (fun pid -> pid <> sleeper) (children_of parent)
          in
          let temps = List.filter_map temp_of workers in
          List.iter (until_in_fifo_open ~parent) workers;
          (* What a compile leaves in its worker's directory goes with it. *)
          List.iter
            (fun temp ->
              Out_channel.with_open_bin
                (Filename.concat temp "left-by-a-compile") (fun oc ->
                  output_string oc "x"))
            temps;
          Unix.kill parent Sys.sigkill;
          ignore (Unix.waitpid [] parent);
          not_equal (list int) ~msg:"the parent's workers" [] workers;
          List.iter until_ended workers;
          not_equal (list string) [] temps;
          equal (list string) ~msg:"their temporary directories" []
            (List.filter Sys.file_exists temps));
    ]

let () = exit (run "Tolk.Compiler_cuda (fork)" [ fork; parent_death ])
