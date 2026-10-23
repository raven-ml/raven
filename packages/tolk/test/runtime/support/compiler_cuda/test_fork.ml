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
  let outcome = compile () in
  if has_infix ~affix:"NVRTC_PATH" outcome then skip ~reason:outcome ();
  outcome

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

let () = exit (run "Tolk.Compiler_cuda (fork)" [ fork ])
