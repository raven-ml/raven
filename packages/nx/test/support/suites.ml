(* Runs the GPU suites that each test directory declares, in one process.

   suites.exe list COMMAND [-- COMMAND]... writes each COMMAND on a line of its
   own, after the digest of its program, so that the list changes when a program
   does.

   suites.exe run FILE... runs the commands of each list FILE in turn, printing
   each before it runs, so that a log names every suite run, and exits 1 if one
   failed.

   A COMMAND is [NAME=VALUE]... PROGRAM [ARG]..., words without spaces, with
   PROGRAM's path relative to the list's directory. PROGRAM runs from its own
   directory, as dune runs a rule's action from the rule's, in the environment
   with each NAME set to VALUE. Every command runs whatever the others' exits,
   so that one failing suite hides no other's report.

   One process asks dune for one job slot (see ../dune). *)

let commands args =
  let rec go cmd acc = function
    | [] -> List.rev (List.rev cmd :: acc)
    | "--" :: rest -> go [] (List.rev cmd :: acc) rest
    | a :: rest -> go (a :: cmd) acc rest
  in
  List.filter (fun c -> c <> []) (go [] [] args)

let name b =
  match String.index_opt b '=' with
  | Some i when i > 0 -> Some (String.sub b 0 i)
  | _ -> None

let rec bindings acc = function
  | b :: rest when name b <> None -> bindings (b :: acc) rest
  | argv -> (List.rev acc, argv)

let fail fmt =
  Printf.ksprintf
    (fun s ->
      prerr_endline s;
      exit 2)
    fmt

(* list *)

let blank w = String.exists (fun c -> c = ' ' || c = '\t' || c = '\n') w

let line cmd =
  match bindings [] cmd with
  | _, [] -> fail "suites.exe list: %s: no program" (String.concat " " cmd)
  | _, prog :: _ ->
      (match List.find_opt blank cmd with
      | Some w -> fail "suites.exe list: %S holds a blank" w
      | None -> ());
      Digest.to_hex (Digest.file prog) ^ " " ^ String.concat " " cmd

let list args = List.iter (fun c -> print_endline (line c)) (commands args)

(* run *)

let environment set =
  let names = List.filter_map name set in
  let kept =
    List.filter
      (fun b ->
        match name b with Some n -> not (List.mem n names) | None -> true)
      (Array.to_list (Unix.environment ()))
  in
  Array.of_list (kept @ set)

let spawn prog args env =
  let here = Sys.getcwd () in
  let exe =
    Filename.concat Filename.current_dir_name (Filename.basename prog)
  in
  Sys.chdir (Filename.dirname prog);
  Fun.protect
    ~finally:(fun () -> Sys.chdir here)
    (fun () ->
      Unix.create_process_env exe
        (Array.of_list (exe :: args))
        env Unix.stdin Unix.stdout Unix.stderr)

(* [p] without a leading "./", which dune's %{dep:} writes. *)
let strip p =
  if String.starts_with ~prefix:"./" p then String.sub p 2 (String.length p - 2)
  else p

let run_one dir cmd =
  match bindings [] cmd with
  | _, [] -> Some (String.concat " " cmd ^ ": no program")
  | set, prog :: args -> (
      let prog = Filename.concat dir (strip prog) in
      Printf.eprintf "suites.exe: %s\n%!"
        (String.concat " " (set @ (prog :: args)));
      let pid = spawn prog args (environment set) in
      match snd (Unix.waitpid [] pid) with
      | WEXITED 0 -> None
      | WEXITED n -> Some (Printf.sprintf "%s exited %d" prog n)
      | WSIGNALED n -> Some (Printf.sprintf "%s killed by signal %d" prog n)
      | WSTOPPED n -> Some (Printf.sprintf "%s stopped by signal %d" prog n))

(* The commands of the list [file], each with the directory its program's path
   is relative to. A line's first word is its program's digest. *)
let read file =
  let words l = List.filter (( <> ) "") (String.split_on_char ' ' l) in
  In_channel.with_open_text file In_channel.input_lines
  |> List.filter_map (fun l ->
      match words l with
      | [] -> None
      | [ _ ] -> fail "suites.exe run: %s: %S has no command" file l
      | _ :: cmd -> Some (Filename.dirname file, cmd))

let run files =
  let failed =
    List.concat_map read files
    |> List.filter_map (fun (dir, cmd) -> run_one dir cmd)
  in
  List.iter (Printf.eprintf "suites.exe: %s\n") failed;
  exit (if failed = [] then 0 else 1)

let () =
  match List.tl (Array.to_list Sys.argv) with
  | "list" :: args -> list args
  | "run" :: files -> run files
  | _ -> fail "usage: suites.exe (list COMMAND [-- COMMAND]... | run FILE...)"
