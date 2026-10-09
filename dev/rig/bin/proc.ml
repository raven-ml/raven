(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let spawn ?(env = Unix.environment ()) prog args ~stdin ~stdout ~stderr =
  Unix.create_process_env prog args env stdin stdout stderr

let pipe () = Unix.pipe ~cloexec:true ()

let write fd s =
  try ignore (Unix.write_substring fd s 0 (String.length s))
  with Unix.Unix_error _ -> ()

(* A descriptor's number, which the launchers pass to the processes they start
   in RIG_REMOTE_REPORT: on POSIX, the only systems rig runs on, a
   [Unix.file_descr] is its number. *)
external fd_number : Unix.file_descr -> int = "%identity"

let inherited fd f =
  Unix.clear_close_on_exec fd;
  Fun.protect ~finally:(fun () -> Unix.set_close_on_exec fd) f

let rec reap pid =
  match Unix.waitpid [ Unix.WNOHANG ] pid with
  | 0, _ -> None
  | _, st -> Some st
  | exception Unix.Unix_error (Unix.EINTR, _, _) -> reap pid

let signal pid s = try Unix.kill pid s with Unix.Unix_error _ -> ()
let kill pid = signal pid Sys.sigkill

(* OCaml names the POSIX signals and gives others as their own number. *)
let name s =
  if s >= 0 then Printf.sprintf "signal %d" s else Sys.signal_to_string s

let cause = function
  | Unix.WEXITED n -> Printf.sprintf "exited with status %d" n
  | Unix.WSIGNALED s | Unix.WSTOPPED s -> Printf.sprintf "killed by %s" (name s)

let status = function
  | Unix.WEXITED n -> n
  | Unix.WSIGNALED s | Unix.WSTOPPED s -> 128 + Sys.signal_to_int s

(* Signals *)

(* The first interrupt, set once; [taken] once [interrupted] gave it. Each
   handler also writes a byte on [wake], which [wait] watches: a signal that
   comes just before [wait] blocks still wakes it. *)
let interrupt = ref None
let taken = ref false
let wake = pipe ()

let signals interrupts =
  let r, w = wake in
  Unix.set_nonblock r;
  Unix.set_nonblock w;
  let note s =
    if !interrupt = None && List.mem s interrupts then interrupt := Some s;
    try ignore (Unix.write_substring w "s" 0 1) with Unix.Unix_error _ -> ()
  in
  let handle s =
    match Sys.signal s (Sys.Signal_handle note) with
    | Sys.Signal_ignore when s <> Sys.sigchld ->
        Sys.set_signal s Sys.Signal_ignore
    | _ -> ()
  in
  List.iter handle (Sys.sigchld :: Sys.sigpipe :: interrupts)

let rec drain fd =
  match Unix.read fd (Bytes.create 64) 0 64 with
  | 0 -> ()
  | _ -> drain fd
  | exception Unix.Unix_error _ -> ()

let wait ?until rs =
  let fds =
    List.filter_map
      (fun r -> if Line.ended r then None else Some (Line.fd r))
      rs
  in
  let timeout =
    match until with
    | None -> -1.
    | Some t -> Float.max 0. (t -. Unix.gettimeofday ())
  in
  (try ignore (Unix.select (fst wake :: fds) [] [] timeout)
   with Unix.Unix_error (Unix.EINTR, _, _) -> ());
  drain (fst wake)

let interrupted () =
  match !interrupt with
  | Some _ as s when not !taken ->
      taken := true;
      s
  | _ -> None

let die_by s =
  Sys.set_signal s Sys.Signal_default;
  Unix.kill (Unix.getpid ()) s;
  exit (128 + Sys.signal_to_int s)
