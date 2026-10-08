(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jobs over real agent processes on the loopback: support/agent.exe serves one
   job and prints how it ended. *)

open Windtrap
module B = Rig.Buffer

(* The most anything the test awaits may take. *)
let patience = 5.

let until ~what cond =
  let t0 = Unix.gettimeofday () in
  while not (cond ()) do
    if Unix.gettimeofday () -. t0 > patience then
      failf "%s: not within %.0f s" what patience;
    Thread.delay 0.01
  done

(* Keys *)

let key = String.init 32 (fun i -> Char.chr (65 + i))
let files = Atomic.make 0

(* A file of this user holding [contents], with permissions [perm]. *)
let write_file ?(perm = 0o600) contents =
  let file =
    Filename.concat
      (Filename.get_temp_dir_name ())
      (Printf.sprintf "rig-remote-%d-%d.key" (Unix.getpid ())
         (Atomic.fetch_and_add files 1))
  in
  (try Sys.remove file with Sys_error _ -> ());
  let oc = open_out_gen [ Open_wronly; Open_creat; Open_excl ] 0o600 file in
  output_string oc contents;
  close_out oc;
  Unix.chmod file perm;
  file

(* Runs [f] with a file of [key], removed after. *)
let with_key_file ?(key = key) f =
  let file = write_file key in
  Fun.protect ~finally:(fun () -> Sys.remove file) (fun () -> f file)

(* Agents *)

let agent_exe =
  Filename.concat (Filename.dirname Sys.executable_name) "support/agent.exe"

type agent = {
  pid : int;
  port : int;
  out : in_channel;
  mutable status : (int * string list) option; (* once ended *)
}

let address a = ("127.0.0.1", a.port)
let machine a = Printf.sprintf "127.0.0.1:%d" a.port

(* Starts an agent of the key in [file]. *)
let start ?(mode = "") file =
  let r, w = Unix.pipe ~cloexec:true () in
  let args = [| agent_exe; file; mode |] in
  let pid = Unix.create_process agent_exe args Unix.stdin w Unix.stderr in
  Unix.close w;
  let out = Unix.in_channel_of_descr r in
  match int_of_string (input_line out) with
  | port -> { pid; port; out; status = None }
  | exception (End_of_file | Failure _) ->
      ignore (Unix.waitpid [] pid);
      fail "the agent did not start"

let controller_exe =
  Filename.concat
    (Filename.dirname Sys.executable_name)
    "support/controller.exe"

(* Starts support/controller.exe in [mode] with [agents], as an agent: its
   output read by {!finish}. *)
let start_controller file mode agents =
  let r, w = Unix.pipe ~cloexec:true () in
  let ports = List.map (fun a -> string_of_int a.port) agents in
  let args = Array.of_list (controller_exe :: file :: mode :: ports) in
  let pid = Unix.create_process controller_exe args Unix.stdin w Unix.stderr in
  Unix.close w;
  { pid; port = 0; out = Unix.in_channel_of_descr r; status = None }

(* The agent's exit code and the lines it printed after its port, once it
   exited. *)
let finish a =
  match a.status with
  | Some s -> s
  | None ->
      let rec lines acc =
        match input_line a.out with
        | l -> lines (l :: acc)
        | exception End_of_file -> List.rev acc
      in
      let printed = lines [] in
      close_in a.out;
      let code =
        match snd (Unix.waitpid [] a.pid) with
        | Unix.WEXITED n -> n
        | Unix.WSIGNALED n | Unix.WSTOPPED n -> -n
      in
      let s = (code, printed) in
      a.status <- Some s;
      s

let exit_w =
  Testable.structural ~pp:(fun ppf (code, lines) ->
      Format.fprintf ppf "exit %d, printed [%s]" code (String.concat "; " lines))

let kill a =
  if a.status = None then begin
    (try Unix.kill a.pid Sys.sigkill with Unix.Unix_error _ -> ());
    ignore (finish a)
  end

(* Runs [f] with [n] agents of [key]; kills those still running after. *)
let with_agents ?(n = 1) ?mode f =
  with_key_file @@ fun file ->
  let agents = List.init n (fun _ -> start ?mode file) in
  Fun.protect ~finally:(fun () -> List.iter kill agents) (fun () -> f agents)

let connect agents =
  match Rig_remote.connect ~key (List.map address agents) with
  | Ok j -> j
  | Error why -> failf "connect: %s" why

(* Runs [f] with a job over [n] fresh agents, closed after unless it failed. *)
let with_job ?n f =
  with_agents ?n @@ fun agents ->
  let j = connect agents in
  Fun.protect ~finally:(fun () -> Rig_remote.close j) (fun () -> f j agents)

let mem h =
  match Rig_remote.devices h "MEM" with
  | Ok ds -> ds
  | Error why -> failf "devices: %s" why

(* Buffers *)

let read_host b =
  let s = Bytes.create (B.length b) in
  B.blit_to_bytes b 0 s 0 (B.length b);
  Bytes.to_string s

(* [b]'s bytes, through a copy to this process. *)
let read b =
  let h = B.create Rig.host (B.length b) in
  B.copy ~src:b ~dst:h;
  read_host h

let far_of_string d s =
  let b = B.create d (String.length s) in
  B.copy ~src:(B.of_string s) ~dst:b;
  b

let lost_why = function Rig.Lost (_, why) -> Some why | _ -> None
