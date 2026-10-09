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

(* [k] as a job's key. *)
let as_key k = Result.get_ok (Rig_remote.key k)
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

(* This process's environment and the variables [vars]. *)
let environment vars =
  Array.append (Unix.environment ())
    (Array.of_list (List.map (fun (k, v) -> k ^ "=" ^ v) vars))

(* Starts an agent of the key in [file], with the variables [vars] added to its
   environment. *)
let start ?(mode = "") ?(vars = []) file =
  let r, w = Unix.pipe ~cloexec:true () in
  let args = [| agent_exe; file; mode |] in
  let pid =
    Unix.create_process_env agent_exe args (environment vars) Unix.stdin w
      Unix.stderr
  in
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
  match Rig_remote.connect ~key:(as_key key) (List.map address agents) with
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

(* Launched programs *)

(* A key as a launcher gives it: 64 hexadecimal characters. *)
let hex_key = String.init 64 (fun i -> "0123456789abcdef".[i * 7 mod 16])

let launched_exe =
  Filename.concat (Filename.dirname Sys.executable_name) "support/launched.exe"

(* The variables a launcher sets for a program whose report descriptor is 3,
   each agent named [mi] for the [i]th. *)
let launch_vars agents =
  let item i a = Printf.sprintf "m%d=127.0.0.1:%d" (i + 1) a.port in
  [
    ("RIG_REMOTE_REPORT", "3");
    ("RIG_REMOTE_AGENTS", String.concat "," (List.mapi item agents));
    ("RIG_REMOTE_KEY", hex_key);
  ]

type launch = {
  lpid : int;
  report : in_channel;
  said : in_channel;
  mutable heard : bool;  (** A line of [said] was read. *)
}

(* Starts support/launched.exe in [mode] with the variables [vars] added to its
   environment, as a launcher does: its descriptor 3 is a pipe read here, the
   report, and its standard output and error another, what it said. With
   [~reader_gone:true] the report's reader is closed before the program starts,
   so each of its report writes fails, and the report reads as no lines. *)
let launch ?(args = []) ?(reader_gone = false) vars mode =
  let rr, rw = Unix.pipe ~cloexec:true () in
  let rr =
    if not reader_gone then rr
    else begin
      Unix.close rr;
      let er, ew = Unix.pipe ~cloexec:true () in
      Unix.close ew;
      er
    end
  in
  let sr, sw = Unix.pipe ~cloexec:true () in
  let sh = "/bin/sh" in
  let argv =
    Array.of_list
      ([ sh; "-c"; "exec \"$0\" \"$@\" 3>&1 >&2"; launched_exe; mode ] @ args)
  in
  let lpid =
    Unix.create_process_env sh argv (environment vars) Unix.stdin rw sw
  in
  Unix.close rw;
  Unix.close sw;
  {
    lpid;
    report = Unix.in_channel_of_descr rr;
    said = Unix.in_channel_of_descr sr;
    heard = false;
  }

let lines ic =
  let rec go acc =
    match input_line ic with
    | l -> go (l :: acc)
    | exception End_of_file -> List.rev acc
  in
  let ls = go [] in
  close_in ic;
  ls

(* The debug runtime, which the sanitize profile links, starts a program's
   standard error with these lines, before anything the program says. *)
let banner =
  [
    "### OCaml runtime: debug mode ###";
    "### set OCAMLRUNPARAM=v=0 to silence this message";
  ]

(* The next line the program said: past the runtime's banner, for the first. *)
let said_line p =
  let rec past = function
    | [] -> input_line p.said
    | b :: bs ->
        let l = input_line p.said in
        if String.equal l b then past bs else l
  in
  let first = not p.heard in
  p.heard <- true;
  if first then past banner else input_line p.said

(* The program's exit code, its report's lines and what it said, once it
   exited. *)
let ended p =
  let report = lines p.report in
  let rec go acc =
    match said_line p with
    | l -> go (l :: acc)
    | exception End_of_file ->
        close_in p.said;
        List.rev acc
  in
  let said = go [] in
  let code =
    match snd (Unix.waitpid [] p.lpid) with
    | Unix.WEXITED n -> n
    | Unix.WSIGNALED n | Unix.WSTOPPED n -> -n
  in
  (code, report, said)

(* Runs [f] with [n] agents of [hex_key]. *)
let with_hex_agents ?(n = 1) ?vars f =
  with_key_file ~key:hex_key @@ fun file ->
  let agents = List.init n (fun _ -> start ?vars file) in
  Fun.protect ~finally:(fun () -> List.iter kill agents) (fun () -> f agents)

(* A raw controller *)

(* The frames of wire.mli, written and read byte by byte, for what a controller
   of rig.remote never sends. *)

let k_request = 1
let k_answer = 2
let k_handover = 3
let k_beat = 8
let k_abort = 9
let k_close = 10
let u8 n = String.make 1 (Char.chr n)

let u32 n =
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 (Int32.of_int n);
  Bytes.to_string b

let u64 n =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int n);
  Bytes.to_string b

let str s = u32 (String.length s) ^ s
let frame kind payload = u64 (String.length payload) ^ u8 kind ^ payload

(* A connection to agent [a] as its controller, once the handshake proved [key].
   A read gives up after [patience]. *)
let raw_controller a =
  let module Wire = Rig_remote_proxy.Wire in
  let fd = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.connect fd (Unix.ADDR_INET (Unix.inet_addr_loopback, a.port));
  match Wire.dial fd ~key ~self:Wire.Controller ~peer:(Wire.Agent 1) with
  | Error why ->
      Unix.close fd;
      failf "dial: %s" why
  | Ok () ->
      Unix.setsockopt_float fd Unix.SO_RCVTIMEO patience;
      fd

let send fd s = ignore (Unix.write_substring fd s 0 (String.length s))

(* The next [n] bytes of [fd], fewer if its stream ends first. *)
let read_n fd n =
  let b = Bytes.create n in
  let rec go off =
    if off = n then off
    else
      match Unix.read fd b off (n - off) with
      | 0 -> off
      | k -> go (off + k)
      | exception Unix.Unix_error (Unix.ECONNRESET, _, _) -> off
      | exception Unix.Unix_error ((Unix.EAGAIN | Unix.EWOULDBLOCK), _, _) ->
          failf "no byte within %.0f s" patience
  in
  Bytes.sub_string b 0 (go 0)

(* The next frame of [fd] other than a beat, its kind and payload, or [None]
   once its stream ends. *)
let rec next_frame fd =
  let h = read_n fd 9 in
  if String.length h < 9 then None
  else
    let n = Int64.to_int (String.get_int64_le h 0) in
    if n < 0 || n > 1 lsl 20 then failf "a frame of %d bytes" n;
    let p = read_n fd n in
    if String.length p < n then None
    else if Char.code h.[8] = k_beat then next_frame fd
    else Some (Char.code h.[8], p)

(* The answer to the request whose payload is [r]: [Ok] its bytes, or [Error]
   the agent's refusal. *)
let ask fd r =
  send fd (frame k_request r);
  match next_frame fd with
  | Some (k, p) when k = k_answer && p.[0] = '\000' ->
      Ok (String.sub p 1 (String.length p - 1))
  | Some (k, p) when k = k_answer ->
      Error (String.sub p 5 (String.length p - 5))
  | Some (k, _) -> failf "a frame of kind %d in place of an answer" k
  | None -> fail "the agent ended the connection in place of an answer"

(* A join of the agent [a] alone. *)
let join_alone fd a =
  ask fd (u8 1 ^ u32 1 ^ str (machine a) ^ str "127.0.0.1" ^ u32 a.port)

(* An allocation of [bytes] of the agent's host memory as [id]. *)
let alloc_host id bytes = u8 3 ^ u64 id ^ u64 0 ^ u8 0 ^ u64 bytes

(* A rail [id] with this process carrying [send] from the agent's machine, each
   a transfer's source, destination and length. *)
let rail_out id send =
  let transfer (src, dst, length) = u64 src ^ u64 dst ^ u64 length in
  u8 7 ^ u64 id ^ u32 0
  ^ u32 (List.length send)
  ^ String.concat "" (List.map transfer send)
  ^ u32 0

(* A hand-over of the host's work at [value]: one copy of [bytes] from memory
   [src] at [src_at] to memory [dst] at [dst_at]. *)
let copy_on_host ~value ~bytes (src, src_at) (dst, dst_at) =
  let region id at = u8 0 ^ u64 id ^ u64 at in
  u64 0 ^ u64 value ^ u32 0 ^ u32 1 ^ u8 1 ^ u64 bytes ^ region src src_at
  ^ region dst dst_at
