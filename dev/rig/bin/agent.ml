(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* The directories driver-less paths read GPU firmware from. *)
let firmware = [ "/lib/firmware" ]
let say l = Proc.write Unix.stdout (Line.to_string l)

(* The job's key: the first line of [fd], read a byte at a time so that nothing
   after it is taken, once [Rig_remote.key] accepts it. A line that grows from a
   key into none is longer than any key: it is refused there, read no
   further. *)
let read_key fd =
  let b = Buffer.create 64 and c = Bytes.create 1 in
  let key () = Rig_remote.key (Buffer.contents b) in
  let rec go was_key =
    match Unix.read fd c 0 1 with
    | 0 when Buffer.length b = 0 -> Error "no key on standard input"
    | n when n = 0 || Bytes.get c 0 = '\n' ->
        Result.map (fun _ -> Buffer.contents b) (key ())
    | _ -> (
        Buffer.add_char b (Bytes.get c 0);
        match key () with
        | Error _ as e when was_key -> e
        | k -> go (Result.is_ok k))
    | exception Unix.Unix_error (Unix.EINTR, _, _) -> go was_key
  in
  go false

let fail why =
  say (Line.Failed why);
  exit 123

(* The half *)

let half address =
  say (Line.Agent Line.version);
  let key =
    match read_key Unix.stdin with Ok k -> k | Error why -> fail why
  in
  Proc.signals [];
  let key_r, key_w = Proc.pipe () and out_r, out_w = Proc.pipe () in
  let env = Array.append (Unix.environment ()) [| "RIG_REMOTE_REPORT=1" |] in
  let exe = Sys.executable_name in
  let pid =
    Proc.spawn ~env exe
      [| exe; "agent"; address |]
      ~stdin:key_r ~stdout:out_w ~stderr:Unix.stderr
  in
  Unix.close key_r;
  Unix.close out_w;
  Proc.write key_w (key ^ "\n");
  Unix.close key_w;
  let lines = Line.reader out_r and input = Line.reader Unix.stdin in
  let reported = ref None and killed = ref false and status = ref None in
  let relay l =
    match Line.of_string l with
    | Some ((Line.Closed | Line.Failed _) as l) when !reported = None ->
        reported := Some l;
        say l
    | Some ((Line.Waiting | Line.Listening _) as l) -> say l
    | _ -> ()
  in
  let rec loop () =
    Proc.wait [ lines; input ];
    if !status = None then status := Proc.reap pid;
    ignore (Line.read input);
    if Line.ended input && !status = None && not !killed then begin
      killed := true;
      Proc.kill pid
    end;
    List.iter relay (Line.read lines);
    if !status = None || not (Line.ended lines) then loop ()
  in
  loop ();
  (* The agent died unreported, unless this half killed it. *)
  match (!status, !reported) with
  | Some (Unix.WSIGNALED s), None when !killed && s = Sys.sigkill -> exit 123
  | Some st, None ->
      say (Line.Died (Proc.cause st));
      exit 123
  | _, r -> exit (if r = Some Line.Closed then 0 else 123)

(* The agent *)

(* Holds the machine's lock for the life of the process: one agent of this user
   runs on a machine at a time. *)
let lock () =
  let path =
    Filename.concat
      (Filename.get_temp_dir_name ())
      (strf "rig-agent-%d.lock" (Unix.getuid ()))
  in
  let take () =
    let fd =
      Unix.openfile path [ Unix.O_RDWR; Unix.O_CREAT; Unix.O_CLOEXEC ] 0o600
    in
    match Unix.lockf fd Unix.F_TLOCK 0 with
    | () -> Ok ()
    | exception Unix.Unix_error ((Unix.EAGAIN | Unix.EACCES), _, _) ->
        say Line.Waiting;
        let rec wait () =
          try Unix.lockf fd Unix.F_LOCK 0
          with Unix.Unix_error (Unix.EINTR, _, _) -> wait ()
        in
        wait ();
        Ok ()
  in
  match Unix.lstat path with
  | { st_kind = Unix.S_REG; st_uid; _ } when st_uid <> Unix.getuid () ->
      Error (strf "%s belongs to another user" path)
  | { st_kind = Unix.S_REG; _ } -> take ()
  | _ -> Error (strf "%s is no regular file" path)
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> take ()
  | exception Unix.Unix_error (e, _, _) ->
      Error (strf "%s: %s" path (Unix.error_message e))

(* Every GPU a path counts, opened by its driver. *)
let gpus (type a) (module D : Rig.Driver with type t = a) count name open_ () =
  let n = count () in
  let rec go acc i =
    if i = n then Ok (List.rev acc)
    else
      match Rig.open_ (module D) ~name:(name i) (fun () -> open_ i) with
      | Ok d -> go (d :: acc) (i + 1)
      | Error _ as e -> e
  in
  go [] 0

let kinds =
  [
    ( "METAL",
      gpus
        (module Rig_metal)
        Rig_metal.count Rig_metal.device_name Rig_metal.open_ );
    ( "CUDA",
      gpus (module Rig_cuda) Rig_cuda.count Rig_cuda.device_name Rig_cuda.open_
    );
    ( "NV",
      gpus
        (module Rig_nv)
        Rig_nv_nvidia.count Rig_nv_nvidia.device_name Rig_nv_nvidia.open_ );
    ( "AMD",
      gpus
        (module Rig_amd)
        Rig_amd_amdgpu.count Rig_amd_amdgpu.device_name Rig_amd_amdgpu.open_ );
    ( "NV-PCI",
      gpus
        (module Rig_nv)
        (fun () -> Rig_nv_pci.count ())
        Rig_nv_pci.device_name
        (fun i -> Rig_nv_pci.open_ ~firmware i) );
    ( "AMD-PCI",
      gpus
        (module Rig_amd)
        (fun () -> Rig_amd_pci.count ())
        Rig_amd_pci.device_name
        (fun i -> Rig_amd_pci.open_ ~firmware i) );
  ]

let agent host port =
  let key =
    match Result.bind (read_key Unix.stdin) Rig_remote.key with
    | Ok k -> k
    | Error why -> fail why
  in
  (match lock () with Ok () -> () | Error why -> fail why);
  let a =
    match Rig_remote.listen ~key host port with
    | Ok a -> a
    | Error why -> fail why
  in
  say (Line.Listening (Address.with_port host (Rig_remote.port a)));
  match Rig_remote.serve a kinds with Ok () -> exit 0 | Error _ -> exit 123
