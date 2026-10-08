(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Links over loopback, each row beside the floor that bounds it: the same bytes
   over a loopback socket set up as a link sets up its own, sent and received
   from C with nothing else. A row's distance to its floor is the link's: its
   frames, threads, queues and the OCaml of its calls. *)

open Rig_remote_bench
module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link
module B = Rig.Buffer

let strf = Printf.sprintf
let kib = 1024
let mib = 1024 * kib
let size n = if n >= mib then strf "%dM" (n / mib) else strf "%dK" (n / kib)

let ok what = function
  | Ok v -> v
  | Error why -> failwith (strf "%s: %s" what why)

let reason = function `Refused why | `Failed why -> why
let check what r = if r <> 0 then failwith (strf "%s: %d" what r)

(* The two ends of a new loopback TCP connection, tuned as a link tunes its
   socket. *)
let connected () =
  let l = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.bind l (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
  Unix.listen l 1;
  let port =
    match Unix.getsockname l with
    | Unix.ADDR_INET (_, p) -> p
    | Unix.ADDR_UNIX _ -> assert false
  in
  let d = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.connect d (Unix.ADDR_INET (Unix.inet_addr_loopback, port));
  let a, _ = Unix.accept ~cloexec:true l in
  Unix.close l;
  tune d;
  tune a;
  (d, a)

(* [forked f] runs [f] in a child process given the accepting end of a new
   connection, and is the dialing end and the child's pid. Thumper measures each
   row in a worker of its own, so the child is the worker's: it ends when the
   worker's end closes, also when the worker is killed. *)
let forked f =
  let d, a = connected () in
  match Unix.fork () with
  | 0 ->
      Unix.close d;
      Unix._exit (match f a with () -> 0 | exception _ -> 1)
  | pid ->
      Unix.close a;
      (d, pid)

let reap pid = ignore (Unix.waitpid [] pid)

(* Requests: a round trip of a request to an agent in another process over
   loopback, and the same bytes there and back. *)

let key = String.make 32 'k'

(* Room for either frame. *)
let message () = Bigarray.(Array1.create char c_layout request_bytes)

(* The agent's end of a job with this process: it answers every allocation,
   allocating nothing, until the job fails. *)
let far fd =
  let j = Link.job () in
  ignore (ok "accept" (Wire.accept fd ~key ~admit:(fun _ -> Ok ())));
  let l = Link.make j fd ~name:"controller" ~peer:Wire.Controller in
  let reply : type r. r Wire.request -> (r, string) result = function
    | Wire.Alloc _ -> Ok true
    | _ -> Error "the bench asks only for allocations"
  in
  let rec serve () =
    match Link.next l with
    | Ok (Wire.Request r) ->
        Link.answer l r (reply r);
        serve ()
    | Ok _ -> serve ()
    | Error _ -> ()
  in
  serve ()

(* The controller's end of a job with one agent. The bench ends the job by
   failing it: the agent sees the failure and leaves. *)
let controller () =
  let fd, pid = forked far in
  let j = Link.job () in
  ok "dial" (Wire.dial fd ~key ~self:Wire.Controller ~peer:(Wire.Agent 1));
  (j, Link.make j fd ~name:"agent" ~peer:(Wire.Agent 1), pid)

let ended (j, _, pid) =
  Link.fail j "the bench ended";
  reap pid

let request_rows =
  Thumper.group "request"
    [
      Thumper.bench_with_setup "alloc" ~setup:controller ~teardown:ended
        (fun (_, l, _) ->
          if not (ok "request" (Result.map_error reason (Link.request l alloc)))
          then failwith "request: refused");
      Thumper.bench_with_setup "floor"
        ~setup:(fun () ->
          ( message (),
            forked (fun a ->
                ignore (echo a (message ()) request_bytes answer_bytes)) ))
        ~teardown:(fun (_, (fd, pid)) ->
          Unix.close fd;
          reap pid)
        (fun (buf, (fd, _)) ->
          check "ask" (ask fd buf request_bytes answer_bytes));
    ]

(* Rails: one transfer of [n] bytes each run, from the controller's end of a
   link to the agent's end, both in this process over loopback: the sending
   machine's work stores [ready] and the receiving machine's waits for
   [arrived]. The floor sends the same bytes on a connection and waits for a
   thread receiving them on its other end. *)

let sizes = [ 4 * kib; 64 * mib ]

type rail = {
  job : Link.job;
  sender : Rig_remote_abi.end_;
  receiver : Rig_remote_abi.end_;
  mutable count : int;
}

let rail n () =
  let d, a = connected () in
  let job = Link.job () in
  let c = Link.make job d ~name:"agent" ~peer:(Wire.Agent 1) in
  let a = Link.make job a ~name:"controller" ~peer:Wire.Controller in
  let t : Rig_remote_abi.transfer = { src = 0; dst = 0; length = n } in
  let sender = Link.rail c ~id:1 ~send:[| t |] ~receive:[||] in
  let receiver = Link.rail a ~id:1 ~send:[||] ~receive:[| t |] in
  { job; sender; receiver; count = 0 }

let run r =
  r.count <- r.count + 1;
  check "rail" (rail_run r.sender r.receiver r.count)

let stream n () =
  let d, a = connected () in
  let st = stream_open d a n in
  if st = 0n then failwith "stream: no memory or thread";
  (st, d, a)

let rail_rows =
  Thumper.group "rail"
    (List.concat_map
       (fun n ->
         [
           Thumper.bench_with_setup (size n) ~setup:(rail n)
             ~teardown:(fun r -> Link.fail r.job "the bench ended")
             run;
           Thumper.bench_with_setup
             (strf "floor-%s" (size n))
             ~setup:(stream n)
             ~teardown:(fun (st, d, a) ->
               stream_close st;
               Unix.close d;
               Unix.close a)
             (fun (st, _, _) -> check "stream" (stream_run st));
         ])
       sizes)

(* Copies: [n] bytes between this process's memory and memory of another
   machine's host, each way, the job's agent [rig agent] in other processes over
   loopback. Their floors are the rails': the same bytes one way over a
   socket. *)

(* [rig agent], the binary built in dev/rig/bin, serving at a port it chooses:
   its process, its standard input, which ends it once closed, and its port. *)
let agent () =
  let exe =
    Filename.concat (Filename.dirname Sys.executable_name) "../../bin/main.exe"
  in
  let key_r, key_w = Unix.pipe ~cloexec:true () in
  let out_r, out_w = Unix.pipe ~cloexec:true () in
  let pid =
    Unix.create_process exe
      [| exe; "agent"; "127.0.0.1:0" |]
      key_r out_w Unix.stderr
  in
  Unix.close key_r;
  Unix.close out_w;
  let line = key ^ "\n" in
  ignore (Unix.write_substring key_w line 0 (String.length line));
  let ic = Unix.in_channel_of_descr out_r in
  let rec port () =
    match In_channel.input_line ic with
    | None -> None
    | Some l when String.starts_with ~prefix:"listening " l ->
        let i = String.rindex l ':' in
        int_of_string_opt (String.sub l (i + 1) (String.length l - i - 1))
    | Some _ -> port ()
  in
  let port = port () in
  close_in ic;
  match port with
  | Some port -> (pid, key_w, port)
  | None ->
      Unix.close key_w;
      reap pid;
      failwith "rig agent: no port"

type copy = {
  job : Rig_remote.t;
  pid : int;
  input : Unix.file_descr;
  near : B.t;
  far : B.t;
}

let copy n () =
  let pid, input, port = agent () in
  match Rig_remote.connect ~key [ ("127.0.0.1", port) ] with
  | Error why ->
      Unix.close input;
      reap pid;
      failwith why
  | Ok job ->
      let h = List.hd (Rig_remote.hosts job) in
      { job; pid; input; near = B.create Rig.host n; far = B.create h n }

let finished c =
  Rig_remote.close c.job;
  Unix.close c.input;
  reap c.pid

let copy_rows =
  Thumper.group "copy"
    (List.concat_map
       (fun n ->
         let row name f =
           Thumper.bench_with_setup
             (strf "%s-%s" name (size n))
             ~setup:(copy n) ~teardown:finished f
         in
         [
           row "to" (fun c -> B.copy ~src:c.near ~dst:c.far);
           row "from" (fun c -> B.copy ~src:c.far ~dst:c.near);
         ])
       sizes)

let () = exit @@ Thumper.run "rig_remote" [ request_rows; copy_rows; rail_rows ]
