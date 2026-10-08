(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Links over loopback, each row beside the floor that bounds it: the same bytes
   over a loopback socket set up as a link sets up its own, sent and received
   from C with nothing else. A row's distance to its floor is the link's: its
   frames, threads, queues and the OCaml of its calls. *)

module Wire = Rig_remote_proxy.Wire
module Link = Rig_remote_proxy.Link

external tune : Unix.file_descr -> unit = "rig_remote_bench_tune"

external ask : Unix.file_descr -> Rig_remote_abi.area -> int -> int -> int
  = "rig_remote_bench_ask"

external echo : Unix.file_descr -> Rig_remote_abi.area -> int -> int -> int
  = "rig_remote_bench_echo"

external stream_open : Unix.file_descr -> Unix.file_descr -> int -> nativeint
  = "rig_remote_bench_stream_open"

external stream_run : nativeint -> int = "rig_remote_bench_stream_run"
external stream_close : nativeint -> unit = "rig_remote_bench_stream_close"

external rail_run : Rig_remote_abi.area -> Rig_remote_abi.area -> int -> int
  = "rig_remote_bench_rail_run"

let strf = Printf.sprintf
let kib = 1024
let mib = 1024 * kib
let size n = if n >= mib then strf "%dM" (n / mib) else strf "%dK" (n / kib)

let ok what = function
  | Ok v -> v
  | Error why -> failwith (strf "%s: %s" what why)

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
let alloc = Wire.Alloc { id = 1; device = 0; memory = `Device; bytes = 4096 }

(* Its frame: a header of 9 bytes, the kind, id, device, memory and bytes; the
   answer's: the header, 0 and the [bool]. *)
let request_bytes = 9 + 1 + 8 + 8 + 1 + 8
let answer_bytes = 9 + 1 + 1

(* Room for either frame. *)
let message () = Bigarray.(Array1.create char c_layout request_bytes)

let reply : type r. r Wire.request -> (r, string) result = function
  | Wire.Alloc _ -> Ok true
  | _ -> Error "the bench asks only for allocations"

(* An agent answering every request until its job fails. *)
let agent fd =
  let j = Link.job () in
  ignore (ok "accept" (Wire.accept fd ~key ~admit:(fun _ -> Ok ())));
  let l = Link.make j fd ~name:"controller" ~peer:Wire.Controller in
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
  let fd, pid = forked agent in
  let j = Link.job () in
  ok "dial" (Wire.dial fd ~key ~self:Wire.Controller ~peer:(Wire.Agent 1));
  (j, Link.make j fd ~name:"agent" ~peer:(Wire.Agent 1), pid)

let request_rows =
  Thumper.group "request"
    [
      Thumper.bench_with_setup "alloc" ~setup:controller
        ~teardown:(fun (j, _, pid) ->
          Link.fail j "the bench ended";
          reap pid)
        (fun (_, l, _) ->
          if not (ok "request" (Link.request l alloc)) then
            failwith "request: refused");
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
  check "rail" (rail_run r.sender.counts r.receiver.counts r.count)

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

(* Windows has no fork for the agent's process. *)
let () =
  exit
  @@ Thumper.run "rig_remote"
       ((if Sys.win32 then [] else [ request_rows ]) @ [ rail_rows ])
