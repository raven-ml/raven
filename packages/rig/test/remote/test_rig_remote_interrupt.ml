(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interrupt walks of rig.remote. A call is made once for each allocation it
   makes, with a landing at that allocation on the calling thread: another
   call from a Gc.Memprof callback, an exit from one, an exception or a
   Ctrl-C. The callback's call answers as it would outside the callback, no
   device is lost, and a launched job's report ends as the job does. A
   connect cut leaves the process able to start another job, and a rail cut
   leaves its job going. An agent's serve, its acceptor and its handshakes,
   cut in a forked child, leave the agent serving or no longer listening,
   and no controller waiting. *)

open Windtrap
open Remote_job

let strf = Printf.sprintf

(* [walk attempt] runs [attempt n] from [n = 1] until an attempt no longer
   reaches its [n]-th allocation. *)
let walk attempt =
  let rec from n = if attempt n then from (n + 1) in
  from 1

let names ds = List.map Rig.name ds

(* Devices *)

(* A devices of MEM from a callback in a devices of NONE, a kind whose opener
   fails, so that each call asks the agent: both answer, the host is not lost,
   and MEM gives the devices it gave first. *)
let devices_in_devices () =
  with_job @@ fun j agents ->
  let h = List.hd (Rig_remote.hosts j) in
  let at = at (List.hd agents) (Rig.name h) in
  let first = ref [] in
  walk (fun n ->
      let outer = ref (Ok []) and inner = ref None in
      let call () = inner := Some (Rig_remote.devices h "MEM") in
      let reached =
        Rig_support.interrupted ~landing:(Call call) n (fun () ->
            outer := Rig_remote.devices h "NONE")
      in
      let msg what = strf "from allocation %d, %s" n what in
      equal ~msg:(msg "NONE")
        (result (list string) string)
        (Error (at ^ ": no such hardware"))
        (Result.map names !outer);
      is_none ~msg:(msg "the host's loss") (Rig.lost h);
      let ds = mem h in
      if !first = [] then begin
        equal ~msg:(msg "MEM") (list string)
          [ "MEM:0@" ^ at; "MEM:1@" ^ at ]
          (names ds);
        first := ds
      end;
      (match !inner with
      | None -> ()
      | Some (Error why) -> failf "%s: %s" (msg "MEM in the callback") why
      | Some (Ok ds') ->
          equal
            ~msg:(msg "MEM in the callback")
            (list bool) [ true; true ]
            (List.map2 Rig.equal !first ds'));
      equal ~msg:(msg "MEM after") (list bool) [ true; true ]
        (List.map2 Rig.equal !first ds);
      reached);
  List.iter
    (fun d ->
      let s = "on " ^ Rig.name d in
      equal ~msg:"a copy there and back" string s (read (far_of_string d s)))
    !first

(* A first devices of MEM from a callback in the first devices of MEM, in a job
   of its own: both give the same devices, or the callback's raises
   Invalid_argument where it opens a device that the interrupted call is
   opening. The host is not lost, and the devices serve copies. *)
let first_in_first () =
  let refusals = ref 0 in
  walk (fun n ->
      with_job @@ fun j agents ->
      let h = List.hd (Rig_remote.hosts j) in
      let at = at (List.hd agents) (Rig.name h) in
      let outer = ref (Ok []) and inner = ref None in
      let call () =
        match Rig_remote.devices h "MEM" with
        | r -> inner := Some r
        | exception Invalid_argument _ -> incr refusals
      in
      let reached =
        Rig_support.interrupted ~landing:(Call call) n (fun () ->
            outer := Rig_remote.devices h "MEM")
      in
      let msg what = strf "from allocation %d, %s" n what in
      let ds =
        match !outer with
        | Ok ds -> ds
        | Error why -> failf "%s: %s" (msg "MEM") why
      in
      equal ~msg:(msg "MEM") (list string)
        [ "MEM:0@" ^ at; "MEM:1@" ^ at ]
        (names ds);
      (match !inner with
      | None -> ()
      | Some (Error why) -> failf "%s: %s" (msg "MEM in the callback") why
      | Some (Ok ds') ->
          equal
            ~msg:(msg "MEM in the callback")
            (list bool) [ true; true ]
            (List.map2 Rig.equal ds ds'));
      is_none ~msg:(msg "the host's loss") (Rig.lost h);
      equal ~msg:(msg "MEM after") (list bool) [ true; true ]
        (List.map2 Rig.equal ds (mem h));
      List.iter
        (fun d ->
          let s = "on " ^ Rig.name d in
          equal
            ~msg:(msg "a copy there and back")
            string s
            (read (far_of_string d s)))
        ds;
      reached);
  greater ~msg:"devices refused" int ~than:0 !refusals

(* A first devices of MEM from a callback in the first devices of MEM, which
   closes the last device it gets, in a job of its own: both give the same
   devices, the last closed, or the callback's raises Invalid_argument where it
   opens a device that the interrupted call is opening. The host is not lost,
   and the first device serves copies. *)
let close_in_first () =
  let refusals = ref 0 and closes = ref 0 in
  walk (fun n ->
      with_job @@ fun j agents ->
      let h = List.hd (Rig_remote.hosts j) in
      let at = at (List.hd agents) (Rig.name h) in
      let outer = ref (Ok []) and inner = ref None in
      let call () =
        match Rig_remote.devices h "MEM" with
        | Ok ds as r ->
            Rig.close (List.nth ds (List.length ds - 1));
            incr closes;
            inner := Some r
        | r -> inner := Some r
        | exception Invalid_argument _ -> incr refusals
      in
      let reached =
        Rig_support.interrupted ~landing:(Call call) n (fun () ->
            outer := Rig_remote.devices h "MEM")
      in
      let msg what = strf "from allocation %d, %s" n what in
      let ds =
        match !outer with
        | Ok ds -> ds
        | Error why -> failf "%s: %s" (msg "MEM") why
      in
      equal ~msg:(msg "MEM") (list string)
        [ "MEM:0@" ^ at; "MEM:1@" ^ at ]
        (names ds);
      (match !inner with
      | None -> ()
      | Some (Error why) -> failf "%s: %s" (msg "MEM in the callback") why
      | Some (Ok ds') ->
          equal
            ~msg:(msg "MEM in the callback")
            (list bool) [ true; true ]
            (List.map2 Rig.equal ds ds');
          equal ~msg:(msg "the losses")
            (list (option string))
            [ None; Some "closed" ] (List.map Rig.lost ds));
      is_none ~msg:(msg "the host's loss") (Rig.lost h);
      equal ~msg:(msg "MEM after") (list bool) [ true; true ]
        (List.map2 Rig.equal ds (mem h));
      let d = List.hd ds in
      let s = "on " ^ Rig.name d in
      equal
        ~msg:(msg "a copy there and back")
        string s
        (read (far_of_string d s));
      reached);
  greater ~msg:"devices refused" int ~than:0 !refusals;
  greater ~msg:"devices closed" int ~than:0 !closes

(* A first devices of MEM cut at each of its allocations by an exception
   there and at each one after it, rig's opens of the proxies included, in a
   job of its own: a devices after it gives the kind's devices, which serve
   copies, and the host is not lost. *)
let raise_in_devices () =
  walk (fun n ->
      with_job @@ fun j agents ->
      let h = List.hd (Rig_remote.hosts j) in
      let at = at (List.hd agents) (Rig.name h) in
      let reached =
        Rig_support.interrupted n (fun () ->
            ignore (Rig_remote.devices h "MEM" : _ result))
      in
      let msg what = strf "from allocation %d, %s" n what in
      let ds =
        match Rig_remote.devices h "MEM" with
        | Ok ds -> ds
        | Error why -> failf "%s: Error %s" (msg "MEM after it") why
        | exception Invalid_argument why ->
            failf "%s raised %S" (msg "MEM after it") why
      in
      equal ~msg:(msg "MEM after it") (list string)
        [ "MEM:0@" ^ at; "MEM:1@" ^ at ]
        (names ds);
      is_none ~msg:(msg "the host's loss") (Rig.lost h);
      List.iter
        (fun d ->
          let s = "on " ^ Rig.name d in
          equal ~msg:(msg "a copy there and back") string s
            (read (far_of_string d s)))
        ds;
      reached)

(* Connects *)

(* An address of the loopback whose listener closes the first connection it
   accepts: a connect to it answers [Error] once it has made the process's
   job, and gives the job back. A port where nothing listens refuses a
   connect at once only on Linux and macOS: Windows tries it again for about
   2 s, which over every allocation of a connect passed 1000 s. *)
let with_closing_address f =
  let fd = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  Unix.bind fd (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
  Unix.listen fd 2;
  let addr = Unix.getsockname fd in
  let port =
    match addr with
    | Unix.ADDR_INET (_, p) -> p
    | Unix.ADDR_UNIX _ -> fail "no port"
  in
  let closer =
    Thread.create (fun () -> Unix.close (fst (Unix.accept ~cloexec:true fd))) ()
  in
  (* A connection of the test's own ends the accept if [f] made none; one
     left in the backlog goes with the listener. *)
  let release () =
    let c = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
    Fun.protect
      ~finally:(fun () -> Unix.close c)
      (fun () -> Unix.connect c addr);
    Thread.join closer
  in
  Fun.protect ~finally:release (fun () -> f ("127.0.0.1", port))

(* A connect after a connect that raised: it answers [Error] for an address
   whose listener closes it, which it can only once the process's job of the
   connect that raised is not open. *)
let connects_after msg =
  with_closing_address @@ fun at ->
  match Rig_remote.connect ~key:(as_key key) [ at ] with
  | Ok j ->
      Rig_remote.close j;
      failf "%s: a connect to %s:%d answered Ok" msg (fst at) (snd at)
  | Error _ -> ()
  | exception Invalid_argument why ->
      failf "%s: a connect after it raised Invalid_argument %S" msg why

(* What a connect answers until it returns. *)
let not_returned : (Rig_remote.t, string) result = Error "not returned"

(* A connect cut at each of its allocations by an exception there and at each
   one after it, as an asynchronous exception's handlers meet it again: a
   connect after it is not refused. *)
let raise_in_connect () =
  walk (fun n ->
      let made = ref not_returned in
      let reached =
        with_agents ~n:2 @@ fun agents ->
        Rig_support.interrupted n (fun () ->
            made :=
              Rig_remote.connect ~key:(as_key key) (List.map address agents))
      in
      (match !made with Ok j -> Rig_remote.close j | Error _ -> ());
      connects_after (strf "from allocation %d" n);
      reached)

(* A connect from a callback inside a lock of rig's on its thread raises
   Invalid_argument where it opens the job's first host: a connect after it is
   not refused. *)
let connect_inside_lock () =
  with_agents @@ fun agents ->
  (match
     Rig_support.inside_lock (fun () ->
         Rig_remote.connect ~key:(as_key key) (List.map address agents))
   with
  | Ok j ->
      Rig_remote.close j;
      fail "a connect inside a lock answered Ok"
  | Error why -> failf "a connect inside a lock answered Error %S" why
  | exception Invalid_argument why ->
      starts_with ~msg:"the refusal" ~affix:"Rig.open_: called from" why);
  connects_after "after a connect inside a lock"

(* Rails *)

let host_record h =
  match Rig.capability h Rig_remote_abi.key with
  | Some (Rig_remote_abi.Host r) -> r
  | _ -> fail "a host's record is Host"

let t5 = { Rig_remote_abi.src = 0; dst = 0; length = 5 }

(* [rail r peer] is a rail of [r]'s machine with [peer]'s, or this process's
   for [None]. *)
let rail (r : Rig_remote_abi.host) peer =
  r.rail peer ~send:[| t5 |] ~receive:[| t5 |]

(* The records of [j]'s two hosts. *)
let two_hosts j =
  match List.map host_record (Rig_remote.hosts j) with
  | [ r1; r2 ] -> (r1, r2)
  | _ -> fail "two hosts"

(* What a rail answers until it returns. *)
let no_rail : (Rig_remote_abi.rail, string) result = Error "not returned"

(* A rail with this process and one with another machine, with a Ctrl-C at
   each of their allocations: a function of a capability blocks signals, so
   the Ctrl-C lands before the rail takes an id, or once it is made, never
   between its two machines' halves. Windows blocks nothing, so there the
   Ctrl-C ends the rail as an exception at that allocation would, and a rail
   after it is made. *)
let break_in_rail () =
  with_job ~n:2 @@ fun j _ ->
  let r1, r2 = two_hosts j in
  let send = [| t5 |] in
  List.iter
    (fun (what, peer) ->
      walk (fun n ->
          let made = ref no_rail in
          let before = Rig_remote_proxy.Link.fresh () in
          let reached =
            Rig_support.interrupted ~landing:Signal n (fun () ->
                made := r1.rail peer ~send ~receive:send)
          in
          let taken = Rig_remote_proxy.Link.fresh () - before - 1 in
          let msg = strf "%s, a Ctrl-C at allocation %d" what n in
          (match !made with
          | Ok _ -> ()
          | Error why when !made != no_rail -> failf "%s: Error %s" msg why
          | Error _ when Sys.win32 ->
              is_ok ~msg:(msg ^ ", a rail after it") (rail r1 peer)
          | Error _ -> equal ~msg:(msg ^ ", ids taken") int 0 taken);
          reached))
    [ ("a rail with this process", None); ("a rail between machines", Some r2) ];
  is_none ~msg:"the job's failure" (Rig_remote.failed j)

(* A rail between machines cut at each of its allocations by an exception
   there and at each one after it: the job goes on, and a rail after it is
   made. *)
let raise_in_rail () =
  with_job ~n:2 @@ fun j _ ->
  let r1, r2 = two_hosts j in
  walk (fun n ->
      let reached =
        Rig_support.interrupted n (fun () -> ignore (rail r1 (Some r2)))
      in
      let msg what = strf "from allocation %d, %s" n what in
      is_none ~msg:(msg "the job's failure") (Rig_remote.failed j);
      is_ok ~msg:(msg "a rail after it") (rail r1 (Some r2));
      reached);
  is_none ~msg:"the job's failure" (Rig_remote.failed j)

(* Agents *)

(* What a listen answers until it returns. *)
let no_agent : (Rig_remote.agent, string) result = Error "not returned"

(* A listen cut at each of its allocations by an exception there and at each
   one after it: a listen that raised holds no descriptor. *)
let raise_in_listen () =
  let held () =
    match Rig_support.descriptors () with
    | Some n -> n
    | None -> skip ~reason:"the system does not count descriptors" ()
  in
  walk (fun n ->
      let before = held () in
      let made = ref no_agent in
      let reached =
        Rig_support.interrupted n (fun () ->
            made := Rig_remote.listen ~key:(as_key key) "127.0.0.1" 0)
      in
      (match !made with
      | Ok _ -> ()
      | Error why when !made != no_agent -> failf "listen: %s" why
      | Error _ ->
          equal
            ~msg:(strf "descriptors after a listen cut from allocation %d" n)
            int before (held ()));
      reached)

(* Serving, in a forked child *)

(* [in_child what f] is [f ()], run in a forked child, so that what one
   attempt leaves, a failed job of the process among it, never changes the
   next one. It fails with the child's failure. *)
let in_child what f =
  Rig_support.needs_fork ();
  let r, w = Unix.pipe ~cloexec:true () in
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      let answer =
        match f () with v -> Ok v | exception e -> Error (Printexc.to_string e)
      in
      let oc = Unix.out_channel_of_descr w in
      Marshal.to_channel oc (answer : ('a, string) result) [];
      close_out oc;
      Unix._exit 0
  | pid -> (
      Unix.close w;
      let ic = Unix.in_channel_of_descr r in
      let answer = In_channel.input_all ic in
      close_in ic;
      match (answer, snd (Unix.waitpid [] pid)) with
      | "", (WEXITED n | WSIGNALED n | WSTOPPED n) ->
          failf "the child ended (%d) without answering%s" n what
      | answer, _ -> (
          match (Marshal.from_string answer 0 : ('a, string) result) with
          | Ok v -> v
          | Error e -> failf "%s%s" e what))

(* The child's start: a thread's [Exit], a landing's, ends it without a word,
   and a write to a connection the agent closed raises [EPIPE]. *)
let child_start () =
  Thread.set_uncaught_exception_handler (function
    | Exit -> ()
    | e -> prerr_endline ("a thread raised " ^ Printexc.to_string e));
  Sys.set_signal Sys.sigpipe Sys.Signal_ignore

(* Whether an allocation's call stack holds the function named [name], or a
   function inside it. *)
let in_fn name =
  let named s =
    match Printexc.Slot.name s with
    | Some n -> String.equal n name || String.starts_with ~prefix:(name ^ ".") n
    | None -> false
  in
  fun (a : Gc.Memprof.allocation) ->
    match Printexc.backtrace_slots a.callstack with
    | Some slots -> Array.exists named slots
    | None -> false

(* Whether an allocation's call stack holds the agent's function [fn]. *)
let in_agent fn = in_fn ("Rig_remote__Agent." ^ fn)

(* [cut_in fn n f] runs [f ()] while every allocation inside the agent's
   [fn], on any thread, is counted, and the [n]-th raises [Exit], one
   asynchronous exception: whether [f] reached the [n]-th. The threads of
   [fn] run one at a time, so the count is the same in each run. Only one
   allocation raises: a handler that must take the agent's lock meets a
   poll there when the lock is contended, where a second exception would cut
   it too. *)
let cut_in fn n f =
  let inside = in_agent fn in
  let left = ref n and armed = ref false in
  let alloc a =
    if !armed && inside a then begin
      decr left;
      if !left = 0 then raise Exit
    end;
    None
  in
  let tracker =
    { Gc.Memprof.null_tracker with alloc_minor = alloc; alloc_major = alloc }
  in
  let profile = Gc.Memprof.start ~sampling_rate:1. tracker in
  armed := true;
  Fun.protect
    ~finally:(fun () ->
      armed := false;
      Gc.Memprof.stop ();
      Gc.Memprof.discard profile)
    f;
  !left <= 0

let agent_here () =
  match Rig_remote.listen ~key:(as_key key) "127.0.0.1" 0 with
  | Ok a -> a
  | Error why -> failf "listen: %s" why

(* Serves [a] on a thread of its own: [served ()] is what serve answered, or
   the exception it raised, once it has returned. *)
let serving a =
  let r = ref None in
  let t =
    Thread.create
      (fun () ->
        r :=
          Some
            (match Rig_remote.serve a [] with
            | v -> Ok v
            | exception e -> Error e))
      ()
  in
  fun () ->
    until ~what:"serve's return" (fun () -> Option.is_some !r);
    Thread.join t;
    Option.get !r

(* A controller's connection to the agent at [port]: [Some fd] once both
   proved the key, [None] if the agent ended it first or listens no more.
   Each read gives up after [patience]. *)
let dial port =
  let module Wire = Rig_remote_proxy.Wire in
  let fd = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Rig_support.recv_timeout fd patience;
  match Unix.connect fd (Unix.ADDR_INET (Unix.inet_addr_loopback, port)) with
  | exception Unix.Unix_error ((Unix.ECONNREFUSED | Unix.ECONNRESET), _, _) ->
      Unix.close fd;
      None
  | () -> (
      match Wire.dial fd ~key ~self:Wire.Controller ~peer:(Wire.Agent 1) with
      | Ok () -> Some fd
      | Error _ ->
          Unix.close fd;
          None)

(* Whether the agent at [port] answers the join of a job of it alone over
   [fd], so that it serves the job. *)
let joins fd port =
  let at = strf "127.0.0.1:%d" port in
  send fd (frame k_request (u8 1 ^ u32 1 ^ str at ^ str "127.0.0.1" ^ u32 port));
  match next_frame fd with
  | Some (k, p) -> k = k_answer && p.[0] = '\000'
  | None -> false

(* Closes the job of [fd]: sends its close, ends the stream once the agent's
   close, abort or end came, and reads [fd] to its end. *)
let ends fd =
  send fd (frame k_close "");
  let rec closed () =
    match next_frame fd with
    | Some (k, _) when k = k_close || k = k_abort -> ()
    | Some _ -> closed ()
    | None -> ()
  in
  closed ();
  (try Unix.shutdown fd Unix.SHUTDOWN_SEND with Unix.Unix_error _ -> ());
  let rec drain () = match next_frame fd with Some _ -> drain () | None -> () in
  drain ();
  Unix.close fd

(* The most a handshake takes: a connection that proves nothing within it is
   dropped ({!Rig_remote.listen}), and a dial gives up. *)
let handshake_s = 10.

(* Whether anything listens at [port] of the loopback. *)
let listens port =
  let fd = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  match Unix.connect fd (Unix.ADDR_INET (Unix.inet_addr_loopback, port)) with
  | () -> true
  | exception Unix.Unix_error (Unix.ECONNREFUSED, _, _) -> false

(* A controller's handshake cut by an exception at each allocation of its
   thread: the agent serves the controller, or gives back what the handshake
   took and serves a second one. *)
let cut_handshake n =
  in_child (strf ", a handshake cut from allocation %d" n) @@ fun () ->
  child_start ();
  let a = agent_here () in
  let port = Rig_remote.port a in
  let served = serving a in
  let first = ref None in
  let reached =
    cut_in "handshake" n (fun () ->
        first := Option.map (fun fd -> (fd, joins fd port)) (dial port))
  in
  (* A job given back ends once its connection has: a controller that comes
     before is refused, and dials again. *)
  let rec second t0 =
    match dial port with
    | Some fd when joins fd port -> fd
    | fd when Unix.gettimeofday () -. t0 < patience ->
        Option.iter Unix.close fd;
        Thread.delay 0.01;
        second t0
    | Some _ -> fail "a second controller is refused"
    | None -> fail "a second controller's handshake failed"
  in
  let fd =
    match !first with
    | Some (fd, true) -> fd
    | first ->
        Option.iter (fun (fd, _) -> Unix.close fd) first;
        second (Unix.gettimeofday ())
  in
  ends fd;
  (match served () with
  | Ok (Ok ()) -> ()
  | Ok (Error why) -> failf "serve answered Error %s" why
  | Error e -> failf "serve raised %s" (Printexc.to_string e));
  reached

let handshake_cut () =
  let rec from n = if cut_handshake n then from (n + 1) else n - 1 in
  greater ~msg:"allocations cut" int ~than:0 (from 1)

(* The acceptor cut by an exception at each of its allocations, while a
   controller connects: serve returns, and the agent listens no more. *)
let cut_acceptor n =
  in_child (strf ", the acceptor cut from allocation %d" n) @@ fun () ->
  child_start ();
  let a = agent_here () in
  let port = Rig_remote.port a in
  let result = ref None in
  let reached =
    cut_in "accept_loop" n (fun () ->
        let served = serving a in
        (match dial port with
        | Some fd when joins fd port -> ends fd
        | Some fd -> Unix.close fd
        | None -> ());
        result := Some (served ()))
  in
  (match !result with
  | Some (Error e) -> failf "serve raised %s" (Printexc.to_string e)
  | Some (Ok _) | None -> ());
  equal ~msg:"the port listens" bool false (listens port);
  reached

let acceptor_cut () =
  let rec from n = if cut_acceptor n then from (n + 1) else n - 1 in
  greater ~msg:"allocations cut" int ~than:0 (from 1)

(* A serve whose own steps, its commands' aside, take [landing] at each of
   their allocations, while a controller joins and closes its job: serve
   ends, the controller's connection ends, and the agent listens no more
   and holds no descriptor. A cut before serve claimed the agent leaves it
   to a later serve, which serves the controller. *)
let cut_serve what landing n =
  in_child (strf ", %s from allocation %d" what n) @@ fun () ->
  child_start ();
  let before =
    match Rig_support.descriptors () with
    | Some n -> n
    | None -> skip ~reason:"the system does not count descriptors" ()
  in
  let a = agent_here () in
  let port = Rig_remote.port a in
  let finished = ref false in
  let controller () =
    ignore (Thread.sigmask Unix.SIG_BLOCK [ Sys.sigint ]);
    (match dial port with
    | Some fd when joins fd port -> ends fd
    | Some fd -> Unix.close fd
    | None -> ());
    finished := true
  in
  let t = Thread.create controller () in
  let not_apply = Fun.negate (in_agent "apply") in
  let reached =
    Rig_support.interrupted ~landing ~from:not_apply n (fun () ->
        ignore (Rig_remote.serve a []))
  in
  (match Rig_remote.serve a [] with
  | Ok () | Error _ -> ()
  | exception Invalid_argument _ -> ());
  (* A connection that comes as the agent stops listening may get no answer:
     its dial gives up after the 10 s a handshake may take. *)
  let t0 = Unix.gettimeofday () in
  while (not !finished) && Unix.gettimeofday () -. t0 < handshake_s +. patience do
    Thread.delay 0.01
  done;
  equal ~msg:"the controller ended" bool true !finished;
  Thread.join t;
  equal ~msg:"the port listens" bool false (listens port);
  until ~what:"the agent's descriptors closed" (fun () ->
      Rig_support.descriptors () = Some before);
  reached

let serve_cut () =
  let once = Rig_support.Call (fun () -> raise Exit) in
  let rec from n = if cut_serve "an exception" once n then from (n + 1) in
  from 1

let serve_break () =
  let rec from n = if cut_serve "a Ctrl-C" Signal n then from (n + 1) in
  from 1

(* Hosts of a start that raised *)

(* An opener of no device, named [name]. *)
let no_device name : Rig_support.Polled.t Rig_edge.opener =
  {
    name;
    driver = (module Rig_support.Polled);
    make = (fun ~word:_ -> Error "no device");
  }

(* Lowers the process limit to 0, in a forked child: no thread or process
   the child starts after it is made, where the limit binds. *)
let refuse () =
  match Rig_support.set_process_limit 0 with
  | 0 -> ()
  | e -> failf "setrlimit: errno %d" e

(* Whether the limit [refuse] sets refuses this process a thread. It binds
   only where the system applies it to threads and to this process: not on
   macOS, nor for a process privileged to pass it. *)
let refuses_threads () =
  refuse ();
  let refused =
    match Thread.create ignore () with
    | t ->
        Thread.join t;
        false
    | exception Sys_error _ -> true
  in
  ignore (Rig_support.set_process_limit (-1) : int);
  refused

(* A connect to [a] with the process limit lowered at its job's watcher's
   first allocation: connect raises Sys_error, and an open of another host
   of the machine answers Error, where an open host would make it raise
   Invalid_argument. *)
let watcher_cut a =
  let raised = ref "" in
  let reached =
    Rig_support.interrupted ~from:(in_fn "Rig_remote.watch")
      ~landing:(Call refuse) 1 (fun () ->
        match Rig_remote.connect ~key:(as_key key) [ address a ] with
        | Ok _ -> raised := "returned"
        | Error why -> raised := "answered Error " ^ why
        | exception Sys_error _ -> raised := "Sys_error")
  in
  ignore (Rig_support.set_process_limit (-1) : int);
  equal ~msg:"the watcher's first allocation reached" bool true reached;
  equal ~msg:"connect" string "Sys_error" !raised;
  match Rig.open_host ~machine:(machine a) (no_device "OTHER") with
  | Ok _ -> fail "a host opened"
  | Error _ -> ()
  | exception Invalid_argument why -> failf "the host is still open: %s" why

(* A connect whose job's watcher the system refuses a thread closes the host
   it opened ({!watcher_cut}), in a forked child, where the process limit
   refuses threads. *)
let watcher_refused () =
  with_agents @@ fun agents ->
  let a = List.hd agents in
  let refused =
    in_child "" @@ fun () ->
    let refused = refuses_threads () in
    if refused then watcher_cut a;
    refused
  in
  if not refused then
    skip ~reason:"the process limit does not refuse this process a thread" ()

(* Launched jobs *)

(* Runs support/launched.exe in [mode] with a landing at each allocation of
   the close of its launched job, whose hosts are closed, one run for each
   from the first on, until [check msg code report words agent] is [false]:
   [msg] names the run, [code] is its exit code, [words] what it prints after
   its hosts' names, and [agent] its agent's exit code and lines. *)
let close_walk mode check =
  walk (fun n ->
      with_hex_agents @@ fun agents ->
      let args = [ string_of_int n ] in
      let code, report, said = ended (launch ~args (launch_vars agents) mode) in
      let msg what = strf "from allocation %d, %s" n what in
      let words =
        match said with
        | "connected" :: "CPU@m1" :: words -> words
        | _ -> failf "%s: [%s]" (msg "its words") (String.concat "; " said)
      in
      check msg code report words (finish (List.hd agents)))

(* The report and the agent's end of a job that closed. *)
let closed msg code report agent =
  equal
    ~msg:(msg "its exit and report")
    (pair int (list string))
    (0, [ "started"; "closed" ])
    (code, report);
  equal ~msg:(msg "the agent's end") exit_w (0, [ "closed" ]) agent

(* "reached" or "past", the last of a run's words, and whether it reached the
   allocation. *)
let reached msg words =
  match words with
  | [ "reached" ] -> true
  | [ "past" ] -> false
  | _ -> failf "%s: [%s]" (msg "its words") (String.concat "; " words)

(* A close from the callback: both return, and the job closed. *)
let close_in_close () =
  close_walk "reenter" (fun msg code report words agent ->
      closed msg code report agent;
      reached msg words)

(* An exception at the allocation and each one after it, then an exit, which
   closes the job again: the job closed. *)
let raise_in_close () =
  close_walk "raise-in-close" (fun msg code report words agent ->
      closed msg code report agent;
      reached msg words)

(* An exit from the callback, whose close at exit runs inside the close: the
   program exits 0, and the job closed, or failed with the reason rig
   refused that close for, which the agent records as the controller's. Both
   are reached. *)
let exit_in_close () =
  let failures = ref 0 and closes = ref 0 in
  close_walk "exit-in-close" (fun msg code report words agent ->
      (match report with
      | [ "started"; "closed" ] ->
          incr closes;
          closed msg code report agent
      | [ "started"; f ] when String.starts_with ~prefix:"failed " f ->
          incr failures;
          let why = String.sub f 7 (String.length f - 7) in
          equal
            ~msg:(msg "its exit and the agent's end")
            (pair int exit_w)
            (0, (2, [ "failed: controller aborted: " ^ why ]))
            (code, agent)
      | _ ->
          failf "%s: [%s]" (msg "its report") (String.concat "; " report));
      match words with
      | [] -> true
      | [ "past" ] -> false
      | _ -> failf "%s: [%s]" (msg "its words") (String.concat "; " words));
  greater ~msg:"closes" int ~than:0 !closes;
  greater ~msg:"failures" int ~than:0 !failures

let () =
  Watchdog.start ();
  exit
    (run "rig_remote interrupt"
       [
         group ~timeout:60. "a call from a callback at any allocation"
           [
             test "a devices in a devices answers" devices_in_devices;
             test "a first devices in a first devices of its kind answers"
               first_in_first;
             test
               "a first devices in a first devices that closes a device answers"
               close_in_first;
             test "a close in a close returns" close_in_close;
             test "an exit in a close reports the job's end" exit_in_close;
             test "an exception in a close reports the close at the exit"
               raise_in_close;
           ];
         group ~timeout:60. "a cut at any allocation"
           [
             test "a devices after a devices cut by an exception answers"
               raise_in_devices;
             test "a connect whose watcher gets no thread closes its host"
               watcher_refused;
             test "a connect after a connect cut by an exception is not refused"
               raise_in_connect;
             test "a connect after a connect inside a lock is not refused"
               connect_inside_lock;
             test "a Ctrl-C in a rail lands before or after it" break_in_rail;
             test "a rail cut by an exception leaves the job going"
               raise_in_rail;
             test "a listen cut by an exception holds no descriptor"
               raise_in_listen;
             test "a handshake cut by an exception leaves the agent serving"
               handshake_cut;
             test "an acceptor cut by an exception ends serve" acceptor_cut;
             test "a serve cut by an exception stops listening" serve_cut;
             test "a serve cut by a Ctrl-C stops listening" serve_break;
           ];
       ])
