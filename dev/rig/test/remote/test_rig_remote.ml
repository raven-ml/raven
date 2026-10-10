(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jobs whose agents are real processes on the loopback (Remote_job). A failed
   job fails the process for good, so the failures run in executables of their
   own; every job here closes. *)

open Windtrap
open Remote_job
module B = Rig.Buffer
module Sub = Rig.Submission

let lost_w =
  Testable.structural ~pp:(fun ppf -> function
    | None -> Format.pp_print_string ppf "not lost"
    | Some why -> Format.fprintf ppf "lost: %S" why)

(* Connecting *)

let two_agents () =
  with_job ~n:2 @@ fun j agents ->
  List.iter2
    (fun h a ->
      equal string ("CPU@" ^ machine a) (Rig.name h);
      let d = List.hd (mem h) in
      let s = "to " ^ machine a in
      equal string s (read (far_of_string d s)))
    (Rig_remote.hosts j) agents;
  Rig_remote.close j;
  equal (list exit_w)
    [ (0, [ "closed" ]); (0, [ "closed" ]) ]
    (List.map finish agents)

let wrong_key () =
  with_agents @@ fun agents ->
  let a = List.hd agents in
  (match
     Rig_remote.connect ~key:(as_key (String.make 32 'x')) [ address a ]
   with
  | Ok j ->
      Rig_remote.close j;
      fail "a job with another key"
  | Error why -> starts_with ~affix:(machine a ^ ": ") why);
  let j = connect agents in
  Rig_remote.close j;
  equal exit_w (0, [ "closed" ]) (finish a)

(* A port nothing listens at: one the system gave and took back. *)
let unreachable () =
  let s = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.bind s (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
  let port =
    match Unix.getsockname s with
    | Unix.ADDR_INET (_, p) -> p
    | Unix.ADDR_UNIX _ -> assert false
  in
  Unix.close s;
  match Rig_remote.connect ~key:(as_key key) [ ("127.0.0.1", port) ] with
  | Ok j ->
      Rig_remote.close j;
      fail "a job with no agent"
  | Error why -> starts_with ~affix:(Printf.sprintf "127.0.0.1:%d: " port) why

(* A second controller proves the key to an agent in a job. *)
let second_controller () =
  with_job @@ fun _ agents ->
  let a = List.hd agents in
  let fd = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close fd)
    (fun () ->
      Unix.connect fd (Unix.ADDR_INET (Unix.inet_addr_loopback, a.port));
      let module Wire = Rig_remote_proxy.Wire in
      let why =
        require_error
          (Wire.dial fd ~key ~self:Wire.Controller ~peer:(Wire.Agent 1))
      in
      contains ~sub:"another job" why)

(* Two controllers prove the key to an idle agent at once: the agent admits one
   and tells the other it serves another job. *)
let two_controllers () =
  with_agents @@ fun agents ->
  let a = List.hd agents in
  let dial () =
    let fd = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
    Unix.connect fd (Unix.ADDR_INET (Unix.inet_addr_loopback, a.port));
    let module Wire = Rig_remote_proxy.Wire in
    (fd, Wire.dial fd ~key ~self:Wire.Controller ~peer:(Wire.Agent 1))
  in
  let second = ref None in
  let other = Thread.create (fun () -> second := Some (dial ())) () in
  let first = dial () in
  Thread.join other;
  let second = Option.get !second in
  let admitted, refused =
    match (first, second) with
    | (fd, Ok ()), (fd', Error why) | (fd', Error why), (fd, Ok ()) ->
        Unix.close fd';
        (fd, why)
    | (fd, r), (fd', r') ->
        Unix.close fd;
        Unix.close fd';
        failf "the dials answered %s and %s"
          (match r with Ok () -> "Ok" | Error e -> e)
          (match r' with Ok () -> "Ok" | Error e -> e)
  in
  contains ~sub:"another job" refused;
  Fun.protect ~finally:(fun () -> Unix.close admitted) @@ fun () ->
  Unix.setsockopt_float admitted Unix.SO_RCVTIMEO patience;
  is_ok ~msg:"the join" ~pp:Format.pp_print_string (join_alone admitted a);
  send admitted (frame k_close "");
  equal ~msg:"the agent's close" (option int) (Some k_close)
    (Option.map fst (next_frame admitted));
  equal exit_w (0, [ "closed" ]) (finish a)

let connect_misuse () =
  with_job @@ fun _ agents ->
  let a = List.hd agents in
  let raises msg f = raises_match ~msg Exn.invalid_arg f in
  let key = as_key key in
  raises "a job is open" (fun () -> Rig_remote.connect ~key [ address a ]);
  raises "no agents" (fun () -> Rig_remote.connect ~key []);
  raises "an address twice" (fun () ->
      Rig_remote.connect ~key [ address a; address a ])

(* An agent that serves two jobs at one address: the second job's machine is
   another machine. *)
let second_connection () =
  with_agents ~mode:"again" @@ fun agents ->
  let a = List.hd agents in
  let first = connect agents in
  let h1 = List.hd (Rig_remote.hosts first) in
  Rig_remote.close first;
  (* The agent prints its port again once it listens for the second job. *)
  equal ~msg:"the agent's first job" string "closed" (input_line a.out);
  equal ~msg:"its second listen" string (string_of_int a.port)
    (input_line a.out);
  let second = connect agents in
  let h2 = List.hd (Rig_remote.hosts second) in
  Rig_remote.close second;
  equal string ("CPU@" ^ machine a) (Rig.name h1);
  equal string ("CPU@" ^ machine a ^ "#2") (Rig.name h2);
  equal exit_w (0, [ "closed" ]) (finish a)

(* An agent's serve returned: it listens no more, though its process goes on. A
   controller finds no listener there. *)
let listens_no_more () =
  with_agents ~mode:"linger" @@ fun agents ->
  let a = List.hd agents in
  Rig_remote.close (connect agents);
  match Rig_remote.connect ~key:(as_key key) [ address a ] with
  | Ok j ->
      Rig_remote.close j;
      fail "a second job at an agent of one"
  | Error why -> not_contains ~sub:"another job" why

(* A process the agent started holds none of its sockets: once close returned,
   nothing listens at the agent's port. *)
(* CR: Wait for a post-exec ready signal and hold the helper on a pipe
   through this assertion. Its sleep can expire during scheduling delays,
   so this test passes even if the helper inherited the listener. Release
   the gate in protected cleanup before finish; have the agent reap the
   helper and check its exit. *)
let helper_holds_nothing () =
  with_agents ~mode:"helper" @@ fun agents ->
  let a = List.hd agents in
  Rig_remote.close (connect agents);
  let s = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  let outcome =
    match Unix.connect s (Unix.ADDR_INET (Unix.inet_addr_loopback, a.port)) with
    | () -> "accepted"
    | exception Unix.Unix_error (e, _, _) -> Unix.error_message e
  in
  Unix.close s;
  equal ~msg:"a connection to the agent's port" string
    (Unix.error_message Unix.ECONNREFUSED)
    outcome;
  equal exit_w (0, [ "closed" ]) (finish a)

let listen_unresolved () =
  is_error (Rig_remote.listen ~key:(as_key key) "no-such-host.invalid" 0)

let connecting =
  group "connect"
    [
      test "two agents serve one job, each its machine, and end with it"
        two_agents;
      test "an agent refuses a controller of another key, then serves the key's"
        wrong_key;
      test "an address nothing listens at is an Error naming it" unreachable;
      test "an agent in a job tells a second controller it serves another"
        second_controller;
      test "an agent admits one of two controllers that come at once"
        two_controllers;
      test "connect raises on no agent, an address twice, an open job"
        connect_misuse;
      test "listen answers Error for a host that does not resolve"
        listen_unresolved;
      test "a second job at one address is another machine, named #2"
        second_connection;
      test "once close returned, no agent listens" listens_no_more;
      test "a process the agent started keeps no listener open"
        helper_holds_nothing;
    ]

(* Hosts and devices *)

let hosts () =
  with_job @@ fun j agents ->
  let h = List.hd (Rig_remote.hosts j) and a = List.hd agents in
  equal string (Rig.arch Rig.host) (Rig.arch h);
  equal bool true (Rig.equal h (Rig.host_of h));
  equal bool false (Rig.shares_host_memory h);
  (match Rig.capability h Rig_remote_abi.key with
  | Some (Rig_remote_abi.Host r) -> equal string (machine a) r.machine
  | _ -> fail "a host's record is Host");
  is_error (Rig.Image.load h "code")

let local_pair () =
  let open_ n = Result.get_ok (Rig.memory_device n) in
  (open_ "LOCAL-MEM:0", open_ "LOCAL-MEM:1")

let devices () =
  with_job @@ fun j agents ->
  let h = List.hd (Rig_remote.hosts j) and a = List.hd agents in
  let ds = mem h in
  equal (list string)
    [ "MEM:0@" ^ machine a; "MEM:1@" ^ machine a ]
    (List.map Rig.name ds);
  equal bool true (List.for_all2 Rig.equal ds (mem h));
  let l0, l1 = local_pair () in
  let d0 = List.nth ds 0 and d1 = List.nth ds 1 in
  equal ~msg:"arch" string (Rig.arch l0) (Rig.arch d0);
  equal ~msg:"budget" int (Rig.budget l0) (Rig.budget d0);
  equal ~msg:"reaches" bool (Rig.reaches l0 l1) (Rig.reaches d0 d1);
  List.iter (fun d -> equal bool true (Rig.equal h (Rig.host_of d))) ds;
  let id d =
    match Rig.capability d Rig_remote_abi.key with
    | Some (Rig_remote_abi.Device { id }) -> id
    | _ -> fail "a device's record is Device"
  in
  not_equal ~msg:"the two devices' ids" int (id d0) (id d1);
  let why = require_error (Rig.Image.load d0 "code") in
  contains ~msg:"says it loads no code" ~sub:"loads no code" why

let device_errors () =
  with_job @@ fun j agents ->
  let h = List.hd (Rig_remote.hosts j) and a = List.hd agents in
  let none = require_error (Rig_remote.devices h "GPU") in
  starts_with ~msg:"a kind the agent serves not" ~affix:(machine a) none;
  let failed = require_error (Rig_remote.devices h "NONE") in
  starts_with ~msg:"an opener's failure" ~affix:(machine a) failed;
  contains ~msg:"the opener's reason" ~sub:"no such hardware" failed;
  raises_match ~msg:"this process's host" Exn.invalid_arg (fun () ->
      Rig_remote.devices Rig.host "MEM");
  raises_match ~msg:"a device" Exn.invalid_arg (fun () ->
      Rig_remote.devices (List.hd (mem h)) "MEM")

let machines =
  group "machine"
    [
      test "a machine's host is the agent's, of this machine's arch" hosts;
      test "devices are the agent's, named after it, the same at each call"
        devices;
      test "devices answers Error naming the machine, raises for no host"
        device_errors;
    ]

(* Copies *)

let sizes = [ 0; 1; 4095; 4096; 4097; 1 lsl 20; 8 lsl 20 ]

let round_trips () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  List.iter
    (fun d ->
      List.iter
        (fun n ->
          let s = String.init n (fun i -> Char.chr (i * 7 land 0xff)) in
          equal
            ~msg:(Printf.sprintf "%d bytes on %s" n (Rig.name d))
            string s
            (read (far_of_string d s)))
        sizes)
    (h :: mem h)

let within_machine () =
  with_job @@ fun j _ ->
  match mem (List.hd (Rig_remote.hosts j)) with
  | [ d0; d1 ] ->
      let a = far_of_string d0 "between devices" in
      let b = B.create d1 (B.length a) in
      B.copy ~src:a ~dst:b;
      equal string "between devices" (read b)
  | _ -> fail "two MEM devices"

let across_machines () =
  with_job ~n:2 @@ fun j _ ->
  match Rig_remote.hosts j with
  | [ h1; h2 ] ->
      let a = far_of_string (List.hd (mem h1)) "here" in
      let b = B.create (List.hd (mem h2)) 4 in
      raises_match Exn.invalid_arg (fun () -> B.copy ~src:a ~dst:b)
  | _ -> fail "two hosts"

(* A law: copies submitted without waiting, between this process's memory and
   two devices of a machine, leave what they leave run one after another. *)

type place = Here of int | There of int * int (* device, buffer *)
type op = { src : place; dst : place; at_src : int; at_dst : int; len : int }

let buffer_size = 64

let pp_place ppf = function
  | Here i -> Format.fprintf ppf "here%d" i
  | There (d, i) -> Format.fprintf ppf "dev%d.%d" d i

let pp_op ppf o =
  Format.fprintf ppf "%a[%d] -> %a[%d] x%d" pp_place o.src o.at_src pp_place
    o.dst o.at_dst o.len

let op_g =
  let open Gen in
  let place =
    of_list ~pp:pp_place
      [ Here 0; Here 1; There (0, 0); There (0, 1); There (1, 0) ]
  in
  let* src, dst =
    such_that
      (fun (s, d) ->
        s <> d
        &&
        match (s, d) with
        | Here _, Here _ -> false
        | There (a, _), There (b, _) -> a = b
        | _ -> true)
      (pair place place)
  in
  let* len = int_range 1 buffer_size in
  let+ at_src = int_range 0 (buffer_size - len)
  and+ at_dst = int_range 0 (buffer_size - len) in
  { src; dst; at_src; at_dst; len }

let device_of = function There (d, _) -> Some d | Here _ -> None

(* CR: Generate and print init with ops. Random draws fresh bytes during
   shrinking and confirmation, so the same ops can test different inputs.
   Use five fixed-size strings of buffer_size bytes in the generator;
   keep their lengths fixed while shrinking their contents. *)
let copies_law ops =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  let devices = [| h; List.hd (mem h) |] in
  let init =
    Array.init 5 (fun _ ->
        String.init buffer_size (fun _ -> Char.chr (Random.int 256)))
  in
  let model = Array.map Bytes.of_string init in
  let index = function Here i -> i | There (0, i) -> 2 + i | There _ -> 4 in
  let buffers =
    Array.init 5 (fun i ->
        if i < 2 then B.of_string init.(i)
        else far_of_string devices.(if i < 4 then 0 else 1) init.(i))
  in
  let last = Array.make 2 0 and run = Sub.Run.make () in
  List.iteri
    (fun k o ->
      let d =
        match (device_of o.src, device_of o.dst) with
        | Some d, _ | None, Some d -> d
        | None, None -> assert false
      in
      let after_into same =
        match o.src with
        | Here _ ->
            List.exists
              (fun (i, o') ->
                i < k && o'.dst = o.src && device_of o'.src = Some d = same)
              (List.mapi (fun i o -> (i, o)) ops)
        | There _ -> false
      in
      cover "a copy from here after a copy into it, on its device"
        (after_into true);
      cover "a copy from here after a copy into it, on another device"
        (after_into false);
      let view p at = B.view buffers.(index p) ~first:at ~length:o.len in
      let s =
        Sub.make devices.(d)
          [|
            {
              Sub.queue = "COPY:0";
              after = [||];
              work =
                Sub.Copy
                  { src = view o.src o.at_src; dst = view o.dst o.at_dst };
            };
          |]
      in
      last.(d) <- Rig.Point.value (Rig.submit s ~run ~buffers:[||] ~waits:[||]);
      Bytes.blit model.(index o.src) o.at_src model.(index o.dst) o.at_dst o.len)
    ops;
  Array.iteri (fun d v -> if v > 0 then Rig.wait devices.(d) v) last;
  Array.iteri
    (fun i b ->
      equal
        ~msg:(Printf.sprintf "buffer %d" i)
        string
        (Bytes.to_string model.(i))
        (if i < 2 then read_host b else read b))
    buffers

let copies =
  group "copy"
    [
      test "bytes go to the host and a device of a machine and come back"
        round_trips;
      test "a copy between two devices of one machine" within_machine;
      test "a copy between two machines raises" across_machines;
      prop ~count:20
        "copies submitted at once leave what they leave one after another"
        (Gen.with_pp
           (Format.pp_print_list
              ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
              pp_op)
           (Gen.list ~size:(Gen.int_range 1 12) op_g))
        copies_law;
    ]

(* Close *)

(* Each agent closes the devices it opened before it returns from serve: one
   left open would print its name after "closed". *)
let close_order () =
  with_agents ~n:2 @@ fun agents ->
  let j = connect agents in
  let hs = Rig_remote.hosts j in
  let ds = List.concat_map mem hs in
  let kept = List.map (fun d -> far_of_string d "kept") (hs @ ds) in
  Rig_remote.close j;
  equal (option string) None (Rig_remote.failure j);
  List.iter
    (fun d -> equal ~msg:(Rig.name d) lost_w (Some "closed") (Rig.lost d))
    (hs @ ds);
  equal (list exit_w)
    [ (0, [ "closed" ]); (0, [ "closed" ]) ]
    (List.map finish agents);
  Rig_remote.close j;
  is_error (Rig_remote.devices (List.hd hs) "MEM");
  ignore (Sys.opaque_identity kept)

(* Work submitted before the close is done when it returns. *)
let close_waits () =
  with_agents @@ fun agents ->
  let j = connect agents in
  let h = List.hd (Rig_remote.hosts j) in
  let n = 8 lsl 20 in
  let s = String.init n (fun i -> Char.chr (i * 13 land 0xff)) in
  let far = far_of_string h s in
  let here = B.create Rig.host n in
  ignore
    (Rig.submit
       (Sub.make h
          [|
            {
              Sub.queue = "COPY:0";
              after = [||];
              work = Sub.Copy { src = far; dst = here };
            };
          |])
       ~run:(Sub.Run.make ()) ~buffers:[||] ~waits:[||]);
  Rig_remote.close j;
  equal ~msg:"the copy's bytes" bool true (read_host here = s);
  equal exit_w (0, [ "closed" ]) (finish (List.hd agents))

let closes =
  group "close"
    [
      test "close ends every device, here and at every agent, which exits 0"
        close_order;
      test "close waits for the work submitted before it" close_waits;
    ]

(* Rails *)

external get64 : Rig_remote_abi.area -> int -> int64 = "%caml_bigstring_get64"

let host_record h =
  match Rig.capability h Rig_remote_abi.key with
  | Some (Rig_remote_abi.Host r) -> r
  | _ -> fail "a host's record is Host"

let t5 = { Rig_remote_abi.src = 0; dst = 0; length = 5 }

let between_agents () =
  with_job ~n:2 @@ fun j _ ->
  match List.map host_record (Rig_remote.hosts j) with
  | [ r1; r2 ] ->
      let rail =
        require_ok (r1.rail (Some r2) ~send:[| t5 |] ~receive:[| t5 |])
      in
      equal ~msg:"no end here" bool true (Option.is_none rail.local);
      rail.release ();
      rail.release ();
      let bad msg send receive =
        raises_match ~msg Exn.invalid_arg (fun () ->
            r1.rail (Some r2) ~send ~receive)
      in
      raises_match ~msg:"its own machine" Exn.invalid_arg (fun () ->
          r1.rail (Some r1) ~send:[| t5 |] ~receive:[||]);
      bad "no transfer" [||] [||];
      bad "length 0" [| { t5 with length = 0 } |] [||];
      bad "src -1" [| { t5 with src = -1 } |] [||];
      bad "dst -1" [||] [| { t5 with dst = -1 } |];
      bad "dst max_int" [||] [| { t5 with dst = max_int } |]
  | _ -> fail "two hosts"

(* A rail with this process: its end here, laid out as the abi says, and a
   transfer sent once its ready count is stored. *)
let with_here () =
  with_job @@ fun j _ ->
  let r = host_record (List.hd (Rig_remote.hosts j)) in
  let t = { t5 with src = 300 } in
  (* [receive] is what this process sends to the agent's machine. *)
  let rail = require_ok (r.rail None ~send:[| t5 |] ~receive:[| t |]) in
  let e = require_some rail.local in
  equal ~msg:"outbound" int (2 * 512) (Bigarray.Array1.dim e.outbound);
  equal ~msg:"inbound" int (2 * 256) (Bigarray.Array1.dim e.inbound);
  List.iter
    (fun at -> equal ~msg:"a count starts at 0" int64 0L (get64 e.counts at))
    [ 0; 128; 256 ];
  String.iteri (fun i c -> e.outbound.{300 + i} <- c) "hello";
  e.ready 1;
  until ~what:"sent" (fun () -> get64 e.counts 128 >= 1L);
  rail.release ()

let other_job () =
  let old = with_job @@ fun j _ -> host_record (List.hd (Rig_remote.hosts j)) in
  with_job @@ fun j _ ->
  let r = host_record (List.hd (Rig_remote.hosts j)) in
  is_error (r.rail (Some old) ~send:[| t5 |] ~receive:[||]);
  Rig_remote.close j;
  is_error (r.rail None ~send:[| t5 |] ~receive:[||])

let rails =
  group "rail"
    [
      test "a rail between two agents has no end here, and releases once"
        between_agents;
      test "a rail with this process sends a transfer once ready" with_here;
      test "a rail to another job's machine, or after the close, is an Error"
        other_job;
    ]

(* Fork *)

(* The child's exit status: whether a use of the job's device raised Lost and
   the job reads as failed there. *)
let child j d =
  let lost =
    match B.create d 16 with _ -> false | exception Rig.Lost _ -> true
  in
  let failed = Option.is_some (Rig_remote.failure j) in
  Unix._exit (if lost && failed then 0 else 1)

let forked () =
  with_agents @@ fun agents ->
  let j = connect agents in
  let d = List.hd (mem (List.hd (Rig_remote.hosts j))) in
  (match Unix.fork () with
  | 0 -> child j d
  | pid -> (
      match Unix.waitpid [] pid with
      | _, Unix.WEXITED 0 -> ()
      | _ -> fail "the child used the job, or its job did not read as failed"));
  equal (option string) None (Rig_remote.failure j);
  equal string "the parent goes on"
    (read (far_of_string d "the parent goes on"));
  Rig_remote.close j;
  equal exit_w (0, [ "closed" ]) (finish (List.hd agents))

let forks =
  group "fork"
    [ test "a child's job is failed there, and the parent's goes on" forked ]

(* Frames rig.remote's controller never sends *)

let answer_pp = Format.pp_print_string

(* Requests an agent cannot apply are refused, and the job goes on: an id the
   job holds already, a rail of no transfer, of an empty one, of one past its
   areas' bound, or of areas larger than the agent's memory. *)
let refused_requests () =
  with_agents @@ fun agents ->
  let a = List.hd agents in
  let fd = raw_controller a in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  is_ok ~msg:"the join" ~pp:answer_pp (join_alone fd a);
  is_ok ~msg:"memory 1" ~pp:answer_pp (ask fd (alloc_host 1 16));
  is_error ~msg:"memory 1 again" (ask fd (alloc_host 1 16));
  is_ok ~msg:"rail 2" ~pp:answer_pp (ask fd (rail_out 2 [ (0, 0, 5) ]));
  is_error ~msg:"rail 2 again" (ask fd (rail_out 2 [ (0, 0, 5) ]));
  is_error ~msg:"a rail as memory 1" (ask fd (rail_out 1 [ (0, 0, 5) ]));
  is_error ~msg:"a rail of no transfer" (ask fd (rail_out 3 []));
  is_error ~msg:"a rail of an empty transfer"
    (ask fd (rail_out 4 [ (0, 0, 0) ]));
  is_error ~msg:"a rail from byte max_int"
    (ask fd (rail_out 5 [ (max_int, 0, 1) ]));
  is_error ~msg:"a rail of 2^61 bytes"
    (ask fd (rail_out 6 [ ((1 lsl 60) - 1, 0, 1) ]));
  send fd (frame k_close "");
  equal ~msg:"the agent's close" (option int) (Some k_close)
    (Option.map fst (next_frame fd));
  equal exit_w (0, [ "closed" ]) (finish a)

(* A hand-over naming bytes outside its memory fails the job: the agent ends
   with it, and does not die of it. *)
let outside_memory () =
  with_agents @@ fun agents ->
  let a = List.hd agents in
  let fd = raw_controller a in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  is_ok ~msg:"the join" ~pp:answer_pp (join_alone fd a);
  is_ok ~msg:"memory 1" ~pp:answer_pp (ask fd (alloc_host 1 16));
  send fd (frame k_handover (copy_on_host ~value:1 ~bytes:8 (1, 12) (1, 0)));
  match finish a with
  | 2, [ why ] -> starts_with ~affix:"failed: " why
  | code, lines ->
      failf "the agent exited %d, printing [%s]" code (String.concat "; " lines)

(* Each upload has its bytes, an empty one included: a hand-over uploads
   nothing, then "abcd", to memory 1, and reads it back. *)
let empty_upload () =
  with_agents @@ fun agents ->
  let a = List.hd agents in
  let fd = raw_controller a in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  is_ok ~msg:"the join" ~pp:answer_pp (join_alone fd a);
  is_ok ~msg:"memory 1" ~pp:answer_pp (ask fd (alloc_host 1 16));
  let local = u8 1 and memory = u8 0 ^ u64 1 ^ u64 0 in
  let copy bytes src dst = u8 1 ^ u64 bytes ^ src ^ dst in
  send fd
    (frame k_handover
       (u64 0 ^ u64 1 ^ u32 0 ^ u32 3 ^ copy 0 local memory
      ^ copy 4 local memory ^ copy 4 memory local ^ "abcd"));
  let k_word = 5 and k_bytes = 6 in
  let bytes = next_frame fd in
  let word = next_frame fd in
  send fd (frame k_close "");
  equal
    (list (option (pair int string)))
    [ Some (k_bytes, u64 0 ^ u64 1 ^ "abcd"); Some (k_word, u64 0 ^ u64 1) ]
    [ bytes; word ];
  equal ~msg:"the agent's close" (option int) (Some k_close)
    (Option.map fst (next_frame fd));
  equal exit_w (0, [ "closed" ]) (finish a)

let frames =
  group "frames"
    [
      test "requests an agent cannot apply are refused, and the job goes on"
        refused_requests;
      test "a hand-over outside its memory fails the job at the agent"
        outside_memory;
      test "an empty upload leaves the next upload its bytes" empty_upload;
    ]

(* Launched programs *)

let report_w = triple int (list string) (list string)

let without_variables () =
  equal ~msg:"no RIG_REMOTE_REPORT" (option string) None
    (Option.map (fun _ -> "a job") (Rig_remote.launched ()));
  raises_match ~msg:"a second call" Exn.invalid_arg (fun () ->
      Rig_remote.launched ())

let started_closed () =
  with_hex_agents ~n:2 @@ fun agents ->
  let e = ended (launch (launch_vars agents) "close") in
  equal report_w
    (0, [ "started"; "closed" ], [ "connected"; "CPU@m1"; "CPU@m2" ])
    e;
  equal (list exit_w)
    [ (0, [ "closed" ]); (0, [ "closed" ]) ]
    (List.map finish agents)

let exits_unclosed () =
  with_hex_agents @@ fun agents ->
  let e = ended (launch (launch_vars agents) "exit") in
  equal report_w (0, [ "started"; "closed" ], [ "connected"; "CPU@m1" ]) e;
  equal (list exit_w) [ (0, [ "closed" ]) ] (List.map finish agents)

(* A report whose reader is gone fails each write, and the program goes on: no
   SIGPIPE ends it. *)
let reader_gone () =
  with_hex_agents @@ fun agents ->
  let e = ended (launch ~reader_gone:true (launch_vars agents) "close") in
  equal report_w (0, [], [ "connected"; "CPU@m1" ]) e;
  equal (list exit_w) [ (0, [ "closed" ]) ] (List.map finish agents)

(* The program kills itself once its process failed: the report was written
   before. *)
let agent_killed () =
  with_hex_agents ~n:2 @@ fun agents ->
  let p = launch (launch_vars agents) "watch" in
  Fun.protect
    ~finally:(fun () ->
      try Unix.kill p.lpid Sys.sigkill with Unix.Unix_error _ -> ())
    (fun () ->
      equal ~msg:"connected" string "connected" (said_line p);
      kill (List.hd agents);
      equal report_w
        ( -Sys.sigkill,
          [ "started"; "failed m1: closed its connection" ],
          [ "CPU@m1"; "CPU@m2" ] )
        (ended p))

(* A port nothing listens at. *)
let dead_port () =
  let s = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.bind s (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
  let port =
    match Unix.getsockname s with
    | Unix.ADDR_INET (_, p) -> p
    | Unix.ADDR_UNIX _ -> assert false
  in
  Unix.close s;
  port

let failed_start (vars, affix) =
  let code, report, said = ended (launch vars "close") in
  equal ~msg:"exit code" int 1 code;
  (match report with
  | [ l ] -> starts_with ~msg:"the report" ~affix:("failed " ^ affix) l
  | ls -> failf "reported [%s]" (String.concat "; " ls));
  match said with
  | [ l ] -> starts_with ~msg:"launched's error" ~affix:("error: " ^ affix) l
  | ls -> failf "said [%s]" (String.concat "; " ls)

let bad_starts =
  let vars ?(agents = "m1=127.0.0.1:1") ?(key = hex_key) () =
    [
      ("RIG_REMOTE_REPORT", "3");
      ("RIG_REMOTE_AGENTS", agents);
      ("RIG_REMOTE_KEY", key);
    ]
  in
  [
    ( "nothing listens",
      (vars ~agents:(Printf.sprintf "m1=127.0.0.1:%d" (dead_port ())) (), "m1: ")
    );
    ("no name", (vars ~agents:"127.0.0.1:1" (), "RIG_REMOTE_AGENTS"));
    ("no port", (vars ~agents:"m1=127.0.0.1" (), "RIG_REMOTE_AGENTS"));
    ("a short key", (vars ~key:"abcd" (), "RIG_REMOTE_KEY"));
    ( "a key not hexadecimal",
      (vars ~key:(String.make 64 'x') (), "RIG_REMOTE_KEY") );
    ( "no key",
      ( [ ("RIG_REMOTE_REPORT", "3"); ("RIG_REMOTE_AGENTS", "m1=127.0.0.1:1") ],
        "RIG_REMOTE_KEY" ) );
  ]

let forked () =
  with_hex_agents @@ fun agents ->
  let e = ended (launch (launch_vars agents) "fork") in
  equal report_w
    (0, [ "started"; "closed" ], [ "connected"; "CPU@m1"; "going on" ])
    e;
  equal (list exit_w) [ (0, [ "closed" ]) ] (List.map finish agents)

(* The program runs a shell that counts the variables it sees and writes on
   descriptor 3, which the report's close-on-exec closed there. *)
let execs () =
  with_hex_agents @@ fun agents ->
  let p =
    launch
      ~args:[ "env | grep -c '^RIG_REMOTE_'; echo leaked >&3" ]
      (launch_vars agents) "exec"
  in
  let code, report, said = ended p in
  equal ~msg:"exit code" int 0 code;
  equal ~msg:"the report" (list string) [ "started"; "closed" ] report;
  equal ~msg:"variables the shell saw" bool true (List.mem "0" said);
  ignore (List.map finish agents)

(* An agent under RIG_REMOTE_REPORT, here its standard output, reports its job's
   end there before support/agent.exe prints its own. *)
let serve_closed () =
  with_hex_agents ~vars:[ ("RIG_REMOTE_REPORT", "1") ] @@ fun agents ->
  (match Rig_remote.connect ~key:(as_key hex_key) (List.map address agents) with
  | Ok j -> Rig_remote.close j
  | Error why -> fail why);
  equal exit_w (0, [ "closed"; "closed" ]) (finish (List.hd agents))

let serve_failed () =
  with_key_file ~key:hex_key @@ fun file ->
  let a = start ~vars:[ ("RIG_REMOTE_REPORT", "1") ] file in
  let c = start_controller file "wait" [ a ] in
  Fun.protect
    ~finally:(fun () -> List.iter kill [ c; a ])
    (fun () ->
      equal ~msg:"the controller" string "connected" (input_line c.out);
      kill c;
      match finish a with
      | 2, [ report; said ] ->
          ends_with ~msg:"the report" ~affix:"closed its connection" report;
          equal ~msg:"the report's reason" string said
            ("failed: " ^ String.sub report 7 (String.length report - 7))
      | code, lines ->
          failf "the agent exited %d, printing [%s]" code
            (String.concat "; " lines))

let launching =
  group "launched"
    [
      test "without the variables it is None, and a second call raises"
        without_variables;
      test "started, then closed, its machines named as the launcher names them"
        started_closed;
      test "a program that exits without close reports closed" exits_unclosed;
      test "a program whose report's reader is gone goes on" reader_gone;
      test "an agent killed: failed, reported before the process fails"
        agent_killed;
      cases ~name:fst "a start that fails reports failed" bad_starts
        (fun (_, c) -> failed_start c);
      test "a child of fork reports nothing, and the job goes on" forked;
      test "a program it runs sees no variable and no report descriptor" execs;
      test "an agent under RIG_REMOTE_REPORT reports its job closed"
        serve_closed;
      test "an agent under RIG_REMOTE_REPORT reports its job failed"
        serve_failed;
    ]

(* Keys *)

(* A key's bytes are never in its error: a key of [n] 'k's has none of four. *)
let key_refused n =
  let why = require_error (Rig_remote.key (String.make n 'k')) in
  contains ~msg:"gives its length" ~sub:(Printf.sprintf "%d bytes" n) why;
  not_contains ~msg:"never its bytes" ~sub:"kkkk" why

let read_ok contents =
  let file = write_file contents in
  Fun.protect
    ~finally:(fun () -> Sys.remove file)
    (fun () -> is_ok (Rig_remote.read_key file))

let refused file =
  let why = require_error (Rig_remote.read_key file) in
  starts_with ~msg:"names the file" ~affix:file why

let refused_perm perm =
  let file = write_file ~perm key in
  Fun.protect ~finally:(fun () -> Sys.remove file) (fun () -> refused file)

let refused_size n =
  let file = write_file (String.make n 'k') in
  Fun.protect ~finally:(fun () -> Sys.remove file) (fun () -> refused file)

(* A FIFO opened for reading waits for a writer: read_key must answer without
   one. If it waits, the test opens a writer to end the wait. *)
let fifo () =
  let file =
    Filename.concat
      (Filename.get_temp_dir_name ())
      (Printf.sprintf "rig-remote-%d.fifo" (Unix.getpid ()))
  in
  (try Sys.remove file with Sys_error _ -> ());
  Unix.mkfifo file 0o600;
  Fun.protect
    ~finally:(fun () -> Sys.remove file)
    (fun () ->
      let r = ref None in
      let t =
        Thread.create (fun () -> r := Some (Rig_remote.read_key file)) ()
      in
      let t0 = Unix.gettimeofday () in
      while !r = None && Unix.gettimeofday () -. t0 < 2. do
        Thread.delay 0.01
      done;
      let waited = !r = None in
      if waited then begin
        let w = Unix.openfile file [ Unix.O_WRONLY; Unix.O_NONBLOCK ] 0 in
        Thread.join t;
        Unix.close w
      end
      else Thread.join t;
      equal ~msg:"read_key waited for a writer" bool false waited;
      refused file)

let keys =
  group "keys"
    [
      test "16 bytes and 4096 bytes are keys" (fun () ->
          is_ok (Rig_remote.key (String.make 16 'k'));
          is_ok
            (Rig_remote.key
               (String.init 4096 (fun i -> Char.chr (i land 0xff)))));
      cases
        ~name:(Printf.sprintf "%d bytes are no key, said by length")
        "key length" [ 0; 15; 4097; 65536 ] key_refused;
      test "a file of 16 bytes and one of 4096, of this user alone, are keys"
        (fun () ->
          read_ok (String.make 16 'a');
          read_ok (String.init 4096 (fun i -> Char.chr (i land 0xff))));
      cases
        ~name:(Printf.sprintf "a file of mode %o is refused, naming it")
        "granting others access"
        [ 0o640; 0o604; 0o620; 0o602; 0o644; 0o666; 0o710; 0o701; 0o711 ]
        refused_perm;
      test "a file of mode 700 is a key" (fun () ->
          let file = write_file ~perm:0o700 key in
          Fun.protect
            ~finally:(fun () -> Sys.remove file)
            (fun () -> is_ok (Rig_remote.read_key file)));
      cases
        ~name:(Printf.sprintf "a file of %d bytes is refused, naming it")
        "size" [ 0; 15; 4097 ] refused_size;
      test "another user's file is refused, naming it" (fun () ->
          refused "/etc/hosts");
      test "a missing file, a directory and a device are refused" (fun () ->
          refused
            (Filename.concat
               (Filename.get_temp_dir_name ())
               "rig-remote-no-such.key");
          refused (Filename.get_temp_dir_name ());
          refused "/dev/null");
      test "a FIFO is refused without waiting for a writer" fifo;
    ]

(* Processes of a job *)

(* Agents whose controller is support/controller.exe, in [mode]. *)
let with_controller ?(n = 2) mode f =
  with_key_file @@ fun file ->
  let agents = List.init n (fun _ -> start file) in
  let c = start_controller file mode agents in
  Fun.protect
    ~finally:(fun () -> List.iter kill (c :: agents))
    (fun () -> f c agents)

(* The most an agent may take to end once its controller's process ended. *)
let end_bound = 5.

let ends_in_bound agents =
  let t0 = Unix.gettimeofday () in
  let ends = List.map finish agents in
  less ~msg:"seconds for the agents to end" float_exact ~than:end_bound
    (Unix.gettimeofday () -. t0);
  ends

(* The controller's process exits without a close: it closes its job at exit,
   and every agent returns from serve with the job closed. *)
let controller_exits () =
  with_controller "exit" @@ fun c agents ->
  equal ~msg:"the controller" exit_w (0, [ "connected" ]) (finish c);
  equal (list exit_w)
    [ (0, [ "closed" ]); (0, [ "closed" ]) ]
    (ends_in_bound agents)

(* The controller's process is killed: its connections end without a close, and
   every agent returns from serve with the job failed. *)
let controller_killed () =
  with_controller "wait" @@ fun c agents ->
  equal ~msg:"the controller" string "connected" (input_line c.out);
  kill c;
  List.iter
    (function
      | 2, [ why ] -> ends_with ~affix:"closed its connection" why
      | code, lines ->
          failf "an agent exited %d, printing [%s]" code
            (String.concat "; " lines))
    (ends_in_bound agents)

(* serve raises once it served, here in an agent that served a job. *)
let served_twice () =
  with_agents ~mode:"twice" @@ fun agents ->
  Rig_remote.close (connect agents);
  equal exit_w (0, [ "closed"; "serve raised" ]) (finish (List.hd agents))

(* Two domains serve one agent at once: one raises, the other serves its job. *)
let served_together () =
  with_agents ~mode:"together" @@ fun agents ->
  Rig_remote.close (connect agents);
  equal exit_w (0, [ "serve raised"; "closed" ]) (finish (List.hd agents))

(* A kind named twice raises before serve waits for a controller; a serve that
   waited instead fails the test after 5 s. *)
let kind_twice () =
  match Rig_remote.listen ~key:(as_key key) "127.0.0.1" 0 with
  | Error why -> fail why
  | Ok a ->
      let r = ref None in
      let kinds = [ ("A", fun () -> Ok []); ("A", fun () -> Ok []) ] in
      let _ =
        Thread.create
          (fun () ->
            r :=
              Some
                (match Rig_remote.serve a kinds with
                | _ -> "returned"
                | exception Invalid_argument _ -> "raised"))
          ()
      in
      until ~what:"serve's raise" (fun () -> !r <> None);
      equal (option string) (Some "raised") !r

let processes =
  group "process"
    [
      test "a controller that exits without close closes its job"
        controller_exits;
      test "a controller killed fails its job at every agent" controller_killed;
      test "serve raises once it served" served_twice;
      test "serve raises in one of two domains that serve at once"
        served_together;
      test "serve raises on a kind named twice" kind_twice;
    ]

let () =
  Watchdog.start ();
  exit
    (run "rig_remote"
       [
         group ~timeout:60. "rig_remote"
           [
             connecting;
             machines;
             copies;
             closes;
             rails;
             forks;
             frames;
             launching;
             keys;
             processes;
           ];
       ])
