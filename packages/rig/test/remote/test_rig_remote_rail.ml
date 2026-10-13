(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A GPU queue that waits on a rail's arrived count when the job fails: the
   failure raises the count to Int64.max_int (Rig_remote_abi), so the wait
   ends, and the work's point raises Lost. The job fails by its agent's death,
   or by its connection's end while the agent lives. Each case runs in a child
   of its own, since a failed job fails the process for good. *)

open Windtrap
open Remote_job
module Sub = Rig.Submission
module A = Rig_amd_abi
module N = Rig_nv_abi

module type Gpu = Rig_gpu_support.Conformance

(* The ring words and commands that write 1 at [marker] once the queue reaches
   them, then hold the queue until the 64 bits at [arrived] are at least 1. An
   AMD ring holds the PM4 packets themselves, with no commands apart; an NV ring
   holds a GPFIFO entry of the methods at [commands]. *)
let amd d ~marker ~arrived ~commands:_ =
  let g = (require_some (Rig.capability d A.Capability.key)).gpu in
  ( A.Packet.encode Fun.id
      (A.Pm4.release_mem g System marker (Data_64 1L)
      @ A.Pm4.wait_64 g arrived Greater_equal 1L ()
      @ A.Pm4.acquire_mem g System),
    "" )

let nv _ ~marker ~arrived ~commands =
  let i v = N.Template.Int (Int64.to_int v) in
  let methods =
    (N.Template.concat
       [
         N.Method.release (i marker) (Int 1);
         N.Method.acquire (i arrived) (Int 1);
       ])
      .bytes
  in
  let words = String.length methods / 4 in
  ((N.Gpfifo.entry (i commands) ~offset:0 ~words).bytes, methods)

(* The 64-bit word at byte [at] of the bytes [a], read with acquire order
   while work may write it. *)
let get64 = Rig_support.load64

(* A relay on the loopback to agent [a], and the function that ends both of
   its connections. *)
let relay a =
  let l = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.bind l (Unix.ADDR_INET (Unix.inet_addr_loopback, 0));
  Unix.listen l 1;
  let port =
    match Unix.getsockname l with Unix.ADDR_INET (_, p) -> p | _ -> 0
  in
  let ends = Atomic.make [] in
  let pump src dst =
    let b = Bytes.create 65536 in
    let rec go () =
      match Unix.read src b 0 65536 with
      | 0 -> ()
      | n ->
          ignore (Unix.write dst b 0 n : int);
          go ()
    in
    try go () with Unix.Unix_error _ -> ()
  in
  let accept () =
    let c, _ = Unix.accept ~cloexec:true l in
    let s = Unix.socket ~cloexec:true Unix.PF_INET Unix.SOCK_STREAM 0 in
    Unix.connect s (Unix.ADDR_INET (Unix.inet_addr_loopback, a.port));
    Atomic.set ends [ c; s ];
    ignore (Thread.create (fun () -> pump c s) ());
    pump s c
  in
  ignore (Thread.create accept ());
  let sever () =
    List.iter (fun fd -> Unix.shutdown fd Unix.SHUTDOWN_ALL) (Atomic.get ends)
  in
  (("127.0.0.1", port), sever)

(* What the child sees: the queue held at its wait, then, once the job failed,
   the wait's outcome and the rail's counts. *)
let case (module G : Gpu) held cut () =
  let g = G.open_ () in
  let d = g.d in
  with_agents @@ fun agents ->
  let a = List.hd agents in
  let address, fail_job =
    match cut with
    | `Kill -> (address a, fun () -> kill a)
    | `Sever -> relay a
  in
  let j =
    match Rig_remote.connect ~key:(as_key key) [ address ] with
    | Ok j -> j
    | Error why -> failf "connect: %s" why
  in
  let r =
    match Rig.capability (List.hd (Rig_remote.hosts j)) Rig_remote_abi.key with
    | Some (Rig_remote_abi.Host r) -> r
    | _ -> fail "a host's record is Host"
  in
  let t = { Rig_remote_abi.src = 0; dst = 0; length = 8 } in
  let rail = require_ok (r.rail None ~send:[| t |] ~receive:[||]) in
  let e = require_some rail.local in
  let borrow b = require_some (B.borrow d b) in
  let page () = B.view (B.create Rig.host 65536) ~first:0 ~length:4096 in
  let counts = B.of_bigarray e.counts and mark = page () and cmds = page () in
  (* The host reads [mark] while the queue's work writes it. *)
  let mark_bytes = B.export mark Bigarray.char in
  let ring, _ = held d ~marker:0L ~arrived:0L ~commands:0L in
  let words = B.create Rig.host (String.length ring) in
  let fixed = List.map borrow [ counts; mark; cmds ] in
  let s =
    Sub.make ~keep:rail
      ~fixed:(List.combine fixed [ Rig.Read; Read_write; Read ])
      d
      [| Sub.words words |]
  in
  let at i = Int64.of_int (Sub.address s (List.nth fixed i)) in
  let ring, methods =
    held d ~marker:(at 1) ~arrived:(Int64.add (at 0) 256L) ~commands:(at 2)
  in
  let write b w =
    B.write b Bigarray.char (fun ba -> String.iteri (fun i c -> ba.{i} <- c) w)
  in
  write words ring;
  write cmds methods;
  let p = Rig.submit s ~run:(Sub.Run.make ()) ~buffers:[||] in
  until ~what:"the queue at its wait" (fun () -> get64 mark_bytes 0 = 1L);
  let reached = Rig.signaled d >= Rig.Point.value p in
  let before =
    Printf.sprintf "arrived %Ld, done %b" (get64 e.counts 256) reached
  in
  fail_job ();
  (* The failure raises the counts, then loses the process's devices: the
     held work may run to its end before the loss or not, and the wait
     returns or raises [Lost] as it did. Either way it answers. *)
  until ~what:"the GPU's loss" (fun () -> Rig.lost d <> None);
  let wait =
    match Rig.Point.wait p with
    | () -> "answered"
    | exception Rig.Lost _ -> "answered"
  in
  let after =
    List.map (fun at -> get64 e.counts at = Int64.max_int) [ 0; 128; 256 ]
  in
  [
    before;
    "wait: " ^ wait;
    "terminal: " ^ String.concat " " (List.map string_of_bool after);
  ]

let status = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> Printf.sprintf "killed by signal %d" n
  | Unix.WSTOPPED n -> Printf.sprintf "stopped by signal %d" n

(* Runs [f] in a child and is the lines it returned, or the exception it
   raised. *)
let in_child f =
  Rig_support.needs_fork ();
  let r, w = Unix.pipe ~cloexec:true () in
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      let lines = try f () with e -> [ Printexc.to_string e ] in
      let oc = Unix.out_channel_of_descr w in
      List.iter (fun l -> output_string oc (l ^ "\n")) lines;
      close_out oc;
      Unix._exit 0
  | pid ->
      Unix.close w;
      let ic = Unix.in_channel_of_descr r in
      let lines = In_channel.input_lines ic in
      close_in ic;
      let _, ended = Unix.waitpid [] pid in
      equal ~msg:"the child" string "exited 0" (status ended);
      lines

let gpus =
  [ ((module Rig_amd_support : Gpu), amd); ((module Rig_nv_support : Gpu), nv) ]

let tests =
  List.concat_map
    (fun ((module G : Gpu), held) ->
      let check cut () =
        if not (G.present ()) then skip ~reason:("no " ^ G.class_ ^ " GPU") ();
        equal (list string)
          [
            "arrived 0, done false"; "wait: answered"; "terminal: true true true";
          ]
          (in_child (case (module G) held cut))
      in
      [
        test (G.class_ ^ ": the agent dies") (check `Kill);
        test (G.class_ ^ ": its connection ends") (check `Sever);
      ])
    gpus

let () =
  Watchdog.start ();
  exit (run "rig_remote.rail" [ group ~timeout:60. "rig_remote" tests ])
