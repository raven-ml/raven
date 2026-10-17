(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A forked child of a process that opened devices. Its own suite: a process
   that ran a domain cannot fork. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?waits s =
  Rig.submit ?waits (Sub.Run.make s) ~buffers

let timeout = 60.
let empty d = Sub.make d [||]
let raises_lost f = match f () with _ -> false | exception Rig.Lost _ -> true

let status = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> "killed by " ^ Sys.signal_to_string n
  | Unix.WSTOPPED n -> "stopped by " ^ Sys.signal_to_string n

(* Runs [child] in a forked child and is what it returned: the child writes its
   answer to a pipe and leaves without running this process's exit. *)
let in_child child =
  let r, w = Unix.pipe () in
  Support.needs_fork ();
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      let lines = try child () with e -> [ Printexc.to_string e ] in
      let oc = Unix.out_channel_of_descr w in
      List.iter (fun l -> output_string oc (l ^ "\n")) lines;
      close_out oc;
      Unix._exit 0
  | pid ->
      Unix.close w;
      let ic = Unix.in_channel_of_descr r in
      let lines = In_channel.input_lines ic in
      close_in ic;
      let _, status = Unix.waitpid [] pid in
      (lines, status)

(* The child sees the device lost, calls no driver function and reads no word:
   memory dropped in it stays, even once the word reads every submitted value,
   which would free it in a process that read it. *)
let test_child () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let d, p = P.open_ "fork:child" in
  let b = ref (Some (B.create d 64)) in
  let pt = submit (empty d) in
  let v = Rig.Point.value pt in
  let calls = P.log p in
  let lines, ended =
    in_child (fun () ->
        let lost = Rig.lost d in
        let submit = raises_lost (fun () -> submit (empty d)) in
        let wait = raises_lost (fun () -> Rig.Point.wait pt) in
        b := None;
        Gc.full_major ();
        ignore (B.create Rig.host 8);
        P.set_word p v;
        Gc.full_major ();
        ignore (B.create Rig.host 8);
        [
          Option.value ~default:"not lost" lost;
          Printf.sprintf "submit raises Lost: %b" submit;
          Printf.sprintf "wait raises Lost: %b" wait;
          Printf.sprintf "driver calls since the fork: %s"
            (String.concat " "
               (List.filteri (fun i _ -> i >= List.length calls) (P.log p)));
        ])
  in
  equal string "exited 0" (status ended);
  equal (list string)
    [
      "forked";
      "submit raises Lost: true";
      "wait raises Lost: true";
      "driver calls since the fork: ";
    ]
    lines;
  equal (option string) None (Rig.lost d);
  Rig.Point.wait pt;
  ignore (Sys.opaque_identity !b)

let page_bytes = 1 lsl 16

(* An io device whose memory is host bytes on a page, whose state its library
   holds, and which counts the regions it holds and its stops. Its reads fault
   while [faulty] is set, and wait while [parked] is. Each application is a
   library of its own, with counts of its own. *)
module Make () = struct
  type t = unit

  (* A host buffer of at least a page, which starts on one. *)
  type region = B.t

  exception Fault of string

  let held = Atomic.make 0
  let stops = Atomic.make 0
  let faulty = Atomic.make false
  let parked = Atomic.make false
  let inside = Atomic.make 0
  let region_key : region Type.Id.t = Type.Id.make ()
  let budget () = max_int

  let alloc () n =
    Atomic.incr held;
    Some (B.create Rig.host (Int.max n page_bytes))

  let free () _ = Atomic.decr held

  let read () _ ~at:_ ~dst:_ ~len:_ =
    if Atomic.get faulty then raise (Fault "a read fault");
    Atomic.incr inside;
    while Atomic.get parked do
      Thread.yield ()
    done;
    Atomic.decr inside

  let write () _ ~at:_ ~src:_ ~len:_ = ()
  let pages () r = Some (B.export r Bigarray.char)
  let prefetch () _ ~at:_ ~len:_ = ()
  let stop () = Atomic.incr stops
end

module Store = Make ()

let open_store name =
  require_ok ~pp:Format.pp_print_string
    (Rig.open_io { name; io = (module Store); make = (fun () -> Ok ()) })

(* An io device's state is its library's, which decides what a fork does to it:
   rig leaves it usable in the child, where a driver's device is lost. *)
let test_io_child () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let io = open_store "fork:io" in
  let d, _ = P.open_ "fork:beside-io" in
  let lines, ended =
    in_child (fun () ->
        [
          Option.value ~default:"not lost" (Rig.lost io);
          Printf.sprintf "bytes made: %d" (B.length (B.create io 8));
          Option.value ~default:"not lost" (Rig.lost d);
        ])
  in
  equal string "exited 0" (status ended);
  equal (list string) [ "not lost"; "bytes made: 8"; "forked" ] lines

(* A forked child frees the io memory it drops, as its parent does: a child that
   makes and drops io buffers in a loop holds a bounded number of them. *)
let test_io_loop () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let io = open_store "fork:io-loop" in
  let lines, ended =
    in_child (fun () ->
        let base = Atomic.get Store.held and most = ref 0 in
        for _ = 1 to 100 do
          ignore (Sys.opaque_identity (B.create io 8));
          Gc.full_major ();
          most := Int.max !most (Atomic.get Store.held - base)
        done;
        [ Printf.sprintf "most held: %d" !most ])
  in
  equal string "exited 0" (status ended);
  equal (list string) [ "most held: 1" ] lines

(* A parent's device works on the parent's copy of io memory: the child frees
   its own copy once dropped, whatever work the device still has on it. *)
let test_io_used () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let io = open_store "fork:io-used" in
  let d, p = P.open_ "fork:io-user" in
  let m = ref (Some (B.create io page_bytes)) in
  let s = Sub.make d [| Support.touching [| Sub.read |] |] in
  ignore (submit s ~buffers:[| require_some (B.borrow d (Option.get !m)) |]);
  let before = Atomic.get Store.held in
  let lines, ended =
    in_child (fun () ->
        m := None;
        Gc.full_major ();
        Gc.full_major ();
        ignore (B.create io 0);
        [ Printf.sprintf "freed: %d" (before - Atomic.get Store.held) ])
  in
  equal string "exited 0" (status ended);
  equal (list string) [ "freed: 1" ] lines;
  ignore (P.run p);
  ignore (Sys.opaque_identity !m)

(* An io device closed while a buffer of it lives has stopped and still holds
   that memory: a child that drops the buffer gives the memory back at the
   device's free_cache, as its parent would. *)
let test_io_closed () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let io = open_store "fork:io-closed" in
  let m = ref (Some (B.create io 8)) in
  Rig.close io;
  let lines, ended =
    in_child (fun () ->
        Rig.free_cache io;
        let before = Atomic.get Store.held in
        m := None;
        Gc.full_major ();
        Rig.free_cache io;
        [ Printf.sprintf "freed: %d" (before - Atomic.get Store.held) ])
  in
  equal string "exited 0" (status ended);
  equal (list string) [ "freed: 1" ] lines;
  ignore (Sys.opaque_identity !m)

(* An io device lost while a thread of the parent is inside a call of it, so
   that its stop is owed at the fork: the child runs no stop of it, in a close
   or a fail, and frees none of its memory, since the stop is the parent's,
   which the thread runs once its call returns. *)
let test_io_owed () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let module Store = Make () in
  let io =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_io
         { name = "fork:io-owed"; io = (module Store); make = (fun () -> Ok ()) })
  in
  let m = B.create io 8 and other = ref (Some (B.create io 8)) in
  let stops = Atomic.get Store.stops in
  let read b () = B.copy ~src:b ~dst:(B.create Rig.host 8) in
  Atomic.set Store.parked true;
  let reader = Thread.create (fun () -> ignore (raises_lost (read m))) () in
  Support.await "a read inside the device" (fun () ->
      Atomic.get Store.inside = 1);
  Atomic.set Store.faulty true;
  let copy = raises_lost (read (Option.get !other)) in
  Atomic.set Store.faulty false;
  let before = Atomic.get Store.stops - stops in
  let held = Atomic.get Store.held in
  let lines, ended =
    in_child (fun () ->
        other := None;
        Gc.full_major ();
        Rig.close io;
        Rig.fail "the child fails";
        [
          Printf.sprintf "stops: %d" (Atomic.get Store.stops - stops);
          Printf.sprintf "freed: %d" (held - Atomic.get Store.held);
        ])
  in
  Atomic.set Store.parked false;
  Thread.join reader;
  equal ~msg:"the faulting copy raised Lost" bool true copy;
  equal ~msg:"the stops before the fork" int 0 before;
  equal string "exited 0" (status ended);
  equal (list string) [ "stops: 0"; "freed: 0" ] lines;
  Rig.close io;
  equal ~msg:"the parent's stop" int 1 (Atomic.get Store.stops - stops)

(* A device that takes the slot a closed io device left is a driver's device:
   a forked child sees it lost. A memory device's last value, which the io
   device's slot keeps and the Polled device starts from, shows that the slot
   passed from one to the next. *)
let test_slot_of_io () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let m = Result.get_ok (Rig.memory_device "fork:before-io") in
  let p = submit (empty m) in
  Rig.Point.wait p;
  Rig.close m;
  Rig.free_cache m;
  let io = open_store "fork:io-slot" in
  Rig.close io;
  Rig.free_cache io;
  let d, _ = P.open_ "fork:after-io" in
  equal ~msg:"the slot passed on" int (Rig.Point.value p) (Rig.submitted d);
  let lines, ended =
    in_child (fun () -> [ Option.value ~default:"not lost" (Rig.lost d) ])
  in
  equal string "exited 0" (status ended);
  equal (list string) [ "forked" ] lines

(* A buffer of [n] bytes on [d] that is unreachable once this returns: its
   address. *)
let[@inline never] dropped d n = Support.at (B.create d n)

(* A device a child opens is its own: memory it drops returns to its cache as in
   any process, while the devices it inherited stay lost. *)
let test_own_device () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let inherited, _ = P.open_ "fork:inherited" in
  let lines, ended =
    in_child (fun () ->
        let d, _ = P.open_ "fork:own" in
        let at = dropped d 4096 in
        Gc.full_major ();
        Gc.full_major ();
        ignore (B.create ~memory:Pinned d 8);
        [
          Option.value ~default:"not lost" (Rig.lost d);
          Printf.sprintf "reused: %b" (Support.at (B.create d 4096) = at);
          Option.value ~default:"not lost" (Rig.lost inherited);
        ])
  in
  equal string "exited 0" (status ended);
  equal (list string) [ "not lost"; "reused: true"; "forked" ] lines

(* A part that holds one unit of a Polled queue. *)
let one_part d =
  let fill = Support.noop in
  let s =
    Sub.make d
      [|
        Sub.fill fill ~params:0;
      |]
  in
  s

let submit_part s =
  let run = Sub.Run.make s in
  Sub.Run.groups run (Sub.block s 0) 0 0 0;
  Rig.submit run ~buffers:[||]

(* A thread of the parent blocked in a hand-over at the fork, holding its
   device's turn and its submission, with a profile being taken, leaves the
   child a working rig: devices the child opens submit, and the parent's are
   lost at once. A child that waits for what the thread holds is killed by its
   alarm. *)
let test_locks () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let io = open_store "fork:io-locked" in
  let d, p = P.open_ ~capacity:1 ~may_block:true "fork:blocked" in
  let s = one_part d in
  ignore (submit_part s);
  let blocked () =
    try ignore (Rig.Profile.take (fun () -> submit_part s))
    with Rig.Lost _ -> ()
  in
  let holder = Thread.create blocked () in
  Support.await "a blocked submit" (fun () -> P.blocked p = 1);
  let lines, ended =
    in_child (fun () ->
        Sys.set_signal Sys.sigalrm Sys.Signal_default;
        ignore (Unix.alarm 5);
        let made = B.length (B.create io 8) in
        let e, _ = P.open_ ~capacity:1 ~may_block:true "fork:opened-in-child" in
        let v = Rig.Point.value (submit_part (one_part e)) in
        Rig.Profile.span "child" (fun () -> ());
        [
          Printf.sprintf "bytes made: %d" made;
          Printf.sprintf "submitted: %d" v;
          Option.value ~default:"not lost" (Rig.lost d);
          Printf.sprintf "submit raises Lost: %b"
            (raises_lost (fun () -> submit_part s));
        ])
  in
  ignore (P.run p);
  Thread.join holder;
  equal string "exited 0" (status ended);
  equal (list string)
    [ "bytes made: 8"; "submitted: 1"; "forked"; "submit raises Lost: true" ]
    lines

(* In a forked child, a buffer over an inherited device's memory raises the
   device's loss, read by the host directly or through a borrow on the host: the
   child may not map that memory, as it maps no Metal buffer on macOS. *)
let test_child_memory () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let d, _ = P.open_ ~copies:false "fork:memory" in
  let m = B.create d 64 in
  let h = require_some (B.borrow Rig.host m) in
  let outcome f =
    match f () with
    | _ -> "returned"
    | exception Rig.Lost (d', why) when Rig.equal d d' -> why
  in
  let lines, ended =
    in_child (fun () ->
        let dst = B.create Rig.host 64 in
        [
          outcome (fun () -> B.wait h Rig.Read_write);
          outcome (fun () -> B.read h Bigarray.char ignore);
          outcome (fun () -> B.read m Bigarray.char ignore);
          outcome (fun () -> B.blit_to_bytes m 0 (Bytes.create 1) 0 1);
          outcome (fun () -> B.copy ~src:h ~dst);
          outcome (fun () -> B.borrow Rig.host m);
        ])
  in
  equal string "exited 0" (status ended);
  equal (list string) (List.init 6 (fun _ -> "forked")) lines;
  B.wait h Rig.Read_write

(* A drain on [d], also on a lost [d], whose {!B.create} then raises. *)
let drain d =
  match B.create d 8 with
  | b -> ignore (Sys.opaque_identity b)
  | exception Rig.Lost _ -> ()

(* A value whose collection counts in [runs]. *)
let counted runs =
  let v = ref () in
  Gc.finalise (fun _ -> Atomic.incr runs) v;
  v

(* Drops the submission [h] holds in a forked child, drains its device and the
   host there, and checks that its value is collected in the child, then in
   the parent once its work is done. *)
let collected_in_child d p runs h =
  let lines, ended =
    in_child (fun () ->
        h := None;
        Gc.full_major ();
        Gc.full_major ();
        drain d;
        drain Rig.host;
        Gc.full_major ();
        [ Printf.sprintf "collected: %d" (Atomic.get runs) ])
  in
  equal string "exited 0" (status ended);
  equal (list string) ~msg:"in the child" [ "collected: 1" ] lines;
  ignore (Sys.opaque_identity !h);
  h := None;
  ignore (P.run p);
  Gc.full_major ();
  Gc.full_major ();
  drain d;
  Gc.full_major ();
  equal ~msg:"in the parent" int 1 (Atomic.get runs)

(* A submission made and submitted before a fork keeps its value in the child
   no longer, though its work is not done. *)
let test_keep () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let runs = Atomic.make 0 in
  let d, p = P.open_ "fork:keep" in
  let h = ref (Some (Sub.make ~keep:(counted runs) d [||])) in
  ignore (submit (Option.get !h));
  collected_in_child d p runs h

(* A value whose submission the parent collected and drained before the fork,
   and whose work is not done, is kept in the child no longer. *)
let[@inline never] kept_and_drained d runs =
  ignore (submit (Sub.make ~keep:(counted runs) d [||]));
  Gc.full_major ();
  Gc.full_major ();
  drain d

let test_keep_drained () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let runs = Atomic.make 0 in
  let d, p = P.open_ "fork:keep-drained" in
  kept_and_drained d runs;
  equal ~msg:"in the parent before the fork" int 0 (Atomic.get runs);
  collected_in_child d p runs (ref None)

(* A child gives back nothing of a word its parent's devices hold: neither a
   consumer's mapping of a producer's word of its own nor the word, of a
   producer whose slot was freed before the fork. The parent gives the word
   back once, after its consumer closed. *)
let test_word_maps () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let consumer, c = P.open_ ~waits_on:[ `Host ] "fork:word-consumer" in
  let d, p = P.open_ ~transport:true "fork:word-producer" in
  let q = submit (empty d) in
  let w = submit ~waits:[| q |] (empty consumer) in
  equal ~msg:"the consumer's waits in its queue" int 1
    (List.length (P.last_waits c));
  ignore (P.run p);
  Rig.Point.wait w;
  Rig.close d;
  Rig.free_cache d;
  let since calls p =
    String.concat " " (List.filteri (fun i _ -> i >= calls) (P.log p))
  in
  let calls_c = List.length (P.log c) and calls_p = List.length (P.log p) in
  let lines, ended =
    in_child (fun () ->
        Rig.free_cache d;
        Rig.free_cache consumer;
        Rig.close consumer;
        Gc.full_major ();
        Rig.free_cache consumer;
        [
          Printf.sprintf "consumer calls: %s" (since calls_c c);
          Printf.sprintf "producer calls: %s" (since calls_p p);
        ])
  in
  equal string "exited 0" (status ended);
  equal (list string) [ "consumer calls: "; "producer calls: " ] lines;
  equal ~msg:"the word's frees while the consumer is open" int 0
    (P.word_frees p);
  Rig.close consumer;
  equal ~msg:"the word's frees once the consumer closed" int 1 (P.word_frees p);
  equal ~msg:"Polled's violations" (list string) [] (P.violations ())

let tests =
  [
    group ~timeout "fork"
      [
        test "a forked child's devices are lost for good" test_child;
        test "a forked child's io devices stay its library's" test_io_child;
        test "a forked child loses a device on a closed io device's slot"
          test_slot_of_io;
        test "a forked child frees the io memory it drops" test_io_loop;
        test "a forked child frees io memory a parent's device uses"
          test_io_used;
        test "a forked child frees the memory of an io device closed before"
          test_io_closed;
        test "a forked child runs no stop its parent owed" test_io_owed;
        test "a device a forked child opens drains as in any process"
          test_own_device;
        test "a forked child uses rig while a parent's thread held a turn"
          test_locks;
        test "a forked child's host borrow of a lost device's memory is lost"
          test_child_memory;
        test "a submission made before a fork keeps nothing in the child"
          test_keep;
        test "a kept value drained before a fork is let go in the child"
          test_keep_drained;
        test "a forked child gives back no word its parent's devices hold"
          test_word_maps;
      ];
  ]

let () = exit (run "rig.fork" tests)
