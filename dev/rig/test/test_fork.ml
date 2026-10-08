(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A forked child of a process that opened devices. Its own suite: a process
   that ran a domain cannot fork. *)

open Windtrap
module C = Rig
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(reads = [||]) ?(writes = [||]) ?(waits = [||]) s =
  C.submit s ~reads ~writes ~waits

let timeout = 60.
let empty d = Sub.make ~reads:0 ~writes:0 d [||]
let raises_lost f = match f () with _ -> false | exception C.Lost _ -> true

let status = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> Printf.sprintf "killed by signal %d" n
  | Unix.WSTOPPED n -> Printf.sprintf "stopped by signal %d" n

(* Runs [child] in a forked child and is what it returned: the child writes its
   answer to a pipe and leaves without running this process's exit. *)
let in_child child =
  let r, w = Unix.pipe () in
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
  let v = C.Point.value (submit (empty d)) in
  let calls = P.log p in
  let lines, ended =
    in_child (fun () ->
        let lost = C.lost d in
        let submit = raises_lost (fun () -> submit (empty d)) in
        let wait = raises_lost (fun () -> C.wait d v) in
        b := None;
        Gc.full_major ();
        ignore (B.create C.host 8);
        P.set_word p v;
        Gc.full_major ();
        ignore (B.create C.host 8);
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
  equal (option string) None (C.lost d);
  C.wait d v;
  ignore (Sys.opaque_identity !b)

let page_bytes = 1 lsl 16

(* An io device whose memory is host bytes on a page, whose state its library
   holds, and which counts the regions it holds. *)
module Store = struct
  type t = unit

  (* A host buffer of at least a page, which starts on one. *)
  type region = B.t

  exception Fault of string

  let held = Atomic.make 0
  let region_key : region Type.Id.t = Type.Id.make ()
  let budget () = max_int

  let alloc () n =
    Atomic.incr held;
    Some (B.create C.host (Int.max n page_bytes))

  let free () _ = Atomic.decr held
  let read () _ ~at:_ ~dst:_ ~len:_ = ()
  let write () _ ~at:_ ~src:_ ~len:_ = ()
  let pages () r = Some (B.bigarray Bigarray.char r)
  let prefetch () _ ~at:_ ~len:_ = ()
  let stop () = ()
end

let open_store name =
  require_ok ~pp:Format.pp_print_string
    (C.open_io (module Store) ~name (fun () -> Ok ()))

(* An io device's state is its library's, which decides what a fork does to it:
   rig leaves it usable in the child, where a driver's device is lost. *)
let test_io_child () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let io = open_store "fork:io" in
  let d, _ = P.open_ "fork:beside-io" in
  let lines, ended =
    in_child (fun () ->
        [
          Option.value ~default:"not lost" (C.lost io);
          Printf.sprintf "bytes made: %d" (B.length (B.create io 8));
          Option.value ~default:"not lost" (C.lost d);
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
  let s = Sub.make ~reads:1 ~writes:0 d [||] in
  ignore (submit s ~reads:[| require_some (B.borrow d (Option.get !m)) |]);
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

(* A buffer of [n] bytes on [d] that is unreachable once this returns: its
   address. *)
let[@inline never] dropped d n = B.address (B.create d n)

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
          Option.value ~default:"not lost" (C.lost d);
          Printf.sprintf "reused: %b" (B.address (B.create d 4096) = at);
          Option.value ~default:"not lost" (C.lost inherited);
        ])
  in
  equal string "exited 0" (status ended);
  equal (list string) [ "not lost"; "reused: true"; "forked" ] lines

(* A thread of the parent inside every lock of rig at the fork, with a
   profile being taken, leaves the child a working rig: the child makes each
   lock anew. A child that still waits for one is killed by its alarm. *)
let test_locks () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let io = open_store "fork:io-locked" in
  let lock = Mutex.create () and cond = Condition.create () in
  let inside = ref false and leave = ref false in
  let hold () =
    Support.locked (fun () ->
        Mutex.protect lock (fun () ->
            inside := true;
            Condition.broadcast cond;
            while not !leave do
              Condition.wait cond lock
            done))
  in
  let holder = Thread.create (fun () -> ignore (C.Profile.take hold)) () in
  Mutex.protect lock (fun () ->
      while not !inside do
        Condition.wait cond lock
      done);
  let lines, ended =
    in_child (fun () ->
        Sys.set_signal Sys.sigalrm Sys.Signal_default;
        ignore (Unix.alarm 5);
        let made = B.length (B.create io 8) in
        let d, _ = P.open_ "fork:opened-in-child" in
        C.Profile.span "child" (fun () -> ());
        [
          Printf.sprintf "bytes made: %d" made;
          Option.value ~default:"opened" (C.lost d);
          "span recorded";
        ])
  in
  Mutex.protect lock (fun () ->
      leave := true;
      Condition.broadcast cond);
  Thread.join holder;
  equal string "exited 0" (status ended);
  equal (list string) [ "bytes made: 8"; "opened"; "span recorded" ] lines

let tests =
  [
    group ~timeout "fork"
      [
        test "a forked child's devices are lost for good" test_child;
        test "a forked child's io devices stay its library's" test_io_child;
        test "a forked child frees the io memory it drops" test_io_loop;
        test "a forked child frees io memory a parent's device uses"
          test_io_used;
        test "a device a forked child opens drains as in any process"
          test_own_device;
        test "a forked child uses rig while a parent's thread held its locks"
          test_locks;
      ];
  ]

let () = exit (run "rig.fork" tests)
