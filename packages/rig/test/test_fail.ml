(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A process's failure. A failed process stays failed, so each case runs in a
   forked child, and the case with two domains, after which the process cannot
   fork, runs last in this one. Its own suite: a process that ran a domain
   cannot fork. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled

let timeout = 60.
let empty d = Sub.make d [||]

let submit ?waits s = Rig.submit ?waits s ~run:(Sub.Run.make ()) ~buffers:[||]

let count call p = List.length (List.filter (( = ) call) (P.log p))
let failure () = Option.value ~default:"none" (Rig.failure ())

let status = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> "killed by " ^ Sys.signal_to_string n
  | Unix.WSTOPPED n -> "stopped by " ^ Sys.signal_to_string n

(* Runs [child] in a forked child and is what it returned: the child writes its
   answer to a pipe and leaves without running this process's exit. *)
let in_child child =
  let r, w = Unix.pipe () in
  Rig_support.needs_fork ();
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

let child_case expected child () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let lines, ended = in_child child in
  equal string "exited 0" (status ended);
  equal (list string) expected lines

(* A release from C with no read claim to end aborts the process with its
   reason, which the child writes to its stderr. *)
let test_release_unclaimed () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let b = B.create Rig.host 8 in
  let r, w = Unix.pipe () in
  Rig_support.needs_fork ();
  match Unix.fork () with
  | 0 ->
      Unix.dup2 w Unix.stderr;
      Rig_support.Reader.release b;
      Unix._exit 0
  | pid ->
      Unix.close w;
      let ic = Unix.in_channel_of_descr r in
      let err = In_channel.input_all ic in
      close_in ic;
      let _, ended = Unix.waitpid [] pid in
      equal string (status (Unix.WSIGNALED Sys.sigabrt)) (status ended);
      equal string "Fatal error: rig_buffer_release: no read claim to end\n" err

(* Loses [d] through a failed hand-over. *)
let lose d p =
  P.fail p;
  try ignore (submit (empty d)) with Rig.Lost _ -> ()

(* A close is no failure; a loss is, as Lost prints it. *)
let test_close_no_failure =
  child_case [ "none"; "none"; "fail:lost lost: the submission failed" ]
  @@ fun () ->
  let before = failure () in
  let c, _ = P.open_ "fail:closed" in
  Rig.close c;
  let closed = failure () in
  let d, p = P.open_ "fail:lost" in
  lose d p;
  [ before; closed; failure () ]

(* The first loss stays the failure, through later losses and a fail. *)
let test_first_loss =
  child_case [ "fail:first lost: the submission failed" ] @@ fun () ->
  let d, p = P.open_ "fail:first" and e, q = P.open_ "fail:second" in
  lose d p;
  lose e q;
  Rig.fail "the world failed";
  [ failure () ]

(* A fail that comes first is the failure, with its reason alone. *)
let test_fail_first =
  child_case [ "the world failed" ] @@ fun () ->
  let d, p = P.open_ "fail:after" in
  Rig.fail "the world failed";
  lose d p;
  [ failure () ]

(* An exception other than a fault escaping a driver's counted call loses the
   device with the exception's constructor's name, and is the process's
   failure; the call raises it, and the device stops once. [prepare]
   readies the call, and is the call: the exception comes at its [call]. *)
let escaping call prepare =
  let name = "fail:escaping-" ^ call in
  child_case
    [
      "raised Stdlib.Exit";
      "Stdlib.Exit";
      "stops: 1";
      name ^ " lost: Stdlib.Exit";
    ]
  @@ fun () ->
  let d, p = P.open_ name in
  let run = prepare d in
  P.before p call (fun () -> raise Exit);
  let raised =
    match run () with
    | () -> "returned"
    | exception e -> "raised " ^ Printexc.to_string e
  in
  [
    raised;
    Option.value ~default:"not lost" (Rig.lost d);
    Printf.sprintf "stops: %d" (P.stops p);
    failure ();
  ]

(* What collected memory gives back to [d]'s driver, from a buffer or an image
   made here and dropped. *)
let[@inline never] dropped make d = ignore (Sys.opaque_identity (make d))

let returning d () =
  Gc.full_major ();
  Rig.free_cache d

let escapes =
  [
    ("alloc", fun d () -> ignore (B.create d 8));
    ( "free",
      fun d ->
        dropped (fun d -> B.create d 64) d;
        returning d );
    ( "map_host",
      fun d ->
        let h = B.create Rig.host (1 lsl 16) in
        fun () -> ignore (B.borrow d h) );
    ( "map_peer",
      fun d ->
        let other, _ = P.open_ "fail:escaping-peer" in
        let m = B.create other 64 in
        fun () -> ignore (B.borrow d m) );
    ("image", fun d () -> ignore (Rig.Image.load d "functions"));
    ("place", fun d () -> ignore (Rig.Image.load d "code:64"));
    ( "entry",
      fun d ->
        let i = Result.get_ok (Rig.Image.load d "functions") in
        fun () -> ignore (Rig.Image.mem i "fill") );
    ( "unload",
      fun d ->
        dropped (fun d -> Result.get_ok (Rig.Image.load d "functions")) d;
        returning d );
  ]

(* An io library whose memory holds nothing. *)
module Nothing = struct
  type t = unit
  type region = unit

  exception Fault of string

  let region_key : region Type.Id.t = Type.Id.make ()
  let budget () = max_int
  let alloc () _ = Some ()
  let free () () = ()
  let read () () ~at:_ ~dst:_ ~len:_ = ()
  let write () () ~at:_ ~src:_ ~len:_ = ()
  let pages () () = None
  let prefetch () () ~at:_ ~len:_ = ()
  let stop () = ()
end

(* A fail loses every device but the host, each stopped once; a device already
   lost keeps its reason; the host's memory goes on; an open afterwards answers
   the reason without calling its opener, also of a name that was open, an io
   open too, and a memory device's; and a second fail changes nothing. *)
let test_fail =
  child_case
    [
      "the world failed";
      "stops: 1";
      "the submission failed";
      "the world failed";
      "host bytes: 8";
      "Error the world failed";
      "Error the world failed";
      "Error the world failed";
      "Error the world failed";
      "opener calls: 0";
      "fail:lost-before lost: the submission failed";
    ]
  @@ fun () ->
  let d, p = P.open_ "fail:device" and e, q = P.open_ "fail:lost-before" in
  lose e q;
  let io = Rig_support.machine "fail-machine" in
  Rig.fail "the world failed";
  Rig.fail "a second failure";
  let calls = ref 0 in
  let open_ name =
    match
      Rig.open_
        {
          name;
          driver = (module P);
          make =
            (fun ~word ->
              incr calls;
              Ok (P.bind (P.make ()) ~word));
        }
    with
    | Ok _ -> "Ok"
    | Error why -> "Error " ^ why
  in
  let open_io name =
    let make () =
      incr calls;
      Ok ()
    in
    match Rig.open_io { name; io = (module Nothing); make } with
    | Ok _ -> "Ok"
    | Error why -> "Error " ^ why
  in
  let memory_device name =
    match Rig.memory_device name with
    | Ok _ -> "Ok"
    | Error why -> "Error " ^ why
  in
  [
    Option.value ~default:"not lost" (Rig.lost d);
    Printf.sprintf "stops: %d" (count "stop" p);
    Option.value ~default:"not lost" (Rig.lost e);
    Option.value ~default:"not lost" (Rig.lost io);
    Printf.sprintf "host bytes: %d" (B.length (B.create Rig.host 8));
    open_ "fail:new";
    open_ "fail:device";
    open_io "fail:io";
    memory_device "fail:memory";
    Printf.sprintf "opener calls: %d" !calls;
    failure ();
  ]

(* A device a fail loses through its waits on another keeps the fail's reason,
   however deep the chain of waits. *)
let test_fail_spread =
  child_case [ "the world failed"; "the world failed"; "the world failed" ]
  @@ fun () ->
  let a, _ = P.open_ "fail:producer" in
  let b, _ = P.open_ ~waits_on:[ `Host ] "fail:middle" in
  let c, _ = P.open_ ~waits_on:[ `Host ] "fail:consumer" in
  let on_a = submit (empty a) in
  let on_b = submit (empty b) ~waits:[| on_a |] in
  ignore (submit (empty c) ~waits:[| on_b |]);
  Rig.fail "the world failed";
  List.map (fun d -> Option.value ~default:"not lost" (Rig.lost d)) [ a; b; c ]

(* Opens race a fail on another domain: every device that opened is lost with
   its reason and stopped once, and every open that answered after the fail
   returned is an error. *)
let test_fail_beside_opens () =
  let opened = Atomic.make [] and failed = Atomic.make false in
  let late = Atomic.make 0 in
  let opener =
    Domain.spawn (fun () ->
        let rec go i =
          let after = Atomic.get failed in
          let name = Printf.sprintf "fail:race-%d" i in
          let p = P.make () in
          match
            Rig.open_ (P.opener name p)
          with
          | Ok d ->
              if after then Atomic.incr late;
              Atomic.set opened ((d, p) :: Atomic.get opened);
              go (i + 1)
          | Error _ -> ()
        in
        go 0)
  in
  while List.length (Atomic.get opened) < 8 do
    Domain.cpu_relax ()
  done;
  Rig.fail "the world failed";
  Atomic.set failed true;
  Domain.join opener;
  equal ~msg:"opens that succeeded once fail returned" int 0 (Atomic.get late);
  List.iter
    (fun (d, p) ->
      equal (option string) (Some "the world failed") (Rig.lost d);
      equal int 1 (count "stop" p))
    (Atomic.get opened)

let tests =
  [
    group ~timeout "an exception escaping a driver"
      (List.map
         (fun (call, prepare) ->
           test
             (call ^ ": loses the device with its text, and is the failure")
             (escaping call prepare))
         escapes);
    group ~timeout "fail"
      [
        test "a close is no failure, a loss is" test_close_no_failure;
        test "the first loss stays the failure" test_first_loss;
        test "a fail that comes first is the failure" test_fail_first;
        test "a fail loses every device and refuses every open" test_fail;
        test "a device lost through its waits keeps the fail's reason"
          test_fail_spread;
        test "a release from C with no claim aborts the process"
          test_release_unclaimed;
        test "opens beside a fail end lost or refused" test_fail_beside_opens;
      ];
  ]

let () = exit (run "rig.fail" tests)
