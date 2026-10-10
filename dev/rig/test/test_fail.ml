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
let submit ?(waits = [||]) s =
  Rig.submit s ~run:(Sub.Run.make ()) ~buffers:[||] ~waits
let count call p = List.length (List.filter (( = ) call) (P.log p))
let failure () = Option.value ~default:"none" (Rig.failure ())

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

let child_case expected child () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let lines, ended = in_child child in
  equal string "exited 0" (status ended);
  equal (list string) expected lines

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

(* A fail loses every device but the host, each stopped once; a device already
   lost keeps its reason; the host's memory goes on; an open afterwards answers
   the reason without calling its opener, also of a name that was open; and a
   second fail changes nothing. *)
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
      Rig.open_ (module P) ~name (fun () ->
          incr calls;
          Ok (P.make ()))
    with
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
          match Rig.open_ (module P) ~name (fun () -> Ok p) with
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
    group ~timeout "fail"
      [
        test "a close is no failure, a loss is" test_close_no_failure;
        test "the first loss stays the failure" test_first_loss;
        test "a fail that comes first is the failure" test_fail_first;
        test "a fail loses every device and refuses every open" test_fail;
        test "a device lost through its waits keeps the fail's reason"
          test_fail_spread;
        test "opens beside a fail end lost or refused" test_fail_beside_opens;
      ];
  ]

let () = exit (run "rig.fail" tests)
