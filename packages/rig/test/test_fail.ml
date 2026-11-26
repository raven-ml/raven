(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Errors and a process's failure. A failed process stays failed, so each case
   that loses a device runs in a forked child, which answers with lines that the
   test compares. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let empty d = Sub.make d [||]
let submit ?waits s = Rig.submit ?waits s ~buffers:[||] ~args:[||]
let count call p = List.length (List.filter (( = ) call) (P.log p))
let failure () = Option.value ~default:"none" (Rig.failure ())
let reason d = Option.value ~default:"not lost" (Rig.lost d)

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
      Fun.protect ~finally:(fun () -> close_in ic) @@ fun () ->
      Support.reaping pid @@ fun () ->
      let lines = In_channel.input_lines ic in
      let _, status = Unix.waitpid [] pid in
      (lines, status)

let child_case expected child () =
  let lines, ended = in_child child in
  equal string "exited 0" (status ended);
  equal (list string) expected lines

(* [child_case], but on Windows, which has no fork, [child] runs in this
   process: every later case that checks the failure is a child case, which
   skips there. *)
let isolated expected child () =
  if Sys.win32 then equal (list string) expected (child ())
  else child_case expected child ()

(* A walk's child answers a line per run, which [line n] states for its [n]-th;
   it lands at least once. *)
let walk_case line child () =
  let lines, ended = in_child child in
  equal string "exited 0" (status ended);
  equal (list string)
    (List.init (List.length lines) (fun n -> line (n + 1)))
    lines;
  greater ~msg:"runs" int ~than:1 (List.length lines)

(* Loses [d] through a failed hand-over. *)
let lose d p =
  P.fail p;
  try ignore (submit (empty d)) with Rig.Lost _ -> ()

(* Errors and failure *)

(* Windtrap's [raises] lets [Sys.Break] through, as it ends a run. *)
let broke f =
  Sys.catch_break true;
  Fun.protect
    ~finally:(fun () -> Sys.catch_break false)
    (fun () ->
      match f () with _ -> "returned" | exception Sys.Break -> "Sys.Break")

(* A submission of one part that holds one unit of a Polled queue. *)
let one_part d =
  Rig.submit
    (Sub.make d [| Sub.fill Support.noop ~params:0 |])
    ~buffers:[||] ~args:(Support.zeros 1)

(* After [Sys.Break] the device is used as before, and once lost its stop runs
   once at its close: the interrupted call left nothing counted in flight.
   [last] is its last point. *)
let still_works d p ~last =
  let lost = reason d in
  let past = Rig.submitted d - Rig.Point.value last in
  Rig.Point.wait last;
  let next = Rig.Point.value (submit (empty d)) - Rig.Point.value last in
  P.fail p;
  let failed =
    match submit (empty d) with
    | _ -> "returned"
    | exception Rig.Lost _ -> "Lost"
  in
  Rig.close d;
  [
    lost;
    Printf.sprintf "values past the last: %d" past;
    Printf.sprintf "the next value: the last + %d" next;
    "a failed submit: " ^ failed;
    Printf.sprintf "stops: %d" (count "stop" p);
  ]

let works =
  [
    "not lost";
    "values past the last: 0";
    "the next value: the last + 1";
    "a failed submit: Lost";
    "stops: 1";
  ]

(* A Ctrl-C during a driver's wait raises inside it: its handler runs in the
   sleep it came in. *)
let test_break_ends_sleep =
  isolated ([ "the wait: Sys.Break"; "sleeps the signal ended: 1" ] @ works)
  @@ fun () ->
  let d, p = P.open_ "fail:break-ends-sleep" in
  let first = one_part d in
  P.interrupt p;
  let wait = broke (fun () -> Rig.Point.wait first) in
  ("the wait: " ^ wait)
  :: Printf.sprintf "sleeps the signal ended: %d" (P.broken_sleeps p)
  :: still_works d p ~last:first

(* After a Ctrl-C during a driver's call outside its waits, the call finished
   and the device works. On Windows nothing blocks the signal: its Sys.Break
   raises at the hook's next poll point, inside the driver's allocation, and
   loses the device. *)
let ended_call d p ~finished ~last =
  let finished =
    Printf.sprintf "the driver's allocation finished: %b" finished
  in
  if Sys.win32 then begin
    let why = reason d in
    Rig.close d;
    [ finished; why; Printf.sprintf "stops: %d" (P.stops p) ]
  end
  else finished :: still_works d p ~last

let call_ended =
  if Sys.win32 then
    [
      "the call: Sys.Break";
      "the driver's allocation finished: false";
      "Stdlib.Sys.Break";
      "stops: 1";
    ]
  else
    "the call: Sys.Break" :: "the driver's allocation finished: true" :: works

(* Outside Windows, a Ctrl-C while a driver's call runs outside its waits lands
   nowhere in the call: the signal comes at the start of the driver's
   allocation, whose hook then allocates, where the handler of a signal the
   thread takes runs. The allocation finishes, the device stays live, and
   Sys.Break raises as the call returns, at its first poll point after the
   driver.

   [`Caller] raises the signal in the calling thread, which blocks it until the
   call gives its mask back. [`Other] raises it in a thread of no OCaml code,
   which takes it: the runtime records it while the calling thread blocks it,
   and the hook's poll points skip it, the join's included. The minor collection
   before it leaves the call no collection, whose poll would look at the
   recorded signals again. *)
let test_break_in_call taker () =
  if taker = `Other && Sys.win32 then
    skip ~reason:"Windows has no SIGINT from a thread" ();
  let name = match taker with `Caller -> "caller" | `Other -> "other" in
  isolated call_ended
    (fun () ->
      let d, p = P.open_ ("fail:break-in-call-" ^ name) in
      let first = one_part d in
      let finished = ref false in
      P.before p "alloc" (fun () ->
          (match taker with
          | `Caller -> Support.interrupt ()
          | `Other ->
              Gc.minor ();
              Support.interrupt_after 0.;
              Support.interrupt_join ());
          ignore (Sys.opaque_identity (List.init 64 Fun.id));
          finished := true);
      let call = broke (fun () -> B.create d 8) in
      ("the call: " ^ call) :: ended_call d p ~finished:!finished ~last:first)
    ()

(* Outside Windows, a Ctrl-C during a wait of a driver's finaliser, which runs
   shielded inside the driver's allocation, takes nothing there: the allocation
   finishes, the device stays live, and Sys.Break raises as the call returns. *)
let test_break_in_shielded_wait =
  isolated call_ended @@ fun () ->
  let d, p = P.open_ "fail:break-shielded" in
  let first = one_part d in
  let finished = ref false in
  P.before p "alloc" (fun () ->
      Rig_edge_primitive.shield ();
      Rig_edge_primitive.unblock ();
      Support.interrupt ();
      Thread.delay 0.;
      Rig_edge_primitive.block ();
      Rig_edge_primitive.unshield ();
      finished := true);
  let call = broke (fun () -> B.create d 8) in
  ("the call: " ^ call) :: ended_call d p ~finished:!finished ~last:first

(* Sys.Break landing in a driver's read of a word of its own, which a wait
   makes, loses nothing: rig blocks no signal across a read of a word. *)
let test_break_in_signaled =
  isolated ("the wait: Sys.Break" :: works) @@ fun () ->
  let d, p = P.open_ ~transport:true "fail:break-signaled" in
  let first = one_part d in
  P.break_word p;
  let wait = broke (fun () -> Rig.Point.wait first) in
  ("the wait: " ^ wait) :: still_works d p ~last:first

(* The signals rig leaves unblocked across a driver's call: those a fault
   raises. *)
let faults = Sys.[ sigsegv; sigbus; sigfpe; sigill; sigtrap; sigabrt; sigsys ]

(* Signals no fault raises, which a process can block, on every system with
   signal masks. *)
let blockable =
  Sys.
    [
      sighup;
      sigint;
      sigquit;
      sigterm;
      sigusr1;
      sigusr2;
      sigpipe;
      sigalrm;
      sigchld;
      sigcont;
      sigtstp;
      sigttin;
      sigttou;
      sigvtalrm;
      sigprof;
      sigurg;
      sigxcpu;
      sigxfsz;
    ]

(* The signals of [among] in the calling thread's mask. *)
let masked among =
  let mask = Thread.sigmask Unix.SIG_BLOCK [] in
  List.filter (fun s -> List.mem s mask) among

let test_mask_in_call () =
  if Sys.win32 then skip ~reason:"Windows has no signal masks" ();
  let d, p = P.open_ "fail:mask" in
  let inside = ref ([], []) in
  P.before p "alloc" (fun () -> inside := (masked faults, masked blockable));
  ignore (B.create d 8);
  let faults_masked, others_masked = !inside in
  equal ~msg:"fault signals blocked" (list int) [] faults_masked;
  equal ~msg:"other signals blocked" (list int) blockable others_masked

(* A fault of a counted call loses its device, whatever exception lands at any
   allocation after it: the device is never left live, and its stop runs once,
   by the close. *)
let test_fault_interrupted =
  walk_case (Printf.sprintf "allocation %d: the engine hung, stops 1")
  @@ fun () ->
  let rec from n lines =
    let d, p = P.open_ (Printf.sprintf "fail:fault-interrupted-%d" n) in
    (* Made before the hook, so the landing is after the fault. *)
    let faulted = ref false and fault = P.Fault "the engine hung" in
    P.before p "alloc" (fun () ->
        faulted := true;
        raise fault);
    let alloc () =
      match B.create d 8 with
      | _ -> failwith "the faulted call returned"
      | exception Rig.Lost _ -> ()
    in
    let reached = Support.interrupted ~from:(fun _ -> !faulted) n alloc in
    let why = reason d in
    Rig.close d;
    let line = Printf.sprintf "allocation %d: %s, stops %d" n why (P.stops p) in
    if reached then from (n + 1) (line :: lines) else List.rev (line :: lines)
  in
  from 1 []

(* A wait that loses its device to the hang bound, with an exception at any
   allocation from the loss on, leaves the device's stop to run once, by the
   close. *)
let test_hang_interrupted =
  walk_case (Printf.sprintf "allocation %d: no progress for 20 ms, stops 1")
  @@ fun () ->
  let rec from n lines =
    let d, p =
      P.open_ ~hang_ms:20 (Printf.sprintf "fail:hang-interrupted-%d" n)
    in
    let first = submit (empty d) in
    P.stall p max_int;
    let wait () =
      match Rig.Point.wait first with
      | () -> failwith "the stalled work completed"
      | exception Rig.Lost (d', "no progress for 20 ms") when Rig.equal d d' ->
          ()
    in
    let reached =
      Support.interrupted ~from:(fun _ -> Rig.lost d <> None) n wait
    in
    Rig.close d;
    let line =
      Printf.sprintf "allocation %d: %s, stops %d" n (reason d) (P.stops p)
    in
    if reached then from (n + 1) (line :: lines) else List.rev (line :: lines)
  in
  from 1 []

(* A Ctrl-C landing at any poll point after an exception escaped a driver leaves
   the stop the loss owes to run once, by the device's close. Backtraces are
   recorded, as OCAMLRUNPARAM=b records them: the handler's fetch of one
   allocates in C, whose sample's callback, and so the signal, comes at a later
   poll point. *)
let test_break_after_escape =
  walk_case (Printf.sprintf "allocation %d: Stdlib.Exit, stops 1") @@ fun () ->
  Printexc.record_backtrace true;
  let rec from n lines =
    let d, p = P.open_ (Printf.sprintf "fail:break-after-escape-%d" n) in
    let escaped = ref false in
    P.before p "alloc" (fun () ->
        escaped := true;
        raise Exit);
    let alloc () =
      match B.create d 8 with
      | _ -> failwith "the allocation returned"
      | exception Exit -> ()
    in
    let reached =
      Support.interrupted
        ~from:(fun _ -> !escaped)
        ~landing:Support.Signal n alloc
    in
    let why = reason d in
    Rig.close d;
    let line = Printf.sprintf "allocation %d: %s, stops %d" n why (P.stops p) in
    if reached then from (n + 1) (line :: lines) else List.rev (line :: lines)
  in
  from 1 []

(* An exception other than a fault escaping a driver's change loses the device
   with the exception's text, and the call raises it again; the device stops
   once, at its close. [prepare d] readies the call and is the call; the
   exception comes at the driver's function [call]. *)
let escaping (call, prepare) =
  let name = "fail:escaping-" ^ call in
  child_case
    [ "raised Stdlib.Exit"; "Stdlib.Exit"; "stops: 0"; "stops: 1" ]
    (fun () ->
      let d, p = P.open_ name in
      let run = prepare d in
      P.before p call (fun () -> raise Exit);
      let raised =
        match run () with
        | () -> "returned"
        | exception e -> "raised " ^ Printexc.to_string e
      in
      let why = reason d in
      let before = Printf.sprintf "stops: %d" (P.stops p) in
      Rig.close d;
      [ raised; why; before; Printf.sprintf "stops: %d" (P.stops p) ])
    ()

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
        let h = B.create Rig.host B.paged_from in
        fun () -> ignore (B.borrow d h) );
    ( "map_peer",
      fun d ->
        let other, _ = P.open_ "fail:escaping-peer" in
        let m = B.create other 64 in
        fun () -> ignore (B.borrow d m) );
    (* Reclamation that another device's call runs: the host's memory returns,
       and with it [d]'s mapping of it. *)
    ( "unmap",
      fun d ->
        dropped (fun d -> B.borrow d (B.create Rig.host B.paged_from)) d;
        returning Rig.host );
    ("image", fun d () -> ignore (Rig.Image.load d "functions"));
    ("place", fun d () -> ignore (Rig.Image.load d "code:64"));
    ( "entry",
      fun d ->
        let i = Result.get_ok (Rig.Image.load d "functions") in
        fun () -> ignore (Rig.Image.resolve i "fill") );
    ( "unload",
      fun d ->
        dropped (fun d -> Result.get_ok (Rig.Image.load d "functions")) d;
        returning d );
  ]

let test_escape_failure =
  child_case [ "fail:escape-failure lost: Stdlib.Exit" ] @@ fun () ->
  let d, p = P.open_ "fail:escape-failure" in
  P.before p "alloc" (fun () -> raise Exit);
  (try ignore (B.create d 8) with Exit -> ());
  [ failure () ]

(* rig.h: a release from C with no read claim of the caller's to end aborts the
   process, so it runs in a forked child whose stderr comes back on a pipe. *)
let test_release_unclaimed () =
  let b = B.create Rig.host 8 in
  let r, w = Unix.pipe () in
  Support.needs_fork ();
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      Unix.dup2 w Unix.stderr;
      Support.Reader.release b;
      Unix._exit 0
  | pid ->
      Unix.close w;
      let ic = Unix.in_channel_of_descr r in
      Fun.protect ~finally:(fun () -> close_in ic) @@ fun () ->
      Support.reaping pid @@ fun () ->
      let err = In_channel.input_all ic in
      let _, ended = Unix.waitpid [] pid in
      equal string (status (Unix.WSIGNALED Sys.sigabrt)) (status ended);
      equal string "Fatal error: rig_buffer_release: no read claim to end\n" err

(* fail *)

let test_fail_loses =
  child_case
    [
      "the world failed";
      "the submission failed";
      "the world failed";
      "host bytes: 8";
    ]
  @@ fun () ->
  let d, _ = P.open_ "fail:device" and e, q = P.open_ "fail:lost-before" in
  lose e q;
  let io = Support.machine "fail-machine" in
  Rig.fail "the world failed";
  [
    reason d;
    reason e;
    reason io;
    Printf.sprintf "host bytes: %d" (B.length (B.create Rig.host 8));
  ]

let test_fail_no_stop =
  child_case [ "stops: 0"; "stops: 1" ] @@ fun () ->
  let d, p = P.open_ "fail:no-stop" in
  Rig.fail "the world failed";
  let before = Printf.sprintf "stops: %d" (count "stop" p) in
  Rig.close d;
  [ before; Printf.sprintf "stops: %d" (count "stop" p) ]

(* An open after a fail of a new name, of a name that was open, an io open, a
   software device's and a machine's host. *)
let test_fail_refuses_opens =
  child_case
    [
      "open_: Error the world failed";
      "open_ of an open name: Error the world failed";
      "open_io: Error the world failed";
      "software: Error the world failed";
      "open_host: Error the world failed";
      "opener calls: 0";
    ]
  @@ fun () ->
  ignore (P.open_ "fail:was-open");
  Rig.fail "the world failed";
  let calls = ref 0 in
  let counted =
    {
      Rig_edge.name = "";
      driver = (module P);
      make =
        (fun ~word ->
          incr calls;
          Ok (P.bind (P.make ()) ~word));
    }
  in
  let answer what = function
    | Ok _ -> what ^ ": Ok"
    | Error why -> what ^ ": Error " ^ why
  in
  (* Bound in turn, so the opener's count is read after every open. *)
  let fresh = answer "open_" (Rig.open_ { counted with name = "fail:new" }) in
  let was_open =
    answer "open_ of an open name"
      (Rig.open_ { counted with name = "fail:was-open" })
  in
  let io = answer "open_io" (Rig.open_io "fail:io") in
  let software = answer "software" (Rig.software "fail:memory") in
  let host =
    answer "open_host"
      (Rig.open_host ~machine:"fail-host" { counted with name = "CPU" })
  in
  [
    fresh;
    was_open;
    io;
    software;
    host;
    Printf.sprintf "opener calls: %d" !calls;
  ]

let test_fail_once =
  child_case [ "the world failed"; "the world failed" ] @@ fun () ->
  let d, _ = P.open_ "fail:once" in
  Rig.fail "the world failed";
  Rig.fail "a second failure";
  [ reason d; failure () ]

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
  List.map reason [ a; b; c ]

(* Opens race a fail on another domain: every device that opened is lost with
   its reason and stopped once, at its close, and every open that answered after
   the fail returned is an error. *)
let test_fail_beside_opens () =
  let opened = Atomic.make [] and failed = Atomic.make false in
  let late = Atomic.make 0 in
  let opener =
    Domain.spawn (fun () ->
        let rec go i =
          let after = Atomic.get failed in
          let name = Printf.sprintf "fail:race-%d" i in
          let p = P.make () in
          match Rig.open_ (P.opener name p) with
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
      equal ~msg:"stops before the close" int 0 (count "stop" p);
      Rig.close d;
      equal ~msg:"stops after the close" int 1 (count "stop" p))
    (Atomic.get opened)

(* failure *)

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

(* A child of a process with no failure sees its inherited devices lost by the
   fork, which is no failure. *)
let test_fork_no_failure () =
  let d, _ = P.open_ "fail:forked" in
  child_case [ "forked"; "none" ] (fun () -> [ reason d; failure () ]) ()

(* The first loss stays the failure, through later losses and a fail. *)
let test_first_loss =
  child_case [ "fail:first lost: the submission failed" ] @@ fun () ->
  let d, p = P.open_ "fail:first" and e, q = P.open_ "fail:second" in
  lose d p;
  lose e q;
  Rig.fail "the world failed";
  [ failure () ]

let test_fail_first =
  child_case [ "the world failed" ] @@ fun () ->
  let d, p = P.open_ "fail:after" in
  Rig.fail "the world failed";
  lose d p;
  [ failure () ]

let test_lose_first =
  child_case [ "fail:given-up lost: given up" ] @@ fun () ->
  let d, _ = P.open_ "fail:given-up" in
  Rig.lose d "given up";
  Rig.fail "the world failed";
  [ failure () ]

let test_lose_after_fail =
  child_case [ "the world failed"; "the world failed" ] @@ fun () ->
  let d, _ = P.open_ "fail:given-up-after" in
  Rig.fail "the world failed";
  Rig.lose d "given up";
  [ reason d; failure () ]

let tests =
  [
    group "errors and failure"
      [
        test "a Ctrl-C during a driver's wait raises inside it"
          test_break_ends_sleep;
        test
          "a Ctrl-C during a driver's call outside its waits raises as it \
           returns"
          (test_break_in_call `Caller);
        test
          "a Ctrl-C another thread takes during a driver's call raises as it \
           returns"
          (test_break_in_call `Other);
        test "a Ctrl-C during a shielded wait raises as the call returns"
          test_break_in_shielded_wait;
        test "Sys.Break in a driver's read of its word loses nothing"
          test_break_in_signaled;
        test
          "a driver's call runs with every signal blocked but those a fault \
           raises"
          test_mask_in_call;
        test "a fault interrupted at any allocation after it loses the device"
          test_fault_interrupted;
        test
          "an exception at any allocation of a wait lost to a hang leaves its \
           stop to run once"
          test_hang_interrupted;
        test
          "a Ctrl-C at any poll point after a driver's exception leaves its \
           stop to run once"
          test_break_after_escape;
        cases ~id:fst
          "an exception escaping a driver's change loses the device with its \
           text"
          escapes escaping;
        test "an exception escaping a driver's change is the process's failure"
          test_escape_failure;
        test "a release from C with no claim aborts the process"
          test_release_unclaimed;
      ];
    group "fail"
      [
        test "a fail loses every live device but the host with its reason"
          test_fail_loses;
        test "a fail runs no stop" test_fail_no_stop;
        test "an open after a fail is its reason, its opener uncalled"
          test_fail_refuses_opens;
        test "only the first fail acts" test_fail_once;
        test "a device lost through its waits keeps the fail's reason"
          test_fail_spread;
        test "an open racing a fail makes a lost device or is refused"
          test_fail_beside_opens;
      ];
    group "failure"
      [
        test "a close is no failure, a loss is" test_close_no_failure;
        test "a fork's losses are no failure" test_fork_no_failure;
        test "the first loss stays the failure" test_first_loss;
        test "a fail that comes first is the failure" test_fail_first;
        test "a lose that comes first is the failure" test_lose_first;
        test "a lose after a fail keeps the fail's reason" test_lose_after_fail;
      ];
  ]

let () = exit (run "rig.fail" tests)
