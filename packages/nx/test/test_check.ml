(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Waiting and checks: a check raises the caller's exception at its first false
   flag in C order, with its index and the data's elements there, each array
   broadcast to the flags' shape; at the call on the host, and on another set at
   a wait on that set or on every set, on any domain, one failure per wait,
   oldest first, each once. A pending check holds none of its data, and checks
   no wait follows hold a bounded window. *)

open Windtrap
open Nx_support
module D = Nx_array.Dtype
module P = Nx.Ptree

let m = Nx_support.memory

module S = (val Nx.devices [ m 0 ])
module S2 = (val Nx.devices [ m 0; m 1 ])
module Two = (val Nx.devices [ m 2; m 3 ])
module C = Nx_support.Counting
module Count = (val Nx.devices ~kernels:(module C) [ m 1 ])

exception At of int array * float * int32 * float
exception Flag of int array
exception Element of string
exception Failed of int

type ('v, 's, 'd) Nx.Prim.payload += Mark : ('v, 's, 'd) Nx.Prim.payload

(* A misuse raises [Invalid_argument] whose message is the function the user
   called, a colon and the reason (nx.mli's Errors). *)
let refuses ~by f =
  let named = function
    | Invalid_argument msg ->
        String.starts_with ~prefix:(by ^ ": ") msg
        && String.length msg > String.length by + 2
    | _ -> false
  in
  raises_match named f

let numel s = Array.fold_left ( * ) 1 s
let pass _ _ = Failure "the check passed its data"

(* A check of nx's primitive, its data a list of arrays. *)
let check ok data fail =
  Nx.Prim.eval ~by:"test" (Nx.Prim.Check { ok; data; fail })

(* A device whose queue runs only when the test runs it or a wait sleeps on it,
   its memory the host's where [host_visible]. *)
let polled ?(host_visible = false) name =
  Rig_support.Polled.open_ ~capacity:(1 lsl 16) ~host_visible
    ~waits_on:[ `Host ] name

(* A set seen through a value on it and two flags of [n] elements, one false at
   its last position. *)
type wait = Sync | To_array | Item | Print | Place_host

let pp_wait ppf w =
  Format.pp_print_string ppf
    (match w with
    | Sync -> "sync"
    | To_array -> "to_array"
    | Item -> "item"
    | Print -> "to_string"
    | Place_host -> "place on the host")

let waits = Gen.of_list ~pp:pp_wait [ Sync; To_array; Item; Print; Place_host ]

let wait w x =
  match w with
  | Sync -> Nx.sync x
  | To_array -> ignore (Nx.to_array x)
  | Item -> ignore (Nx.item [ 0 ] x)
  | Print -> ignore (Nx.to_string x)
  | Place_host -> ignore (Nx.place Nx.Host.on x)

type side = { wait : wait -> unit; issue : bool -> int -> unit }

let side (type d) (module D : Nx.Devices with type d = d) n =
  let x = Nx.place D.on (Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |]) in
  let flags b =
    let ok = Array.init n (fun k -> k < n - 1 || b) in
    Nx.place D.on (Nx.create Nx.bool [| n |] ok)
  in
  let pass = flags true and fail = flags false in
  {
    wait = (fun w -> wait w x);
    issue =
      (fun fails id ->
        check (if fails then fail else pass) [] (fun _ () -> Failed id));
  }

(* S's checks are read at their issue, since its device runs its work as it is
   submitted. Held's device runs its own queue at times no test chooses, so its
   checks of one flag stay pending until a wait or an issue reads them, and two
   readers may meet. *)
let pending_device, _ =
  Rig_support.Polled.open_ ~capacity:(1 lsl 16) ~host_visible:false
    ~runs:`Itself ~waits_on:[ `Host ] "nx-check-pending"

module Held = (val Nx.devices [ pending_device ])

let s_side = side (module S) 3
let t_side = side (module Held) 1

(* Checks *)

let at_call =
  let ok = [| true; false; true |] in
  cases "a check on the host raises at the call, leaving no failure to a wait"
    ~id:(fun (name, _) -> name)
    [
      ( "on the host",
        fun () -> check (Nx.create Nx.bool [| 3 |] ok) [] (fun _ () -> Failed 0)
      );
      ( "of a host creation",
        fun () ->
          check
            Nx.Infix.(Nx.arange Nx.int32 0 3 1 <>$ 1l)
            []
            (fun _ () -> Failed 0) );
    ]
    (fun (_, f) ->
      raises (Failed 0) f;
      Nx.wait_checks ())

let test_every_failure () =
  let ok b = Nx.place S.on (Nx.create Nx.bool [| 2 |] [| true; b |]) in
  let issue b id = check (ok b) [] (fun _ () -> Failed id) in
  issue false 1;
  issue true 2;
  issue false 3;
  raises (Failed 1) (fun () -> Nx.sync (ok true));
  (* A check issued once the set holds a failure is computed too. *)
  issue false 4;
  raises (Failed 3) (fun () -> Nx.sync (ok true));
  raises (Failed 4) (fun () -> Nx.sync (ok true));
  Nx.sync (ok true)

let test_built_at_wait () =
  let ok b = Nx.place S.on (Nx.create Nx.bool [| 2 |] [| true; b |]) in
  let data =
    ( Nx.place S.on (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]),
      Nx.place S.on (Nx.create Nx.int32 [| 1 |] [| 9l |]) )
  in
  let found = ref 0 in
  let fail i (a, b) =
    incr found;
    At (i, Nx.item [] a, Nx.item [] b, 0.)
  in
  let desc = P.(pair tensor tensor) in
  Nx.check desc (ok false) data fail;
  Nx.check desc (ok true) data fail;
  equal ~msg:"built at no issue" int 0 !found;
  raises (At ([| 1 |], 2., 9l, 0.)) (fun () -> Nx.sync (fst data));
  equal ~msg:"built at the wait" int 1 !found;
  Nx.sync (fst data);
  equal ~msg:"built once" int 1 !found

let test_backtrace () =
  let raise_exit _ () = raise (Sys.opaque_identity Exit) in
  let ok b = Nx.place S.on (Nx.create Nx.bool [| 2 |] [| true; b |]) in
  let at_wait () =
    Nx.check P.unit (ok false) () raise_exit;
    Nx.sync (ok true)
  in
  (* The failure is found at the second issue, and [fail] runs at the wait. *)
  let found_at_issue () =
    Nx.check P.unit (ok false) () raise_exit;
    Nx.check P.unit (ok true) () raise_exit;
    Nx.sync (ok true)
  in
  equal ~msg:"at a wait" string "test_check.ml" (raised_in at_wait);
  equal ~msg:"found at an issue" string "test_check.ml"
    (raised_in found_at_issue)

(* Two systhreads of one process share its pending checks. The order below is
   held by two semaphores: the main thread's wait takes the oldest failure and
   stops inside its [fail], after a read of the value [fail] receives, where the
   other thread waits on the set, then resumes once that wait returns. *)
let test_two_systhreads () =
  let ok = Nx.place S.on (Nx.create Nx.bool [| 2 |] [| true; false |]) in
  let data = Nx.place S.on (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]) in
  let go = Semaphore.Binary.make false in
  let waited = Semaphore.Binary.make false in
  let raised = ref None in
  let other =
    Thread.create
      (fun () ->
        Semaphore.Binary.acquire go;
        (raised := match Nx.sync data with () -> None | exception e -> Some e);
        Semaphore.Binary.release waited)
      ()
  in
  (* [Nx.item] waits on the set while the failure is built. *)
  let failing i x =
    let e = At (i, Nx.item [] x, 0l, 0.) in
    Semaphore.Binary.release go;
    Semaphore.Binary.acquire waited;
    e
  in
  Nx.check P.tensor ok data failing;
  check ok [] (fun _ () -> Failed 2);
  raises (At ([| 1 |], 2., 0l, 0.)) (fun () -> Nx.sync data);
  Thread.join other;
  (match !raised with
  | Some (Failed 2) -> ()
  | Some e -> raise e
  | None -> fail "the other thread's wait passed a pending failure");
  Nx.sync data

(* The systhreads of a domain share its checks. A signal every 20 us of CPU time
   yields the running systhread at its next poll point, so that two issuers
   interleave inside nx's own updates. It is the virtual timer's: the real-time
   one is windtrap's, for its timeouts. Each of [sets] sets gets one failing
   check a round, and every one must raise. *)
let test_systhreads () =
  let sets =
    Array.init 6000 (fun _ ->
        let module D = (val Nx.devices [ m 0 ]) in
        side (module D) 3)
  in
  let tick = { Unix.it_interval = 2e-5; it_value = 2e-5 } in
  let still = { Unix.it_interval = 0.; it_value = 0. } in
  let yield = Sys.Signal_handle (fun _ -> Thread.yield ()) in
  let issue first () =
    for i = 0 to (Array.length sets / 2) - 1 do
      sets.((2 * i) + first).issue true i
    done
  in
  let round () =
    let a = Thread.create (issue 0) () and b = Thread.create (issue 1) () in
    Thread.join a;
    Thread.join b;
    Array.fold_left
      (fun lost side ->
        match side.wait Sync with () -> lost + 1 | exception Failed _ -> lost)
      0 sets
  in
  let rounds () = List.init 5 (fun _ -> round ()) in
  (* Windows has neither the virtual timer nor its signal: there the systhreads
     interleave at the runtime's own ticks. *)
  let lost =
    if Sys.win32 then rounds ()
    else begin
      let previous = Sys.signal Sys.sigvtalrm yield in
      ignore (Unix.setitimer Unix.ITIMER_VIRTUAL tick);
      Fun.protect
        ~finally:(fun () ->
          ignore (Unix.setitimer Unix.ITIMER_VIRTUAL still);
          Sys.set_signal Sys.sigvtalrm previous)
        rounds
    end
  in
  equal (list int) [ 0; 0; 0; 0; 0 ] lost

(* The checks pending on either set, as [(set, id, fails)] in issue order. *)
type set = On_s | On_t
type model = { mutable checks : (set * int * bool) list }

let sets =
  Gen.of_list
    ~pp:(fun ppf s ->
      Format.pp_print_string ppf (if s = On_s then "S" else "Held"))
    [ On_s; On_t ]

let side_of = function On_s -> s_side | On_t -> t_side
let other = function On_s -> On_t | On_t -> On_s
let fails_on md s = List.exists (fun (s', _, f) -> s' = s && f) md.checks

(* A wait on the sets [on] accepts raises the oldest failure among their checks
   and leaves the later ones pending. *)
let settle md on =
  let rec go = function
    | [] -> ([], None)
    | ((s, _, _) as c) :: rest when not (on s) ->
        let rest, raised = go rest in
        (c :: rest, raised)
    | (_, _, false) :: rest -> go rest
    | (_, id, true) :: rest -> (rest, Some id)
  in
  let mine = List.filter (fun (s, _, _) -> on s) md.checks in
  cover "a wait passes a check before a failing one"
    (match mine with (_, _, false) :: _ :: _ -> true | _ -> false);
  let left, raised = go md.checks in
  cover "a wait leaves a later failure pending"
    (List.exists (fun (s, _, f) -> on s && f) left);
  md.checks <- left;
  raised

(* Raises every failure pending on a set, so that no test sees another's. *)
let rec drain side =
  match side.wait Sync with () -> () | exception Failed _ -> drain side

(* The checks are the process's, so every session shares one model, which a
   start empties. A program's end drains what it left. *)
let model = { checks = [] }

let session =
  abstract "w" ~release:(fun () ->
      drain s_side;
      drain t_side)

(* A wait on every set raises the oldest failure of either, in issue order.
   Listed twice, so that it often meets failures on both sets. *)
let every =
  command "wait on every set"
    (session ^-> returns (option int))
    (fun md -> settle md (fun _ -> true))
    (fun () ->
      match Nx.wait_checks () with () -> None | exception Failed id -> Some id)

(* On two domains a wait on every set may leave a check issued while it runs, so
   a [None] leaves the model as it was, and a later wait raises what it left. A
   failure it raises is the oldest. *)
let every_judged =
  command "wait on every set, judged"
    (session ^-> judges (option int))
    (fun md -> function
      | Ok None -> ()
      | Ok (Some id) ->
          equal ~msg:"the oldest failure" (option int)
            (settle md (fun _ -> true))
            (Some id)
      | Error e -> raise e)
    (fun () ->
      match Nx.wait_checks () with () -> None | exception Failed id -> Some id)

(* On two domains each wait and issue is explained by some order of the calls: a
   wait raises a failure whichever domain issued it. *)
let pending_checks ?count ?domains ~across name =
  stateful ?count ?domains ~steps:30 name
    ([
       command "start"
         (Gen.unit @-> makes session)
         (fun () ->
           model.checks <- [];
           model)
         (fun () ->
           drain s_side;
           drain t_side);
       command "issue"
         (session ^-> sets @-> Gen.bool @-> Gen.int_range 0 99 @-> returns unit)
         (fun md s fails id -> md.checks <- md.checks @ [ (s, id, fails) ])
         (fun () s fails id -> (side_of s).issue fails id);
       command "wait"
         (session ^-> sets @-> waits @-> returns (option int))
         (fun md s _ ->
           let raised = settle md (( = ) s) in
           cover "a wait on a set with a failure on the other"
             (raised = None && fails_on md (other s));
           raised)
         (fun () s w ->
           match (side_of s).wait w with
           | () -> None
           | exception Failed id -> Some id);
     ]
    @ across)

let test_holds_none () =
  let n = 1 lsl 16 in
  let ok = Array.init n (fun k -> k <> n - 1) in
  let ok = Nx.place S.on (Nx.create Nx.bool [| n |] ok) in
  (* The data, a copy that owns its memory, is reachable from no value of the
     test once [issue] returns. *)
  let issue () =
    let d = Nx.create Nx.float32 [| n |] (Array.make n 1.) in
    let d = Nx.copy (Nx.place S.on d) in
    Nx.check P.tensor ok d (fun i x -> At (i, Nx.item [] x, 0l, 0.))
  in
  let settled () =
    for _ = 1 to 4 do
      Gc.full_major ();
      Rig.free_cache (m 0)
    done;
    Rig.allocated (m 0)
  in
  let before = settled () in
  issue ();
  at_most ~msg:"bytes held past the data's" int
    ~than:(before + (n * 4 / 2))
    (settled ());
  raises (At ([| n - 1 |], 1., 0l, 0.)) (fun () -> Nx.sync ok)

let test_through_donation () =
  let d = Nx.place S.on (Nx.create Nx.float32 [| 3 |] [| 4.; 5.; 6. |]) in
  let ok = Nx.place S.on (Nx.create Nx.bool [| 3 |] [| true; false; true |]) in
  check ok [ d ] (fun idx (v, ()) -> At (idx, v, 0l, 0.));
  let negated = Nx.neg (Nx.donate d) in
  raises (At ([| 1 |], 5., 0l, 0.)) (fun () -> Nx.sync negated);
  equal (array float_exact) [| -4.; -5.; -6. |] (Nx.to_array negated)

(* The bytes [d] holds once the collector and [d]'s cache have released what no
   value reaches: a chain of finalisers frees its memory one round late. *)
let settled d =
  for _ = 1 to 4 do
    Gc.full_major ();
    Rig.free_cache d
  done;
  Rig.allocated d

let donated_before_wait (n, fails) =
  (* A copy owns its memory: a placement onto a software device borrows the
     host's, which no donation writes. *)
  let d = Nx.create Nx.float32 [| n |] (Array.init n float_of_int) in
  let d = Nx.copy (Nx.place S.on d) in
  let ok = Array.init n (fun k -> not (fails && k = n - 1)) in
  let ok = Nx.place S.on (Nx.create Nx.bool [| n |] ok) in
  check ok [ d ] (fun idx (v, ()) -> At (idx, v, 0l, 0.));
  (* The count includes cached memory, which a refused donation would take
     unseen. *)
  let before = settled (m 0) in
  let negated = Nx.neg (Nx.donate d) in
  equal ~msg:"bytes allocated" int before (Rig.allocated (m 0));
  if fails then
    raises
      (At ([| n - 1 |], float_of_int (n - 1), 0l, 0.))
      (fun () -> Nx.sync negated);
  Nx.sync negated

let test_donated_after_sync () =
  (* A copy owns its memory: a placement onto a software device borrows the
     host's, which no donation writes. *)
  let d = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let d = Nx.copy (Nx.place S.on d) in
  let ok = Nx.place S.on (Nx.create Nx.bool [| 3 |] [| true; true; true |]) in
  check ok [ d ] (fun idx (v, ()) -> At (idx, v, 0l, 0.));
  Nx.sync d;
  (* The count includes cached memory, which a refused donation would take
     unseen. *)
  let before = settled (m 0) in
  ignore (Nx.neg (Nx.donate d));
  equal int before (Rig.allocated (m 0))

let test_kept_in_place () =
  (* The host addresses [a], and [b] copies into it, in a queue the test runs,
     so the flag and the datum are unwritten at issue. *)
  let a, _ = polled ~host_visible:true "nx-check-held" in
  let b, bq = polled "nx-check-held-from" in
  let module A = (val Nx.devices ~kernels:(module Nx_cpu) [ a ]) in
  let module B = (val Nx.devices [ b ]) in
  let via x = Nx.place A.on (Nx.place B.on x) in
  let ok = via (Nx.create Nx.bool [| 1 |] [| false |]) in
  let d = via (Nx.create Nx.float32 [| 1 |] [| 5. |]) in
  check ok [ d ] (fun idx (v, ()) -> At (idx, v, 0l, 0.));
  while Rig_support.Polled.run bq > 0 do
    ()
  done;
  let grown f =
    Rig.free_cache a;
    let before = Rig.allocated a in
    ignore (f ());
    Rig.allocated a - before
  in
  greater ~msg:"bytes a donation before the read allocates" int ~than:0
    (grown (fun () -> Nx.neg (Nx.donate d)));
  raises (At ([| 0 |], 5., 0l, 0.)) (fun () -> Nx.sync ok);
  equal ~msg:"bytes a donation after the read allocates" int 0
    (grown (fun () -> Nx.logical_not (Nx.donate ok)))

(* On two devices the host does not address, a check of one flag on the second,
   its datum on the first: the datum's element moves to the second, and the
   second copies it and then the flag into memory the host addresses, by the
   value it has submitted once the check is issued, which the check is read
   after. *)
let test_apart_off_host () =
  let p0, q0 = polled "nx-check-apart-0" in
  let p1, q1 = polled "nx-check-apart-1" in
  let module A = (val Nx.devices [ p0; p1 ]) in
  let on d x = Nx.place (Nx.Placement.device A.v d) x in
  let ok = on p1 (Nx.create Nx.bool [||] [| false |]) in
  let d = on p0 (Nx.create Nx.float32 [||] [| 5. |]) in
  let rec drain () =
    if Rig_support.Polled.run q0 + Rig_support.Polled.run q1 > 0 then drain ()
  in
  drain ();
  let mark = ref 0 in
  let (), events =
    Rig.Profile.take (fun () ->
        check ok [ d ] (fun idx (v, ()) -> At (idx, v, 0l, 0.));
        mark := Rig.submitted p1;
        drain ())
  in
  let into_host =
    List.filter_map
      (function
        | Rig.Profile.Copy { src; dst; bytes; point; _ }
          when Rig.equal src p1 && Rig.equal dst Rig.host ->
            let on_p1 pt =
              Option.fold ~none:false ~some:(Rig.equal p1) (Rig.Point.device pt)
              && Rig.Point.value pt <= !mark
            in
            Some (bytes, Option.fold ~none:false ~some:on_p1 point)
        | _ -> None)
      events
  in
  equal ~msg:"copies of the element and the flag, submitted at the issue"
    (slist (pair int bool) compare)
    [ (4, true); (1, true) ]
    into_host;
  raises (At ([||], 5., 0l, 0.)) (fun () -> Nx.sync ok)

let test_earlier_raises () =
  let lost, _ = polled "nx-check-lost-at-issue" in
  let module L = (val Nx.devices [ lost; m 2 ]) in
  let on d = Nx.Placement.device L.v d in
  let flags d = Nx.place (on d) (Nx.create Nx.bool [| 1 |] [| false |]) in
  let x = Nx.place (on (m 2)) (Nx.create Nx.float32 [| 1 |] [| 0. |]) in
  check (flags lost) [] (fun _ () -> Failed 1);
  let late = flags (m 2) in
  Rig.lose lost "lost by the test";
  raises_match
    (function Rig.Lost _ -> true | _ -> false)
    (fun () -> check late [] (fun _ () -> Failed 2));
  raises (Failed 2) (fun () -> Nx.sync x);
  Nx.sync x

(* The wait reads a check whose flag a Polled device copies, and sleeps at the
   device's gate, holding the claim; the issue on the set's other device, on the
   test's thread, returns without reading. An issue that waited for the claim
   would never return. *)
let test_issue_beside_wait () =
  let p, pq = polled "nx-check-claimed" in
  let module L = (val Nx.devices [ p; m 2 ]) in
  let on d = Nx.Placement.device L.v d in
  let flag d b = Nx.place (on d) (Nx.create Nx.bool [| 1 |] [| b |]) in
  let x = Nx.place (on (m 2)) (Nx.create Nx.float32 [| 1 |] [| 0. |]) in
  check (flag p false) [] (fun _ () -> Failed 1);
  Rig_support.Polled.gate pq;
  let raised = ref None and waited = Atomic.make false in
  let waiter =
    Thread.create
      (fun () ->
        (raised := match Nx.sync x with () -> None | exception e -> Some e);
        Atomic.set waited true)
      ()
  in
  let asleep () = Rig_support.Polled.sleepers pq > 0 in
  while (not (asleep ())) && not (Atomic.get waited) do
    Thread.yield ()
  done;
  if Atomic.get waited then begin
    Rig_support.Polled.open_gate pq;
    fail "the wait returned without reading the pending check"
  end;
  check (flag (m 2) true) [] (fun _ () -> Failed 2);
  let sleeping = Rig_support.Polled.sleepers pq in
  Rig_support.Polled.open_gate pq;
  Thread.join waiter;
  equal ~msg:"waits asleep through the issue" int 1 sleeping;
  (match !raised with
  | Some (Failed 1) -> ()
  | Some e -> raise e
  | None -> fail "the wait raised no failure");
  Nx.sync x

let test_issue_beside_kept () =
  (* As in [test_kept_in_place], and a second check issued while [b]'s copies
     into the first check's buffers are still queued. A host wait on them would
     run [b]'s queue, as a wait sleeps, so the queue is how many it left. *)
  let a, _ = polled ~host_visible:true "nx-check-held-beside" in
  let b, bq = polled "nx-check-held-beside-from" in
  let module A = (val Nx.devices ~kernels:(module Nx_cpu) [ a ]) in
  let module B = (val Nx.devices [ b ]) in
  let via x = Nx.place A.on (Nx.place B.on x) in
  let ok = via (Nx.create Nx.bool [| 1 |] [| false |]) in
  let d = via (Nx.create Nx.float32 [| 1 |] [| 5. |]) in
  check ok [ d ] (fun idx (v, ()) -> At (idx, v, 0l, 0.));
  let written = Nx.place A.on (Nx.create Nx.bool [| 1 |] [| true |]) in
  let queued = Rig_support.Polled.queued bq in
  check written [] (fun _ () -> Failed 0);
  equal ~msg:"copies still queued" int queued (Rig_support.Polled.queued bq);
  raises (At ([| 0 |], 5., 0l, 0.)) (fun () -> Nx.sync ok)

(* A loop that draws over a fresh parameter at [at] each step and never waits on
   its set: once warm, device [d] holds as many bytes after 200 more steps as
   after the first, its collected memory moved and its cache freed alike. *)
let bounded at d =
  let n = 1 lsl 16 in
  let p = Nx.create Nx.float32 [| n |] (Array.make n 0.5) in
  let step () = ignore (Nx.Rng.bernoulli (Nx.copy (Nx.place at p))) in
  let settled () =
    for _ = 1 to 4 do
      Gc.full_major ();
      Rig.free_cache d
    done;
    Rig.allocated d
  in
  step ();
  let warm = settled () in
  for _ = 1 to 200 do
    step ()
  done;
  at_most int ~than:warm (settled ())

let test_read_raises () =
  (* A Polled device runs its queue only when the test runs it, so its copy of
     the flag is still queued when it is lost. *)
  let lost, _ = polled "nx-check-lost" in
  let module L = (val Nx.devices [ lost; m 2 ]) in
  let on d = Nx.Placement.device L.v d in
  let flags d = Nx.place (on d) (Nx.create Nx.bool [| 1 |] [| false |]) in
  check (flags lost) [] (fun _ () -> Failed 1);
  check (flags (m 2)) [] (fun _ () -> Failed 2);
  let x = Nx.place (on (m 2)) (Nx.create Nx.float32 [| 1 |] [| 0. |]) in
  Rig.lose lost "lost by the test";
  raises_match
    (function Rig.Lost _ -> true | _ -> false)
    (fun () -> Nx.sync x);
  raises (Failed 2) (fun () -> Nx.sync x);
  Nx.sync x

let checks =
  group "checks"
    [
      at_call;
      test "a wait raises a failure another domain issued" (fun () ->
          Domain.join (Domain.spawn (fun () -> s_side.issue true 8));
          raises (Failed 8) (fun () -> s_side.wait Sync);
          s_side.wait Sync);
      test "every failure raises, oldest first, one per wait" test_every_failure;
      test "a failure's exception is built at the wait that raises it"
        test_built_at_wait;
      test "what fail raises keeps the backtrace of its raise" test_backtrace;
      test
        "a wait inside fail raises none, and another thread's raises the next"
        test_two_systhreads;
      test "checks issued from two systhreads all raise" test_systhreads;
      (* A program leaves a later failure pending about one time in ten on one
         domain and four in ten on two. *)
      pending_checks ~count:300 ~across:[ every; every ]
        "a wait raises its set's oldest failure, once";
      pending_checks ~count:60 ~domains:2
        ~across:[ every_judged; every_judged ]
        "on two domains, a wait raises its set's oldest failure, once";
      test "a check holds none of its data while its failure is pending"
        test_holds_none;
      test "a check keeps its data's elements through a donation"
        test_through_donation;
      cases "a donation of checked data before any wait reuses its memory"
        ~id:(fun (n, fails) ->
          let outcome = if fails then "failing" else "passing" in
          Printf.sprintf "%d flags, %s" n outcome)
        [ (1, false); (1, true); (3, false); (3, true) ]
        donated_before_wait;
      test "a donation after a sync reuses the checked data's memory"
        test_donated_after_sync;
      test "a check kept in place holds its data until it is read"
        test_kept_in_place;
      test
        "a check off host memory submits copies of its flag and element into \
         host memory as it is issued"
        test_apart_off_host;
      test "an issue whose earlier check raises keeps the new check"
        test_earlier_raises;
      test "an issue returns while a wait holds its set's claim"
        test_issue_beside_wait;
      test "an issue beside a check kept in place waits for nothing"
        test_issue_beside_kept;
      test "checks no wait follows hold bounded memory" (fun () ->
          bounded S.on (m 0));
      test "a check on a device then idle leaves later checks bounded"
        (fun () ->
          let on k = Nx.Placement.device Two.v (m k) in
          let p = Nx.create Nx.float32 [| 4 |] (Array.make 4 0.5) in
          ignore (Nx.Rng.bernoulli (Nx.copy (Nx.place (on 3) p)));
          bounded (on 2) (m 2));
      test "a read that raises takes its check off" test_read_raises;
    ]

(* Waiting *)

let f32 = Nx.float32

(* A host value placed on [src], then at [p]; the value on [src] is reachable
   from no value once this returns. *)
let[@inline never] [@local never] placed_through src p n =
  Nx.place p (Nx.place src (Nx.create f32 [| n |] (Array.make n 1.)))

(* Two devices whose queues run only when the test runs them: the copy from the
   first to the second reads the first's memory once it runs. *)
let test_queued_memory () =
  let n = 1 lsl 16 in
  let a, qa = polled "nx-check-queued-from" in
  let b, qb = polled "nx-check-queued-to" in
  let module A = (val Nx.devices [ a ]) in
  let module B = (val Nx.devices [ b ]) in
  let settled () =
    for _ = 1 to 4 do
      Gc.full_major ();
      Rig.free_cache a
    done;
    Rig.allocated a
  in
  let run () =
    while Rig_support.Polled.run qa + Rig_support.Polled.run qb > 0 do
      ()
    done
  in
  (* One uncounted round warms the devices. *)
  ignore (placed_through A.on B.on n);
  run ();
  let before = settled () in
  let y = placed_through A.on B.on n in
  let queued = settled () in
  run ();
  let ran = settled () in
  at_least ~msg:"bytes held while the copy is queued" int
    ~than:(before + (4 * n))
    queued;
  equal ~msg:"the copy's elements" (array float_exact) (Array.make n 1.)
    (Nx.to_array y);
  at_most ~msg:"bytes held once it ran" int ~than:before ran

let waiting =
  group "waiting"
    [
      test "memory queued work reads stays allocated until the work runs"
        test_queued_memory;
    ]

(* sync *)

let trace i ~by op = Nx.Prim.results ~by (fun _ f -> Nx.Prim.traced i f Mark) op

let sync =
  group "sync"
    [
      test "sync of a host value returns" (fun () ->
          Nx.sync (Nx.arange Nx.int32 0 3 1));
      test "sync raises naming itself for a dead value" (fun () ->
          let x = Nx.place S.on (Nx.create f32 [| 2 |] [| 1.; 2. |]) in
          ignore (Nx.neg (Nx.donate x));
          refuses ~by:"Nx.sync" (fun () -> Nx.sync x));
      test "sync raises naming itself for a traced value" (fun () ->
          Nx.Prim.interpret ~name:"test.trace" Extent trace (fun _ ->
              let x = Nx.neg (Nx.create f32 [| 2 |] [| 1.; 2. |]) in
              raises
                (Invalid_argument "Nx.sync: the value is traced by test.trace")
                (fun () -> Nx.sync x)));
      test "sync raises naming itself for a value whose interpretation returned"
        (fun () ->
          let x =
            Nx.Prim.interpret ~name:"test.trace" Values trace (fun i ->
                Nx.Prim.traced i
                  (Nx.Prim.form (Nx.create f32 [| 2 |] [| 1.; 2. |]))
                  Mark)
          in
          raises
            (Invalid_argument
               "Nx.sync: a value of test.trace, used after test.trace returned")
            (fun () -> Nx.sync x));
    ]

(* wait_checks. Later is minted after S, so a wait that went through the sets in
   their order would raise S's failures before Later's. *)

module Later = (val Nx.devices [ m 1 ])

let flags (type d) (module D : Nx.Devices with type d = d) b =
  Nx.place D.on (Nx.create Nx.bool [| 2 |] [| true; b |])

let issue d b id = check (flags d b) [] (fun _ () -> Failed id)
let s_set = (module S : Nx.Devices with type d = S.d)
let later = (module Later : Nx.Devices with type d = Later.d)

let test_inside_build () =
  let inner = ref None in
  check (flags s_set false) [] (fun _ () ->
      inner :=
        Some (match Nx.wait_checks () with () -> false | exception _ -> true);
      Failed 1);
  issue later false 2;
  raises (Failed 1) Nx.wait_checks;
  equal ~msg:"the wait inside the build raised" (option bool) (Some false)
    !inner;
  raises (Failed 2) Nx.wait_checks;
  Nx.wait_checks ()

let test_reading_raises () =
  (* A Polled device runs its queue only when the test runs it, so its copy of
     the flag is still queued when it is lost. *)
  let lost, _ = polled "nx-check-lost-every" in
  let module L = (val Nx.devices [ lost ]) in
  let ok = Nx.place L.on (Nx.create Nx.bool [| 1 |] [| false |]) in
  check ok [] (fun _ () -> Failed 1);
  issue s_set false 2;
  Rig.lose lost "lost by the test";
  raises_match (function Rig.Lost _ -> true | _ -> false) Nx.wait_checks;
  raises (Failed 2) Nx.wait_checks;
  Nx.wait_checks ()

(* The value's copy to a gated Polled device stays queued, and a wait on it
   would sleep at the gate. The check's copy to a second Polled device stays
   queued too, until the wait reads it. A wait that waited for the value's copy
   would never return. *)
let test_no_other_work () =
  let p, pq = polled "nx-check-unread-every" in
  let r, rq = polled "nx-check-read-every" in
  let module P = (val Nx.devices [ p ]) in
  let module R = (val Nx.devices [ r ]) in
  Rig_support.Polled.gate pq;
  let x = Nx.place P.on (Nx.create f32 [| 1 |] [| 1. |]) in
  check
    (Nx.place R.on (Nx.create Nx.bool [| 1 |] [| true |]))
    []
    (fun _ () -> Failed 1);
  let pending = Rig_support.Polled.queued rq in
  Nx.wait_checks ();
  let queued = Rig_support.Polled.queued pq in
  Rig_support.Polled.open_gate pq;
  greater ~msg:"the check's copy was queued at the issue" int ~than:0 pending;
  equal ~msg:"the wait read the check" int 0 (Rig_support.Polled.queued rq);
  greater ~msg:"the value's copy stayed queued" int ~than:0 queued;
  Nx.sync x

(* The wait sleeps at a gated Polled device, reading a check issued before it,
   while the other domain issues failing checks on S and raises each with a wait
   on S, all issued after the call. The gate opens once the other domain has
   raised one, and the wait returns while that domain goes on, raising none of
   its checks. A wait that took the later checks would never catch up. *)
let test_beside_domain () =
  let p, pq = polled "nx-check-every-busy" in
  let module P = (val Nx.devices [ p ]) in
  check
    (Nx.place P.on (Nx.create Nx.bool [| 1 |] [| true |]))
    []
    (fun _ () -> Failed 0);
  Rig_support.Polled.gate pq;
  let raised = ref (Some (Failed (-1))) and waited = Atomic.make false in
  let waiter =
    Thread.create
      (fun () ->
        (raised :=
           match Nx.wait_checks () with () -> None | exception e -> Some e);
        Atomic.set waited true)
      ()
  in
  let asleep () = Rig_support.Polled.sleepers pq > 0 in
  while (not (asleep ())) && not (Atomic.get waited) do
    Thread.yield ()
  done;
  let stop = Atomic.make false and rounds = Atomic.make 0 in
  let other =
    Domain.spawn (fun () ->
        while not (Atomic.get stop) do
          issue s_set false 1;
          (try Nx.sync (flags s_set true) with Failed _ -> ());
          Atomic.incr rounds
        done)
  in
  while Atomic.get rounds = 0 do
    Thread.yield ()
  done;
  Rig_support.Polled.open_gate pq;
  Thread.join waiter;
  Atomic.set stop true;
  Domain.join other;
  (match !raised with None -> () | Some e -> raise e);
  Nx.wait_checks ()

let wait_checks =
  group "wait_checks"
    [
      test "raises every set's failures, oldest first, one per wait" (fun () ->
          (* The first check's flag reaches its device's queue, which runs only
             when a wait sleeps on it: it is pending at the wait, older than the
             failures read at their issue. *)
          let p, _ = polled "nx-check-oldest-pending" in
          let module P = (val Nx.devices [ p ]) in
          check
            (Nx.place P.on (Nx.create Nx.bool [| 1 |] [| false |]))
            []
            (fun _ () -> Failed 0);
          issue later false 1;
          issue s_set false 2;
          issue s_set true 3;
          issue later false 4;
          raises (Failed 0) Nx.wait_checks;
          raises (Failed 1) Nx.wait_checks;
          raises (Failed 2) Nx.wait_checks;
          raises (Failed 4) Nx.wait_checks;
          Nx.wait_checks ());
      test "leaves a set's wait its own set's oldest failure" (fun () ->
          issue later false 1;
          issue s_set false 2;
          raises (Failed 2) (fun () -> Nx.sync (flags s_set true));
          raises (Failed 1) Nx.wait_checks;
          Nx.wait_checks ());
      test "inside a failure's build raises nothing and drops nothing"
        test_inside_build;
      test "raises what reading a check raises in its turn, taking it off"
        test_reading_raises;
      test "waits for no work that a check does not read" test_no_other_work;
      test "returns while another domain issues and raises checks"
        test_beside_domain;
    ]

(* Nx.check *)

type at = Host | One | Split of int

let pp_at ppf = function
  | Host -> Format.pp_print_string ppf "host"
  | One -> Format.pp_print_string ppf "one device"
  | Split a -> Format.fprintf ppf "split along %d" a

(* Flags of a shape with some extents of 0 or 1, mostly true, on the host, on
   one device, or split along an axis, whose extent doubles to divide evenly. *)
let flags_case =
  let open Gen in
  (let* kind = int_range 0 2 in
   let* s =
     array ~size:(int_range 0 3)
       (frequency [ (1, constant 0); (1, constant 1); (4, int_range 2 4) ])
   in
   let r = Array.length s in
   let* at =
     if kind = 2 && r > 0 then map (fun a -> Split a) (int_range 0 (r - 1))
     else constant (if kind = 0 then Host else One)
   in
   let s =
     match at with
     | Split a -> Array.mapi (fun i e -> if i = a then 2 * e else e) s
     | Host | One -> s
   in
   let+ ok =
     array
       ~size:(constant (numel s))
       (frequency [ (6, constant true); (1, constant false) ])
   in
   (s, at, ok))
  |> with_pp (fun ppf (s, at, ok) ->
      Format.fprintf ppf "%a %a %s" pp_ints s pp_at at
        (String.concat ""
           (Array.to_list (Array.map (fun b -> if b then "1" else "0") ok))))

(* Data of three arrays: one of the flags' shape, a 0-d one, and one of the
   flags' last axis alone, broadcast along the others. The failure raises at the
   call on the host and at the next wait elsewhere, once. *)
let law_first_false (s, at, ok) =
  let n = numel s and r = Array.length s in
  let first = Array.find_index not ok in
  let falses = Array.fold_left (fun k b -> if b then k else k + 1) 0 ok in
  let fails = Option.is_some first in
  cover "passes" ((not fails) && n > 0);
  cover "fails" fails;
  cover "no element" (n = 0);
  cover "rank 0" (r = 0);
  cover "fails past the first element" (Option.value ~default:0 first > 0);
  cover "fails at two flags" (falses > 1);
  cover "fails on one device" (at = One && fails);
  cover "fails split" ((match at with Split _ -> true | _ -> false) && fails);
  cover "fails split along an inner axis"
    ((match at with Split a -> a > 0 | _ -> false) && fails);
  let xs = Array.init n float_of_int in
  let last = if r = 0 then [||] else [| s.(r - 1) |] in
  let expected =
    Option.map
      (fun j ->
        let i = index s j in
        let column = if r = 0 then 100. else 100. +. float_of_int i.(r - 1) in
        At (i, xs.(j), 7l, column))
      first
  in
  let run (type d) (cut : d Nx.Placement.t) (whole : d Nx.Placement.t) =
    let data =
      ( Nx.place cut (Nx.create Nx.float32 s xs),
        ( Nx.place whole (Nx.create Nx.int32 [||] [| 7l |]),
          Nx.place whole
            (Nx.create Nx.float32 last
               (Array.init (numel last) (fun j -> 100. +. float_of_int j))) ) )
    in
    let ok = Nx.place cut (Nx.create Nx.bool s ok) in
    let desc = P.(pair tensor (pair tensor tensor)) in
    let fail i (a, (b, c)) = At (i, Nx.item [] a, Nx.item [] b, Nx.item [] c) in
    let issue () = Nx.check desc ok data fail in
    (match (at, expected) with
    | Host, Some e -> raises e issue
    | (One | Split _), Some e ->
        issue ();
        raises e (fun () -> Nx.sync ok)
    | _, None -> issue ());
    Nx.sync ok
  in
  match at with
  | Host -> run Nx.Host.on Nx.Host.on
  | One -> run S.on S.on
  | Split axis -> run (S2.split ~axis) S2.on

let first_false =
  prop
    ~examples:
      [
        (* Two false flags on one device, the first past the first element. *)
        ([| 3 |], One, [| true; false; false |]);
        (* Cut along the inner axis, false at position 5 alone. *)
        ([| 2; 4 |], Split 1, Array.init 8 (fun k -> k <> 5));
        (* Empty flags on the host. *)
        ([| 2; 0 |], Host, [||]);
      ]
    "the first false flag raises, once, with its index and the data there"
    flags_case law_first_false

(* Two devices whose memory the host addresses, with nx.cpu's kernels, and a
   device whose copies into them run only when the host waits: flags and data
   placed through it are unwritten when the check is issued, so the check stays
   pending until a wait. *)
let cut_a, _ = polled ~host_visible:true "nx-check-cut-a"
let cut_b, _ = polled ~host_visible:true "nx-check-cut-b"
let cut_from, _ = polled "nx-check-cut-from"

module Cut = (val Nx.devices ~kernels:(module Nx_cpu) [ cut_a; cut_b ])
module From = (val Nx.devices [ cut_from ])

(* A shape of rank 1 or 2, the axis the flags are cut along, and the false
   flags' positions. *)
let pending_cuts =
  let open Gen in
  (let* s = array ~size:(int_range 1 2) (int_range 1 4) in
   let* axis = int_range 0 (Array.length s - 1) in
   let n = numel s in
   let+ falses = list ~size:(int_range 0 3) (int_range 0 (n - 1)) in
   (s, axis, falses))
  |> with_pp (fun ppf (s, axis, falses) ->
      Format.fprintf ppf "%a cut along %d, false at [%s]" pp_ints s axis
        (String.concat "; " (List.map string_of_int falses)))

(* The device of [s] cut along [axis] in two that holds position [j]. *)
let device_of s axis j =
  if (index s j).(axis) < (s.(axis) + 1) / 2 then 0 else 1

let pending_cut (s, axis, falses) =
  let n = numel s in
  let first = List.fold_left min n falses in
  let others = List.filter (fun j -> j <> first) falses in
  cover "passes" (falses = []);
  cover "fails" (falses <> []);
  cover "one flag" (n = 1);
  cover "cut along an inner axis" (axis > 0 && s.(axis) > 1);
  cover "the first false on the second device, the first's later"
    (first < n
    && device_of s axis first = 1
    && List.exists (fun j -> device_of s axis j = 0) others);
  let through x = Nx.place (Cut.split ~axis) (Nx.place From.on x) in
  let flags = Array.init n (fun j -> not (List.mem j falses)) in
  let ok = through (Nx.create Nx.bool s flags) in
  let xs = Array.init n float_of_int in
  let seven = Nx.place From.on (Nx.create Nx.int32 [||] [| 7l |]) in
  let data = (through (Nx.create Nx.float32 s xs), Nx.place Cut.on seven) in
  let fail i (a, b) = At (i, Nx.item [] a, Nx.item [] b, 0.) in
  Nx.check P.(pair tensor tensor) ok data fail;
  if first < n then
    raises (At (index s first, xs.(first), 7l, 0.)) (fun () -> Nx.sync ok);
  Nx.sync ok

let pending_over_devices =
  prop
    ~examples:
      [
        (* Cut along the inner axis: position 2 lies on the second device, 4 on
           the first. *)
        ([| 2; 4 |], 1, [ 4; 2 ]);
        ([| 3; 4 |], 0, [ 9; 1 ]);
        ([| 1 |], 0, [ 0 ]);
        ([| 4 |], 0, []);
      ]
    "a check pending over several devices raises its first false in C order"
    pending_cuts pending_cut

(* A datum that is a sum of the devices' terms ([~sum:true]) gives its closed
   element at the first false, pending or read at its issue. *)
let test_summed_datum () =
  let a = Array.init 8 float_of_int and b = Array.init 12 float_of_int in
  let product i j =
    List.fold_left
      (fun t k -> t +. (a.((4 * i) + k) *. b.((3 * k) + j)))
      0. [ 0; 1; 2; 3 ]
  in
  let flags = Array.init 6 (fun k -> k <> 5) in
  let run (type d) (module D : Nx.Devices with type d = d) ~pending =
    let through p x =
      if pending then Nx.place p (Nx.place From.on x) else Nx.place p x
    in
    let x = through (D.split ~axis:1) (Nx.create Nx.float32 [| 2; 4 |] a) in
    let y = through (D.split ~axis:0) (Nx.create Nx.float32 [| 4; 3 |] b) in
    let sum = Nx.matmul ~sum:true x y in
    let ok = through D.on (Nx.create Nx.bool [| 2; 3 |] flags) in
    Nx.check P.tensor ok sum (fun i v -> At (i, Nx.item [] v, 0l, 0.));
    raises (At ([| 1; 2 |], product 1 2, 0l, 0.)) (fun () -> Nx.sync ok)
  in
  run (module Cut) ~pending:true;
  run (module S2) ~pending:false

(* The 0-d value a failure receives for data at [data], flags at [ok]. *)
let lies_at (type d) ~(ok : d Nx.Placement.t) ~(data : d Nx.Placement.t)
    (expected : d Nx.Placement.t) =
  let ok = Nx.full_at ok Nx.bool [| 2 |] false in
  let x = Nx.place data (Nx.create f32 [| 2 |] [| 5.; 6. |]) in
  let lies = ref None in
  Nx.check P.tensor ok x (fun i x ->
      lies := Some (Nx.placement x);
      At (i, Nx.item [] x, 0l, 0.));
  raises (At ([| 0 |], 5., 0l, 0.)) (fun () -> Nx.sync x);
  equal (option (placement ())) (Some expected) !lies

(* One array, whose walk passes the array it is given through [f]. *)
let through f = P.v (fun c x -> f (P.Walk.tensor c x))

let test_stand_ins () =
  let at = S2.split ~axis:0 in
  let ok = Nx.full_at at Nx.bool [| 2 |] true in
  let x = Nx.place at (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]) in
  let seen = ref [] in
  let record y =
    seen := (Nx.shape y, Nx.placement y) :: !seen;
    y
  in
  Nx.check (through record) ok x pass;
  Nx.sync ok;
  match !seen with
  | [] -> fail "the walk saw no array"
  | (shape, lies) :: _ ->
      equal ~msg:"the stand-in's shape" (array int) [||] shape;
      equal ~msg:"the stand-in's placement" (placement ()) S2.on lies

let test_computes_nothing () =
  let ok = Nx.full_at Count.on Nx.bool [| 4 |] true in
  let data =
    ( Nx.full_at Count.on Nx.float32 [| 4 |] 1.,
      Nx.full_at Count.on Nx.int32 [| 4 |] 2l )
  in
  let calls f =
    C.reset ();
    f ();
    Nx.sync ok;
    C.calls ()
  in
  let prim =
    calls (fun () ->
        check ok [ fst data; snd data ] (fun _ _ -> Failure "checked"))
  in
  let full = calls (fun () -> Nx.check P.(pair tensor tensor) ok data pass) in
  equal ~msg:"kernel calls of the primitive check, as of Nx.check" int prim full

let two_flags () = Nx.create Nx.bool [| 2 |] [| true; true |]
let three () = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |]

let check_group =
  let key () = Nx.Rng.key 7 in
  group "check"
    [
      first_false;
      cases "the data's elements reach the exception at every dtype on the host"
        ~id:(fun (D.Any dt) -> D.name dt)
        D.all
        (fun (D.Any dt) ->
          let vs = [| D.zero dt; D.one dt; D.of_float dt 3. |] in
          let data = Nx.create dt [| 3 |] vs in
          let ok = Nx.create Nx.bool [| 3 |] [| true; true; false |] in
          let show v = Format.asprintf "%a" (D.pp_value dt) v in
          raises
            (Element (show vs.(2)))
            (fun () ->
              Nx.check P.tensor ok data (fun _ x ->
                  Element (show (Nx.item [] x)))));
      (* The data's elements at the first false flag, of one flag or of several,
         at every dtype, on a device. *)
      cases "the data's elements reach a deferred failure at every dtype"
        ~id:(fun (D.Any dt, flags) ->
          Printf.sprintf "%s, %d flags" (D.name dt) flags)
        (List.concat_map (fun dt -> [ (dt, 1); (dt, 3) ]) D.all)
        (fun (D.Any dt, flags) ->
          let vs = [| D.zero dt; D.one dt; D.of_float dt 3. |] in
          let vs = Array.sub vs (3 - flags) flags in
          let data = Nx.place S.on (Nx.create dt [| flags |] vs) in
          let ok = Array.init flags (fun k -> k < flags - 1) in
          let ok = Nx.place S.on (Nx.create Nx.bool [| flags |] ok) in
          let show v = Format.asprintf "%a" (D.pp_value dt) v in
          Nx.check P.tensor ok data (fun _ x -> Element (show (Nx.item [] x)));
          raises (Element (show vs.(flags - 1))) (fun () -> Nx.sync ok));
      test "a check with no data passes Ptree.unit and ()" (fun () ->
          let ok = Nx.create Nx.bool [| 2 |] [| true; false |] in
          raises (Flag [| 1 |]) (fun () ->
              Nx.check P.unit ok () (fun i () -> Flag i)));
      test "an empty check off the host passes" (fun () ->
          let ok = Nx.place S.on (Nx.create Nx.bool [| 0; 2 |] [||]) in
          check ok [] (fun _ () -> Failed 0);
          Nx.sync ok);
      cases "data that does not broadcast to the flags raises at the call"
        ~id:fst
        [
          ( "on the host",
            fun () ->
              Nx.check P.tensor (two_flags ()) (three ()) (fun _ _ ->
                  Failure "checked") );
          ( "on another set",
            fun () ->
              Nx.check P.tensor
                (Nx.place S.on (two_flags ()))
                (Nx.place S.on (three ()))
                (fun _ _ -> Failure "checked") );
        ]
        (fun (_, f) -> refuses ~by:"Nx.check" f);
      cases
        "a failure's 0-d values lie whole on each device of their data's \
         placement"
        ~id:fst
        [
          ("on one device", fun () -> lies_at ~ok:S.on ~data:S.on S.on);
          ( "cut over two devices",
            fun () -> lies_at ~ok:S2.on ~data:(S2.split ~axis:0) S2.on );
          ( "on one device of a set",
            fun () ->
              let d = Nx.Placement.device S2.v (m 1) in
              lies_at ~ok:d ~data:d d );
        ]
        (fun (_, run) -> run ());
      test "a flipped datum gives its element" (fun () ->
          let ok = Array.init 8 (fun k -> k <> 5) in
          let ok = Nx.place S.on (Nx.create Nx.bool [| 8 |] ok) in
          let xs = Array.init 8 float_of_int in
          let data =
            Nx.flip (Nx.place S.on (Nx.create Nx.float32 [| 8 |] xs))
          in
          Nx.check P.tensor ok data (fun i x -> At (i, Nx.item [] x, 0l, 0.));
          raises (At ([| 5 |], 2., 0l, 0.)) (fun () -> Nx.sync data));
      pending_over_devices;
      test "a datum that is a sum gives its closed element" test_summed_datum;
      group "the rebuild tried at the call"
        [
          test "is over 0-d stand-ins at each array's placement, uncut"
            test_stand_ins;
          test "computes nothing on the devices" test_computes_nothing;
          test "raises where the walk computes on a stand-in" (fun () ->
              let ok = Nx.create Nx.bool [| 2 |] [| true; false |] in
              let x = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
              raises
                (Invalid_argument
                   "Nx.check: the structure's walk refuses 0-d arrays: Nx.add: \
                    a value traced by Nx.check reached eager execution")
                (fun () -> Nx.check (through (fun y -> Nx.add y y)) ok x pass));
          test "raises where the walk reads a stand-in" (fun () ->
              let ok = Nx.create Nx.bool [| 2 |] [| true; false |] in
              let x = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
              let read y =
                ignore (Nx.to_array y);
                y
              in
              refuses ~by:"Nx.check" (fun () ->
                  Nx.check (through read) ok x pass));
        ];
      group "walks that refuse 0-d arrays"
        [
          test "raise at the call on the host, naming Nx.check" (fun () ->
              let ok = Nx.create Nx.bool [| 2 |] [| true; false |] in
              refuses ~by:"Nx.check" (fun () ->
                  Nx.check Nx.Rng.ptree ok (key ()) pass));
          test "raise at the call on another set, leaving nothing pending"
            (fun () ->
              let ok =
                Nx.place S.on (Nx.create Nx.bool [| 2 |] [| true; false |])
              in
              let key = Nx.Rng.place S.on (key ()) in
              refuses ~by:"Nx.check" (fun () ->
                  Nx.check Nx.Rng.ptree ok key pass);
              Nx.sync ok;
              Nx.check P.unit
                (Nx.place S.on (Nx.create Nx.bool [| 1 |] [| true |]))
                ()
                (fun i () -> Flag i);
              Nx.sync ok);
        ];
    ]

let () =
  exit (run "nx check" [ checks; waiting; sync; wait_checks; check_group ])
