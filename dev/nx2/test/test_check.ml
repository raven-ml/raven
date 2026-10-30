(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Checks and waits: the first false flag in C order raises the caller's
   exception with its index and the data's elements there, each array broadcast
   to the flags' shape; at the call on the host, at the next wait on another
   set, on the domain that issued it, earliest first and once. A check holds its
   data's memory until a wait drains it, within a bounded window where none
   does. *)

open Windtrap
open Nx_support
module D = Nx_array.Dtype
module P = Nx.Ptree

let m = Nx_support.memory

module S = (val Nx.devices [ m 0 ])
module S2 = (val Nx.devices [ m 0; m 1 ])
module T = (val Nx.devices [ m 1 ])
module Two = (val Nx.devices [ m 2; m 3 ])
module C = Nx_support.Counting
module Count = (val Nx.devices ~kernels:(module C) [ m 1 ])

exception At of int array * float * int32 * float
exception Flag of int array
exception Element of string
exception Failed of int

type ('v, 's, 'd) Nx.Prim.payload += Mark : ('v, 's, 'd) Nx.Prim.payload

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f
let numel s = Array.fold_left ( * ) 1 s
let pass _ _ = Failure "the check passed its data"

(* A check of nx's primitive, its data a list of arrays. *)
let check ok data fail =
  Nx.Prim.eval ~by:"test" (Nx.Prim.Check { ok; data; fail })

type at = Host | One | Split

let pp_at ppf a =
  Format.pp_print_string ppf
    (match a with Host -> "host" | One -> "one device" | Split -> "split")

(* Flags of a shape with some extents of 0 or 1, mostly true, on the host, on
   one device, or split along their first axis, whose extent doubles to divide
   evenly. *)
let flags =
  let open Gen in
  (let* at = of_list ~pp:pp_at [ Host; One; Split ] in
   let* s =
     array ~size:(int_range 0 3)
       (frequency [ (1, constant 0); (1, constant 1); (4, int_range 2 4) ])
   in
   let at = if at = Split && Array.length s = 0 then One else at in
   let s =
     if at = Split then Array.mapi (fun a e -> if a = 0 then 2 * e else e) s
     else s
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

(* Where a case's flags and data lie: [cut] for the flags and the data of their
   shape, [whole] for the others. *)
type 'd where = {
  cut : 'v 's. ('v, 's, Nx.host) Nx.t -> ('v, 's, 'd) Nx.t;
  whole : 'v 's. ('v, 's, Nx.host) Nx.t -> ('v, 's, 'd) Nx.t;
}

(* Data of three arrays: one of the flags' shape, a 0-d one, and one of the
   flags' last axis alone, broadcast along the others. The failure raises at the
   call on the host and at the next wait elsewhere, once. *)
let law (s, at, ok) =
  let n = numel s and r = Array.length s in
  let first = Array.find_index not ok in
  let falses = Array.fold_left (fun k b -> if b then k else k + 1) 0 ok in
  cover "passes" (Option.is_none first && n > 0);
  cover "fails" (Option.is_some first);
  cover "no element" (n = 0);
  cover "rank 0" (r = 0);
  cover "fails past the first element" (Option.value ~default:0 first > 0);
  cover "fails at two flags" (falses > 1);
  cover "fails on one device" (at = One && Option.is_some first);
  cover "fails split" (at = Split && Option.is_some first);
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
  let run (type d) (p : d where) =
    let data =
      ( p.cut (Nx.create Nx.float32 s xs),
        ( p.whole (Nx.create Nx.int32 [||] [| 7l |]),
          p.whole
            (Nx.create Nx.float32 last
               (Array.init (numel last) (fun j -> 100. +. float_of_int j))) ) )
    in
    let ok = p.cut (Nx.create Nx.bool s ok) in
    let desc = P.(pair tensor (pair tensor tensor)) in
    let fail i (a, (b, c)) = At (i, Nx.item [] a, Nx.item [] b, Nx.item [] c) in
    let issue () = Nx.check desc ok data fail in
    (match (at, expected) with
    | Host, Some e -> raises e issue
    | (One | Split), Some e ->
        issue ();
        raises e (fun () -> Nx.sync ok)
    | _, None -> issue ());
    Nx.sync ok
  in
  match at with
  | Host -> run { cut = Fun.id; whole = Fun.id }
  | One ->
      let on x = Nx.place S.on x in
      run { cut = on; whole = on }
  | Split ->
      run
        {
          cut = (fun x -> Nx.place (S2.split ~axis:0) x);
          whole = (fun x -> Nx.place S2.on x);
        }

let first_false =
  prop "the first false flag raises, once, with its index and the data there"
    flags law

let host =
  group "on the host"
    [
      cases
        ~name:(fun (D.Any dt) -> D.name dt)
        "the data's elements reach the exception at every dtype" D.all
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
      test "empty flags pass, and a check without data passes unit" (fun () ->
          let data = Nx.create Nx.float32 [| 2; 0 |] [||] in
          Nx.check P.tensor
            (Nx.create Nx.bool [| 2; 0 |] [||])
            data
            (fun _ _ -> Failure "an empty check failed");
          let ok = Nx.create Nx.bool [| 2 |] [| true; false |] in
          raises (Flag [| 1 |]) (fun () ->
              Nx.check P.unit ok () (fun i () -> Flag i)));
      test "data that does not broadcast to the flags raises" (fun () ->
          let ok = Nx.create Nx.bool [| 2 |] [| true; true |] in
          let data = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
          invalid ~by:"Nx.check" (fun () ->
              Nx.check P.tensor ok data (fun _ _ -> Failure "checked")));
    ]

let elsewhere =
  group "on another set"
    [
      test "a failure raises at the next wait on the flags' set" (fun () ->
          let ok =
            Nx.place S.on (Nx.create Nx.bool [| 3 |] [| true; false; false |])
          in
          Nx.check P.unit ok () (fun i () -> Flag i);
          raises (Flag [| 1 |]) (fun () -> Nx.sync ok);
          Nx.sync ok);
      test "what fail raises keeps the backtrace of its raise" (fun () ->
          let raise_exit _ () = raise (Sys.opaque_identity Exit) in
          let ok b = Nx.place S.on (Nx.create Nx.bool [| 2 |] [| true; b |]) in
          let at_wait () =
            Nx.check P.unit (ok false) () raise_exit;
            Nx.sync (ok true)
          in
          (* The failure is found, and [fail] runs, at the second issue. *)
          let at_issue () =
            Nx.check P.unit (ok false) () raise_exit;
            Nx.check P.unit (ok true) () raise_exit;
            Nx.sync (ok true)
          in
          equal ~msg:"at a wait" string "test_check.ml" (raised_in at_wait);
          equal ~msg:"at an issue" string "test_check.ml" (raised_in at_issue));
      test "a failure's 0-d values lie where their data lies" (fun () ->
          let ok = Nx.full_at S.on Nx.bool [| 2 |] false in
          let data =
            Nx.place S.on (Nx.create Nx.float32 [| 2 |] [| 5.; 6. |])
          in
          let lies = ref None in
          Nx.check P.tensor ok data (fun i x ->
              lies := Some (Nx.placement x);
              At (i, Nx.item [] x, 0l, 0.));
          raises (At ([| 0 |], 5., 0l, 0.)) (fun () -> Nx.sync data);
          equal ~msg:"the 0-d value's placement"
            (option (placement ()))
            (Some S.on) !lies);
      test "a failure found at a later check's issue raises at the next wait"
        (fun () ->
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
          equal ~msg:"found at the second issue" int 1 !found;
          raises (At ([| 1 |], 2., 9l, 0.)) (fun () -> Nx.sync (fst data));
          Nx.sync (fst data));
    ]

(* Waits *)

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

(* A set seen through a value on it and two flags, one false at position 1. *)
type side = { wait : wait -> unit; issue : bool -> int -> unit }

let side (type d) (module D : Nx.Devices with type d = d) =
  let x = Nx.place D.on (Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |]) in
  let flags b = Nx.place D.on (Nx.create Nx.bool [| 3 |] [| true; b; true |]) in
  let pass = flags true and fail = flags false in
  {
    wait = (fun w -> wait w x);
    issue =
      (fun fails id ->
        check (if fails then fail else pass) [] (fun _ () -> Failed id));
  }

let s_side = side (module S)
let t_side = side (module T)

(* The checks pending on each set, as [(id, fails)] in issue order. *)
type model = {
  mutable on_s : (int * bool) list;
  mutable on_t : (int * bool) list;
}

type set = On_s | On_t

let sets =
  Gen.of_list
    ~pp:(fun ppf s ->
      Format.pp_print_string ppf (if s = On_s then "S" else "T"))
    [ On_s; On_t ]

let side_of = function On_s -> s_side | On_t -> t_side
let pending md = function On_s -> md.on_s | On_t -> md.on_t

let set_pending md s l =
  match s with On_s -> md.on_s <- l | On_t -> md.on_t <- l

(* A wait drains its set's checks and raises the earliest failure, the later
   ones dropped. *)
let settle md s =
  let rec go = function
    | [] -> None
    | (_, false) :: rest ->
        cover "a wait passes a check before a failing one" (rest <> []);
        go rest
    | (id, true) :: rest ->
        cover "a wait drops a later failure" (List.exists snd rest);
        Some id
  in
  let raised = go (pending md s) in
  set_pending md s [];
  raised

(* Drains a set's checks, so that no test sees another's. *)
let drain side = try side.wait Sync with Failed _ -> ()

(* The checks are the domain's, so every session shares one model, which a start
   empties. A program's end drains what it left. *)
let model = { on_s = []; on_t = [] }

let session =
  abstract "w" ~release:(fun () ->
      drain s_side;
      drain t_side)

let pending_checks =
  stateful ~steps:30
    "a wait drains its set's checks, raising the earliest failure"
    [
      command "start"
        (Gen.unit @-> makes session)
        (fun () ->
          model.on_s <- [];
          model.on_t <- [];
          model)
        (fun () ->
          drain s_side;
          drain t_side);
      command "issue"
        (session ^-> sets @-> Gen.bool @-> Gen.int_range 0 99 @-> returns unit)
        (fun md s fails id -> set_pending md s (pending md s @ [ (id, fails) ]))
        (fun () s fails id -> (side_of s).issue fails id);
      command "wait"
        (session ^-> sets @-> waits @-> returns (option int))
        (fun md s _ ->
          let raised = settle md s in
          cover "a wait on a set with a failure on the other"
            (raised = None
            && List.exists snd (pending md (if s = On_s then On_t else On_s)));
          raised)
        (fun () s w ->
          match (side_of s).wait w with
          | () -> None
          | exception Failed id -> Some id);
    ]

(* At the call *)

let at_call =
  let ok = [| true; false; true |] in
  cases
    ~name:(fun (name, _) -> name)
    "a check raises at the call"
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
    (fun (_, f) -> raises (Failed 0) f)

(* Systhreads *)

(* The systhreads of a domain share its checks. A signal every 20 us of CPU time
   yields the running systhread at its next poll point, so that two issuers
   interleave inside nx's own updates. It is the virtual timer's: the real-time
   one is windtrap's, for its timeouts. Each of [sets] sets gets one failing
   check a round, and every one must raise. *)
let systhreads () =
  let sets =
    Array.init 6000 (fun _ ->
        let module D = (val Nx.devices [ m 0 ]) in
        side (module D))
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
  (* Windows has neither the virtual timer nor its signal: there the
     systhreads interleave at the runtime's own ticks. *)
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

(* Domains, donation and sync *)

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

let others =
  group "domains, donation and sync"
    [
      test "a check raises only on the domain that issued it" (fun () ->
          let raised =
            Domain.join
              (Domain.spawn (fun () ->
                   s_side.issue true 7;
                   match s_side.wait Sync with
                   | () -> None
                   | exception Failed id -> Some id))
          in
          equal (option int) (Some 7) raised;
          Domain.join (Domain.spawn (fun () -> s_side.issue true 8));
          s_side.wait Sync);
      test "a check keeps its data's elements through a donation" (fun () ->
          let d =
            Nx.place S.on (Nx.create Nx.float32 [| 3 |] [| 4.; 5.; 6. |])
          in
          let ok =
            Nx.place S.on (Nx.create Nx.bool [| 3 |] [| true; false; true |])
          in
          check ok [ d ] (fun idx (v, ()) -> At (idx, v, 0l, 0.));
          let negated = Nx.neg (Nx.donate d) in
          raises (At ([| 1 |], 5., 0l, 0.)) (fun () -> Nx.sync negated);
          equal (array float_exact) [| -4.; -5.; -6. |] (Nx.to_array negated));
      test "a donation after a sync reuses the checked data's memory" (fun () ->
          (* A copy owns its memory: a placement onto a memory device borrows
             the host's, which no donation writes. *)
          let d = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
          let d = Nx.copy (Nx.place S.on d) in
          let ok = Nx.create Nx.bool [| 3 |] [| true; true; true |] in
          let ok = Nx.place S.on ok in
          check ok [ d ] (fun idx (v, ()) -> At (idx, v, 0l, 0.));
          Nx.sync d;
          (* The count includes cached memory, which a refused donation would
             take unseen. *)
          Rig.free_cache (m 0);
          let before = Rig.allocated (m 0) in
          ignore (Nx.neg (Nx.donate d));
          equal int before (Rig.allocated (m 0)));
      test "a check that raises at issue holds none of its data" (fun () ->
          let d = Nx.place S.on (Nx.create Nx.float32 [| 1 |] [| 1. |]) in
          let ok = Nx.place S.on (Nx.create Nx.bool [| 1 |] [| false |]) in
          check ok [ d ] (fun _ _ -> raise Exit);
          (* The wait that reads the failing check again ends its hold. *)
          let drain () = try Nx.sync d with Exit -> () in
          Fun.protect ~finally:drain (fun () ->
              let d' = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
              let d' = Nx.copy (Nx.place S.on d') in
              let ok' = Nx.create Nx.bool [| 3 |] [| true; true; true |] in
              let ok' = Nx.place S.on ok' in
              raises Exit (fun () -> check ok' [ d' ] (fun _ _ -> Failed 0));
              Rig.free_cache (m 0);
              let before = Rig.allocated (m 0) in
              ignore (Nx.neg (Nx.donate d'));
              equal int before (Rig.allocated (m 0))));
      test "checks no wait follows hold bounded memory" (fun () ->
          bounded S.on (m 0));
      test "a check on a device then idle leaves later checks bounded"
        (fun () ->
          let on k = Nx.Placement.device Two.v (m k) in
          let p = Nx.create Nx.float32 [| 4 |] (Array.make 4 0.5) in
          ignore (Nx.Rng.bernoulli (Nx.copy (Nx.place (on 3) p)));
          bounded (on 2) (m 2));
      test "a print that raises a failed check prints nothing" (fun () ->
          let d = Nx.place S.on (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]) in
          let ok =
            Nx.place S.on (Nx.create Nx.bool [| 2 |] [| false; true |])
          in
          check ok [ d ] (fun idx (v, ()) -> At (idx, v, 0l, 0.));
          let b = Buffer.create 16 in
          let ppf = Format.formatter_of_buffer b in
          raises (At ([| 0 |], 1., 0l, 0.)) (fun () -> Nx.pp ppf d);
          Format.pp_print_flush ppf ();
          equal string "" (Buffer.contents b));
      test "an empty check passes" (fun () ->
          let ok = Nx.place S.on (Nx.create Nx.bool [| 0; 2 |] [||]) in
          check ok [] (fun _ () -> Failed 0);
          Nx.sync ok);
      test "sync returns for a host creation" (fun () ->
          Nx.sync (Nx.arange Nx.int32 0 3 1));
      test "sync refuses a dead value" (fun () ->
          let x = Nx.place S.on (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]) in
          ignore (Nx.neg (Nx.donate x));
          raises_match (Exn.invalid_arg ~substring:"Nx.sync: ") (fun () ->
              Nx.sync x));
      test "sync refuses a traced value" (fun () ->
          let traced i ~by op =
            Nx.Prim.results ~by (fun _ form -> Nx.Prim.traced i form Mark) op
          in
          Nx.Prim.interpret ~name:"test.trace" Extent traced (fun _ ->
              let x = Nx.neg (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]) in
              raises
                (Invalid_argument "Nx.sync: the value is traced by test.trace")
                (fun () -> Nx.sync x)));
    ]

(* Two systhreads of one domain share its pending checks. The order below is
   held by two semaphores: the issuing thread builds a failure found at its
   second issue and stops inside [fail], where the other thread waits on the
   set, then resumes once that wait returns. *)
let threads =
  group "from two systhreads"
    [
      test "a wait while another thread builds a failure raises it, once"
        (fun () ->
          let ok b = Nx.place S.on (Nx.create Nx.bool [| 2 |] [| true; b |]) in
          let data =
            Nx.place S.on (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |])
          in
          let go = Semaphore.Binary.make false in
          let waited = Semaphore.Binary.make false in
          let builder = Thread.id (Thread.self ()) and stopped = ref false in
          let failing i x =
            let e = At (i, Nx.item [] x, 0l, 0.) in
            if Thread.id (Thread.self ()) = builder && not !stopped then begin
              stopped := true;
              Semaphore.Binary.release go;
              Semaphore.Binary.acquire waited
            end;
            e
          in
          let raised = ref None in
          let other =
            Thread.create
              (fun () ->
                Semaphore.Binary.acquire go;
                (raised :=
                   match Nx.sync data with () -> None | exception e -> Some e);
                Semaphore.Binary.release waited)
              ()
          in
          Nx.check P.tensor (ok false) data failing;
          Nx.check P.tensor (ok true) data failing;
          (* Without the stop the other thread waits after the issues. *)
          if not !stopped then Semaphore.Binary.release go;
          Thread.join other;
          let again =
            match Nx.sync data with () -> None | exception e -> Some e
          in
          equal ~msg:"the failure is built at the second issue" bool true
            !stopped;
          (match !raised with
          | Some (At (i, v, _, _)) ->
              equal ~msg:"the index" (array int) [| 1 |] i;
              equal ~msg:"the element" float_exact 2. v
          | Some e -> raise e
          | None -> fail "the other thread's wait passed a failed check");
          match again with
          | None -> ()
          | Some (At _) -> fail "a later wait raised the failure again"
          | Some e -> raise e);
    ]

(* One array, whose walk passes the array it is given through [f]. *)
let through f = P.v (fun c x -> f (P.Walk.tensor c x))

let tried =
  group "the rebuild tried at the call"
    [
      test "is over 0-d stand-ins at each array's placement, uncut" (fun () ->
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
              equal ~msg:"the stand-in's placement" (placement ()) S2.on lies);
      test "computes nothing on the devices" (fun () ->
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
          let bare = calls (fun () -> Nx.check P.unit ok () pass) in
          let full =
            calls (fun () -> Nx.check P.(pair tensor tensor) ok data pass)
          in
          equal ~msg:"kernel calls with data, as without" int bare full);
      test "raises where the walk computes on a stand-in" (fun () ->
          let ok = Nx.create Nx.bool [| 2 |] [| true; false |] in
          let x = Nx.create Nx.float32 [| 2 |] [| 1.; 2. |] in
          raises
            (Invalid_argument
               "Nx.check: the structure's walk refuses 0-d arrays: Nx.add: a \
                value traced by Nx.check reached eager execution") (fun () ->
              Nx.check (through (fun y -> Nx.add y y)) ok x pass));
    ]

let walks =
  let key () = Nx.Rng.key 7 and keys () = P.v Nx.Rng.walk in
  group "walks that refuse 0-d arrays"
    [
      test "raise at the call on the host, naming Nx.check" (fun () ->
          let ok = Nx.create Nx.bool [| 2 |] [| true; false |] in
          invalid ~by:"Nx.check" (fun () -> Nx.check (keys ()) ok (key ()) pass));
      test "raise at the call on another set, leaving nothing pending"
        (fun () ->
          let ok =
            Nx.place S.on (Nx.create Nx.bool [| 2 |] [| true; false |])
          in
          let key = Nx.Rng.place S.on (key ()) in
          invalid ~by:"Nx.check" (fun () -> Nx.check (keys ()) ok key pass);
          Nx.sync ok;
          Nx.check P.unit
            (Nx.place S.on (Nx.create Nx.bool [| 1 |] [| true |]))
            ()
            (fun i () -> Flag i);
          Nx.sync ok);
    ]

let () =
  exit
    (run "nx check"
       [
         first_false;
         host;
         elsewhere;
         pending_checks;
         at_call;
         threads;
         test "checks issued from two systhreads all raise" systhreads;
         tried;
         walks;
         others;
       ])
