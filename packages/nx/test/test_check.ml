(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Checks and waits: the first false flag in C order raises the caller's
   exception with its index and the data's elements there, each array broadcast
   to the flags' shape; at the call on the host, and on another set at a wait on
   that set, on any domain, one failure per wait, oldest first, each once. A
   pending check holds none of its data, and checks no wait follows hold a
   bounded window. *)

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

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f
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
      cases "the data's elements reach the exception at every dtype"
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

(* The data's elements at the first false flag, of one flag or of several, at
   every dtype, on a device. *)
let dtypes_elsewhere =
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
      raises (Element (show vs.(flags - 1))) (fun () -> Nx.sync ok))

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
          (* The failure is found at the second issue, and [fail] runs at the
             wait. *)
          let found_at_issue () =
            Nx.check P.unit (ok false) () raise_exit;
            Nx.check P.unit (ok true) () raise_exit;
            Nx.sync (ok true)
          in
          equal ~msg:"at a wait" string "test_check.ml" (raised_in at_wait);
          equal ~msg:"found at an issue" string "test_check.ml"
            (raised_in found_at_issue));
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
      test "a failure's exception is built at the wait that raises it"
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
          equal ~msg:"built at no issue" int 0 !found;
          raises (At ([| 1 |], 2., 9l, 0.)) (fun () -> Nx.sync (fst data));
          equal ~msg:"built at the wait" int 1 !found;
          Nx.sync (fst data);
          equal ~msg:"built once" int 1 !found);
      test "a datum cut along an inner axis gives its element" (fun () ->
          let at = S2.split ~axis:1 in
          let ok = Array.init 8 (fun k -> k <> 5) in
          let ok = Nx.place at (Nx.create Nx.bool [| 2; 4 |] ok) in
          let xs = Array.init 8 float_of_int in
          let data = Nx.place at (Nx.create Nx.float32 [| 2; 4 |] xs) in
          Nx.check P.tensor ok data (fun i x -> At (i, Nx.item [] x, 0l, 0.));
          raises (At ([| 1; 1 |], 5., 0l, 0.)) (fun () -> Nx.sync data));
      test "a flipped datum gives its element" (fun () ->
          let ok = Array.init 8 (fun k -> k <> 5) in
          let ok = Nx.place S.on (Nx.create Nx.bool [| 8 |] ok) in
          let xs = Array.init 8 float_of_int in
          let data = Nx.flip (Nx.place S.on (Nx.create Nx.float32 [| 8 |] xs)) in
          Nx.check P.tensor ok data (fun i x -> At (i, Nx.item [] x, 0l, 0.));
          raises (At ([| 5 |], 2., 0l, 0.)) (fun () -> Nx.sync data));
      test "every failure raises, oldest first, one per wait" (fun () ->
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
          Nx.sync (ok true));
    ]

(* Pending over several devices

   Two devices whose memory the host addresses, with nx.cpu's kernels, and a
   device whose copies into them run only when the host waits: flags and
   data placed through it are unwritten when the check is issued, so the
   check stays pending until a wait. *)

let cut_a, _ =
  Rig_support.Polled.open_ ~capacity:(1 lsl 16) ~host_visible:true
    ~waits_on:[ `Host ] "nx-check-cut-a"

let cut_b, _ =
  Rig_support.Polled.open_ ~capacity:(1 lsl 16) ~host_visible:true
    ~waits_on:[ `Host ] "nx-check-cut-b"

let cut_from, _ =
  Rig_support.Polled.open_ ~capacity:(1 lsl 16) ~host_visible:false
    ~waits_on:[ `Host ] "nx-check-cut-from"

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
        (* Cut along the inner axis: position 2 lies on the second device, 4
           on the first. *)
        ([| 2; 4 |], 1, [ 4; 2 ]);
        ([| 3; 4 |], 0, [ 9; 1 ]);
        ([| 1 |], 0, [ 0 ]);
        ([| 4 |], 0, []);
      ]
    "a check pending over several devices raises its first false in C order"
    pending_cuts pending_cut

(* A datum that is a sum of the devices' terms ([~sum:true]) gives its closed
   element at the first false, pending or read at its issue. *)
let summed_datum () =
  let a = Array.init 8 float_of_int and b = Array.init 12 float_of_int in
  let product i j =
    List.fold_left (fun t k -> t +. (a.((4 * i) + k) *. b.((3 * k) + j))) 0.
      [ 0; 1; 2; 3 ]
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

(* On two devices the host does not address, a check of one flag on the
   second, its datum on the first: the datum's element moves to the second,
   and the second copies it and then the flag into memory the host
   addresses, by the value it has submitted once the check is issued, which
   the check is read after. *)
let apart_off_host () =
  let p0, q0 =
    Rig_support.Polled.open_ ~capacity:(1 lsl 16) ~host_visible:false
      ~waits_on:[ `Host ] "nx-check-apart-0"
  in
  let p1, q1 =
    Rig_support.Polled.open_ ~capacity:(1 lsl 16) ~host_visible:false
      ~waits_on:[ `Host ] "nx-check-apart-1"
  in
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
  equal ~msg:"copies into memory the host addresses, before the mark"
    (list (pair int bool))
    [ (4, true); (1, true) ]
    into_host;
  raises (At ([||], 5., 0l, 0.)) (fun () -> Nx.sync ok)

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

(* A set seen through a value on it and two flags of [n] elements, one false
   at its last position. *)
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

(* S's checks are read at their issue, since its device runs its work as it
   is submitted. Held's device runs its own queue at times no test chooses, so
   its checks of one flag stay pending until a wait or an issue reads them,
   and two readers may meet. *)
let pending_device, _ =
  Rig_support.Polled.open_ ~capacity:(1 lsl 16) ~host_visible:false
    ~runs:`Itself ~waits_on:[ `Host ] "nx-check-pending"

module Held = (val Nx.devices [ pending_device ])

let s_side = side (module S) 3
let t_side = side (module Held) 1

(* The checks pending on each set, as [(id, fails)] in issue order. *)
type model = {
  mutable on_s : (int * bool) list;
  mutable on_t : (int * bool) list;
}

type set = On_s | On_t

let sets =
  Gen.of_list
    ~pp:(fun ppf s ->
      Format.pp_print_string ppf (if s = On_s then "S" else "Held"))
    [ On_s; On_t ]

let side_of = function On_s -> s_side | On_t -> t_side
let pending md = function On_s -> md.on_s | On_t -> md.on_t

let set_pending md s l =
  match s with On_s -> md.on_s <- l | On_t -> md.on_t <- l

(* A wait raises its set's oldest failure and leaves the later ones pending. *)
let settle md s =
  let rec go = function
    | [] -> ([], None)
    | (_, false) :: rest ->
        cover "a wait passes a check before a failing one" (rest <> []);
        go rest
    | (id, true) :: rest ->
        cover "a wait leaves a later failure pending" (List.exists snd rest);
        (rest, Some id)
  in
  let left, raised = go (pending md s) in
  set_pending md s left;
  raised

(* Raises every failure pending on a set, so that no test sees another's. *)
let rec drain side =
  match side.wait Sync with () -> () | exception Failed _ -> drain side

(* The checks are the process's, so every session shares one model, which a
   start empties. A program's end drains what it left. *)
let model = { on_s = []; on_t = [] }

let session =
  abstract "w" ~release:(fun () ->
      drain s_side;
      drain t_side)

(* On two domains each wait and issue is explained by some order of the calls:
   a wait raises a failure whichever domain issued it. *)
let pending_checks ?count ?domains name =
  stateful ?count ?domains ~steps:30 name
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
  cases "a check raises at the call"
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
      test "a wait raises a failure another domain issued" (fun () ->
          Domain.join (Domain.spawn (fun () -> s_side.issue true 8));
          raises (Failed 8) (fun () -> s_side.wait Sync);
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
          (* A copy owns its memory: a placement onto a software device borrows
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
      cases "a donation of checked data before any wait reuses its memory"
        ~id:(fun (n, fails) ->
          let outcome = if fails then "failing" else "passing" in
          Printf.sprintf "%d flags, %s" n outcome)
        [ (1, false); (1, true); (3, false); (3, true) ]
        (fun (n, fails) ->
          (* A copy owns its memory: a placement onto a software device borrows
             the host's, which no donation writes. *)
          let d = Nx.create Nx.float32 [| n |] (Array.init n float_of_int) in
          let d = Nx.copy (Nx.place S.on d) in
          let ok = Array.init n (fun k -> not (fails && k = n - 1)) in
          let ok = Nx.place S.on (Nx.create Nx.bool [| n |] ok) in
          check ok [ d ] (fun idx (v, ()) -> At (idx, v, 0l, 0.));
          (* The count includes cached memory, which a refused donation would
             take unseen. *)
          Gc.full_major ();
          Rig.free_cache (m 0);
          let before = Rig.allocated (m 0) in
          let negated = Nx.neg (Nx.donate d) in
          equal ~msg:"bytes allocated" int before (Rig.allocated (m 0));
          if fails then
            raises
              (At ([| n - 1 |], float_of_int (n - 1), 0l, 0.))
              (fun () -> Nx.sync negated);
          Nx.sync negated);
      test "checks no wait follows hold bounded memory" (fun () ->
          bounded S.on (m 0));
      test "a check on a device then idle leaves later checks bounded"
        (fun () ->
          let on k = Nx.Placement.device Two.v (m k) in
          let p = Nx.create Nx.float32 [| 4 |] (Array.make 4 0.5) in
          ignore (Nx.Rng.bernoulli (Nx.copy (Nx.place (on 3) p)));
          bounded (on 2) (m 2));
      test "Nx.check holds none of its data while a failure is pending"
        (fun () ->
          let n = 1 lsl 16 in
          let ok = Array.init n (fun k -> k <> n - 1) in
          let ok = Nx.place S.on (Nx.create Nx.bool [| n |] ok) in
          (* The data, a copy that owns its memory, is reachable from no value
             of the test once [issue] returns. *)
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
          raises (At ([| n - 1 |], 1., 0l, 0.)) (fun () -> Nx.sync ok));
      test "a read that raises takes its check off" (fun () ->
          (* A Polled device runs its queue only when the test runs it, so its
             copy of the flag is still queued when it is lost. *)
          let lost, _ =
            Rig_support.Polled.open_ ~capacity:(1 lsl 16) ~host_visible:false
              ~waits_on:[ `Host ] "nx-check-lost"
          in
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
          Nx.sync x);
      test "an issue whose earlier check raises keeps the new check" (fun () ->
          let lost, _ =
            Rig_support.Polled.open_ ~capacity:(1 lsl 16) ~host_visible:false
              ~waits_on:[ `Host ] "nx-check-lost-at-issue"
          in
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
          Nx.sync x);
      test "a check kept in place holds its data until it is read" (fun () ->
          (* The host addresses [a], and [b] copies into it, in a queue the
             test runs, so the flag and the datum are unwritten at issue. *)
          let a, _ =
            Rig_support.Polled.open_ ~capacity:(1 lsl 16) ~host_visible:true
              ~waits_on:[ `Host ] "nx-check-held"
          in
          let b, bq =
            Rig_support.Polled.open_ ~capacity:(1 lsl 16) ~host_visible:false
              ~waits_on:[ `Host ] "nx-check-held-from"
          in
          let module A = (val Nx.devices ~kernels:(module Nx_cpu) [ a ]) in
          let module B = (val Nx.devices [ b ]) in
          let via x = Nx.place A.on (Nx.place B.on x) in
          let ok = via (Nx.create Nx.bool [| 1 |] [| false |]) in
          let d = via (Nx.create Nx.float32 [| 1 |] [| 5. |]) in
          check ok [ d ] (fun idx (v, ()) -> At (idx, v, 0l, 0.));
          while Rig_support.Polled.run bq > 0 do
            ()
          done;
          let allocates f =
            Rig.free_cache a;
            let before = Rig.allocated a in
            ignore (f ());
            Rig.allocated a > before
          in
          equal ~msg:"a donation before the read allocates" bool true
            (allocates (fun () -> Nx.neg (Nx.donate d)));
          raises (At ([| 0 |], 5., 0l, 0.)) (fun () -> Nx.sync ok);
          equal ~msg:"a donation after the read allocates" bool false
            (allocates (fun () -> Nx.logical_not (Nx.donate ok))));
      test "an issue returns while a wait holds its set's claim" (fun () ->
          (* The wait reads a check whose flag a Polled device copies, and
             sleeps at the device's gate, holding the claim; the issue on the
             set's other device, on a thread of its own, returns without
             reading. Only a failure waits for the bound, then opens the
             gate. *)
          let polled, pq =
            Rig_support.Polled.open_ ~capacity:(1 lsl 16) ~host_visible:false
              ~waits_on:[ `Host ] "nx-check-claimed"
          in
          let module L = (val Nx.devices [ polled; m 2 ]) in
          let on d = Nx.Placement.device L.v d in
          let flag d b = Nx.place (on d) (Nx.create Nx.bool [| 1 |] [| b |]) in
          let x = Nx.place (on (m 2)) (Nx.create Nx.float32 [| 1 |] [| 0. |]) in
          check (flag polled false) [] (fun _ () -> Failed 1);
          Rig_support.Polled.gate pq;
          let raised = ref None and waited = Atomic.make false in
          let waiter =
            Thread.create
              (fun () ->
                (raised :=
                   match Nx.sync x with () -> None | exception e -> Some e);
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
          let flag = flag (m 2) true and issued = Atomic.make false in
          let issuer =
            Thread.create
              (fun () ->
                check flag [] (fun _ () -> Failed 2);
                Atomic.set issued true)
              ()
          in
          let bound = Unix.gettimeofday () +. 5. in
          while (not (Atomic.get issued)) && Unix.gettimeofday () < bound do
            Thread.delay 0.001
          done;
          let returned = Atomic.get issued in
          let sleeping = Rig_support.Polled.sleepers pq in
          Rig_support.Polled.open_gate pq;
          Thread.join issuer;
          Thread.join waiter;
          equal ~msg:"the issue returned while the wait slept" bool true
            returned;
          equal ~msg:"the wait slept" int 1 sleeping;
          (match !raised with
          | Some (Failed 1) -> ()
          | Some e -> raise e
          | None -> fail "the wait raised no failure");
          Nx.sync x);
      test "an issue beside a check kept in place waits for nothing"
        (fun () ->
          (* As above, and a second check issued while [b]'s copies into the
             first check's buffers are still queued. A host wait on them would
             run [b]'s queue, as a wait sleeps, so the queue is how many it
             left. *)
          let a, _ =
            Rig_support.Polled.open_ ~capacity:(1 lsl 16) ~host_visible:true
              ~waits_on:[ `Host ] "nx-check-held-beside"
          in
          let b, bq =
            Rig_support.Polled.open_ ~capacity:(1 lsl 16) ~host_visible:false
              ~waits_on:[ `Host ] "nx-check-held-beside-from"
          in
          let module A = (val Nx.devices ~kernels:(module Nx_cpu) [ a ]) in
          let module B = (val Nx.devices [ b ]) in
          let via x = Nx.place A.on (Nx.place B.on x) in
          let ok = via (Nx.create Nx.bool [| 1 |] [| false |]) in
          let d = via (Nx.create Nx.float32 [| 1 |] [| 5. |]) in
          check ok [ d ] (fun idx (v, ()) -> At (idx, v, 0l, 0.));
          let written = Nx.place A.on (Nx.create Nx.bool [| 1 |] [| true |]) in
          let queued = Rig_support.Polled.queued bq in
          check written [] (fun _ () -> Failed 0);
          equal ~msg:"copies still queued" int queued
            (Rig_support.Polled.queued bq));
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
          raises_match (Exn.invalid_arg ~sub:"Nx.sync: ") (fun () -> Nx.sync x));
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

(* Two systhreads of one process share its pending checks. The order below is
   held by two semaphores: the main thread's wait takes the oldest failure and
   stops inside its [fail], after a read of the value [fail] receives, where
   the other thread waits on the set, then resumes once that wait returns. *)
let threads =
  group "from two systhreads"
    [
      test
        "a wait inside fail raises none, and another thread's raises the next"
        (fun () ->
          let ok =
            Nx.place S.on (Nx.create Nx.bool [| 2 |] [| true; false |])
          in
          let data =
            Nx.place S.on (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |])
          in
          let go = Semaphore.Binary.make false in
          let waited = Semaphore.Binary.make false in
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
          Nx.sync data);
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
          let prim =
            calls (fun () ->
                check ok [ fst data; snd data ] (fun _ _ -> Failure "checked"))
          in
          let full =
            calls (fun () -> Nx.check P.(pair tensor tensor) ok data pass)
          in
          equal ~msg:"kernel calls of the primitive check, as of Nx.check" int
            prim full);
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
  let key () = Nx.Rng.key 7 in
  group "walks that refuse 0-d arrays"
    [
      test "raise at the call on the host, naming Nx.check" (fun () ->
          let ok = Nx.create Nx.bool [| 2 |] [| true; false |] in
          invalid ~by:"Nx.check" (fun () ->
              Nx.check Nx.Rng.ptree ok (key ()) pass));
      test "raise at the call on another set, leaving nothing pending"
        (fun () ->
          let ok =
            Nx.place S.on (Nx.create Nx.bool [| 2 |] [| true; false |])
          in
          let key = Nx.Rng.place S.on (key ()) in
          invalid ~by:"Nx.check" (fun () -> Nx.check Nx.Rng.ptree ok key pass);
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
         dtypes_elsewhere;
         elsewhere;
         pending_over_devices;
         test "a datum that is a sum gives its closed element" summed_datum;
         test "a check of one flag off host memory copies before its mark"
           apart_off_host;
         pending_checks "a wait raises its set's oldest failure, once";
         pending_checks ~count:20 ~domains:2
           "on two domains, a wait raises its set's oldest failure, once";
         at_call;
         threads;
         test "checks issued from two systhreads all raise" systhreads;
         tried;
         walks;
         others;
       ])
