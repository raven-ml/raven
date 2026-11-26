(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module L = Rig.Local

let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false

(* A name no other case opened. *)
let names = Atomic.make 0

let fresh what =
  Printf.sprintf "local:%s-%d" what (Atomic.fetch_and_add names 1)

let software what = Result.get_ok (Rig.software (fresh what))

(* A value whose collection a weak pointer shows: a fresh block each time. *)
type value = { id : int }

let ids = Atomic.make 0
let made () = { id = Atomic.fetch_and_add ids 1 }

(* The id of the value a weak pointer holds, while it holds one. *)
let held w = Option.map (fun v -> v.id) (Weak.get w 0)

(* Values per device *)

(* The model of a device: why it is lost, once it is, whether an open of its
   name replaced it, the value its gets answer once one answered, and the values
   of the devices its name had before. *)
type model = {
  mutable lost : string option;
  mutable replaced : bool;
  mutable value : int option;
  before : int list;
}

(* A name and its device. *)
type device = { name : string; d : Rig.t }

let device_t =
  abstract "d"
    ~release:(fun s -> Rig.close s.d)
    ~pp:(fun ppf m ->
      Format.fprintf ppf "%s, %s"
        (Option.value m.lost ~default:"open")
        (match m.value with None -> "no value" | Some v -> string_of_int v))

(* Made once, as rig.mli asks of a key. *)
let key = L.new_key (fun _ -> made ())

let open_named () =
  let name = fresh "model" in
  { name; d = Result.get_ok (Rig.software name) }

(* Opens the name anew: an open of a lost device's name runs its owed stop, and
   a device still open is closed first. *)
let reopen s =
  if Rig.lost s.d = None then Rig.close s.d;
  { s with d = Result.get_ok (Rig.software s.name) }

let lose m = if m.lost = None then m.lost <- Some "given up"

let get_judged m outcome =
  match (outcome, m.lost) with
  | Error (Rig.Lost (_, why)), Some why' -> equal string why' why
  | Ok v, None -> (
      match m.value with
      | Some v' -> equal ~msg:"the value every get answers" int v' v
      | None ->
          equal ~msg:"values of the name's earlier devices" (list int) []
            (List.filter (( = ) v) m.before);
          m.value <- Some v)
  | Ok v, Some why -> failf "the get of a device lost (%s) answered %d" why v
  | Error e, _ -> raise e

let model_commands =
  [
    command "open"
      (Gen.unit @-> makes device_t)
      (fun () -> { lost = None; replaced = false; value = None; before = [] })
      open_named;
    command "reopen"
      (device_t ^-> makes device_t)
      ~pre:(fun m -> not m.replaced)
      (fun m ->
        let before = Option.to_list m.value @ m.before in
        if m.lost = None then m.lost <- Some "closed";
        m.replaced <- true;
        { lost = None; replaced = false; value = None; before })
      reopen;
    command "get"
      (device_t ^-> judges int)
      get_judged
      (fun s -> (L.get key s.d).id);
    command "lose"
      (device_t ^-> returns unit)
      lose
      (fun s -> Rig.lose s.d "given up");
    command "close"
      (device_t ^-> returns unit)
      (fun m -> if m.lost = None then m.lost <- Some "closed")
      (fun s -> Rig.close s.d);
  ]

(* Gets [k] on [d] into [w], and leaves no other reference to the value. *)
let[@inline never] [@local never] get_weak k d w =
  let v = L.get k d in
  Weak.set w 0 (Some v);
  v.id

(* The ways that run a device's stop. *)
let stops =
  [
    ("a close", fun _ d -> Rig.close d);
    ( "a loss, then a close",
      fun _ d ->
        Rig.lose d "given up";
        Rig.close d );
    ( "a loss, then an open of its name",
      fun name d ->
        Rig.lose d "given up";
        Rig.close (Result.get_ok (Rig.software name)) );
  ]

let test_dropped_at_stop (_, stop) =
  let name = fresh "dropped" in
  let d = Result.get_ok (Rig.software name) in
  let k = L.new_key (fun _ -> made ()) in
  let w = Weak.create 1 in
  let id = get_weak k d w in
  Gc.full_major ();
  equal ~msg:"while open" (option int) (Some id) (held w);
  stop name d;
  Gc.full_major ();
  equal ~msg:"once stopped" (option int) None (held w)

(* An init that closes its device: the store comes after the stop. *)
let test_init_stops () =
  let d = software "races" in
  let w = Weak.create 1 in
  let k =
    L.new_key (fun d ->
        Rig.close d;
        let v = made () in
        Weak.set w 0 (Some v);
        v)
  in
  raises_match (lost d) (fun () -> L.get k d);
  Gc.full_major ();
  equal (option int) None (held w)

(* The init closes its device and opens another, which continues its slot, and
   gets the key there: the outer get must not answer that device's value. *)
let test_init_reopens () =
  let d = software "first" in
  ignore (Rig.submit (Rig.Submission.make d [||]) ~buffers:[||] ~args:[||]);
  let key = ref None and other = ref None in
  let k =
    L.new_key (fun dev ->
        if Rig.equal dev d then (
          Rig.close d;
          let e = software "second" in
          equal ~msg:"e continues d's slot" int (Rig.submitted d)
            (Rig.submitted e);
          other := Some (e, L.get (Option.get !key) e));
        made ())
  in
  key := Some k;
  raises_match (lost d) (fun () -> L.get k d);
  let e, v = Option.get !other in
  equal ~msg:"e's value" int v.id (L.get k e).id;
  Rig.close e

(* Local.get *)

let test_keys () =
  let d = software "keys" and e = software "keys" in
  let k = L.new_key (fun d -> Rig.name d)
  and k' = L.new_key (fun _ -> made ()) in
  equal string (Rig.name d) (L.get k d);
  equal string (Rig.name e) (L.get k e);
  not_equal int (L.get k' d).id (L.get k' e).id;
  equal string "CPU" (L.get k Rig.host);
  Rig.close d;
  Rig.close e

let test_hit_words () =
  let d = software "words" in
  let k = L.new_key (fun _ -> made ()) in
  ignore (L.get k d);
  equal int 0
    (Rig_support.allocated_bytes (fun () ->
         for _ = 1 to 10_000 do
           ignore (Sys.opaque_identity (L.get k d))
         done));
  Rig.close d

let test_init_domain () =
  let d = software "domain" in
  let k = L.new_key (fun _ -> (Domain.self () :> int)) in
  let caller, value =
    Domain.join (Domain.spawn (fun () -> ((Domain.self () :> int), L.get k d)))
  in
  equal int caller value;
  Rig.close d

let test_init_gets_other () =
  let d = software "other" in
  let inner = L.new_key (fun _ -> made ()) in
  let outer = L.new_key (fun d -> (L.get inner d).id) in
  let v = L.get outer d in
  equal ~msg:"the other key's value" int (L.get inner d).id v;
  Rig.close d

(* Two domains get one key on one device with no value. A's init waits until B's
   get, whose init ran and stored, has returned: A's value comes second. *)
let test_race_held () =
  let d = software "held" in
  let a_in_init = Semaphore.Binary.make false
  and b_returned = Semaphore.Binary.make false in
  let first = Atomic.make true in
  let k =
    L.new_key (fun _ ->
        if Atomic.exchange first false then (
          Semaphore.Binary.release a_in_init;
          Semaphore.Binary.acquire b_returned);
        made ())
  in
  let b =
    Domain.spawn (fun () ->
        Semaphore.Binary.acquire a_in_init;
        let v = L.get k d in
        Semaphore.Binary.release b_returned;
        v)
  in
  let a = L.get k d in
  let b = Domain.join b in
  equal ~msg:"A's get answers B's value" int b.id a.id;
  equal ~msg:"a later get" int b.id (L.get k d).id;
  Rig.close d

(* Two keys made once, and two devices per program, so a get's walk stays at
   most two keys long. Gets of both keys on one device race their stores. *)
let racing_keys =
  [| L.new_key (fun _ -> made ()); L.new_key (fun _ -> made ()) |]

let pair_t = abstract "p" ~release:(Array.iter Rig.close)
let index = Gen.int_range 0 1

(* The value of key [k] on device [i], at [values.(k).(i)]. *)
let racing_judged values k i outcome =
  match outcome with
  | Error e -> raise e
  | Ok v -> (
      match values.(k).(i) with
      | Some v' -> equal ~msg:"the value every get answers" int v' v
      | None -> values.(k).(i) <- Some v)

let racing_commands =
  [
    command "devices"
      (Gen.unit @-> makes pair_t)
      (fun () -> Array.init 2 (fun _ -> [| None; None |]))
      (fun () -> Array.init 2 (fun _ -> software "racing"));
    command "get"
      (pair_t ^-> index @-> index @-> judges int)
      racing_judged
      (fun ds k i -> (L.get racing_keys.(k) ds.(i)).id);
  ]

let test_nested () =
  let d = software "nested" in
  let depth = ref 0 in
  let rec k =
    lazy
      (L.new_key (fun d ->
           incr depth;
           if !depth = 1 then ignore (L.get (Lazy.force k) d);
           made ()))
  in
  let k = Lazy.force k in
  let outer = L.get k d in
  equal ~msg:"inits" int 2 !depth;
  equal ~msg:"the nested value" int outer.id (L.get k d).id;
  Rig.close d

let test_init_raises () =
  let d = software "raises" in
  let inits = ref 0 in
  let k =
    L.new_key (fun _ ->
        incr inits;
        if !inits = 1 then failwith "first" else made ())
  in
  raises (Failure "first") (fun () -> L.get k d);
  let v = L.get k d in
  equal ~msg:"inits" int 2 !inits;
  equal ~msg:"the stored value" int v.id (L.get k d).id;
  Rig.close d

let losses =
  [
    ("by a close", fun d -> Rig.close d);
    ("by lose", fun d -> Rig.lose d "given up");
  ]

let test_lost (_, lose) =
  let d = software "lost" in
  let k = L.new_key (fun _ -> made ()) and k' = L.new_key (fun _ -> made ()) in
  ignore (L.get k d);
  lose d;
  raises_match ~msg:"with a value" (lost d) (fun () -> L.get k d);
  raises_match ~msg:"without" (lost d) (fun () -> L.get k' d);
  Rig.close d

let tests =
  [
    group "values per device"
      [
        stateful "a device's gets answer one value, its own, until it is lost"
          model_commands;
        cases ~id:fst "a device's value is collected once its stop has run"
          stops test_dropped_at_stop;
        test "a get whose init closes its device keeps nothing" test_init_stops;
        test
          "a get whose init closes its device raises Lost, and a later device \
           of its slot keeps its own value"
          test_init_reopens;
      ];
    group "Local.get"
      [
        test "each key and device has its own value" test_keys;
        test "a get that finds its value allocates nothing" test_hit_words;
        test "an init runs on the calling domain" test_init_domain;
        test "an init may get another key" test_init_gets_other;
        test "a get racing another's init answers the first value stored"
          test_race_held;
        stateful ~domains:2 ~steps:1
          "gets racing on a device with no value all answer one value"
          racing_commands;
        test "an init that gets its own key answers the nested value"
          test_nested;
        test "an init that raises stores nothing" test_init_raises;
        cases ~id:fst "a lost device's get raises Lost, with a value or not"
          losses test_lost;
      ];
  ]

let () = exit (run "rig.local" tests)
