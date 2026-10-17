(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module L = Rig.Local

let timeout = Rig_support.time 60.
let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false

(* A name no other case opened. *)
let names = Atomic.make 0
let fresh what = Printf.sprintf "local:%s-%d" what (Atomic.fetch_and_add names 1)

(* A value whose collection a weak pointer shows: a fresh block each time. *)
type value = { id : int }

let ids = Atomic.make 0
let made () = { id = Atomic.fetch_and_add ids 1 }

(* Gets, closes and reopens *)

(* The model of a device: whether it is closed, whether a reopen replaced it
   as its name's device, the value its gets answer once one answered, and the
   values of the devices its name had before. *)
type model = {
  mutable closed : bool;
  mutable replaced : bool;
  mutable value : int option;
  before : int list;
}

(* A name and its device. *)
type device = { name : string; d : Rig.t }

let device_t =
  abstract "d" ~release:(fun s -> Rig.close s.d) ~pp:(fun ppf m ->
      Format.fprintf ppf "%s, %s"
        (if m.closed then "closed" else "open")
        (match m.value with None -> "no value" | Some v -> string_of_int v))

let key = L.new_key (fun _ -> made ())

let open_named () =
  let name = fresh "model" in
  { name; d = Result.get_ok (Rig.memory_device name) }

(* Closes the name's device if open, then opens the name anew. *)
let reopen s =
  Rig.close s.d;
  { s with d = Result.get_ok (Rig.memory_device s.name) }

let get_judged m outcome =
  match (outcome, m.closed) with
  | Error (Rig.Lost (_, why)), true -> equal string "closed" why
  | Ok v, false -> (
      match m.value with
      | Some v' -> equal ~msg:"the value every get answers" int v' v
      | None ->
          equal ~msg:"a value of the name's earlier devices" bool false
            (List.mem v m.before);
          m.value <- Some v)
  | Ok v, true -> failf "a closed device's get answered %d" v
  | Error e, _ -> raise e

let model_commands =
  [
    command "open" (Gen.unit @-> makes device_t)
      (fun () -> { closed = false; replaced = false; value = None; before = [] })
      open_named;
    command "reopen" (device_t ^-> makes device_t)
      ~pre:(fun m -> not m.replaced)
      (fun m ->
        let before = Option.to_list m.value @ m.before in
        m.closed <- true;
        m.replaced <- true;
        { closed = false; replaced = false; value = None; before })
      reopen;
    command "get" (device_t ^-> judges int) get_judged (fun s ->
        (L.get key s.d).id);
    command "close" (device_t ^-> returns unit)
      (fun m -> m.closed <- true)
      (fun s -> Rig.close s.d);
  ]

(* Racing gets *)

(* Every device a domain's get draws from, opened once for every case: a
   program makes keys, which cost no device. *)
let racing_devices =
  Array.init 2 (fun _ -> Result.get_ok (Rig.memory_device (fresh "racing")))

(* The model of a key: the value its gets answer on each device, once one
   answered. *)
let key_t = abstract "k"
let device_index = Gen.int_range 0 (Array.length racing_devices - 1)

let racing_judged values i outcome =
  match outcome with
  | Error e -> raise e
  | Ok v -> (
      match values.(i) with
      | Some v' -> equal ~msg:"the value every get answers" int v' v
      | None -> values.(i) <- Some v)

let racing_commands =
  [
    command "key" (Gen.unit @-> makes key_t)
      (fun () -> Array.make (Array.length racing_devices) None)
      (fun () -> L.new_key (fun _ -> made ()));
    command "get"
      (key_t ^-> device_index @-> judges int)
      racing_judged
      (fun k i -> (L.get k racing_devices.(i)).id);
  ]

(* Values *)

(* An init that gets its own key on its device runs a nested init, whose value
   both answer. *)
let test_nested () =
  let d = Result.get_ok (Rig.memory_device (fresh "nested")) in
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

(* An init that raises stores nothing: the next get runs init again. *)
let test_init_raises () =
  let d = Result.get_ok (Rig.memory_device (fresh "raises")) in
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

(* Keys are independent, and a device's value is its own. *)
let test_keys () =
  let d = Result.get_ok (Rig.memory_device (fresh "keys"))
  and e = Result.get_ok (Rig.memory_device (fresh "keys")) in
  let k = L.new_key (fun d -> Rig.name d) and k' = L.new_key (fun _ -> made ()) in
  equal string (Rig.name d) (L.get k d);
  equal string (Rig.name e) (L.get k e);
  not_equal int (L.get k' d).id (L.get k' e).id;
  equal string "CPU" (L.get k Rig.host);
  Rig.close d;
  Rig.close e

(* A get that finds its value allocates nothing. *)
let test_hit_words () =
  let d = Result.get_ok (Rig.memory_device (fresh "words")) in
  let k = L.new_key (fun _ -> made ()) in
  ignore (L.get k d);
  let before = Gc.minor_words () in
  for _ = 1 to 10_000 do
    ignore (Sys.opaque_identity (L.get k d))
  done;
  equal int 0 (int_of_float (Gc.minor_words () -. before) / 10_000);
  Rig.close d

(* Lifetime *)

let[@inline never] get_weak k d w = Weak.set w 0 (Some (L.get k d))

(* A closed device's value is collected: close drops it. *)
let test_closed_collected () =
  let d = Result.get_ok (Rig.memory_device (fresh "collected")) in
  let k = L.new_key (fun _ -> made ()) in
  let w = Weak.create 1 in
  get_weak k d w;
  Gc.full_major ();
  equal ~msg:"while open" bool true (Weak.check w 0);
  Rig.close d;
  Gc.full_major ();
  equal ~msg:"once closed" bool false (Weak.check w 0);
  raises_match (lost d) (fun () -> L.get k d)

(* A value whose init raced the stop is dropped too, and the get raises Lost:
   here the init closes its device. *)
let test_init_races_stop () =
  let d = Result.get_ok (Rig.memory_device (fresh "races")) in
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
  equal bool false (Weak.check w 0)

(* An init that closes its device and opens another in its slot leaves the
   get raising Lost: the other device's value is its own. *)
let test_init_reopens () =
  let d = Result.get_ok (Rig.memory_device (fresh "first")) in
  ignore
    (Rig.submit
       (Rig.Submission.make d [||])
       ~run:(Rig.Submission.Run.make ()) ~buffers:[||]);
  let key = ref None and other = ref None in
  let k =
    L.new_key (fun dev ->
        if Rig.equal dev d then (
          Rig.close d;
          let e = Result.get_ok (Rig.memory_device (fresh "second")) in
          equal ~msg:"e continues d's slot" int (Rig.submitted d)
            (Rig.submitted e);
          ignore (L.get (Option.get !key) e);
          other := Some e);
        made ())
  in
  key := Some k;
  raises_match (lost d) (fun () -> L.get k d);
  Option.iter Rig.close !other

(* A lost device's get raises, whether it had a value or not. *)
let test_lost () =
  let d = Result.get_ok (Rig.memory_device (fresh "lost")) in
  let k = L.new_key (fun _ -> made ()) and k' = L.new_key (fun _ -> made ()) in
  ignore (L.get k d);
  Rig.close d;
  raises_match ~msg:"with a value" (lost d) (fun () -> L.get k d);
  raises_match ~msg:"without" (lost d) (fun () -> L.get k' d)

let tests =
  [
    group ~timeout "values"
      [
        test "an init that gets its own key answers the nested value"
          test_nested;
        test "an init that raises stores nothing" test_init_raises;
        test "each key and device has its own value" test_keys;
        test "a get that finds its value allocates nothing" test_hit_words;
        test "a lost device's get raises Lost" test_lost;
        stateful "gets answer one value per device until it closes"
          model_commands;
        stateful ~domains:2 "racing first gets answer one value"
          racing_commands;
      ];
    group ~timeout "lifetime"
      [
        test "a closed device's value is collected" test_closed_collected;
        test "a value whose init raced the stop is collected"
          test_init_races_stop;
        test "an init that reopens the slot leaves the get lost"
          test_init_reopens;
      ];
  ]

let () = exit (run "rig.local" tests)
