(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Lost devices, on test memories whose driver reports a fault a test names
   ([Nx_test.Faulty]). Once a device is lost, every operation that reads or
   computes on a value it holds raises [Nx_device.Lost] naming it and the fault;
   values elsewhere, copies made before the loss included, compute as before;
   opening the device again gives a fresh, unequal device. *)

open Windtrap
open Nx_test

let elements = array float_exact

let iota shape =
  Nx.create Nx.float32 shape
    (Array.init (Ref.numel shape) (fun i -> float_of_int (i + 1)))

let memory_of v = Nx.Device.memory (List.hd (Nx.Placement.devices v))

(* [Lost (m, why)] for [d]'s memory [m]. *)
let lost_on d why = function
  | Nx_device.Lost (m, why') -> m == Nx.Device.memory d && why' = why
  | _ -> false

(* Operations *)

(* Every kind of operation on a value [x], with a host value [h] of its shape:
   each reads or computes on [x]. *)
type op = { name : string; run : (float, Nx.float32_elt) Nx.t -> unit }

let op name f = { name; run = (fun x -> ignore (Sys.opaque_identity (f x))) }

let ops h =
  [
    op "a unary operation" Nx.neg;
    op "a binary operation with a host value" (fun x -> Nx.add x h);
    op "a binary operation with itself" (fun x -> Nx.add x x);
    op "a reduction" Nx.sum;
    op "a scan" (fun x -> Nx.cumsum (Nx.flatten x));
    op "a product" (fun x ->
        let v = Nx.flatten x in
        let n = (Nx.shape v).(0) in
        Nx.matmul (Nx.reshape [| n; 1 |] v) (Nx.reshape [| 1; n |] v));
    op "a constant beside it" Nx.zeros_like;
    op "a copy" Nx.copy;
    op "a read" Nx.to_array;
    op "a read of a movement" (fun x -> Nx.to_array (Nx.transpose x));
    op "a print" (Format.asprintf "%a" Nx.pp);
    op "a placement on the host" (Nx.place Nx.Placement.host);
    op "a placement on another device"
      (Nx.place (Nx.Placement.on (Nx.Device.cpu 1)));
  ]

let pp_op ppf o = Format.pp_print_string ppf o.name
let shapes = Gen.array ~size:(Gen.int_range 0 3) (Gen.int_range 0 3)

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let shapes = Gen.with_pp pp_shape shapes
let kinds = Gen.of_list ~pp:pp_op (ops (iota [||]))

(* Each case draws an operation by its name, made over a host value of the drawn
   shape. *)
let on_lost =
  prop "every operation on a lost device's value raises Lost naming it"
    (Gen.pair shapes kinds) (fun (shape, kind) ->
      cover "no element" (Ref.numel shape = 0);
      cover "a scalar" (Array.length shape = 0);
      let d = Faulty.device 1 in
      let x = Nx.place (Nx.Placement.on d) (iota shape) in
      let o = List.find (fun o -> o.name = kind.name) (ops (iota shape)) in
      Faulty.lose d "memory lost";
      raises_match (lost_on d "memory lost") (fun () -> o.run x))

let elsewhere =
  prop "values the lost device does not hold compute as before"
    (Gen.pair shapes kinds) (fun (shape, kind) ->
      let d = Faulty.device 2 and other = Faulty.device 3 in
      let h = iota shape in
      let x = Nx.place (Nx.Placement.on d) h in
      let copies =
        [
          ("a host value", h);
          ("a copy on the host", Nx.place Nx.Placement.host x);
          ("a copy on another device", Nx.place (Nx.Placement.on other) x);
        ]
      in
      Faulty.lose d "page fault";
      let o = List.find (fun o -> o.name = kind.name) (ops h) in
      List.iter
        (fun (what, v) ->
          o.run v;
          equal ~msg:what elements (Nx.to_array h) (Nx.to_array v))
        copies)

(* Reopening *)

let reopening () =
  let d = Faulty.device 4 in
  equal ~msg:"an open of a live device" bool true
    (Nx.Device.equal d (Faulty.device 4));
  let x = Nx.place (Nx.Placement.on d) (iota [| 3 |]) in
  Faulty.lose d "hardware exception";
  let d' = Faulty.device 4 in
  equal ~msg:"a fresh device" bool false (Nx.Device.equal d d');
  equal ~msg:"of the same name" string (Nx.Device.name d) (Nx.Device.name d');
  let y = Nx.place (Nx.Placement.on d') (iota [| 3 |]) in
  equal ~msg:"computes" elements [| 2.; 4.; 6. |] (Nx.to_array (Nx.add y y));
  raises_match (lost_on d "hardware exception") (fun () -> Nx.to_array x);
  raises_match (lost_on d "hardware exception") (fun () ->
      Nx.place (Nx.Placement.on d) (iota [| 3 |]));
  equal ~msg:"a lost value keeps its shape" (array int) [| 3 |] (Nx.shape x);
  match Nx.to_array x with
  | _ -> fail "read a lost value"
  | exception e ->
      equal ~msg:"printed" string
        (Nx.Device.name d ^ " lost: hardware exception")
        (Printexc.to_string e)

(* The state machine *)

(* A model of devices and the values on them. A device is the opening of a slot
   that each run starts live; an opening of a lost slot is a new device. *)
module Model = struct
  exception Lost

  type device = { mutable lost : bool }
  type world = { current : device array }

  (* A value on a device, or on the host, and the device it was copied from. *)
  type value = {
    on : device option;
    elements : float array;
    source : device option;
  }

  let world () = { current = Array.init 2 (fun _ -> { lost = false }) }

  let device w slot =
    cover "a lost device is opened again" w.current.(slot).lost;
    if w.current.(slot).lost then w.current.(slot) <- { lost = false };
    w.current.(slot)

  let alive v = match v.on with Some d when d.lost -> raise Lost | _ -> ()

  let same d d' =
    cover "a lost device and a live one" (d.lost <> d'.lost);
    d == d'

  let place d elements =
    if d.lost then raise Lost;
    { on = Some d; elements; source = None }

  let host elements = { on = None; elements; source = None }

  (* [add v w] for [v] and [w] on one device, or one on the host. *)
  let add v w =
    alive v;
    alive w;
    let on = match v.on with Some _ -> v.on | None -> w.on in
    { on; elements = Array.map2 ( +. ) v.elements w.elements; source = None }

  let read v =
    let lost = match v.on with Some d -> d.lost | None -> false in
    cover "a lost device's value is read" lost;
    cover "a lost device's value of no elements is read"
      (lost && Array.length v.elements = 0);
    cover "a copy is read once its source is lost"
      ((not lost) && match v.source with Some d -> d.lost | None -> false);
    if lost then Error () else Ok v.elements

  let move v d =
    if d.lost then raise Lost;
    alive v;
    { v with on = Some d; source = v.on }

  let to_host v =
    alive v;
    { v with on = None; source = v.on }

  let lose d = d.lost <- true
end

type value = { v : (float, Nx.float32_elt) Nx.t }

(* A world's slots are memories of [Faulty] of its own, so that each run starts
   with live devices that no other run used. *)
let worlds = Atomic.make 10
let world = abstract "w"
let device = abstract "d"
let value = abstract "v"
let slots = Gen.of_list ~pp:Format.pp_print_int [ 0; 1 ]

let contents =
  Gen.with_pp
    (fun ppf a ->
      Format.fprintf ppf "[|%s|]"
        (String.concat "; " (Array.to_list (Array.map string_of_float a))))
    (Gen.array
       ~size:(Gen.one_of [ Gen.constant 0; Gen.constant 3 ])
       (Gen.map float_of_int (Gen.int_range (-4) 4)))

let vector a = Nx.create Nx.float32 [| Array.length a |] a

(* Same length, and on one device or the host. *)
let compatible (v : Model.value) (w : Model.value) =
  Array.length v.elements = Array.length w.elements
  && match (v.on, w.on) with Some d, Some d' -> d == d' | _ -> true

(* A read's outcome: its elements, or that it raised [Lost] naming the value's
   device and its fault. *)
let read_outcome = result elements unit

let commands =
  [
    command "world"
      (Gen.unit @-> makes world)
      Model.world
      (fun () -> Atomic.fetch_and_add worlds 2);
    command "device"
      (world ^-> slots @-> makes device)
      Model.device
      (fun base slot -> Faulty.device (base + slot));
    command "same"
      (device ^-> device ^-> returns bool)
      Model.same Nx.Device.equal;
    command "place"
      (device ^-> contents @-> makes value)
      Model.place
      (fun d a -> { v = Nx.place (Nx.Placement.on d) (vector a) });
    command "host"
      (contents @-> makes value)
      Model.host
      (fun a -> { v = vector a });
    command "add" ~pre:compatible
      (value ^-> value ^-> makes value)
      Model.add
      (fun x y -> { v = Nx.add x.v y.v });
    command "read"
      (value ^-> returns read_outcome)
      Model.read
      (fun x ->
        match Nx.to_array x.v with
        | a -> Ok a
        | exception (Nx_device.Lost (m, why) as e) ->
            if m == memory_of (Nx.placement x.v) && why = "fault" then Error ()
            else raise e);
    command "move"
      (value ^-> device ^-> makes value)
      Model.move
      (fun x d -> { v = Nx.place (Nx.Placement.on d) x.v });
    command "to host"
      (value ^-> makes value)
      Model.to_host
      (fun x -> { v = Nx.place Nx.Placement.host x.v });
    command "lose"
      (device ^-> returns unit)
      Model.lose
      (fun d -> Faulty.lose d "fault");
  ]

let machine =
  stateful ~count:300 ~steps:25
    "devices lose their values at a fault and none other, and reopen fresh"
    commands

let () =
  exit
    (run "nx lost devices"
       [
         group "operations" [ on_lost; elsewhere ];
         group "devices"
           [
             test
               "reopening a lost device gives a fresh one, and its values stay \
                lost"
               reopening;
             machine;
           ];
       ])
