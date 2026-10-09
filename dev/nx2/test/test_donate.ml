(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Donation through Nx: when a donor and its handles die, the one-read rule,
   handles passed on by movements, sharers keeping their elements, and values
   that donation leaves alone. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype

let f32 = Nx.float32
let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let elements x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))

(* A value in memory of its own: computed, so no Repr crossing shares it. *)
let fresh xs =
  let a =
    Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 [| Array.length xs |] xs)
  in
  Nx.add a (Nx.zeros_like a)

let test_dies_at_its_consumer () =
  let x = fresh [| 1.; 2. |] in
  let d = Nx.donate x in
  equal ~msg:"usable before" (array float_exact) [| 2.; 4. |]
    (elements (Nx.add x x));
  let y = Nx.mul d (fresh [| 3.; 4. |]) in
  equal ~msg:"the result" (array float_exact) [| 3.; 8. |] (elements y);
  invalid ~sub:"Nx.add: operand 1 was donated to Nx.mul" (fun () -> Nx.add x x);
  invalid ~sub:"Nx.copy: operand 1 was donated to Nx.mul" (fun () -> Nx.copy d);
  equal ~msg:"its shape answers" (array int) [| 2 |] (Nx.shape x)

let test_one_read () =
  let x = fresh [| 1.; 2. |] in
  let d = Nx.donate x in
  invalid ~sub:"Nx.donate: Nx.mul reads one donation twice" (fun () ->
      Nx.mul d d);
  let x = fresh [| 1.; 2. |] in
  invalid ~sub:"Nx.donate: Nx.add reads one donation twice" (fun () ->
      Nx.add (Nx.donate x) (Nx.donate x));
  let x = fresh [| 1.; 2. |] in
  let d = Nx.donate x in
  invalid ~sub:"Nx.donate: Nx.mul reads one donation twice" (fun () ->
      Nx.mul d (Nx.reshape [| 2 |] d))

let test_passed_on () =
  let x = fresh [| 1.; 2.; 3.; 4. |] in
  let d = Nx.reshape [| 2; 2 |] (Nx.donate x) in
  equal ~msg:"the donor lives until the final consumer" (array float_exact)
    [| 2.; 4.; 6.; 8. |]
    (elements (Nx.add x x));
  let y = Nx.copy d in
  equal ~msg:"the consumer's result" (array float_exact) [| 1.; 2.; 3.; 4. |]
    (elements y);
  invalid ~sub:"Nx.add: operand 1 was donated to Nx.copy" (fun () -> Nx.add x x);
  invalid ~sub:"Nx.copy: operand 1 was donated to Nx.copy" (fun () -> Nx.copy d)

let test_sharer_keeps () =
  let x = fresh [| 1.; 2. |] in
  let v = Nx.reshape [| 1; 2 |] x in
  let y = Nx.add (Nx.donate x) (fresh [| 10.; 20. |]) in
  equal ~msg:"the result" (array float_exact) [| 11.; 22. |] (elements y);
  equal ~msg:"a view made earlier keeps its elements" (array float_exact)
    [| 1.; 2. |] (elements v)

let test_left_alone () =
  let c = Nx.zeros f32 [| 2 |] in
  let d = Nx.donate c in
  equal ~msg:"a value of every set, read twice" (array float_exact) [| 0.; 0. |]
    (elements (Nx.add d d));
  equal ~msg:"and still usable" (array float_exact) [| 0.; 0. |]
    (elements (Nx.add c c))

(* On two domains: reading a value and consuming it in an addition. Each run is
   explained by some order: a read before the consumer gives the elements, after
   it raises; one consumer of a value wins and any other raises. *)
type model = { xs : float array; mutable dead : bool }

let value = abstract "x"
let make_ref xs = { xs; dead = false }
let make_sys xs = fresh xs

let read_ref m =
  if m.dead then Error () else Ok (Array.map (fun v -> v +. v) m.xs)

let read_sys x =
  match elements (Nx.add x x) with
  | ys -> Ok ys
  | exception Invalid_argument _ -> Error ()

let consume_ref m =
  if m.dead then Error ()
  else begin
    m.dead <- true;
    Ok (Array.map (fun v -> v +. 1.) m.xs)
  end

let consume_sys x =
  match
    elements
      (Nx.add (Nx.donate x) (Nx.add (Nx.zeros_like x) (Nx.scalar f32 1.)))
  with
  | ys -> Ok ys
  | exception Invalid_argument _ -> Error ()

let outcome = result (array float_exact) unit

let floats2 =
  Gen.with_pp
    (fun ppf a ->
      Format.fprintf ppf "[%s]"
        (String.concat "; " (Array.to_list (Array.map string_of_float a))))
    (Gen.array ~size:(Gen.int_range 1 3) (Gen.of_list [ 0.; 1.; -2.; 3.5 ]))

let commands =
  [
    command "make" (floats2 @-> makes value) make_ref make_sys;
    command "read" (value ^-> returns outcome) read_ref read_sys;
    command "consume" (value ^-> returns outcome) consume_ref consume_sys;
  ]

let () =
  exit
    (run "nx donate"
       [
         group "donation"
           [
             test "a donor and its handle die at the consumer, naming it"
               test_dies_at_its_consumer;
             test "one donation reaches an operation once" test_one_read;
             test "a one-to-one movement passes the handle on" test_passed_on;
             test "a value sharing the memory keeps its elements"
               test_sharer_keeps;
             test "a value of every set is not donated" test_left_alone;
             stateful ~domains:2 "reads and consumers on two domains" commands;
           ];
       ])
