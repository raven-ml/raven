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
      Nx.mul d (Nx.reshape [| 2 |] d));
  (* A reshape of a transpose copies: the two handles share no memory. *)
  let x = Nx.reshape [| 2; 2 |] (fresh [| 1.; 2.; 3.; 4. |]) in
  let c = Nx.reshape [| 4 |] (Nx.transpose (Nx.donate x)) in
  invalid ~sub:"Nx.donate: Nx.add reads one donation twice" (fun () ->
      Nx.add (Nx.reshape [| 4 |] (Nx.donate x)) c);
  equal ~msg:"and kills nothing" (array float_exact) [| 2.; 4.; 6.; 8. |]
    (elements (Nx.add x x))

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

let test_refused_kills_none () =
  let x1 = fresh [| 1.; 2. |] in
  let d1 = Nx.donate x1 in
  ignore (Nx.mul (Nx.donate x1) (fresh [| 3.; 4. |]));
  let x2 = fresh [| 5.; 6. |] in
  invalid ~sub:"Nx.add: operand 2 was donated to Nx.mul" (fun () ->
      Nx.add (Nx.donate x2) d1);
  equal ~msg:"the other donor lives" (array float_exact) [| 10.; 12. |]
    (elements (Nx.add x2 x2))

(* An operation that aligns its operands, as [add] broadcasts [1] to [2],
   checks them all before it consumes any. *)
let test_aligned_refusal () =
  let x2 = fresh [| 1.; 2. |] in
  let d2 = Nx.donate x2 in
  ignore (Nx.mul (Nx.donate x2) (fresh [| 3.; 4. |]));
  let x1 = fresh [| 5. |] in
  invalid ~sub:"Nx.add: operand 2 was donated to Nx.mul" (fun () ->
      Nx.add (Nx.donate x1) d2);
  equal ~msg:"the other donor lives" (array float_exact) [| 10. |]
    (elements (Nx.add x1 x1))

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

(* On two domains: handles given to consumers and movements. A handle's state
   is the set of what it may be: free, or taken by a named function. In
   sequence it is known; an operation of two handles that loses one to another
   domain may have taken the other. Every run is explained by some order, every
   result is computed from the elements, and a refusal names a function that
   took one of the operation's handles. *)
type handle = { ys : float array; mutable states : string option list }

let handle = abstract "h"
let make_handle ys = { ys; states = [ None ] }
let make_handle_sys ys = Nx.donate (fresh ys)
let taken = result (array float_exact) string

(* What a refusal names: the function the handle went to, or [twice]. *)
let taker msg =
  let has s =
    let n = String.length s and m = String.length msg in
    let rec go i = i + n <= m && (String.sub msg i n = s || go (i + 1)) in
    go 0
  in
  if has "reads one donation twice" then "twice"
  else
    match
      List.find_opt
        (fun f -> has ("donated to " ^ f))
        [ "Nx.add"; "Nx.mul"; "Nx.reshape" ]
    with
    | Some f -> f
    | None -> msg

let refused f =
  match f () with ys -> Ok ys | exception Invalid_argument m -> Error (taker m)

let free h = List.mem None h.states
let union a b = List.sort_uniq compare (a @ b)

(* Whether [h] may have gone to [e]. An [add2] claims its handles one at a time
   and keeps those it took when it loses another, so a handle it took can be
   seen dead before the order reaches it: [add2 h2 h3] takes [h2] on one
   domain, [add2 h3 h2] on the other loses [h2] to it, and the first loses [h3]
   to a later [add2 h1 h3] there. A free handle may then have gone to an
   [Nx.add] in flight, and is dead from then on. *)
let lost_to e h = List.mem (Some e) h.states || (e = "Nx.add" && free h)

(* A one-handle operation [by] whose result's elements are [f]'s. *)
let judge1 by f h = function
  | Ok (Ok ys) ->
      if not (free h) then fail (by ^ " took a taken handle");
      equal (array float_exact) (f h.ys) ys;
      h.states <- [ Some by ]
  | Ok (Error e) ->
      cover "a later operation raises naming the first" true;
      if not (lost_to e h) then fail (by ^ " raised naming " ^ e);
      h.states <- [ Some e ]
  | Error exn -> raise exn

(* Twos beside [h]: its shape answers whether it lives or not. *)
let twos h = fresh (Array.make (Nx.shape h).(0) 2.)
let add_ref = judge1 "Nx.add" (Array.map (fun v -> v +. 2.))
let mul_ref = judge1 "Nx.mul" (Array.map (fun v -> v *. 2.))
let add_sys h = refused (fun () -> elements (Nx.add h (twos h)))
let mul_sys h = refused (fun () -> elements (Nx.mul h (twos h)))

(* The movement's result is a handle no command reads. *)
let reshape_ref = judge1 "Nx.reshape" (fun _ -> [||])

let reshape_sys h =
  refused (fun () ->
      ignore (Nx.reshape [| 1; (Nx.shape h).(0) |] h);
      [||])

(* [add h1 h2] checks both handles, then takes them in an order the promise
   leaves open: a loser lost one handle, and may have taken the other if it was
   free. *)
let add2_ref h1 h2 out =
  if h1 == h2 then
    match out with
    | Ok r -> equal taken (Error "twice") r
    | Error exn -> raise exn
  else
    match out with
    | Ok (Ok ys) ->
        if not (free h1 && free h2) then fail "Nx.add took a taken handle";
        cover "two handles consumed" true;
        equal (array float_exact) (Array.map2 ( +. ) h1.ys h2.ys) ys;
        h1.states <- [ Some "Nx.add" ];
        h2.states <- [ Some "Nx.add" ]
    | Ok (Error e) ->
        let lost = lost_to e in
        let taken h =
          if free h then union h.states [ Some "Nx.add" ] else h.states
        in
        if not (lost h1 || lost h2) then fail ("Nx.add raised naming " ^ e);
        cover "one handle refused, the other free"
          ((lost h1 && free h2) || (lost h2 && free h1));
        let s1 = if lost h1 then [ Some e ] else [] in
        let s1 = if lost h2 then union s1 (taken h1) else s1 in
        let s2 = if lost h2 then [ Some e ] else [] in
        let s2 = if lost h1 then union s2 (taken h2) else s2 in
        h1.states <- s1;
        h2.states <- s2
    | Error exn -> raise exn

let add2_sys h1 h2 = refused (fun () -> elements (Nx.add h1 h2))

(* [add h1 h2] and [add h2 h1] on two domains, released together in each round
   over a fresh pair: at least one succeeds, and a loser names [Nx.add]. *)
let rounds = 2000

let test_crossed () =
  let pairs =
    Array.init rounds (fun _ ->
        (Nx.donate (fresh [| 1.; 2. |]), Nx.donate (fresh [| 10.; 20. |])))
  in
  let arrived = Atomic.make 0 in
  let meet r =
    Atomic.incr arrived;
    while Atomic.get arrived < 2 * (r + 1) do
      Domain.cpu_relax ()
    done
  in
  let race flip =
    Array.mapi
      (fun r (h1, h2) ->
        meet r;
        refused (fun () ->
            elements (if flip then Nx.add h2 h1 else Nx.add h1 h2)))
      pairs
  in
  let other = Domain.spawn (fun () -> race true) in
  let mine = race false in
  let theirs = Domain.join other in
  Array.iteri
    (fun r a ->
      match (a, theirs.(r)) with
      | Ok ys, Error e | Error e, Ok ys ->
          equal (array float_exact) [| 11.; 22. |] ys;
          equal string "Nx.add" e
      | Ok _, Ok _ -> fail (Printf.sprintf "round %d: both consumed" r)
      | Error e, Error e' ->
          fail (Printf.sprintf "round %d: both lost, to %s and %s" r e e'))
    mine

let pair =
  Gen.with_pp
    (fun ppf a ->
      Format.fprintf ppf "[%s]"
        (String.concat "; " (Array.to_list (Array.map string_of_float a))))
    (Gen.array ~size:(Gen.int_range 2 2) (Gen.of_list [ 0.; 1.; -2.; 3.5 ]))

let handles =
  [
    command "make" (pair @-> makes handle) make_handle make_handle_sys;
    command "add" (handle ^-> judges taken) add_ref add_sys;
    command "mul" (handle ^-> judges taken) mul_ref mul_sys;
    command "reshape" (handle ^-> judges taken) reshape_ref reshape_sys;
    command "add2" (handle ^-> handle ^-> judges taken) add2_ref add2_sys;
  ]

(* Consumers of two handles alone, so that two domains often take one pair in
   opposite orders. *)
let crossed =
  [
    command "make" (pair @-> makes handle) make_handle make_handle_sys;
    command "add2" (handle ^-> handle ^-> judges taken) add2_ref add2_sys;
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
             test "a refused consumer kills no donor" test_refused_kills_none;
             test "an aligning consumer checks before it consumes"
               test_aligned_refusal;
             test "a value of every set is not donated" test_left_alone;
             stateful ~domains:2 "reads and consumers on two domains" commands;
             stateful ~domains:2 "handles taken on two domains" handles;
             stateful ~domains:2 "two-handle consumers on two domains" crossed;
             test "of crossed consumers on two domains, one succeeds"
               test_crossed;
           ];
       ])
