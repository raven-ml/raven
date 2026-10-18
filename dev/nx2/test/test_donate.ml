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

(* As [fresh], of shape [s]: a reshape would make a view, which shares. *)
let fresh_at s xs =
  let a = Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 s xs) in
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

(* A function that computes through several operations kills its donors only
   when it returns: refused, from its own checks or from an operation inside,
   it leaves each donor and its handle usable. *)
let test_refused_composite () =
  let x = fresh_at [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let d = Nx.donate x in
  invalid ~sub:"Nx.softmax" (fun () -> Nx.softmax ~axes:[ 5 ] d);
  invalid ~sub:"Nx.roll" (fun () -> Nx.roll ~axis:4 1 d);
  let a = fresh_at [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let da = Nx.donate a in
  invalid ~sub:"Nx.lstsq" (fun () -> Nx.lstsq da (fresh [| 1.; 2.; 3. |]));
  equal ~msg:"the donor lives" (array float_exact)
    [| 2.; 4.; 6.; 8.; 10.; 12. |]
    (elements (Nx.add x x));
  equal ~msg:"its handle reaches a later consumer" (array float_exact)
    [| 1.; 2.; 3.; 4.; 5.; 6. |]
    (elements (Nx.copy d));
  invalid ~sub:"Nx.add: operand 1 was donated to Nx.copy" (fun () ->
      Nx.add x x);
  equal ~msg:"the other donor's handle computes" (array float_exact)
    [| 3.; 7. |]
    (elements (Nx.sum ~axes:[ 1 ] da));
  (* [map_segments] reads both handles in operations before it calls [f]; an
     operation [f] makes refuses, then its own check of [f]'s result. *)
  let ids = Nx.create Nx.int64 [| 2 |] [| 0L; 1L |] in
  let x = fresh_at [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let di = Nx.donate ids and dx = Nx.donate x in
  let segments f = Nx.map_segments ~segments:2 di f dx in
  invalid ~sub:"Nx.reshape" (fun () ->
      segments (fun _ rows -> Nx.reshape [| 3 |] rows));
  invalid ~sub:"Nx.map_segments: f gave" (fun () ->
      segments (fun _ rows -> Nx.reshape [| -1 |] (Nx.mul_s rows 2.)));
  equal ~msg:"the ids live" (array int64) [| 0L; 1L |]
    (elements (Nx.copy ids));
  equal ~msg:"the rows' donor lives" (array float_exact) [| 2.; 4.; 6.; 8. |]
    (elements (Nx.add x x));
  equal ~msg:"their handles reach a later consumer" (array float_exact)
    [| 2.; 4.; 6.; 8. |]
    (elements (segments (fun _ rows -> Nx.mul_s rows 2.)))

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

(* An operation an interpretation receives consumes its handles as one
   computed now does: an observer whose rule computes each operation, and a
   tracer whose rule computes none, both kill the donor, naming the
   consumer. *)
type ('v, 's, 'd) Nx.Prim.payload += Tag : ('v, 's, 'd) Nx.Prim.payload

let test_dies_when_received () =
  let x = fresh [| 1.; 2. |] in
  let y =
    Nx.Prim.interpret ~name:"test.observer" Extent
      (fun _ ~by op -> Nx.Prim.eval ~by op)
      (fun _ -> Nx.mul (Nx.donate x) (fresh [| 3.; 4. |]))
  in
  equal ~msg:"the observer's result" (array float_exact) [| 3.; 8. |]
    (elements y);
  invalid ~sub:"Nx.add: operand 1 was donated to Nx.mul" (fun () -> Nx.add x x);
  let x = fresh [| 1.; 2. |] and t = fresh [| 3.; 4. |] in
  Nx.Prim.interpret ~name:"test.tracer" Values
    (fun i ~by op ->
      Nx.Prim.results ~by (fun _ form -> Nx.Prim.traced i form Tag) op)
    (fun i ->
      let t = Nx.Prim.traced i (Nx.Prim.form t) Tag in
      ignore (Nx.mul (Nx.donate x) t));
  invalid ~sub:"Nx.add: operand 1 was donated to Nx.mul" (fun () -> Nx.add x x);
  (* A consumer that broadcasts its handle gives the rule a movement, which
     is then the final consumer: under the observer, and under an Extent
     rule that keeps its operands and computes nothing. *)
  let wide = Nx.reshape [| 3; 2 |] (fresh (Array.make 6 10.)) in
  let x = fresh [| 1.; 2. |] in
  let y =
    Nx.Prim.interpret ~name:"test.observer" Extent
      (fun _ ~by op -> Nx.Prim.eval ~by op)
      (fun _ -> Nx.add (Nx.donate x) wide)
  in
  equal ~msg:"the broadcast sum" (array float_exact)
    [| 11.; 12.; 11.; 12.; 11.; 12. |]
    (elements (Nx.reshape [| 6 |] y));
  invalid ~sub:"Nx.add: operand 1 was donated to Nx.add" (fun () -> Nx.add x x);
  let x = fresh [| 1.; 2. |] in
  Nx.Prim.interpret ~name:"test.keeper" Extent
    (fun i ~by op ->
      Nx.Prim.results ~by (fun _ form -> Nx.Prim.traced i form Tag) op)
    (fun _ -> ignore (Nx.add (Nx.donate x) wide));
  invalid ~sub:"Nx.add: operand 1 was donated to Nx.add" (fun () -> Nx.add x x)

(* Reuse: a consumer that writes its result into donated memory takes no
   memory of the result's size, counted on a memory device of these tests'
   own. *)
let device = Nx_support.memory 3

module Mem = (val Nx.devices [ device ])

(* [xs] of shape [s] on [device] in memory of its own: computed there, since a
   placement onto it borrows the host's memory, which shares it. *)
let on_mem s xs =
  let a =
    Nx.place Mem.on (Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 s xs))
  in
  Nx.add a (Nx.zeros_like a)

let positions s ps = Nx.place Mem.on (Nx.create Nx.int64 s ps)

(* The bytes [device] holds after four rounds of collection, each of which
   returns to the device what it collected. *)
let settled () =
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create device 0);
    Rig.free_cache device
  done;
  Rig.allocated device

(* [f (make ())] and the bytes [device] holds more after it, its cache
   included, so that a donor freed after a copy still counts. One uncounted
   run comes first, and the count before is read settled, so that no memory
   collected late serves the counted run. *)
let grown make f =
  ignore (f (make ()));
  let inputs = make () in
  let before = settled () in
  let y = f inputs in
  (y, Rig.allocated device - before)

(* A cache of [rows] rows of [width] float32, row [r] holding [r], and its
   bytes. *)
let width = 4096
let rows = 8
let size = 4 * rows * width
let original = Array.init (rows * width) (fun i -> Float.of_int (i / width))

(* The cache's elements with row [r] holding [v]. *)
let with_row r v =
  Array.mapi (fun i x -> if i / width = r then v else x) original

let cache () = on_mem [| rows; width |] original
let nines s = on_mem s (Array.make width 9.)

let in_place ~msg ~bytes grew =
  less ~msg:(msg ^ ": no memory of the result's size") int ~than:bytes grew

let copied ~msg ~bytes grew =
  at_least ~msg:(msg ^ ": a fresh result") int ~than:bytes grew

let test_set_in_place () =
  let check ~msg idx s =
    let y, grew =
      grown
        (fun () -> (cache (), nines s))
        (fun (c, v) -> Nx.set idx v (Nx.donate c))
    in
    in_place ~msg ~bytes:size grew;
    equal ~msg (array float_exact) (with_row 5 9.) (elements y)
  in
  let pos = Nx.place Mem.on (Nx.scalar Nx.int64 5L) in
  check ~msg:"a window held in data" Nx.[ D (pos, 1); A ] [| 1; width |];
  check ~msg:"a written position" Nx.[ I 5 ] [| width |];
  check ~msg:"a list of positions" Nx.[ L [ 5 ] ] [| 1; width |];
  check ~msg:"positions on two axes" Nx.[ L [ 5 ]; R (0, width) ] [| 1; width |];
  let c = cache () in
  ignore (Nx.set Nx.[ I 5 ] (nines [| width |]) (Nx.donate c));
  invalid ~sub:"donated to Nx.set" (fun () -> Nx.copy c)

let test_scatter_in_place () =
  let p = positions [| 1; width |] (Array.make width 5L) in
  let y, grew =
    grown
      (fun () -> (cache (), nines [| 1; width |]))
      (fun (c, u) -> Nx.scatter ~axis:0 p u (Nx.donate c))
  in
  in_place ~msg:"scatter" ~bytes:size grew;
  equal (array float_exact) (with_row 5 9.) (elements y)

let test_init_in_place () =
  let a = on_mem [| 2; rows |] (Array.make (2 * rows) 1.) in
  let b = on_mem [| 2; width |] (Array.make (2 * width) 1.) in
  let plus_two =
    Array.init (rows * width) (fun i -> Float.of_int (i / width) +. 2.)
  in
  let p = Nx.Pattern.v "n i, n o -> i o | n" in
  let y, grew =
    grown cache (fun g -> Nx.contract f32 ~init:(Nx.donate g) p a b)
  in
  in_place ~msg:"contract" ~bytes:size grew;
  equal (array float_exact) plus_two (elements y);
  (* [b]'s names before [a]'s: the kernel's result is the transpose. *)
  let p = Nx.Pattern.v "n o, n i -> i o | n" in
  let y, grew =
    grown cache (fun g -> Nx.contract f32 ~init:(Nx.donate g) p b a)
  in
  copied ~msg:"contract, transposed" ~bytes:size grew;
  equal (array float_exact) plus_two (elements y)

(* A value that is not donated, or that shares the donated memory, keeps its
   elements, and the result is the same. *)
let test_kept () =
  let v = nines [| 1; width |] in
  let c = cache () in
  let y, grew = grown (fun () -> c) (fun c -> Nx.set Nx.[ L [ 5 ] ] v c) in
  copied ~msg:"not donated" ~bytes:size grew;
  equal ~msg:"the result" (array float_exact) (with_row 5 9.) (elements y);
  equal ~msg:"the operand" (array float_exact) original (elements c);
  let shared () =
    let c = cache () in
    (c, Nx.reshape [| rows * width |] c)
  in
  let (y, flat), grew =
    grown shared (fun (c, flat) ->
        (Nx.set Nx.[ L [ 5 ] ] v (Nx.donate c), flat))
  in
  copied ~msg:"shared" ~bytes:size grew;
  equal ~msg:"the result" (array float_exact) (with_row 5 9.) (elements y);
  equal ~msg:"the view" (array float_exact) original (elements flat)

(* An operation may read the donor beside its handle. Where it reads it at
   other indices than the result's own, as a scatter's updates or a
   contraction's operand, it computes into fresh memory; where at the
   result's own index, as an elementwise operand, it may still write there. *)
let test_donor_beside () =
  let p =
    positions [| rows; width |]
      (Array.init (rows * width) (fun i ->
           Int64.of_int (rows - 1 - (i / width))))
  in
  let y, grew = grown cache (fun c -> Nx.scatter ~axis:0 p c (Nx.donate c)) in
  copied ~msg:"the scatter's updates" ~bytes:size grew;
  equal ~msg:"rows reversed" (array float_exact)
    (Array.init (rows * width) (fun i -> Float.of_int (rows - 1 - (i / width))))
    (elements y);
  let c = cache () in
  ignore (Nx.scatter ~axis:0 p c (Nx.donate c));
  invalid ~sub:"donated to Nx.scatter" (fun () -> Nx.copy c);
  let n = 128 in
  let ones = on_mem [| n; n |] (Array.make (n * n) 1.) in
  let sums i = Float.of_int ((n * n * i) + (n * (n - 1) / 2)) in
  let p = Nx.Pattern.v "i k, k j -> i j | k" in
  let y, grew =
    grown
      (fun () -> on_mem [| n; n |] (Array.init (n * n) Float.of_int))
      (fun g -> Nx.contract f32 ~init:(Nx.donate g) p g ones)
  in
  copied ~msg:"the contraction's operand" ~bytes:(4 * n * n) grew;
  equal ~msg:"init plus the product" (array float_exact)
    (Array.init (n * n) (fun i -> Float.of_int i +. sums (i / n)))
    (elements y);
  let y, grew = grown cache (fun c -> Nx.add (Nx.donate c) c) in
  in_place ~msg:"an elementwise operand" ~bytes:size grew;
  let doubled = Array.map (fun x -> 2. *. x) original in
  equal ~msg:"doubled" (array float_exact) doubled (elements y);
  let c = cache () in
  let flat = Nx.reshape [| rows * width |] c in
  let y = Nx.add (Nx.donate c) c in
  equal ~msg:"doubled, shared" (array float_exact) doubled (elements y);
  equal ~msg:"the view" (array float_exact) original (elements flat)

(* A consumer that takes a composite's donor while the composite runs, here a
   rule at [softmax]'s first operation, computes into fresh memory, so that
   [softmax]'s later operations read the donor's elements; [softmax], which
   lost its donor, raises naming that consumer. *)
let test_inside_composite () =
  let run () =
    let c = cache () and taken = ref None in
    let rule _ ~by op =
      if Option.is_none !taken then begin
        taken := Some (0, [||]);
        let before = settled () in
        let y = Nx.add (Nx.donate c) (nines [| width |]) in
        taken := Some (Rig.allocated device - before, elements y)
      end;
      Nx.Prim.eval ~by op
    in
    invalid ~sub:"Nx.softmax: operand 1 was donated to Nx.add" (fun () ->
        Nx.Prim.interpret ~name:"test.taker" Extent rule (fun _ ->
            Nx.softmax ~axes:[ 1 ] (Nx.donate c)));
    Option.get !taken
  in
  ignore (run ());
  let grew, y = run () in
  copied ~msg:"the consumer" ~bytes:size grew;
  equal ~msg:"the consumer's result" (array float_exact)
    (Array.map (fun x -> x +. 9.) original)
    y

(* A donated call gives the bits of the same call undonated: a scatter of
   every combine with repeated positions, positions outside the axis, NaN,
   -0. and infinities, and a bfloat16 init summed in float32 over a drawn
   number of terms. *)
type scatter_case = {
  combine : Nx.combine;
  target : float array;
  targets : int64 array;
  updates : float array;
}

let specials = [ 0.; -0.; 1.5; -2.; Float.nan; Float.infinity; 3e38 ]

let scatter_case =
  Gen.with_pp
    (fun ppf c ->
      let fs a =
        String.concat "; " (Array.to_list (Array.map Float.to_string a))
      in
      Format.fprintf ppf "%s target [%s] at [%s] updates [%s]"
        (match c.combine with
        | Nx.Set -> "Set"
        | Add -> "Add"
        | Max -> "Max"
        | Min -> "Min")
        (fs c.target)
        (String.concat "; "
           (Array.to_list (Array.map Int64.to_string c.targets)))
        (fs c.updates))
    (let open Gen in
     let* combine = of_list Nx.[ Set; Add; Max; Min ] in
     let* n = int_range 1 6 in
     let* m = int_range 0 12 in
     let* target = array ~size:(constant n) (of_list specials) in
     let* targets =
       array ~size:(constant m)
         (map Int64.of_int (int_range (-1) n))
     in
     let+ updates = array ~size:(constant m) (of_list specials) in
     { combine; target; targets; updates })

let bits32 x = elements (Nx.bitcast D.Int32 x)

let law_scatter_bits =
  prop "a donated scatter gives the undonated one's bits" scatter_case
    (fun c ->
      let m = Array.length c.updates and n = Array.length c.target in
      let repeated =
        Array.length (Array.of_list (List.sort_uniq compare (Array.to_list c.targets)))
        < m
      in
      cover "a repeated position" repeated;
      cover "a position outside the axis"
        (Array.exists (fun t -> t < 0L || t >= Int64.of_int n) c.targets);
      cover "a NaN target" (Array.exists Float.is_nan c.target);
      let p = Nx.create Nx.int64 [| m |] c.targets in
      let u = fresh c.updates in
      let combine = c.combine in
      let kept = Nx.scatter ~combine ~axis:0 p u (fresh c.target) in
      let given = Nx.scatter ~combine ~axis:0 p u (Nx.donate (fresh c.target)) in
      equal (array int32) (bits32 kept) (bits32 given))

let terms = Gen.int_range 1 2100

let law_init_bits =
  prop "a donated bfloat16 init gives the undonated one's bits" terms (fun k ->
      cover "more terms than a block of 1024" (k > 1024);
      let value i = Float.of_int ((i * 37 mod 11) - 5) /. 3. in
      let a = fresh_at [| k; 3 |] (Array.init (3 * k) value) in
      let b = fresh_at [| k; 5 |] (Array.init (5 * k) (fun i -> value (i + 7))) in
      let init () =
        Nx.cast D.Bfloat16 (fresh_at [| 3; 5 |] (Array.init 15 value))
      in
      let a = Nx.cast D.Bfloat16 a and b = Nx.cast D.Bfloat16 b in
      let p = Nx.Pattern.v "k i, k o -> i o | k" in
      let bits y = elements (Nx.bitcast D.Uint16 y) in
      let kept = Nx.contract Nx.bfloat16 ~init:(init ()) p a b in
      let given = Nx.contract Nx.bfloat16 ~init:(Nx.donate (init ())) p a b in
      equal (array int) (bits kept) (bits given))

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

(* Naming: a function consumes a donated operand for the user, so the donor
   and the handle the user holds die naming the function the user called,
   computed now or received by an interpretation, whatever functions that one
   computes through. *)
let test_dies_at_the_call () =
  let square () = fresh_at [| 2; 2 |] [| 2.; 0.; 0.; 2. |] in
  let pair () = fresh [| 1.; 2. |] in
  let lo = Nx.create Nx.int64 [| 1 |] [| 0L |] in
  let hi = Nx.create Nx.int64 [| 1 |] [| 2L |] in
  let observed f =
    Nx.Prim.interpret ~name:"test.observer" Extent
      (fun _ ~by op -> Nx.Prim.eval ~by op)
      (fun _ -> f ())
  in
  let dies name make f =
    List.iter
      (fun (how, run) ->
        let x = make () in
        let d = Nx.donate x in
        run (fun () -> ignore (f d));
        let sub = "Nx.copy: operand 1 was donated to " ^ name in
        let msg = Printf.sprintf "%s %s's" how name in
        raises_match ~msg:(msg ^ " donor") (Exn.invalid_arg ~substring:sub)
          (fun () -> Nx.copy x);
        raises_match ~msg:(msg ^ " handle") (Exn.invalid_arg ~substring:sub)
          (fun () -> Nx.copy d))
      [ ("computed", fun f -> f ()); ("received", observed) ]
  in
  dies "Nx.trace" square (fun d -> Nx.trace d);
  dies "Nx.inv" square Nx.inv;
  dies "Nx.cholesky" square (fun d -> Nx.cholesky ~triangle:Upper d);
  dies "Nx.mean" pair (fun d -> Nx.mean d);
  dies "Nx.rfft" square (fun d -> Nx.rfft Nx.complex64 d);
  dies "Nx.complex" pair (fun d -> Nx.complex Nx.complex64 ~re:d ~im:(pair ()));
  dies "Nx.reduce_ranges"
    (fun () -> Nx.cast Nx.float16 (pair ()))
    (Nx.reduce_ranges Nx.Add ~lo ~hi)

(* Every public function that takes a value, called by one table of cases. A
   case makes fresh operands, passes its operand [i] as [d i x], and prints the
   results. Two promises run over the table: on a dead operand the function
   raises naming itself, the one the user called, whatever it computes through;
   a donated operand gives the undonated result. *)

(* What a function does with a dead operand: raise naming itself, or print
   [death]. *)
type dead = Raises | Prints of string

type case = {
  fn : string;  (* The function, as the user writes it. *)
  by : string;  (* The name its refusals give. *)
  dt : string;
  operands : int;
  dead : dead;
  run :
    ('v 's. int -> ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t) -> string;
}

let product s = Array.fold_left ( * ) 1 s

(* A host value of shape [s] in memory of its own, its elements spread over both
   signs. *)
let data (type v s) (dt : (v, s) Nx.dtype) s : (v, s, Nx.host) Nx.t =
  let n = product s in
  let x =
    Nx.cast dt
      (Nx.create Nx.float64 s
         (Array.init n (fun i -> Float.of_int ((i * 7 mod 11) - 4) /. 2.)))
  in
  Nx.add x (Nx.zeros_like x)

(* As [data], of [int64] positions [xs] at shape [s], [[|n|]] by default. *)
let ints ?s xs =
  let s = Option.value s ~default:[| Array.length xs |] in
  let x = Nx.create Nx.int64 s (Array.map Int64.of_int xs) in
  Nx.add x (Nx.zeros_like x)

let int32s xs =
  let x =
    Nx.create Nx.int32 [| Array.length xs |] (Array.map Int32.of_int xs)
  in
  Nx.add x (Nx.zeros_like x)

let complex_data s = Nx.complex Nx.complex64 ~re:(data f32 s) ~im:(data f32 s)
let show x = Nx.to_string x
let shows l = String.concat "; " l
let showl l = shows (List.map show l)

(* A case of [Nx.fn], whose refusals name [Nx.by], [Nx.fn] by default. *)
let c ?by ?(dead = Raises) fn operands run dt =
  let by = "Nx." ^ Option.value by ~default:fn in
  { fn = "Nx." ^ fn; by; dt; operands; dead; run }

(* The cases of a group, at the dtype named [dt]. *)
let table dt cases = List.map (fun case -> case dt) cases

(* The functions of every dtype. *)
let any_dtype (type v s) (dt : (v, s) Nx.dtype) =
  let x s = data dt s and one = D.one dt in
  let nonzero s =
    let a = x s in
    Nx.add_s (Nx.mul a a) one
  in
  let ij_jk = Nx.Pattern.v "i j, j k -> i k | j" in
  let dead = Prints (D.name dt ^ " [2; 3], donated to Nx.copy") in
  table (D.name dt)
    [
      c "copy" 1 (fun d -> show (Nx.copy (d 0 (x [| 2; 3 |]))));
      c "map_item" 1 (fun d -> show (Nx.map_item Fun.id (d 0 (x [| 2; 3 |]))));
      c "to_array" 1 (fun d ->
          show (Nx.create dt [| 6 |] (Nx.to_array (d 0 (x [| 2; 3 |])))));
      c "item" 1 (fun d ->
          show (Nx.scalar dt (Nx.item [ 1; 2 ] (d 0 (x [| 2; 3 |])))));
      c "sync" 1 (fun d ->
          Nx.sync (d 0 (x [| 3 |]));
          "synced");
      c "to_string" ~dead 1 (fun d -> Nx.to_string (d 0 (x [| 2; 3 |])));
      c "pp" ~dead 1 (fun d -> Format.asprintf "%a" Nx.pp (d 0 (x [| 2; 3 |])));
      c "place" 1 (fun d -> show (Nx.place Nx.Host.on (d 0 (x [| 2; 3 |]))));
      c "Repr.array" 1 (fun d ->
          show
            (Nx.Repr.of_array Nx.Host.v
               (Option.get (Nx.Repr.array (d 0 (x [| 2; 3 |]))))));
      c "Repr.shards" 1 (fun d ->
          string_of_int
            (Iarray.length (Option.get (Nx.Repr.shards (d 0 (x [| 2; 3 |]))))));
      c "reshape" 1 (fun d -> show (Nx.reshape [| 3; 2 |] (d 0 (x [| 2; 3 |]))));
      c "broadcast_to" 1 (fun d ->
          show (Nx.broadcast_to [| 2; 2; 3 |] (d 0 (x [| 2; 3 |]))));
      c "broadcast_arrays" 2 (fun d ->
          showl (Nx.broadcast_arrays [ d 0 (x [| 2; 1 |]); d 1 (x [| 3 |]) ]));
      c "squeeze" 1 (fun d -> show (Nx.squeeze (d 0 (x [| 2; 1; 3 |]))));
      c "unsqueeze" 1 (fun d ->
          show (Nx.unsqueeze ~axes:[ 0 ] (d 0 (x [| 3 |]))));
      c "flatten" 1 (fun d -> show (Nx.flatten (d 0 (x [| 2; 3 |]))));
      c "transpose" 1 (fun d -> show (Nx.transpose (d 0 (x [| 2; 3 |]))));
      c "moveaxis" 1 (fun d -> show (Nx.moveaxis 0 1 (d 0 (x [| 2; 3 |]))));
      c "swapaxes" 1 (fun d -> show (Nx.swapaxes 0 1 (d 0 (x [| 2; 3 |]))));
      c "flip" 1 (fun d -> show (Nx.flip (d 0 (x [| 2; 3 |]))));
      c "sliding_window" 1 (fun d ->
          show (Nx.sliding_window ~window:2 (d 0 (x [| 5 |]))));
      c "split" 1 (fun d -> showl (Nx.split ~axis:0 2 (d 0 (x [| 4; 2 |]))));
      c "concatenate" 2 (fun d ->
          show
            (Nx.concatenate ~axis:0 [ d 0 (x [| 2; 3 |]); d 1 (x [| 1; 3 |]) ]));
      c "stack" 2 (fun d ->
          show (Nx.stack [ d 0 (x [| 3 |]); d 1 (x [| 3 |]) ]));
      c "pad" 1 (fun d ->
          show (Nx.pad [| (1, 0); (0, 1) |] one (d 0 (x [| 2; 3 |]))));
      c "roll" 1 (fun d -> show (Nx.roll 1 (d 0 (x [| 5 |]))));
      c "roll" 1 (fun d -> show (Nx.roll ~axis:1 2 (d 0 (x [| 2; 3 |]))));
      c "tile" 1 (fun d -> show (Nx.tile [| 2; 1 |] (d 0 (x [| 2; 3 |]))));
      c "repeat" 1 (fun d -> show (Nx.repeat ~axis:0 2 (d 0 (x [| 2; 3 |]))));
      c "rearrange" 1 (fun d ->
          show (Nx.rearrange (Nx.Pattern.v "a b -> b a") (d 0 (x [| 2; 3 |]))));
      c "slice" 1 (fun d ->
          show
            (Nx.slice [ Nx.R (0, 1); Nx.Rs (2, -4, -1) ] (d 0 (x [| 2; 3 |]))));
      c "slice" 1 (fun d ->
          show (Nx.slice [ Nx.L [ 1; 0 ] ] (d 0 (x [| 2; 3 |]))));
      c "slice" 2 (fun d ->
          show
            (Nx.slice
               [ Nx.A; Nx.T (d 0 (ints [| 2; 0 |])) ]
               (d 1 (x [| 2; 3 |]))));
      c "slice" 2 (fun d ->
          let start = d 0 (ints ~s:[||] [| 1 |]) in
          show (Nx.slice [ Nx.D (start, 2) ] (d 1 (x [| 4; 2 |]))));
      c "get" 1 (fun d -> show (Nx.get [ 1 ] (d 0 (x [| 2; 3 |]))));
      c "take" 2 (fun d ->
          show (Nx.take ~axis:1 (d 0 (ints [| 2; 0; 2 |])) (d 1 (x [| 2; 3 |]))));
      c "take_along_axis" 2 (fun d ->
          let ids = d 0 (ints ~s:[| 2; 1 |] [| 1; 0 |]) in
          show (Nx.take_along_axis ~axis:1 ids (d 1 (x [| 2; 3 |]))));
      c "set" 2 (fun d ->
          show (Nx.set [ Nx.I 0 ] (d 0 (x [| 3 |])) (d 1 (x [| 2; 3 |]))));
      c "set" 3 (fun d ->
          show
            (Nx.set
               [ Nx.T (d 0 (ints [| 1 |])) ]
               (d 1 (x [| 1; 3 |]))
               (d 2 (x [| 2; 3 |]))));
      c "scatter" 3 (fun d ->
          let ids = d 0 (ints ~s:[| 2; 2 |] [| 2; 0; 1; 1 |]) in
          show
            (Nx.scatter ~axis:1 ids (d 1 (x [| 2; 2 |])) (d 2 (x [| 2; 3 |]))));
      c "scatter" 3 (fun d ->
          let ids = d 0 (ints ~s:[| 2; 3 |] [| 1; 1; 0; 1; 0; 0 |]) in
          show
            (Nx.scatter ~combine:Add ~axis:0 ids
               (d 1 (x [| 2; 3 |]))
               (d 2 (x [| 2; 3 |]))));
      c "add" 2 (fun d -> show (Nx.add (d 0 (x [| 2; 3 |])) (d 1 (x [| 3 |]))));
      c "sub" 2 (fun d ->
          show (Nx.sub (d 0 (x [| 2; 3 |])) (d 1 (x [| 2; 3 |]))));
      c "mul" 2 (fun d ->
          show (Nx.mul (d 0 (x [| 2; 3 |])) (d 1 (x [| 2; 1 |]))));
      c "div" 2 (fun d ->
          show (Nx.div (d 0 (x [| 2; 3 |])) (d 1 (nonzero [| 2; 3 |]))));
      c "neg" 1 (fun d -> show (Nx.neg (d 0 (x [| 2; 3 |]))));
      c "square" 1 (fun d -> show (Nx.square (d 0 (x [| 2; 3 |]))));
      c "add_s" 1 (fun d -> show (Nx.add_s (d 0 (x [| 3 |])) one));
      c "sub_s" 1 (fun d -> show (Nx.sub_s (d 0 (x [| 3 |])) one));
      c "mul_s" 1 (fun d -> show (Nx.mul_s (d 0 (x [| 3 |])) one));
      c "div_s" 1 (fun d -> show (Nx.div_s (d 0 (x [| 3 |])) one));
      c "rsub_s" 1 (fun d -> show (Nx.rsub_s one (d 0 (x [| 3 |]))));
      c "rdiv_s" 1 (fun d -> show (Nx.rdiv_s one (d 0 (nonzero [| 3 |]))));
      c "equal" 2 (fun d -> show (Nx.equal (d 0 (x [| 4 |])) (d 1 (x [| 4 |]))));
      c "not_equal" 2 (fun d ->
          show (Nx.not_equal (d 0 (x [| 4 |])) (d 1 (Nx.flip (x [| 4 |])))));
      c "equal_s" 1 (fun d -> show (Nx.equal_s (d 0 (x [| 4 |])) one));
      c "not_equal_s" 1 (fun d -> show (Nx.not_equal_s (d 0 (x [| 4 |])) one));
      c "array_equal" 2 (fun d ->
          show (Nx.array_equal (d 0 (x [| 3 |])) (d 1 (x [| 3 |]))));
      c "where" 3 (fun d ->
          let cond = d 0 (Nx.not_equal_s (x [| 2; 3 |]) one) in
          show
            (Nx.where cond (d 1 (x [| 2; 3 |])) (d 2 (Nx.flip (x [| 2; 3 |])))));
      c "sum" 1 (fun d -> show (Nx.sum (d 0 (x [| 2; 3 |]))));
      c "sum" 1 (fun d ->
          show (Nx.sum ~axes:[ 1 ] ~keepdims:true (d 0 (x [| 2; 3 |]))));
      c "prod" 1 (fun d -> show (Nx.prod ~axes:[ 0 ] (d 0 (x [| 2; 3 |]))));
      c "mean" 1 (fun d -> show (Nx.mean (d 0 (x [| 2; 3 |]))));
      c "cumsum" 1 (fun d -> show (Nx.cumsum (d 0 (x [| 2; 3 |]))));
      c "cumsum" 1 (fun d -> show (Nx.cumsum ~axis:1 (d 0 (x [| 2; 3 |]))));
      c "cumprod" 1 (fun d -> show (Nx.cumprod ~axis:0 (d 0 (x [| 2; 3 |]))));
      c "cast" 1 (fun d -> show (Nx.cast Nx.complex128 (d 0 (x [| 3 |]))));
      c "bitcast" 1 (fun d -> show (Nx.bitcast Nx.uint8 (d 0 (x [| 2 |]))));
      c "contract" 2 (fun d ->
          show (Nx.contract dt ij_jk (d 0 (x [| 2; 3 |])) (d 1 (x [| 3; 2 |]))));
      c "contract" 3 (fun d ->
          show
            (Nx.contract
               ~init:(d 2 (x [| 2; 2 |]))
               dt ij_jk
               (d 0 (x [| 2; 3 |]))
               (d 1 (x [| 3; 2 |]))));
      c "einsum" 2 (fun d ->
          show (Nx.einsum ij_jk [ d 0 (x [| 2; 3 |]); d 1 (x [| 3; 2 |]) ]));
      c "matmul" 2 (fun d ->
          show (Nx.matmul (d 0 (x [| 2; 3 |])) (d 1 (x [| 3; 2 |]))));
      c "dot" 2 (fun d -> show (Nx.dot (d 0 (x [| 3 |])) (d 1 (x [| 3 |]))));
      c "vdot" 2 (fun d -> show (Nx.vdot (d 0 (x [| 3 |])) (d 1 (x [| 3 |]))));
      c "matrix_transpose" 1 (fun d ->
          show (Nx.matrix_transpose (d 0 (x [| 2; 3 |]))));
      c "tril" 1 (fun d -> show (Nx.tril (d 0 (x [| 3; 3 |]))));
      c "triu" 1 (fun d -> show (Nx.triu ~k:1 (d 0 (x [| 3; 3 |]))));
      c "diagonal" 1 (fun d -> show (Nx.diagonal (d 0 (x [| 3; 3 |]))));
      c "trace" 1 (fun d -> show (Nx.trace (d 0 (x [| 3; 3 |]))));
      c "diag" 1 (fun d -> show (Nx.diag (d 0 (x [| 3 |]))));
      c "diag" 1 (fun d -> show (Nx.diag (d 0 (x [| 3; 3 |]))));
    ]

(* The functions of floats. *)
let floats (type s) (dt : (float, s) Nx.dtype) =
  let x s = data dt s in
  let positive s = Nx.add_s (Nx.abs (x s)) 0.5 in
  (* In (0, 1). *)
  let unit s = Nx.div_s (positive s) 4. in
  let ids () = ints [| 0; 1; 0; 1 |] in
  table (D.name dt)
    [
      c "mod_" 2 (fun d ->
          show (Nx.mod_ (d 0 (x [| 4 |])) (d 1 (positive [| 4 |]))));
      c "pow" 2 (fun d ->
          show (Nx.pow (d 0 (positive [| 4 |])) (d 1 (x [| 4 |]))));
      c "fma" 3 (fun d ->
          show (Nx.fma (d 0 (x [| 4 |])) (d 1 (x [| 4 |])) (d 2 (x [| 4 |]))));
      c "maximum" 2 (fun d ->
          show (Nx.maximum (d 0 (x [| 4 |])) (d 1 (Nx.flip (x [| 4 |])))));
      c "minimum" 2 (fun d ->
          show (Nx.minimum (d 0 (x [| 4 |])) (d 1 (Nx.flip (x [| 4 |])))));
      c "recip" 1 (fun d -> show (Nx.recip (d 0 (positive [| 4 |]))));
      c "abs" 1 (fun d -> show (Nx.abs (d 0 (x [| 4 |]))));
      c "sign" 1 (fun d -> show (Nx.sign (d 0 (x [| 4 |]))));
      c "pow_s" 1 (fun d -> show (Nx.pow_s (d 0 (positive [| 4 |])) 1.5));
      c "mod_s" 1 (fun d -> show (Nx.mod_s (d 0 (x [| 4 |])) 1.5));
      c "maximum_s" 1 (fun d -> show (Nx.maximum_s (d 0 (x [| 4 |])) 0.));
      c "minimum_s" 1 (fun d -> show (Nx.minimum_s (d 0 (x [| 4 |])) 0.));
      c "rpow_s" 1 (fun d -> show (Nx.rpow_s 2. (d 0 (x [| 4 |]))));
      c "less_s" 1 (fun d -> show (Nx.less_s (d 0 (x [| 4 |])) 0.));
      c "less_equal_s" 1 (fun d -> show (Nx.less_equal_s (d 0 (x [| 4 |])) 0.));
      c "greater_s" 1 (fun d -> show (Nx.greater_s (d 0 (x [| 4 |])) 0.));
      c "greater_equal_s" 1 (fun d ->
          show (Nx.greater_equal_s (d 0 (x [| 4 |])) 0.));
      c "less" 2 (fun d ->
          show (Nx.less (d 0 (x [| 4 |])) (d 1 (Nx.flip (x [| 4 |])))));
      c "less_equal" 2 (fun d ->
          show (Nx.less_equal (d 0 (x [| 4 |])) (d 1 (Nx.flip (x [| 4 |])))));
      c "greater" 2 (fun d ->
          show (Nx.greater (d 0 (x [| 4 |])) (d 1 (Nx.flip (x [| 4 |])))));
      c "greater_equal" 2 (fun d ->
          show (Nx.greater_equal (d 0 (x [| 4 |])) (d 1 (Nx.flip (x [| 4 |])))));
      c "sqrt" 1 (fun d -> show (Nx.sqrt (d 0 (positive [| 4 |]))));
      c "rsqrt" 1 (fun d -> show (Nx.rsqrt (d 0 (positive [| 4 |]))));
      c "hypot" 2 (fun d ->
          show (Nx.hypot (d 0 (x [| 4 |])) (d 1 (positive [| 4 |]))));
      c "exp" 1 (fun d -> show (Nx.exp (d 0 (x [| 4 |]))));
      c "exp2" 1 (fun d -> show (Nx.exp2 (d 0 (x [| 4 |]))));
      c "expm1" 1 (fun d -> show (Nx.expm1 (d 0 (x [| 4 |]))));
      c "log" 1 (fun d -> show (Nx.log (d 0 (positive [| 4 |]))));
      c "log2" 1 (fun d -> show (Nx.log2 (d 0 (positive [| 4 |]))));
      c "log1p" 1 (fun d -> show (Nx.log1p (d 0 (positive [| 4 |]))));
      c "sin" 1 (fun d -> show (Nx.sin (d 0 (x [| 4 |]))));
      c "cos" 1 (fun d -> show (Nx.cos (d 0 (x [| 4 |]))));
      c "tan" 1 (fun d -> show (Nx.tan (d 0 (x [| 4 |]))));
      c "asin" 1 (fun d -> show (Nx.asin (d 0 (unit [| 4 |]))));
      c "acos" 1 (fun d -> show (Nx.acos (d 0 (unit [| 4 |]))));
      c "atan" 1 (fun d -> show (Nx.atan (d 0 (x [| 4 |]))));
      c "atan2" 2 (fun d ->
          show (Nx.atan2 (d 0 (x [| 4 |])) (d 1 (positive [| 4 |]))));
      c "sinh" 1 (fun d -> show (Nx.sinh (d 0 (x [| 4 |]))));
      c "cosh" 1 (fun d -> show (Nx.cosh (d 0 (x [| 4 |]))));
      c "tanh" 1 (fun d -> show (Nx.tanh (d 0 (x [| 4 |]))));
      c "asinh" 1 (fun d -> show (Nx.asinh (d 0 (x [| 4 |]))));
      c "acosh" 1 (fun d ->
          show (Nx.acosh (d 0 (Nx.add_s (positive [| 4 |]) 1.))));
      c "atanh" 1 (fun d -> show (Nx.atanh (d 0 (unit [| 4 |]))));
      c "erf" 1 (fun d -> show (Nx.erf (d 0 (x [| 4 |]))));
      c "erfc" 1 (fun d -> show (Nx.erfc (d 0 (x [| 4 |]))));
      c "erfinv" 1 (fun d -> show (Nx.erfinv (d 0 (unit [| 4 |]))));
      c "ndtr" 1 (fun d -> show (Nx.ndtr (d 0 (x [| 4 |]))));
      c "log_ndtr" 1 (fun d -> show (Nx.log_ndtr (d 0 (x [| 4 |]))));
      c "ndtri" 1 (fun d -> show (Nx.ndtri (d 0 (unit [| 4 |]))));
      c "lgamma" 1 (fun d -> show (Nx.lgamma (d 0 (positive [| 4 |]))));
      c "digamma" 1 (fun d -> show (Nx.digamma (d 0 (positive [| 4 |]))));
      c "lbeta" 2 (fun d ->
          show (Nx.lbeta (d 0 (positive [| 4 |])) (d 1 (positive [| 4 |]))));
      c "gammainc" 2 (fun d ->
          show (Nx.gammainc (d 0 (positive [| 4 |])) (d 1 (positive [| 4 |]))));
      c "gammaincc" 2 (fun d ->
          show (Nx.gammaincc (d 0 (positive [| 4 |])) (d 1 (positive [| 4 |]))));
      c "log_gammainc" 2 (fun d ->
          show
            (Nx.log_gammainc (d 0 (positive [| 4 |])) (d 1 (positive [| 4 |]))));
      c "log_gammaincc" 2 (fun d ->
          show
            (Nx.log_gammaincc (d 0 (positive [| 4 |])) (d 1 (positive [| 4 |]))));
      c "gammaincinv" 2 (fun d ->
          show (Nx.gammaincinv (d 0 (positive [| 4 |])) (d 1 (unit [| 4 |]))));
      c "gammainccinv" 2 (fun d ->
          show (Nx.gammainccinv (d 0 (positive [| 4 |])) (d 1 (unit [| 4 |]))));
      c "betainc" 3 (fun d ->
          show
            (Nx.betainc
               (d 0 (positive [| 4 |]))
               (d 1 (positive [| 4 |]))
               (d 2 (unit [| 4 |]))));
      c "betaincc" 3 (fun d ->
          show
            (Nx.betaincc
               (d 0 (positive [| 4 |]))
               (d 1 (positive [| 4 |]))
               (d 2 (unit [| 4 |]))));
      c "log_betainc" 3 (fun d ->
          show
            (Nx.log_betainc
               (d 0 (positive [| 4 |]))
               (d 1 (positive [| 4 |]))
               (d 2 (unit [| 4 |]))));
      c "log_betaincc" 3 (fun d ->
          show
            (Nx.log_betaincc
               (d 0 (positive [| 4 |]))
               (d 1 (positive [| 4 |]))
               (d 2 (unit [| 4 |]))));
      c "i0e" 1 (fun d -> show (Nx.i0e (d 0 (x [| 4 |]))));
      c "i1e" 1 (fun d -> show (Nx.i1e (d 0 (x [| 4 |]))));
      c "floor" 1 (fun d -> show (Nx.floor (d 0 (x [| 4 |]))));
      c "ceil" 1 (fun d -> show (Nx.ceil (d 0 (x [| 4 |]))));
      c "round" 1 (fun d -> show (Nx.round (d 0 (x [| 4 |]))));
      c "trunc" 1 (fun d -> show (Nx.trunc (d 0 (x [| 4 |]))));
      c "logical_and" 2 (fun d ->
          show (Nx.logical_and (d 0 (x [| 4 |])) (d 1 (Nx.flip (x [| 4 |])))));
      c "logical_or" 2 (fun d ->
          show (Nx.logical_or (d 0 (x [| 4 |])) (d 1 (Nx.flip (x [| 4 |])))));
      c "logical_xor" 2 (fun d ->
          show (Nx.logical_xor (d 0 (x [| 4 |])) (d 1 (Nx.flip (x [| 4 |])))));
      c "logical_not" 1 (fun d -> show (Nx.logical_not (d 0 (x [| 4 |]))));
      c "isnan" 1 (fun d -> show (Nx.isnan (d 0 (Nx.log (x [| 4 |])))));
      c "isinf" 1 (fun d -> show (Nx.isinf (d 0 (Nx.log (x [| 4 |])))));
      c "isfinite" 1 (fun d -> show (Nx.isfinite (d 0 (Nx.log (x [| 4 |])))));
      c "clamp" 1 (fun d -> show (Nx.clamp ~min:0. (d 0 (x [| 4 |]))));
      c "clamp" 1 (fun d ->
          show (Nx.clamp ~min:(-1.) ~max:1. (d 0 (x [| 4 |]))));
      c "max" 1 (fun d -> show (Nx.max (d 0 (x [| 2; 3 |]))));
      c "min" 1 (fun d -> show (Nx.min ~axes:[ 1 ] (d 0 (x [| 2; 3 |]))));
      c "all" 1 (fun d -> show (Nx.all (d 0 (x [| 2; 3 |]))));
      c "any" 1 (fun d -> show (Nx.any ~axes:[ 0 ] (d 0 (x [| 2; 3 |]))));
      c "argmax" 1 (fun d -> show (Nx.argmax (d 0 (x [| 2; 3 |]))));
      c "argmax" 1 (fun d -> show (Nx.argmax ~axis:0 (d 0 (x [| 2; 3 |]))));
      c "argmin" 1 (fun d -> show (Nx.argmin ~axis:1 (d 0 (x [| 2; 3 |]))));
      c "var" 1 (fun d -> show (Nx.var (d 0 (x [| 2; 3 |]))));
      c "std" 1 (fun d -> show (Nx.std ~ddof:1 (d 0 (x [| 5 |]))));
      c "logsumexp" 1 (fun d -> show (Nx.logsumexp (d 0 (x [| 2; 3 |]))));
      c "logmeanexp" 1 (fun d ->
          show (Nx.logmeanexp ~axes:[ 1 ] (d 0 (x [| 2; 3 |]))));
      c "cummax" 1 (fun d -> show (Nx.cummax ~axis:0 (d 0 (x [| 2; 3 |]))));
      c "cummin" 1 (fun d -> show (Nx.cummin (d 0 (x [| 2; 3 |]))));
      c "ewma" 1 (fun d -> show (Nx.ewma ~alpha:0.5 (d 0 (x [| 2; 3 |]))));
      c "ewma" 1 (fun d ->
          show (Nx.ewma ~axis:0 ~alpha:0.5 (d 0 (x [| 2; 3 |]))));
      c "sigmoid" 1 (fun d -> show (Nx.sigmoid (d 0 (x [| 4 |]))));
      c "softplus" 1 (fun d -> show (Nx.softplus (d 0 (x [| 4 |]))));
      c "softmax" 1 (fun d -> show (Nx.softmax (d 0 (x [| 2; 3 |]))));
      c "log_softmax" 1 (fun d ->
          show (Nx.log_softmax ~axes:[ 0 ] (d 0 (x [| 2; 3 |]))));
      c "conv" 2 (fun d ->
          show (Nx.conv (d 0 (x [| 1; 4; 1 |])) (d 1 (x [| 1; 2; 1 |]))));
      c "conv_transpose" 2 (fun d ->
          show
            (Nx.conv_transpose (d 0 (x [| 1; 4; 1 |])) (d 1 (x [| 1; 2; 1 |]))));
      c "correlate" 2 (fun d ->
          show (Nx.correlate (d 0 (x [| 4 |])) (d 1 (x [| 2 |]))));
      c "correlate" 2 (fun d ->
          show (Nx.correlate ~padding:`Full (d 0 (x [| 4 |])) (d 1 (x [| 2 |]))));
      c "extract_patches" 1 (fun d ->
          show
            (Nx.extract_patches ~kernel_size:[| 2 |] ~stride:[| 1 |]
               ~dilation:[| 1 |]
               ~padding:[| (0, 0) |]
               (d 0 (x [| 2; 4 |]))));
      c "combine_patches" 1 (fun d ->
          show
            (Nx.combine_patches ~output_size:[| 4 |] ~kernel_size:[| 2 |]
               ~stride:[| 1 |] ~dilation:[| 1 |]
               ~padding:[| (0, 0) |]
               (d 0 (x [| 2; 3 |]))));
      c "sort" 1 (fun d ->
          let v, i = Nx.sort (d 0 (x [| 2; 3 |])) in
          shows [ show v; show i ]);
      c "sort" 1 (fun d ->
          let v, i = Nx.sort ~descending:true ~axis:0 (d 0 (x [| 2; 3 |])) in
          shows [ show v; show i ]);
      c "argsort" 1 (fun d -> show (Nx.argsort (d 0 (x [| 5 |]))));
      c "top_k" 1 (fun d ->
          let v, i = Nx.top_k ~k:2 (d 0 (x [| 2; 3 |])) in
          shows [ show v; show i ]);
      c "order_key" 1 (fun d -> show (Nx.order_key Nx.uint64 (d 0 (x [| 4 |]))));
      c "lexsort" 1 (fun d -> show (Nx.lexsort (d 0 (x [| 5; 2 |]))));
      c "searchsorted" 2 (fun d ->
          let s = d 0 (Nx.cumsum (Nx.abs (x [| 4 |]))) in
          show (Nx.searchsorted ~side:`Left s (d 1 (x [| 3 |]))));
      c "searchsorted" 2 (fun d ->
          let s = d 0 (Nx.cumsum (Nx.abs (x [| 4 |]))) in
          show (Nx.searchsorted ~side:`Right s (d 1 (x [| 3 |]))));
      c "unique" 1 (fun d ->
          let g = Nx.unique (d 0 (x [| 6 |])) in
          showl [ g.ids; g.first; g.counts; g.count ]);
      c "unique" 1 (fun d ->
          let g = Nx.unique (d 0 (x [| 4; 2 |])) in
          showl [ g.ids; g.counts ]);
      c "quantile" 1 (fun d ->
          show (Nx.quantile [| 0.25; 0.5 |] (d 0 (x [| 5 |]))));
      c "histogram" 2 (fun d ->
          let edges = d 0 (Nx.add (x [| 3 |]) (Nx.arange_f dt 0. 3. 1.)) in
          show (Nx.histogram [ (edges, d 1 (x [| 6 |])) ]));
      c "histogram" 3 (fun d ->
          let edges = d 0 (Nx.add (x [| 3 |]) (Nx.arange_f dt 0. 3. 1.)) in
          show
            (Nx.histogram
               ~weights:(d 2 (x [| 6 |]))
               [ (edges, d 1 (x [| 6 |])) ]));
      c "vector_norm" 1 (fun d -> show (Nx.vector_norm (d 0 (x [| 4 |]))));
      c "vector_norm" 1 (fun d ->
          show (Nx.vector_norm ~ord:3. (d 0 (x [| 4 |]))));
      c "matrix_norm" 1 (fun d -> show (Nx.matrix_norm (d 0 (x [| 2; 2 |]))));
      c "matrix_norm" 1 (fun d ->
          show (Nx.matrix_norm ~ord:`One (d 0 (x [| 2; 2 |]))));
      c "rfft" 1 (fun d -> show (Nx.rfft Nx.complex64 (d 0 (x [| 2; 4 |]))));
      c "rfft" 1 (fun d ->
          show (Nx.rfft Nx.complex64 ~norm:Forward (d 0 (x [| 4 |]))));
      c "dct" 1 (fun d -> show (Nx.dct (d 0 (x [| 2; 4 |]))));
      c "dct" 1 (fun d -> show (Nx.dct ~type_:Type_1 (d 0 (x [| 2; 4 |]))));
      c "idct" 1 (fun d -> show (Nx.idct ~norm:Ortho (d 0 (x [| 4 |]))));
      c "dst" 1 (fun d -> show (Nx.dst ~type_:Type_1 (d 0 (x [| 4 |]))));
      c "idst" 1 (fun d -> show (Nx.idst (d 0 (x [| 4 |]))));
      c "idst" 1 (fun d -> show (Nx.idst ~type_:Type_4 (d 0 (x [| 4 |]))));
      c "stft" 1 (fun d ->
          show (Nx.stft Nx.complex64 ~window:2 (d 0 (x [| 4 |]))));
      c "stft" 2 (fun d ->
          show
            (Nx.stft Nx.complex64 ~window:4 ~step:2
               ~win:(d 1 (x [| 4 |]))
               (d 0 (x [| 8 |]))));
      c "reduce_segments" 2 (fun d ->
          show
            (Nx.reduce_segments Add ~segments:2
               (d 0 (ids ()))
               (d 1 (x [| 4; 2 |]))));
      c "reduce_segments" 2 (fun d ->
          show
            (Nx.reduce_segments Max ~segments:2
               (d 0 (ids ()))
               (d 1 (x [| 4 |]))));
      c "reduce_ranges" 3 (fun d ->
          let lo = d 0 (ids ()) and hi = d 1 (Nx.add_s (ids ()) 2L) in
          show (Nx.reduce_ranges Add ~lo ~hi (d 2 (x [| 5; 2 |]))));
      c "reduce_ranges" 3 (fun d ->
          let lo = d 0 (ids ()) and hi = d 1 (Nx.add_s (ids ()) 3L) in
          show (Nx.reduce_ranges Max ~lo ~hi (d 2 (x [| 5 |]))));
      c "map_segments" 2 (fun d ->
          show
            (Nx.map_segments ~segments:2
               (d 0 (ids ()))
               (fun _ r -> Nx.mul_s r 2.)
               (d 1 (x [| 4; 2 |]))));
    ]

(* The routines' functions. *)
let routines (type v s) (dt : (v, s) Nx.dtype) =
  let x s = data dt s in
  (* Symmetric positive definite: [1.25] on the diagonal, [0.25] off it. *)
  let spd () =
    let v i = Nx.Dtype.of_float dt (if i mod 4 = 0 then 1.25 else 0.25) in
    let a = Nx.create dt [| 3; 3 |] (Array.init 9 v) in
    Nx.add a (Nx.zeros_like a)
  in
  table (D.name dt)
    [
      c "cholesky" 1 (fun d -> show (Nx.cholesky (d 0 (spd ()))));
      c "cholesky" 1 (fun d ->
          show (Nx.cholesky ~triangle:Upper (d 0 (spd ()))));
      c "qr" 1 (fun d ->
          let q, r = Nx.qr (d 0 (spd ())) in
          showl [ q; r ]);
      c "lu" 1 (fun d ->
          let p, l, u = Nx.lu (d 0 (spd ())) in
          shows [ show p; show l; show u ]);
      c "svd" 1 (fun d ->
          let u, s, vh = Nx.svd (d 0 (spd ())) in
          showl [ u; s; vh ]);
      c "eigh" 1 (fun d ->
          showl
            (let w, v = Nx.eigh (d 0 (spd ())) in
             [ w; v ]));
      c "eigh" 1 (fun d -> show (fst (Nx.eigh ~triangle:Upper (d 0 (spd ())))));
      c "det" 1 (fun d -> show (Nx.det (d 0 (spd ()))));
      c "slogdet" 1 (fun d ->
          showl
            (let s, l = Nx.slogdet (d 0 (spd ())) in
             [ s; l ]));
      c "solve" 2 (fun d -> show (Nx.solve (d 0 (spd ())) (d 1 (x [| 3 |]))));
      c "solve_triangular" 2 (fun d ->
          show
            (Nx.solve_triangular (d 0 (Nx.tril (spd ()))) (d 1 (x [| 3; 1 |]))));
      c "solve_triangular" 2 (fun d ->
          show
            (Nx.solve_triangular ~transpose:Transpose
               (d 0 (Nx.tril (spd ())))
               (d 1 (x [| 3; 1 |]))));
      c "inv" 1 (fun d -> show (Nx.inv (d 0 (spd ()))));
      c "lstsq" 2 (fun d ->
          let x, r, k, s = Nx.lstsq (d 0 (spd ())) (d 1 (x [| 3 |])) in
          shows [ show x; show r; show k; show s ]);
      c "lstsq" 1 (fun d ->
          (* The donor beside its handle. *)
          let a = spd () in
          let x, _, _, _ = Nx.lstsq (d 0 a) a in
          show x);
      c "pinv" 1 (fun d -> show (Nx.pinv (d 0 (spd ()))));
      c "pinv" 1 (fun d -> show (Nx.pinv ~hermitian:true (d 0 (spd ()))));
      c "matrix_norm" 1 (fun d ->
          show (Nx.matrix_norm ~ord:`Two (d 0 (spd ()))));
    ]

(* The functions of complex numbers. *)
let complexes =
  let z s = complex_data s in
  table "complex64"
    [
      c "real" 1 (fun d -> show (Nx.real f32 (d 0 (z [| 4 |]))));
      c "imag" 1 (fun d -> show (Nx.imag f32 (d 0 (z [| 4 |]))));
      c "magnitude" 1 (fun d -> show (Nx.magnitude f32 (d 0 (z [| 4 |]))));
      c "angle" 1 (fun d -> show (Nx.angle f32 (d 0 (z [| 4 |]))));
      c "conjugate" 1 (fun d -> show (Nx.conjugate (d 0 (z [| 4 |]))));
      c "complex" 2 (fun d ->
          show
            (Nx.complex Nx.complex64
               ~re:(d 0 (data f32 [| 4 |]))
               ~im:(d 1 (data f32 [| 4 |]))));
      c "isinf" 1 (fun d -> show (Nx.isinf (d 0 (z [| 4 |]))));
      c "fft" 1 (fun d -> show (Nx.fft (d 0 (z [| 2; 4 |]))));
      c "fft" 1 (fun d -> show (Nx.fft ~norm:Ortho (d 0 (z [| 4 |]))));
      c "ifft" 1 (fun d -> show (Nx.ifft ~norm:Ortho (d 0 (z [| 4 |]))));
      c "irfft" 1 (fun d -> show (Nx.irfft f32 (d 0 (z [| 3 |]))));
      c "istft" 1 (fun d ->
          show (Nx.istft f32 ~window:2 ~step:1 (d 0 (z [| 2; 2 |]))));
      c "istft" 2 (fun d ->
          show
            (Nx.istft f32 ~window:4 ~step:2
               ~win:(d 1 (Nx.add_s (Nx.abs (data f32 [| 4 |])) 1.))
               (d 0 (z [| 3; 3 |]))));
      c "eig" 1 (fun d ->
          let w, v =
            Nx.eig (d 0 (Nx.add (Nx.eye Nx.complex64 2) (z [| 2; 2 |])))
          in
          showl [ w; v ]);
    ]

(* The functions of integers. *)
let integers =
  let x s = Nx.cast Nx.int32 (data f32 s) in
  let ids () = int32s [| 1; 0; 1; 1 |] in
  table "int32"
    [
      c "bitwise_and" 2 (fun d ->
          show (Nx.bitwise_and (d 0 (x [| 4 |])) (d 1 (Nx.flip (x [| 4 |])))));
      c "bitwise_or" 2 (fun d ->
          show (Nx.bitwise_or (d 0 (x [| 4 |])) (d 1 (Nx.flip (x [| 4 |])))));
      c "bitwise_xor" 2 (fun d ->
          show (Nx.bitwise_xor (d 0 (x [| 4 |])) (d 1 (Nx.flip (x [| 4 |])))));
      c "bitwise_not" 1 (fun d -> show (Nx.bitwise_not (d 0 (x [| 4 |]))));
      c "lshift" 1 (fun d -> show (Nx.lshift (d 0 (x [| 4 |])) 1));
      c "rshift" 1 (fun d -> show (Nx.rshift (d 0 (x [| 4 |])) 1));
      c "mod_" 2 (fun d -> show (Nx.mod_ (d 0 (x [| 4 |])) (d 1 (ids ()))));
      c "pow" 2 (fun d -> show (Nx.pow (d 0 (x [| 4 |])) (d 1 (ids ()))));
      c "abs" 1 (fun d -> show (Nx.abs (d 0 (x [| 4 |]))));
      c "sign" 1 (fun d -> show (Nx.sign (d 0 (x [| 4 |]))));
      c "recip" 1 (fun d -> show (Nx.recip (d 0 (x [| 4 |]))));
      c "maximum" 2 (fun d ->
          show (Nx.maximum (d 0 (x [| 4 |])) (d 1 (ids ()))));
      c "less" 2 (fun d -> show (Nx.less (d 0 (x [| 4 |])) (d 1 (ids ()))));
      c "max" 1 (fun d -> show (Nx.max (d 0 (x [| 4 |]))));
      c "positions" 1 (fun d -> show (Nx.positions (d 0 (Nx.abs (x [| 4 |])))));
      c "positions" 1 (fun d ->
          show (Nx.positions ~size:4 (d 0 (Nx.abs (x [| 4 |])))));
      c "one_hot" 1 (fun d -> show (Nx.one_hot ~num_classes:2 (d 0 (ids ()))));
      c "unique" 1 (fun d ->
          let g = Nx.unique (d 0 (x [| 6 |])) in
          showl [ g.ids; g.first; g.counts; g.count ]);
      c "sort" 1 (fun d ->
          showl
            (let v, _ = Nx.sort (d 0 (x [| 5 |])) in
             [ v ]));
    ]

(* The functions of booleans. *)
let booleans =
  let b s = Nx.greater_s (data f32 s) 0. in
  table "bool"
    [
      c "logical_and" 2 (fun d ->
          show (Nx.logical_and (d 0 (b [| 4 |])) (d 1 (Nx.flip (b [| 4 |])))));
      c "logical_not" 1 (fun d -> show (Nx.logical_not (d 0 (b [| 4 |]))));
      c "count" 1 (fun d -> show (Nx.count (d 0 (b [| 2; 3 |]))));
      c "count" 1 (fun d -> show (Nx.count ~axes:[ 1 ] (d 0 (b [| 2; 3 |]))));
      c "all" 1 (fun d -> show (Nx.all (d 0 (b [| 4 |]))));
      c "Infix.( && )" ~by:"logical_and" 2 (fun d ->
          show Nx.Infix.(d 0 (b [| 4 |]) && d 1 (Nx.flip (b [| 4 |]))));
      c "Infix.( || )" ~by:"logical_or" 2 (fun d ->
          show Nx.Infix.(d 0 (b [| 4 |]) || d 1 (Nx.flip (b [| 4 |]))));
    ]

(* The operators, at one dtype: each is a function above. *)
let operators =
  let x s = data f32 s in
  let x2 fn by op =
    c fn ~by 2 (fun d ->
        show (op (d 0 (x [| 4 |])) (d 1 (Nx.flip (x [| 4 |])))))
  in
  let xs fn by op = c fn ~by 1 (fun d -> show (op (d 0 (x [| 4 |])) 1.5)) in
  table "float32"
    [
      x2 "( + )" "add" Nx.( + );
      x2 "( - )" "sub" Nx.( - );
      x2 "( * )" "mul" Nx.( * );
      x2 "( / )" "div" Nx.( / );
      x2 "( ** )" "pow" Nx.( ** );
      c "( ~- )" ~by:"neg" 1 (fun d -> show Nx.(-d 0 (x [| 4 |])));
      xs "( +$ )" "add_s" Nx.( +$ );
      xs "( -$ )" "sub_s" Nx.( -$ );
      xs "( *$ )" "mul_s" Nx.( *$ );
      xs "( /$ )" "div_s" Nx.( /$ );
      x2 "Infix.( = )" "equal" Nx.Infix.( = );
      x2 "Infix.( <> )" "not_equal" Nx.Infix.( <> );
      x2 "Infix.( < )" "less" Nx.Infix.( < );
      x2 "Infix.( <= )" "less_equal" Nx.Infix.( <= );
      x2 "Infix.( > )" "greater" Nx.Infix.( > );
      x2 "Infix.( >= )" "greater_equal" Nx.Infix.( >= );
      xs "Infix.( =$ )" "equal_s" Nx.Infix.( =$ );
      xs "Infix.( <>$ )" "not_equal_s" Nx.Infix.( <>$ );
      xs "Infix.( <$ )" "less_s" Nx.Infix.( <$ );
      xs "Infix.( <=$ )" "less_equal_s" Nx.Infix.( <=$ );
      xs "Infix.( >$ )" "greater_s" Nx.Infix.( >$ );
      xs "Infix.( >=$ )" "greater_equal_s" Nx.Infix.( >=$ );
    ]

(* The samplers and keys of [Nx.Rng], at one dtype. *)
let rng =
  let x s = data f32 s in
  let positive s = Nx.add_s (Nx.abs (x s)) 0.5 in
  let unit s = Nx.div_s (positive s) 4. in
  let key d i = Nx.Rng.of_tensor (d i (int32s [| 7; 11 |])) in
  let shown_key k = show (Nx.Rng.to_tensor k) in
  table "float32"
    [
      c "Rng.split" 1 (fun d ->
          shows (Array.to_list (Array.map shown_key (Nx.Rng.split (key d 0)))));
      c "Rng.split_batch" 1 (fun d ->
          shown_key (Nx.Rng.split_batch ~n:3 (key d 0)));
      c "Rng.fold_in" 1 (fun d -> shown_key (Nx.Rng.fold_in (key d 0) 5));
      c "Rng.fold_in_tensor" 2 (fun d ->
          shown_key (Nx.Rng.fold_in_tensor (key d 0) (d 1 (int32s [| 1; 2 |]))));
      c "Rng.place" 1 (fun d -> shown_key (Nx.Rng.place Nx.Host.on (key d 0)));
      c "Rng.bits" 1 (fun d -> show (Nx.Rng.bits ~key:(key d 0) [| 3 |]));
      c "Rng.uniform" 1 (fun d ->
          show (Nx.Rng.uniform ~key:(key d 0) f32 [| 3 |]));
      c "Rng.normal" 1 (fun d ->
          show (Nx.Rng.normal ~key:(key d 0) f32 [| 3 |]));
      c "Rng.exponential" 1 (fun d ->
          show (Nx.Rng.exponential ~key:(key d 0) f32 [| 3 |]));
      c "Rng.randint" 1 (fun d ->
          show (Nx.Rng.randint ~key:(key d 0) ~high:5 [| 3 |]));
      c "Rng.permutation" 1 (fun d ->
          show (Nx.Rng.permutation ~key:(key d 0) 4));
      c "Rng.bernoulli" 2 (fun d ->
          show (Nx.Rng.bernoulli ~key:(key d 1) (d 0 (unit [| 4 |]))));
      c "Rng.gamma" 2 (fun d ->
          show (Nx.Rng.gamma ~key:(key d 1) (d 0 (positive [| 4 |]))));
      c "Rng.beta" 3 (fun d ->
          show
            (Nx.Rng.beta ~key:(key d 2)
               (d 0 (positive [| 4 |]))
               (d 1 (positive [| 4 |]))));
      c "Rng.von_mises" 2 (fun d ->
          show (Nx.Rng.von_mises ~key:(key d 1) (d 0 (positive [| 4 |]))));
      c "Rng.poisson" 2 (fun d ->
          show (Nx.Rng.poisson ~key:(key d 1) (d 0 (positive [| 4 |]))));
      c "Rng.binomial" 3 (fun d ->
          show
            (Nx.Rng.binomial ~key:(key d 2)
               (d 0 (int32s [| 1; 4; 0; 7 |]))
               (d 1 (unit [| 4 |]))));
      c "Rng.truncated_normal" 3 (fun d ->
          show
            (Nx.Rng.truncated_normal ~key:(key d 2)
               (d 0 (Nx.neg (positive [| 4 |])))
               (d 1 (positive [| 4 |]))));
      c "Rng.dirichlet" 2 (fun d ->
          show (Nx.Rng.dirichlet ~key:(key d 1) (d 0 (positive [| 2; 3 |]))));
      c "Rng.categorical" 2 (fun d ->
          show (Nx.Rng.categorical ~key:(key d 1) (d 0 (x [| 2; 3 |]))));
    ]

(* Host crossings at one dtype. *)
let host =
  table "float64"
    [
      c "to_bigarray" 1 (fun d ->
          show
            (Nx.of_bigarray
               (Nx.to_bigarray Bigarray.float64
                  (d 0 (data Nx.float64 [| 2; 3 |])))));
    ]

let all_cases =
  List.concat
    [
      any_dtype Nx.float64;
      any_dtype Nx.float16;
      any_dtype Nx.complex64;
      any_dtype Nx.int32;
      floats Nx.float64;
      floats Nx.float16;
      routines Nx.float64;
      routines Nx.complex64;
      complexes;
      integers;
      booleans;
      operators;
      rng;
      host;
    ]

(* Each case with its name, its function's and dtype's, with its rank among
   their cases past the first. *)
let named_cases =
  let seen = Hashtbl.create 64 in
  List.map
    (fun c ->
      let key = (c.fn, c.dt) in
      let k = Option.value (Hashtbl.find_opt seen key) ~default:0 + 1 in
      Hashtbl.replace seen key k;
      let name =
        if k = 1 then Printf.sprintf "%s, %s" c.fn c.dt
        else Printf.sprintf "%s, %s, case %d" c.fn c.dt k
      in
      (name, c))
    all_cases

let case_count = List.length named_cases
let case_at = Array.of_list named_cases

(* [x] dead: donated to [Nx.copy], which consumed it. *)
let kill x =
  ignore (Nx.copy (Nx.donate x));
  x

let test_dead_names =
  let each =
    List.concat_map
      (fun (name, c) -> List.init c.operands (fun i -> (name, c, i)))
      named_cases
  in
  cases
    ~name:(fun (name, _, i) -> Printf.sprintf "%s, operand %d" name (i + 1))
    "a dead operand names the function called" each
    (fun (name, c, i) ->
      let run () = c.run (fun j x -> if j = i then kill x else x) in
      match c.dead with
      | Prints death -> equal ~msg:name string death (run ())
      | Raises ->
          let named = function
            | Invalid_argument m ->
                String.starts_with ~prefix:(c.by ^ ": operand ") m
                && String.ends_with ~suffix:" was donated to Nx.copy" m
            | _ -> false
          in
          raises_match ~msg:name named run)

(* A case: one of the table's, and the operands it donates, as a mask. *)
let donations =
  let pp ppf (k, mask) =
    Format.fprintf ppf "%s, donating mask %d" (fst case_at.(k)) mask
  in
  Gen.with_pp pp
    (let open Gen in
     let* k = int_range 0 (case_count - 1) in
     let c = snd case_at.(k) in
     let+ mask = int_range 1 ((1 lsl c.operands) - 1) in
     (k, mask))

(* Each case with each operand donated alone, then with all. *)
let seeds =
  List.concat
    (List.mapi
       (fun k (_, c) ->
         List.init c.operands (fun i -> (k, 1 lsl i))
         @ if c.operands > 1 then [ (k, (1 lsl c.operands) - 1) ] else [])
       named_cases)

let law_donated (k, mask) =
  let name, c = case_at.(k) in
  let plain = c.run (fun _ x -> x) in
  let donated =
    c.run (fun i x -> if mask land (1 lsl i) <> 0 then Nx.donate x else x)
  in
  equal ~msg:name string plain donated

let test_donated =
  prop ~examples:seeds "a donated operand gives the undonated result" donations
    law_donated

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
             test "a refused composite kills no donor" test_refused_composite;
             test "an aligning consumer checks before it consumes"
               test_aligned_refusal;
             test "a value of every set is not donated" test_left_alone;
             test "a donor dies at a consumer an interpretation receives"
               test_dies_when_received;
             test "a donor dies naming the function the user called"
               test_dies_at_the_call;
             test_dead_names;
             test_donated;
             stateful ~domains:2 "reads and consumers on two domains" commands;
             stateful ~domains:2 "handles taken on two domains" handles;
             stateful ~domains:2 "two-handle consumers on two domains" crossed;
             test "of crossed consumers on two domains, one succeeds"
               test_crossed;
           ];
         group "reuse"
           [
             test "set writes into a donated target" test_set_in_place;
             test "scatter writes into a donated target" test_scatter_in_place;
             test "contract writes into a donated init" test_init_in_place;
             test "an undonated or shared operand keeps its elements" test_kept;
             test "an operation may read the donor beside its handle"
               test_donor_beside;
             test "a consumer inside a composite computes beside its donor"
               test_inside_composite;
             law_scatter_bits;
             law_init_bits;
           ];
       ])
