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
             test "a donor dies at a consumer an interpretation receives"
               test_dies_when_received;
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
             law_scatter_bits;
             law_init_bits;
           ];
       ])
