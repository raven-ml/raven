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
  invalid ~sub:"Nx.copy: operand 1 was donated to Nx.copy" (fun () -> Nx.copy d);
  (* The handle a movement read and the one it passed on are handles of one
     donor: whichever is consumed first, the other raises naming that
     consumer. *)
  let h = Nx.donate (fresh [| 1.; 2. |]) in
  let v = Nx.reshape [| 1; 2 |] h in
  equal ~msg:"the handle read" (array float_exact) [| 1.; 2. |]
    (elements (Nx.copy h));
  invalid ~sub:"Nx.copy: operand 1 was donated to Nx.copy" (fun () -> Nx.copy v);
  let h = Nx.donate (fresh [| 1.; 2. |]) in
  let v = Nx.reshape [| 1; 2 |] h in
  ignore (Nx.neg v);
  invalid ~sub:"Nx.copy: operand 1 was donated to Nx.neg" (fun () -> Nx.copy h)

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
    (elements (Nx.add x2 x2));
  (* Refused on its first operand, an add kills no handle of its second,
     which it would broadcast. *)
  let dead = fresh_at [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  ignore (Nx.copy (Nx.donate dead));
  let b = Nx.donate (fresh [| 7.; 8. |]) in
  invalid ~sub:"Nx.add: operand 1 was donated to Nx.copy" (fun () ->
      Nx.add dead b);
  equal ~msg:"the broadcast operand's handle lives" (array float_exact)
    [| 7.; 8. |] (elements (Nx.copy b))

(* An operation that broadcasts a handle, refused by its dtype or by a dead
   operand, kills nothing and names the operand that died by its place in the
   call. *)
let test_broadcast_refused () =
  let ints s xs =
    let a = Nx.Repr.of_array Nx.Host.v (A.of_array D.Int32 s xs) in
    Nx.add a (Nx.zeros_like a)
  in
  let y = ints [| 2 |] [| 1l; 2l |] in
  let z = ints [| 2; 2 |] [| 3l; 4l; 5l; 6l |] in
  let h = Nx.donate y in
  invalid ~sub:"Nx.atan2: Atan2 does not take int32" (fun () -> Nx.atan2 h z);
  equal ~msg:"the integers' handle reaches a later consumer" (array int32)
    [| 1l; 2l |]
    (elements (Nx.copy h));
  let dead = fresh_at [| 2; 2 |] [| 1.; 0.; 1.; 0. |] in
  ignore (Nx.copy (Nx.donate dead));
  let h = Nx.donate (fresh [| 1.; 2. |]) in
  invalid ~sub:"Nx.logical_and: operand 2 was donated to Nx.copy" (fun () ->
      Nx.logical_and h dead);
  equal ~msg:"the floats' handle reaches a later consumer" (array float_exact)
    [| 1.; 2. |]
    (elements (Nx.copy h))

(* Inside an interpretation that receives every operation, movements
   included, an operation that broadcasts a handle and refuses its dtype
   kills nothing either: it refuses before it broadcasts. *)
let test_broadcast_refused_received () =
  let received f =
    Nx.Prim.interpret ~name:"test.extent" Extent
      (fun _ ~by op -> Nx.Prim.eval ~by op)
      (fun _ -> f ())
  in
  let ints s xs =
    let a = Nx.Repr.of_array Nx.Host.v (A.of_array D.Int32 s xs) in
    Nx.add a (Nx.zeros_like a)
  in
  let y = ints [| 2 |] [| 1l; 2l |] in
  let z = ints [| 2; 2 |] [| 3l; 4l; 5l; 6l |] in
  let h = Nx.donate y in
  invalid ~sub:"Nx.atan2: Atan2 does not take int32" (fun () ->
      received (fun () -> Nx.atan2 h z));
  invalid ~sub:"Nx.atan2: Atan2 does not take int32" (fun () ->
      received (fun () -> Nx.atan2 z h));
  equal ~msg:"the integers' handle reaches a later consumer" (array int32)
    [| 1l; 2l |]
    (elements (Nx.copy h));
  let truths s =
    Nx.less (fresh_at s (Array.make (Array.fold_left ( * ) 1 s) 0.))
      (fresh_at s (Array.make (Array.fold_left ( * ) 1 s) 1.))
  in
  let p = Nx.donate (truths [| 2 |]) and q = truths [| 2; 2 |] in
  invalid ~sub:"Nx.fma: Fma does not take bool" (fun () ->
      received (fun () -> Nx.fma p q q));
  equal ~msg:"the booleans' handle reaches a later consumer" (array bool)
    [| true; true |]
    (elements (Nx.copy p))

(* Inside an interpretation that receives every operation, a call that
   broadcasts its operands is the one consumer of their handles, as it is
   eagerly: a check only the call's last operation makes, as one donation
   read twice, or a rule refusing that operation's dtype, kills no donor. *)
let test_broadcast_one_consumer () =
  (* A float32 map, which [received ~refuse:true] refuses. *)
  let map (type r) (op : r Nx.Prim.t) =
    match op with
    | Map { outs = [ dt ]; _ } -> Option.is_some (D.equal_witness dt f32)
    | _ -> false
  in
  let received ?(refuse = false) f =
    Nx.Prim.interpret ~name:"test.extent" Extent
      (fun _ ~by op ->
        if refuse && map op then invalid_arg "test.extent: no float32 map";
        Nx.Prim.eval ~by op)
      (fun _ -> f ())
  in
  let x = fresh [| 1.; 2. |] in
  let h = Nx.donate x in
  let h' = Nx.reshape [| 1; 2 |] h in
  let wide = fresh_at [| 2; 2 |] [| 1.; 0.; 1.; 0. |] in
  let c = Nx.less wide (fresh_at [| 2; 2 |] (Array.make 4 0.5)) in
  let twice by = "Nx.donate: " ^ by ^ " reads one donation twice" in
  let both by f =
    invalid ~sub:(twice by) f;
    invalid ~sub:(twice by) (fun () -> received f)
  in
  both "Nx.where" (fun () -> Nx.where c h h);
  both "Nx.add" (fun () -> Nx.add h' h);
  both "Nx.less" (fun () -> Nx.less h' h);
  both "Nx.logical_and" (fun () -> Nx.logical_and h' h);
  equal ~msg:"the donor lives" (array float_exact) [| 2.; 4. |]
    (elements (Nx.add x x));
  invalid ~sub:"test.extent: no float32 map" (fun () ->
      received ~refuse:true (fun () -> Nx.add h wide));
  invalid ~sub:"test.extent: no float32 map" (fun () ->
      received ~refuse:true (fun () -> Nx.where c wide h'));
  equal ~msg:"the handle passed on lives" (array float_exact) [| 1.; 2. |]
    (elements (Nx.copy h'));
  invalid ~sub:"Nx.add: operand 1 was donated to Nx.copy" (fun () ->
      Nx.add x x)

(* Inside an interpretation that receives every operation, a call that moves
   a handle before the operation that reads it is the one consumer of its
   handles, as it is eagerly: a check only the operation makes, as one
   donation read twice, or a rule refusing the operation, kills no donor; a
   call that returns kills its donor once, naming the call. *)
let test_moved_one_consumer () =
  let is (type r) kind (op : r Nx.Prim.t) =
    match (kind, op) with
    | `Assemble, Assemble _ | `Gather, Gather _ | `Contract, Contract _ -> true
    | _ -> false
  in
  let received ?refuse f =
    Nx.Prim.interpret ~name:"test.extent" Extent
      (fun _ ~by op ->
        (match refuse with
        | Some kind when is kind op -> invalid_arg "test.extent: refused"
        | _ -> ());
        Nx.Prim.eval ~by op)
      (fun _ -> f ())
  in
  let twice by f =
    let sub = "Nx.donate: " ^ by ^ " reads one donation twice" in
    invalid ~sub f;
    invalid ~sub (fun () -> received f)
  in
  let refused kind f =
    invalid ~sub:"test.extent: refused" (fun () -> received ~refuse:kind f)
  in
  let lives msg x xs =
    equal ~msg (array float_exact) xs (elements (Nx.add x (Nx.zeros_like x)))
  in
  let dies by x =
    invalid ~sub:("operand 1 was donated to " ^ by) (fun () -> Nx.add x x)
  in
  let ints s xs =
    let a = Nx.create Nx.int64 s xs in
    Nx.add a (Nx.zeros_like a)
  in
  (* A stack lifts each operand to the result's rank. *)
  let x = fresh [| 1.; 2. |] and y = fresh [| 3.; 4. |] in
  let h = Nx.donate x in
  twice "Nx.stack" (fun () -> Nx.stack [ h; h ]);
  refused `Assemble (fun () -> Nx.stack [ h; y ]);
  lives "the stack's donor" x [| 1.; 2. |];
  equal ~msg:"the stack" (array float_exact) [| 1.; 2.; 3.; 4. |]
    (elements (Nx.reshape [| 4 |] (received (fun () -> Nx.stack [ h; y ]))));
  dies "Nx.stack" x;
  (* Positions [2; 1] broadcast a [1; 3] operand along its other axis. *)
  let x = fresh_at [| 1; 3 |] [| 1.; 2.; 3. |] in
  let h = Nx.donate x and p = ints [| 2; 1 |] [| 2L; 0L |] in
  let q = Nx.donate (ints [| 1; 3 |] [| 0L; 1L; 2L |]) in
  let q' = Nx.reshape [| 3; 1 |] q in
  twice "Nx.take_along_axis" (fun () -> Nx.take_along_axis ~axis:1 q' q);
  refused `Gather (fun () -> Nx.take_along_axis ~axis:1 p h);
  lives "the gather's donor" x [| 1.; 2.; 3. |];
  equal ~msg:"the gather" (array float_exact) [| 3.; 1. |]
    (elements
       (Nx.reshape [| 2 |]
          (received (fun () -> Nx.take_along_axis ~axis:1 p h))));
  dies "Nx.take_along_axis" x;
  (* A [1; 2; 2] operand against a [2; 2; 2] one drops its unit axis. *)
  let x = fresh_at [| 1; 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let y = fresh_at [| 2; 2; 2 |] [| 1.; 0.; 0.; 1.; 2.; 0.; 0.; 2. |] in
  let h = Nx.donate x in
  let h' = Nx.broadcast_to [| 2; 2; 2 |] h in
  twice "Nx.matmul" (fun () -> Nx.matmul h h');
  refused `Contract (fun () -> Nx.matmul h y);
  lives "the product's donor" x [| 1.; 2.; 3.; 4. |];
  equal ~msg:"the product" (array float_exact)
    [| 1.; 2.; 3.; 4.; 2.; 4.; 6.; 8. |]
    (elements (Nx.reshape [| 8 |] (received (fun () -> Nx.matmul h y))));
  dies "Nx.matmul" x;
  (* Three operands make two contractions. *)
  let chain = Nx.Pattern.v "i j, j k, k l -> i l | j k" in
  let id = fresh_at [| 2; 2 |] [| 1.; 0.; 0.; 1. |] in
  let two = fresh_at [| 2; 2 |] [| 2.; 0.; 0.; 2. |] in
  let x = fresh_at [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let h = Nx.donate x in
  twice "Nx.einsum" (fun () -> Nx.einsum chain [ h; id; h ]);
  refused `Contract (fun () -> Nx.einsum chain [ h; id; two ]);
  lives "the einsum's donor" x [| 1.; 2.; 3.; 4. |];
  let product = [| 2.; 4.; 6.; 8. |] in
  equal ~msg:"the einsum" (array float_exact) product
    (elements
       (Nx.reshape [| 4 |]
          (received (fun () -> Nx.einsum chain [ h; id; two ]))));
  dies "Nx.einsum" x;
  let x = fresh_at [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  equal ~msg:"the eager einsum" (array float_exact) product
    (elements (Nx.reshape [| 4 |] (Nx.einsum chain [ Nx.donate x; id; two ])));
  dies "Nx.einsum" x

(* As [test_moved_one_consumer], for selections: a set that reshapes its
   target to scatter into it, and a slice that slices its operand before it
   gathers along positions held in data. *)
let test_selected_one_consumer () =
  let refusing f =
    Nx.Prim.interpret ~name:"test.extent" Extent
      (fun (type r) _ ~by (op : r Nx.Prim.t) : r ->
        match op with
        | Scatter _ | Gather _ -> invalid_arg "test.extent: refused"
        | _ -> Nx.Prim.eval ~by op)
      (fun _ -> f ())
  in
  let x = fresh_at [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let v = fresh [| 5.; 6. |] in
  let p = Nx.create Nx.int64 [| 1 |] [| 1L |] in
  invalid ~sub:"test.extent: refused" (fun () ->
      refusing (fun () -> Nx.set [ L [ 1; 0 ]; I 0 ] v (Nx.donate x)));
  invalid ~sub:"test.extent: refused" (fun () ->
      refusing (fun () -> Nx.slice [ I 0; T p ] (Nx.donate x)));
  equal ~msg:"the donor lives" (array float_exact) [| 2.; 4.; 6.; 8. |]
    (elements (Nx.reshape [| 4 |] (Nx.add x x)))

(* A complex value of float32 parts at complex128 casts each part, then
   assembles them: operations that read the parts once, eagerly too. *)
let test_parts_one_consumer () =
  let x = fresh [| 1.; 2. |] in
  let h = Nx.donate x in
  invalid ~sub:"Nx.donate: Nx.complex reads one donation twice" (fun () ->
      Nx.complex Nx.complex128 ~re:h ~im:h);
  equal ~msg:"the donor lives" (array float_exact) [| 2.; 4. |]
    (elements (Nx.add x x));
  let z = Nx.complex Nx.complex128 ~re:h ~im:(fresh [| 3.; 4. |]) in
  equal ~msg:"the real parts" (array float_exact) [| 1.; 2. |]
    (elements (Nx.real Nx.float32 z));
  equal ~msg:"the imaginary parts" (array float_exact) [| 3.; 4. |]
    (elements (Nx.imag Nx.float32 z));
  invalid ~sub:"operand 1 was donated to Nx.complex" (fun () -> Nx.add x x)

(* Inside an interpretation that receives every operation, a call that
   broadcasts a handle consumes it once, when it returns: the donor lives
   while the call's operations run, and dies naming the call. *)
let test_broadcast_consumed_once () =
  let x = fresh [| 1.; 2. |] in
  let wide = fresh_at [| 2; 2 |] [| 10.; 20.; 30.; 40. |] in
  let during = ref [] in
  let y =
    Nx.Prim.interpret ~name:"test.extent" Extent
      (fun _ ~by op ->
        let r = Nx.Prim.eval ~by op in
        during := elements (Nx.copy x) :: !during;
        r)
      (fun _ -> Nx.add (Nx.donate x) wide)
  in
  equal ~msg:"the sum" (array float_exact) [| 11.; 22.; 31.; 42. |]
    (elements (Nx.reshape [| 4 |] y));
  equal ~msg:"the donor during the call's operations" (list (array float_exact))
    [ [| 1.; 2. |]; [| 1.; 2. |] ]
    !during;
  invalid ~sub:"Nx.add: operand 1 was donated to Nx.add" (fun () -> Nx.add x x)

(* Inside an interpretation that receives every operation, an operation
   consumes its handles once the rule returns: one the rule refuses, or whose
   own checks refuse it while the rule computes it, kills no donor. *)
let test_received_refused () =
  let refusing f =
    Nx.Prim.interpret ~name:"test.extent" Extent
      (fun _ ~by _ -> invalid_arg (by ^ ": refused by test.extent"))
      (fun _ -> f ())
  in
  let forwarding f =
    Nx.Prim.interpret ~name:"test.extent" Extent
      (fun _ ~by op -> Nx.Prim.eval ~by op)
      (fun _ -> f ())
  in
  let x = fresh [| 1.; 2. |] and y = fresh [| 3.; 4. |] in
  invalid ~sub:"Nx.add: refused by test.extent" (fun () ->
      refusing (fun () -> Nx.add (Nx.donate x) y));
  equal ~msg:"the sum's donor lives" (array float_exact) [| 2.; 4. |]
    (elements (Nx.add x x));
  let truths () = Nx.less (fresh [| 0.; 2. |]) (fresh [| 1.; 1. |]) in
  let b = truths () and u = truths () in
  let p = Nx.create Nx.int64 [| 2 |] [| 1L; 0L |] in
  invalid ~sub:"Nx.scatter: Add does not take bool" (fun () ->
      forwarding (fun () -> Nx.scatter ~combine:Add ~axis:0 p u (Nx.donate b)));
  equal ~msg:"the scatter's target lives" (array bool) [| true; false |]
    (elements (Nx.copy b))

(* A kind refusing its operands' dtypes kills nothing, though its result would
   fit a handle's memory: an operation checks before it takes the memory. *)
let test_dtype_refused () =
  let ints xs =
    let a = Nx.Repr.of_array Nx.Host.v (A.of_array D.Int32 [| 2 |] xs) in
    Nx.add a (Nx.zeros_like a)
  in
  let y = ints [| 1l; 2l |] and x = ints [| 3l; 4l |] in
  let refused sub f = invalid ~sub f in
  refused "Nx.atan2: Atan2 does not take int32" (fun () ->
      Nx.atan2 (Nx.donate y) x);
  refused "Nx.atan2: Atan2 does not take int32" (fun () ->
      Nx.atan2 x (Nx.donate y));
  refused "Nx.sin: Sin does not take int32" (fun () -> Nx.sin (Nx.donate y));
  equal ~msg:"the integers live" (array int32) [| 2l; 4l |]
    (elements (Nx.add y y));
  let f = fresh [| 1.; 2. |] and g = fresh [| 3.; 4. |] in
  refused "Nx.bitwise_and: And does not take float32" (fun () ->
      Nx.bitwise_and (Nx.donate f) g);
  equal ~msg:"the floats live" (array float_exact) [| 2.; 4. |]
    (elements (Nx.add f f));
  let truths () = Nx.less (fresh [| 0.; 2. |]) (fresh [| 1.; 1. |]) in
  let p = truths () and q = truths () and r = truths () in
  let fma = "Nx.fma: Fma does not take bool" in
  refused fma (fun () -> Nx.fma (Nx.donate p) q r);
  refused fma (fun () -> Nx.fma q (Nx.donate p) r);
  refused fma (fun () -> Nx.fma q r (Nx.donate p));
  equal ~msg:"the booleans live" (array bool) [| true; false |]
    (elements (Nx.copy p))

(* An operation whose rule refuses its operands kills nothing, though its
   result would take a handle's memory: a scatter's combine refusing the
   target's dtype or updates of another shape than the positions, a
   contraction's init of another shape than its result, a map's load of
   another shape than its loop. *)
let test_rule_refused () =
  let truths () = Nx.less (fresh [| 0.; 2. |]) (fresh [| 1.; 1. |]) in
  let b = truths () and u = truths () in
  let p = Nx.create Nx.int64 [| 2 |] [| 1L; 0L |] in
  invalid ~sub:"Nx.scatter: Add does not take bool" (fun () ->
      Nx.scatter ~combine:Add ~axis:0 p u (Nx.donate b));
  equal ~msg:"the booleans live" (array bool) [| true; false |]
    (elements (Nx.copy b));
  let x = fresh [| 1.; 2. |] in
  invalid ~sub:"Nx.scatter" (fun () ->
      Nx.scatter ~axis:0 p (fresh [| 1.; 2.; 3. |]) (Nx.donate x));
  equal ~msg:"the scatter's target lives" (array float_exact) [| 1.; 2. |]
    (elements (Nx.copy x));
  let a = fresh_at [| 2; 3 |] (Array.make 6 1.) in
  let g = fresh_at [| 2; 3 |] (Array.make 6 2.) in
  invalid ~sub:"test: init of shape" (fun () ->
      Nx.Prim.eval ~by:"test"
        (Contract
           {
             order = Chain;
             batch = [||];
             contracting = [| (1, 1) |];
             acc = D.Any D.Float32;
             out = f32;
             a;
             b = a;
             init = Some (Nx.donate g);
             a_scale = None;
             b_scale = None;
             segments = None;
           }));
  equal ~msg:"the init lives" (array float_exact) (Array.make 6 2.)
    (elements (Nx.copy g));
  let x = fresh [| 1.; 2. |] in
  let prog =
    Nx_kernel.Prog.v
      ~ins:[| D.Any D.Float32; D.Any D.Float32 |]
      [| Nx_kernel.Prog.In 0 |] ~outs:[| 0 |]
  in
  invalid ~sub:"test: load 1 has shape" (fun () ->
      Nx.Prim.eval ~by:"test"
        (Map
           {
             layout = Nx_array.Layout.contiguous [| 2 |];
             prog;
             outs = [ f32 ];
             loads = [| Plain (Nx.donate x); Plain (fresh [| 1.; 2.; 3. |]) |];
           }));
  equal ~msg:"the map's load lives" (array float_exact) [| 1.; 2. |]
    (elements (Nx.copy x))

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

(* A placement onto memory devices borrows the host's memory: the shards of a
   value placed there lie in one allocation, apart or interleaved, and a
   device whose window is empty holds none of it. *)
module Two = (val Nx.devices [ Nx_support.memory 1; Nx_support.memory 2 ])

let test_shards_of_one_allocation () =
  let donated s axis =
    let msg = Printf.sprintf "[%d; %d] split along %d" s.(0) s.(1) axis in
    let xs = Array.init (s.(0) * s.(1)) (fun i -> i mod 3 = 0) in
    let x =
      Nx.place (Two.split ~axis)
        (Nx.Repr.of_array Nx.Host.v (A.of_array D.Bool s xs))
    in
    let y = Nx.logical_not (Nx.donate x) in
    equal ~msg (array bool) (Array.map not xs) (elements y);
    invalid ~sub:"Nx.copy: operand 1 was donated to Nx.logical_not" (fun () ->
        Nx.copy x)
  in
  donated [| 1; 1 |] 0;
  donated [| 1; 1 |] 1;
  donated [| 4; 2 |] 0;
  donated [| 4; 2 |] 1

(* A consumer that is no operation, such as a compiled call, consumes its
   handles as an operation does: it reads each handle as its arrays and the
   donor beside it as it was, kills the donors naming itself, and kills none
   where it refuses one. *)
let consume xs = Nx.Prim.consume ~by:"test.call" xs
let read_f32 (Nx.Prim.Any v) = elements (Nx.cast f32 v)

let test_is_handle () =
  let x = fresh [| 1.; 2. |] in
  let d = Nx.donate x in
  equal ~msg:"a handle" bool true (Nx.Prim.donated d);
  equal ~msg:"its donor" bool false (Nx.Prim.donated x);
  equal ~msg:"a handle passed on" bool true
    (Nx.Prim.donated (Nx.reshape [| 1; 2 |] (Nx.donate (fresh [| 1.; 2. |]))));
  equal ~msg:"a value of every set" bool false
    (Nx.Prim.donated (Nx.donate (Nx.zeros f32 [| 2 |])));
  ignore (consume [ Any d ]);
  equal ~msg:"a consumed handle" bool true (Nx.Prim.donated d)

let test_consume () =
  let x = fresh [| 1.; 2. |] and y = fresh [| 3.; 4. |] in
  let read = consume [ Any (Nx.donate x); Any x; Any y ] in
  equal ~msg:"read as called"
    (list (array float_exact))
    [ [| 1.; 2. |]; [| 1.; 2. |]; [| 3.; 4. |] ]
    (List.map read_f32 read);
  equal ~msg:"no handle read" (list bool) [ false; false; false ]
    (List.map (fun (Nx.Prim.Any v) -> Nx.Prim.donated v) read);
  invalid ~sub:"Nx.add: operand 1 was donated to test.call" (fun () ->
      Nx.add x x);
  equal ~msg:"another operand lives" (array float_exact) [| 6.; 8. |]
    (elements (Nx.add y y))

let test_consume_refused () =
  let x1 = fresh [| 1.; 2. |] in
  let d1 = Nx.donate x1 in
  ignore (Nx.mul (Nx.donate x1) (fresh [| 3.; 4. |]));
  let x2 = fresh [| 5.; 6. |] in
  invalid ~sub:"test.call: operand 2 was donated to Nx.mul" (fun () ->
      consume [ Any (Nx.donate x2); Any d1 ]);
  equal ~msg:"the other donor lives" (array float_exact) [| 10.; 12. |]
    (elements (Nx.add x2 x2));
  let x = fresh [| 1.; 2. |] in
  invalid ~sub:"Nx.donate: test.call reads one donation twice" (fun () ->
      consume [ Any (Nx.donate x); Any (Nx.donate x) ]);
  equal ~msg:"two handles of one donor kill nothing" (array float_exact)
    [| 2.; 4. |]
    (elements (Nx.add x x));
  let d = Nx.donate (fresh [| 1.; 2. |]) in
  ignore (consume [ Any d ]);
  invalid ~sub:"test.call: operand 1 was donated to test.call" (fun () ->
      consume [ Any d ])

(* Of an operation and a call over the same two handles, in either order, the
   first consumes them and the second raises naming it. *)
let test_call_order () =
  let pair () =
    (Nx.donate (fresh [| 1.; 2. |]), Nx.donate (fresh [| 10.; 20. |]))
  in
  let h1, h2 = pair () in
  equal ~msg:"the operation" (array float_exact) [| 11.; 22. |]
    (elements (Nx.add h1 h2));
  invalid ~sub:"test.call: operand 1 was donated to Nx.add" (fun () ->
      consume [ Any h2; Any h1 ]);
  let h1, h2 = pair () in
  ignore (consume [ Any h2; Any h1 ]);
  invalid ~sub:"Nx.add: operand 1 was donated to test.call" (fun () ->
      Nx.add h1 h2)

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
  (* A consumer that broadcasts its handle consumes it when it returns:
     under the observer, and under an Extent rule that keeps its operands
     and computes nothing. *)
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

(* The bytes [d] holds after four rounds of collection, each of which returns
   to the device what it collected. *)
let settled_on d =
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create d 0);
    Rig.free_cache d
  done;
  Rig.allocated d

let settled () = settled_on device

(* [f (make ())] and the bytes [d] holds more after it, its cache included, so
   that a donor freed after a copy still counts. One uncounted run comes
   first, and the count before is read settled, so that no memory collected
   late serves the counted run. *)
let grown_on d make f =
  ignore (f (make ()));
  let inputs = make () in
  let before = settled_on d in
  let y = f inputs in
  (y, Rig.allocated d - before)

let grown make f = grown_on device make f

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

(* A value made from OCaml data or a bigarray's copy, or over an array its
   caller gives up, owns its memory, which its donation then writes; one over
   an array its caller keeps shares it. *)
let test_made_in_place () =
  let n = 1 lsl 16 in
  let xs = Array.make n 1. in
  let negated ~msg make =
    let y, grew = grown_on Rig.host make (fun x -> Nx.neg (Nx.donate x)) in
    equal ~msg (array float_exact) (Array.make n (-1.)) (elements y);
    grew
  in
  let in_place ~msg make =
    less ~msg:(msg ^ ": no memory of the result's size") int ~than:(4 * n)
      (negated ~msg make)
  in
  in_place ~msg:"Nx.create" (fun () -> Nx.create f32 [| n |] xs);
  in_place ~msg:"Nx.init" (fun () -> Nx.init f32 [| n |] (fun _ -> 1.));
  in_place ~msg:"Nx.of_bigarray" (fun () ->
      Nx.of_bigarray
        (Bigarray.genarray_of_array1
           (Bigarray.Array1.init Bigarray.float32 Bigarray.c_layout n (fun _ ->
                1.))));
  in_place ~msg:"Nx.Repr.unsafe_of_array" (fun () ->
      Nx.Repr.unsafe_of_array Nx.Host.v (A.of_array D.Float32 [| n |] xs));
  at_least ~msg:"Nx.Repr.of_array: a fresh result" int ~than:(4 * n)
    (negated ~msg:"Nx.Repr.of_array" (fun () ->
         Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 [| n |] xs)))

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

(* A read for the extent of a call leaves the memory to the next donation,
   where a crossing ({!Nx.Repr.array}) shares it for good. *)
let test_read_in_place () =
  let read c =
    Nx.Repr.read c (fun a ->
        Nx_array.read Bigarray.float32 a (fun b ->
            ignore (Bigarray.Genarray.get b [| 5; 0 |])));
    c
  in
  let y, grew =
    grown
      (fun () -> (read (cache ()), nines [| width |]))
      (fun (c, u) -> Nx.set Nx.[ I 5 ] u (Nx.donate c))
  in
  in_place ~msg:"after Repr.read" ~bytes:size grew;
  equal (array float_exact) (with_row 5 9.) (elements y);
  let _, grew =
    grown
      (fun () ->
        let c = cache () in
        ignore (Nx.Repr.array c);
        (c, nines [| width |]))
      (fun (c, u) -> Nx.set Nx.[ I 5 ] u (Nx.donate c))
  in
  copied ~msg:"after Repr.array" ~bytes:size grew

(* [donated_during_read donate] reads a cache with [Nx.Repr.read] and, while
   the read holds it, runs [donate c], which donates it: the donation computes
   into fresh memory and leaves the read's bytes as they were. *)
let donated_during_read donate =
  let c = cache () in
  let y =
    Nx.Repr.read c (fun a ->
        let before = A.to_array a in
        let y = donate c in
        let fresh = Option.get (Nx.Repr.array y) in
        equal ~msg:"fresh memory" bool false
          (Rig.Buffer.overlaps (A.buffer a) (A.buffer fresh));
        equal ~msg:"the read's bytes" (array float_exact) before (A.to_array a);
        y)
  in
  equal (array float_exact) (with_row 5 9.) (elements y)

let set_row c = Nx.set Nx.[ I 5 ] (nines [| width |]) (Nx.donate c)

let test_read_donated_inside () = donated_during_read set_row

(* The second domain starts inside [f], after Repr.read has taken its claim,
   and [f] joins it before returning, so its donation runs while the claim is
   held on every run. *)
let test_read_donated_across () =
  donated_during_read (fun c -> Domain.join (Domain.spawn (fun () -> set_row c)))

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

(* An eager broadcasting call on one domain, while an [Extent] lives on
   another, frees its donor's memory at once: the extent is not around it.
   The donor stays reachable, so no collection explains the bytes. *)
let test_extent_elsewhere () =
  let n = 4096 in
  let wide = on_mem [| 2; n |] (Array.make (2 * n) 1.) in
  let run () =
    let x = on_mem [| n |] (Array.make n 2.) in
    let eager () =
      let before = settled () in
      let y = Nx.add (Nx.donate x) wide in
      Rig.free_cache device;
      (y, Rig.allocated device - before)
    in
    let y, grew =
      Nx.Prim.interpret ~name:"test.extent" Extent
        (fun _ ~by op -> Nx.Prim.eval ~by op)
        (fun _ -> Domain.join (Domain.spawn eager))
    in
    (x, y, grew)
  in
  ignore (run ());
  let x, y, grew = run () in
  less ~msg:"the donor freed" int ~than:(4 * 2 * n) grew;
  equal ~msg:"the sum" (array float_exact) (Array.make (2 * n) 3.)
    (elements (Nx.reshape [| 2 * n |] y));
  invalid ~sub:"donated to Nx.add" (fun () -> Nx.copy x)

(* An eager call that reads a donation in several operations, as an einsum
   of three operands or a slice at two lists of positions held in data,
   frees the donor's memory once it returns, as one operation does. The
   donors stay reachable, so no collection explains the bytes. A slice that
   gives its operand itself keeps its elements. *)
let test_several_free () =
  let p = Nx.Pattern.v "i j, j k, k l -> i l | j k" in
  let ones = on_mem [| width; 1 |] (Array.make width 1.) in
  let two = on_mem [| 1; 1 |] [| 2. |] in
  let freed f =
    let run () =
      let c = cache () in
      let before = settled () in
      let y = f c in
      Rig.free_cache device;
      (c, y, Rig.allocated device - before)
    in
    ignore (run ());
    run ()
  in
  let c, y, grew = freed (fun c -> Nx.einsum p [ Nx.donate c; ones; two ]) in
  less ~msg:"the einsum's donor freed" int ~than:(-size / 2) grew;
  equal ~msg:"the einsum" (array float_exact)
    (Array.init rows (fun r -> Float.of_int (2 * r * width)))
    (elements (Nx.reshape [| rows |] y));
  invalid ~sub:"donated to Nx.einsum" (fun () -> Nx.copy c);
  let cols_at = positions [| 1 |] [| 7L |] in
  let c, y, grew =
    freed (fun c ->
        let rows_at = positions [| 1 |] [| 3L |] in
        Nx.slice [ T (Nx.donate rows_at); T cols_at ] (Nx.donate c))
  in
  less ~msg:"the slice's donor freed" int ~than:(-size / 2) grew;
  equal ~msg:"the slice" (array float_exact) [| 3. |]
    (elements (Nx.reshape [| 1 |] y));
  invalid ~sub:"donated to Nx.slice" (fun () -> Nx.copy c);
  let c = cache () in
  let y = Nx.slice [ A; A ] (Nx.donate c) in
  equal ~msg:"the operand itself" (array float_exact) original (elements y)

(* A Values rule may keep the arrays it receives, as a tape keeps a
   derivative's primals. An eager call that reads a donation in several
   operations, one of which the rule receives beside a traced value, leaves
   the donor's memory to the arrays the rule kept. *)
type ('v, 's, 'd) Nx.Prim.payload +=
  | Held : ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.Prim.payload

let test_kept_by_rule () =
  let p = Nx.Pattern.v "i j, j k, k l -> i l | j k" in
  let ones = on_mem [| width; 1 |] (Array.make width 1.) in
  let two = on_mem [| 1; 1 |] [| 2. |] in
  let kept = ref [] in
  let rule i ~by op =
    let held (type v s d) (x : (v, s, d) Nx.t) : (v, s, d) Nx.t =
      match Nx.Prim.payload i x with Some (Held y) -> y | _ -> x
    in
    let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
    List.iter
      (fun (Nx.Prim.Any x) ->
        if Nx.Prim.owner x = None && Nx.shape x = [| rows; width |] then
          kept := (fun () -> elements (Nx.cast f32 x)) :: !kept)
      xs;
    Nx.Prim.eval ~by (Nx.Prim.map held op)
  in
  let c = cache () in
  let y =
    Nx.Prim.interpret ~name:"test.tape" Values rule (fun i ->
        let ones = Nx.Prim.traced i (Nx.Prim.form ones) (Held ones) in
        Nx.einsum p [ Nx.donate c; ones; two ])
  in
  equal ~msg:"the einsum" (array float_exact)
    (Array.init rows (fun r -> Float.of_int (2 * r * width)))
    (elements (Nx.reshape [| rows |] y));
  invalid ~sub:"donated to Nx.einsum" (fun () -> Nx.copy c);
  equal ~msg:"the arrays the rule received" int 1 (List.length !kept);
  List.iter
    (fun read -> equal ~msg:"a kept array" (array float_exact) original (read ()))
    !kept

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
   domain gives back the other. Every run is explained by some order, every
   result is computed from the elements, and a refusal names a function that
   held one of the operation's handles. *)
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
        [ "Nx.add"; "Nx.mul"; "Nx.sub"; "test.call" ]
    with
    | Some f -> f
    | None -> msg

let refused f =
  match f () with ys -> Ok ys | exception Invalid_argument m -> Error (taker m)

let free h = List.mem None h.states
let union a b = List.sort_uniq compare (a @ b)

(* Whether [e], a consumer of two handles, may hold a handle in flight: it
   claims them one at a time and gives back those it took when it loses
   another, so a free handle can be seen held by it. A consumer of one handle
   claims nothing more once it holds it. *)
let in_flight e = e = "Nx.sub" || e = "test.call"

(* Whether [h] may have gone to [e]: taken by it, or free and held by it in
   flight. *)
let lost_to e h = List.mem (Some e) h.states || (in_flight e && free h)

(* The states of [h] once a call lost it to [e]: taken by [e], or, held in
   flight, free again. *)
let lost h e =
  let held = if free h then [ Some e ] else [] in
  let back = if free h && in_flight e then [ None ] else [] in
  union (List.filter (( = ) (Some e)) h.states) (union held back)

(* A one-handle operation [by] whose result's elements are [f]'s. *)
let judge1 by f h = function
  | Ok (Ok ys) ->
      if not (free h) then fail (by ^ " took a taken handle");
      equal (array float_exact) (f h.ys) ys;
      h.states <- [ Some by ]
  | Ok (Error e) ->
      cover "a later operation raises naming the first" true;
      if not (lost_to e h) then fail (by ^ " raised naming " ^ e);
      h.states <- lost h e
  | Error exn -> raise exn

(* Twos beside [h]: its shape answers whether it lives or not. *)
let twos h = fresh (Array.make (Nx.shape h).(0) 2.)
let add_ref = judge1 "Nx.add" (Array.map (fun v -> v +. 2.))
let mul_ref = judge1 "Nx.mul" (Array.map (fun v -> v *. 2.))
let add_sys h = refused (fun () -> elements (Nx.add h (twos h)))
let mul_sys h = refused (fun () -> elements (Nx.mul h (twos h)))

(* A movement reads a free handle and kills nothing. Its result is a handle no
   command reads. *)
let reshape_ref h = function
  | Ok (Ok _) ->
      if not (free h) then fail "Nx.reshape read a taken handle";
      h.states <- [ None ]
  | Ok (Error e) ->
      if not (lost_to e h) then fail ("Nx.reshape raised naming " ^ e);
      h.states <- lost h e
  | Error exn -> raise exn

let reshape_sys h =
  refused (fun () ->
      ignore (Nx.reshape [| 1; (Nx.shape h).(0) |] h);
      [||])

(* [sub h1 h2] checks both handles, then takes them in an order the promise
   leaves open: a loser lost one handle, and gave back the other. The result's
   elements are [f]'s. *)
let pair_ref f by h1 h2 out =
  if h1 == h2 then
    match out with
    | Ok r -> equal taken (Error "twice") r
    | Error exn -> raise exn
  else
    match out with
    | Ok (Ok ys) ->
        if not (free h1 && free h2) then fail (by ^ " took a taken handle");
        cover "two handles consumed" true;
        equal (array float_exact) (Array.map2 f h1.ys h2.ys) ys;
        h1.states <- [ Some by ];
        h2.states <- [ Some by ]
    | Ok (Error e) ->
        let to_e = lost_to e in
        if not (to_e h1 || to_e h2) then fail (by ^ " raised naming " ^ e);
        cover "one handle refused, the other free"
          ((to_e h1 && free h2) || (to_e h2 && free h1));
        (* Either may be the one lost; the other keeps its state. *)
        let after h other =
          union
            (if to_e h then lost h e else [])
            (if to_e other then h.states else [])
        in
        let s1 = after h1 h2 and s2 = after h2 h1 in
        h1.states <- s1;
        h2.states <- s2
    | Error exn -> raise exn

let sub2_ref = pair_ref ( -. ) "Nx.sub"
let sub2_sys h1 h2 = refused (fun () -> elements (Nx.sub h1 h2))

(* A call of two handles, which consumes them as an operation does. *)
let call2_ref = pair_ref ( +. ) "test.call"

let call2_sys h1 h2 =
  refused (fun () ->
      match consume [ Any h1; Any h2 ] with
      | [ a; b ] -> Array.map2 ( +. ) (read_f32 a) (read_f32 b)
      | _ -> fail "a call's reads, one per operand")

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
    command "sub2" (handle ^-> handle ^-> judges taken) sub2_ref sub2_sys;
  ]

(* Consumers of two handles alone, so that two domains often take one pair in
   opposite orders. *)
let crossed =
  [
    command "make" (pair @-> makes handle) make_handle make_handle_sys;
    command "sub2" (handle ^-> handle ^-> judges taken) sub2_ref sub2_sys;
    command "call2" (handle ^-> handle ^-> judges taken) call2_ref call2_sys;
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
      c "flatten" 1 (fun d -> show (Nx.flatten (d 0 (x [||]))));
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
      c "rearrange" 1 (fun d ->
          show
            (Nx.rearrange ~sizes:[ ("b", 3) ]
               (Nx.Pattern.v "a (b c) -> c a b")
               (d 0 (x [| 2; 6 |]))));
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
      c "mean" 1 (fun d ->
          show (Nx.mean ~axes:[ 0 ] ~keepdims:true (d 0 (x [| 2; 3 |]))));
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
      c "einsum" 3 (fun d ->
          show
            (Nx.einsum
               (Nx.Pattern.v "i j, j k, k l -> l i | j k")
               [ d 0 (x [| 2; 3 |]); d 1 (x [| 3; 2 |]); d 2 (x [| 2; 2 |]) ]));
      c "matmul" 2 (fun d ->
          show (Nx.matmul (d 0 (x [| 2; 3 |])) (d 1 (x [| 3; 2 |]))));
      c "contract" 2 (fun d ->
          show
            (Nx.contract dt
               (Nx.Pattern.v "i j, j k -> k i | j")
               (d 0 (x [| 2; 3 |]))
               (d 1 (x [| 3; 2 |]))));
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
      c "any" 1 (fun d ->
          show (Nx.any ~axes:[ 0 ] ~keepdims:true (d 0 (x [| 2; 3 |]))));
      c "argmax" 1 (fun d -> show (Nx.argmax (d 0 (x [| 2; 3 |]))));
      c "argmax" 1 (fun d -> show (Nx.argmax ~axis:0 (d 0 (x [| 2; 3 |]))));
      c "argmin" 1 (fun d -> show (Nx.argmin ~axis:1 (d 0 (x [| 2; 3 |]))));
      c "argmin" 1 (fun d ->
          show (Nx.argmin ~axis:1 ~keepdims:true (d 0 (x [| 2; 3 |]))));
      c "var" 1 (fun d -> show (Nx.var (d 0 (x [| 2; 3 |]))));
      c "std" 1 (fun d -> show (Nx.std ~ddof:1 (d 0 (x [| 5 |]))));
      c "logsumexp" 1 (fun d -> show (Nx.logsumexp (d 0 (x [| 2; 3 |]))));
      c "logsumexp" 1 (fun d ->
          show (Nx.logsumexp ~axes:[ 1 ] ~keepdims:true (d 0 (x [| 2; 3 |]))));
      c "logmeanexp" 1 (fun d ->
          show (Nx.logmeanexp ~axes:[ 1 ] (d 0 (x [| 2; 3 |]))));
      c "logmeanexp" 1 (fun d ->
          show (Nx.logmeanexp ~keepdims:true (d 0 (x [| 2; 3 |]))));
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
      c "count" 1 (fun d ->
          show (Nx.count ~axes:[ 1 ] ~keepdims:true (d 0 (b [| 2; 3 |]))));
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

(* Attention, at one dtype: the query is operand 1, then the keys, the values
   and a page table, then the lengths. *)
let attention =
  let x s = data f32 s in
  let shown (o, lse) = showl [ o; lse ] in
  let pair d =
    {
      Nx.q = d 0 (ints [| 0; 1; 2 |]);
      k = d 1 (ints [| 2; 1; 0 |]);
      len = d 2 (ints [| 3; 3; 3 |]);
      tq = 2;
    }
  in
  table "float32"
    [
      c "attention" 3 (fun d ->
          let q = d 0 (x [| 1; 2; 2 |]) in
          let kv =
            Nx.Dense { k = d 1 (x [| 1; 3; 2 |]); v = d 2 (x [| 1; 3; 2 |]) }
          in
          shown (Nx.attention ~kv q));
      c "attention" 4 (fun d ->
          let q = d 0 (x [| 1; 2; 2 |]) in
          let kv =
            Nx.Dense { k = d 1 (x [| 1; 3; 2 |]); v = d 2 (x [| 1; 3; 2 |]) }
          in
          let lengths = d 3 (ints ~s:[||] [| 2 |]) in
          shown (Nx.attention ~lengths ~mask:Nx.causal ~kv q));
      c "attention" 4 (fun d ->
          let q = d 0 (x [| 1; 2; 2 |]) in
          let kv =
            Nx.Paged
              {
                k = d 1 (x [| 2; 1; 2; 2 |]);
                v = d 2 (x [| 2; 1; 2; 2 |]);
                table = d 3 (ints [| 1; 0 |]);
              }
          in
          shown (Nx.attention ~kv q));
      c "causal" 3 (fun d -> show (Nx.causal (pair d)));
    ]

(* An array a score returns is an operand after the call's own: a donated one
   gives the undonated result and dies naming [Nx.attention]. One that a
   function reads in the score dies naming that function, which the user
   called. *)
let test_captured () =
  let run score =
    let q = data f32 [| 1; 2; 2 |] in
    let kv =
      Nx.Dense { k = data f32 [| 1; 3; 2 |]; v = data f32 [| 1; 3; 2 |] }
    in
    let o, lse = Nx.attention ~score ~kv q in
    showl [ o; lse ]
  in
  List.iter
    (fun (name, score) ->
      let bias = data f32 [| 1; 1; 3 |] in
      let plain = run (score bias) in
      equal ~msg:name string plain (run (score (Nx.donate bias)));
      invalid ~sub:("Nx.copy: operand 1 was donated to " ^ name) (fun () ->
          Nx.copy bias))
    [
      ("Nx.attention", fun bias _ _ -> bias);
      ("Nx.add", fun bias _ s -> Nx.add s bias);
    ]

(* Functions over structures, at one dtype: a structure's arrays are operands
   in walk order, after the operands before it in the call. *)
type ('a, 'd) keyed = { held : 'a; key : 'd Nx.Rng.key }

module Keyed = struct
  type ('a, 'd) t = ('a, 'd) keyed

  let walk c x =
    let open Nx.Ptree.Walk in
    let held = field c "held" leaf x.held in
    let key = field c "key" Nx.Rng.walk x.key in
    { held; key }
end

exception Failed of string

let structures =
  let module P = Nx.Ptree in
  let x s = data f32 s in
  let two () = P.pair P.tensor P.tensor in
  let shown2 (a, b) = shows [ show a; show b ] in
  let f (a, b) (a', b') = (Nx.add a a', Nx.mul b b') in
  table "float32"
    [
      c "associative_scan" 2 (fun d ->
          shown2
            (Nx.associative_scan (two ()) f
               (d 0 (x [| 4 |]), d 1 (x [| 4; 2 |]))));
      c "check" 3 (fun d ->
          let ok = d 0 (Nx.greater_s (x [| 4 |]) 0.) in
          let fail _ (a, b) = Failed (shown2 (a, b)) in
          let data = (d 1 (x [| 4 |]), d 2 (x [| 1 |])) in
          match Nx.check (two ()) ok data fail with
          | () -> "passed"
          | exception Failed s -> s);
      c "Ptree.dot" 4 (fun d ->
          show
            (P.dot (two ()) f32
               (d 0 (x [| 3 |]), d 1 (complex_data [| 2 |]))
               (d 2 (x [| 3 |]), d 3 (complex_data [| 2 |]))));
      c "Ptree.norm" 2 (fun d ->
          show
            (P.norm (two ()) f32
               (d 0 (x [| 3 |]), d 1 (complex_data [| 2 |]))));
      c "Ptree.scale" 3 (fun d ->
          shown2
            (P.scale (two ())
               (d 0 (x [||]))
               (d 1 (x [| 3 |]), d 2 (complex_data [| 2 |]))));
      c "Ptree.axpy" 5 (fun d ->
          shown2
            (P.axpy (two ())
               (d 0 (x [||]))
               (d 1 (x [| 3 |]), d 2 (complex_data [| 2 |]))
               (d 3 (x [| 3 |]), d 4 (complex_data [| 2 |]))));
      c "Ptree.place" 2 (fun d ->
          let key = Nx.Rng.of_tensor (d 1 (int32s [| 7; 11 |])) in
          let y =
            P.place (module Keyed) Nx.Host.on { held = d 0 (x [| 3 |]); key }
          in
          shows [ show y.held; show (Nx.Rng.to_tensor y.key) ]);
      c "Ptree.cast" 1 (fun d ->
          let y =
            P.cast (module Keyed) Nx.float64
              { held = d 0 (x [| 3 |]); key = Nx.Rng.key 7 }
          in
          show y.held);
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
      attention;
      structures;
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
          let want =
            Printf.sprintf "%s: operand %d was donated to Nx.copy" c.by (i + 1)
          in
          raises ~msg:name (Invalid_argument want) run)

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

(* Each case with every donation mask, up to three operands; past three, each
   operand donated alone, then all. *)
let seeds =
  List.concat
    (List.mapi
       (fun k (_, c) ->
         let n = c.operands in
         if n <= 3 then List.init ((1 lsl n) - 1) (fun m -> (k, m + 1))
         else List.init n (fun i -> (k, 1 lsl i)) @ [ (k, (1 lsl n) - 1) ])
       named_cases)

let law_donated (k, mask) =
  let name, c = case_at.(k) in
  let plain = c.run (fun _ x -> x) in
  let donating : 'v 's. int -> ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t
      =
   fun i x -> if mask land (1 lsl i) <> 0 then Nx.donate x else x
  in
  equal ~msg:name string plain (c.run donating);
  let received =
    Nx.Prim.interpret ~name:"test.extent" Extent
      (fun _ ~by op -> Nx.Prim.eval ~by op)
      (fun _ -> c.run donating)
  in
  equal ~msg:(name ^ ", received") string plain received

let test_donated =
  prop ~examples:seeds
    "a donated operand gives the undonated result, received too" donations
    law_donated

(* Case [c] inside an Extent whose rule refuses the [k]th operation it
   receives, a movement too, none for [k = 0], donating the operands in
   [mask]: the operations it received then, whether the call raised, and the
   donors it left dead. *)
let refused_at k c mask =
  let n = ref 0 and donors = ref [] in
  let rule _ ~by op =
    incr n;
    if !n = k then invalid_arg "test.extent: refused";
    Nx.Prim.eval ~by op
  in
  let donate : 'v 's. int -> ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t
      =
   fun i x ->
    if mask land (1 lsl i) = 0 then x
    else begin
      let dead () =
        match Nx.copy x with
        | _ -> None
        | exception Invalid_argument _ -> Some i
      in
      donors := dead :: !donors;
      Nx.donate x
    end
  in
  let raised =
    match
      Nx.Prim.interpret ~name:"test.extent" Extent rule (fun _ -> c.run donate)
    with
    | _ -> false
    | exception _ -> true
  in
  (!n, raised, List.filter_map (fun dead -> dead ()) !donors)

(* The rule refusing each operation of a case in turn, with each of the
   case's donation masks of [seeds]. *)
let test_refused_received =
  cases
    ~name:(fun (k, mask) ->
      Printf.sprintf "%s, donating mask %d" (fst case_at.(k)) mask)
    "a call refused inside an Extent kills no donor" seeds
    (fun (k, mask) ->
      let name, c = case_at.(k) in
      let total, _, _ = refused_at 0 c mask in
      let killed = ref [] in
      for j = 1 to total do
        match refused_at j c mask with
        | _, true, (_ :: _ as dead) ->
            List.iter
              (fun i ->
                killed :=
                  Printf.sprintf "operation %d of %d, operand %d" j total
                    (i + 1)
                  :: !killed)
              dead
        | _ -> ()
      done;
      equal ~msg:name (list string) [] (List.rev !killed))

(* A traced value of a Values interpretation, which its rule keeps nothing
   for. *)
type ('v, 's, 'd) Nx.Prim.payload += Mark : ('v, 's, 'd) Nx.Prim.payload

(* Case [c] under a Values interpretation that traces operand 0 and refuses
   the [k]th operation it receives, none for [k = 0], donating the operands
   in [mask]: as {!refused_at}. The rule's results are traced, so every
   operation that reads one reaches the rule. *)
let refused_traced k c mask =
  let n = ref 0 and donors = ref [] in
  let rule i ~by op =
    incr n;
    if !n = k then invalid_arg "test.values: refused";
    Nx.Prim.results ~by (fun _ form -> Nx.Prim.traced i form Mark) op
  in
  let raised =
    match
      Nx.Prim.interpret ~name:"test.values" Values rule (fun i ->
          let pass :
              'v 's. int -> ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t =
           fun j x ->
            if j = 0 then Nx.Prim.traced i (Nx.Prim.form x) Mark
            else if mask land (1 lsl j) = 0 then x
            else begin
              let dead () =
                match Nx.copy x with
                | _ -> None
                | exception Invalid_argument _ -> Some j
              in
              donors := dead :: !donors;
              Nx.donate x
            end
          in
          c.run pass)
    with
    | _ -> false
    | exception _ -> true
  in
  (!n, raised, List.filter_map (fun dead -> dead ()) !donors)

(* Each case of several operands, operand 0 traced, with every donation mask
   of the others up to three operands; past three, each other operand alone,
   then all. *)
let traced_seeds =
  List.concat
    (List.mapi
       (fun k (_, c) ->
         let n = c.operands in
         if n < 2 then []
         else if n <= 3 then
           List.init ((1 lsl (n - 1)) - 1) (fun m -> (k, 2 * (m + 1)))
         else
           List.init (n - 1) (fun i -> (k, 1 lsl (i + 1)))
           @ [ (k, (1 lsl n) - 2) ])
       named_cases)

let test_refused_traced =
  cases
    ~name:(fun (k, mask) ->
      Printf.sprintf "%s, donating mask %d" (fst case_at.(k)) mask)
    "a call refused beside a traced operand kills no donor" traced_seeds
    (fun (k, mask) ->
      let name, c = case_at.(k) in
      let total, _, _ = refused_traced 0 c mask in
      let killed = ref [] in
      for j = 1 to total do
        match refused_traced j c mask with
        | _, true, (_ :: _ as dead) ->
            List.iter
              (fun i ->
                killed :=
                  Printf.sprintf "operation %d of %d, operand %d" j total
                    (i + 1)
                  :: !killed)
              dead
        | _ -> ()
      done;
      equal ~msg:name (list string) [] (List.rev !killed))

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
             test "a dtype refusal kills no donor" test_dtype_refused;
             test "a rule's refusal kills no donor" test_rule_refused;
             test "a refused broadcast kills no donor" test_broadcast_refused;
             test "a refused broadcast an interpretation receives kills no donor"
               test_broadcast_refused_received;
             test
               "a call an interpretation receives is its broadcasts' one \
                consumer" test_broadcast_one_consumer;
             test
               "a call an interpretation receives is its movements' one \
                consumer" test_moved_one_consumer;
             test
               "a selection an interpretation receives is its movements' one \
                consumer" test_selected_one_consumer;
             test "a complex value reads its parts once"
               test_parts_one_consumer;
             test
               "a broadcast handle an interpretation receives dies when the \
                call returns" test_broadcast_consumed_once;
             test "an operation an interpretation refuses kills no donor"
               test_received_refused;
             test "a refused composite kills no donor" test_refused_composite;
             test "an aligning consumer checks before it consumes"
               test_aligned_refusal;
             test "a value of every set is not donated" test_left_alone;
             test "a value whose shards lie in one allocation is donated"
               test_shards_of_one_allocation;
             test "a donor dies at a consumer an interpretation receives"
               test_dies_when_received;
             test "a donor dies naming the function the user called"
               test_dies_at_the_call;
             test_dead_names;
             test_donated;
             test_refused_received;
             test_refused_traced;
             test "a captured array dies at the call that reads it"
               test_captured;
             stateful ~domains:2 "reads and consumers on two domains" commands;
             stateful ~domains:2 "handles taken on two domains" handles;
             stateful ~domains:2 "two-handle consumers on two domains" crossed;
             test "of crossed consumers on two domains, one succeeds"
               test_crossed;
             test "a consumer that is no operation consumes as one"
               test_consume;
             test "a refused consumer that is no operation kills no donor"
               test_consume_refused;
             test "donated tells a handle, live or not" test_is_handle;
             test "an operation and a call, in either order, one consumes"
               test_call_order;
           ];
         group "reuse"
           [
             test "a value that owns its memory is written in place"
               test_made_in_place;
             test "set writes into a donated target" test_set_in_place;
             test "scatter writes into a donated target" test_scatter_in_place;
             test "contract writes into a donated init" test_init_in_place;
             test "a value read by Repr.read is still written in place"
               test_read_in_place;
             test "a donation during Repr.read computes into fresh memory"
               test_read_donated_inside;
             test
               "a donation on another domain during Repr.read computes into \
                fresh memory" test_read_donated_across;
             test "an undonated or shared operand keeps its elements" test_kept;
             test "an operation may read the donor beside its handle"
               test_donor_beside;
             test "a consumer inside a composite computes beside its donor"
               test_inside_composite;
             test
               "an eager call beside an Extent on another domain frees its \
                donor" test_extent_elsewhere;
             test "an eager call of several operations frees its donor"
               test_several_free;
             test "an eager call leaves the arrays a rule kept"
               test_kept_by_rule;
             law_scatter_bits;
             law_init_bits;
           ];
       ])
