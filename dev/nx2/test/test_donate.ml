(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Donation through Nx: when a donor and its handles die, the one-read rule,
   handles passed on by movements, sharers keeping their elements, what an
   interpretation reads, and values that donation leaves alone. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype

let f32 = Nx.float32
let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f

(* [x] as messages name it: [float32 [2; 3]]. *)
let text x =
  Printf.sprintf "%s [%s]"
    (D.name (Nx.dtype x))
    (String.concat "; " (List.map string_of_int (Array.to_list (Nx.shape x))))

(* What [by] raises for a dead [x] that [consumer] consumed. *)
let dead_in by x consumer =
  Printf.sprintf "%s: %s was donated to %s" by (text x) consumer
let elements x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))

(* A value in memory of its own: computed, so no Repr crossing shares it. *)
let fresh xs =
  let a =
    Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 [| Array.length xs |] xs)
  in
  Nx.add a (Nx.zeros_like a)

(* As [fresh], of shape [s]: a reshape would make a view, which shares. *)
let fresh_at s xs =
  let a =
    Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 (Iarray.of_array s) xs)
  in
  Nx.add a (Nx.zeros_like a)

let test_dies_at_its_consumer () =
  let x = fresh [| 1.; 2. |] in
  let d = Nx.donate x in
  equal ~msg:"usable before" (array float_exact) [| 2.; 4. |]
    (elements (Nx.add x x));
  let y = Nx.mul d (fresh [| 3.; 4. |]) in
  equal ~msg:"the result" (array float_exact) [| 3.; 8. |] (elements y);
  invalid ~sub:"Nx.add: float32 [2] was donated to Nx.mul" (fun () ->
      Nx.add x x);
  invalid ~sub:"Nx.copy: float32 [2] was donated to Nx.mul" (fun () ->
      Nx.copy d);
  equal ~msg:"its shape answers" (array int) [| 2 |] (Nx.shape x)

let test_passed_on () =
  let x = fresh [| 1.; 2.; 3.; 4. |] in
  let d = Nx.reshape [| 2; 2 |] (Nx.donate x) in
  equal ~msg:"the donor lives until the final consumer" (array float_exact)
    [| 2.; 4.; 6.; 8. |]
    (elements (Nx.add x x));
  let y = Nx.copy d in
  equal ~msg:"the consumer's result" (array float_exact) [| 1.; 2.; 3.; 4. |]
    (elements y);
  invalid ~sub:"Nx.add: float32 [4] was donated to Nx.copy" (fun () ->
      Nx.add x x);
  invalid ~sub:"Nx.copy: float32 [2; 2] was donated to Nx.copy" (fun () ->
      Nx.copy d);
  (* The handle a movement read and the one it passed on are handles of one
     donor: whichever is consumed first, the other raises naming that
     consumer. *)
  let h = Nx.donate (fresh [| 1.; 2. |]) in
  let v = Nx.reshape [| 1; 2 |] h in
  equal ~msg:"the handle read" (array float_exact) [| 1.; 2. |]
    (elements (Nx.copy h));
  invalid ~sub:"Nx.copy: float32 [1; 2] was donated to Nx.copy" (fun () ->
      Nx.copy v);
  let h = Nx.donate (fresh [| 1.; 2. |]) in
  let v = Nx.reshape [| 1; 2 |] h in
  ignore (Nx.neg v);
  invalid ~sub:"Nx.copy: float32 [2] was donated to Nx.neg" (fun () ->
      Nx.copy h);
  (* A slice of every element is the operand itself. *)
  let x = fresh_at [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  equal ~msg:"a slice of every element" (array float_exact)
    [| 1.; 2.; 3.; 4. |]
    (elements (Nx.reshape [| 4 |] (Nx.slice [ A; A ] (Nx.donate x))))

(* A sampler reads its key and kills no donation of it, though its draw reads
   the key in one bitcast: the key lives, and draws the same again. *)
let test_key_lent () =
  let words () = Nx.cast Nx.int32 (fresh [| 1.; 2. |]) in
  let draw f t = elements (Nx.cast f32 (f (Nx.Rng.of_tensor t))) in
  let lent ~msg f =
    let expected = draw f (words ()) in
    let t = words () in
    equal ~msg (array float_exact) expected (draw f (Nx.donate t));
    equal ~msg:(msg ^ ": the key lives") (array float_exact) expected
      (draw f t)
  in
  lent ~msg:"Nx.Rng.permutation" (fun key -> Nx.Rng.permutation ~key 5);
  lent ~msg:"Nx.Rng.uniform" (fun key -> Nx.Rng.uniform ~key f32 [| 3 |])

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
  invalid ~sub:"Nx.add: float32 [2] was donated to Nx.mul" (fun () ->
      Nx.add (Nx.donate x2) d1);
  equal ~msg:"the other donor lives" (array float_exact) [| 10.; 12. |]
    (elements (Nx.add x2 x2));
  (* Refused on its first operand, an add kills no handle of its second,
     which it would broadcast. *)
  let dead = fresh_at [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  ignore (Nx.copy (Nx.donate dead));
  let b = Nx.donate (fresh [| 7.; 8. |]) in
  invalid ~sub:"Nx.add: float32 [2; 2] was donated to Nx.copy" (fun () ->
      Nx.add dead b);
  equal ~msg:"the broadcast operand's handle lives" (array float_exact)
    [| 7.; 8. |] (elements (Nx.copy b));
  (* A function that computes a narrow float at float32 refuses before the
     cast that would consume its operand. *)
  let narrow s = Nx.cast Nx.bfloat16 (fresh_at s (Array.make (s.(0) * s.(1)) 1.)) in
  let x3 = narrow [| 2; 2 |] in
  invalid ~sub:"Nx.matrix_rank: rtol" (fun () ->
      Nx.matrix_rank ~rtol:(-1.) (Nx.donate x3));
  equal ~msg:"matrix_rank's refused donor lives" (array int) [| 2; 2 |]
    (Nx.shape (Nx.copy x3));
  let x4 = narrow [| 2; 3 |] in
  invalid ~sub:"Nx.cond: " (fun () -> Nx.cond f32 ~p:`One (Nx.donate x4));
  equal ~msg:"cond's refused donor lives" (array int) [| 2; 3 |]
    (Nx.shape (Nx.copy x4));
  (* A norm refuses before the cast to its precision, which would consume
     its operand. *)
  List.iter
    (fun (name, s, f) ->
      let x = narrow s in
      invalid ~sub:(name ^ ": ") (fun () -> f (Nx.donate x));
      equal ~msg:(name ^ "'s refused donor lives") (array int) s
        (Nx.shape (Nx.copy x)))
    [
      ("Nx.vector_norm", [| 2; 2 |], fun d ->
          ignore (Nx.vector_norm f32 ~ord:Float.nan d));
      ("Nx.vector_norm", [| 2; 2 |], fun d ->
          ignore (Nx.vector_norm f32 ~axes:[ 2 ] d));
      ("Nx.vector_norm", [| 2; 0 |], fun d ->
          ignore (Nx.vector_norm Nx.float64 ~ord:Float.infinity d));
      ("Nx.matrix_norm", [| 2; 0 |], fun d ->
          ignore (Nx.matrix_norm Nx.float64 ~ord:`Two d));
      ("Nx.matrix_norm", [| 2; 0 |], fun d ->
          ignore (Nx.matrix_norm f32 ~ord:`NegOne d));
      ("Nx.cond", [| 2; 2 |], fun d -> ignore (Nx.cond Nx.float8_e4m3fn d));
      ("Nx.cond", [| 2; 0 |], fun d -> ignore (Nx.cond f32 d));
    ];
  (* A routine that takes its result's dtype first refuses its operand's
     dtype before it consumes the operand. *)
  List.iter
    (fun (name, f) ->
      let x = narrow [| 2; 2 |] in
      invalid ~sub:(name ^ ": ") (fun () -> f (Nx.donate x));
      equal ~msg:(name ^ "'s refused donor lives") (array int) [| 2; 2 |]
        (Nx.shape (Nx.copy x)))
    [
      ("Nx.svd", fun d -> ignore (Nx.svd f32 d));
      ("Nx.eigh", fun d -> ignore (Nx.eigh f32 d));
      ("Nx.eig", fun d -> ignore (Nx.eig Nx.complex64 d));
      ("Nx.slogdet", fun d -> ignore (Nx.slogdet f32 d));
      ("Nx.lstsq", fun d -> ignore (Nx.lstsq f32 d (narrow [| 2; 2 |])));
    ];
  (* A composite that slices its operand into one map, or conjugates it,
     refuses before either. *)
  let x5 = fresh [| 1.; 2.; 3. |] in
  invalid ~sub:"Nx.cross: " (fun () ->
      Nx.cross (Nx.donate x5) (fresh [| 1.; 2. |]));
  equal ~msg:"cross's refused donor lives" (array int) [| 3 |]
    (Nx.shape (Nx.copy x5));
  let z6 = Nx.complex Nx.complex64 ~re:(fresh [| 1.; 2. |]) ~im:(fresh [| 3.; 4. |]) in
  invalid ~sub:"Nx.hfft: " (fun () -> Nx.hfft Nx.float32 ~axes:[ 1 ] (Nx.donate z6));
  equal ~msg:"hfft's refused donor lives" (array int) [| 2 |]
    (Nx.shape (Nx.copy z6));
  let z7 = Nx.complex Nx.complex64 ~re:(fresh [| 1.; 2. |]) ~im:(fresh [| 3.; 4. |]) in
  invalid ~sub:"Nx.vecdot: " (fun () ->
      Nx.vecdot (Nx.donate z7) (Nx.zeros Nx.complex64 [| 3 |]));
  let parts () = fresh_at [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let z8 = Nx.complex Nx.complex64 ~re:(parts ()) ~im:(parts ()) in
  invalid ~sub:"Nx.vecdot: " (fun () ->
      Nx.vecdot (Nx.donate z8) (Nx.zeros Nx.complex64 [| 3; 2 |]));
  equal ~msg:"vecdot's refused donors live" (array int) [| 2; 2 |]
    [| Nx.numel (Nx.copy z7); Nx.ndim (Nx.copy z8) |];
  let x9 = fresh_at [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  invalid ~sub:"Nx.multi_dot: " (fun () ->
      Nx.multi_dot [ Nx.donate x9; fresh_at [| 2; 2 |] [| 1.; 2.; 3.; 4. |]; fresh [| 1.; 2.; 3. |] ]);
  equal ~msg:"multi_dot's refused donor lives" (array int) [| 2; 2 |]
    (Nx.shape (Nx.copy x9));
  (* A value that fits no count refuses before the mask is read; one that
     fits all but the count refuses once it has consumed the mask, and
     leaves the target. *)
  let m10 = Nx.greater_s (fresh [| 1.; 3. |]) 0. and x10 = fresh [| 1.; 3. |] in
  invalid ~sub:"Nx.set: " (fun () ->
      Nx.set [ Nx.M (Nx.donate m10) ] (Nx.zeros f32 [| 2; 2 |]) (Nx.donate x10));
  equal ~msg:"set's refused mask and target live" (array int) [| 2; 2 |]
    [| Nx.numel (Nx.copy m10); Nx.numel (Nx.copy x10) |];
  let m13 = Nx.greater_s (fresh [| 1.; 3. |]) 0. and x13 = fresh [| 1.; 3. |] in
  invalid ~sub:"Nx.set: " (fun () ->
      Nx.set [ Nx.M (Nx.donate m13) ] (Nx.zeros f32 [| 3 |]) (Nx.donate x13));
  invalid ~sub:(dead_in "Nx.copy" m13 "Nx.set") (fun () -> Nx.copy m13);
  equal ~msg:"set's target lives past its count" (array int) [| 2 |]
    (Nx.shape (Nx.copy x13));
  let dead = fresh [| 1.; 3. |] in
  ignore (Nx.copy (Nx.donate dead));
  let m11 = Nx.greater_s (fresh [| 1.; 3. |]) 0. in
  invalid ~sub:"Nx.slice: float32 [2] was donated to Nx.copy" (fun () ->
      Nx.slice [ Nx.M (Nx.donate m11) ] dead);
  equal ~msg:"slice's refused mask lives" (array int) [| 2 |]
    (Nx.shape (Nx.copy m11));
  let c12 = Nx.greater_s (fresh [| 1.; 3. |]) 0. in
  invalid ~sub:"Nx.extract: " (fun () ->
      Nx.extract ~fill:300 (Nx.donate c12) (Nx.zeros Nx.uint8 [| 2 |]));
  equal ~msg:"extract's refused condition lives" (array int) [| 2 |]
    (Nx.shape (Nx.copy c12))

(* An operation that broadcasts a handle, refused by its dtype or by a dead
   operand, kills nothing and names the dead operand. *)
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
  invalid ~sub:"Nx.logical_and: float32 [2; 2] was donated to Nx.copy"
    (fun () -> Nx.logical_and h dead);
  equal ~msg:"the floats' handle reaches a later consumer" (array float_exact)
    [| 1.; 2. |]
    (elements (Nx.copy h))

(* A function of several operations reads its handles as live values over
   their memory, shared, and kills nothing: the donor beside its handle, one
   donation twice, an einsum of three operands, a slice and a set at
   positions held in data. The donor keeps its elements, and its handle
   reaches a later consumer. *)
let test_several_reads () =
  let x = fresh [| 1.; 2. |] in
  let h = Nx.donate x in
  let column v = Nx.reshape [| 2; 1 |] v in
  let solution a b =
    let s, _, _, _ = Nx.lstsq f32 (column a) (column b) in
    elements s
  in
  let expected = solution (fresh [| 1.; 2. |]) (fresh [| 1.; 2. |]) in
  equal ~msg:"the donor beside its handle" (array float_exact) expected
    (solution h x);
  equal ~msg:"one donation twice" (array float_exact) expected (solution h h);
  equal ~msg:"the donor" (array float_exact) [| 1.; 2. |] (elements x);
  equal ~msg:"its handle's consumer" (array float_exact) [| 2.; 4. |]
    (elements (Nx.add h x));
  invalid ~sub:"Nx.copy: float32 [2] was donated to Nx.add" (fun () ->
      Nx.copy x);
  let id = fresh_at [| 2; 2 |] [| 1.; 0.; 0.; 1. |] in
  let a = fresh_at [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let chain = Nx.Pattern.v "i j, j k, k l -> i l | j k" in
  equal ~msg:"the einsum" (array float_exact) [| 7.; 10.; 15.; 22. |]
    (elements (Nx.reshape [| 4 |] (Nx.einsum chain [ Nx.donate a; id; a ])));
  equal ~msg:"the einsum's donor" (array float_exact) [| 1.; 2.; 3.; 4. |]
    (elements (Nx.reshape [| 4 |] a));
  let positions () =
    let p = Nx.create Nx.int64 [| 2 |] [| 1L; 0L |] in
    Nx.add p (Nx.zeros_like p)
  in
  let m = fresh_at [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let p = positions () in
  equal ~msg:"the slice" (array float_exact)
    (elements (Nx.slice [ T (positions ()); T (positions ()) ] m))
    (elements (Nx.slice [ T p; T (Nx.donate p) ] m));
  equal ~msg:"the slice's donor" (array int64) [| 1L; 0L |] (elements p);
  let p = positions () and v = Nx.create Nx.int64 [| 2 |] [| 10L; 20L |] in
  equal ~msg:"the set" (array int64) [| 20L; 10L |]
    (elements (Nx.set [ T (Nx.donate p) ] v p));
  let a = fresh_at [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  equal ~msg:"the cube" (array float_exact) [| 37.; 54.; 81.; 118. |]
    (elements (Nx.reshape [| 4 |] (Nx.matrix_power (Nx.donate a) 3)));
  equal ~msg:"the cube's donor" (array float_exact) [| 1.; 2.; 3.; 4. |]
    (elements (Nx.reshape [| 4 |] a));
  let x = fresh_at [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  invalid ~sub:"Nx.softmax" (fun () -> Nx.softmax ~axes:[ 5 ] (Nx.donate x));
  equal ~msg:"a refused function's donor" (array float_exact)
    [| 1.; 2.; 3.; 4.; 5.; 6. |]
    (elements (Nx.reshape [| 6 |] x))

(* An interpretation reads a handle as its arrays and kills nothing: an
   observer of every operation, which receives the broadcasts too, and a
   tracer beside a traced operand, and two handles of one donor. The
   donor's memory is shared from then on, so that its next donation computes
   into fresh memory, where without an interpretation it writes in place. *)
type ('v, 's, 'd) Nx.Prim.payload += Tag : ('v, 's, 'd) Nx.Prim.payload

let test_lent () =
  let peek x = Nx.Repr.read x A.to_array in
  let in_place x =
    let b = Nx.Repr.read x A.buffer in
    let y = Nx.neg (Nx.donate x) in
    Rig.Buffer.overlaps b (A.buffer (Option.get (Nx.Repr.array y)))
  in
  equal ~msg:"no interpretation: in place" bool true
    (in_place (fresh [| 1.; 2. |]));
  let observed f =
    Nx.Prim.interpret ~name:"test.observer" Extent
      (fun _ ~by op -> Nx.Prim.eval ~by op)
      (fun _ -> f ())
  in
  let x = fresh [| 1.; 2. |] in
  equal ~msg:"the observer's result" (array float_exact) [| 3.; 8. |]
    (elements (observed (fun () -> Nx.mul (Nx.donate x) (fresh [| 3.; 4. |]))));
  equal ~msg:"the observer's donor" (array float_exact) [| 1.; 2. |] (peek x);
  equal ~msg:"the observer's donor: fresh memory" bool false (in_place x);
  let x = fresh [| 1.; 2. |] in
  let wide = fresh_at [| 2; 2 |] [| 10.; 20.; 30.; 40. |] in
  equal ~msg:"the broadcast sum" (array float_exact) [| 11.; 22.; 31.; 42. |]
    (elements
       (Nx.reshape [| 4 |] (observed (fun () -> Nx.add (Nx.donate x) wide))));
  equal ~msg:"the broadcast donor" (array float_exact) [| 1.; 2. |] (peek x);
  equal ~msg:"two donations of one value" (array float_exact) [| 2.; 4. |]
    (elements (observed (fun () -> Nx.add (Nx.donate x) (Nx.donate x))));
  equal ~msg:"their donor" (array float_exact) [| 1.; 2. |] (peek x);
  let x = fresh [| 1.; 2. |] and t = fresh [| 3.; 4. |] in
  Nx.Prim.interpret ~name:"test.tracer" Values
    (fun i ~by op ->
      Nx.Prim.results ~by (fun _ form -> Nx.Prim.traced i form Tag) op)
    (fun i ->
      let t = Nx.Prim.traced i (Nx.Prim.form t) Tag in
      ignore (Nx.mul (Nx.donate x) t));
  equal ~msg:"the tracer's donor" (array float_exact) [| 1.; 2. |] (peek x);
  equal ~msg:"the tracer's donor: fresh memory" bool false (in_place x)

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
             at = None;
           }));
  equal ~msg:"the map's load lives" (array float_exact) [| 1.; 2. |]
    (elements (Nx.copy x))

(* An operation that aligns its operands, as [add] broadcasts [1] to [2],
   checks them all before it consumes any. *)
let test_aligned_refusal () =
  let x2 = fresh [| 1.; 2. |] in
  let d2 = Nx.donate x2 in
  ignore (Nx.mul (Nx.donate x2) (fresh [| 3.; 4. |]));
  let x1 = fresh [| 5. |] in
  invalid ~sub:"Nx.add: float32 [2] was donated to Nx.mul" (fun () ->
      Nx.add (Nx.donate x1) d2);
  equal ~msg:"the other donor lives" (array float_exact) [| 10. |]
    (elements (Nx.add x1 x1))

(* A creation has bytes from the call on, so a donation of one dies as any
   value's does. *)
let test_creation_dies () =
  let c = Nx.zeros f32 [| 2 |] in
  let d = Nx.donate c in
  equal ~msg:"a creation, read twice" (array float_exact) [| 0.; 0. |]
    (elements (Nx.add d d));
  invalid ~sub:"Nx.add: float32 [2] was donated to Nx.add" (fun () ->
      Nx.add c c)

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
        (Nx.Repr.of_array Nx.Host.v
           (A.of_array D.Bool (Iarray.of_array s) xs))
    in
    let y = Nx.logical_not (Nx.donate x) in
    equal ~msg (array bool) (Array.map not xs) (elements y);
    invalid ~sub:(dead_in "Nx.copy" x "Nx.logical_not") (fun () -> Nx.copy x)
  in
  donated [| 1; 1 |] 0;
  donated [| 1; 1 |] 1;
  donated [| 4; 2 |] 0;
  donated [| 4; 2 |] 1

let test_is_handle () =
  let x = fresh [| 1.; 2. |] in
  let d = Nx.donate x in
  equal ~msg:"a handle" bool true (Nx.Prim.donated d);
  equal ~msg:"its donor" bool false (Nx.Prim.donated x);
  equal ~msg:"a handle passed on" bool true
    (Nx.Prim.donated (Nx.reshape [| 1; 2 |] (Nx.donate (fresh [| 1.; 2. |]))));
  equal ~msg:"a donated creation" bool true
    (Nx.Prim.donated (Nx.donate (Nx.zeros f32 [| 2 |])));
  ignore (Nx.copy d);
  equal ~msg:"a consumed handle" bool true (Nx.Prim.donated d)

(* A call may give back what it read of a donor: an array outside the
   vector, which Ptree.scale and axpy keep, or one Ptree.cast keeps at its
   dtype. Given the donor beside its handle, or beside one a movement passed
   on, it gives two values over one memory, and a donation of one leaves the
   other's elements as they were. *)
module Both = struct
  type ('a, _) t = 'a * 'a

  let walk c (w, v) =
    let open Nx.Ptree.Walk in
    let w = field c "w" leaf w in
    let v = field c "v" leaf v in
    (w, v)
end

let test_given_back_twice () =
  let ints s =
    let x = Nx.create Nx.int32 s [| 1l; 2l |] in
    Nx.add x (Nx.zeros_like x)
  in
  let two = Nx.Ptree.(pair tensor tensor) and k = Nx.scalar f32 2. in
  let apart ~msg (w, v) =
    ignore (Nx.neg (Nx.donate w));
    equal ~msg (array int32) [| 1l; 2l |] (elements v)
  in
  let x = ints [| 2 |] in
  apart ~msg:"Ptree.scale" (Nx.Ptree.scale two k (Nx.donate x, x));
  let x = ints [| 2 |] in
  apart ~msg:"Ptree.axpy"
    (Nx.Ptree.axpy two k (ints [| 2 |], ints [| 2 |]) (Nx.donate x, x));
  let x = ints [| 2 |] in
  apart ~msg:"Ptree.cast"
    (Nx.Ptree.cast (module Both) Nx.int32 (Nx.donate x, x));
  let x = ints [| 1; 2 |] in
  let w, v = Nx.Ptree.scale two k (Nx.reshape [| 2 |] (Nx.donate x), x) in
  apart ~msg:"Ptree.scale, a handle passed on" (w, Nx.reshape [| 2 |] v)

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

(* A donated value on one device is written in place beside a value whole
   on each device of its set, which that device reads by a view of its
   window, and beside one placed on the same device by a call of its own:
   the result lies where the donor does, over its memory. *)
module Pair = (val Nx.devices [ device; Nx_support.memory 0 ])

let test_beside_in_place () =
  let n = 1 lsl 16 in
  let host v = Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 [| n |] v) in
  let alone () = Nx.Placement.device Pair.v device in
  let donor () =
    let a = Nx.place (alone ()) (host (Array.make n 1.)) in
    Nx.add a (Nx.zeros_like a)
  in
  let beside ~msg p =
    let y, grew =
      grown
        (fun () -> (donor (), Nx.place (p ()) (host (Array.make n 2.))))
        (fun (x, b) -> Nx.add (Nx.donate x) b)
    in
    in_place ~msg ~bytes:(4 * n) grew;
    equal ~msg (array float_exact) (Array.make n 3.) (elements y)
  in
  beside ~msg:"whole on each device" (fun () -> Pair.on);
  beside ~msg:"on its device" alone

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

(* A handle a movement passes on as a view reads the donor's memory at the
   result's own index, C-contiguous: its consumer writes there. *)
let test_view_in_place () =
  let y, grew =
    grown cache (fun c -> Nx.neg (Nx.reshape [| rows * width |] (Nx.donate c)))
  in
  in_place ~msg:"through a reshape" ~bytes:size grew;
  equal (array float_exact) (Array.map Float.neg original) (elements y)

(* where writes into its donated second or third operand; a donated condition,
   of another dtype than the result, dies naming it. *)
let test_where_in_place () =
  let n = rows * width in
  let even () =
    Nx.place Mem.on (Nx.init Nx.bool [| n |] (fun i -> i.(0) mod 2 = 0))
  in
  let ones () = on_mem [| n |] (Array.make n 1.) in
  let twos () = on_mem [| n |] (Array.make n 2.) in
  let chosen = Array.init n (fun i -> if i mod 2 = 0 then 1. else 2.) in
  let donated ~msg f =
    let y, grew = grown (fun () -> (even (), ones (), twos ())) f in
    in_place ~msg ~bytes:(4 * n) grew;
    equal ~msg (array float_exact) chosen (elements y)
  in
  donated ~msg:"the second" (fun (c, a, b) -> Nx.where c (Nx.donate a) b);
  donated ~msg:"the third" (fun (c, a, b) -> Nx.where c a (Nx.donate b));
  let c = even () in
  equal ~msg:"a donated condition" (array float_exact) chosen
    (elements (Nx.where (Nx.donate c) (ones ()) (twos ())));
  invalid ~sub:"donated to Nx.where" (fun () -> Nx.copy c)

(* A value over every other element of an array its caller gives up owns
   that memory, which is not C-contiguous: its donation computes into fresh
   memory. *)
let test_strided_copied () =
  let n = 1 lsl 16 in
  let make () =
    let a =
      A.of_array D.Float32
        [| 2 * n |]
        (Array.init (2 * n) (fun i -> Float.of_int (i mod 2)))
    in
    let odd = A.move (Slice [| { start = 1; count = n; step = 2 } |]) a in
    Nx.Repr.unsafe_of_array Nx.Host.v (Option.get odd)
  in
  let y, grew = grown_on Rig.host make (fun x -> Nx.neg (Nx.donate x)) in
  copied ~msg:"every other element" ~bytes:(4 * n) grew;
  equal (array float_exact) (Array.make n (-1.)) (elements y)

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

(* A function of several operations reads its handles as shared memory: a
   consumer of its donor inside it, here a rule at [softmax]'s first
   operation, computes into fresh memory and leaves the elements [softmax]
   reads as they were. *)
let test_inside_composite () =
  let expected = elements (Nx.softmax ~axes:[ 1 ] (cache ())) in
  let c = cache () and taken = ref None in
  let rule _ ~by op =
    if Option.is_none !taken then
      taken :=
        Some
          (match Nx.add (Nx.donate c) (nines [| width |]) with
          | _ -> "consumed"
          | exception Invalid_argument m -> m);
    Nx.Prim.eval ~by op
  in
  let y =
    Nx.Prim.interpret ~name:"test.taker" Extent rule (fun _ ->
        Nx.softmax ~axes:[ 1 ] (Nx.donate c))
  in
  equal ~msg:"the consumer" (option string) (Some "consumed") !taken;
  equal ~msg:"the softmax" (array float_exact) expected (elements y)

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

(* An operation whose kernels decline once its checks pass raises with its
   donors dead and returns their memory, as a consumer that raises does, the
   memory it took for its result included: on a set whose kernels decline
   [Add], a sum that may write into its handle's memory, and one beside a
   wider operand, which may not. The operands stay reachable, so no
   collection explains the bytes. *)
module Dec = (val Nx.devices ~kernels:(module Nx_support.Declining) [ device ])

(* [xs] of shape [s] on [Dec] in memory of its own: a copy, since the set's
   kernels decline the sum [on_mem] computes. *)
let on_dec s xs =
  Nx.copy
    (Nx.place Dec.on (Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 s xs)))

let test_declined_frees () =
  let n = 1 lsl 14 in
  let freed ~msg y =
    let x = on_dec [| n |] (Array.make n 1.) in
    let before = settled () in
    invalid ~sub:"does not compute Add" (fun () -> Nx.add (Nx.donate x) y);
    let freed = before - settled () in
    ignore (Sys.opaque_identity y);
    invalid ~sub:(dead_in "Nx.copy" x "Nx.add") (fun () -> Nx.copy x);
    at_least ~msg int ~than:(4 * n) freed
  in
  freed ~msg:"a sum that may write in place" (on_dec [| n |] (Array.make n 2.));
  freed ~msg:"a sum beside a wider operand"
    (on_dec [| 2; n |] (Array.make (2 * n) 2.))

(* A Values rule may keep the arrays it receives, as a tape keeps a
   derivative's primals. The donor lives, and its next donation computes into
   fresh memory, leaving the kept arrays' elements as they were. *)
type ('v, 's, 'd) Nx.Prim.payload +=
  | Held : ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.Prim.payload

let test_kept_by_rule () =
  let kept = ref [] in
  let rule i ~by op =
    let held (type v s d) (x : (v, s, d) Nx.t) : (v, s, d) Nx.t =
      match Nx.Prim.payload i x with Some (Held y) -> y | _ -> x
    in
    let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
    List.iter
      (fun (Nx.Prim.Any x) ->
        if Nx.Prim.owner x = None then
          kept := (fun () -> elements (Nx.cast f32 x)) :: !kept)
      xs;
    Nx.Prim.eval ~by (Nx.Prim.map held op)
  in
  let c = cache () and zeros = cache () in
  let y =
    Nx.Prim.interpret ~name:"test.tape" Values rule (fun i ->
        let t = Nx.Prim.traced i (Nx.Prim.form zeros) (Held zeros) in
        Nx.sub (Nx.donate c) t)
  in
  equal ~msg:"the difference" (array float_exact) (Array.make (rows * width) 0.)
    (elements y);
  equal ~msg:"the arrays the rule received" int 1 (List.length !kept);
  let negated = Nx.neg (Nx.donate c) in
  equal ~msg:"the donor's next consumer" (array float_exact)
    (Array.map Float.neg original) (elements negated);
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

(* Two handles of one donor in one operation, the same handle twice or two
   donations of one value, give what the operation gives two undonated copies,
   and kill the donor and the handles naming it: a product, which may write
   into the memory, a sum beside the transpose, which reads it in another
   layout, a matrix product, which reads it at other indices than the
   result's own, and a scatter of a matrix into itself. *)
let square =
  Gen.with_pp
    (fun ppf (n, xs) ->
      Format.fprintf ppf "%dx%d [%s]" n n
        (String.concat "; " (Array.to_list (Array.map Float.to_string xs))))
    (let open Gen in
     let* n = int_range 0 4 in
     let+ xs = array ~size:(constant (n * n)) (of_list specials) in
     (n, xs))

let law_one_donor_twice =
  prop "two handles of one donor give the undonated result" square
    (fun (n, xs) ->
      cover "no element" (n = 0);
      cover "one element" (n = 1);
      cover "a matrix" (n >= 2);
      let s = [| n; n |] in
      let rows = Array.init (n * n) (fun i -> Int64.of_int (n - 1 - (i / n))) in
      let p = Nx.create Nx.int64 s rows in
      let ops =
        [
          ("Nx.mul", fun a b -> Nx.mul a b);
          ("Nx.add", fun a b -> Nx.add a (Nx.transpose b));
          ("Nx.matmul", fun a b -> Nx.matmul a b);
          ("Nx.scatter", fun a b -> Nx.scatter ~axis:0 p a b);
        ]
      in
      List.iter
        (fun (by, f) ->
          let dies y =
            invalid ~sub:(dead_in "Nx.copy" y by) (fun () -> Nx.copy y)
          in
          let kept = f (fresh_at s xs) (fresh_at s xs) in
          let x = fresh_at s xs in
          let h = Nx.donate x in
          equal
            ~msg:(by ^ ", one handle twice")
            (array int32) (bits32 kept)
            (bits32 (f h h));
          List.iter dies [ x; h ];
          let x = fresh_at s xs in
          let h = Nx.donate x and h' = Nx.donate x in
          equal ~msg:(by ^ ", two donations") (array int32) (bits32 kept)
            (bits32 (f h h'));
          List.iter dies [ x; h; h' ])
        ops)

(* Two handles of one donor, each through a chain of movements, or a handle
   through one beside its donor, give in one operation what the operation
   gives undonated copies through the same movements. The donor dies once,
   naming the operation, and so does every handle the chains passed on. A
   recut, a reshape of a transpose, copies; the others make views. *)
type movement = Transpose | Flip | Flatten | Recut | Whole

let movement_name = function
  | Transpose -> "transpose"
  | Flip -> "flip"
  | Flatten -> "flatten"
  | Recut -> "recut"
  | Whole -> "whole"

let move n x = function
  | Transpose -> Nx.transpose x
  | Flip -> Nx.flip x
  | Flatten -> Nx.reshape [| n; n |] (Nx.reshape [| n * n |] x)
  | Recut -> Nx.reshape [| n; n |] (Nx.reshape [| n * n |] (Nx.transpose x))
  | Whole -> Nx.slice [ A; A ] x

let chain n x ms = List.fold_left (move n) x ms

let movements =
  Gen.list ~size:(Gen.int_range 0 3)
    (Gen.of_list
       ~pp:(fun ppf m -> Format.pp_print_string ppf (movement_name m))
       [ Transpose; Flip; Flatten; Recut; Whole ])

let moved_case =
  let names ms = String.concat "; " (List.map movement_name ms) in
  Gen.with_pp
    (fun ppf ((n, xs), m1, m2, beside) ->
      Format.fprintf ppf "%dx%d [%s] through [%s], %s" n n
        (String.concat "; " (Array.to_list (Array.map Float.to_string xs)))
        (names m1)
        (if beside then "beside the donor"
         else Printf.sprintf "beside [%s]" (names m2)))
    Gen.(quad square movements movements bool)

let law_moved_twice =
  prop "two handles of one donor through movements give the undonated result"
    moved_case (fun ((n, xs), m1, m2, beside) ->
      let copies m = n >= 2 && List.mem Recut m in
      cover "a copy passed on" (copies m1 || ((not beside) && copies m2));
      cover "the donor beside a moved handle" (beside && m1 <> []);
      cover "two moved handles" ((not beside) && m1 <> [] && m2 <> []);
      let s = [| n; n |] in
      let second x h = if beside then x else chain n h m2 in
      List.iter
        (fun (by, f) ->
          let copy () = fresh_at s xs in
          let kept = f (chain n (copy ()) m1) (second (copy ()) (copy ())) in
          let x = copy () in
          let h = Nx.donate x in
          let a = chain n h m1 and b = second x h in
          equal ~msg:by (array int32) (bits32 kept) (bits32 (f a b));
          let dies y =
            invalid ~sub:(dead_in "Nx.copy" y by) (fun () -> Nx.copy y)
          in
          List.iter dies [ x; h; a; b ])
        [
          ("Nx.mul", fun a b -> Nx.mul a b);
          ("Nx.add", fun a b -> Nx.add a b);
          ("Nx.matmul", fun a b -> Nx.matmul a b);
        ])

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

(* On two domains: reading a value and consuming it in an addition or in a
   function of several operations. Each run is explained by some order: a read
   before the consumer gives the elements, after it raises; one consumer of a
   value wins and any other raises. *)
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

(* A function of several operations reading its handle, the number of
   distinct keys: it kills nothing, and reads the elements as they were or
   raises where a consumer overlaps it. *)
let composite_ref m =
  if m.dead then Error ()
  else
    let distinct = List.sort_uniq compare (Array.to_list m.xs) in
    Ok [| Float.of_int (List.length distinct) |]

let composite_sys x =
  match elements (Nx.cast f32 (Nx.unique (Nx.donate x)).count) with
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
    command "composite" (value ^-> returns outcome) composite_ref composite_sys;
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

(* What a refusal names: the function the handle went to. *)
let taker msg =
  let has s =
    let n = String.length s and m = String.length msg in
    let rec go i = i + n <= m && (String.sub msg i n = s || go (i + 1)) in
    go 0
  in
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

(* Whether [h] may have gone to [e]: taken by it, or free and taken by it in
   flight. A consumer of two handles claims them one at a time and keeps
   those it took when it loses another, so a free handle may then have gone
   to one in flight, and is dead from then on. *)
let lost_to e h =
  List.mem (Some e) h.states || ((e = "Nx.sub" || e = "test.call") && free h)

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

(* A movement reads a free handle and kills nothing. Its result is a handle no
   command reads. *)
let reshape_ref h = function
  | Ok (Ok _) ->
      if not (free h) then fail "Nx.reshape read a taken handle";
      h.states <- [ None ]
  | Ok (Error e) ->
      if not (lost_to e h) then fail ("Nx.reshape raised naming " ^ e);
      h.states <- [ Some e ]
  | Error exn -> raise exn

let reshape_sys h =
  refused (fun () ->
      ignore (Nx.reshape [| 1; (Nx.shape h).(0) |] h);
      [||])

(* [sub h1 h2] checks both handles, then takes them in an order the promise
   leaves open: a loser lost one handle, and may have taken the other if it was
   free. [sub h h] reads one handle. The result's elements are [f]'s. *)
let pair_ref f by h1 h2 out =
  if h1 == h2 then judge1 by (Array.map (fun v -> f v v)) h1 out
  else
    match out with
    | Ok (Ok ys) ->
        if not (free h1 && free h2) then fail (by ^ " took a taken handle");
        cover "two handles consumed" true;
        equal (array float_exact) (Array.map2 f h1.ys h2.ys) ys;
        h1.states <- [ Some by ];
        h2.states <- [ Some by ]
    | Ok (Error e) ->
        let lost = lost_to e in
        let taken h = if free h then union h.states [ Some by ] else h.states in
        if not (lost h1 || lost h2) then fail (by ^ " raised naming " ^ e);
        cover "one handle refused, the other free"
          ((lost h1 && free h2) || (lost h2 && free h1));
        let s1 = if lost h1 then [ Some e ] else [] in
        let s1 = if lost h2 then union s1 (taken h1) else s1 in
        let s2 = if lost h2 then [ Some e ] else [] in
        let s2 = if lost h1 then union s2 (taken h2) else s2 in
        h1.states <- s1;
        h2.states <- s2
    | Error exn -> raise exn

let sub2_ref = pair_ref ( -. ) "Nx.sub"
let sub2_sys h1 h2 = refused (fun () -> elements (Nx.sub h1 h2))

let pair =
  Gen.with_pp
    (fun ppf a ->
      Format.fprintf ppf "[%s]"
        (String.concat "; " (Array.to_list (Array.map string_of_float a))))
    (Gen.array ~size:(Gen.int_range 2 2) (Gen.of_list [ 0.; 1.; -2.; 3.5 ]))

(* A consumer of several operations, [h1 - h2] read from its operands'
   arrays. *)
let host_floats a = A.to_array (A.expect D.Float32 a)

let call_ref = pair_ref ( -. ) "test.call"

(* It writes [h1 - h2] over [h1]'s memory where it may take it, reading
   each element of [h1] through the taken array before writing it. *)
let difference h1 h2 =
  Nx.Prim.Consumer.with_ ~by:"test.call" [ Any h1; Any h2 ] (fun c ->
      let read i = host_floats (Nx.Prim.Consumer.arrays c i).(0) in
      let b = read 1 in
      let n = Array.length b in
      match
        Nx.Prim.Consumer.take c 0 D.Float32 (A.Layout.contiguous [| n |])
      with
      | Some a ->
          for k = 0 to n - 1 do
            A.set a [| k |] (A.get a [| k |] -. b.(k))
          done;
          A.to_array a
      | None -> Array.map2 ( -. ) (read 0) b)

let call_sys h1 h2 = refused (fun () -> difference h1 h2)

let handles =
  [
    command "make" (pair @-> makes handle) make_handle make_handle_sys;
    command "call" (handle ^-> handle ^-> judges taken) call_ref call_sys;
    command "add" (handle ^-> judges taken) add_ref add_sys;
    command "mul" (handle ^-> judges taken) mul_ref mul_sys;
    command "reshape" (handle ^-> judges taken) reshape_ref reshape_sys;
    command "sub2" (handle ^-> handle ^-> judges taken) sub2_ref sub2_sys;
  ]

(* A consumer of two handles alone, so that two domains often take one pair in
   opposite orders. *)
let crossed =
  [
    command "make" (pair @-> makes handle) make_handle make_handle_sys;
    command "sub2" (handle ^-> handle ^-> judges taken) sub2_ref sub2_sys;
    command "call" (handle ^-> handle ^-> judges taken) call_ref call_sys;
  ]

(* On one domain: donors, their handles and values of their own, across
   calls. A donor's fate is whether it lives or the function it died to. A
   value is a donor's own or a handle of it, with its shape and elements: a
   movement of a handle is a handle of the same donor, and a movement of a
   donor's own value is a view, a value of its own. An operation and a
   consumer check every operand before they kill anything, then kill the
   donor of each handle they read, once. A read and a function of several
   operations kill nothing. Every result is computed from the elements, and
   every refusal names the function the donor died to. *)
type donor = { mutable fate : string option }
type held = { of_ : donor; handle : bool; dims : int array; xs : float array }

let held =
  abstract "v" ~invariant:(fun r s ->
      equal ~msg:"donated tells a handle" bool r.handle (Nx.Prim.donated s);
      equal ~msg:"its shape answers" (array int) r.dims (Nx.shape s))

(* [by]'s refusal of [r], whose donor died to [f]. *)
let refusal by r f =
  Printf.sprintf "%s: float32 [%s] was donated to %s" by
    (String.concat "; " (List.map string_of_int (Array.to_list r.dims)))
    f

let told f =
  match f () with ys -> Ok ys | exception Invalid_argument m -> Error m

let made xs = { of_ = { fate = None }; handle = false; dims = [| 2 |]; xs }

(* [Ok (f ())] where [r] lives, else [by]'s refusal. *)
let living by r f =
  match r.of_.fate with Some f' -> Error (refusal by r f') | None -> Ok (f ())

(* [f ()], a value [by] makes of [r], where [r] lives; raises as [by] does
   otherwise. *)
let making by r f =
  match r.of_.fate with
  | Some f' -> invalid_arg (refusal by r f')
  | None -> f ()

let donate_r r = making "Nx.donate" r (fun () -> { r with handle = true })

let moved by r dims xs =
  making by r (fun () ->
      if r.handle then { r with dims; xs }
      else { of_ = { fate = None }; handle = false; dims; xs })

let toggled dims = if Array.length dims = 1 then [| 1; 2 |] else [| 2 |]
let reshape_r r = moved "Nx.reshape" r (toggled r.dims) r.xs
let reshape_s x = Nx.reshape (toggled (Nx.shape x)) x
let reversed xs = Array.of_list (List.rev (Array.to_list xs))
let flip_r r = moved "Nx.flip" r r.dims (reversed r.xs)

(* [r]'s donor dead to [by], where [r] is a handle. *)
let killed by r = if r.handle then r.of_.fate <- Some by

let neg_r r =
  living "Nx.neg" r (fun () ->
      killed "Nx.neg" r;
      Array.map Float.neg r.xs)

let neg_s x = told (fun () -> elements (Nx.neg x))

(* A function [by] of two operands whose result's elements are [f]'s. It
   refuses naming a dead operand, killing nothing, or kills the donor of each
   handle it read. Which dead operand it names is left open. *)
let pair_judged by f a b = function
  | Ok (Ok ys) ->
      if Option.is_some a.of_.fate || Option.is_some b.of_.fate then
        fail (by ^ " read a dead operand");
      equal (array float_exact) (Array.map2 f a.xs b.xs) ys;
      killed by a;
      killed by b
  | Ok (Error m) ->
      let names r =
        match r.of_.fate with Some f' -> refusal by r f' = m | None -> false
      in
      cover "an operand refused, the call killing nothing" true;
      if not (names a || names b) then fail (by ^ " refused: " ^ m)
  | Error exn -> raise exn

let add_s x y = told (fun () -> elements (Nx.add x y))
let call_s x y = told (fun () -> difference x y)
let read_r r = living "Nx.to_array" r (fun () -> r.xs)
let read_s x = told (fun () -> Nx.to_array x)
let split_r r = living "Nx.split" r (fun () -> r.xs)

let split_s x =
  told (fun () ->
      Array.concat (List.map elements (Nx.split ~axis:(-1) 2 x)))

let lives =
  [
    command "make" (pair @-> makes held) made fresh;
    command "donate" (held ^-> makes held) donate_r Nx.donate;
    command "reshape" (held ^-> makes held) reshape_r reshape_s;
    command "flip" (held ^-> makes held) flip_r (fun x -> Nx.flip x);
    command "neg" (held ^-> returns taken) neg_r neg_s;
    command "add"
      (held ^-> held ^-> judges taken)
      (pair_judged "Nx.add" ( +. ))
      add_s;
    command "call"
      (held ^-> held ^-> judges taken)
      (pair_judged "test.call" ( -. ))
      call_s;
    command "read" (held ^-> returns taken) read_r read_s;
    command "split" (held ^-> returns taken) split_r split_s;
  ]

(* Naming: a function of one operation consumes a donated operand for the
   user, so the donor and the handle the user holds die naming the function
   the user called, whatever functions that one computes through. *)
let test_dies_at_the_call () =
  let square () = fresh_at [| 2; 2 |] [| 2.; 0.; 0.; 2. |] in
  let dies name f =
    let x = square () in
    let d = Nx.donate x in
    ignore (f d);
    let sub = dead_in "Nx.copy" x name in
    raises_match ~msg:(name ^ "'s donor") (Exn.invalid_arg ~substring:sub)
      (fun () -> Nx.copy x);
    raises_match ~msg:(name ^ "'s handle") (Exn.invalid_arg ~substring:sub)
      (fun () -> Nx.copy d)
  in
  dies "Nx.trace" (fun d -> Nx.trace d);
  dies "Nx.qr" (fun d -> fst (Nx.qr d));
  dies "Nx.matmul" (fun d -> Nx.matmul d (square ()))

(* A call that keeps none of its operand still raises for a dead one: an
   fft cut to no point, patches of an empty axis padded. *)
let test_none_kept () =
  let dead x =
    ignore (Nx.copy (Nx.donate x));
    x
  in
  let parts = [| 1.; 2.; 3.; 4. |] in
  let z = dead (Nx.complex Nx.complex64 ~re:(fresh parts) ~im:(fresh parts)) in
  invalid ~sub:(dead_in "Nx.fft" z "Nx.copy") (fun () -> Nx.fft ~s:[ 0 ] z);
  let x = dead (fresh_at [| 1; 0 |] [||]) in
  invalid ~sub:(dead_in "Nx.extract_patches" x "Nx.copy") (fun () ->
      Nx.extract_patches ~kernel_size:[| 1 |] ~stride:[| 1 |] ~dilation:[| 1 |]
        ~padding:[| (1, 1) |] x)

(* Where a call's operands reach different operations, a donor beside its
   own handle raises naming the call once the handle's operation consumed
   it: here [q]'s sum consumes it before [k]'s comparison reads it. *)
let test_beside_elsewhere () =
  let p = Nx.cast Nx.int64 (fresh [| 0.; 1.; 2. |]) in
  let len = Nx.cast Nx.int64 (fresh [| 3.; 3.; 3. |]) in
  let pair = { Nx.q = Nx.donate p; k = p; len; tq = 2 } in
  invalid ~sub:(dead_in "Nx.causal" p "Nx.causal") (fun () -> Nx.causal pair)

(* Every public function that takes a value, called by one table of cases. A
   case makes fresh operands, passes its operand [i] as [d i x], and prints the
   results. Two promises run over the table: on a dead operand the function
   raises naming itself, the one the user called, whatever it computes through;
   a donated operand gives the undonated result, and where each operand reaches
   one operation, through movements, whatever else the call computes from its
   result, the call consumes it: the donor and its handle die naming the
   call. *)

(* What a function does with a dead operand: raise naming itself, or print
   [death]. *)
type dead = Raises | Prints of string

type case = {
  fn : string;  (* The function, as the user writes it. *)
  by : string;  (* The name its refusals give. *)
  dt : string;
  operands : int;
  dead : dead;
  consumes : bool;  (* Each operand reaches one operation. *)
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
let c ?by ?(dead = Raises) ?(consumes = false) fn operands run dt =
  let by = "Nx." ^ Option.value by ~default:fn in
  { fn = "Nx." ^ fn; by; dt; operands; dead; consumes; run }

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
            (Iarray.length (Nx.Repr.shards (d 0 (x [| 2; 3 |])))));
      c "reshape" 1 (fun d -> show (Nx.reshape [| 3; 2 |] (d 0 (x [| 2; 3 |]))));
      c "broadcast_to" 1 (fun d ->
          show (Nx.broadcast_to [| 2; 2; 3 |] (d 0 (x [| 2; 3 |]))));
      c "broadcast_arrays" 2 (fun d ->
          showl (Nx.broadcast_arrays [ d 0 (x [| 2; 1 |]); d 1 (x [| 3 |]) ]));
      c "broadcast_arrays" 2 (fun d ->
          showl (Nx.broadcast_arrays [ d 0 (x [| 2; 3 |]); d 1 (x [| 3 |]) ]));
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
      c ~consumes:true "roll" 1 (fun d -> show (Nx.roll 1 (d 0 (x [| 5 |]))));
      c ~consumes:true "roll" 1 (fun d ->
          show (Nx.roll ~axis:1 2 (d 0 (x [| 2; 3 |]))));
      c ~consumes:true "fftshift" 1 (fun d ->
          show (Nx.fftshift ~axes:[ 0; 1 ] (d 0 (x [| 2; 3 |]))));
      c "tile" 1 (fun d -> show (Nx.tile [| 2; 1 |] (d 0 (x [| 2; 3 |]))));
      c "tile" 1 (fun d -> show (Nx.tile [| 1; 1 |] (d 0 (x [| 2; 3 |]))));
      c "repeat" 1 (fun d -> show (Nx.repeat ~axis:0 2 (d 0 (x [| 2; 3 |]))));
      c "repeat" 1 (fun d -> show (Nx.repeat ~axis:0 1 (d 0 (x [| 2; 3 |]))));
      c "rearrange" 1 (fun d ->
          show (Nx.rearrange (Nx.Pattern.v "a b -> b a") (d 0 (x [| 2; 3 |]))));
      c "rearrange" 1 (fun d ->
          show (Nx.rearrange (Nx.Pattern.v "a b -> a b") (d 0 (x [| 2; 3 |]))));
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
      c "slice" 1 (fun d ->
          show (Nx.slice [ Nx.A; Nx.A ] (d 0 (x [| 2; 3 |]))));
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
      c ~consumes:true "square" 1 (fun d ->
          show (Nx.square (d 0 (x [| 2; 3 |]))));
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
      c ~consumes:true "array_equal" 2 (fun d ->
          show (Nx.array_equal (d 0 (x [| 3 |])) (d 1 (x [| 3 |]))));
      c "where" 3 (fun d ->
          let cond = d 0 (Nx.not_equal_s (x [| 2; 3 |]) one) in
          show
            (Nx.where cond (d 1 (x [| 2; 3 |])) (d 2 (Nx.flip (x [| 2; 3 |])))));
      c "sum" 1 (fun d -> show (Nx.sum (d 0 (x [| 2; 3 |]))));
      c "sum" 1 (fun d ->
          show (Nx.sum ~axes:[ 1 ] ~keepdims:true (d 0 (x [| 2; 3 |]))));
      c "prod" 1 (fun d -> show (Nx.prod ~axes:[ 0 ] (d 0 (x [| 2; 3 |]))));
      c ~consumes:true "mean" 1 (fun d -> show (Nx.mean (d 0 (x [| 2; 3 |]))));
      c ~consumes:true "mean" 1 (fun d ->
          show (Nx.mean ~axes:[ 0 ] ~keepdims:true (d 0 (x [| 2; 3 |]))));
      c "cumsum" 1 (fun d -> show (Nx.cumsum (d 0 (x [| 2; 3 |]))));
      c "cumsum" 1 (fun d -> show (Nx.cumsum ~axis:1 (d 0 (x [| 2; 3 |]))));
      c "cumprod" 1 (fun d -> show (Nx.cumprod ~axis:0 (d 0 (x [| 2; 3 |]))));
      c "cast" 1 (fun d -> show (Nx.cast Nx.complex128 (d 0 (x [| 3 |]))));
      c "cast" 1 (fun d -> show (Nx.cast dt (d 0 (x [| 3 |]))));
      c "bitcast" 1 (fun d -> show (Nx.bitcast Nx.uint8 (d 0 (x [| 2 |]))));
      c "bitcast" 1 (fun d -> show (Nx.bitcast dt (d 0 (x [| 2 |]))));
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
      c ~consumes:true "vdot" 2 (fun d ->
          show (Nx.vdot (d 0 (x [| 3 |])) (d 1 (x [| 3 |]))));
      c ~consumes:true "cross" 2 (fun d ->
          show (Nx.cross (d 0 (x [| 2; 3 |])) (d 1 (x [| 3 |]))));
      c "matrix_transpose" 1 (fun d ->
          show (Nx.matrix_transpose (d 0 (x [| 2; 3 |]))));
      c "matrix_transpose" 1 (fun d ->
          show (Nx.matrix_transpose (d 0 (x [| 3 |]))));
      c "conjugate" 1 (fun d -> show (Nx.conjugate (d 0 (x [| 3 |]))));
      c "tril" 1 (fun d -> show (Nx.tril (d 0 (x [| 3; 3 |]))));
      c "triu" 1 (fun d -> show (Nx.triu ~k:1 (d 0 (x [| 3; 3 |]))));
      c "diagonal" 1 (fun d -> show (Nx.diagonal (d 0 (x [| 3; 3 |]))));
      c "trace" 1 (fun d -> show (Nx.trace (d 0 (x [| 3; 3 |]))));
      c "diag" 1 (fun d -> show (Nx.diag (d 0 (x [| 3 |]))));
      c "diag" 1 (fun d -> show (Nx.diag (d 0 (x [| 3; 3 |]))));
      c ~consumes:true "tensordot" 2 (fun d ->
          show
            (Nx.tensordot ~axes:([ 1 ], [ 0 ])
               (d 0 (x [| 2; 3 |]))
               (d 1 (x [| 3; 2 |]))));
      c ~consumes:true "tensordot" 2 (fun d ->
          show (Nx.tensordot (d 0 (x [| 2; 3 |])) (d 1 (x [| 2; 3 |]))));
      c ~consumes:true "inner" 2 (fun d ->
          show (Nx.inner (d 0 (x [| 2; 3 |])) (d 1 (x [| 3 |]))));
      c ~consumes:true "outer" 2 (fun d ->
          show (Nx.outer (d 0 (x [| 2 |])) (d 1 (x [| 3 |]))));
      c ~consumes:true "vecdot" 2 (fun d ->
          show (Nx.vecdot (d 0 (x [| 2; 3 |])) (d 1 (x [| 3 |]))));
      c ~consumes:true "kron" 2 (fun d ->
          show (Nx.kron (d 0 (x [| 2; 1 |])) (d 1 (x [| 3 |]))));
      c ~consumes:true "multi_dot" 2 (fun d ->
          show (Nx.multi_dot [ d 0 (x [| 3 |]); d 1 (x [| 3; 2 |]) ]));
      c ~consumes:true "multi_dot" 3 (fun d ->
          show
            (Nx.multi_dot
               [ d 0 (x [| 2; 3 |]); d 1 (x [| 3; 1 |]); d 2 (x [| 1; 2 |]) ]));
      c "matrix_power" 1 (fun d -> show (Nx.matrix_power (d 0 (x [| 2; 2 |])) 0));
      c ~consumes:true "matrix_power" 1 (fun d ->
          show (Nx.matrix_power (d 0 (x [| 2; 2 |])) 1));
      c ~consumes:true "matrix_power" 1 (fun d ->
          show (Nx.matrix_power (d 0 (x [| 2; 2 |])) 2));
      c "matrix_power" 1 (fun d ->
          show (Nx.matrix_power (d 0 (x [| 1; 2; 2 |])) 3));
      c "meshgrid" 2 (fun d ->
          showl (Nx.meshgrid ~indexing:`Xy [ d 0 (x [| 2 |]); d 1 (x [| 3 |]) ]));
      c ~consumes:true "slice" 2 (fun d ->
          let mask = d 0 (Nx.not_equal_s (x [| 2; 3 |]) one) in
          show (Nx.slice [ Nx.M mask ] (d 1 (x [| 2; 3 |]))));
      c ~consumes:true "set" 3 (fun d ->
          let mask = d 0 (Nx.not_equal_s (x [| 3 |]) one) in
          show (Nx.set [ Nx.A; Nx.M mask ] (d 1 (x [| 2; 1 |])) (d 2 (x [| 2; 3 |]))));
      c ~consumes:true "extract" 2 (fun d ->
          show (Nx.extract (d 0 (x [| 2; 3 |])) (d 1 (x [| 6 |]))));
      c ~consumes:true "extract" 2 (fun d ->
          show (Nx.extract ~size:4 ~fill:one (d 0 (x [| 2; 3 |])) (d 1 (x [| 6 |]))));
      c ~consumes:true "nonzero" 1 (fun d ->
          showl (Nx.nonzero (d 0 (x [| 2; 3 |]))));
      c ~consumes:true "argwhere" 1 (fun d ->
          show (Nx.argwhere (d 0 (x [| 2; 3 |]))));
      c "( *@ )" ~by:"matmul" 2 (fun d ->
          show Nx.(d 0 (x [| 2; 3 |]) *@ d 1 (x [| 3; 2 |])));
      c "( .%{} )" ~by:"get" 1 (fun d -> show Nx.((d 0 (x [| 2; 3 |])).%{[ 1 ]}));
      c "( .${} )" ~by:"slice" 1 (fun d ->
          show Nx.((d 0 (x [| 2; 3 |])).${[ A; R (0, 2) ]}));
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
      c ~consumes:true "vector_norm" 1 (fun d ->
          show (Nx.vector_norm dt (d 0 (x [| 4 |]))));
      c ~consumes:true "vector_norm" 1 (fun d ->
          show (Nx.vector_norm dt ~ord:3. (d 0 (x [| 4 |]))));
      c ~consumes:true "vector_norm" 1 (fun d ->
          show (Nx.vector_norm Nx.float64 (d 0 (x [| 4 |]))));
      c ~consumes:true "matrix_norm" 1 (fun d ->
          show (Nx.matrix_norm dt (d 0 (x [| 2; 2 |]))));
      c ~consumes:true "matrix_norm" 1 (fun d ->
          show (Nx.matrix_norm dt ~ord:`One (d 0 (x [| 2; 2 |]))));
      c ~consumes:true "matrix_norm" 1 (fun d ->
          show (Nx.matrix_norm dt ~ord:`Two (d 0 (x [| 2; 2 |]))));
      c ~consumes:true "matrix_norm" 1 (fun d ->
          show (Nx.matrix_norm Nx.float16 ~ord:`Nuc (d 0 (x [| 2; 2 |]))));
      c ~consumes:true "cond" 1 (fun d ->
          show (Nx.cond dt (d 0 (positive [| 2; 2 |]))));
      c "cond" 1 (fun d ->
          show (Nx.cond Nx.float64 ~p:`Inf (d 0 (positive [| 2; 2 |]))));
      c "cond" 1 (fun d ->
          show (Nx.cond Nx.float64 ~p:`One (d 0 (positive [| 2; 2 |]))));
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
      c ~consumes:true "isnan" 1 (fun d ->
          show (Nx.isnan (d 0 (Nx.log (x [| 4 |])))));
      c "isinf" 1 (fun d -> show (Nx.isinf (d 0 (Nx.log (x [| 4 |])))));
      c "isfinite" 1 (fun d -> show (Nx.isfinite (d 0 (Nx.log (x [| 4 |])))));
      c "clamp" 1 (fun d -> show (Nx.clamp (d 0 (x [| 4 |]))));
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
      c "argmin" 1 (fun d ->
          show (Nx.argmin ~keepdims:true (d 0 (x [| 2; 3 |]))));
      c ~consumes:true "var" 1 (fun d -> show (Nx.var (d 0 (x [| 2; 3 |]))));
      c ~consumes:true "std" 1 (fun d ->
          show (Nx.std ~ddof:1 (d 0 (x [| 5 |]))));
      c "logsumexp" 1 (fun d -> show (Nx.logsumexp (d 0 (x [| 2; 3 |]))));
      c "logsumexp" 1 (fun d ->
          show (Nx.logsumexp ~axes:[ 1 ] ~keepdims:true (d 0 (x [| 2; 3 |]))));
      c ~consumes:true "logmeanexp" 1 (fun d ->
          show (Nx.logmeanexp ~axes:[ 1 ] (d 0 (x [| 2; 3 |]))));
      c ~consumes:true "logmeanexp" 1 (fun d ->
          show (Nx.logmeanexp ~keepdims:true (d 0 (x [| 2; 3 |]))));
      c "cummax" 1 (fun d -> show (Nx.cummax ~axis:0 (d 0 (x [| 2; 3 |]))));
      c "cummin" 1 (fun d -> show (Nx.cummin (d 0 (x [| 2; 3 |]))));
      c ~consumes:true "ewma" 1 (fun d ->
          show (Nx.ewma ~alpha:0.5 (d 0 (x [| 2; 3 |]))));
      c ~consumes:true "ewma" 1 (fun d ->
          show (Nx.ewma ~axis:0 ~alpha:0.5 (d 0 (x [| 2; 3 |]))));
      c "sigmoid" 1 (fun d -> show (Nx.sigmoid (d 0 (x [| 4 |]))));
      c "softplus" 1 (fun d -> show (Nx.softplus (d 0 (x [| 4 |]))));
      c "softmax" 1 (fun d -> show (Nx.softmax (d 0 (x [| 2; 3 |]))));
      c "log_softmax" 1 (fun d ->
          show (Nx.log_softmax ~axes:[ 0 ] (d 0 (x [| 2; 3 |]))));
      c "conv" 2 (fun d ->
          show (Nx.conv (d 0 (x [| 1; 4; 1 |])) (d 1 (x [| 1; 2; 1 |]))));
      c "conv" 2 (fun d ->
          show (Nx.conv (d 0 (x [| 1; 4; 1 |])) (d 1 (x [| 1; 0; 1 |]))));
      c "conv_transpose" 2 (fun d ->
          show
            (Nx.conv_transpose (d 0 (x [| 1; 4; 1 |])) (d 1 (x [| 1; 2; 1 |]))));
      c "correlate" 2 (fun d ->
          show (Nx.correlate (d 0 (x [| 4 |])) (d 1 (x [| 2 |]))));
      c "conv" 2 (fun d ->
          show
            (Nx.conv
               ~padding:[| (1, 1) |]
               (d 0 (x [| 1; 4; 2 |]))
               (d 1 (x [| 3; 2; 2 |]))));
      c "conv_transpose" 2 (fun d ->
          show
            (Nx.conv_transpose ~stride:[| 2 |]
               (d 0 (x [| 1; 3; 2 |]))
               (d 1 (x [| 2; 2; 2 |]))));
      c "rms_norm" 1 (fun d -> show (Nx.rms_norm (d 0 (x [| 2; 3 |]))));
      c "correlate" 2 (fun d ->
          show (Nx.correlate ~padding:`Full (d 0 (x [| 4 |])) (d 1 (x [| 2 |]))));
      c "extract_patches" 1 (fun d ->
          show
            (Nx.extract_patches ~kernel_size:[| 2 |] ~stride:[| 1 |]
               ~dilation:[| 1 |]
               ~padding:[| (0, 0) |]
               (d 0 (x [| 2; 4 |]))));
      c ~consumes:true "combine_patches" 1 (fun d ->
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
      c ~consumes:true "searchsorted" 2 (fun d ->
          let s = d 0 (Nx.cumsum (Nx.abs (x [| 4 |]))) in
          show (Nx.searchsorted ~side:`Left s (d 1 (x [| 3 |]))));
      c ~consumes:true "searchsorted" 2 (fun d ->
          let s = d 0 (Nx.cumsum (Nx.abs (x [| 4 |]))) in
          show (Nx.searchsorted ~side:`Right s (d 1 (x [| 3 |]))));
      c "unique" 1 (fun d ->
          let g = Nx.unique (d 0 (x [| 6 |])) in
          showl [ g.ids; g.first; g.counts; g.count ]);
      c "unique" 1 (fun d ->
          let g = Nx.unique (d 0 (x [| 4; 2 |])) in
          showl [ g.ids; g.counts ]);
      c ~consumes:true "quantile" 1 (fun d ->
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
      c ~consumes:true "rfft" 1 (fun d ->
          show (Nx.rfft Nx.complex64 (d 0 (x [| 2; 4 |]))));
      c ~consumes:true "ihfft" 1 (fun d ->
          show (Nx.ihfft Nx.complex64 (d 0 (x [| 2; 4 |]))));
      c ~consumes:true "rfft" 1 (fun d ->
          show (Nx.rfft Nx.complex64 ~norm:Forward (d 0 (x [| 4 |]))));
      c "dct" 1 (fun d -> show (Nx.dct (d 0 (x [| 2; 4 |]))));
      c "dct" 1 (fun d -> show (Nx.dct ~type_:Type_1 (d 0 (x [| 2; 4 |]))));
      c "idct" 1 (fun d -> show (Nx.idct ~norm:Ortho (d 0 (x [| 4 |]))));
      c "dst" 1 (fun d -> show (Nx.dst ~type_:Type_1 (d 0 (x [| 4 |]))));
      c "idst" 1 (fun d -> show (Nx.idst (d 0 (x [| 4 |]))));
      c "idst" 1 (fun d -> show (Nx.idst ~type_:Type_4 (d 0 (x [| 4 |]))));
      c ~consumes:true "stft" 1 (fun d ->
          show (Nx.stft Nx.complex64 ~window:2 (d 0 (x [| 4 |]))));
      c ~consumes:true "stft" 2 (fun d ->
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

(* The routines' functions, of [dt], their real results in [r] and eig's in
   [c]. *)
let routines (type v s r c) (dt : (v, s) Nx.dtype) (r : (float, r) Nx.dtype)
    (c' : (Complex.t, c) Nx.dtype) =
  let x s = data dt s in
  (* Symmetric positive definite: [1.25] on the diagonal, [0.25] off it. *)
  let spd () =
    let v i = Nx.Dtype.of_float dt (if i mod 4 = 0 then 1.25 else 0.25) in
    let a = Nx.create dt [| 3; 3 |] (Array.init 9 v) in
    Nx.add a (Nx.zeros_like a)
  in
  table (D.name dt)
    [
      c ~consumes:true "cholesky" 1 (fun d ->
          show (Nx.cholesky (d 0 (spd ()))));
      c ~consumes:true "cholesky" 1 (fun d ->
          show (Nx.cholesky ~triangle:Upper (d 0 (spd ()))));
      c "qr" 1 (fun d ->
          let q, r = Nx.qr (d 0 (spd ())) in
          showl [ q; r ]);
      c ~consumes:true "lu" 1 (fun d ->
          let p, l, u = Nx.lu (d 0 (spd ())) in
          shows [ show p; show l; show u ]);
      c ~consumes:true "svd" 1 (fun d ->
          let u, s, vh = Nx.svd r (d 0 (spd ())) in
          shows [ show u; show s; show vh ]);
      c ~consumes:true "svd" 1 (fun d ->
          let _, s, _ = Nx.svd ~vectors:false Nx.float16 (d 0 (spd ())) in
          show s);
      c ~consumes:true "eigh" 1 (fun d ->
          let w, v = Nx.eigh r (d 0 (spd ())) in
          shows [ show w; show v ]);
      c ~consumes:true "eigh" 1 (fun d ->
          show (fst (Nx.eigh ~triangle:Upper r (d 0 (spd ())))));
      c ~consumes:true "eig" 1 (fun d ->
          let w, v = Nx.eig c' (d 0 (spd ())) in
          showl [ w; v ]);
      c ~consumes:true "det" 1 (fun d -> show (Nx.det (d 0 (spd ()))));
      c ~consumes:true "slogdet" 1 (fun d ->
          let s, l = Nx.slogdet r (d 0 (spd ())) in
          shows [ show s; show l ]);
      c ~consumes:true "solve" 2 (fun d ->
          show (Nx.solve (d 0 (spd ())) (d 1 (x [| 3 |]))));
      c ~consumes:true "solve_triangular" 2 (fun d ->
          show
            (Nx.solve_triangular (d 0 (Nx.tril (spd ()))) (d 1 (x [| 3; 1 |]))));
      c ~consumes:true "solve_triangular" 2 (fun d ->
          show
            (Nx.solve_triangular ~transpose:Transpose
               (d 0 (Nx.tril (spd ())))
               (d 1 (x [| 3; 1 |]))));
      c ~consumes:true "inv" 1 (fun d -> show (Nx.inv (d 0 (spd ()))));
      c ~consumes:true "matrix_power" 1 (fun d ->
          show (Nx.matrix_power (d 0 (x [| 2; 2 |])) (-3)));
      c "lstsq" 2 (fun d ->
          let x, res, k, s = Nx.lstsq r (d 0 (spd ())) (d 1 (x [| 3 |])) in
          shows [ show x; show res; show k; show s ]);
      c "lstsq" 1 (fun d ->
          (* The donor beside its handle. *)
          let a = spd () in
          let x, _, _, _ = Nx.lstsq r (d 0 a) a in
          show x);
      c ~consumes:true "pinv" 1 (fun d -> show (Nx.pinv (d 0 (spd ()))));
      c ~consumes:true "pinv" 1 (fun d ->
          show (Nx.pinv ~hermitian:true (d 0 (spd ()))));
      c ~consumes:true "matrix_rank" 1 (fun d ->
          show (Nx.matrix_rank (d 0 (spd ()))));
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
      c ~consumes:true "complex" 2 (fun d ->
          show
            (Nx.complex Nx.complex64
               ~re:(d 0 (data f32 [| 4 |]))
               ~im:(d 1 (data f32 [| 4 |]))));
      c "isinf" 1 (fun d -> show (Nx.isinf (d 0 (z [| 4 |]))));
      c ~consumes:true "fft" 1 (fun d -> show (Nx.fft (d 0 (z [| 2; 4 |]))));
      c ~consumes:true "fft" 1 (fun d ->
          show (Nx.fft ~norm:Ortho (d 0 (z [| 4 |]))));
      c ~consumes:true "ifft" 1 (fun d ->
          show (Nx.ifft ~norm:Ortho (d 0 (z [| 4 |]))));
      c ~consumes:true "irfft" 1 (fun d ->
          show (Nx.irfft f32 (d 0 (z [| 3 |]))));
      c ~consumes:true "hfft" 1 (fun d -> show (Nx.hfft f32 (d 0 (z [| 3 |]))));
      c ~consumes:true "istft" 1 (fun d ->
          show (Nx.istft f32 ~window:2 ~step:1 (d 0 (z [| 2; 2 |]))));
      c "istft" 2 (fun d ->
          show
            (Nx.istft f32 ~window:4 ~step:2
               ~win:(d 1 (Nx.add_s (Nx.abs (data f32 [| 4 |])) 1.))
               (d 0 (z [| 3; 3 |]))));
      c ~consumes:true "eig" 1 (fun d ->
          let w, v =
            Nx.eig Nx.complex64
              (d 0 (Nx.add (Nx.eye Nx.complex64 2) (z [| 2; 2 |])))
          in
          showl [ w; v ]);
      c ~consumes:true "vector_norm" 1 (fun d ->
          show (Nx.vector_norm f32 (d 0 (z [| 4 |]))));
      c ~consumes:true "vector_norm" 1 (fun d ->
          show (Nx.vector_norm Nx.float64 ~ord:1. (d 0 (z [| 4 |]))));
      c ~consumes:true "matrix_norm" 1 (fun d ->
          show (Nx.matrix_norm f32 ~ord:`Inf (d 0 (z [| 2; 2 |]))));
      c ~consumes:true "matrix_norm" 1 (fun d ->
          show (Nx.matrix_norm Nx.bfloat16 ~ord:`Two (d 0 (z [| 2; 2 |]))));
      c ~consumes:true "cond" 1 (fun d ->
          show
            (Nx.cond f32 (d 0 (Nx.add (Nx.eye Nx.complex64 2) (z [| 2; 2 |])))));
      c ~consumes:true "cond" 1 (fun d ->
          show
            (Nx.cond Nx.float64 ~p:`One
               (d 0 (Nx.add (Nx.eye Nx.complex64 2) (z [| 2; 2 |])))));
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
      c ~consumes:true "positions" 1 (fun d ->
          show (Nx.positions (d 0 (Nx.abs (x [| 4 |])))));
      c ~consumes:true "positions" 1 (fun d ->
          show (Nx.positions ~size:4 (d 0 (Nx.abs (x [| 4 |])))));
      c "one_hot" 1 (fun d -> show (Nx.one_hot ~num_classes:2 (d 0 (ids ()))));
      c "unique" 1 (fun d ->
          let g = Nx.unique (d 0 (x [| 6 |])) in
          showl [ g.ids; g.first; g.counts; g.count ]);
      c "sort" 1 (fun d ->
          showl
            (let v, _ = Nx.sort (d 0 (x [| 5 |])) in
             [ v ]));
      c ~consumes:true "unravel_index" 1 (fun d ->
          showl (Nx.unravel_index (d 0 (ints [| 0; 4; 7 |])) [| 2; 3 |]));
      c "( land )" ~by:"bitwise_and" 2 (fun d ->
          show Nx.(d 0 (x [| 4 |]) land d 1 (Nx.flip (x [| 4 |]))));
      c "( lor )" ~by:"bitwise_or" 2 (fun d ->
          show Nx.(d 0 (x [| 4 |]) lor d 1 (Nx.flip (x [| 4 |]))));
      c "( lxor )" ~by:"bitwise_xor" 2 (fun d ->
          show Nx.(d 0 (x [| 4 |]) lxor d 1 (Nx.flip (x [| 4 |]))));
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
      xs "( **$ )" "pow_s" Nx.( **$ );
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
      c ~consumes:true "causal" 3 (fun d -> show (Nx.causal (pair d)));
    ]

(* An array a score returns or reads is an operand after the call's own, or
   reaches the interpretation that records the score: a donated one gives the
   undonated result and lives, since attention computes in several
   operations. *)
let test_captured () =
  let run score =
    let q = data f32 [| 1; 2; 2 |] in
    let kv =
      Nx.Dense { k = data f32 [| 1; 3; 2 |]; v = data f32 [| 1; 3; 2 |] }
    in
    let o, lse = Nx.attention ~score ~kv q in
    showl [ o; lse ]
  in
  let returned bias _ _ = bias and added bias _ s = Nx.add s bias in
  let bias = data f32 [| 1; 1; 3 |] in
  let plain = run (returned bias) in
  equal ~msg:"returned" string plain (run (returned (Nx.donate bias)));
  equal ~msg:"returned: the donor lives" string
    (show (data f32 [| 1; 1; 3 |]))
    (show bias);
  let bias = data f32 [| 1; 1; 3 |] in
  let plain = run (added bias) in
  equal ~msg:"read" string plain (run (added (Nx.donate bias)));
  equal ~msg:"read: the donor lives" string
    (show (data f32 [| 1; 1; 3 |]))
    (show bias)

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
      routines Nx.float64 Nx.float64 Nx.complex128;
      routines Nx.complex64 Nx.float32 Nx.complex64;
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
      let killed = ref "" in
      let dead x =
        killed := text x;
        kill x
      in
      let run () = c.run (fun j x -> if j = i then dead x else x) in
      match c.dead with
      | Prints death -> equal ~msg:name string death (run ())
      | Raises ->
          let raised =
            match run () with
            | _ -> "no exception"
            | exception Invalid_argument m -> m
          in
          let want = c.by ^ ": " ^ !killed ^ " was donated to Nx.copy" in
          equal ~msg:name string want raised)

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
  let named = "donated to " ^ c.by in
  (* What a copy of [y] gives that the case does not: a death that names
     another function, or, from a consuming case, a value alive. *)
  let wrong ~consumed y =
    match Nx.copy y with
    | _ -> if consumed then [ text y ^ " lives" ] else []
    | exception Invalid_argument m ->
        if String.ends_with ~suffix:named m then [] else [ m ]
  in
  let donors = ref [] in
  let donating : 'v 's. int -> ('v, 's, Nx.host) Nx.t -> ('v, 's, Nx.host) Nx.t
      =
   fun i x ->
    if mask land (1 lsl i) = 0 then x
    else begin
      let h = Nx.donate x in
      let donor ~consumed = wrong ~consumed x in
      let handle ~consumed = wrong ~consumed h in
      donors := (donor, handle) :: !donors;
      h
    end
  in
  equal ~msg:name string plain (c.run donating);
  (* The donor first: a copy of a live handle would consume it. *)
  let deaths (donor, handle) =
    let d = donor ~consumed:true in
    d @ handle ~consumed:true
  in
  if c.consumes then
    equal ~msg:(name ^ ", deaths") (list string) []
      (List.concat_map deaths !donors);
  donors := [];
  let received =
    Nx.Prim.interpret ~name:"test.extent" Extent
      (fun _ ~by op -> Nx.Prim.eval ~by op)
      (fun _ -> c.run donating)
  in
  equal ~msg:(name ^ ", received") string plain received;
  equal ~msg:(name ^ ", deaths received") (list string) []
    (List.concat_map (fun (donor, _) -> donor ~consumed:false) !donors)

let test_donated =
  prop ~examples:seeds
    "a donated operand gives the undonated result, received too, and dies \
     where one operation reads it"
    donations law_donated

(* Consumers: a function of several operations that consumes as one
   operation does. *)
module C = Nx.Prim.Consumer

let ended = "test.call: the consumer ended its handles"

let test_consumer_kills () =
  let x = fresh [| 1.; 2.; 3. |] and y = fresh [| 4.; 5.; 6. |] in
  let kept =
    C.with_ ~by:"test.call" [ Any (Nx.donate x); Any y ] (fun c ->
        equal ~msg:"the handle reads the donor" (array float_exact)
          [| 1.; 2.; 3. |]
          (host_floats (C.arrays c 0).(0));
        equal ~msg:"an undonated operand" (array float_exact) [| 4.; 5.; 6. |]
          (host_floats (C.arrays c 1).(0));
        invalid ~sub:(dead_in "Nx.copy" x "test.call") (fun () -> Nx.copy x);
        c)
  in
  equal ~msg:"an undonated operand lives" (array float_exact)
    [| 4.; 5.; 6. |] (elements y);
  raises (Invalid_argument ended) (fun () -> C.arrays kept 0);
  raises (Invalid_argument ended) (fun () ->
      C.take kept 0 D.Float32 (A.Layout.contiguous [| 3 |]))

let test_consumer_checks_first () =
  let x = fresh [| 1. |] and dead = fresh [| 2. |] in
  ignore (Nx.copy (Nx.donate dead));
  invalid ~sub:(dead_in "test.call" dead "Nx.copy") (fun () ->
      C.with_ ~by:"test.call" [ Any (Nx.donate x); Any dead ] ignore);
  equal ~msg:"the other donor lives" (array float_exact) [| 1. |] (elements x)

let test_consumer_refuses () =
  let x = fresh [| 1. |] in
  C.with_ ~by:"test.call" [ Any x; Any (Nx.zeros f32 [| 2 |]) ] (fun c ->
      equal ~msg:"a creation's bytes" (array float_exact) [| 0.; 0. |]
        (host_floats (C.arrays c 1).(0));
      invalid ~sub:"test.call: no operand 2" (fun () -> C.arrays c 2))

let layout_of n = A.Layout.contiguous [| n |]

(* A consumer reads operands with bytes alone: [arrays] and [take] raise
   naming it for a traced value. *)
let test_consumer_no_bytes =
  let traced k =
    Nx.Prim.interpret ~name:"test.tracer" Values
      (fun i ~by op ->
        Nx.Prim.results ~by (fun _ form -> Nx.Prim.traced i form Tag) op)
      (fun i -> k (Nx.Prim.traced i (Nx.Prim.form (fresh [| 1.; 2. |])) Tag))
  in
  let operands = [ ("a traced value", traced) ] in
  let readers =
    [
      ("arrays", fun c -> ignore (C.arrays c 0));
      ("take", fun c -> ignore (C.take c 0 D.Float32 (layout_of 2)));
    ]
  in
  let each =
    List.concat_map (fun r -> List.map (fun o -> (r, o)) operands) readers
  in
  cases
    ~name:(fun ((r, _), (o, _)) -> r ^ " of " ^ o)
    "a consumer refuses an operand without bytes" each
    (fun ((_, read), (_, given)) ->
      given (fun x ->
          invalid ~sub:"test.call: operand 0" (fun () ->
              C.with_ ~by:"test.call" [ Any x ] read)))

(* A handle's memory is taken for a result only where nothing else reads it:
   the handle alone among the operands, owning its memory, of the result's
   dtype and layout. *)
let test_consumer_takes () =
  let some = function Some _ -> true | None -> false in
  let take xs i dt l =
    C.with_ ~by:"test.call" xs (fun c -> some (C.take c i dt l))
  in
  let h () = Nx.donate (fresh [| 1.; 2. |]) in
  equal ~msg:"a handle alone" bool true
    (take [ Any (h ()) ] 0 D.Float32 (layout_of 2));
  equal ~msg:"another dtype" bool false
    (take [ Any (h ()) ] 0 D.Int32 (layout_of 2));
  equal ~msg:"another layout" bool false
    (take [ Any (h ()) ] 0 D.Float32 (A.Layout.contiguous [| 1; 2 |]));
  equal ~msg:"an undonated operand" bool false
    (take [ Any (fresh [| 1.; 2. |]) ] 0 D.Float32 (layout_of 2));
  let x = fresh [| 1.; 2. |] in
  equal ~msg:"beside its donor" bool false
    (take [ Any (Nx.donate x); Any x ] 0 D.Float32 (layout_of 2));
  let x = fresh [| 1.; 2. |] in
  let view = Nx.flip x in
  equal ~msg:"beside a view of it" bool false
    (take [ Any (Nx.donate x); Any view ] 0 D.Float32 (layout_of 2));
  equal ~msg:"the view keeps its elements" (array float_exact) [| 2.; 1. |]
    (elements view);
  let x = fresh [| 1.; 2. |] in
  ignore (Nx.Repr.array x);
  equal ~msg:"shared by a crossing" bool false
    (take [ Any (Nx.donate x) ] 0 D.Float32 (layout_of 2));
  C.with_ ~by:"test.call" [ Any (h ()) ] (fun c ->
      let first = some (C.take c 0 D.Float32 (layout_of 2)) in
      let second = some (C.take c 0 D.Float32 (layout_of 2)) in
      equal ~msg:"taken once" (Testable.pair bool bool) (true, false)
        (first, second))

(* An update in place, [y = x + 1] over [x]'s taken memory: the taken array
   holds [x]'s elements until it is written, and needs no memory of the
   result's size. A consumer that takes none returns the donor's memory
   when it ends, even when it raises, and one that raises returns the memory
   it took, though the taken array stays reachable. *)
let test_consumer_memory () =
  let n = 1 lsl 16 in
  let bytes = 4 * n in
  let make () = on_mem [| n |] (Array.init n Float.of_int) in
  let increment x =
    C.with_ ~by:"test.call" [ Any (Nx.donate x) ] (fun c ->
        match C.take c 0 D.Float32 (layout_of n) with
        | Some a ->
            let ys = Array.map (fun v -> v +. 1.) (A.to_array a) in
            A.transfer ~src:(A.of_array D.Float32 [| n |] ys) ~dst:a;
            Some (Nx.Repr.unsafe_of_array Mem.v a)
        | None -> None)
  in
  let y, grew = grown make increment in
  in_place ~msg:"taken" ~bytes grew;
  equal ~msg:"its elements" (array float_exact)
    (Array.init n (fun k -> Float.of_int k +. 1.))
    (elements (Option.get y));
  let freed f =
    let x = make () in
    let before = settled () in
    (try f x with Exit -> ());
    let freed = before - settled () in
    ignore (Sys.opaque_identity x);
    freed
  in
  at_least ~msg:"a donor's memory returns at the end, its donor reachable"
    int ~than:bytes
    (freed (fun x ->
         C.with_ ~by:"test.call" [ Any (Nx.donate x) ] (fun _ -> ())));
  at_least ~msg:"and when the consumer raises" int ~than:bytes
    (freed (fun x ->
         C.with_ ~by:"test.call" [ Any (Nx.donate x) ] (fun _ -> raise Exit)));
  let kept = ref None in
  at_least ~msg:"the memory it took, when the consumer raises" int ~than:bytes
    (freed (fun x ->
         C.with_ ~by:"test.call" [ Any (Nx.donate x) ] (fun c ->
             kept := C.take c 0 D.Float32 (layout_of n);
             raise Exit)));
  equal ~msg:"it took the memory" bool true (Option.is_some !kept);
  ignore (Sys.opaque_identity !kept)

(* A consumer that ends returns the memory of every handle it claimed, in
   either order of its operands: the donor's, which a view passed on reads,
   and a copy's, which a movement made. Every value stays reachable across
   the count, so no collection returns either. *)
let test_consumer_frees_every_handle =
  let k = 256 in
  let bytes = 4 * k * k in
  cases
    ~name:(function
      | true -> "a copy, then a view" | false -> "a view, then a copy")
    "a consumer frees the memory of every handle it claimed" [ true; false ]
    (fun copy_first ->
      let x = on_mem [| k; k |] (Array.init (k * k) Float.of_int) in
      let view = Nx.reshape [| k * k |] (Nx.donate x) in
      (* A reshape of a transpose copies. *)
      let copy = Nx.reshape [| k * k |] (Nx.transpose (Nx.donate x)) in
      let before = settled () in
      C.with_ ~by:"test.call"
        (if copy_first then [ Any copy; Any view ] else [ Any view; Any copy ])
        ignore;
      let freed = before - settled () in
      ignore (Sys.opaque_identity (x, view, copy));
      at_least ~msg:"the donor's memory and the copy's" int ~than:(2 * bytes)
        freed)

let () =
  exit
    (run "nx donate"
       [
         group "donation"
           [
             test "a donor and its handle die at the consumer, naming it"
               test_dies_at_its_consumer;
             law_one_donor_twice;
             law_moved_twice;
             stateful "donors and handles across calls" lives;
             test "a one-to-one movement passes the handle on" test_passed_on;
             test "a value sharing the memory keeps its elements"
               test_sharer_keeps;
             test "a refused consumer kills no donor" test_refused_kills_none;
             test "a dtype refusal kills no donor" test_dtype_refused;
             test "a rule's refusal kills no donor" test_rule_refused;
             test "a refused broadcast kills no donor" test_broadcast_refused;
             test "an aligning consumer checks before it consumes"
               test_aligned_refusal;
             test "a donated creation dies" test_creation_dies;
             test "a value whose shards lie in one allocation is donated"
               test_shards_of_one_allocation;
             test "a function of several operations kills nothing"
               test_several_reads;
             test "a sampler lends its key" test_key_lent;
             test "an interpretation reads a handle and kills nothing"
               test_lent;
             test "a donor dies naming the function the user called"
               test_dies_at_the_call;
             test "a donor beside its handle in another operation raises"
               test_beside_elsewhere;
             test "a call that keeps none of a dead operand raises"
               test_none_kept;
             test_dead_names;
             test_donated;
             test "a captured array lives" test_captured;
             stateful ~domains:2 "reads and consumers on two domains" commands;
             stateful ~domains:2 "handles taken on two domains" handles;
             stateful ~domains:2 "two-handle consumers on two domains" crossed;
             test "donated tells a handle, live or not" test_is_handle;
             test "a consumer kills its handles' donors, naming itself"
               test_consumer_kills;
             test "a consumer checks every operand before it claims"
               test_consumer_checks_first;
             test "a consumer reads its operands' arrays by position"
               test_consumer_refuses;
             test_consumer_no_bytes;
             test "a consumer takes memory nothing else reads"
               test_consumer_takes;
             test "a consumer updates taken memory in place and frees the rest"
               test_consumer_memory;
             test_consumer_frees_every_handle;
             test "two values a call gives back over one donor stay apart"
               test_given_back_twice;
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
             test "a value on one device is written in place beside others"
               test_beside_in_place;
             test "a donation during Repr.read computes into fresh memory"
               test_read_donated_inside;
             test
               "a donation on another domain during Repr.read computes into \
                fresh memory" test_read_donated_across;
             test "a handle a view passes on is written in place"
               test_view_in_place;
             test "where writes into a donated value operand"
               test_where_in_place;
             test "a strided operand computes into fresh memory"
               test_strided_copied;
             test "an undonated or shared operand keeps its elements" test_kept;
             test "an operation may read the donor beside its handle"
               test_donor_beside;
             test "a consumer inside a composite leaves its reads alone"
               test_inside_composite;
             test
               "an eager call beside an Extent on another domain frees its \
                donor" test_extent_elsewhere;
             test "a declining operation returns its donors' memory"
               test_declined_frees;
             test "a donation leaves the arrays a rule kept"
               test_kept_by_rule;
             law_scatter_bits;
             law_init_bits;
           ];
       ])
