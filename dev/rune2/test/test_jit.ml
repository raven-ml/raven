(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.jit: a compiled call computes eager's values within rune2.mli's
   classes, exact functions bit for bit, every NaN read as one, and the others
   within their bounds; a call retraces exactly when its key changes; inside
   it reads and keyless draws raise, naming Rune.jit. *)

open Windtrap
open Compose
module Rune = Rune2
module T = Nx.Ptree

let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let f32 = Nx.float32
let f64 = Nx.float64
let v xs = Nx.create f32 [| Array.length xs |] xs
let pair_fn () = T.(tensor @-> tensor @-> returns tensor)

(* Exact compositions *)

type case =
  | Case : (float, 's) Nx.dtype * int array * operand * operand * expr -> case

let cases =
  let open Gen in
  (let* s = shape in
   let* e = expr ~rank:(Array.length s) in
   let* x = operand s and+ y = operand s in
   let+ (Nx.Dtype.Any dt) =
     of_list [ Nx.Dtype.Any Nx.float32; Any Nx.float64 ]
   in
   match Nx.Dtype.kind dt with
   | Float -> Case (dt, s, x, y, exact e)
   | _ -> assert false)
  |> with_pp (fun ppf (Case (dt, s, x, y, e)) ->
      Format.fprintf ppf "%a %a, x %a, y %a: %a" Nx.pp_dtype dt pp_shape s
        pp_view x.view pp_view y.view pp e)

let rec reduces = function
  | X | Y | K _ -> false
  | Un (Max _, _) -> true
  | Un (_, e) -> reduces e
  | Bin (_, e, e') -> reduces e || reduces e'

let law (Case (dt, s, xo, yo, e)) =
  let x = make dt s xo and y = make dt s yo in
  cover "a zero-size shape" (numel s = 0);
  cover "a one-element shape" (numel s = 1 && Array.length s > 0);
  cover "a 0-d shape" (Array.length s = 0);
  cover "a view operand" (xo.view <> Plain || yo.view <> Plain);
  cover "a reduction" (reduces e);
  cover "a composition of over 6 functions" (size e > 6);
  let f x y = eval x y e in
  let eager = outcome (fun () -> bits (f x y)) in
  let jitted = outcome (fun () -> bits (Rune.jit (pair_fn ()) f x y)) in
  cover "a refused operation" (match eager with Raised _ -> true | _ -> false);
  equal (outcome_t bits_t) eager jitted

(* Every dtype, through the functions every dtype has that it computes
   exactly. *)
type any_case =
  | Any_case :
      ('v, 's) Nx.dtype * int array * operand * operand * expr
      -> any_case

let any_cases =
  let open Gen in
  (let* s = shape in
   let* e = expr ~rank:(Array.length s) in
   let* x = operand s and+ y = operand s in
   let+ (Nx.Dtype.Any dt) = of_list Nx.Dtype.all in
   Any_case (dt, s, x, y, exact_any dt (generic e)))
  |> with_pp (fun ppf (Any_case (dt, s, x, y, e)) ->
      Format.fprintf ppf "%a %a, x %a, y %a: %a" Nx.pp_dtype dt pp_shape s
        pp_view x.view pp_view y.view pp e)

let law_any (Any_case (dt, s, xo, yo, e)) =
  let x = make_any dt s xo and y = make_any dt s yo in
  let kind : type v. v Nx.Dtype.kind -> string = function
    | Float ->
        if Nx.Dtype.equal dt Nx.float64 || Nx.Dtype.equal dt Nx.float32 then
          "wide float"
        else "narrow float"
    | Complex -> "complex"
    | Signed | Unsigned -> "integer"
    | Boolean -> "boolean"
  in
  let k = kind (Nx.Dtype.kind dt) in
  List.iter
    (fun c -> cover ("a " ^ c ^ " dtype") (k = c))
    [ "narrow float"; "complex"; "integer"; "boolean" ];
  cover "a sub-byte dtype" (Nx.Dtype.bits dt < 8);
  let f x y = eval_any x y e in
  let eager = outcome (fun () -> codes (f x y)) in
  let jitted = outcome (fun () -> codes (Rune.jit (pair_fn ()) f x y)) in
  equal (outcome_t codes_t) eager jitted

(* Bounded functions

   A function of one kind is within the sum of its compiled bound and its
   eager one, in ulps, of eager's result: each is within its own of the
   correctly rounded result. Special values are exact in both. *)

type unary = {
  name : string;
  f : 's 'd. (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t;
  bound : int * int;  (** At float32, at float64. *)
}

(* rune2.mli's bound, then nx's. *)
let unaries =
  [
    { name = "exp"; f = Nx.exp; bound = (9 + 2, 9 + 2) };
    { name = "log"; f = Nx.log; bound = (9 + 1, 9 + 2) };
    { name = "sin"; f = Nx.sin; bound = (4 + 2, 4 + 2) };
    { name = "cos"; f = Nx.cos; bound = (9 + 2, 9 + 2) };
    { name = "tan"; f = Nx.tan; bound = (19 + 4, 19 + 2) };
    { name = "asin"; f = Nx.asin; bound = (5 + 2, 9 + 2) };
    { name = "acos"; f = Nx.acos; bound = (5 + 2, 9 + 2) };
    { name = "atan"; f = Nx.atan; bound = (2 + 2, 7 + 2) };
    { name = "sinh"; f = Nx.sinh; bound = (21 + 3, 21 + 2) };
    { name = "cosh"; f = Nx.cosh; bound = (10 + 2, 10 + 2) };
    { name = "tanh"; f = Nx.tanh; bound = (18 + 2, 18 + 2) };
    { name = "expm1"; f = Nx.expm1; bound = (21 + 1, 21 + 2) };
    { name = "log1p"; f = Nx.log1p; bound = (12 + 1, 12 + 2) };
    { name = "erf"; f = Nx.erf; bound = (2 + 2, 3 + 2) };
  ]

(* A float's rank among the floats of its width in the order of their values,
   [-0] one below [+0]. *)
let rank ~wide x =
  let m, negative =
    if wide then
      let b = Int64.bits_of_float x in
      (Int64.logand b Int64.max_int, Int64.compare b 0L < 0)
    else
      let b = Int32.bits_of_float x in
      (Int64.of_int32 (Int32.logand b Int32.max_int), Int32.compare b 0l < 0)
  in
  if negative then Int64.(sub (neg m) 1L) else m

(* Whether [a] and [b] are at most [n] ranks apart, or both NaN. *)
let within ~wide n a b =
  if Float.is_nan a || Float.is_nan b then Float.is_nan a && Float.is_nan b
  else
    let ra = rank ~wide a and rb = rank ~wide b in
    let lo = Int64.min ra rb and hi = Int64.max ra rb in
    (* Ranks of both signs may be too far apart to subtract. *)
    Int64.compare hi (Int64.add lo (Int64.of_int n)) <= 0
    && Int64.compare (Int64.sub hi (Int64.of_int n)) lo <= 0

let floats =
  Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf x ->
      Format.fprintf ppf "%h" x)

let ulps ~wide n =
  Testable.make
    ~pp:(fun ppf a -> floats ppf (Array.to_list a))
    ~equal:(fun a b ->
      Array.length a = Array.length b && Array.for_all2 (within ~wide n) a b)

type unary_case =
  | Unary_case :
      (float, 's) Nx.dtype * unary * int array * operand
      -> unary_case

let unary_cases =
  let open Gen in
  (let* s = shape in
   let* u = of_list unaries and+ x = operand s in
   let+ wide = bool in
   if wide then Unary_case (f64, u, s, x) else Unary_case (f32, u, s, x))
  |> with_pp (fun ppf (Unary_case (dt, u, s, x)) ->
      Format.fprintf ppf "%s at %a %a, x %a" u.name Nx.pp_dtype dt pp_shape s
        pp_view x.view)

let law_unary (Unary_case (dt, u, s, xo)) =
  let x = make dt s xo in
  let wide = Nx.Dtype.equal dt f64 in
  cover "float64" wide;
  cover "a view operand" (xo.view <> Plain);
  cover "a special operand"
    (Array.exists
       (fun x -> Float.is_nan x || Float.is_integer x)
       (Nx.to_array x));
  let n = if wide then snd u.bound else fst u.bound in
  equal (ulps ~wide n) (Nx.to_array (u.f x)) (Nx.to_array (Rune.jit' u.f x))

(* A sum of [n] terms is within [γ(n - 1) Σ|x|] of the exact one, eagerly and
   compiled, so the two are within twice that of each other. A sum whose terms
   are not all finite is the same special value in both. *)
let gamma ~wide k =
  let u = if wide then Float.ldexp 1. (-53) else Float.ldexp 1. (-24) in
  let k = Float.of_int k in
  k *. u /. (1. -. (k *. u))

let sums_within ~wide ~terms ~magnitudes =
  let close i x y =
    let m = magnitudes.(i) in
    if Float.is_nan x || Float.is_nan y then Float.is_nan x && Float.is_nan y
    else if not (Float.is_finite m && Float.is_finite x) then x = y
    else Float.abs (x -. y) <= 2. *. gamma ~wide (terms - 1) *. m
  in
  Testable.make
    ~pp:(fun ppf a -> floats ppf (Array.to_list a))
    ~equal:(fun a b ->
      Array.length a = Array.length b
      && Array.for_all Fun.id (Array.mapi (fun i x -> close i x b.(i)) a))

type sum_case =
  | Sum_case : (float, 's) Nx.dtype * int array * int * operand -> sum_case

let sum_cases =
  let open Gen in
  (let* s = shape in
   let s = if Array.length s = 0 then [| 3 |] else s in
   let* axis = int_range 0 (Array.length s - 1) and+ x = operand s in
   let+ wide = bool in
   if wide then Sum_case (f64, s, axis, x) else Sum_case (f32, s, axis, x))
  |> with_pp (fun ppf (Sum_case (dt, s, a, x)) ->
      Format.fprintf ppf "sum~%d at %a %a, x %a" a Nx.pp_dtype dt pp_shape s
        pp_view x.view)

let law_sum (Sum_case (dt, s, axis, xo)) =
  let x = make dt s xo in
  let wide = Nx.Dtype.equal dt f64 in
  cover "float64" wide;
  cover "over 2 terms" (s.(axis) > 2);
  let sum x = Nx.sum ~axes:[ axis ] x in
  let magnitudes = Nx.to_array (Nx.cast f64 (sum (Nx.abs x))) in
  equal
    (sums_within ~wide ~terms:s.(axis) ~magnitudes)
    (Nx.to_array (sum x))
    (Nx.to_array (Rune.jit' sum x))

(* Reads and draws *)

let test_read () =
  invalid ~sub:"Nx.item: the value is traced by Rune.jit" (fun () ->
      Rune.jit' (fun x -> Nx.scalar f32 (Nx.item [ 0 ] x)) (v [| 1. |]));
  (* A value that depends on no argument is staged too. *)
  let c = v [| 2. |] in
  invalid ~sub:"traced by Rune.jit" (fun () ->
      Rune.jit'
        (fun x -> Nx.add x (Nx.scalar f32 (Nx.item [ 0 ] (Nx.mul c c))))
        c)

let test_keyless () =
  invalid ~sub:"no key inside Rune.jit" (fun () ->
      Rune.jit'
        (fun x -> Nx.add x (Nx.Rng.uniform f32 [| 2 |]))
        (v [| 1.; 2. |]))

(* A key that depends on no argument draws the same values on every call, the
   eager draw's: a uniform draw is exact. *)
let test_constant_key () =
  let f x = Nx.add x (Nx.Rng.uniform ~key:(Nx.Rng.key 7) f32 [| 3 |]) in
  let x = v [| 0.; 1.; 2. |] in
  let g = Rune.jit' f in
  let a = bits (g x) and b = bits (g x) in
  equal bits_t (bits (f x)) a;
  equal bits_t a b

(* The function's code runs once per key. *)
let test_runs_once () =
  let n = ref 0 in
  let g =
    Rune.jit' (fun x ->
        incr n;
        Nx.neg x)
  in
  ignore (g (v [| 1. |]));
  ignore (g (v [| 2. |]));
  equal ~msg:"one key" int 1 !n;
  ignore (g (v [| 1.; 2. |]));
  equal ~msg:"two keys" int 2 !n

(* Arguments rebuild their structure; a result of several arrays. *)
let test_structures () =
  let s =
    Nx.Ptree.(
      pair tensor (list tensor) @-> tensor @-> returns (pair tensor tensor))
  in
  let f (a, bs) c = (Nx.add a c, List.fold_left Nx.mul c bs) in
  let a = v [| 1.; 2. |] and bs = [ v [| 3.; 4. |]; v [| -0.; 5. |] ] in
  let c = v [| 0.5; -1. |] in
  let p, q = Rune.jit s f (a, bs) c and p', q' = f (a, bs) c in
  equal bits_t (bits p') (bits p);
  equal bits_t (bits q') (bits q)

(* Values of every set: an argument of every set is computed for the call, so
   one key serves its values; a function of such arguments alone gives a value
   of every set, as eagerly. *)
let test_every_set () =
  let traces = ref 0 in
  let g =
    Rune.jit (pair_fn ()) (fun x y ->
        incr traces;
        Nx.add x y)
  in
  let x = v [| 1.; 2. |] in
  equal bits_t
    (bits (Nx.add x (Nx.ones f32 [| 2 |])))
    (bits (g x (Nx.ones f32 [| 2 |])));
  equal bits_t
    (bits (Nx.add x (Nx.full f32 [| 2 |] 3.)))
    (bits (g x (Nx.full f32 [| 2 |] 3.)));
  equal ~msg:"one trace" int 1 !traces;
  let y = Rune.jit' (fun x -> Nx.add x x) (Nx.full f32 [| 2 |] 1.5) in
  equal ~msg:"of every set" bool true (Option.is_none (Nx.placement y));
  equal bits_t (bits (Nx.full f32 [| 2 |] 3.)) (bits y)

(* A jit inside a jit is staged into the outer program, a call over constants
   included. *)
let test_nested () =
  let inner = Rune.jit' (fun x -> Nx.mul x x) in
  let c = v [| 3.; -0. |] in
  let outer = Rune.jit' (fun x -> Nx.add (inner x) (inner c)) in
  let x = v [| 1.; 2. |] in
  equal bits_t (bits (Nx.add (Nx.mul x x) (Nx.mul c c))) (bits (outer x))

(* Fresh results: a result that is an argument is a copy, compiled and
   evaluated. *)
let test_fresh () =
  let x = v [| 1.; 2. |] in
  let b = Nx.Repr.read x Nx_array.buffer in
  let shares y = Rig.Buffer.overlaps b (Nx_array.buffer (Option.get (Nx.Repr.array y))) in
  let compiled = Rune.jit' (fun x -> x) x in
  equal ~msg:"compiled" bool false (shares compiled);
  equal bits_t (bits x) (bits compiled);
  let looped =
    Rune.jit' (fun x -> fst (Rune.scan' ~f:(fun c r -> (c, r)) ~init:x (Nx.reshape [| 2; 1 |] x))) x
  in
  equal ~msg:"evaluated" bool false (shares looped);
  equal bits_t (bits x) (bits looped)

(* An argument whose memory starts off 16 bytes binds where it lies, through a
   program compiled for that phase. *)
let test_phase () =
  let f = Rune.jit' (fun x -> Nx.add x x) in
  let at first =
    let b = Rig.Buffer.create Rig.host 64 in
    let a =
      Nx_array.v Nx.float32 (Nx_array.Layout.contiguous (Iarray.of_list [ 4 ]))
        (Rig.Buffer.view b ~first ~length:16)
    in
    Nx_array.(set a (Iarray.of_list [ 0 ]) 1.);
    Nx_array.(set a (Iarray.of_list [ 1 ]) (-0.));
    Nx_array.(set a (Iarray.of_list [ 2 ]) 2.5);
    Nx_array.(set a (Iarray.of_list [ 3 ]) Float.nan);
    Nx.Repr.of_array Nx.Host.v a
  in
  List.iter
    (fun first ->
      let x = at first in
      equal ~msg:(Printf.sprintf "at byte %d" first) bits_t
        (bits (Nx.add x x)) (bits (f x)))
    [ 0; 4; 8; 12; 0 ]

(* A program that holds a loop is evaluated, giving eager's values. *)
let test_loop () =
  let step c r = (Nx.add c r, Nx.mul c r) in
  let xs = Nx.create f32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let init = v [| 0.5; -0. |] in
  let eager = Rune.scan' ~f:step ~init xs in
  let jitted =
    Rune.jit
      T.(tensor @-> tensor @-> returns (pair tensor tensor))
      (fun init xs -> Rune.scan' ~f:step ~init xs)
      init xs
  in
  equal bits_t (bits (fst eager)) (bits (fst jitted));
  equal bits_t (bits (snd eager)) (bits (snd jitted))

(* The cache

   A call retraces exactly when its key, the abstract value of its arguments,
   differs from every earlier call's of the same compiled function: here its
   shape, at each dtype's own compiled function, as rune2.mli states the key.
   Layouts, offsets and donation vary too, and change no key. The model is the
   set of keys traced, and each call answers whether it traced, which the
   function records on the domain that traces it. *)

type layout = Contiguous | Transposed | Offset of int

type arg = {
  wide : bool;
  rows : int;
  layout : layout;
  donated : bool;
  value : float;
}

let pp_arg ppf a =
  Format.fprintf ppf "%s [%d; 2] %s%s %g"
    (if a.wide then "float64" else "float32")
    a.rows
    (match a.layout with
    | Contiguous -> "contiguous"
    | Transposed -> "transposed"
    | Offset k -> Printf.sprintf "at element %d" k)
    (if a.donated then " donated" else "")
    a.value

let args =
  let open Gen in
  (let* wide = bool and+ rows = int_range 1 2 in
   let+ layout =
     frequency
       [
         (2, constant Contiguous);
         (1, constant Transposed);
         (2, map (fun k -> Offset k) (int_range 0 8));
       ]
   and+ donated = bool
   and+ value = float_range (-2.) 2. in
   { wide; rows; layout; donated; value })
  |> with_pp pp_arg

let key a = (a.wide, a.rows)

module Keys = Set.Make (struct
  type t = bool * int

  let compare = compare
end)

let make_arg (type s) (dt : (float, s) Nx.dtype) a =
  let n = a.rows * 2 in
  let x =
    match a.layout with
    | Contiguous -> Nx.create dt [| a.rows; 2 |] (Array.make n a.value)
    | Transposed ->
        Nx.transpose (Nx.create dt [| 2; a.rows |] (Array.make n a.value))
    | Offset k ->
        let flat = Nx.create dt [| k + n |] (Array.make (k + n) a.value) in
        Nx.reshape [| a.rows; 2 |] (Nx.slice [ Nx.R (k, k + n) ] flat)
  in
  if a.donated then Nx.donate x else x

let traced_here = Domain.DLS.new_key (fun () -> false)

type compiled = {
  at32 : Nx.host Nx.float32_t -> Nx.host Nx.float32_t;
  at64 : Nx.host Nx.float64_t -> Nx.host Nx.float64_t;
}

let negated x =
  Domain.DLS.set traced_here true;
  Nx.neg x

(* Whether the call traced, once its result has the negated value. *)
let system_call g a =
  Domain.DLS.set traced_here false;
  let value, expected =
    if a.wide then
      ( Nx.to_array (g.at64 (make_arg f64 a)),
        Nx.item [] (Nx.neg (Nx.scalar f64 a.value)) )
    else
      ( Nx.to_array (g.at32 (make_arg f32 a)),
        Nx.item [] (Nx.neg (Nx.scalar f32 a.value)) )
  in
  if not (Array.for_all (fun y -> y = expected) value) then
    failwith "a wrong value";
  Domain.DLS.get traced_here

let compiled = abstract "jit"

let cache_commands =
  let start () = ref Keys.empty in
  let model seen a =
    let k = key a in
    let fresh = not (Keys.mem k !seen) in
    seen := Keys.add k !seen;
    fresh
  in
  let system () = { at32 = Rune.jit' negated; at64 = Rune.jit' negated } in
  [
    command "start" (Gen.unit @-> makes compiled) start system;
    command "call" (compiled ^-> args @-> returns bool) model system_call;
  ]

let () =
  exit
    (run "rune2 jit"
       [
         group "classes"
           [
             prop "jit gives eager's bits for exact functions" cases law;
             prop "jit gives eager's bits for exact functions at every dtype"
               any_cases law_any;
             prop "a function of one kind is within its bounds" unary_cases
               law_unary;
             prop "a sum is within its bound" sum_cases law_sum;
           ];
         group "jit"
           [
             test "a read raises naming Rune.jit" test_read;
             test "a keyless draw raises naming Rune.jit" test_keyless;
             test "a constant key draws the same each call" test_constant_key;
             test "the function runs once per key" test_runs_once;
             test "structures in and out" test_structures;
             test "values of every set" test_every_set;
             test "a jit inside a jit" test_nested;
             test "results are fresh" test_fresh;
             test "an argument off 16 bytes" test_phase;
             test "a program with a loop" test_loop;
             stateful ~domains:2 ~count:10 ~steps:8
               "a call retraces iff its abstract value is new" cache_commands;
           ];
       ])
