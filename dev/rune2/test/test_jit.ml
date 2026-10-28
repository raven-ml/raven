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

(* Host arguments: their values do not change the key, and a function of host
   arguments gives host values, as eagerly. *)
let test_host_values () =
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
  equal ~msg:"on the host" bool true
    (Nx.Placement.equal (Nx.placement y) Nx.Host.on);
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

(* Staged loops: a loop's step is traced once and compiles as one loop of the
   program, which runs no eager kernel, whatever its number of trips. *)

module C = Nx_support.Counting
module Count = (val Nx.devices ~kernels:(module C) [ Nx_support.memory 0 ])

let count (type d) (x : (float, Nx.float32_elt, d) Nx.t) = Nx.place Count.on x

(* The occurrences of [word] in [s]. *)
let occurrences word s =
  let n = String.length word and k = ref 0 in
  for i = 0 to String.length s - n do
    if String.sub s i n = word then incr k
  done;
  !k

let test_staged_scan () =
  let traces = ref 0 in
  let step c r =
    incr traces;
    (Nx.add c (Nx.mul c r), Nx.mul_s c 2.)
  in
  let rows n =
    count (Nx.reshape [| n; 2 |] (Nx.linspace f32 (-0.01) 0.01 (2 * n)))
  in
  let xs = rows 100 in
  let init = count (v [| 0.5; -0. |]) in
  let f init xs = Rune.scan' ~f:step ~init xs in
  let printed xs =
    Format.asprintf "%a" Rune.pp_program
      (Rune.program
         T.(pair tensor tensor)
         T.(pair tensor tensor)
         (fun (a, b) -> f a b)
         (init, xs))
  in
  let short = printed (rows 3) and long = printed xs in
  equal ~msg:"one scan" int 1 (occurrences "scan" long);
  equal ~msg:"its step once" int
    (occurrences "Map" short)
    (occurrences "Map" long);
  let g = Rune.jit T.(tensor @-> tensor @-> returns (pair tensor tensor)) f in
  traces := 0;
  ignore (g init xs);
  equal ~msg:"one trace" int 1 !traces;
  traces := 0;
  C.reset ();
  let c, ys = g init xs in
  equal ~msg:"no retrace" int 0 !traces;
  equal ~msg:"no eager kernel" int 0 (C.calls ());
  let c', ys' = f init xs in
  equal ~msg:"carry" bits_t (bits c') (bits c);
  equal ~msg:"outputs" bits_t (bits ys') (bits ys)

let test_staged_iterate () =
  let traces = ref 0 in
  let until x =
    incr traces;
    let s = Nx.sum x in
    Nx.less s (Nx.scalar_like s 0.01)
  in
  let halve x = Nx.mul_s x 0.5 in
  let g = Rune.jit' (fun x -> Rune.iterate' ~max:50 ~until ~f:halve x) in
  let x = count (v [| 1.; 2.; 3. |]) in
  ignore (g x);
  traces := 0;
  C.reset ();
  let y = g x in
  equal ~msg:"no retrace" int 0 !traces;
  equal ~msg:"no eager kernel" int 0 (C.calls ());
  equal bits_t (bits (Rune.iterate' ~max:50 ~until ~f:halve x)) (bits y);
  invalid ~sub:"Rune.iterate': until is still false after max = 3 steps"
    (fun () ->
      Rune.jit' (fun x -> Rune.iterate' ~max:3 ~until ~f:halve x) x)

(* A custom rule's primal, a linear call's function and a remat's function
   compile with the program that calls them: a compiled call over them runs no
   eager kernel, under reverse mode too. The functions are exact, so the
   compiled values are eager's bits. *)
let test_constructs () =
  let tensor = T.tensor in
  let square x = Nx.mul x x in
  let square_vjp =
    Rune.custom_vjp tensor tensor (fun x ->
        (square x, fun ct -> Nx.mul ct (Nx.mul_s x 2.)))
  in
  let square_jvp =
    Rune.custom_jvp tensor tensor (fun x ->
        (square x, fun dx -> Nx.mul dx (Nx.mul_s x 2.)))
  in
  let triple =
    Rune.custom_transpose tensor tensor
      (fun v -> Nx.mul_s v 3.)
      ~transpose:(fun w -> Nx.mul_s w 3.)
  in
  let twice =
    Rune.remat T.(tensor @-> returns tensor) (fun x -> square (square x))
  in
  let x = count (v [| 0.5; -1.; 2. |]) in
  let compiled name f =
    let g = Rune.jit' f in
    let expected = f x in
    ignore (g x);
    C.reset ();
    let y = g x in
    equal ~msg:(name ^ ": no eager kernel") int 0 (C.calls ());
    equal ~msg:name bits_t (bits expected) (bits y)
  in
  let grad f x = Rune.grad' (fun x -> Nx.sum (f x)) x in
  compiled "custom_vjp" (fun x -> Nx.add (square_vjp x) x);
  compiled "custom_jvp" (fun x -> Nx.add (square_jvp x) x);
  compiled "custom_transpose" triple;
  compiled "remat" twice;
  compiled "grad of a custom_vjp" (grad square_vjp);
  compiled "grad of a custom_jvp" (grad square_jvp);
  compiled "grad of a custom_transpose" (grad triple);
  compiled "grad of a remat" (grad twice)

(* Donation

   A compiled call consumes the donated handles among its arguments: their
   donors die. A result takes a donated argument's memory where it reads the
   argument only at its own index and nothing else of the call reads it, and
   its values are the same either way. *)

let fresh xs = Nx.create f32 [| Array.length xs |] xs

(* [g (Nx.donate x)] and whether it wrote its result into [x]'s memory. *)
let in_place g x =
  let b = Nx.Repr.read x Nx_array.buffer in
  let y = g (Nx.donate x) in
  let a = Option.get (Nx.Repr.array y) in
  (Rig.Buffer.overlaps b (Nx_array.buffer a), y)

let dead x = invalid ~sub:"Rune.jit" (fun () -> ignore (Nx.to_array x))

let test_consumed () =
  let g = Rune.jit (pair_fn ()) Nx.add in
  let x = fresh [| 1.; 2. |] and y = fresh [| 3.; 4. |] in
  equal bits_t (bits (fresh [| 4.; 6. |])) (bits (g (Nx.donate x) y));
  dead x;
  equal ~msg:"the other argument lives" bits_t (bits (fresh [| 3.; 4. |]))
    (bits y)

let test_lending () =
  let xs = [| 1.; -0.; 2.5; Float.infinity |] in
  let check ~msg ~lent f =
    let expected = bits (f (fresh xs)) in
    let x = fresh xs in
    let took, y = in_place (Rune.jit' f) x in
    equal ~msg bits_t expected (bits y);
    equal ~msg:(msg ^ ": in place") bool lent took;
    dead x
  in
  check ~msg:"an elementwise result" ~lent:true (fun x -> Nx.neg (Nx.mul x x));
  check ~msg:"the argument itself" ~lent:true (fun x -> x);
  check ~msg:"a result that reads the argument at other indices" ~lent:false
    (fun x -> Nx.sub x (Nx.max x));
  check ~msg:"a reversed result" ~lent:false (fun x -> Nx.flip x);
  let pair = T.(tensor @-> returns (pair tensor tensor)) in
  let g = Rune.jit pair (fun x -> (Nx.neg x, Nx.add x x)) in
  let x = fresh xs in
  let b = Nx.Repr.read x Nx_array.buffer in
  let y, z = g (Nx.donate x) in
  let shares v = Rig.Buffer.overlaps b (Nx_array.buffer (Option.get (Nx.Repr.array v))) in
  equal ~msg:"two results that read it: neither in place"
    (Testable.pair bool bool)
    (false, false) (shares y, shares z);
  equal bits_t (bits (Nx.neg (fresh xs))) (bits y);
  equal bits_t (bits (Nx.add (fresh xs) (fresh xs))) (bits z)

(* A donor whose memory another value reads lends nothing: the call computes
   into fresh memory, and the reader keeps its values. *)
let test_shared () =
  let x = fresh [| 1.; 2. |] in
  let view = Nx.reshape [| 2; 1 |] x in
  ignore (Nx.Repr.array view);
  let took, y = in_place (Rune.jit' (fun x -> Nx.neg x)) x in
  equal ~msg:"in place" bool false took;
  equal bits_t (bits (fresh [| -1.; -2. |])) (bits y);
  equal ~msg:"the reader" bits_t (bits (Nx.create f32 [| 2; 1 |] [| 1.; 2. |]))
    (bits view)

(* An evaluated program may read a donated handle many times; the call still
   consumes it. *)
let test_evaluated_donation () =
  let f x =
    fst
      (Rune.scan'
         ~f:(fun c r -> (Nx.add c (Nx.mul x r), r))
         ~init:x (Nx.reshape [| 2; 1 |] x))
  in
  let expected = bits (f (fresh [| 1.; 2. |])) in
  let x = fresh [| 1.; 2. |] in
  equal bits_t expected (bits (Rune.jit' f (Nx.donate x)));
  dead x

type donation_case =
  | Donation_case :
      (float, 's) Nx.dtype * int array * operand * operand * expr
      -> donation_case

let pp_donation ppf (Donation_case (dt, s, x, y, e)) =
  Format.fprintf ppf "%a %a, x %a, y %a: %a" Nx.pp_dtype dt pp_shape s
    pp_view x.view pp_view y.view pp e

let donation_cases =
  let open Gen in
  (let* s = shape in
   let* e = expr ~rank:(Array.length s) in
   let* x = operand s and+ y = operand s in
   let+ wide = bool in
   let e = exact e in
   if wide then Donation_case (f64, s, x, y, e)
   else Donation_case (f32, s, x, y, e))
  |> with_pp pp_donation

(* A result that takes the argument's memory, and one that reads it along an
   axis and so does not. *)
let donation_examples =
  let plain = { view = Plain; values = [| 1.; -0.; 2.5 |] } in
  [
    Donation_case (f32, [| 3 |], plain, plain, Un (Neg, Bin (Mul, X, Y)));
    Donation_case (f32, [| 3 |], plain, plain, Bin (Sub, X, Un (Max 0, X)));
  ]

(* A call over a donated argument gives the bits of one over the argument
   kept, whether its result took the argument's memory or not. *)
let law_donation (Donation_case (dt, s, xo, yo, e)) =
  let f x y = eval x y e in
  let eager = outcome (fun () -> bits (f (make dt s xo) (make dt s yo))) in
  let x = make dt s xo and y = make dt s yo in
  let memory =
    if numel s = 0 then None else Some (Nx.Repr.read x Nx_array.buffer)
  in
  let result = ref None in
  let jitted =
    outcome (fun () ->
        let r = Rune.jit (pair_fn ()) f (Nx.donate x) y in
        result := Some r;
        bits r)
  in
  let took =
    match (memory, !result) with
    | Some b, Some r -> (
        match Nx.Repr.array r with
        | Some a -> Rig.Buffer.overlaps b (Nx_array.buffer a)
        | None -> false)
    | _ -> false
  in
  cover "a zero-size shape" (numel s = 0);
  cover "a view operand" (xo.view <> Plain);
  cover "the result takes the argument's memory" took;
  cover "the result is fresh" (Option.is_some !result && not took);
  equal (outcome_t bits_t) eager jitted

(* A contraction reads its [init] at its own index and its operands along
   their contracted axes: an [init] lends, and one that is also an operand
   does not. *)
let test_contract_init () =
  let p = Nx.Pattern.v "i j, j k -> i k | j" in
  let m () = Nx.create f32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let a = m () in
  let into w = Nx.contract ~init:w f32 p a a in
  let expected = bits (into (m ())) in
  let took, y = in_place (Rune.jit' into) (m ()) in
  equal bits_t expected (bits y);
  equal ~msg:"an init: in place" bool true took;
  let f x = Nx.contract ~init:x f32 p x x in
  let expected = bits (f (m ())) in
  let took, y = in_place (Rune.jit' f) (m ()) in
  equal bits_t expected (bits y);
  equal ~msg:"an init that is an operand: in place" bool false took

(* A scatter's target lends to the scatter, first. *)
let test_scatter_lends () =
  let p = Nx.create Nx.int64 [| 2 |] [| 2L; 0L |] in
  let u = Nx.create f32 [| 2 |] [| 9.; -0. |] in
  let f x = Nx.scatter ~axis:0 p u x in
  let m () = Nx.create f32 [| 3 |] [| 1.; 2.; 3. |] in
  let expected = bits (f (m ())) in
  let took, y = in_place (Rune.jit' f) (m ()) in
  equal bits_t expected (bits y);
  equal ~msg:"in place" bool true took

(* Checks: the first failing check in the order the function made them
   raises its exception, with its data at its first failing index in C
   order. *)

exception Small of int array * float
exception Signed of int array * int32 * bool

let checked x =
  Nx.check Nx.Ptree.tensor
    (Nx.less x (Nx.scalar_like x 3.))
    x
    (fun i d -> Small (i, Nx.item [] d));
  Nx.check
    Nx.Ptree.(pair tensor tensor)
    (Nx.greater_equal x (Nx.scalar_like x 0.))
    (Nx.cast Nx.int32 x, Nx.isnan x)
    (fun i (n, b) -> Signed (i, Nx.item [] n, Nx.item [] b));
  Nx.neg x

type failure = None_raised | Small_at of int list * float | Signed_at of int list * int32 * bool

let raised f =
  match f () with
  | _ -> None_raised
  | exception Small (i, d) -> Small_at (Array.to_list i, d)
  | exception Signed (i, n, b) -> Signed_at (Array.to_list i, n, b)

let failure =
  Testable.make
    ~pp:(fun ppf -> function
      | None_raised -> Format.pp_print_string ppf "no failure"
      | Small_at (i, d) ->
          Format.fprintf ppf "Small at [%s] %g"
            (String.concat "; " (List.map string_of_int i)) d
      | Signed_at (i, n, b) ->
          Format.fprintf ppf "Signed at [%s] %ld %b"
            (String.concat "; " (List.map string_of_int i)) n b)
    ~equal:( = )

let test_checks () =
  let g = Rune.jit' checked in
  let m xs = Nx.create f32 [| 2; 2 |] xs in
  let at xs = raised (fun () -> g (m xs)) in
  equal ~msg:"passing" bits_t (bits (Nx.neg (m [| 1.; 2.; -0.; 2.5 |])))
    (bits (g (m [| 1.; 2.; -0.; 2.5 |])));
  equal ~msg:"the first check's first failure" failure
    (Small_at ([ 1; 0 ], 4.)) (at [| 1.; 2.; 4.; 5. |]);
  equal ~msg:"the first check that fails, in order" failure
    (Small_at ([ 0; 1 ], 7.)) (at [| -1.; 7.; 0.; 0. |]);
  equal ~msg:"the second check, its data of two dtypes" failure
    (Signed_at ([ 1; 1 ], -2l, false)) (at [| 1.; 2.; 0.; -2.5 |]);
  equal ~msg:"one element" failure (Small_at ([ 0 ], 9.))
    (raised (fun () -> Rune.jit' checked (Nx.create f32 [| 1 |] [| 9. |])));
  let empty = Nx.create f32 [| 0 |] [||] in
  equal ~msg:"an empty check passes" bits_t (bits (Nx.neg empty))
    (bits (g empty));
  (* A check reads its data, so the argument it reads lends nothing. *)
  let x = m [| 1.; 2.; -0.; 2.5 |] in
  let took, y = in_place g x in
  equal ~msg:"read by a check: in place" bool false took;
  equal bits_t (bits (Nx.neg (m [| 1.; 2.; -0.; 2.5 |]))) (bits y)

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

(* Derived programs

   Under a transformation a call is a call of a program derived from its key's,
   once per derivation: its kind, which arguments have a tangent or lanes,
   their abstract values, and the maps the program reads. The model is the set
   of derivations met: a call under a new one derives its program, and reverse
   mode two more (the derivative's part known at the call, and the transpose of
   the rest), the derivative being forward mode's, as the program holds no
   custom rule; a call at a new shape traces. Each result is checked against
   eager's within the compiled sine's bound. *)

type mask = bool * bool
type how = Plain | Jvp of mask | Grad of mask | Map of mask
type derivation = { how : how; n : int }

let pp_mask ppf (a, b) =
  Format.fprintf ppf "%s"
    (match (a, b) with
    | true, true -> "x, y"
    | true, false -> "x"
    | false, true -> "y"
    | false, false -> "none")

let pp_derivation ppf d =
  (match d.how with
  | Plain -> Format.fprintf ppf "plain"
  | Jvp m -> Format.fprintf ppf "jvp along %a" pp_mask m
  | Grad m -> Format.fprintf ppf "grad of %a" pp_mask m
  | Map m -> Format.fprintf ppf "vmap of %a" pp_mask m);
  Format.fprintf ppf " at [%d]" d.n

let derivations =
  let open Gen in
  let mask = of_list [ (true, false); (false, true); (true, true) ] in
  (let* n = int_range 2 3 in
   let+ how =
     frequency
       [
         (1, constant Plain);
         (2, map (fun m -> Jvp m) mask);
         (2, map (fun m -> Grad m) mask);
         (2, map (fun m -> Map m) mask);
       ]
   in
   { how; n })
  |> with_pp pp_derivation

let derived_traced = Domain.DLS.new_key (fun () -> false)

let product x y =
  Domain.DLS.set derived_traced true;
  Nx.mul x (Nx.sin y)

(* [d] of [f] at [x] and [y], as one array. *)
let apply f d x y =
  let pair = T.(pair tensor tensor) in
  let flat = Nx.flatten in
  match d.how with
  | Plain -> f x y
  | Jvp (a, b) -> (
      let ones = Nx.ones f64 [| d.n |] in
      match (a, b) with
      | true, true ->
          snd (Rune.jvp pair T.tensor (fun (x, y) -> f x y) (x, y) (ones, ones))
      | true, false -> snd (Rune.jvp' (fun x -> f x y) x ones)
      | _ -> snd (Rune.jvp' (fun y -> f x y) y ones))
  | Grad (a, b) -> (
      let loss x y = Nx.sum (f x y) in
      match (a, b) with
      | true, true ->
          let gx, gy = Rune.grad pair (fun (x, y) -> loss x y) (x, y) in
          Nx.concatenate ~axis:0 [ gx; gy ]
      | true, false -> Rune.grad' (fun x -> loss x y) x
      | _ -> Rune.grad' (fun y -> loss x y) y)
  | Map (a, b) -> (
      let rows v = Nx.stack ~axis:0 [ v; Nx.neg v ] in
      match (a, b) with
      | true, true ->
          flat
            (Rune.vmap
               T.(pair tensor tensor @-> returns tensor)
               (fun (x, y) -> f x y)
               (rows x, rows y))
      | true, false -> flat (Rune.vmap' (fun x -> f x y) (rows x))
      | _ -> flat (Rune.vmap' (fun y -> f x y) (rows y)))

let derived_call g d =
  let x = Nx.linspace f64 0.5 1.5 d.n and y = Nx.linspace f64 (-1.) 2. d.n in
  Domain.DLS.set derived_traced false;
  let before = Rune.derivations () in
  let got = apply g d x y in
  let made = Rune.derivations () - before in
  let traced = Domain.DLS.get derived_traced in
  (* The compiled sine within 4 ulps, its derivative within its cosine's. *)
  equal ~msg:"value"
    (array (float_rel ~rel:1e-14 ~abs:1e-300))
    (Nx.to_array (apply product d x y))
    (Nx.to_array got);
  (traced, made)

module Seen = Set.Make (struct
  type t = string * mask * int

  let compare = compare
end)

let derivation_commands =
  let start () = (ref Seen.empty, ref [])
  and model (seen, shapes) d =
    let traced = not (List.mem d.n !shapes) in
    if traced then shapes := d.n :: !shapes;
    let meet k =
      let fresh = not (Seen.mem k !seen) in
      seen := Seen.add k !seen;
      Bool.to_int fresh
    in
    let made =
      match d.how with
      | Plain -> 0
      | Jvp m -> meet ("jvp", m, d.n)
      | Grad m -> meet ("jvp", m, d.n) + (2 * meet ("grad", m, d.n))
      | Map m -> meet ("vmap", m, d.n)
    in
    (traced, made)
  in
  let system () = Rune.jit T.(tensor @-> tensor @-> returns tensor) product in
  let fn = abstract "jit" in
  [
    command "start" (Gen.unit @-> makes fn) start system;
    command "call"
      (fn ^-> derivations @-> returns (pair bool int))
      model derived_call;
  ]

(* On two domains a call's derivations may be made by either domain's calls,
   so a call shows whether it traced, and its values. *)
let shared_commands =
  let start () = ref [] in
  let model shapes d =
    let traced = not (List.mem d.n !shapes) in
    if traced then shapes := d.n :: !shapes;
    traced
  in
  let system () = Rune.jit T.(tensor @-> tensor @-> returns tensor) product in
  let fn = abstract "jit" in
  [
    command "start" (Gen.unit @-> makes fn) start system;
    command "call"
      (fn ^-> derivations @-> returns bool)
      model
      (fun g d -> fst (derived_call g d));
  ]

(* A program reading a map's lane index, mapped under a map of another name
   inside that map and under the map itself inside another: equal masks, lanes
   and shapes, so only the name read tells the derivations apart. *)
let test_named_maps () =
  let a = Rune.axis () and b = Rune.axis () in
  let f x = Nx.add x (Nx.cast f64 (Rune.lane_index a x)) in
  let g = Rune.jit' f in
  let xs = Nx.reshape [| 2; 2; 2 |] (Nx.arange_f f64 0. 8. 1.) in
  let nested f ~outer ~inner =
    Rune.vmap' ~axis:outer (fun x -> Rune.vmap' ~axis:inner f x) xs
  in
  let check outer inner =
    equal
      ~msg:
        (if inner == a then "the map read inside" else "the map read outside")
      bits_t
      (bits (nested f ~outer ~inner))
      (bits (nested g ~outer ~inner))
  in
  check a b;
  check b a;
  check a b

(* A program that stacks a map's lanes is traced again where that map's lanes
   change. *)
let test_map_lanes () =
  let a = Rune.axis () in
  let traces = ref 0 in
  let f x = Nx.sum ~axes:[ 0 ] (Rune.lanes a x) in
  let g =
    Rune.jit' (fun x ->
        incr traces;
        f x)
  in
  List.iter
    (fun n ->
      let m = Float.of_int (3 * n) in
      let xs = Nx.reshape [| n; 3 |] (Nx.arange_f f64 0. m 1.) in
      equal bits_t
        (bits (Rune.vmap' ~axis:a f xs))
        (bits (Rune.vmap' ~axis:a g xs)))
    [ 2; 3; 2 ];
  equal ~msg:"one trace per lane count" int 2 !traces

(* The function runs once per key, whatever is derived from it. *)
let test_once_per_key () =
  let traces = ref 0 in
  let g =
    Rune.jit' (fun x ->
        incr traces;
        Nx.mul x (Nx.sin x))
  in
  let x = Nx.linspace f64 0. 1. 3 in
  let xs = Nx.reshape [| 2; 3 |] (Nx.linspace f64 0. 1. 6) in
  let sum f x = Nx.sum (f x) in
  ignore (g x);
  ignore (Rune.jvp' g x x);
  ignore (Rune.grad' (sum g) x);
  ignore (Rune.grad' (sum (Rune.grad' (sum g))) x);
  ignore (Rune.vmap' (Rune.grad' (sum g)) xs);
  ignore (Rune.jit' (Rune.grad' (sum g)) x);
  ignore (Rune.jacfwd' (Rune.grad' (sum g)) x);
  equal ~msg:"one key" int 1 !traces;
  ignore (Rune.vmap' g xs);
  equal ~msg:"a map over arguments of the key's forms" int 1 !traces

(* Under an enclosing jit, a derived call is staged: the printed program calls
   no other. *)
let test_inlined () =
  let g = Rune.jit' (fun x -> Nx.mul x (Nx.sin x)) in
  let f x = Rune.grad' (fun x -> Nx.sum (g x)) x in
  let x = Nx.linspace f64 0. 1. 3 in
  let printed =
    Format.asprintf "%a" Rune.pp_program (Rune.program T.tensor T.tensor f x)
  in
  equal ~msg:"calls" int 0 (occurrences "call" printed);
  equal (array (float_rel ~rel:1e-14 ~abs:1e-300))
    (Nx.to_array (Rune.grad' (fun x -> Nx.sum (Nx.mul x (Nx.sin x))) x))
    (Nx.to_array (Rune.jit' f x))

(* A scan's step inlines a call over its variables: the map around the scan
   answers the lane index that the call's program reads, as it answers the
   step's own. *)
let test_inlined_in_step () =
  let a = Rune.axis () in
  let lane x = Nx.add x (Nx.cast f64 (Rune.lane_index a x)) in
  let g = Rune.jit' lane in
  let xs = Nx.reshape [| 2; 3 |] (Nx.linspace f64 0. 1. 6) in
  let run f x =
    fst
      (Rune.scan' ~f:(fun c r -> (Nx.mul (f c) r, Nx.neg c)) ~init:x
         (Nx.reshape [| 3; 1 |] x))
  in
  equal bits_t
    (bits (Rune.vmap' ~axis:a (run lane) xs))
    (bits (Rune.vmap' ~axis:a (run g) xs))

(* Compositions of transformations around a compiled call give eager's values,
   the compiled sine and cosine within their bounds. *)
let test_compositions () =
  let close ~msg a b =
    equal ~msg
      (array (float_rel ~rel:1e-13 ~abs:1e-13))
      (Nx.to_array (Nx.flatten a))
      (Nx.to_array (Nx.flatten b))
  in
  let f x = Nx.mul (Nx.sin x) (Nx.cos (Nx.mul x x)) in
  let g = Rune.jit' f in
  let sum h x = Nx.sum (h x) in
  let x = Nx.linspace f64 (-1.) 1.5 4 in
  let xs = Nx.reshape [| 3; 4 |] (Nx.linspace f64 (-1.) 2. 12) in
  let both msg t = close ~msg (t f) (t g) in
  both "grad of grad" (fun f -> Rune.grad' (sum (Rune.grad' (sum f))) x);
  both "jvp of grad" (fun f -> snd (Rune.jvp' (Rune.grad' (sum f)) x x));
  both "grad of jvp" (fun f ->
      Rune.grad' (fun x -> Nx.sum (snd (Rune.jvp' f x x))) x);
  both "jacfwd of grad" (fun f -> Rune.jacfwd' (Rune.grad' (sum f)) x);
  both "jacrev" (fun f -> Rune.jacrev' f x);
  both "vmap of grad" (fun f -> Rune.vmap' (Rune.grad' (sum f)) xs);
  both "grad of vmap" (fun f -> Rune.grad' (sum (Rune.vmap' f)) xs);
  both "vjp applied twice" (fun f ->
      let _, pb = Rune.vjp' f x in
      Nx.add (pb x) (pb (Nx.neg x)));
  both "linearize" (fun f ->
      let _, l = Rune.linearize T.tensor T.tensor f x in
      Nx.add (l x) (l (Nx.ones f64 [| 4 |])));
  both "linear_transpose" (fun f ->
      let l =
        Rune.linear_transpose T.tensor T.tensor (fun v -> Nx.mul v (f x)) x
      in
      l x);
  let remat = Rune.remat T.(tensor @-> returns tensor) f in
  close ~msg:"grad of a remat in a jit" (Rune.grad' (sum f) x)
    (Rune.grad' (sum (Rune.jit' remat)) x);
  (* Lanes and lane indices through a derived transpose. *)
  let a = Rune.axis () in
  let laned x =
    Nx.mul (f x)
      (Nx.add (Nx.cast f64 (Rune.lane_index a x)) (Nx.sum (Rune.lanes a x)))
  in
  let lg = Rune.jit' laned in
  close ~msg:"vmap of grad, reading the map"
    (Rune.vmap' ~axis:a (Rune.grad' (sum laned)) xs)
    (Rune.vmap' ~axis:a (Rune.grad' (sum lg)) xs);
  close ~msg:"grad of vmap, reading the map"
    (Rune.grad' (sum (Rune.vmap' ~axis:a laned)) xs)
    (Rune.grad' (sum (Rune.vmap' ~axis:a lg)) xs);
  (* An addition counts once per execution, under each derivation. *)
  let t = Rune.Total.make () in
  let adding f x =
    Rune.Total.add t (Nx.sum (Nx.mul x x));
    f x
  in
  let collected h =
    let zero = Nx.scalar f64 0. in
    snd (Rune.Total.collect t ~zero (fun () -> ignore (h ())))
  in
  let totals msg h =
    close ~msg (collected (fun () -> h (adding f)))
      (collected (fun () -> h (Rune.jit' (adding f))))
  in
  totals "a total under jvp" (fun f -> Rune.jvp' f x x);
  totals "a total under grad" (fun f -> Rune.grad' (sum f) x);
  totals "a total under vmap" (fun f -> Rune.vmap' f xs)

(* A remat stages every operation its function applies, a compiled call's over
   its captures included: evaluated, the remat runs their kernels. *)
let test_remat_stages () =
  let c = count (v [| 0.5; 2. |]) in
  let g = Rune.jit' (fun c -> Nx.mul c c) in
  let r = Rune.remat T.(tensor @-> returns tensor) (fun x -> Nx.add x (g c)) in
  let x = count (v [| 1.; -1. |]) in
  ignore (r x);
  C.reset ();
  let y = r x in
  equal ~msg:"the call's kernel and the addition's" int 2 (C.calls ());
  equal bits_t (bits (Nx.add x (Nx.mul c c))) (bits y)

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
             test "host arguments" test_host_values;
             test "a jit inside a jit" test_nested;
             test "results are fresh" test_fresh;
             test "an argument off 16 bytes" test_phase;
             test "a program with a loop" test_loop;
             test "a donated argument is consumed" test_consumed;
             test "a result takes a donated argument's memory" test_lending;
             test "a shared donor lends nothing" test_shared;
             test "an evaluated program consumes its donations"
               test_evaluated_donation;
             prop ~examples:donation_examples "donation keeps the values"
               donation_cases law_donation;
             test "a contraction's init lends, an operand does not"
               test_contract_init;
             test "a scatter's target lends" test_scatter_lends;
             test "checks raise as eagerly" test_checks;
             stateful ~domains:2 ~count:10 ~steps:8
               "a call retraces iff its abstract value is new" cache_commands;
           ];
         group "constructs"
           [
             test "a scan's step traces once and compiles one loop"
               test_staged_scan;
             test "an iterate compiles one loop" test_staged_iterate;
             test "custom rules and remats compile with their program"
               test_constructs;
           ];
         group "derived"
           [
             stateful ~count:20 ~steps:8
               "a derived program is made iff its derivation is new"
               derivation_commands;
             stateful ~domains:2 ~count:10 ~steps:8
               "derivations on two domains trace once per key" shared_commands;
             test "a program reading a map's name derives apart"
               test_named_maps;
             test "a program stacking lanes retraces per lane count"
               test_map_lanes;
             test "the function runs once per key under every derivation"
               test_once_per_key;
             test "a derived call is staged under an enclosing jit"
               test_inlined;
             test "a step inlines a call over its variables"
               test_inlined_in_step;
             test "compositions around a call give eager's values"
               test_compositions;
             test "a remat stages a call over its captures" test_remat_stages;
           ];
       ])
