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
let v64 xs = Nx.create f64 [| Array.length xs |] xs

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

(* Bitcasts

   A compiled bitcast gives eager's bits for every pair of dtypes, a narrower
   dtype splitting each element along a new last axis and a wider one joining
   the last axis's, over operands of random bits in every view, from the
   start of their storage or one element into it, inside a byte for a sub-byte
   dtype. It compiles whole: no call is evaluated. *)

type bitcast_case =
  | Bitcast_case :
      ('v, 's) Nx.dtype * ('w, 'r) Nx.dtype * view * bool * int
      -> bitcast_case

(* [random_bits dt s st] is a value of [dt] and shape [s] of random bits. *)
let random_bits (type v s) (dt : (v, s) Nx.dtype) s st : (v, s, Nx.host) Nx.t =
  let n = numel s in
  let byte () = Random.State.int st 256 in
  match Nx.Dtype.bits dt with
  | 1 ->
      Nx.bitcast dt
        (Nx.create Nx.bit s (Array.init n (fun _ -> Random.State.bool st)))
  | 4 ->
      Nx.bitcast dt
        (Nx.create Nx.uint4 s (Array.init n (fun _ -> Random.State.int st 16)))
  | 8 -> Nx.bitcast dt (Nx.create Nx.uint8 s (Array.init n (fun _ -> byte ())))
  | b ->
      let w = b / 8 in
      Nx.bitcast dt
        (Nx.create Nx.uint8 (Array.append s [| w |])
           (Array.init (n * w) (fun _ -> byte ())))

(* [viewed v ~offset s make] is a value of shape [s] as the view [v] of one
   [make] gives: its last axis stepped by 2 where strided, its first broadcast,
   its storage's first element skipped where [offset]. *)
let viewed v ~offset s make =
  let r = Array.length s in
  let make s =
    if not offset then make s
    else
      let n = numel s in
      Nx.reshape s (Nx.slice [ Nx.R (1, n + 1) ] (make [| n + 1 |]))
  in
  match v with
  | Flipped -> Nx.flip (make s)
  | Strided when r > 0 ->
      let s' = Array.copy s in
      s'.(r - 1) <- 2 * s.(r - 1);
      let step i = if i = r - 1 then Nx.Rs (0, s'.(i), 2) else Nx.A in
      Nx.slice (List.init r step) (make s')
  | Broadcast when r > 0 && s.(0) > 0 ->
      let s1 = Array.copy s in
      s1.(0) <- 1;
      Nx.broadcast_to s (make s1)
  | Plain | Strided | Broadcast -> make s

(* [raw x] is [x]'s shape and each element's bits, read through a bitcast to
   the unsigned integers of its width, a complex128's as two [uint64]s. *)
let raw (type v s d) (x : (v, s, d) Nx.t) =
  let each f y = Array.map f (Nx.to_array y) in
  let elements =
    match Nx.Dtype.bits (Nx.dtype x) with
    | 1 -> each (fun b -> if b then "1" else "0") (Nx.bitcast Nx.bit x)
    | 4 -> each string_of_int (Nx.bitcast Nx.uint4 x)
    | 8 -> each string_of_int (Nx.bitcast Nx.uint8 x)
    | 16 -> each string_of_int (Nx.bitcast Nx.uint16 x)
    | 32 -> each (Printf.sprintf "%lx") (Nx.bitcast Nx.uint32 x)
    | _ -> each (Printf.sprintf "%Lx") (Nx.bitcast Nx.uint64 x)
  in
  (Nx.shape x, elements)

let views = [ Plain; Flipped; Strided; Broadcast ]

(* Every pair of dtypes, the views and offsets in turn. *)
let bitcast_examples =
  List.concat_map
    (fun src -> List.map (fun dt -> (src, dt)) Nx.Dtype.all)
    Nx.Dtype.all
  |> List.mapi (fun i (Nx.Dtype.Any src, Nx.Dtype.Any dt) ->
      Bitcast_case (src, dt, List.nth views (i mod 4), i / 4 mod 2 = 1, i))

let bitcast_cases =
  let open Gen in
  (let* (Nx.Dtype.Any src) = of_list Nx.Dtype.all
   and+ (Nx.Dtype.Any dt) = of_list Nx.Dtype.all in
   let+ view = of_list views
   and+ offset = bool
   and+ seed = int_range 0 1_000_000 in
   Bitcast_case (src, dt, view, offset, seed))
  |> with_pp (fun ppf (Bitcast_case (src, dt, v, offset, seed)) ->
      Format.fprintf ppf "%a to %a, %a%s, seed %d" Nx.pp_dtype src Nx.pp_dtype
        dt pp_view v
        (if offset then " at an offset" else "")
        seed)

let law_bitcast (Bitcast_case (src, dt, view, offset, seed)) =
  let st = Random.State.make [| seed |] in
  let bs = Nx.Dtype.bits src and bt = Nx.Dtype.bits dt in
  let dim () = Random.State.int st 4 in
  let s =
    if bt > bs then
      Array.append
        (Array.init (Random.State.int st 3) (fun _ -> dim ()))
        [| bt / bs |]
    else Array.init (Random.State.int st 4) (fun _ -> dim ())
  in
  let x = viewed view ~offset s (fun s -> random_bits src s st) in
  let sub_byte d = Nx.Dtype.bits d < 8 in
  cover "a narrowing" (bt < bs);
  cover "a widening" (bt > bs);
  cover "one width" (bt = bs);
  cover "a sub-byte narrowing" (bt < bs && sub_byte dt);
  cover "a sub-byte widening" (bt > bs && sub_byte src);
  cover "between sub-byte dtypes" (bt <> bs && sub_byte src && sub_byte dt);
  cover "a complex dtype"
    (bt <> bs && (Nx.Dtype.is Complex src || Nx.Dtype.is Complex dt));
  cover "a zero-size operand" (numel s = 0);
  cover "a 0-d operand" (Array.length s = 0);
  cover "a sub-byte operand inside a byte" (offset && bs < 8 && numel s > 0);
  List.iter
    (fun v ->
      cover
        (Format.asprintf "a %a operand" pp_view v)
        (view = v && numel s > 1))
    views;
  let before = Rune.evaluations () in
  let y = Rune.jit' (Nx.bitcast dt) x in
  equal ~msg:"compiled" int 0 (Rune.evaluations () - before);
  equal codes_t (raw (Nx.bitcast dt x)) (raw y)

(* Each nx.quant format's decode compiles whole and gives eager's values. *)
let test_quant_decodes () =
  let st = Random.State.make [| 7 |] in
  let bytes s =
    Nx.create Nx.uint8 s
      (Array.init (numel s) (fun _ -> Random.State.int st 256))
  in
  let n = 3 and k = 256 in
  let compiled name f =
    let before = Rune.evaluations () in
    let y = f () in
    equal ~msg:(name ^ ": compiled") int 0 (Rune.evaluations () - before);
    y
  in
  let blocks name make (values, size) =
    let b = bytes [| n; k / values * size |] in
    let decode b = Nx_quant.dequant f32 (make b) in
    let y = compiled name (fun () -> Rune.jit' decode b) in
    equal ~msg:name bits_t (bits (decode b)) (bits y)
  in
  blocks "q8_0" Nx_quant.q8_0 (32, 34);
  blocks "q4_k" Nx_quant.q4_k (256, 144);
  blocks "q6_k" Nx_quant.q6_k (256, 210);
  blocks "mxfp4_blocks" Nx_quant.mxfp4_blocks (32, 17);
  let codes = bytes [| n; k / 2 |] and scales = bytes [| n; k / 32 |] in
  let decode scales codes =
    Nx_quant.dequant f32 (Nx_quant.mxfp4 ~scales codes)
  in
  let y =
    compiled "mxfp4" (fun () -> Rune.jit (pair_fn ()) decode scales codes)
  in
  equal ~msg:"mxfp4" bits_t (bits (decode scales codes)) (bits y)

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
    { name = "log"; f = Nx.log; bound = (1 + 1, 1 + 2) };
    { name = "sin"; f = Nx.sin; bound = (4 + 2, 4 + 2) };
    { name = "cos"; f = Nx.cos; bound = (9 + 2, 9 + 2) };
    { name = "tan"; f = Nx.tan; bound = (19 + 4, 19 + 2) };
    { name = "asin"; f = Nx.asin; bound = (5 + 2, 9 + 2) };
    { name = "acos"; f = Nx.acos; bound = (5 + 2, 9 + 2) };
    { name = "atan"; f = Nx.atan; bound = (2 + 2, 7 + 2) };
    { name = "sinh"; f = Nx.sinh; bound = (23 + 3, 23 + 2) };
    { name = "cosh"; f = Nx.cosh; bound = (10 + 2, 10 + 2) };
    { name = "tanh"; f = Nx.tanh; bound = (18 + 2, 18 + 2) };
    { name = "expm1"; f = Nx.expm1; bound = (23 + 1, 23 + 2) };
    { name = "log1p"; f = Nx.log1p; bound = (5 + 1, 5 + 2) };
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

(* A function searched at a width gives eager's bits, as do the programs
   derived from it, over operands drawn to break it: -0., infinities, NaN.
   The operands share a shape, so each kernel is searched once. *)
let searched_f =
  Rune.jit ~beam:1 ~parallel:2 (pair_fn ()) (fun x y -> Nx.sub (Nx.mul x x) y)

let searched_cases =
  let open Gen in
  let element =
    frequency
      [
        (3, map (fun i -> pool.(i)) (int_range 0 (Array.length pool - 1)));
        (2, float_range (-4.) 4.);
      ]
  in
  let four = array ~size:(constant 4) element in
  pair four four

let law_searched (xs, ys) =
  let has p = Array.exists p xs || Array.exists p ys in
  cover "a -0." (has (fun e -> e = 0. && 1. /. e < 0.));
  cover "an infinity" (has (fun e -> Float.abs e = Float.infinity));
  cover "a NaN" (has Float.is_nan);
  let x = v xs and y = v ys in
  let f x y = Nx.sub (Nx.mul x x) y in
  equal bits_t (bits (f x y)) (bits (searched_f x y));
  equal ~msg:"its gradient" bits_t
    (bits (Rune.grad' (fun x -> Nx.sum (f x y)) x))
    (bits (Rune.grad' (fun x -> Nx.sum (searched_f x y)) x))

(* A constant on another device that the function places where it computes is
   copied there: the call compiles and gives eager's bits, at every dtype and
   for views of the constant, an offset one included. A result that is such a
   constant, unplaced, lies on its device: the call evaluates. *)
module Away = (val Nx.devices [ Nx_support.memory 2 ])

type placed_case =
  | Placed_case :
      ('v, 's) Nx.dtype * int array * operand * bool
      -> placed_case

let placed_cases =
  let open Gen in
  (let* s = shape in
   let* o = operand s in
   let+ (Nx.Dtype.Any dt) = of_list Nx.Dtype.all and+ offset = bool in
   Placed_case (dt, s, o, offset))
  |> with_pp (fun ppf (Placed_case (dt, s, o, offset)) ->
      Format.fprintf ppf "%a %a, %a%s" Nx.pp_dtype dt pp_shape s pp_view o.view
        (if offset then ", one element in" else ""))

(* [o]'s value of [dt] and [s], as a view of an array on [Away], one element
   into a longer one where [offset]. *)
let away (type v s) (dt : (v, s) Nx.dtype) s o ~offset : (v, s, Away.d) Nx.t =
  let r = Array.length s in
  let create s vs =
    let x = Nx.cast dt (Nx.create Nx.float64 s vs) in
    if not offset then Nx.place Away.on x
    else
      let n = numel s in
      let longer = Nx.concatenate ~axis:0 [ Nx.flatten x; Nx.flatten x ] in
      Nx.reshape s (Nx.slice [ Nx.R (1, n + 1) ] (Nx.place Away.on longer))
  in
  match o.view with
  | Plain -> create s o.values
  | Flipped -> Nx.flip (create s o.values)
  | Strided when r = 0 -> create s (Array.sub o.values 0 1)
  | Strided ->
      let s' = Array.copy s in
      s'.(r - 1) <- 2 * s.(r - 1);
      let every i = if i = r - 1 then Nx.Rs (0, s'.(i), 2) else Nx.A in
      Nx.slice (List.init r every) (create s' o.values)
  | Broadcast when r = 0 -> create s o.values
  | Broadcast ->
      let s1 = Array.copy s in
      s1.(0) <- 1;
      if s.(0) = 0 then create s [||]
      else Nx.broadcast_to s (create s1 o.values)

let law_placed (Placed_case (dt, s, o, offset)) =
  cover "a sub-byte dtype" (Nx.Dtype.bits dt < 8);
  cover "a view" (o.view <> Plain);
  cover "an offset constant" (offset && numel s > 1);
  cover "a zero-size shape" (numel s = 0);
  let c = away dt s o ~offset in
  let f x = (Nx.neg x, Nx.place Nx.Host.on c) in
  let x = Nx.ones f32 [| 2 |] in
  let calls = Rune.evaluations () in
  let _, y =
    Rune.jit T.(tensor @-> returns (pair tensor tensor)) f x
  in
  equal ~msg:"compiled" int 0 (Rune.evaluations () - calls);
  equal codes_t (codes (snd (f x))) (codes y)

(* A set of the host and a memory device: a program over it computes on the
   host. *)
module Pair = (val Nx.devices [ Rig.host; Nx_support.memory 3 ])

let test_constant_result () =
  let on r x = Nx.place (Nx.Placement.device Pair.v r) x in
  let c = on (Nx_support.memory 3) (v [| 0.5; -2. |]) in
  let x = on Rig.host (v [| 1.; 2. |]) in
  let calls = Rune.evaluations () in
  let y, d =
    Rune.jit T.(tensor @-> returns (pair tensor tensor))
      (fun x -> (Nx.neg x, c)) x
  in
  equal ~msg:"evaluated" int 1 (Rune.evaluations () - calls);
  equal bits_t (bits (v [| -1.; -2. |])) (bits y);
  equal ~msg:"on its device" bool true
    (Nx.Placement.equal (Nx.placement d) (Nx.placement c));
  equal bits_t (bits (v [| 0.5; -2. |])) (bits d)

(* A jit inside a jit is staged into the outer program, a call over constants
   included. *)
let test_nested () =
  let inner = Rune.jit' (fun x -> Nx.mul x x) in
  let c = v [| 3.; -0. |] in
  let outer = Rune.jit' (fun x -> Nx.add (inner x) (inner c)) in
  let x = v [| 1.; 2. |] in
  equal bits_t (bits (Nx.add (Nx.mul x x) (Nx.mul c c))) (bits (outer x))

(* An integer running sum wraps at its type: over the greatest value [m] twice,
   it is [m] then [m + m] wrapped, whose maximum is [m]. *)
let test_wrapped_max () =
  let case (Nx.Dtype.Any dt) =
    match Nx.Dtype.kind dt with
    | Signed | Unsigned ->
        let m = Nx.Dtype.max_value dt in
        let x = Nx.broadcast_to [| 2 |] (Nx.create dt [| 1 |] [| m |]) in
        let f x = Nx.max ~axes:[ 0 ] (Nx.cumsum ~axis:0 x) in
        equal
          ~msg:(Format.asprintf "%a" Nx.pp_dtype dt)
          codes_t
          (codes (Nx.scalar dt m))
          (codes (Rune.jit' f x))
    | Float | Complex | Boolean -> ()
  in
  List.iter case Nx.Dtype.all

(* Fresh results: compiled and evaluated, a result has memory of its own, apart
   from the arguments', the captures' and the other results'. *)
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
  equal bits_t (bits x) (bits looped);
  let c = v [| 5.; -0. |] in
  let rooted =
    Rune.jit
      T.(tensor @-> returns (pair tensor tensor))
      (fun x ->
        let r =
          Rune.root T.tensor ~residual:(fun y -> Nx.sub y x) (fun () -> x)
        in
        (r, c))
  in
  let r, k = rooted x in
  let held = Nx.Repr.read c Nx_array.buffer in
  let held_by y =
    Rig.Buffer.overlaps held (Nx_array.buffer (Option.get (Nx.Repr.array y)))
  in
  equal ~msg:"evaluated, a captured result" bool false (held_by k);
  equal bits_t (bits x) (bits r);
  equal bits_t (bits c) (bits k);
  (* A result that is a view of an argument or a capture. *)
  let viewed =
    Rune.jit
      T.(tensor @-> returns (pair tensor (pair tensor tensor)))
      (fun x ->
        let r =
          Rune.root T.tensor ~residual:(fun y -> Nx.sub y x) (fun () -> x)
        in
        (r, (Nx.reshape [| 2; 1 |] x, Nx.reshape [| 1; 2 |] c)))
  in
  let _, (vx, vc) = viewed x in
  equal ~msg:"evaluated, a view of an argument" bool false (shares vx);
  equal ~msg:"evaluated, a view of a capture" bool false (held_by vc);
  equal bits_t (bits x) (bits (Nx.reshape [| 2 |] vx));
  equal bits_t (bits c) (bits (Nx.reshape [| 2 |] vc));
  (* A capture a construct's function returns: a scan's carry, a branch. *)
  let carried =
    Rune.jit
      T.(tensor @-> returns (pair tensor (pair tensor tensor)))
      (fun x ->
        let r =
          Rune.root T.tensor ~residual:(fun y -> Nx.sub y x) (fun () -> x)
        in
        let s =
          fst
            (Rune.scan' ~f:(fun _ r -> (c, r)) ~init:x (Nx.reshape [| 2; 1 |] x))
        in
        let b =
          Rune.cond' (Nx.greater_s (Nx.sum x) 0.) (fun () -> c) (fun () -> x)
        in
        (r, (s, b)))
  in
  let _, (s, b) = carried x in
  equal ~msg:"evaluated, a scan's captured carry" bool false (held_by s);
  equal ~msg:"evaluated, a captured branch" bool false (held_by b);
  equal bits_t (bits c) (bits s);
  equal bits_t (bits c) (bits b);
  (* A result and a view of it. *)
  let twice =
    Rune.jit
      T.(tensor @-> returns (pair tensor tensor))
      (fun x ->
        let r =
          Rune.root T.tensor ~residual:(fun y -> Nx.sub y x) (fun () ->
              Nx.add_s x 1.)
        in
        (r, Nx.reshape [| 2; 1 |] r))
  in
  let r, w = twice x in
  let mem y = Nx_array.buffer (Option.get (Nx.Repr.array y)) in
  equal ~msg:"evaluated, a view of a result" bool false
    (Rig.Buffer.overlaps (mem r) (mem w));
  equal bits_t (bits (Nx.add_s x 1.)) (bits r);
  equal bits_t (bits r) (bits (Nx.reshape [| 2 |] w))

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
   argument only at its own index and the stores have an order, and its values
   are the same either way. *)

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
  equal ~msg:"two results that read it: the first in place"
    (Testable.pair bool bool)
    (true, false) (shares y, shares z);
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

(* Two donated leaves [(w, v)] and two results: [v' = e1 (w, v)], and [w' =
   e2 (w, v')] where [through], reading the leaf [v] through [v']'s memory, or
   [e2 (w, v)] apart, in the order [(w', v')], or [(v', w')] where [swap];
   and a third result [e3 (w', v')], which reads both. *)
type state_case =
  | State_case : {
      dt : (float, 's) Nx.dtype;
      s : int array;
      w : operand;
      v : operand;
      e1 : expr;
      e2 : expr;
      e3 : expr;
      through : bool;
      swap : bool;
    }
      -> state_case

let pp_state ppf (State_case c) =
  Format.fprintf ppf "%a %a, w %a, v %a: v' = %a, w' = %a over %s, %a%s"
    Nx.pp_dtype c.dt pp_shape c.s pp_view c.w.view pp_view c.v.view pp c.e1 pp
    c.e2
    (if c.through then "v'" else "v")
    pp c.e3
    (if c.swap then ", (v', w')" else "")

let state_cases =
  let open Gen in
  (let* s = shape in
   let rank = Array.length s in
   let* e1 = expr ~rank and+ e2 = expr ~rank and+ e3 = expr ~rank in
   let* w = operand s and+ v = operand s in
   let+ through = bool and+ swap = bool and+ wide = bool in
   let e1 = exact e1 and e2 = exact e2 and e3 = exact e3 in
   if wide then State_case { dt = f64; s; w; v; e1; e2; e3; through; swap }
   else State_case { dt = f32; s; w; v; e1; e2; e3; through; swap })
  |> with_pp pp_state

let state_step (State_case c) w v =
  let v' = eval w v c.e1 in
  let w' = eval w (if c.through then v' else v) c.e2 in
  let third = eval w' v' c.e3 in
  if c.swap then (v', (w', third)) else (w', (v', third))

let state_examples =
  let plain = { view = Plain; values = [| 1.; -0.; 2.5; -3. |] } in
  let case ?(through = true) ?(swap = false) ?(e3 = Bin (Add, X, Y)) e1 e2 =
    State_case
      { dt = f32; s = [| 4 |]; w = plain; v = plain; e1; e2; e3; through; swap }
  in
  [
    (* A momentum step, in either order. *)
    case
      (Bin (Add, Bin (Mul, Y, K 0.5), Bin (Mul, Bin (Sub, X, K 2.), K 2.)))
      (Bin (Sub, X, Bin (Mul, Y, K 0.5)));
    case ~swap:true
      (Bin (Add, Bin (Mul, Y, K 0.5), Bin (Mul, Bin (Sub, X, K 2.), K 2.)))
      (Bin (Sub, X, Bin (Mul, Y, K 0.5)));
    (* Results that read each other's leaf before their stores. *)
    case ~through:false (Bin (Add, X, Y)) (Bin (Sub, Y, X));
    (* A leaf returned as it is. *)
    case (Bin (Add, Y, K 1.)) X;
    case ~swap:true (Bin (Add, Y, K 1.)) (Bin (Mul, X, Y));
  ]

(* A step over two donated leaves gives the bits of one over the leaves kept,
   whichever results take their memory. *)
let law_states (State_case c as case) =
  let two = T.(pair tensor tensor) in
  let step (w, v) = state_step case w v in
  let g = Rune.jit T.(two @-> returns (pair tensor two)) step in
  let make () = (make c.dt c.s c.w, make c.dt c.s c.v) in
  let both (a, (b, d)) = (bits a, (bits b, bits d)) in
  let t = outcome_t Testable.(pair bits_t (pair bits_t bits_t)) in
  let eager = outcome (fun () -> both (step (make ()))) in
  let w, v = make () in
  let memory x =
    if numel c.s = 0 then None else Some (Nx.Repr.read x Nx_array.buffer)
  in
  let leaves = List.filter_map memory [ w; v ] in
  let result = ref None in
  let calls = Rune.evaluations () in
  let jitted =
    outcome (fun () ->
        let r = g (Nx.donate w, Nx.donate v) in
        result := Some r;
        both r)
  in
  (* The stores have an order: the call compiles. *)
  (match jitted with
  | Value _ -> equal ~msg:"compiled" int 0 (Rune.evaluations () - calls)
  | Raised _ -> ());
  let lent x =
    match Nx.Repr.array x with
    | Some a ->
        List.exists (fun b -> Rig.Buffer.overlaps b (Nx_array.buffer a)) leaves
    | None -> false
  in
  let took =
    match !result with Some (a, (b, _)) -> [ lent a; lent b ] | None -> []
  in
  cover "a zero-size shape" (numel c.s = 0);
  cover "a view operand" (c.w.view <> Plain || c.v.view <> Plain);
  cover "both results take a leaf's memory" (took = [ true; true ]);
  cover "one result takes a leaf's memory"
    (took = [ true; false ] || took = [ false; true ]);
  equal t eager jitted

(* A momentum step's results each take a leaf's memory, call after call: the
   parameters' update reads the velocity's from where it was stored, whatever
   the order of the results, at every dtype. Where each result reads the other's
   leaf before its store, no order of the stores exists, and the later result
   takes fresh memory. [run ~msg ~start ~same step lends] runs [step] compiled
   over the donated leaves [start ()] three times against eager, [same] its
   results' equality, and [lends] whether each result took a leaf's memory. *)
let test_states () =
  let address x = Nx.Repr.read x Nx_array.buffer in
  let run ~msg ~start ~same step lends =
    let two = T.(pair tensor tensor) in
    let g = Rune.jit T.(two @-> returns two) step in
    let compiled = ref (start ()) and eager = ref (start ()) in
    for _ = 1 to 3 do
      let leaves = [ address (fst !compiled); address (snd !compiled) ] in
      compiled := g (Nx.donate (fst !compiled), Nx.donate (snd !compiled));
      eager := step !eager;
      let took x =
        List.exists (fun b -> Rig.Buffer.overlaps b (address x)) leaves
      in
      equal ~msg:(msg ^ ": in place") (Testable.pair bool bool) lends
        (took (fst !compiled), took (snd !compiled));
      same (fst !eager) (fst !compiled);
      same (snd !eager) (snd !compiled)
    done
  in
  let floats ~msg a b = equal ~msg bits_t (bits a) (bits b) in
  let start () =
    (fresh [| 1.; -0.; 2.5; -3. |], fresh [| 0.5; 2.; -1.; 4. |])
  in
  let x = fresh [| 0.25; 1.; -2.; 3. |] in
  let momentum (w, v) =
    let v = Nx.add (Nx.mul_s v 0.9) (Nx.mul_s (Nx.sub w x) 2.) in
    (Nx.sub w (Nx.mul_s v 0.1), v)
  in
  let run_f ~msg = run ~msg ~start ~same:(floats ~msg) in
  run_f ~msg:"leaves updated apart"
    (fun (w, v) -> (Nx.add_s w 1., Nx.add_s v 1.))
    (true, true);
  run_f ~msg:"a momentum step" momentum (true, true);
  run_f ~msg:"a momentum step, the velocity first"
    (fun s ->
      let w, v = momentum s in
      (v, w))
    (true, true);
  run_f ~msg:"a cycle" (fun (a, b) -> (Nx.add a b, Nx.sub b a)) (true, false);
  (* Sub-byte and complex leaves, stored and read back. *)
  let ints = Nx.create Nx.int4 [| 5 |] in
  run ~msg:"int4"
    ~start:(fun () -> (ints [| 7; -8; 0; 3; -1 |], ints [| 1; 2; -3; 4; 0 |]))
    ~same:(fun a b ->
      equal ~msg:"int4" (array int) (Nx.to_array a) (Nx.to_array b))
    (fun (w, v) ->
      let v = Nx.add v (Nx.sub w (Nx.ones_like w)) in
      (Nx.sub w v, v))
    (true, true);
  let complex re =
    Nx.create Nx.complex64 [| 3 |]
      (Array.map (fun r -> { Complex.re = r; im = -.r }) re)
  in
  (* Read through a copy: a view of the result would share the memory the next
     call takes. *)
  let parts ~msg a b =
    let b = Nx.copy b in
    floats ~msg (Nx.real f32 a) (Nx.real f32 b);
    floats ~msg (Nx.imag f32 a) (Nx.imag f32 b)
  in
  run ~msg:"complex64"
    ~start:(fun () -> (complex [| 1.; -0.; 2.5 |], complex [| 0.5; 2.; -1. |]))
    ~same:(parts ~msg:"complex64")
    (fun (w, v) ->
      let v = Nx.add v (Nx.sub w (Nx.ones_like w)) in
      (Nx.sub w v, v))
    (true, true)

(* A result that takes no leaf's memory reads a lent result from its store: a
   loop over a lent result runs once, and a scatter into a donated pool lends
   it while another result reads the pool's new value. Each call compiles and
   gives eager's values. *)
let test_readers () =
  let address x = Nx.Repr.read x Nx_array.buffer in
  let rows = Nx.reshape [| 2; 3 |] (fresh [| 0.5; -2.; 1.5; 0.25; 3.; -1. |]) in
  let f (w, a) =
    let w' = Nx.add_s w 1. in
    let c, _ =
      Rune.scan T.tensor T.tensor T.unit
        ~f:(fun c r -> (Nx.mul c r, ()))
        ~init:w' rows
    in
    (w', (Nx.add a c, c))
  in
  let two = T.(pair tensor tensor) in
  let g = Rune.jit T.(two @-> returns (pair tensor two)) f in
  let start () = (fresh [| 1.; -0.; 2.5 |], fresh [| 0.5; 2.; -1. |]) in
  let ew, (ea, ec) = f (start ()) in
  let w, a = start () in
  let bw = address w and ba = address a in
  let calls = Rune.evaluations () in
  let w', (a', c) = g (Nx.donate w, Nx.donate a) in
  equal ~msg:"compiled" int 0 (Rune.evaluations () - calls);
  equal ~msg:"in place" (Testable.pair bool bool) (true, true)
    (Rig.Buffer.overlaps bw (address w'), Rig.Buffer.overlaps ba (address a'));
  equal bits_t (bits ew) (bits w');
  equal bits_t (bits ea) (bits a');
  equal bits_t (bits ec) (bits c);
  let p = Nx.create Nx.int64 [| 2 |] [| 2L; 0L |] in
  let u = fresh [| 9.; -0. |] in
  let step pool =
    let pool = Nx.scatter ~axis:0 p u pool in
    (pool, Nx.sum pool)
  in
  let h = Rune.jit T.(tensor @-> returns (pair tensor tensor)) step in
  let pool () = fresh [| 1.; 2.; 3. |] in
  let ep, es = step (pool ()) in
  let q = pool () in
  let bq = address q in
  let q', sum = h (Nx.donate q) in
  equal ~msg:"the pool in place" bool true
    (Rig.Buffer.overlaps bq (address q'));
  equal bits_t (bits ep) (bits q');
  equal bits_t (bits es) (bits sum);
  (* A loop's stop, a check, reads a lent result from its store and a leaf as
     it was: it would run between the two stores, after the parameters' and
     before the velocity's, which the parameters' reads. The velocity takes
     fresh memory. *)
  let step (w, v) =
    let v' = Nx.add_s v 1. in
    let w' = Nx.mul w v' in
    let until c = Nx.less (Nx.max c) (Nx.scalar f32 1e30) in
    ignore (Rune.iterate' ~max:3 ~until ~f:Nx.neg (Nx.add w' v));
    (w', v')
  in
  let k = Rune.jit T.(two @-> returns two) step in
  let ew, ev = step (start ()) in
  let w, v = start () in
  let bw = address w and bv = address v in
  let calls = Rune.evaluations () in
  let w', v' = k (Nx.donate w, Nx.donate v) in
  equal ~msg:"a loop's stop: compiled" int 0 (Rune.evaluations () - calls);
  equal ~msg:"a loop's stop: in place" (Testable.pair bool bool) (true, false)
    (Rig.Buffer.overlaps bw (address w'), Rig.Buffer.overlaps bv (address v'));
  equal bits_t (bits ew) (bits w');
  equal bits_t (bits ev) (bits v')

(* A donated argument that a compiled remat's backward pass reads lends to a
   result that reads it at its own index: the backward pass reads it as it was,
   before the store over it. *)
let test_remat_lends () =
  let cube x = Nx.mul x (Nx.mul x x) in
  let r = Rune.remat T.(tensor @-> returns tensor) cube in
  let f x = (Nx.add_s x 1., Rune.grad' (fun x -> Nx.sum (r x)) x) in
  let g = Rune.jit T.(tensor @-> returns (pair tensor tensor)) f in
  let xs = [| 1.; -0.; 2.5; -3. |] in
  let ey, eg = f (fresh xs) in
  let x = fresh xs in
  let b = Nx.Repr.read x Nx_array.buffer in
  let y, d = g (Nx.donate x) in
  equal ~msg:"in place" bool true
    (Rig.Buffer.overlaps b (Nx.Repr.read y Nx_array.buffer));
  equal bits_t (bits ey) (bits y);
  equal ~msg:"the gradient" bits_t (bits eg) (bits d)

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

type failure =
  | None_raised
  | Small_at of int list * float
  | Signed_at of int list * int32 * bool
  | Refused of string

let raised f =
  match f () with
  | _ -> None_raised
  | exception Small (i, d) -> Small_at (Array.to_list i, d)
  | exception Signed (i, n, b) -> Signed_at (Array.to_list i, n, b)
  | exception Invalid_argument m -> Refused m

let failure =
  Testable.make
    ~pp:(fun ppf -> function
      | None_raised -> Format.pp_print_string ppf "no failure"
      | Small_at (i, d) ->
          Format.fprintf ppf "Small at [%s] %g"
            (String.concat "; " (List.map string_of_int i)) d
      | Signed_at (i, n, b) ->
          Format.fprintf ppf "Signed at [%s] %ld %b"
            (String.concat "; " (List.map string_of_int i)) n b
      | Refused m -> Format.fprintf ppf "Invalid_argument %S" m)
    ~equal:(fun a b ->
      match (a, b) with
      | Small_at (i, d), Small_at (j, e) -> i = j && Float.equal d e
      | _ -> a = b)

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

(* A loop's step that checks, on the host, compiles with the loop: the call
   raises the first failure of the first step that fails, as the steps written
   out raise it. Compiled, a result takes a donated argument's memory, which an
   evaluated program never lends. *)
let test_loop_checks () =
  let scan xs =
    fst
      (Rune.scan' ~init:(Nx.zeros_like (Nx.get [ 0 ] xs))
         ~f:(fun c r -> (Nx.add c (checked r), Nx.neg c))
         xs)
  in
  let g = Rune.jit' scan in
  let rows xs = Nx.create f32 [| 3; 2 |] xs in
  let both msg xs =
    equal ~msg failure
      (raised (fun () -> scan (rows xs)))
      (raised (fun () -> g (rows xs)))
  in
  let passing = rows [| 1.; 2.; 0.; 2.5; 1.; 1. |] in
  let lending =
    Rune.jit
      T.(tensor @-> tensor @-> returns (pair tensor tensor))
      (fun w xs -> (Nx.neg w, scan xs))
  in
  let w = v [| 1.; 2. |] in
  let b = Nx.Repr.read w Nx_array.buffer in
  let nw, y = lending (Nx.donate w) passing in
  let a = Option.get (Nx.Repr.array nw) in
  equal ~msg:"compiled: the result took the donated memory" bool true
    (Rig.Buffer.overlaps b (Nx_array.buffer a));
  equal ~msg:"passing" bits_t (bits (scan passing)) (bits y);
  both "the first step that fails" [| 1.; 2.; -1.; 1.; 7.; 1. |];
  both "the first check that fails in that step" [| 1.; 2.; 5.; -1.; -1.; 9. |];
  both "the second check" [| 1.; 2.; 1.; -2.5; 1.; 9. |];
  let iterate x =
    Rune.iterate' ~max:4
      ~until:(fun x -> Nx.greater (Nx.sum x) (Nx.scalar f32 4.))
      ~f:(fun x -> Nx.add (Nx.neg (checked x)) (Nx.scalar_like x 1.))
      x
  in
  let h = Rune.jit' iterate in
  let at xs = Nx.create f32 [| 2 |] xs in
  equal ~msg:"an iterate's step" failure
    (raised (fun () -> iterate (at [| 0.5; -2. |])))
    (raised (fun () -> h (at [| 0.5; -2. |])));
  (* A check reads its data, so a donated argument that a step's or an until's
     check reads lends nothing, where an unchecked loop over it lets it lend. *)
  let lends loop =
    let g =
      Rune.jit' (fun w ->
          ignore (loop w);
          Nx.mul_s w 4.)
    in
    let w = rows [| 1.; 2.; 0.; 2.5; 1.; 1. |] in
    let b = Nx.Repr.read w Nx_array.buffer in
    let a = Option.get (Nx.Repr.array (g (Nx.donate w))) in
    Rig.Buffer.overlaps b (Nx_array.buffer a)
  in
  let scan_with f xs =
    fst (Rune.scan' ~init:(Nx.zeros f32 [| 2 |]) ~f:(fun c r -> (f c r, c)) xs)
  in
  let until_with f xs =
    Rune.iterate' ~max:0
      ~until:(fun c -> Nx.all (Nx.less (f c) (Nx.scalar f32 9.)))
      ~f:Fun.id xs
  in
  equal ~msg:"an unchecked step lends" bool true
    (lends (scan_with (fun c r -> Nx.add c r)));
  equal ~msg:"a step's check: no lending" bool false
    (lends (scan_with (fun c r -> Nx.add c (checked r))));
  equal ~msg:"an unchecked until lends" bool true (lends (until_with Fun.id));
  equal ~msg:"an until's check: no lending" bool false
    (lends (until_with checked))

(* An iterate stops at the first failure of the program, as the steps written
   out stop: one in its step, one before it, or one before it in an enclosing
   step, each starting from a carry its until does not stop. Without that, the
   compiled iterate runs to its max of 10^9 steps, minutes past the test's
   limit. *)
let test_loop_stops () =
  let at xs = Nx.create f32 [| 2 |] xs in
  let long ~f x =
    Rune.iterate' ~max:1_000_000_000
      ~until:(fun x -> Nx.less (Nx.sum x) (Nx.scalar f32 (-1.)))
      ~f x
  in
  let step x = Nx.add x (Nx.scalar_like x 1.) in
  let failing x = long ~f:step (checked x) in
  let in_step x = long ~f:(fun x -> step (step (Nx.neg (checked x)))) x in
  let in_scan xs =
    fst
      (Rune.scan' ~init:(Nx.zeros f32 [| 2 |])
         ~f:(fun c r -> (Nx.add c (long ~f:step (checked r)), Nx.neg c))
         xs)
  in
  let both msg f x =
    equal ~msg failure
      (raised (fun () -> f x))
      (raised (fun () -> Rune.jit' f x))
  in
  both "a failure before the iterate" failing (at [| -1.; -7. |]);
  both "a failure in its step" in_step (at [| 1.; 2. |]);
  both "a failure in an enclosing step" in_scan
    (Nx.create f32 [| 2; 2 |] [| -1.; -2.; 1.; 2. |])

(* Loops whose steps check, drawn: rows of values that pass or fail each check,
   a check of no element beside them, and an iterate whose until checks too;
   compiled and eager raise the same failure, with its index and data. *)
type checked_loop = { rows : float array array; max : int; stop : float }

let checked_loops =
  let open Gen in
  let value = of_list [ 1.; 2.; 0.; -0.; 2.5; 4.; 5.; -1.; Float.nan ] in
  (let* n = int_range 0 4 in
   let* rows = array ~size:(constant n) (array ~size:(constant 2) value) in
   let+ max = int_range 0 4 and+ stop = of_list [ 0.; 2.; 6.; 50. ] in
   { rows; max; stop })
  |> with_pp (fun ppf c ->
      Format.fprintf ppf "rows [%s], max %d, stop at %g"
        (String.concat "; "
           (Array.to_list
              (Array.map
                 (fun r ->
                   String.concat " "
                     (Array.to_list (Array.map string_of_float r)))
                 c.rows)))
        c.max c.stop)

let law_checked_loop c =
  let n = Array.length c.rows in
  let xs =
    Nx.create f32 [| n; 2 |] (Array.concat (Array.to_list c.rows))
  in
  let step c r =
    let none = Nx.slice [ Nx.R (0, 0) ] r in
    Nx.check T.tensor (Nx.less none none) none (fun i _ -> Small (i, 0.));
    (Nx.add c (checked r), Nx.neg c)
  in
  let scan xs =
    fst (Rune.scan' ~init:(Nx.zeros f32 [| 2 |]) ~f:step xs)
  in
  let iterate x =
    Rune.iterate' ~max:c.max
      ~until:(fun x ->
        let s = Nx.sum (Nx.abs x) in
        Nx.check T.tensor
          (Nx.less (Nx.reshape [| 1 |] s) (Nx.scalar f32 30.))
          s
          (fun i d -> Small (i, Nx.item [] d));
        Nx.greater s (Nx.scalar f32 c.stop))
      ~f:(fun x -> Nx.add (checked x) (Nx.scalar_like x 1.5))
      x
  in
  let eager = raised (fun () -> scan xs) in
  (* A scan of no row returns its init, of every set, which no program
     computes: that one evaluates. *)
  let compiled ~compiles f x =
    let before = Rune.evaluations () in
    let r = raised (fun () -> Rune.jit' f x) in
    if compiles then
      equal ~msg:"compiled" int 0 (Rune.evaluations () - before);
    r
  in
  cover "no row" (n = 0);
  cover "a step fails" (eager <> None_raised);
  cover "every step passes" (n > 0 && eager = None_raised);
  equal ~msg:"scan" failure eager (compiled ~compiles:(n > 0) scan xs);
  let x =
    if n = 0 then Nx.create f32 [| 2 |] [| 1.; 0.5 |] else Nx.get [ 0 ] xs
  in
  let e = raised (fun () -> iterate x) in
  cover "an iterate's check fails"
    (match e with Small_at _ | Signed_at _ -> true | _ -> false);
  cover "an iterate reaches max" (match e with Refused _ -> true | _ -> false);
  equal ~msg:"iterate" failure e (compiled ~compiles:true iterate x)

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
   change, once per lane count, and each call runs its count's program: one
   repeated, one met again after another, one new. *)
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
    [ 2; 3; 3; 2; 4; 3; 2 ];
  equal ~msg:"one trace per lane count" int 3 !traces

(* Two systhreads of one domain meeting a new key: the second waits for the
   first's trace, and [f] runs once. The first's trace starts the second and
   waits until it is about to call, so that the second calls while the trace
   runs: the first resumes once the second blocks, unless the runtime's tick
   switches threads in the few microseconds before the second's call, which
   then finds the value made. Either order gives one trace. *)
let test_threads () =
  let traces = Atomic.make 0 in
  let m = Mutex.create () and c = Condition.create () and ready = ref false in
  let x = Nx.create f64 [| 2 |] [| 3.; -0. |] in
  let g = ref (fun y -> y) and second = ref None and thread = ref None in
  let call () =
    Mutex.protect m (fun () ->
        ready := true;
        Condition.signal c);
    second := Some (!g x)
  in
  g :=
    Rune.jit' (fun y ->
        if Atomic.fetch_and_add traces 1 = 0 then begin
          thread := Some (Thread.create call ());
          Mutex.protect m (fun () ->
              while not !ready do
                Condition.wait c m
              done)
        end;
        Nx.mul y y);
  let first = !g x in
  Option.iter Thread.join !thread;
  equal ~msg:"one trace" int 1 (Atomic.get traces);
  equal bits_t (bits (Nx.mul x x)) (bits first);
  equal bits_t (bits (Nx.mul x x)) (bits (Option.get !second))

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

(* Under reverse mode, the part of a compiled call computed at the call returns
   the residuals it computes, and the rest reads an argument, which the caller
   holds, as it is: a pullback over the call holds the bytes the call's result
   does, where it would hold a copy of the argument too. *)
let test_forwarded () =
  let host = Rig.host in
  let settled () =
    for _ = 1 to 4 do
      Gc.full_major ();
      ignore (Rig.Buffer.create host 0);
      Rig.free_cache host
    done;
    Rig.allocated host
  in
  let n = 65536 in
  let w = Nx.full f32 [| n |] 0.5 in
  let f x = Nx.mul (Nx.mul x x) w in
  let x = Nx.linspace f32 (-1.) 1. n in
  (* The bytes [keep ()] holds, with a compiled function of its own, whose
     program keeps its last call's buffers; after one uncounted run. *)
  let held keep =
    ignore (Sys.opaque_identity (keep ()));
    let before = settled () in
    let kept = keep () in
    let held = settled () - before in
    ignore (Sys.opaque_identity kept);
    held
  in
  let calls = Rune.evaluations () in
  let both f =
    let plain = held (fun () -> let g = Rune.jit' f in Obj.repr (g, g x)) in
    let pullback =
      held (fun () -> let g = Rune.jit' f in Obj.repr (g, Rune.vjp' g x))
    in
    (plain, pullback)
  in
  let plain, pullback = both f in
  equal ~msg:"the result" int (4 * n) plain;
  equal ~msg:"the result and the pullback" int plain pullback;
  equal ~msg:"evaluated calls" int 0 (Rune.evaluations () - calls);
  ignore (Sys.opaque_identity (x, w))

(* A donated argument that the linear part of a compiled call reads is
   consumed by the part computed at the call, which returns it for the
   pullback, as one handle or two of one donor. *)
let test_forwarded_donation () =
  let g = Rune.jit (pair_fn ()) (fun x w -> Nx.mul x w) in
  let x = fresh [| 1.; 2.; -0.5 |] in
  let w = fresh [| 3.; -0.5; 0.25 |] in
  let y, pb = Rune.vjp' (fun x -> g x (Nx.donate w)) x in
  equal bits_t (bits (fresh [| 3.; -1.; -0.125 |])) (bits y);
  equal bits_t
    (bits (fresh [| 3.; -0.5; 0.25 |]))
    (bits (pb (fresh [| 1.; 1.; 1. |])));
  dead w;
  let g3 =
    Rune.jit
      T.(tensor @-> tensor @-> tensor @-> returns tensor)
      (fun x a b -> Nx.mul x (Nx.mul a b))
  in
  let w = fresh [| 3.; -0.5; 0.25 |] in
  let _, pb = Rune.vjp' (fun x -> g3 x (Nx.donate w) (Nx.donate w)) x in
  equal ~msg:"two handles of one donor" bits_t
    (bits (fresh [| 9.; 0.25; 0.0625 |]))
    (bits (pb (fresh [| 1.; 1.; 1. |])));
  (* Each of [a] and [b] a residual: the handle's returned, its donor
     beside it read from the call too. *)
  let g3 =
    Rune.jit
      T.(tensor @-> tensor @-> tensor @-> returns tensor)
      (fun x a b -> Nx.add (Nx.mul x a) (Nx.mul x b))
  in
  let w = fresh [| 3.; -0.5; 0.25 |] in
  let _, pb = Rune.vjp' (fun x -> g3 x (Nx.donate w) w) x in
  equal ~msg:"a handle beside its donor" bits_t
    (bits (fresh [| 6.; -1.; 0.5 |]))
    (bits (pb (fresh [| 1.; 1.; 1. |])))

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

(* A compiled call inside a remat's function, a loop's step or a custom rule
   stays a call, which runs compiled where the construct evaluates its program:
   the construct gives its function's bits, the compiled sine's, over its
   arguments and its captures alike. *)
let test_calls_stay () =
  let g = Rune.jit' Nx.sin in
  let host x = Nx.place Nx.Host.on x in
  let x = host (Nx.linspace f64 (-3.) 3. 1024) in
  let c = host (Nx.linspace f64 0. 7. 1024) in
  let f x = Nx.add (g x) (g c) in
  equal ~msg:"the compiled sine differs from eager's here" bool false
    (bits (g x) = bits (Nx.sin x));
  let expected = bits (f x) in
  let r = Rune.remat T.(tensor @-> returns tensor) f in
  equal ~msg:"remat" bits_t expected (bits (r x));
  let rows = Nx.reshape [| 1; 1024 |] x in
  let scanned =
    fst
      (Rune.scan' ~f:(fun _ r -> (f r, Nx.neg r)) ~init:(Nx.zeros_like x) rows)
  in
  equal ~msg:"a scan's step" bits_t expected (bits scanned);
  let rule =
    Rune.custom_jvp T.tensor T.tensor (fun x -> (f x, fun dx -> dx))
  in
  equal ~msg:"a custom rule" bits_t expected (bits (rule x))

(* Totals in a compiled loop: each step's additions chained from the running
   total in turn, as eagerly, bit for bit for exact functions; a loop of no step
   leaves the running total, a negative zero included. *)
let test_loop_totals () =
  let t = Rune.Total.make () and u = Rune.Total.make () in
  let rows n =
    Nx.place Nx.Host.on
      (Nx.reshape [| n; 3 |] (Nx.linspace f64 (-1.) 1. (3 * n)))
  in
  let scan xs =
    fst
      (Rune.scan' ~init:(Nx.zeros f64 [| 3 |])
         ~f:(fun c r ->
           Rune.Total.add t (Nx.sum (Nx.mul r r));
           Rune.Total.add u (Nx.mul_s r 0.1);
           let inner, _ =
             Rune.scan' ~init:c
               ~f:(fun d s ->
                 Rune.Total.add t (Nx.sum (Nx.abs s));
                 (Nx.add d s, Nx.neg d))
               (Nx.reshape [| 3; 1 |] r)
           in
           (Nx.add c inner, Nx.neg c))
         xs)
  in
  let iterate x =
    Rune.iterate' ~max:20
      ~until:(fun x ->
        Rune.Total.add t (Nx.sum x);
        Nx.less (Nx.sum (Nx.abs x)) (Nx.scalar f64 0.1))
      ~f:(fun x ->
        Rune.Total.add u (Nx.neg x);
        Nx.mul_s x 0.5)
      x
  in
  let collected f x =
    let zero = Nx.scalar f64 (-0.) and zu = Nx.full f64 [| 3 |] (-0.) in
    let (y, a), b =
      Rune.Total.collect u ~zero:zu (fun () ->
          Rune.Total.collect t ~zero (fun () -> f x))
    in
    (bits y, bits a, bits b)
  in
  (* A scan of no row returns its init, here of every set, which no program
     computes: that one is evaluated. *)
  let same ?(compiles = true) msg f x =
    let before = Rune.evaluations () in
    let compiled = collected (Rune.jit' f) x in
    if compiles then
      equal ~msg:(msg ^ ": compiled") int 0 (Rune.evaluations () - before);
    equal ~msg (Testable.triple bits_t bits_t bits_t) (collected f x) compiled
  in
  same "scan" scan (rows 4);
  same ~compiles:false "no row" scan (rows 0);
  same "iterate" iterate (v64 [| 1.; -2.; 0.5 |]);
  same "iterate of no step" iterate (v64 [| 0.; 0.01; 0. |])

(* A loop's step that runs an iterate compiles whole: the inner loop's failure
   after its steps is a check of the outer step's. *)
let test_nested_iterate () =
  let halve x = Nx.mul_s x 0.5 in
  let small x = Nx.less (Nx.sum (Nx.abs x)) (Nx.scalar f64 0.01) in
  let scan xs =
    fst
      (Rune.scan' ~init:(Nx.zeros f64 [| 3 |])
         ~f:(fun c r ->
           let d = Rune.iterate' ~max:12 ~until:small ~f:halve (Nx.add r c) in
           (Nx.add c d, d))
         xs)
  in
  let xs =
    Nx.place Nx.Host.on (Nx.reshape [| 2; 3 |] (Nx.linspace f64 (-1.) 1. 6))
  in
  let before = Rune.evaluations () in
  let y = Rune.jit' scan xs in
  equal ~msg:"compiled" int 0 (Rune.evaluations () - before);
  equal bits_t (bits (scan xs)) (bits y);
  let big = Nx.mul_s xs 1e6 in
  invalid ~sub:"Rune.iterate': until is still false after max = 12 steps"
    (fun () -> Rune.jit' scan big)

(* The loops a derivative or a map makes of an iterate compile: under jvp,
   and under a map whose lanes stop together, with eager's values and the
   first failure of the steps written out. *)
let test_derived_iterates () =
  let double x = Nx.mul_s x 2. in
  let large x = Nx.greater_s (Nx.sum x) 50. in
  let iterate x = Rune.iterate' ~max:8 ~until:large ~f:double x in
  let jvp x = Rune.jvp' iterate x (Nx.ones_like x) in
  let x = Nx.place Nx.Host.on (v64 [| 1.; 0.5 |]) in
  let compiled msg f x =
    let before = Rune.evaluations () in
    let y = f x in
    equal ~msg:(msg ^ ": compiled") int 0 (Rune.evaluations () - before);
    y
  in
  let y, t =
    compiled "jvp" (Rune.jit T.(tensor @-> returns (pair tensor tensor)) jvp) x
  in
  let y', t' = jvp x in
  equal ~msg:"jvp value" bits_t (bits y') (bits y);
  equal ~msg:"jvp tangent" bits_t (bits t') (bits t);
  (* A count no lane reads stops every lane at once. *)
  let counted x =
    fst
      (Rune.iterate
         T.(pair tensor tensor)
         ~max:8
         ~until:(fun (_, k) -> Nx.greater_equal_s k 3.)
         ~f:(fun (x, k) -> (double x, Nx.add_s k 1.))
         (x, Nx.place Nx.Host.on (Nx.scalar f64 0.)))
  in
  let xs =
    Nx.place Nx.Host.on (Nx.reshape [| 2; 2 |] (v64 [| 1.; 2.; 3.; 4. |]))
  in
  equal ~msg:"vmap" bits_t
    (bits (Rune.vmap' counted xs))
    (bits (compiled "vmap" (Rune.jit' (Rune.vmap' counted)) xs));
  (* A check in the step fails at its fifth step, before [max] does. *)
  let checked x =
    Rune.iterate' ~max:6
      ~until:(fun x -> Nx.greater_s (Nx.sum x) 1e9)
      ~f:(fun x ->
        Nx.check T.tensor (Nx.less_s x 10.) x (fun _ _ -> Exit);
        double x)
      x
  in
  let jvp_checked x = snd (Rune.jvp' checked x (Nx.ones_like x)) in
  raises Exit (fun () -> jvp_checked x);
  let before = Rune.evaluations () in
  raises Exit (fun () -> Rune.jit' jvp_checked x);
  invalid ~sub:"until is still false after max = 6 steps" (fun () ->
      Rune.jit' jvp_checked (Nx.mul_s x 1e-3));
  equal ~msg:"checked: compiled" int 0 (Rune.evaluations () - before)

(* A call under jvp derives apart for which totals' running values carry a
   tangent: two calls of one key, the tangent on the first total's running
   value, then on the second's, each give their own sums' tangents. *)
let test_seed_tangents () =
  let t1 = Rune.Total.make () and t2 = Rune.Total.make () in
  let g =
    Rune.jit' (fun c ->
        Rune.Total.add t1 (Nx.mul_s c 2.);
        Rune.Total.add t2 (Nx.mul_s c 3.);
        c)
  in
  let c = v64 [| 1.; 2. |] in
  let sums first x =
    let zero = Nx.zeros f64 [| 2 |] in
    let (_, s1), s2 =
      Rune.Total.collect t2 ~zero (fun () ->
          Rune.Total.collect t1 ~zero (fun () ->
              Rune.Total.add (if first then t1 else t2) x;
              g c))
    in
    Nx.stack [ s1; s2 ]
  in
  let x = v64 [| 5.; 7. |] and dx = v64 [| 1.; 1. |] in
  let tangent first = bits (snd (Rune.jvp' (sums first) x dx)) in
  let ones = Nx.ones f64 [| 2 |] and zeros = Nx.zeros f64 [| 2 |] in
  equal ~msg:"the first's" bits_t
    (bits (Nx.stack [ ones; zeros ]))
    (tangent true);
  equal ~msg:"the second's" bits_t
    (bits (Nx.stack [ zeros; ones ]))
    (tangent false)

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
             prop ~count:20 ~examples:bitcast_examples
               "a bitcast gives eager's bits" bitcast_cases law_bitcast;
             test "nx.quant's decodes compile" test_quant_decodes;
           ];
         group "jit"
           [
             test "a read raises naming Rune.jit" test_read;
             test "a keyless draw raises naming Rune.jit" test_keyless;
             test "a constant key draws the same each call" test_constant_key;
             test "the function runs once per key" test_runs_once;
             test "structures in and out" test_structures;
             test "host arguments" test_host_values;
             prop "a function searched at a width" searched_cases law_searched;
             prop "a constant placed where the program computes" placed_cases
               law_placed;
             test "a constant result off the program's device"
               test_constant_result;
             test "a jit inside a jit" test_nested;
             test "an integer maximum of a running sum wraps"
               test_wrapped_max;
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
             prop ~examples:state_examples
               "a step over donated leaves keeps the values" state_cases
               law_states;
             test "a step's results lend in an order of their stores"
               test_states;
             test "an argument a remat's backward pass reads lends"
               test_remat_lends;
             test "a result that takes no memory reads a lent one's store"
               test_readers;
             test "a contraction's init lends, an operand does not"
               test_contract_init;
             test "a scatter's target lends" test_scatter_lends;
             test "checks raise as eagerly" test_checks;
             stateful ~domains:2 ~count:10 ~steps:8
               "a call retraces iff its abstract value is new" cache_commands;
             test "two threads of a domain trace once" test_threads;
           ];
         group "constructs"
           [
             test "a scan's step traces once and compiles one loop"
               test_staged_scan;
             test "an iterate compiles one loop" test_staged_iterate;
             test "custom rules and remats compile with their program"
               test_constructs;
             test "a loop's checks raise as the steps written out"
               test_loop_checks;
             test ~timeout:60. "an iterate stops at the program's first failure"
               test_loop_stops;
             prop "drawn loops' checks raise as the steps written out"
               checked_loops law_checked_loop;
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
             test "the known part of a call forwards its arguments"
               test_forwarded;
             test "a forwarded donated argument lives on for the pullback"
               test_forwarded_donation;
             test "compositions around a call give eager's values"
               test_compositions;
             test "a call in a construct stays a call" test_calls_stay;
             test "totals in compiled loops" test_loop_totals;
             test "an iterate in a step compiles with the loop"
               test_nested_iterate;
             test "an iterate's derived loops compile" test_derived_iterates;
             test "a call derives apart for its seeds' tangents"
               test_seed_tangents;
           ];
       ])
