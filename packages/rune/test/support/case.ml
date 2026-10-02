(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Gen
module Op = Nx.Op

type dtype = D : ('a, 'b) Nx.dtype -> dtype

let pp_dtype ppf (D d) = Nx.pp_dtype ppf d
let float64 = D Nx.float64

type instance =
  | Instance : {
      f : ('a, 'b) Nx.t list -> ('c, 'd) Nx.t list;
      x : ('a, 'b) Nx.t list;
      linear : (('a, 'b) Nx.t list -> ('c, 'd) Nx.t list) option;
      extra : Row.t list option;
      pp : Format.formatter -> unit;
    }
      -> instance

let pp_instance ppf (Instance i) = i.pp ppf

type kind = Tangent | Plain | Integer

type t = {
  row : Row.t;
  kind : kind;
  difference : float * float;
  smooth : dtype -> instance Gen.t;
  finite : instance Gen.t;
  complex : instance Gen.t option;
  dtypes : dtype list;
  derivative : (float -> float) option;
}

(* Shapes and values *)

let numel s = Array.fold_left ( * ) 1 s
let dim = frequency [ (1, constant 0); (2, constant 1); (5, int_range 2 3) ]
let full_dim = frequency [ (1, constant 1); (3, int_range 2 3) ]

let shape ?(dim = dim) lo hi =
  let* rank = int_range lo hi in
  array ~size:(constant rank) dim

let range lo hi = float_range lo hi

let away lo hi =
  let+ m = range lo hi and+ negative = bool in
  if negative then -.m else m

(* [tensor d g shape] is a tensor of [d] whose elements [g] draws as floats. *)
let tensor d g shape =
  let+ a = array ~size:(constant (numel shape)) g in
  Nx.cast d (Nx.create Nx.float64 shape a)

(* [ctensor d (re, im) shape] is a complex tensor of [d] whose components [re]
   and [im] draw. *)
let ctensor d (re, im) shape =
  let+ a =
    array
      ~size:(constant (numel shape))
      (let+ re = re and+ im = im in
       { Complex.re; im })
  in
  Nx.cast d (Nx.create Nx.complex128 shape a)

(* [distinct d shape] has elements at least [0.04] apart: a shuffle of a ramp
   with a little noise, so no two are tied. *)
let distinct d shape =
  let n = numel shape in
  let* order = permutation (List.init n Fun.id) in
  let+ noise = array ~size:(constant n) (range 0. 0.01) in
  let ramp = Array.of_list order in
  Nx.cast d
    (Nx.create Nx.float64 shape
       (Array.mapi
          (fun i k -> (0.05 *. float_of_int k) -. 1. +. noise.(i))
          ramp))

(* Complex elements of both components in [[-2, 2]]. *)
let plane = (range (-2.) 2., range (-2.) 2.)

(* Elements drawn among a few values, so ties and zeros are common. *)
let tied = of_list [ -1.; 0.; 0.; 1.; 2. ]

let pp_tensors ppf xs =
  Format.pp_print_list ~pp_sep:Format.pp_print_space
    (fun ppf x -> Format.fprintf ppf "@[%a@]" Nx.pp x)
    ppf xs

let pp_with desc xs ppf = Format.fprintf ppf "@[<v>%s@ %a@]" desc pp_tensors xs

(* Instances *)

let instance ?linear ?(extra = Some []) desc f x =
  Instance { f; x; linear; extra; pp = pp_with desc x }

(* [take tracked xs] is the elements of [xs] where [tracked] holds. *)
let take tracked xs = List.filteri (fun i _ -> List.nth tracked i) xs

(* [merge tracked all xs] is [all] with the tracked positions read from [xs]. *)
let rec merge tracked all xs =
  match (tracked, all, xs) with
  | true :: tracked, _ :: all, x :: xs -> x :: merge tracked all xs
  | false :: tracked, a :: all, xs -> a :: merge tracked all xs
  | [], [], [] -> []
  | _ -> invalid_arg "Case.merge"

(* The nonempty subsets of [n] operands, as tracked flags. *)
let patterns n =
  let rec subsets n =
    if n = 0 then [ [] ]
    else List.concat_map (fun s -> [ true :: s; false :: s ]) (subsets (n - 1))
  in
  let pp ppf t =
    Format.fprintf ppf "tracked %s"
      (String.concat "" (List.map (fun b -> if b then "x" else "-") t))
  in
  of_list ~pp (List.filter (List.exists Fun.id) (subsets n))

(* How a row of several operands is linear in the ones it tracks. *)
type ('x, 'y) linearity =
  | Not_linear
  | Coefficients (* Linear, with the constants as coefficients. *)
  | Zeroed (* Linear, with the constants contributing nothing. *)
  | Map of ('x list -> 'y list)
(* Linear, as this map. *)

(* [nary desc op operands tracked linearity] is the instance of [op] at the
   tracked [operands], the others captured. *)
let nary desc op operands tracked linearity =
  let f xs = op (merge tracked operands xs) in
  let linear =
    match linearity with
    | Not_linear -> None
    | Coefficients -> Some f
    | Zeroed ->
        let zeros = List.map Nx.zeros_like operands in
        Some (fun xs -> op (merge tracked zeros xs))
    | Map m -> Some m
  in
  instance ?linear desc f (take tracked operands)

let one = function [ x ] -> x | _ -> invalid_arg "Case: one operand"
let two = function [ a; b ] -> (a, b) | _ -> invalid_arg "Case: two operands"

(* Unary *)

let unary_domain : Nx_backend.unary -> float Gen.t = function
  | Neg | Exp | Sinh | Cosh | Tanh -> range (-5.) 5.
  | Recip -> away 0.1 3.
  | Sqrt -> range 0.1 5.
  | Log -> range 0.1 10.
  | Log1p -> range (-0.9) 10.
  | Expm1 -> range (-5.) 5.
  | Sin | Cos -> range (-6.) 6.
  | Tan -> range (-1.4) 1.4
  | Asin | Acos -> range (-0.9) 0.9
  | Atan -> range (-10.) 10.
  | Erf -> range (-4.) 4.
  | Abs -> away 0.01 3.
  | Sign | Trunc | Ceil | Floor | Round -> range (-3.) 3.

let unary_finite : Nx_backend.unary -> float Gen.t = function
  | Abs -> frequency [ (1, constant 0.); (3, range (-3.) 3.) ]
  | k -> unary_domain k

(* Complex domains stay off the branch cuts and the poles. *)
let unary_complex : Nx_backend.unary -> (float Gen.t * float Gen.t) option =
  function
  | Neg -> Some (range (-3.) 3., range (-3.) 3.)
  | Recip -> Some (away 0.3 2., away 0.3 2.)
  | Sqrt | Log -> Some (range 0.1 2., range (-2.) 2.)
  | Exp | Sin | Cos | Sinh | Cosh -> Some (range (-1.5) 1.5, range (-1.5) 1.5)
  | Tanh -> Some (range (-1.5) 1.5, range (-1.2) 1.2)
  | Tan -> Some (range (-1.2) 1.2, range (-1.5) 1.5)
  | Asin | Acos -> Some (range (-0.8) 0.8, range (-1.5) 1.5)
  | Atan -> Some (range (-2.) 2., range (-0.8) 0.8)
  | Abs | Sign -> Some (away 0.1 2., away 0.1 2.)
  | Log1p | Expm1 | Erf | Trunc | Ceil | Floor | Round -> None

let unary_derivative : Nx_backend.unary -> (float -> float) option = function
  | Neg -> Some (fun _ -> -1.)
  | Recip -> Some (fun x -> -1. /. (x *. x))
  | Sqrt -> Some (fun x -> 1. /. (2. *. Float.sqrt x))
  | Exp -> Some Float.exp
  | Log -> Some (fun x -> 1. /. x)
  | Log1p -> Some (fun x -> 1. /. (1. +. x))
  | Expm1 -> Some Float.exp
  | Sin -> Some Float.cos
  | Cos -> Some (fun x -> -.Float.sin x)
  | Tan -> Some (fun x -> 1. /. (Float.cos x *. Float.cos x))
  | Asin -> Some (fun x -> 1. /. Float.sqrt (1. -. (x *. x)))
  | Acos -> Some (fun x -> -1. /. Float.sqrt (1. -. (x *. x)))
  | Atan -> Some (fun x -> 1. /. (1. +. (x *. x)))
  | Sinh -> Some Float.cosh
  | Cosh -> Some Float.sinh
  | Tanh -> Some (fun x -> 1. /. (Float.cosh x *. Float.cosh x))
  | Erf -> Some (fun x -> 1.12837916709551257390 *. Float.exp (-.(x *. x)))
  | Abs -> Some (fun x -> if x > 0. then 1. else -1.)
  | Sign | Trunc | Ceil | Floor | Round -> None

let reals = [ D Nx.float16; D Nx.bfloat16; D Nx.float32; D Nx.float64 ]
let complexes = [ D Nx.complex64; D Nx.complex128 ]
let wide = [ D Nx.float32; D Nx.float64; D Nx.complex64; D Nx.complex128 ]

let case ?(kind = Tangent) ?(difference = (1e-6, 1e-8)) ?finite ?complex
    ?(dtypes = reals) ?derivative row smooth =
  let finite = match finite with Some g -> g | None -> smooth float64 in
  { row; kind; difference; smooth; finite; complex; dtypes; derivative }

let unary_instance k g =
  let* s = shape 0 3 in
  let+ x = g s in
  let f x = [ Op.eval (Unary (k, one x)) ] in
  let linear = match k with Nx_backend.Neg -> Some f | _ -> None in
  instance ?linear (Row.name (Unary k)) f [ x ]

let unary_case (k : Nx_backend.unary) =
  let kind =
    match k with Sign | Trunc | Ceil | Floor | Round -> Plain | _ -> Tangent
  in
  let complex =
    Option.map
      (fun dom -> unary_instance k (ctensor Nx.complex128 dom))
      (unary_complex k)
  in
  case ~kind
    ~finite:(unary_instance k (tensor Nx.float64 (unary_finite k)))
    ?complex
    ~dtypes:(if Option.is_some complex then reals @ complexes else reals)
    ?derivative:(unary_derivative k) (Unary k)
    (fun (D d) -> unary_instance k (tensor d (unary_domain k)))

(* Binary *)

let pair_of a b =
  let+ a = a and+ b = b in
  (a, b)

let binary_domain : Nx_backend.binary -> (float * float) Gen.t = function
  | Add | Sub | Mul -> pair_of (range (-3.) 3.) (range (-3.) 3.)
  | Fdiv -> pair_of (range (-3.) 3.) (away 0.1 3.)
  | Atan2 -> pair_of (away 0.05 3.) (away 0.1 3.)
  | Pow -> pair_of (range 0.1 3.) (range (-2.) 2.)
  | Maximum | Minimum ->
      let+ a = range (-3.) 3. and+ d = away 0.01 1. in
      (a, a +. d)
  | Mod ->
      (* [a / b] at least [0.05] from an integer. *)
      let+ b = away 0.5 2. and+ q = int_range (-3) 2 and+ r = range 0.05 0.95 in
      ((float_of_int q +. r) *. b, b)
  | Idiv | And | Or | Xor -> invalid_arg "Case: an integer operation"

let binary_finite : Nx_backend.binary -> (float * float) Gen.t = function
  | Add | Sub | Mul | Maximum | Minimum -> pair_of tied tied
  | k -> binary_domain k

(* A sum's tangent along one operand is that operand's tangent, with no zero
   added for the other: [-0.] stays [-0.]. *)
let binary_linearity (k : Nx_backend.binary) tracked =
  match (k, tracked) with
  | (Add | Sub), [ true; true ] -> Coefficients
  | (Add | Sub), [ true; false ] | Add, [ false; true ] -> Map Fun.id
  | Sub, [ false; true ] -> Map (fun x -> [ Op.eval (Unary (Neg, one x)) ])
  | Mul, ([ true; false ] | [ false; true ]) | Fdiv, [ true; false ] ->
      Coefficients
  | _ -> Not_linear

let binary_of k xs =
  let a, b = two xs in
  [ Op.eval (Binary (k, a, b)) ]

let binary_instance k a b =
  let+ a = a and+ b = b and+ tracked = patterns 2 in
  nary (Row.name (Binary k)) (binary_of k) [ a; b ] tracked
    (binary_linearity k tracked)

let operand_pair d dom =
  let* s = shape 0 3 in
  let+ ps = array ~size:(constant (numel s)) dom in
  let side f = Nx.cast d (Nx.create Nx.float64 s (Array.map f ps)) in
  (side fst, side snd)

let real_binary k dom (D d) =
  let* a, b = operand_pair d dom in
  binary_instance k (constant a) (constant b)

let binary_complex : Nx_backend.binary -> _ = function
  | Add | Sub | Mul -> Some (plane, plane)
  | Fdiv -> Some (plane, (away 0.3 2., away 0.3 2.))
  | Pow ->
      Some ((range 0.1 2., range (-1.) 1.), (range (-1.) 1., range (-1.) 1.))
  | Idiv | Mod | Atan2 | Maximum | Minimum | And | Or | Xor -> None

(* An integer operation of the integer cast of a real operand: the cast drops
   every tangent, so the operation never meets one. *)
let integer_instance row op =
  let* s = shape 0 3 in
  let+ x = tensor Nx.float64 (range (-20.) 20.) s
  and+ c = tensor Nx.int32 (away 1. 5.) s in
  instance
    ~extra:(Some [ Row.Cast; Row.Cast ])
    (Row.name row)
    (fun x -> [ Nx.cast Nx.float64 (op (Nx.cast Nx.int32 (one x)) c) ])
    [ x ]

let binary_case (k : Nx_backend.binary) =
  match k with
  | Idiv | And | Or | Xor ->
      let g =
        integer_instance (Binary k) (fun a b -> Op.eval (Binary (k, a, b)))
      in
      case ~kind:Integer ~dtypes:[ float64 ] (Binary k) (fun _ -> g)
  | Add | Sub | Mul | Fdiv | Mod | Pow | Atan2 | Maximum | Minimum ->
      let complex =
        Option.map
          (fun (da, db) ->
            let* s = shape 0 3 in
            binary_instance k
              (ctensor Nx.complex128 da s)
              (ctensor Nx.complex128 db s))
          (binary_complex k)
      in
      case
        ~finite:(real_binary k (binary_finite k) float64)
        ?complex
        ~dtypes:(if Option.is_some complex then reals @ complexes else reals)
        (Binary k)
        (real_binary k (binary_domain k))

(* Selection *)

let compare_case k =
  case ~kind:Plain (Compare k) (fun (D d) ->
      let* a, b = operand_pair d (pair_of tied tied) in
      let+ tracked = patterns 2 in
      nary (Row.name (Compare k))
        (fun xs ->
          let a, b = two xs in
          [ Op.eval (Compare (k, a, b)) ])
        [ a; b ] tracked Not_linear)

let where_instance g =
  let* s = shape 0 3 in
  let+ c = array ~size:(constant (numel s)) bool
  and+ a = g s
  and+ b = g s
  and+ tracked = patterns 2 in
  let c = Nx.create Nx.bool s c in
  nary "where"
    (fun xs ->
      let a, b = two xs in
      [ Op.eval (Where (c, a, b)) ])
    [ a; b ] tracked Zeroed

let where_case =
  case ~dtypes:(reals @ complexes)
    ~complex:(where_instance (ctensor Nx.complex128 plane))
    Where
    (fun (D d) -> where_instance (tensor d (range (-3.) 3.)))

(* A multiply-add is linear in its sum and one factor together. *)
let fma_instance g =
  let* s = shape 0 3 in
  let+ a = g s and+ b = g s and+ c = g s and+ tracked = patterns 3 in
  nary "fma"
    (fun xs ->
      match xs with
      | [ a; b; c ] -> [ Op.eval (Fma (a, b, c)) ]
      | _ -> invalid_arg "Case: three operands")
    [ a; b; c ] tracked
    (match tracked with
    | [ true; false; true ] | [ false; true; true ] -> Coefficients
    | _ -> Not_linear)

let fma_case =
  case
    ~finite:(fma_instance (tensor Nx.float64 tied))
    Fma
    (fun (D d) -> fma_instance (tensor d (range (-3.) 3.)))

(* Reductions and scans *)

(* A running product is drawn with zeros, where its rule is exact at every
   order; a reduced product away from them, whose rule is exact at one zero only
   (its edges hold the zeros). *)
let reduce_values ~running (k : Nx_backend.reduce) d s =
  match k with
  | Sum -> tensor d (range (-3.) 3.) s
  | Prod when running ->
      tensor d (frequency [ (1, constant 0.); (4, away 0.01 2.) ]) s
  | Prod -> tensor d (away 0.01 2.) s
  | Max | Min -> distinct d s

let reduce_finite (k : Nx_backend.reduce) s =
  match k with
  | Sum -> tensor Nx.float64 (range (-3.) 3.) s
  | Prod | Max | Min -> tensor Nx.float64 tied s

let reduce_instance (k : Nx_backend.reduce) values =
  let extremum = match k with Max | Min -> true | Sum | Prod -> false in
  let* s = shape ~dim:(if extremum then full_dim else dim) 1 3 in
  let* axes = subsequence (List.init (Array.length s) Fun.id) in
  let axes = if extremum && axes = [] then [ 0 ] else axes in
  let+ x = values s in
  let axes = Array.of_list axes in
  let f x = [ Op.eval (Reduce (k, axes, one x)) ] in
  let linear = match k with Sum -> Some f | Prod | Max | Min -> None in
  instance ?linear
    (Format.asprintf "%s over [%s]" (Row.name (Reduce k))
       (String.concat "; " (Array.to_list (Array.map string_of_int axes))))
    f [ x ]

let scan_instance (k : Nx_backend.reduce) values =
  let* s = shape 1 3 in
  let* axis = int_range 0 (Array.length s - 1) in
  let+ x = values s in
  let f x = [ Op.eval (Scan (k, axis, one x)) ] in
  let linear = match k with Sum -> Some f | Prod | Max | Min -> None in
  instance ?linear
    (Printf.sprintf "%s along %d" (Row.name (Scan k)) axis)
    f [ x ]

let cumulative_complex : Nx_backend.reduce -> _ = function
  | Sum -> Some (ctensor Nx.complex128 (range (-1.5) 1.5, range (-1.5) 1.5))
  | Prod -> Some (ctensor Nx.complex128 (away 0.1 1.5, away 0.1 1.5))
  | Max | Min -> None

let reduce_case k =
  let complex = Option.map (reduce_instance k) (cumulative_complex k) in
  case
    ~finite:(reduce_instance k (reduce_finite k))
    ?complex
    ~dtypes:(if Option.is_some complex then reals @ complexes else reals)
    (Reduce k)
    (fun (D d) -> reduce_instance k (reduce_values ~running:false k d))

let scan_case k =
  let complex = Option.map (scan_instance k) (cumulative_complex k) in
  case
    ~finite:(scan_instance k (reduce_finite k))
    ?complex
    ~dtypes:(if Option.is_some complex then reals @ complexes else reals)
    (Scan k)
    (fun (D d) -> scan_instance k (reduce_values ~running:true k d))

let arg_reduce_case k =
  case ~kind:Plain (Arg_reduce k) (fun (D d) ->
      let* s = shape ~dim:full_dim 1 3 in
      let* axis = int_range 0 (Array.length s - 1) in
      let+ x = tensor d tied s in
      instance
        (Printf.sprintf "%s along %d" (Row.name (Arg_reduce k)) axis)
        (fun x -> [ Op.eval (Arg_reduce (k, axis, one x)) ])
        [ x ])

let sort_instance ~indices values =
  let* s = shape 1 3 in
  let* axis = int_range 0 (Array.length s - 1) in
  let+ descending = bool and+ x = values s in
  let desc =
    Printf.sprintf "%s along %d%s"
      (if indices then "argsort" else "sort")
      axis
      (if descending then ", descending" else "")
  in
  if indices then
    instance desc
      (fun x -> [ Op.eval (Argsort { descending; axis; x = one x }) ])
      [ x ]
  else
    instance desc
      (fun x -> [ Op.eval (Sort { descending; axis; x = one x }) ])
      [ x ]

let sort_case =
  case
    ~finite:(sort_instance ~indices:false (tensor Nx.float64 tied))
    Sort
    (fun (D d) -> sort_instance ~indices:false (distinct d))

let argsort_case =
  case ~kind:Plain Argsort (fun (D d) ->
      sort_instance ~indices:true (tensor d tied))

(* Grouping: rows of words among a few, so that rows repeat. *)

let group_case =
  case ~kind:Integer ~dtypes:[ float64 ] Group (fun _ ->
      let* s = shape 2 2 in
      let+ x = tensor Nx.float64 (of_list [ 0.; 1.; 2. ]) s in
      instance
        ~extra:(Some [ Row.Cast; Row.Cast ])
        "group"
        (fun x ->
          [
            Nx.cast Nx.float64
              (Op.eval
                 (Group { by = "Case.group"; x = Nx.cast Nx.uint64 (one x) }));
          ])
        [ x ])

(* Assembly *)

let pad_instance d g =
  let* s = shape 0 3 in
  let+ x = g s
  and+ padding =
    array
      ~size:(constant (Array.length s))
      (pair_of (int_range 0 2) (int_range 0 2))
  and+ fill = away 0.5 3. in
  let pad v x = [ Op.eval (Pad (padding, v, one x)) ] in
  Instance
    {
      f = pad (Nx_dtype.of_float d fill);
      x = [ x ];
      linear = Some (pad (Nx_dtype.zero d));
      extra = Some [];
      pp =
        pp_with
          (Printf.sprintf "pad [%s] with %g"
             (String.concat "; "
                (Array.to_list
                   (Array.map
                      (fun (a, b) -> Printf.sprintf "%d, %d" a b)
                      padding)))
             fill)
          [ x ];
    }

let pad_case =
  case ~dtypes:(reals @ complexes)
    ~complex:(pad_instance Nx.complex128 (ctensor Nx.complex128 plane))
    Pad
    (fun (D d) -> pad_instance d (tensor d (range (-3.) 3.)))

let cat_instance g =
  let* s = shape ~dim:full_dim 1 3 in
  let* axis = int_range 0 (Array.length s - 1) in
  let* n = int_range 1 3 in
  let* lengths = array ~size:(constant n) (int_range 0 2) in
  let* pieces =
    Array.fold_right
      (fun len acc ->
        let+ x = g (Array.mapi (fun i d -> if i = axis then len else d) s)
        and+ acc = acc in
        x :: acc)
      lengths (constant [])
  in
  let+ tracked = patterns n in
  nary
    (Printf.sprintf "cat along %d" axis)
    (fun xs -> [ Op.eval (Cat (axis, xs)) ])
    pieces tracked Zeroed

let cat_case =
  case ~dtypes:(reals @ complexes)
    ~complex:(cat_instance (ctensor Nx.complex128 plane))
    Cat
    (fun (D d) -> cat_instance (tensor d (range (-3.) 3.)))

(* Conversions *)

let cast_instance g targets =
  let* s = shape 0 3 in
  let* (D t) = of_list ~pp:pp_dtype targets in
  let+ x = g s in
  let f x = [ Op.eval (Convert (Cast, t, one x)) ] in
  instance ~linear:f (Format.asprintf "cast to %a" Nx.pp_dtype t) f [ x ]

let cast_case =
  let exact = [ D Nx.float64; D Nx.complex128 ] in
  case ~dtypes:(reals @ complexes)
    ~complex:(cast_instance (ctensor Nx.complex128 plane) exact)
    Cast
    (fun (D d) ->
      cast_instance
        (tensor d (range (-3.) 3.))
        (if D d = float64 then exact else [ D Nx.float64 ]))

(* A bitcast between a complex dtype and the float of its components, which
   holds them along a last axis of two. Any other bitcast has no tangent, which
   the edges check. *)
let bitcast_instance dst x =
  let f x = [ Op.eval (Convert (Bitcast, dst, one x)) ] in
  instance ~linear:f (Format.asprintf "bitcast to %a" Nx.pp_dtype dst) f [ x ]

let bitcast_case =
  let widen d dst =
    let* s = shape 0 2 in
    let+ x = tensor d (range (-3.) 3.) (Array.append s [| 2 |]) in
    bitcast_instance dst x
  and narrow d dst =
    let* s = shape 0 2 in
    let+ x = ctensor d plane s in
    bitcast_instance dst x
  in
  case ~dtypes:wide ~complex:(narrow Nx.complex128 Nx.float64) Bitcast
    (fun (D d) ->
      match d with
      | Float32 -> widen d Nx.complex64
      | Float64 -> widen d Nx.complex128
      | Complex64 -> narrow d Nx.float32
      | Complex128 -> narrow d Nx.float64
      | _ -> invalid_arg "Case: a bitcast of a float or complex dtype")

let threefry_case =
  case ~kind:Integer ~dtypes:[ float64 ] Threefry (fun _ ->
      let* s = shape 0 2 in
      let s = Array.append s [| 2 |] in
      let+ x = tensor Nx.float64 (range (-20.) 20.) s
      and+ key = tensor Nx.int32 (range (-1000.) 1000.) s in
      instance
        ~extra:(Some [ Row.Cast; Row.Cast ])
        "threefry"
        (fun x ->
          [
            Nx.cast Nx.float64
              (Op.eval (Threefry (key, Nx.cast Nx.int32 (one x))));
          ])
        [ x ])

(* Indexed access *)

let gather_instance g =
  let* s = shape ~dim:full_dim 1 3 in
  let* axis = int_range 0 (Array.length s - 1) in
  let n = s.(axis) in
  let* m = int_range 0 3 in
  let is = Array.mapi (fun i d -> if i = axis then m else d) s in
  let+ x = g s
  and+ indices =
    tensor Nx.int64
      (frequency
         [
           (6, map float_of_int (int_range 0 (n - 1)));
           (1, of_list [ -1.; float_of_int n ]);
         ])
      is
  in
  let f x = [ Op.eval (Gather (axis, indices, one x)) ] in
  instance ~linear:f
    (Format.asprintf "gather along %d at %a" axis Nx.pp indices)
    f [ x ]

let gather_case =
  case ~dtypes:(reals @ complexes)
    ~complex:(gather_instance (ctensor Nx.complex128 plane))
    Gather
    (fun (D d) -> gather_instance (tensor d (range (-3.) 3.)))

let scatter_name : Nx_backend.scatter -> string = function
  | `Set -> "set"
  | `Add -> "add"
  | `Max -> "max"
  | `Min -> "min"

(* [scatter_instance mode operands] scatters by [mode] the updates into the
   element [operands s is] draws, of shapes [is] and [s]. *)
let scatter_instance (mode : Nx_backend.scatter) operands =
  let* s = shape ~dim:full_dim 1 3 in
  let* axis = int_range 0 (Array.length s - 1) in
  let* m = int_range 1 3 in
  let is = Array.mapi (fun i d -> if i = axis then m else d) s in
  let+ into, updates = operands s is
  and+ indices =
    tensor Nx.int64 (map float_of_int (int_range 0 (s.(axis) - 1))) is
  and+ tracked = patterns 2 in
  nary
    (Format.asprintf "scatter (%s) along %d at %a" (scatter_name mode) axis
       Nx.pp indices)
    (fun xs ->
      let updates, into = two xs in
      [
        Op.eval (Scatter { mode; unique = false; axis; indices; updates; into });
      ])
    [ updates; into ] tracked
    (match mode with `Set | `Add -> Zeroed | `Max | `Min -> Not_linear)

(* [apart g s is] is the element and the updates [g s] and [g is] draw. *)
let apart g s is =
  let+ into = g s and+ updates = g is in
  (into, updates)

(* [together g s is] is the element and the updates cut from one tensor [g]
   draws, so that the elements of [distinct] stay untied across the two. *)
let together g s is =
  let+ x = g [| numel s + numel is |] in
  let cut lo n shape = Nx.reshape shape (Nx.slice [ Nx.R (lo, lo + n) ] x) in
  (cut 0 (numel s) s, cut (numel s) (numel is) is)

let scatter_case (mode : Nx_backend.scatter) =
  match mode with
  | `Set | `Add ->
      case ~dtypes:(reals @ complexes)
        ~complex:(scatter_instance mode (apart (ctensor Nx.complex128 plane)))
        (Scatter mode)
        (fun (D d) -> scatter_instance mode (apart (tensor d (range (-3.) 3.))))
  | `Max | `Min ->
      case
        ~finite:(scatter_instance mode (apart (tensor Nx.float64 tied)))
        (Scatter mode)
        (fun (D d) -> scatter_instance mode (together (distinct d)))

let update_instance g =
  let* s = shape ~dim:full_dim 1 3 in
  let* vs = array ~size:(constant (Array.length s)) (int_range 0 3) in
  let vs = Array.mapi (fun i d -> Int.min d s.(i)) vs in
  let* starts =
    Array.fold_right
      (fun hi acc ->
        let+ v = int_range 0 hi and+ acc = acc in
        Int64.of_int v :: acc)
      (Array.mapi (fun i d -> s.(i) - d) vs)
      (constant [])
  in
  let starts = Nx.create Nx.int64 [| Array.length s |] (Array.of_list starts) in
  let+ x = g s and+ v = g vs and+ tracked = patterns 2 in
  nary
    (Format.asprintf "update at %a" Nx.pp starts)
    (fun xs ->
      let x, v = two xs in
      [ Op.eval (Update (x, starts, v)) ])
    [ x; v ] tracked Zeroed

let update_case =
  case ~dtypes:(reals @ complexes)
    ~complex:(update_instance (ctensor Nx.complex128 plane))
    Update
    (fun (D d) -> update_instance (tensor d (range (-3.) 3.)))

(* Windows *)

type window = {
  kernel_size : int array;
  stride : int array;
  dilation : int array;
  padding : (int * int) array;
  spatial : int array;
}

let window =
  let* k = int_range 1 2 in
  let each g = array ~size:(constant k) g in
  let* kernel_size = each (int_range 1 2) in
  let* stride = each (int_range 1 3) in
  let* dilation = each (int_range 1 2) in
  let* padding = each (pair_of (int_range 0 1) (int_range 0 1)) in
  let* extra = each (int_range 0 3) in
  let spatial =
    Array.init k (fun i ->
        Int.max 1
          ((dilation.(i) * (kernel_size.(i) - 1))
          + 1 + extra.(i)
          - fst padding.(i)
          - snd padding.(i)))
  in
  constant { kernel_size; stride; dilation; padding; spatial }

let windows w =
  Array.fold_left ( * ) 1
    (Array.mapi
       (fun i n ->
         (n
         + fst w.padding.(i)
         + snd w.padding.(i)
         - ((w.dilation.(i) * (w.kernel_size.(i) - 1)) + 1))
         / w.stride.(i)
         + 1)
       w.spatial)

let pp_window ppf w =
  let ints a = String.concat "; " (Array.to_list (Array.map string_of_int a)) in
  Format.fprintf ppf "kernel [%s] stride [%s] dilation [%s] spatial [%s]"
    (ints w.kernel_size) (ints w.stride) (ints w.dilation) (ints w.spatial)

let unfold_instance g =
  let* w = window in
  let* lead = shape ~dim:full_dim 0 1 in
  let+ x = g (Array.append lead w.spatial) in
  let { kernel_size; stride; dilation; padding; _ } = w in
  let f x =
    [ Op.eval (Unfold { kernel_size; stride; dilation; padding; x = one x }) ]
  in
  instance ~linear:f (Format.asprintf "unfold %a" pp_window w) f [ x ]

let fold_instance g =
  let* w = window in
  let* lead = shape ~dim:full_dim 0 1 in
  let k = Array.fold_left ( * ) 1 w.kernel_size in
  let+ x = g (Array.append lead [| k; windows w |]) in
  let { kernel_size; stride; dilation; padding; spatial = output_size } = w in
  let f x =
    [
      Op.eval
        (Fold { output_size; kernel_size; stride; dilation; padding; x = one x });
    ]
  in
  instance ~linear:f (Format.asprintf "fold %a" pp_window w) f [ x ]

let unfold_case =
  case ~dtypes:(reals @ complexes)
    ~complex:(unfold_instance (ctensor Nx.complex128 plane))
    Unfold
    (fun (D d) -> unfold_instance (tensor d (range (-3.) 3.)))

let fold_case =
  case ~dtypes:(reals @ complexes)
    ~complex:(fold_instance (ctensor Nx.complex128 plane))
    Fold
    (fun (D d) -> fold_instance (tensor d (range (-3.) 3.)))

(* Products *)

let matmul_instance g =
  let* m = int_range 1 3 in
  let* k = int_range 0 3 in
  let* n = int_range 1 3 in
  (* Each operand's leading axes are a suffix of [lead], each axis of extent one
     where the operand broadcasts it, in either operand or both. *)
  let* lead = shape ~dim:(int_range 1 3) 0 2 in
  let r = Array.length lead in
  let side =
    let+ drop = int_range 0 r
    and+ ones =
      array ~size:(constant r)
        (frequency [ (2, constant false); (1, constant true) ])
    in
    Array.sub
      (Array.mapi (fun i d -> if ones.(i) then 1 else d) lead)
      drop (r - drop)
  in
  let* lead_a = side in
  let* lead_b = side in
  let+ a = g (Array.append lead_a [| m; k |])
  and+ b = g (Array.append lead_b [| k; n |])
  and+ tracked = patterns 2 in
  nary "matmul"
    (fun xs ->
      let a, b = two xs in
      [ Op.eval (Matmul (a, b)) ])
    [ a; b ] tracked
    (match tracked with [ true; true ] -> Not_linear | _ -> Coefficients)

let matmul_case =
  case ~dtypes:wide
    ~complex:(matmul_instance (ctensor Nx.complex128 plane))
    Matmul
    (fun (D d) -> matmul_instance (tensor d (range (-2.) 2.)))

(* Fourier transforms *)

let axes_of s =
  let+ axes = subsequence (List.init (Array.length s) Fun.id) in
  Array.of_list (if axes = [] then [ Array.length s - 1 ] else axes)

let pp_axes axes =
  String.concat "; " (Array.to_list (Array.map string_of_int axes))

let fft_instance d =
  let* s = shape ~dim:full_dim 1 3 in
  let* axes = axes_of s in
  let+ inverse = bool and+ x = ctensor d plane s in
  let f x = [ Op.eval (Fft { inverse; axes; x = one x }) ] in
  instance ~linear:f
    (Printf.sprintf "%s over [%s]"
       (if inverse then "ifft" else "fft")
       (pp_axes axes))
    f [ x ]

let fft_case =
  let smooth (D d) =
    match Nx_dtype.equal_witness d Nx.complex64 with
    | Some Type.Equal -> fft_instance Nx.complex64
    | None -> fft_instance Nx.complex128
  in
  case ~dtypes:complexes
    ~finite:(fft_instance Nx.complex128)
    ~complex:(fft_instance Nx.complex128)
    Fft smooth

let rfft_instance x_dtype dtype =
  let* s = shape ~dim:full_dim 1 3 in
  let* axes = axes_of s in
  let+ x = tensor x_dtype (range (-2.) 2.) s in
  let f x = [ Op.eval (Rfft { dtype; axes; x = one x }) ] in
  instance ~linear:f (Printf.sprintf "rfft over [%s]" (pp_axes axes)) f [ x ]

let rfft_case =
  case ~dtypes:[ D Nx.float32; D Nx.float64 ] Rfft (fun (D d) ->
      match Nx_dtype.equal_witness d Nx.float32 with
      | Some Type.Equal -> rfft_instance Nx.float32 Nx.complex64
      | None -> rfft_instance Nx.float64 Nx.complex128)

let irfft_instance x_dtype dtype =
  let* s = shape ~dim:full_dim 1 3 in
  let* axes = axes_of s in
  let* sizes =
    option
      (let+ last = int_range 1 6 in
       Array.mapi
         (fun i a -> if i = Array.length axes - 1 then last else s.(a))
         axes)
  in
  let+ x = ctensor x_dtype plane s in
  let f x = [ Op.eval (Irfft { dtype; axes; s = sizes; x = one x }) ] in
  instance ~linear:f
    (Printf.sprintf "irfft over [%s]%s" (pp_axes axes)
       (match sizes with None -> "" | Some s -> " to [" ^ pp_axes s ^ "]"))
    f [ x ]

let irfft_case =
  let smooth (D d) =
    match Nx_dtype.equal_witness d Nx.complex64 with
    | Some Type.Equal -> irfft_instance Nx.complex64 Nx.float32
    | None -> irfft_instance Nx.complex128 Nx.float64
  in
  case ~dtypes:complexes ~finite:(smooth (D Nx.complex128))
    ~complex:(smooth (D Nx.complex128)) Irfft smooth

let contiguous_case =
  let inst g =
    let* s = shape 0 3 in
    let+ x = g s in
    let f x = [ Op.eval (Contiguous (one x)) ] in
    instance ~linear:f "contiguous" f [ x ]
  in
  case ~dtypes:(reals @ complexes)
    ~complex:(inst (ctensor Nx.complex128 plane))
    Contiguous
    (fun (D d) -> inst (tensor d (range (-3.) 3.)))

(* Linear algebra. Matrices are built inside each factorization's domain. *)

let matrices ?(lead = shape ~dim:(int_range 1 2) 0 1) d n m g =
  let* lead = lead in
  g d (Array.append lead [| n; m |])

(* Values away from zero, so that no reflection of nx's QR meets a column that
   is already reduced, where its sign convention jumps. *)
let real_values d s = tensor d (away 0.1 1.) s

(* Both components away from zero, as {!real_values}: nx's complex QR also fixes
   its phase differently where a reflection's leading element is real. *)
let complex_values d s = ctensor d (away 0.1 1., away 0.1 1.) s
let adjoint x = Nx.conjugate (Nx.matrix_transpose x)
let scalar_of d v = Nx.full d [||] (Nx_dtype.of_float d v)

(* [B Bᴴ + m I] in the lower triangle, which the factorization reads in both
   modes, and noise above it. *)
let cholesky_instance values d =
  let* n = int_range 1 3 in
  let* upper = bool in
  let* b = matrices d n n values in
  let lead = Array.sub (Nx.shape b) 0 (Nx.ndim b - 2) in
  let+ shift = range 0.5 2.
  and+ noise = matrices ~lead:(constant lead) d n n values in
  let a =
    Nx.add (Nx.matmul b (adjoint b)) (Nx.mul (Nx.eye d n) (scalar_of d shift))
  in
  let x = Nx.add (Nx.tril a) (Nx.triu ~k:1 noise) in
  let f x = [ Op.eval (Cholesky { upper; x = one x }) ] in
  instance
    (Printf.sprintf "cholesky%s" (if upper then " upper" else ""))
    f [ x ]

let cholesky_case =
  case ~dtypes:wide ~complex:(cholesky_instance complex_values Nx.complex128)
    Cholesky (fun (D d) -> cholesky_instance real_values d)

(* [Q diag (s) Vᴴ] with [s] in [[0.5, 2]]: of full rank, with a condition number
   of at most 4. *)
let conditioned values d m n =
  let k = Int.min m n in
  let* lead = shape ~dim:(int_range 1 2) 0 1 in
  let+ u = values d (Array.append lead [| m; k |])
  and+ v = values d (Array.append lead [| n; k |])
  and+ s = tensor d (range 0.5 2.) (Array.append lead [| k |]) in
  let q x = fst (Nx.qr x) in
  let scaled = Nx.mul (q u) (Nx.unsqueeze ~axes:[ -2 ] s) in
  Nx.matmul scaled (adjoint (q v))

let qr_instance values d =
  let* m = int_range 1 4 in
  let* n = int_range 1 4 in
  let* reduced = if m > n then constant true else bool in
  let+ x =
    if m >= n then conditioned values d m n
    else
      let* left = conditioned values d m m in
      let lead = Array.sub (Nx.shape left) 0 (Nx.ndim left - 2) in
      let+ right = values d (Array.append lead [| m; n - m |]) in
      Nx.concatenate ~axis:(-1) [ left; right ]
  in
  let f x =
    let q, r = Op.eval (Qr { reduced; x = one x }) in
    [ q; r ]
  in
  instance
    (Printf.sprintf "qr %s" (if reduced then "reduced" else "complete"))
    f [ x ]

let qr_case =
  case ~dtypes:wide ~complex:(qr_instance complex_values Nx.complex128) Qr
    (fun (D d) -> qr_instance real_values d)

let diag_matrix d n v = Nx.mul (Nx.eye d n) (Nx.unsqueeze ~axes:[ -2 ] v)

(* [Pᵀ L U] of [m] rows and [n] columns, [L] unit lower trapezoidal with [|l_ij|
   <= 0.25] and [U] upper trapezoidal with a diagonal in [[1, 2]]: partial
   pivoting picks the row of [L]'s one at each step, by a margin of four, so a
   small change of the matrix changes no pivot. *)
let lu_instance values d =
  let* m = int_range 1 3 in
  let* n = int_range 1 3 in
  let k = Int.min m n in
  let* lead = shape ~dim:(int_range 1 2) 0 1 in
  let* perm = permutation (List.init m Fun.id) in
  let+ l = values d (Array.append lead [| m; k |])
  and+ u = values d (Array.append lead [| k; n |])
  and+ diag = tensor d (away 1. 2.) (Array.append lead [| k |]) in
  let quarter = Nx_dtype.of_float d 0.25 in
  let l = Nx.add (Nx.mul_s (Nx.tril ~k:(-1) l) quarter) (Nx.eye ~m:k d m) in
  let u =
    Nx.add
      (Nx.mul_s (Nx.triu ~k:1 u) quarter)
      (Nx.mul (Nx.eye ~m:n d k) (Nx.unsqueeze ~axes:[ -1 ] diag))
  in
  let rows =
    Nx.create Nx.int64 [| m |] (Array.of_list (List.map Int64.of_int perm))
  in
  let x = Nx.take ~axis:(-2) ~indices:rows (Nx.matmul l u) in
  let f x =
    let packed, _, _ = Op.eval (Lu (one x)) in
    [ packed ]
  in
  instance (Printf.sprintf "lu of %d x %d" m n) f [ x ]

let lu_case =
  case ~dtypes:wide ~complex:(lu_instance complex_values Nx.complex128) Lu
    (fun (D d) -> lu_instance real_values d)

(* The triangle the solve reads has a diagonal in [[1, 2]] and small entries;
   the other triangle, and the diagonal under [unit_diag], hold noise ten times
   larger, which the solve never reads. *)
let solve_instance values d =
  let* n = int_range 1 3 in
  let* upper = bool in
  let* transpose = bool in
  let* unit_diag = bool in
  let* lead = shape ~dim:(int_range 1 2) 0 1 in
  let* vector = bool in
  let* k = int_range 1 2 in
  let b_shape = Array.append lead (if vector then [| n |] else [| n; k |]) in
  let square = Array.append lead [| n; n |] in
  let+ raw = values d square
  and+ noise = values d square
  and+ diag = tensor d (away 1. 2.) (Array.append lead [| n |])
  and+ b = values d b_shape
  and+ tracked = patterns 2 in
  let scale v x = Nx.mul_s x (Nx_dtype.of_float d v) in
  let read, unread =
    if upper then ((fun x -> Nx.triu ~k:1 x), fun x -> Nx.tril ~k:(-1) x)
    else ((fun x -> Nx.tril ~k:(-1) x), fun x -> Nx.triu ~k:1 x)
  in
  let diagonal =
    if unit_diag then scale 10. (Nx.mul (Nx.eye d n) noise)
    else diag_matrix d n diag
  in
  let a =
    Nx.add
      (Nx.add (read (scale (0.5 /. float_of_int n) raw)) diagonal)
      (unread (scale 10. noise))
  in
  nary
    (Printf.sprintf "solve_triangular%s%s%s%s"
       (if upper then " upper" else " lower")
       (if transpose then " transposed" else "")
       (if unit_diag then " unit_diag" else "")
       (if vector then ", vector" else ""))
    (fun xs ->
      let a, b = two xs in
      [ Op.eval (Solve_triangular { upper; transpose; unit_diag; a; b }) ])
    [ a; b ] tracked
    (match tracked with [ false; true ] -> Coefficients | _ -> Not_linear)

let solve_case =
  case ~dtypes:wide ~complex:(solve_instance complex_values Nx.complex128)
    Solve_triangular (fun (D d) -> solve_instance real_values d)

(* The spectral factorizations. Their vectors are compared through forms their
   sign or phase cancels from, so that a central difference of nx's factors,
   whose phase nx does not fix, can judge them: [A Aᴴ] and [Aᴴ A] from the
   singular vectors, [Q Λ Qᴴ] from the eigenvectors of a Hermitian matrix, and
   [Σ λₖ vₖ vₖᴴ] from those of a general one, whose eigenvalues, which nx does
   not order, enter as power sums. *)

(* [separated d lead n] is [n] values per matrix at least [0.25] apart in [[0.3,
   1.25]], in a shuffled order. *)
let separated d lead n =
  let+ blocks =
    array
      ~size:(constant (numel lead))
      (let* order = permutation (List.init n Fun.id) in
       let+ noise = array ~size:(constant n) (range 0. 0.05) in
       Array.of_list
         (List.mapi
            (fun i k -> 0.3 +. (0.3 *. float_of_int k) +. noise.(i))
            order))
  in
  Nx.cast d
    (Nx.create Nx.float64
       (Array.append lead [| n |])
       (Array.concat (Array.to_list blocks)))

let orthonormal x = fst (Nx.qr x)
let columns_scaled q s = Nx.mul q (Nx.unsqueeze ~axes:[ -2 ] s)

let svd_instance values d =
  let* m = int_range 1 3 in
  let* n = int_range 1 3 in
  let k = Int.min m n in
  let* lead = shape ~dim:(int_range 1 2) 0 1 in
  let+ u = values d (Array.append lead [| m; k |])
  and+ v = values d (Array.append lead [| n; k |])
  and+ s = separated d lead k in
  let x =
    Nx.matmul (columns_scaled (orthonormal u) s) (adjoint (orthonormal v))
  in
  let f x =
    let u, s, vh = Op.eval (Svd { full_matrices = false; x = one x }) in
    let s = Nx.cast (Nx.dtype u) s in
    let us = columns_scaled u s and vs = columns_scaled (adjoint vh) s in
    [ s; Nx.matmul us (adjoint us); Nx.matmul vs (adjoint vs); Nx.matmul us vh ]
  in
  instance ~extra:None (Printf.sprintf "svd of %d x %d" m n) f [ x ]

let eigh_instance values d =
  let* n = int_range 1 3 in
  let* lead = shape ~dim:(int_range 1 2) 0 1 in
  let* vectors = bool in
  let+ q = values d (Array.append lead [| n; n |])
  and+ w = separated d lead n
  and+ signs = tensor d (of_list [ -1.; 1. ]) (Array.append lead [| n |]) in
  let q = orthonormal q in
  let x = Nx.matmul (columns_scaled q (Nx.mul w signs)) (adjoint q) in
  let x = Nx.mul_s (Nx.add x (adjoint x)) (Nx_dtype.of_float d 0.5) in
  (* eigh's domain is the Hermitian matrices: every direction is taken through
     its Hermitian part. *)
  let f x =
    let x = one x in
    let x =
      Nx.mul_s (Nx.add x (adjoint x)) (Nx_dtype.of_float (Nx.dtype x) 0.5)
    in
    let w, q = Op.eval (Eigh { vectors; x }) in
    let w = Nx.cast (Nx.dtype x) w in
    match q with
    | None -> [ w ]
    | Some q -> [ w; Nx.matmul (columns_scaled q w) (adjoint q) ]
  in
  instance ~extra:None
    (Printf.sprintf "eigh of %d x %d%s" n n
       (if vectors then "" else ", values"))
    f [ x ]

(* [V B V⁻¹] with [V = I + 0.3 Q], [Q] orthonormal, so that [V]'s condition
   number is at most 13/7, and [B] the eigenvalues on a diagonal; on a real
   matrix, when [pair], its first two rows hold [[λ₀, -0.4], [0.4, λ₀]], whose
   eigenvalues are the complex pair [λ₀ ± 0.4i]. *)
let eig_instance values d =
  let* n = int_range 1 3 in
  let* vectors = bool in
  let* pair =
    if n >= 2 && not (Nx_dtype.is_complex d) then bool else constant false
  in
  let+ q = values d [| n; n |] and+ w = separated Nx.float64 [||] n in
  let w = Nx.to_array w in
  let b =
    Array.init (n * n) (fun p ->
        let i = p / n and j = p mod n in
        match (i, j) with
        | 0, 1 when pair -> -0.4
        | 1, 0 when pair -> 0.4
        | 1, 1 when pair -> w.(0)
        | _ -> if i = j then w.(i) else 0.)
  in
  let v =
    Nx.add (Nx.eye d n) (Nx.mul_s (orthonormal q) (Nx_dtype.of_float d 0.3))
  in
  let x =
    Nx.matmul
      (Nx.matmul v (Nx.cast d (Nx.create Nx.float64 [| n; n |] b)))
      (Nx.inv v)
  in
  let f x =
    let w, vecs = Op.eval (Eig { vectors; x = one x }) in
    let power k = Nx.sum ~axes:[ -1 ] ~keepdims:true (Nx.pow_s w k) in
    let sums =
      Nx.concatenate ~axis:(-1)
        [
          power Complex.one;
          power { re = 2.; im = 0. };
          power { re = 3.; im = 0. };
        ]
    in
    match vecs with
    | None -> [ sums ]
    | Some v -> [ sums; Nx.matmul (columns_scaled v w) (adjoint v) ]
  in
  instance ~extra:None
    (Printf.sprintf "eig of %d x %d%s%s" n n
       (if vectors then "" else ", values")
       (if pair then ", a complex pair" else ""))
    f [ x ]

let svd_case =
  case ~dtypes:wide ~complex:(svd_instance complex_values Nx.complex128)
    ~difference:(1e-6, 1e-6) Svd (fun (D d) -> svd_instance real_values d)

let eigh_case =
  case ~dtypes:wide ~complex:(eigh_instance complex_values Nx.complex128)
    ~difference:(1e-6, 1e-6) Eigh (fun (D d) -> eigh_instance real_values d)

let eig_case =
  case ~dtypes:wide ~complex:(eig_instance complex_values Nx.complex128)
    ~difference:(1e-3, 1e-3) Eig (fun (D d) -> eig_instance real_values d)

(* Movements *)

let ints a = String.concat "; " (Array.to_list (Array.map string_of_int a))

let move_instance (m : Row.move) g =
  let at desc move =
    let* s = shape 0 3 in
    let* move = move s in
    let+ x = g s in
    let f x = [ Op.eval (Move (one x, move)) ] in
    instance ~linear:f (desc move) f [ x ]
  in
  let pp_move : Nx.Op.move -> string = function
    | Reshape s -> "reshape to [" ^ ints s ^ "]"
    | Expand s -> "expand to [" ^ ints s ^ "]"
    | Permute p -> "permute [" ^ ints p ^ "]"
    | Shrink r ->
        "shrink ["
        ^ String.concat "; "
            (Array.to_list
               (Array.map (fun (a, b) -> Printf.sprintf "%d, %d" a b) r))
        ^ "]"
    | Flip f ->
        "flip ["
        ^ String.concat "; " (Array.to_list (Array.map string_of_bool f))
        ^ "]"
    | Window { axis; size; step } ->
        Printf.sprintf "window along %d of %d every %d" axis size step
  in
  match m with
  | Reshape ->
      at pp_move (fun s ->
          let n = numel s and r = Array.length s in
          let merged =
            if r >= 2 then
              Array.append [| s.(0) * s.(1) |] (Array.sub s 2 (r - 2))
            else s
          in
          of_list
            [
              Nx.Op.Reshape [| n |];
              Reshape (Array.of_list (List.rev (Array.to_list s)));
              Reshape (Array.append [| 1 |] s);
              Reshape merged;
            ])
  | Expand ->
      let* target = shape ~dim:full_dim 0 3 in
      let* ones = array ~size:(constant (Array.length target)) bool in
      let s = Array.mapi (fun i d -> if ones.(i) then 1 else d) target in
      let+ x = g s in
      let move = Nx.Op.Expand target in
      let f x = [ Op.eval (Move (one x, move)) ] in
      instance ~linear:f (pp_move move) f [ x ]
  | Permute ->
      at pp_move (fun s ->
          let+ p = permutation (List.init (Array.length s) Fun.id) in
          Nx.Op.Permute (Array.of_list p))
  | Shrink ->
      at pp_move (fun s ->
          let+ r =
            Array.fold_right
              (fun n acc ->
                let+ lo = int_range 0 n
                and+ len = int_range 0 n
                and+ acc = acc in
                let lo = Int.min lo n in
                (lo, Int.min n (lo + len)) :: acc)
              s (constant [])
          in
          Nx.Op.Shrink (Array.of_list r))
  | Flip ->
      at pp_move (fun s ->
          let+ f = array ~size:(constant (Array.length s)) bool in
          Nx.Op.Flip f)
  | Window ->
      let* s = shape ~dim:full_dim 1 3 in
      let* axis = int_range 0 (Array.length s - 1) in
      let* size = int_range 1 s.(axis) in
      let* step = int_range 1 (size + 2) in
      let+ x = g s in
      let move = Nx.Op.Window { axis; size; step } in
      let f x = [ Op.eval (Move (one x, move)) ] in
      instance ~linear:f (pp_move move) f [ x ]

let move_case m =
  case ~dtypes:(reals @ complexes)
    ~complex:(move_instance m (ctensor Nx.complex128 plane))
    (Move m)
    (fun (D d) -> move_instance m (tensor d (range (-3.) 3.)))

(* Test devices over host memory, so that a value can be placed off the host. *)
let d1 = Nx.Device.v (Cpu 1)
let d2 = Nx.Device.v (Cpu 2)

let placements =
  [
    ("on one device", fun _ -> Some (Nx.Placement.on d1));
    ("replicated on two", fun _ -> Some (Nx.Placement.replicated [ d1; d2 ]));
    ( "split by rows over two",
      fun s ->
        if Array.length s > 0 && s.(0) mod 2 = 0 then
          Some (Nx.Placement.sharded ~axis:0 [ d1; d2 ])
        else None );
  ]

let place_case =
  let inst g =
    let* s = shape 0 3 in
    let* name, p =
      of_list ~pp:(fun ppf (n, _) -> Format.pp_print_string ppf n) placements
    in
    let+ x = g s in
    let p = Option.value (p s) ~default:(Nx.Placement.on d1) in
    let f x = [ Op.eval (Place (p, one x)) ] in
    instance ~linear:f ("place " ^ name) f [ x ]
  in
  case ~dtypes:(reals @ complexes)
    ~complex:(inst (ctensor Nx.complex128 plane))
    Place
    (fun (D d) -> inst (tensor d (range (-3.) 3.)))

(* The table *)

let of_row : Row.t -> t = function
  | Unary k -> unary_case k
  | Binary k -> binary_case k
  | Compare k -> compare_case k
  | Where -> where_case
  | Fma -> fma_case
  | Reduce k -> reduce_case k
  | Scan k -> scan_case k
  | Arg_reduce k -> arg_reduce_case k
  | Sort -> sort_case
  | Argsort -> argsort_case
  | Group -> group_case
  | Pad -> pad_case
  | Cat -> cat_case
  | Cast -> cast_case
  | Bitcast -> bitcast_case
  | Threefry -> threefry_case
  | Gather -> gather_case
  | Scatter mode -> scatter_case mode
  | Update -> update_case
  | Unfold -> unfold_case
  | Fold -> fold_case
  | Matmul -> matmul_case
  | Fft -> fft_case
  | Rfft -> rfft_case
  | Irfft -> irfft_case
  | Contiguous -> contiguous_case
  | Cholesky -> cholesky_case
  | Qr -> qr_case
  | Lu -> lu_case
  | Svd -> svd_case
  | Eig -> eig_case
  | Eigh -> eigh_case
  | Solve_triangular -> solve_case
  | Move m -> move_case m
  | Place -> place_case
  | Read -> invalid_arg "Case.of_row: a read's result is a buffer"
  | Check -> invalid_arg "Case.of_row: a check has no result"

let all =
  List.map of_row
    (List.filter (fun r -> r <> Row.Read && r <> Row.Check) Row.all)
