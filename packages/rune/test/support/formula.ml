(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

type unary = Sin | Tanh | Exp_tanh | Neg | Log1p_sq | Abs | Relu
type binary = Add | Sub | Mul | Div_safe | Maximum

type term =
  | X
  | Const of int array * float array
  | Un of unary * term
  | Bin of binary * term * term
  | Where of int array * bool array * term * term
  | Sum of int * bool * term
  | Max of int * term
  | Permute of int list * term
  | Reshape of int array * term
  | Flip of int * term
  | Pad of (int * int) array * term
  | Shrink of (int * int) array * term
  | Expand of int * term
  | Concat of int * term * term
  | Matmul of term * term
  | Take of int * int array * term

type t = { term : term; input : int array; output : int array }

(* Evaluation *)

let unary op x =
  match op with
  | Sin -> Nx.sin x
  | Tanh -> Nx.tanh x
  | Exp_tanh -> Nx.exp (Nx.tanh x)
  | Neg -> Nx.neg x
  | Log1p_sq -> Nx.log (Nx.add_s (Nx.mul x x) 1.)
  | Abs -> Nx.abs x
  | Relu -> Nx.relu x

let binary op a b =
  match op with
  | Add -> Nx.add a b
  | Sub -> Nx.sub a b
  | Mul -> Nx.mul a b
  | Div_safe -> Nx.div a (Nx.add_s (Nx.mul b b) 1.)
  | Maximum -> Nx.maximum a b

let rec term t x =
  match t with
  | X -> x
  | Const (s, c) -> Nx.create Nx.float64 s c
  | Un (op, p) -> unary op (term p x)
  | Bin (op, p, q) -> binary op (term p x) (term q x)
  | Where (s, m, p, q) -> Nx.where (Nx.create Nx.bool s m) (term p x) (term q x)
  | Sum (axis, keepdims, p) -> Nx.sum ~axes:[ axis ] ~keepdims (term p x)
  | Max (axis, p) -> Nx.max ~axes:[ axis ] (term p x)
  | Permute (axes, p) -> Nx.transpose ~axes (term p x)
  | Reshape (s, p) -> Nx.reshape s (term p x)
  | Flip (axis, p) -> Nx.flip ~axes:[ axis ] (term p x)
  | Pad (widths, p) -> Nx.pad widths 0. (term p x)
  | Shrink (ranges, p) -> Nx.shrink ranges (term p x)
  | Expand (n, p) ->
      let y = term p x in
      Nx.broadcast_to (Array.append [| n |] (Nx.shape y)) y
  | Concat (axis, p, q) -> Nx.concatenate ~axis [ term p x; term q x ]
  | Matmul (p, q) -> Nx.matmul (term p x) (term q x)
  | Take (axis, indices, p) ->
      let indices =
        Nx.create Nx.int64
          [| Array.length indices |]
          (Array.map Int64.of_int indices)
      in
      Nx.take ~axis ~indices (term p x)

let eval p x = term p.term x
let objective p x = Nx.sum (eval p x)

(* Families *)

let all_families =
  [ "elementwise"; "reduction"; "movement"; "matmul"; "where"; "take" ]

let rec uses acc = function
  | X | Const _ -> acc
  | Un (_, p) -> uses ("elementwise" :: acc) p
  | Bin (_, p, q) -> uses (uses ("elementwise" :: acc) p) q
  | Where (_, _, p, q) -> uses (uses ("where" :: acc) p) q
  | Sum (_, _, p) | Max (_, p) -> uses ("reduction" :: acc) p
  | Permute (_, p)
  | Reshape (_, p)
  | Flip (_, p)
  | Pad (_, p)
  | Shrink (_, p)
  | Expand (_, p) ->
      uses ("movement" :: acc) p
  | Concat (_, p, q) -> uses (uses ("movement" :: acc) p) q
  | Matmul (p, q) -> uses (uses ("matmul" :: acc) p) q
  | Take (_, _, p) -> uses ("take" :: acc) p

let families p =
  let used = uses [] p.term in
  List.filter (fun f -> List.mem f used) all_families

(* Printing *)

let unary_name = function
  | Sin -> "sin"
  | Tanh -> "tanh"
  | Exp_tanh -> "exp∘tanh"
  | Neg -> "neg"
  | Log1p_sq -> "log1p_sq"
  | Abs -> "abs"
  | Relu -> "relu"

let binary_name = function
  | Add -> "+"
  | Sub -> "-"
  | Mul -> "*"
  | Div_safe -> "/(1+_²)"
  | Maximum -> "max"

let pp_list pp_elt ppf l =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
    pp_elt ppf l

let pp_shape ppf s =
  Format.fprintf ppf "[%a]" (pp_list Format.pp_print_int) (Array.to_list s)

let pp_pair ppf (a, b) = Format.fprintf ppf "(%d, %d)" a b

let rec pp_term ppf = function
  | X -> Format.pp_print_string ppf "x"
  | Const (s, c) ->
      Format.fprintf ppf "C%a[%a]" pp_shape s
        (pp_list Format.pp_print_float)
        (Array.to_list c)
  | Un (op, p) -> Format.fprintf ppf "%s(%a)" (unary_name op) pp_term p
  | Bin (op, p, q) ->
      Format.fprintf ppf "(%a %s %a)" pp_term p (binary_name op) pp_term q
  | Where (s, m, p, q) ->
      Format.fprintf ppf "where(M%a[%a], %a, %a)" pp_shape s
        (pp_list Format.pp_print_bool)
        (Array.to_list m) pp_term p pp_term q
  | Sum (axis, keepdims, p) ->
      Format.fprintf ppf "sum[%d%s](%a)" axis
        (if keepdims then ", keepdims" else "")
        pp_term p
  | Max (axis, p) -> Format.fprintf ppf "max[%d](%a)" axis pp_term p
  | Permute (axes, p) ->
      Format.fprintf ppf "permute[%a](%a)"
        (pp_list Format.pp_print_int)
        axes pp_term p
  | Reshape (s, p) -> Format.fprintf ppf "reshape%a(%a)" pp_shape s pp_term p
  | Flip (axis, p) -> Format.fprintf ppf "flip[%d](%a)" axis pp_term p
  | Pad (w, p) ->
      Format.fprintf ppf "pad[%a](%a)" (pp_list pp_pair) (Array.to_list w)
        pp_term p
  | Shrink (r, p) ->
      Format.fprintf ppf "shrink[%a](%a)" (pp_list pp_pair) (Array.to_list r)
        pp_term p
  | Expand (n, p) -> Format.fprintf ppf "expand[%d](%a)" n pp_term p
  | Concat (axis, p, q) ->
      Format.fprintf ppf "concat[%d](%a, %a)" axis pp_term p pp_term q
  | Matmul (p, q) -> Format.fprintf ppf "(%a @ %a)" pp_term p pp_term q
  | Take (axis, i, p) ->
      Format.fprintf ppf "take[%d; %a](%a)" axis
        (pp_list Format.pp_print_int)
        (Array.to_list i) pp_term p

let pp ppf p = Format.fprintf ppf "x%a ↦ %a" pp_shape p.input pp_term p.term

(* Generation *)

let numel s = Array.fold_left ( * ) 1 s

(* A program's values stay small enough to evaluate many times a test. *)
let max_numel = 192
let max_rank = 4
let element = Gen.float_range (-2.) 2.

(* An axis of zero elements is drawn about once in ten, one of one element about
   twice. *)
let dim =
  Gen.frequency
    [ (1, Gen.constant 0); (2, Gen.constant 1); (7, Gen.int_range 2 3) ]

let shape =
  Gen.bind (Gen.int_range 0 3) (fun r -> Gen.array ~size:(Gen.constant r) dim)

let floats n = Gen.array ~size:(Gen.constant n) element

(* [s] with each of its axes kept or made one element, for a broadcast operand,
   and its leading axes possibly dropped. *)
let broadcastable s =
  let open Gen in
  let* drop = int_range 0 (Array.length s) in
  let+ ones = array ~size:(constant (Array.length s - drop)) bool in
  Array.mapi (fun i one -> if one then 1 else s.(drop + i)) ones

let const s = Gen.map (fun c -> Const (s, c)) (floats (numel s))

let unaries ~kinks =
  Gen.of_list
    ((if kinks then [ Abs; Relu ] else [])
    @ [ Sin; Tanh; Exp_tanh; Neg; Log1p_sq ])

let binaries ~kinks =
  Gen.of_list ((if kinks then [ Maximum ] else []) @ [ Add; Sub; Mul; Div_safe ])

(* The other branch of a selection: a constant, or the program itself through a
   unary operation, so that its value reaches the result twice. *)
let other ~kinks p s =
  let open Gen in
  one_of [ const s; map (fun op -> Un (op, p)) (unaries ~kinks) ]

let remove i s =
  Array.of_list (List.filteri (fun j _ -> j <> i) (Array.to_list s))

(* The steps that apply to a program [p] of shape [s], each with its weight and
   the program and shape it makes. *)
let steps ~kinks p s =
  let open Gen in
  let r = Array.length s in
  let axis = int_range 0 (max 0 (r - 1)) in
  let when_ c l = if c then l () else [] in
  List.concat
    [
      [
        (3, map (fun op -> (Un (op, p), s)) (unaries ~kinks));
        ( 3,
          let* op = binaries ~kinks in
          let* q = bind (broadcastable s) const in
          let+ left = bool in
          ((if left then Bin (op, q, p) else Bin (op, p, q)), s) );
        ( 2,
          let* op = binaries ~kinks in
          let+ u = unaries ~kinks in
          (Bin (op, p, Un (u, p)), s) );
        ( 1,
          let* m =
            bind (broadcastable s) (fun ms ->
                map (fun b -> (ms, b)) (array ~size:(constant (numel ms)) bool))
          in
          let+ q = other ~kinks p s in
          (Where (fst m, snd m, p, q), s) );
        ( 1,
          let+ n = int_range 0 2 in
          (Expand (n, p), Array.append [| n |] s) );
      ];
      when_ (r >= 1) (fun () ->
          [
            ( 2,
              let* a = axis in
              let+ keepdims = bool in
              ( Sum (a, keepdims, p),
                if keepdims then
                  Array.mapi (fun i d -> if i = a then 1 else d) s
                else remove a s ) );
            (1, map (fun a -> (Flip (a, p), s)) axis);
            ( 1,
              let+ w =
                array ~size:(constant r) (pair (int_range 0 1) (int_range 0 1))
              in
              (Pad (w, p), Array.mapi (fun i d -> d + fst w.(i) + snd w.(i)) s)
            );
            ( 1,
              let+ ranges =
                array ~size:(constant r) (pair (int_range 0 3) (int_range 0 3))
              in
              let ranges =
                Array.mapi
                  (fun i (a, b) ->
                    let a = min a s.(i) and b = min b s.(i) in
                    (min a b, max a b))
                  ranges
              in
              (Shrink (ranges, p), Array.map (fun (a, b) -> b - a) ranges) );
            ( 1,
              let* a = axis in
              let concat q n =
                ( Concat (a, p, q),
                  Array.mapi (fun i d -> if i = a then d + n else d) s )
              in
              one_of
                [
                  (let* n = dim in
                   let+ q =
                     const (Array.mapi (fun i d -> if i = a then n else d) s)
                   in
                   concat q n);
                  map (fun u -> concat (Un (u, p)) s.(a)) (unaries ~kinks);
                ] );
            ( 1,
              let* a = axis in
              let* k = int_range 0 3 in
              let+ indices =
                array ~size:(constant k)
                  (frequency
                     [
                       (6, int_range 0 (max 0 (s.(a) - 1)));
                       (1, constant s.(a));
                       (1, constant (-1));
                     ])
              in
              ( Take (a, indices, p),
                Array.mapi (fun i d -> if i = a then k else d) s ) );
            ( 1,
              let+ shape =
                of_list
                  ([
                     [| numel s |];
                     Array.append [| 1 |] s;
                     Array.append s [| 1 |];
                   ]
                  @ when_ (r >= 2) (fun () ->
                      [
                        Array.append [| s.(0) * s.(1) |] (Array.sub s 2 (r - 2));
                      ]))
              in
              (Reshape (shape, p), shape) );
            ( 2,
              let* n = dim in
              let+ q = const [| s.(r - 1); n |] in
              (Matmul (p, q), Array.append (Array.sub s 0 (r - 1)) [| n |]) );
          ]);
      (* A maximum along an empty axis raises. *)
      when_
        (kinks && Array.exists (fun d -> d > 0) s)
        (fun () ->
          let nonempty =
            List.filter (fun a -> s.(a) > 0) (List.init r Fun.id)
          in
          [
            ( 1,
              let+ a = of_list nonempty in
              (Max (a, p), remove a s) );
          ]);
      when_ (r >= 2) (fun () ->
          [
            ( 1,
              let+ axes = permutation (List.init r Fun.id) in
              (Permute (axes, p), Array.of_list (List.map (fun i -> s.(i)) axes))
            );
            ( 1,
              let swap =
                List.init r (fun i ->
                    if i = r - 1 then r - 2 else if i = r - 2 then r - 1 else i)
              in
              constant
                ( Matmul (p, Permute (swap, p)),
                  Array.append (Array.sub s 0 (r - 1)) [| s.(r - 2) |] ) );
          ]);
    ]

let fits s = Array.length s <= max_rank && numel s <= max_numel

let rec grow ~kinks n (p, s) =
  if n = 0 then Gen.constant (p, s)
  else
    let options =
      List.map
        (fun (w, g) ->
          (w, Gen.map (fun ((_, s') as r) -> if fits s' then Some r else None) g))
        (steps ~kinks p s)
    in
    Gen.bind (Gen.frequency options) (function
      | Some r -> grow ~kinks (n - 1) r
      | None -> grow ~kinks (n - 1) (p, s))

let program ~kinks =
  let open Gen in
  let* input = shape in
  let* n = int_range 1 5 in
  let+ term, output = grow ~kinks n (X, input) in
  { term; input; output }

let gen = Gen.with_pp pp (program ~kinks:true)
let smooth = Gen.with_pp pp (program ~kinks:false)
let pp_point ppf x = Nx.pp ppf x

let point s =
  Gen.with_pp pp_point
    (Gen.map (fun a -> Nx.create Nx.float64 s a) (floats (numel s)))

let points n s =
  let s = Array.append [| n |] s in
  Gen.with_pp pp_point
    (Gen.map (fun a -> Nx.create Nx.float64 s a) (floats (numel s)))
