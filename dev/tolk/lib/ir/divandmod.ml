(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Division and modulo folding for index-typed expressions. *)

module K = Kernel

(* Int64 arithmetic *)

let floordiv a b =
  let q = Int64.div a b and r = Int64.rem a b in
  if r <> 0L && Int64.compare a 0L < 0 <> (Int64.compare b 0L < 0) then
    Int64.sub q 1L
  else q

let floormod a b = Int64.sub a (Int64.mul (floordiv a b) b)

(* C-style truncation division and remainder (rounds toward zero),
   matching the IR's Idiv constant folding (Int64.div). *)
let cdiv a b = Int64.div a b
let cmod a b = Int64.rem a b

let rec gcd a b =
  let a = Int64.abs a in
  let b = Int64.abs b in
  if b = 0L then a else gcd b (Int64.rem a b)

let min4 a b c d = min a (min b (min c d))
let max4 a b c d = max a (max b (max c d))

(* Node helpers *)

let iconst v = K.const (Const.int64 Dtype.index v)

let const_int_val node =
  match K.view node with
  | Const { value; _ } -> (
      match Const.view value with Int v -> Some v | _ -> None)
  | _ -> None

(* Recursive interval analysis: compute the tightest lower (vmin) and upper
   (vmax) bounds of an index expression. For Mul, Idiv, and Mod the extremes
   can occur at any corner of the two operands' ranges, so we evaluate all
   four products/quotients and take the min/max respectively. Division and
   modulo bail to Int64.min/max_int when the divisor range spans zero. *)

let rec vmin node =
  match K.view node with
  | Const { value; _ } -> (
      match Const.view value with
      | Int v -> v
      | Bool b -> if b then 1L else 0L
      | Float _ -> Int64.min_int)
  | Range _ -> 0L
  | Define_var { lo; _ } -> Int64.of_int lo
  | Special _ -> 0L
  | Binary { op = `Add; lhs; rhs; _ } -> Int64.add (vmin lhs) (vmin rhs)
  | Binary { op = `Sub; lhs; rhs; _ } -> Int64.sub (vmin lhs) (vmax rhs)
  | Binary { op = `Mul; lhs; rhs; _ } ->
      let a = vmin lhs and b = vmax lhs and c = vmin rhs and d = vmax rhs in
      min4 (Int64.mul a c) (Int64.mul a d) (Int64.mul b c) (Int64.mul b d)
  | Binary { op = `Idiv; lhs; rhs; _ } ->
      if vmin rhs > 0L then
        let xlo = vmin lhs and xhi = vmax lhs
        and ylo = vmin rhs and yhi = vmax rhs in
        min4 (Int64.div xlo ylo) (Int64.div xlo yhi)
          (Int64.div xhi ylo) (Int64.div xhi yhi)
      else Int64.min_int
  | Binary { op = `Mod; lhs; _ } ->
      if vmin lhs >= 0L then 0L else Int64.min_int
  | Binary { op = `Max; lhs; rhs; _ } -> max (vmin lhs) (vmin rhs)
  | Binary { op = `And; lhs; rhs; _ } ->
      if vmin lhs >= 0L && vmin rhs >= 0L then 0L else Int64.min_int
  | Unary { op = `Neg; src; _ } -> Int64.neg (vmax src)
  | Cast { src; dtype } ->
      let dt = Dtype.any_to_val dtype in
      if Dtype.is_int dt then vmin src else Int64.min_int
  | Ternary { op = `Where; b; c; _ } -> min (vmin b) (vmin c)
  | _ -> Int64.min_int

and vmax node =
  match K.view node with
  | Const { value; _ } -> (
      match Const.view value with
      | Int v -> v
      | Bool b -> if b then 1L else 0L
      | Float _ -> Int64.max_int)
  | Range { size; _ } -> Int64.sub (vmax size) 1L
  | Define_var { hi; _ } -> Int64.of_int hi
  | Special { size; _ } -> Int64.sub (vmax size) 1L
  | Binary { op = `Add; lhs; rhs; _ } -> Int64.add (vmax lhs) (vmax rhs)
  | Binary { op = `Sub; lhs; rhs; _ } -> Int64.sub (vmax lhs) (vmin rhs)
  | Binary { op = `Mul; lhs; rhs; _ } ->
      let a = vmin lhs and b = vmax lhs and c = vmin rhs and d = vmax rhs in
      max4 (Int64.mul a c) (Int64.mul a d) (Int64.mul b c) (Int64.mul b d)
  | Binary { op = `Idiv; lhs; rhs; _ } ->
      if vmin rhs > 0L then
        let xlo = vmin lhs and xhi = vmax lhs
        and ylo = vmin rhs and yhi = vmax rhs in
        max4 (Int64.div xlo ylo) (Int64.div xlo yhi)
          (Int64.div xhi ylo) (Int64.div xhi yhi)
      else Int64.max_int
  | Binary { op = `Mod; lhs; rhs; _ } ->
      if vmin lhs >= 0L then min (vmax lhs) (Int64.sub (vmax rhs) 1L)
      else Int64.max_int
  | Binary { op = `Max; lhs; rhs; _ } -> max (vmax lhs) (vmax rhs)
  | Unary { op = `Neg; src; _ } -> Int64.neg (vmin src)
  | Cast { src; dtype } ->
      let dt = Dtype.any_to_val dtype in
      if Dtype.is_int dt then vmax src else Int64.max_int
  | Ternary { op = `Where; b; c; _ } -> max (vmax b) (vmax c)
  | _ -> Int64.max_int

let split_add node =
  let rec go acc node =
    match K.view node with
    | Binary { op = `Add; lhs; rhs; _ } -> go (go acc rhs) lhs
    | _ -> node :: acc
  in
  go [] node

let const_factor node =
  match K.view node with
  | Const { value; _ } -> (
      match Const.view value with Int v -> v | _ -> 1L)
  | Binary { op = `Mul; rhs; _ } -> (
      match const_int_val rhs with Some v -> v | None -> 1L)
  | _ -> 1L

let sum_nodes = function
  | [] -> iconst 0L
  | [ x ] -> x
  | x :: rest ->
      List.fold_left (fun acc t -> K.binary ~op:`Add ~lhs:acc ~rhs:t) x rest

let pop_const node =
  let terms = split_add node in
  let consts, non_consts =
    List.partition (fun t -> Option.is_some (const_int_val t)) terms
  in
  let const_sum =
    List.fold_left
      (fun acc t ->
        match const_int_val t with Some v -> Int64.add acc v | None -> acc)
      0L consts
  in
  (sum_nodes non_consts, const_sum)

(* Check whether an expression is statically divisible by the constant [c],
   returning [Some (node / c)] with the quotient simplified. Recurses into
   Add (all summands must be divisible) and Mul (the constant factor must
   be divisible), returning None if any sub-expression is not evenly divisible. *)
let rec divides node c =
  if c = 1L then Some node
  else if c = 0L then None
  else
    match K.view node with
    | Const { value; dtype } -> (
        match Const.view value with
        | Int v when Int64.rem v c = 0L ->
            Some (K.const (Const.int64 dtype (Int64.div v c)))
        | _ -> None)
    | Binary { op = `Add; _ } ->
        let terms = split_add node in
        let divided = List.filter_map (fun t -> divides t c) terms in
        if List.length divided = List.length terms then Some (sum_nodes divided)
        else None
    | Binary { op = `Mul; lhs; rhs; _ } -> (
        match K.view rhs with
        | Const { value; dtype } -> (
            match Const.view value with
            | Int v when Int64.rem v c = 0L ->
                let q = Int64.div v c in
                if q = 1L then Some lhs
                else Some (K.binary ~op:`Mul ~lhs
                       ~rhs:(K.const (Const.int64 dtype q)))
            | _ -> None)
        | _ -> None)
    | _ -> None

let rec cartesian = function
  | [] -> [ [] ]
  | choices :: rest ->
      let rest_products = cartesian rest in
      List.concat_map
        (fun c -> List.map (fun rp -> c :: rp) rest_products)
        choices

(* fold_divmod_general *)

let is_index_dtype dtype =
  Dtype.is_int dtype && Dtype.equal (Dtype.scalar_of dtype) Dtype.index

let ( ||| ) a b = match a with Some _ -> a | None -> b ()

(* Top-level algebraic simplifier for Idiv and Mod on index expressions.
   Tries a cascade of strategies via the short-circuit combinator (|||):
   cancel when the quotient is provably constant, fold nested div/mod,
   remove redundant inner mods, linearize binary-valued numerators,
   exploit congruences mod c, factor out the GCD, and finally try
   recursive nesting by candidate divisors. Returns Some simplified_node
   on the first strategy that succeeds, or None. *)
let rec fold_divmod_general node =
  match K.view node with
  | Binary { op = ((`Idiv | `Mod) as op); lhs = x; rhs = y; dtype }
    when is_index_dtype dtype ->
      let is_mod = op = `Mod in
      let x_min = vmin x and x_max = vmax x
      and y_min = vmin y and y_max = vmax y in
      if y_min = 0L && y_max = 0L then None
      else
        cancel_divmod ~is_mod x y x_min x_max y_min y_max
        ||| fun () ->
        let x_peeled, const = pop_const x in
        let uops_no_const = split_add x_peeled in
        let const_denom =
          match const_int_val y with
          | Some c when c > 0L ->
              fold_const_denom ~is_mod ~x ~x_min ~x_peeled ~uops_no_const
                ~const ~c ~y
          | _ -> None
        in
        const_denom
        ||| fun () ->
        let all_uops = split_add x in
        divide_by_gcd ~is_mod ~op ~x ~y all_uops
        ||| fun () -> factor_remainder ~is_mod ~x_min ~y_min ~x ~y all_uops
  | _ -> None

and cancel_divmod ~is_mod x y x_min x_max y_min y_max =
  if Int64.mul y_min y_max > 0L then
    let q1 = cdiv x_min y_min and q2 = cdiv x_min y_max
    and q3 = cdiv x_max y_min and q4 = cdiv x_max y_max in
    if q1 = q2 && q2 = q3 && q3 = q4 then
      if is_mod then
        Some (K.binary ~op:`Sub ~lhs:x
                ~rhs:(K.binary ~op:`Mul ~lhs:(iconst q1) ~rhs:y))
      else Some (iconst q1)
    else None
  else None

and fold_const_denom ~is_mod ~x ~x_min ~x_peeled ~uops_no_const ~const ~c ~y =
  nested_div_mod ~is_mod ~x ~c ~y
  ||| fun () -> remove_nested_mod ~is_mod ~x_min ~uops_no_const ~const ~c ~y
  ||| fun () ->
  let decomp =
    List.map
      (fun u ->
        let f = const_factor u in
        let t = match divides u f with Some d -> d | None -> u in
        (t, f))
      uops_no_const
  in
  let terms = List.map fst decomp and factors = List.map snd decomp in
  fold_binary_numerator ~is_mod ~terms ~factors ~const ~c
  ||| fun () -> fold_congruence ~is_mod ~x_min ~terms ~factors ~const ~c
  ||| fun () -> gcd_with_remainder ~is_mod ~x_peeled ~factors ~const ~c
  ||| fun () -> nest_by_factor ~is_mod ~x_min ~x ~terms ~factors ~const ~c

and nested_div_mod ~is_mod ~x ~c ~y =
  match K.view x with
  | Binary { op = `Mod; lhs = x0; rhs = mod_rhs; _ } -> (
      match divides mod_rhs c with
      | Some k ->
          if is_mod then Some (K.binary ~op:`Mod ~lhs:x0 ~rhs:y)
          else
            Some (K.binary ~op:`Mod
                    ~lhs:(K.binary ~op:`Idiv ~lhs:x0 ~rhs:y) ~rhs:k)
      | None -> None)
  | _ -> None

and remove_nested_mod ~is_mod ~x_min ~uops_no_const ~const ~c ~y =
  if not (is_mod && x_min >= 0L) then None
  else
    let new_xs, changed =
      List.fold_right
        (fun u (acc, ch) ->
          match K.view u with
          | Binary { op = `Mod; lhs = u0; rhs = mr; _ } -> (
              match divides mr c with
              | Some _ -> (u0 :: acc, true)
              | None -> (u :: acc, ch))
          | _ -> (u :: acc, ch))
        uops_no_const ([], false)
    in
    if not changed then None
    else
      let new_x =
        K.binary ~op:`Add ~lhs:(sum_nodes new_xs) ~rhs:(iconst const)
      in
      if vmin new_x >= 0L then Some (K.binary ~op:`Mod ~lhs:new_x ~rhs:y)
      else None

and fold_binary_numerator ~is_mod ~terms ~factors ~const ~c =
  match terms, factors with
  | [ v ], [ f ] when Int64.sub (vmax v) (vmin v) = 1L ->
      let eval = if is_mod then cmod else cdiv in
      let y1 = eval (Int64.add (Int64.mul f (vmin v)) const) c in
      let y2 = eval (Int64.add (Int64.mul f (vmax v)) const) c in
      Some
        (K.binary ~op:`Add
           ~lhs:(K.binary ~op:`Mul ~lhs:(iconst (Int64.sub y2 y1))
                   ~rhs:(K.binary ~op:`Sub ~lhs:v ~rhs:(iconst (vmin v))))
           ~rhs:(iconst y1))
  | _ -> None

and fold_congruence ~is_mod ~x_min ~terms ~factors ~const ~c =
  if x_min < 0L then None
  else
    let rem_choices =
      List.map
        (fun f ->
          let r = floormod f c in
          if Int64.mul r 2L = c then [ r; Int64.sub r c ]
          else
            let rc = Int64.sub r c in
            if Int64.abs r <= Int64.abs rc then [ r ] else [ rc ])
        factors
    in
    List.find_map
      (fun rems ->
        let rem_terms =
          List.filter_map
            (fun (r, v) ->
              if r = 0L then None
              else if r = 1L then Some v
              else Some (K.binary ~op:`Mul ~lhs:v ~rhs:(iconst r)))
            (List.combine rems terms)
        in
        let const_rem = floormod const c in
        let all_rem =
          if const_rem <> 0L then rem_terms @ [ iconst const_rem ]
          else rem_terms
        in
        let rem = sum_nodes all_rem in
        let rem_lo = floordiv (vmin rem) c in
        let rem_hi = floordiv (vmax rem) c in
        if rem_lo <> rem_hi then None
        else if is_mod then
          Some (K.binary ~op:`Sub ~lhs:rem
                  ~rhs:(iconst (Int64.mul rem_lo c)))
        else
          let div_terms =
            List.filter_map
              (fun ((f, r), v) ->
                let q = floordiv (Int64.sub f r) c in
                if q = 0L then None
                else if q = 1L then Some v
                else Some (K.binary ~op:`Mul ~lhs:v ~rhs:(iconst q)))
              (List.combine (List.combine factors rems) terms)
          in
          let const_div = Int64.add (floordiv const c) rem_lo in
          let all_div =
            if const_div <> 0L then div_terms @ [ iconst const_div ]
            else div_terms
          in
          Some (sum_nodes all_div))
      (cartesian rem_choices)

and gcd_with_remainder ~is_mod ~x_peeled ~factors ~const ~c =
  if vmin x_peeled < 0L then None
  else
    let g = List.fold_left gcd c factors in
    if g <= 1L then None
    else
      match divides x_peeled g with
      | None -> None
      | Some divided ->
          let cg = Int64.div c g in
          let new_x =
            K.binary ~op:`Add ~lhs:divided
              ~rhs:(iconst (floormod (floordiv const g) cg))
          in
          if vmin new_x < 0L then None
          else if is_mod then
            Some
              (K.binary ~op:`Add
                 ~lhs:(K.binary ~op:`Mul
                         ~lhs:(K.binary ~op:`Mod ~lhs:new_x ~rhs:(iconst cg))
                         ~rhs:(iconst g))
                 ~rhs:(iconst (floormod const g)))
          else
            Some
              (K.binary ~op:`Add
                 ~lhs:(K.binary ~op:`Idiv ~lhs:new_x ~rhs:(iconst cg))
                 ~rhs:(iconst (floordiv const c)))

(* Try to simplify (x div c) or (x mod c) by factoring through an
   intermediate divisor: for each non-trivial factor f that divides c,
   compute (x/f) and recursively simplify that, then finish with the
   remaining (c/f). Among all candidates that succeed, pick the one
   whose backward slice (expression size) is smallest. *)
and nest_by_factor ~is_mod ~x_min ~x ~terms ~factors ~const ~c =
  if x_min < 0L then None
  else
    let x_peeled, _ = pop_const x in
    let uops_no_const = split_add x_peeled in
    let candidates =
      List.filter_map
        (fun (u, f) ->
          match K.view u with
          | Const _ -> None
          | _ ->
              let af = Int64.abs f in
              if af > 1L && af < c && Int64.rem c f = 0L then Some af
              else None)
        (List.combine uops_no_const factors)
      |> List.sort_uniq Int64.compare
    in
    let results =
      List.filter_map
        (fun div ->
          let xd = K.binary ~op:`Idiv ~lhs:x ~rhs:(iconst div) in
          match fold_divmod_general xd with
          | Some newxs when vmin newxs >= 0L ->
              let cd = Int64.div c div in
              if not is_mod then
                Some (List.length (K.backward_slice newxs),
                      K.binary ~op:`Idiv ~lhs:newxs ~rhs:(iconst cd))
              else
                let b_parts =
                  List.filter_map
                    (fun (f, t) ->
                      let r = floormod f div in
                      if r <> 0L then
                        Some (K.binary ~op:`Mul ~lhs:t ~rhs:(iconst r))
                      else None)
                    (List.combine factors terms)
                in
                let cr = floormod const div in
                let b_parts =
                  if cr <> 0L then b_parts @ [ iconst cr ] else b_parts
                in
                let b = sum_nodes b_parts in
                if vmin b >= 0L && vmax b < div then
                  let r =
                    K.binary ~op:`Add
                      ~lhs:(K.binary ~op:`Mul
                              ~lhs:(K.binary ~op:`Mod ~lhs:newxs
                                      ~rhs:(iconst cd))
                              ~rhs:(iconst div))
                      ~rhs:b
                  in
                  Some (List.length (K.backward_slice r), r)
                else None
          | _ -> None)
        candidates
    in
    match results with
    | [] -> None
    | first :: rest ->
        let _, best =
          List.fold_left
            (fun ((bc, _) as best) ((cc, _) as cur) ->
              if cc < bc then cur else best)
            first rest
        in
        Some best

and divide_by_gcd ~is_mod ~op ~x ~y all_uops =
  let gcd_val =
    List.fold_left (fun acc u -> gcd acc (const_factor u))
      (const_factor y) all_uops
  in
  if gcd_val <= 1L then None
  else
    match divides x gcd_val, divides y gcd_val with
    | Some x_div, Some y_div ->
        let ret = K.binary ~op:(op :> Op.binary) ~lhs:x_div ~rhs:y_div in
        if is_mod then
          Some (K.binary ~op:`Mul ~lhs:ret ~rhs:(iconst gcd_val))
        else Some ret
    | _ -> None

and factor_remainder ~is_mod ~x_min ~y_min ~x ~y all_uops =
  if y_min < 0L || x_min < 0L then None
  else
    let quo, rem =
      List.fold_right
        (fun u (q, r) ->
          match const_int_val y with
          | Some yv ->
              let cf = const_factor u in
              if Int64.rem cf yv <> cf then
                let base =
                  match divides u cf with Some d -> d | None -> u
                in
                let r_part =
                  K.binary ~op:`Mul ~lhs:base ~rhs:(iconst (floormod cf yv))
                in
                let q_part =
                  if is_mod then iconst 0L
                  else K.binary ~op:`Mul ~lhs:base
                         ~rhs:(iconst (floordiv cf yv))
                in
                (q_part :: q, r_part :: r)
              else (q, u :: r)
          | None -> (q, u :: r))
        all_uops ([], [])
    in
    if quo = [] then None
    else
      let new_x =
        K.binary ~op:`Add ~lhs:(sum_nodes rem) ~rhs:(iconst 0L)
      in
      if vmin new_x < 0L then None
      else if is_mod then Some (K.binary ~op:`Mod ~lhs:new_x ~rhs:y)
      else
        Some (K.binary ~op:`Add
                ~lhs:(K.binary ~op:`Idiv ~lhs:new_x ~rhs:y)
                ~rhs:(sum_nodes quo))

(* Fast inline rules *)

let fast_div_combine node =
  match K.view node with
  | Binary { op = `Idiv; lhs; rhs = d; dtype } when is_index_dtype dtype -> (
      match const_int_val d with
      | Some dv when dv > 0L -> (
          let try_pattern inner_div a =
            match K.view inner_div with
            | Binary { op = `Idiv; lhs = x; rhs = c; _ } -> (
                match const_int_val c, const_int_val a with
                | Some cv, Some av when cv > 0L && av >= 0L && vmin x >= 0L ->
                    Some
                      (K.binary ~op:`Idiv
                         ~lhs:(K.binary ~op:`Add ~lhs:x
                                 ~rhs:(iconst (Int64.mul av cv)))
                         ~rhs:(iconst (Int64.mul cv dv)))
                | _ -> None)
            | _ -> None
          in
          match K.view lhs with
          | Binary { op = `Add; lhs = l; rhs = r; _ } -> (
              match try_pattern l r with
              | Some _ as result -> result
              | None -> try_pattern r l)
          | _ -> None)
      | _ -> None)
  | _ -> None

let neg_divisor_div node =
  match K.view node with
  | Binary { op = `Idiv; lhs = x; rhs = d; dtype }
    when is_index_dtype dtype && vmax d < 0L ->
      Some (K.unary ~op:`Neg
              ~src:(K.binary ~op:`Idiv ~lhs:x ~rhs:(K.unary ~op:`Neg ~src:d)))
  | _ -> None

let neg_dividend_div node =
  match K.view node with
  | Binary { op = `Idiv; lhs = x; rhs = d; dtype }
    when is_index_dtype dtype && vmax x <= 0L ->
      Some (K.unary ~op:`Neg
              ~src:(K.binary ~op:`Idiv ~lhs:(K.unary ~op:`Neg ~src:x) ~rhs:d))
  | _ -> None

let const_split_div node =
  match K.view node with
  | Binary { op = `Idiv; lhs; rhs = d; dtype } when is_index_dtype dtype -> (
      match const_int_val d with
      | Some dv when dv > 0L -> (
          let try_split x c_node =
            match const_int_val c_node with
            | Some cv when floormod cv dv <> cv ->
                if vmin x >= 0L && vmin lhs >= 0L then
                  Some
                    (K.binary ~op:`Add
                       ~lhs:(K.binary ~op:`Idiv
                               ~lhs:(K.binary ~op:`Add ~lhs:x
                                       ~rhs:(iconst (floormod cv dv)))
                               ~rhs:(iconst dv))
                       ~rhs:(iconst (floordiv cv dv)))
                else None
            | _ -> None
          in
          match K.view lhs with
          | Binary { op = `Add; lhs = l; rhs = r; _ } -> (
              match try_split l r with
              | Some _ as result -> result
              | None -> try_split r l)
          | _ -> None)
      | _ -> None)
  | _ -> None

let neg_dividend_mod node =
  match K.view node with
  | Binary { op = `Mod; lhs = x; rhs = d; dtype }
    when is_index_dtype dtype && vmax x <= 0L ->
      Some (K.unary ~op:`Neg
              ~src:(K.binary ~op:`Mod ~lhs:(K.unary ~op:`Neg ~src:x) ~rhs:d))
  | _ -> None

let neg_divisor_mod node =
  match K.view node with
  | Binary { op = `Mod; lhs = x; rhs = d; dtype }
    when is_index_dtype dtype && vmax d < 0L ->
      Some (K.binary ~op:`Mod ~lhs:x ~rhs:(K.unary ~op:`Neg ~src:d))
  | _ -> None

(* Entry point *)

let div_and_mod_symbolic =
  K.first_match
    [ fast_div_combine; neg_divisor_div; neg_dividend_div; const_split_div;
      fold_divmod_general; neg_dividend_mod; neg_divisor_mod ]
